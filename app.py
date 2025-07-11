from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": ["http://localhost:3000", "https://sellytics.sprintifyhq.com"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})
logger.info("CORS configured for origins: http://localhost:3000, https://sellytics.sprintifyhq.com")

# Ensure CORS headers for all responses, including errors
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    logger.debug(f"Added CORS headers to response: {response.headers}")
    return response

# Initialize Supabase client
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logger.error("Missing SUPABASE_URL or SUPABASE_KEY")
    raise ValueError("Supabase credentials not found")
supabase: Client = create_client(supabase_url, supabase_key)

# Helper function to convert NumPy types to Python types
def convert_to_python_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    return obj

# Cache database queries
@lru_cache(maxsize=32)
def fetch_sales_data(store_id=None):
    try:
        query = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at")
        if store_id:
            query = query.eq("store_id", store_id)
        sales = query.execute().data
        logger.info(f"Fetched {len(sales)} sales records for store_id {store_id or 'all'}")
        return pd.DataFrame(sales) if sales else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching sales data: {str(e)}")
        raise

@lru_cache(maxsize=32)
def fetch_inventory_data(store_id=None):
    try:
        query = supabase.table("dynamic_inventory").select("dynamic_product_id, store_id, available_qty, reorder_level, safety_stock, updated_at")
        if store_id:
            query = query.eq("store_id", store_id)
        inventory = query.execute().data
        logger.info(f"Fetched {len(inventory)} inventory records for store_id {store_id or 'all'}")
        return pd.DataFrame(inventory) if inventory else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching inventory data: {str(e)}")
        raise

# Sales forecasting with Prophet
def forecast_demand(store_id=None):
    try:
        sales_df = fetch_sales_data(store_id)
        if sales_df.empty:
            logger.info("No sales data found")
            return []
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        sales_df["month"] = sales_df["sold_at"].dt.to_period("M").dt.to_timestamp()

        forecasts = []
        for (product_id, store_id) in sales_df[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_data = sales_df[(sales_df["dynamic_product_id"] == product_id) & (sales_df["store_id"] == store_id)]
            monthly_sales = product_data.groupby("month")["quantity"].sum().reset_index()
            monthly_sales.columns = ["ds", "y"]

            if len(monthly_sales) < 2:
                logger.info(f"Skipping forecast for product_id {product_id}, store_id {store_id}: insufficient data")
                continue

            model = Prophet(
                yearly_seasonality=False,  # Disable to avoid holidays
                weekly_seasonality=False,
                daily_seasonality=False,
                holidays=None,  # Explicitly disable holidays
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            model.fit(monthly_sales)
            future = model.make_future_dataframe(periods=1, freq="M")
            forecast = model.predict(future)
            predicted_demand = max(0, forecast["yhat"].iloc[-1])

            inventory = fetch_inventory_data(store_id)
            inventory = inventory[(inventory["dynamic_product_id"] == product_id) & (inventory["store_id"] == store_id)]
            product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
            store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data

            if inventory.empty:
                logger.warning(f"No inventory data for product_id {product_id}, store_id {store_id}")
                current_stock = 0
                reorder_level = 0
            else:
                current_stock = int(inventory["available_qty"].iloc[-1])
                reorder_level = int(inventory["reorder_level"].iloc[-1]) if inventory["reorder_level"].iloc[-1] is not None else 0

            product_name = product[0]["name"] if product and len(product) > 0 else f"Product ID: {product_id}"
            shop_name = store[0]["shop_name"] if store and len(store) > 0 else f"Store ID: {store_id}"

            recommendation = "Restock recommended" if predicted_demand > current_stock + reorder_level else "No restock needed"

            forecast_entry = {
                "dynamic_product_id": int(product_id),
                "store_id": int(store_id),
                "predicted_demand": round(float(predicted_demand), 2),
                "current_stock": current_stock,
                "product_name": product_name,
                "shop_name": shop_name,
                "recommendation": recommendation,
                "forecast_period": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            forecasts.append(forecast_entry)

        if forecasts:
            logger.info(f"Inserting {len(forecasts)} forecasts into Supabase")
            supabase.table("forecasts").insert(forecasts).execute()
        return forecasts
    except Exception as e:
        logger.error(f"Error in forecast_demand: {str(e)}")
        raise

# Theft detection
def detect_theft(store_id=None):
    try:
        sales_df = fetch_sales_data(store_id)
        inventory_df = fetch_inventory_data(store_id)
        if sales_df.empty or inventory_df.empty:
            logger.info("No sales or inventory data for theft detection")
            return []

        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        inventory_df["updated_at"] = pd.to_datetime(inventory_df["updated_at"], utc=True)

        theft_incidents = []
        for (product_id, store_id) in inventory_df[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_sales = sales_df[(sales_df["dynamic_product_id"] == product_id) & (sales_df["store_id"] == store_id)]
            product_inventory = inventory_df[(inventory_df["dynamic_product_id"] == product_id) & (inventory_df["store_id"] == store_id)]

            if product_inventory.empty or len(product_inventory) < 2:
                logger.info(f"Skipping theft detection for product_id {product_id}, store_id {store_id}: insufficient inventory data")
                continue

            product_inventory = product_inventory.sort_values("updated_at")
            inventory_changes = product_inventory["available_qty"].diff().dropna()

            sales_period = product_sales[(product_sales["sold_at"] >= product_inventory["updated_at"].min()) & 
                                        (product_sales["sold_at"] <= product_inventory["updated_at"].max())]
            total_sold = sales_period["quantity"].sum()

            for idx, change in inventory_changes.items():
                if change < 0:
                    expected_change = -total_sold
                    if abs(change) > abs(expected_change) * 1.2:
                        product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
                        store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data
                        product_name = product[0]["name"] if product and len(product) > 0 else f"Product ID: {product_id}"
                        shop_name = store[0]["shop_name"] if store and len(store) > 0 else f"Store ID: {store_id}"
                        theft_incidents.append({
                            "dynamic_product_id": int(product_id),
                            "store_id": int(store_id),
                            "inventory_change": float(change),
                            "expected_change": float(expected_change),
                            "timestamp": product_inventory.loc[idx, "updated_at"].isoformat(),
                            "product_name": product_name,
                            "shop_name": shop_name,
                            "created_at": datetime.now(timezone.utc).isoformat()
                        })

        if theft_incidents:
            logger.info(f"Inserting {len(theft_incidents)} theft incidents into Supabase")
            supabase.table("theft_incidents").insert(theft_incidents).execute()
        return theft_incidents
    except Exception as e:
        logger.error(f"Error in detect_theft: {str(e)}")
        raise

# Anomaly detection
def detect_anomalies(store_id=None):
    try:
        sales_df = fetch_sales_data(store_id)
        if sales_df.empty:
            logger.info("No sales data found for anomalies")
            return []
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)

        anomalies = []
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        for (product_id, store_id) in sales_df[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_data = sales_df[(sales_df["dynamic_product_id"] == product_id) & (sales_df["store_id"] == store_id)]
            if len(product_data) < 3:
                logger.info(f"Skipping anomaly detection for product_id {product_id}, store_id {store_id}: insufficient data")
                continue

            product_data = product_data.copy()
            product_data["anomaly"] = iso_forest.fit_predict(product_data[["quantity"]])
            anomaly_rows = product_data[product_data["anomaly"] == -1].index

            for idx in anomaly_rows:
                row = product_data.loc[idx]
                product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
                store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data
                if not product:
                    logger.warning(f"No product found for product_id {product_id}")
                if not store:
                    logger.warning(f"No store found for store_id {store_id}")
                anomalies.append({
                    "dynamic_product_id": int(row["dynamic_product_id"]),
                    "store_id": int(row["store_id"]),
                    "quantity": int(row["quantity"]),
                    "sold_at": row["sold_at"].isoformat(),
                    "anomaly_type": "Outlier",
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

        if anomalies:
            logger.info(f"Inserting {len(anomalies)} anomalies into Supabase")
            supabase.table("anomalies").insert(anomalies).execute()
        return anomalies
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
        raise

# Sales trends
def sales_trends(store_id=None):
    try:
        sales_df = fetch_sales_data(store_id)
        if sales_df.empty:
            logger.info(f"No sales data found for store_id {store_id or 'all'}")
            return {
                "top_products": {},
                "top_stores": {},
                "monthly_trends": {},
                "yearly_growth": {}
            }
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        sales_df["month"] = sales_df["sold_at"].dt.to_period("M").astype(str)
        sales_df["year"] = sales_df["sold_at"].dt.year

        trends = {
            "top_products": sales_df.groupby("dynamic_product_id")["quantity"].sum().nlargest(5).to_dict(),
            "top_stores": {store_id: sales_df["quantity"].sum()} if store_id else {},
            "monthly_trends": sales_df.groupby("month")["quantity"].sum().to_dict(),
            "yearly_growth": sales_df.groupby("year")["quantity"].sum().pct_change().fillna(0).to_dict()
        }
        # Convert NumPy types to Python types
        trends = convert_to_python_types(trends)
        logger.info(f"Generated trends for store_id {store_id or 'all'}: {trends}")
        return trends
    except Exception as e:
        logger.error(f"Error in sales_trends: {str(e)}")
        raise

# Inquiry processing
def process_inquiry(inquiry_text):
    try:
        inquiry_text = inquiry_text.lower()
        responses = {
            "stock": "Check the inventory dashboard or contact support.",
            "availability": "Availability is updated real-time on the platform.",
            "order": "Place orders via the platform or support.",
            "delivery": "Delivery depends on location; provide more details.",
            "price": "Prices are in the product catalog.",
            "theft": "Potential theft incidents are flagged in the dashboard. Please review the alerts."
        }
        for key, response in responses.items():
            if key in inquiry_text:
                return response
        return "Thanks for your inquiry. Please provide more details or contact support."
    except Exception as e:
        logger.error(f"Error in process_inquiry: {str(e)}")
        raise

def handle_inquiries():
    try:
        inquiries = supabase.table("customer_inquiries").select("id, inquiry_text").eq("status", "pending").execute().data
        logger.info(f"Found {len(inquiries)} pending inquiries")
        processed = []
        for inquiry in inquiries:
            response = process_inquiry(inquiry["inquiry_text"])
            supabase.table("customer_inquiries").update({
                "response_text": response,
                "status": "responded",
                "created_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", inquiry["id"]).execute()
            processed.append({"inquiry_text": inquiry["inquiry_text"], "response_text": response})
        
        recent_inquiries = supabase.table("customer_inquiries").select("inquiry_text, response_text").order("created_at", desc=True).limit(10).execute().data
        return recent_inquiries
    except Exception as e:
        logger.error(f"Error in handle_inquiries: {str(e)}")
        raise

@app.route('/forecast', methods=['GET'])
def forecast_endpoint():
    try:
        store_id = request.args.get('store_id')
        if store_id:
            try:
                store_id = int(store_id)
            except ValueError:
                return jsonify({"error": "Invalid store_id format"}), 400

        forecasts = forecast_demand(store_id)
        anomalies = detect_anomalies(store_id)
        theft_incidents = detect_theft(store_id)
        trends = sales_trends(store_id)
        return jsonify({
            "forecasts": forecasts,
            "anomalies": anomalies,
            "theft_incidents": theft_incidents,
            "trends": trends
        }), 200
    except Exception as e:
        logger.error(f"Error in /forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/inquiries', methods=['GET'])
def inquiries_endpoint():
    try:
        inquiries = handle_inquiries()
        return jsonify({"inquiries": inquiries}), 200
    except Exception as e:
        logger.error(f"Error in /inquiries: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/theft', methods=['GET'])
def theft_endpoint():
    try:
        store_id = request.args.get('store_id')
        if store_id:
            try:
                store_id = int(store_id)
            except ValueError:
                return jsonify({"error": "Invalid store_id format"}), 400
        theft_incidents = detect_theft(store_id)
        return jsonify({"theft_incidents": theft_incidents}), 200
    except Exception as e:
        logger.error(f"Error in /theft: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root_endpoint():
    return jsonify({"message": "Sellytics AI Agent Backend"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))