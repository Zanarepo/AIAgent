
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime, timezone
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": ["http://localhost:3000", "https://sellytics.sprintifyhq.com"],
    "methods": ["GET", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})
logger.info("CORS configured for origins: http://localhost:3000, https://sellytics.sprintifyhq.com")

# Initialize Supabase client
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logger.error("Missing SUPABASE_URL or SUPABASE_KEY")
    raise ValueError("Supabase credentials not found")
supabase: Client = create_client(supabase_url, supabase_key)

# Helper function to convert NumPy types
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

# Anomaly detection
def detect_anomalies(store_id=None):
    try:
        # Fetch limited sales data to reduce memory usage
        query = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").limit(500)
        if store_id:
            query = query.eq("store_id", store_id)
        sales = query.execute().data
        sales_df = pd.DataFrame(sales) if sales else pd.DataFrame()
        if sales_df.empty:
            logger.info("No sales data found for anomalies")
            return []
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        logger.debug(f"Anomaly detection - Sales data shape: {sales_df.shape}, unique products: {len(sales_df['dynamic_product_id'].unique())}")

        anomalies = []
        iso_forest = IsolationForest(n_estimators=20, max_samples=128, contamination=0.05, random_state=42, n_jobs=1)
        for (product_id, store_id) in sales_df[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_data = sales_df[(sales_df["dynamic_product_id"] == product_id) & (sales_df["store_id"] == store_id)]
            logger.debug(f"Processing anomaly detection for product_id {product_id}, store_id {store_id}, data size: {len(product_data)}")
            if len(product_data) < 3:
                logger.info(f"Skipping anomaly detection for product_id {product_id}, store_id {store_id}: insufficient data")
                continue
            if len(product_data) > 500:
                logger.info(f"Skipping anomaly detection for product_id {product_id}, store_id {store_id}: dataset too large ({len(product_data)} rows)")
                continue

            product_data = product_data.copy()
            try:
                product_data["anomaly"] = iso_forest.fit_predict(product_data[["quantity"]])
                anomaly_rows = product_data[product_data["anomaly"] == -1].index
                logger.debug(f"Found {len(anomaly_rows)} anomalies for product_id {product_id}, store_id {store_id}")
            except Exception as e:
                logger.error(f"Error in IsolationForest for product_id {product_id}, store_id {store_id}: {str(e)}\n{traceback.format_exc()}")
                continue

            for idx in anomaly_rows:
                row = product_data.loc[idx]
                product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
                store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data
                product_name = product[0]["name"] if product and len(product) > 0 else f"Product ID: {product_id}"
                shop_name = store[0]["shop_name"] if store and len(store) > 0 else f"Store ID: {store_id}"
                anomalies.append({
                    "dynamic_product_id": int(row["dynamic_product_id"]),
                    "store_id": int(row["store_id"]),
                    "quantity": int(row["quantity"]),
                    "sold_at": row["sold_at"].isoformat(),
                    "anomaly_type": "High" if row["quantity"] > product_data["quantity"].mean() else "Low",
                    "product_name": product_name,
                    "shop_name": shop_name,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

        if anomalies:
            logger.info(f"Inserting {len(anomalies)} anomalies into Supabase")
            supabase.table("anomalies").insert(anomalies).execute()
        return convert_to_python_types(anomalies)
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}\n{traceback.format_exc()}")
        raise

# Analyze sales patterns for restock and avoid-restock recommendations
def analyze_sales_patterns(store_id=None):
    try:
        # Fetch limited sales data to reduce memory usage
        query = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").limit(500)
        if store_id:
            query = query.eq("store_id", store_id)
        sales = query.execute().data
        sales_df = pd.DataFrame(sales) if sales else pd.DataFrame()
        if sales_df.empty:
            logger.info("No sales data found for sales patterns")
            return {"restock_recommendations": [], "avoid_restock": [], "high_demand_periods": []}
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        sales_df["month"] = sales_df["sold_at"].dt.to_period("M").astype(str)
        logger.debug(f"Sales patterns - Sales data shape: {sales_df.shape}, unique products: {len(sales_df['dynamic_product_id'].unique())}")

        # Aggregate sales by product, store, and month
        monthly_sales = sales_df.groupby(["dynamic_product_id", "store_id", "month"])["quantity"].sum().reset_index()
        
        # Calculate thresholds for high/low sales
        mean_sales = monthly_sales["quantity"].mean()
        high_sales_threshold = mean_sales + monthly_sales["quantity"].std()
        low_sales_threshold = mean_sales - monthly_sales["quantity"].std()

        restock_recommendations = []
        avoid_restock = []
        high_demand_periods = []
        for (product_id, store_id) in monthly_sales[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_data = monthly_sales[(monthly_sales["dynamic_product_id"] == product_id) & (monthly_sales["store_id"] == store_id)]
            product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
            store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data
            product_name = product[0]["name"] if product and len(product) > 0 else f"Product ID: {product_id}"
            shop_name = store[0]["shop_name"] if store and len(store) > 0 else f"Store ID: {store_id}"

            # Identify high/low sales and peak periods
            for _, row in product_data.iterrows():
                quantity = row["quantity"]
                month = row["month"]
                if quantity >= high_sales_threshold:
                    restock_recommendations.append({
                        "dynamic_product_id": int(product_id),
                        "store_id": int(store_id),
                        "product_name": product_name,
                        "shop_name": shop_name,
                        "month": month,
                        "quantity_sold": int(quantity),
                        "recommendation": f"Restock {product_name} for {month} due to high demand",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    })
                    high_demand_periods.append({
                        "dynamic_product_id": int(product_id),
                        "store_id": int(store_id),
                        "product_name": product_name,
                        "shop_name": shop_name,
                        "month": month,
                        "quantity_sold": int(quantity),
                        "recommendation": f"High demand for {product_name} in {month}; plan inventory accordingly",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    })
                elif quantity <= low_sales_threshold:
                    avoid_restock.append({
                        "dynamic_product_id": int(product_id),
                        "store_id": int(store_id),
                        "product_name": product_name,
                        "shop_name": shop_name,
                        "month": month,
                        "quantity_sold": int(quantity),
                        "recommendation": f"Purchase {product_name} less frequently for {month} due to low sales",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    })

        # Insert recommendations into Supabase
        if restock_recommendations or avoid_restock or high_demand_periods:
            logger.info(f"Inserting {len(restock_recommendations)} restock, {len(avoid_restock)} avoid restock, {len(high_demand_periods)} high-demand periods into Supabase")
            supabase.table("restock_recommendations").insert(restock_recommendations + avoid_restock + high_demand_periods).execute()

        return convert_to_python_types({
            "restock_recommendations": restock_recommendations,
            "avoid_restock": avoid_restock,
            "high_demand_periods": high_demand_periods
        })
    except Exception as e:
        logger.error(f"Error in analyze_sales_patterns: {str(e)}\n{traceback.format_exc()}")
        raise

# Recommendations endpoint
@app.route('/recommendations', methods=['GET'])
def recommendations_endpoint():
    try:
        store_id = request.args.get('store_id')
        if store_id:
            try:
                store_id = int(store_id)
                logger.debug(f"Processing recommendations for store_id: {store_id}")
            except ValueError:
                logger.error("Invalid store_id format")
                return jsonify({"error": "Invalid store_id format"}), 400
        anomalies = detect_anomalies(store_id)
        sales_patterns = analyze_sales_patterns(store_id)
        response = {
            "anomalies": anomalies,
            "restock_recommendations": sales_patterns["restock_recommendations"],
            "avoid_restock": sales_patterns["avoid_restock"],
            "high_demand_periods": sales_patterns["high_demand_periods"]
        }
        logger.debug(f"Recommendations endpoint response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in /recommendations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8000)))