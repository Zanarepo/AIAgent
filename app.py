from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import spacy
from datetime import datetime, timedelta, timezone
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize Supabase client
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logger.error("Missing SUPABASE_URL or SUPABASE_KEY")
    raise ValueError("Supabase credentials not found")
supabase: Client = create_client(supabase_url, supabase_key)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Sales forecasting with Prophet
def forecast_demand():
    try:
        sales = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").execute().data
        if not sales:
            logger.info("No sales data found")
            return []
        sales_df = pd.DataFrame(sales)
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        sales_df["date"] = sales_df["sold_at"].dt.date

        daily_sales = sales_df.groupby(["dynamic_product_id", "store_id", "date"])["quantity"].sum().reset_index()
        forecasts = []
        for (product_id, store_id) in daily_sales[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_data = daily_sales[(daily_sales["dynamic_product_id"] == product_id) & (daily_sales["store_id"] == store_id)][["date", "quantity"]]
            if len(product_data) < 3:
                continue

            # Prepare data for Prophet
            prophet_df = product_data.rename(columns={"date": "ds", "quantity": "y"})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            predicted_demand = max(0, forecast["yhat"].iloc[-1])

            inventory = supabase.table("dynamic_inventory").select("available_qty, reorder_level, safety_stock").eq("dynamic_product_id", product_id).eq("store_id", store_id).execute().data
            product = supabase.table("dynamic_product").select("name, purchase_price").eq("id", product_id).execute().data
            store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data

            current_stock = int(inventory[0]["available_qty"]) if inventory else 0
            reorder_level = int(inventory[0]["reorder_level"]) if inventory and inventory[0]["reorder_level"] is not None else 0
            safety_stock = int(inventory[0]["safety_stock"]) if inventory and inventory[0]["safety_stock"] is not None else 0
            product_name = product[0]["name"] if product else f"Product ID: {product_id}"
            purchase_price = float(product[0]["purchase_price"]) if product and product[0]["purchase_price"] is not None else 0.0
            shop_name = store[0]["shop_name"] if store else f"Store ID: {store_id}"

            restock_qty = max(0, int(predicted_demand + safety_stock - current_stock))
            recommendation = f"Restock {restock_qty} units" if restock_qty > 0 else "No restock needed"

            forecast = {
                "dynamic_product_id": int(product_id),
                "store_id": int(store_id),
                "predicted_demand": round(float(predicted_demand), 2),
                "current_stock": current_stock,
                "restock_quantity": restock_qty,
                "product_name": product_name,
                "shop_name": shop_name,
                "purchase_price": purchase_price,
                "recommendation": recommendation,
                "forecast_period": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            forecasts.append(forecast)

        if forecasts:
            logger.info(f"Inserting {len(forecasts)} forecasts into Supabase")
            supabase.table("forecasts").insert(forecasts).execute()
        return forecasts
    except Exception as e:
        logger.error(f"Error in forecast_demand: {str(e)}")
        raise

# Anomaly detection with Isolation Forest
def detect_anomalies():
    try:
        sales = supabase.table("dynamic_sales").select("id, dynamic_product_id, store_id, quantity, sold_at").execute().data
        if not sales:
            logger.info("No sales data found for anomalies")
            return []
        sales_df = pd.DataFrame(sales)
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)

        anomalies = []
        for (product_id, store_id) in sales_df[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_sales = sales_df[(sales_df["dynamic_product_id"] == product_id) & (sales_df["store_id"] == store_id)][["quantity", "sold_at"]]
            if len(product_sales) < 5:
                continue

            features = product_sales[["quantity"]]
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            product_sales["anomaly"] = iso_forest.fit_predict(features)
            anomaly_rows = product_sales[product_sales["anomaly"] == -1]

            for _, row in anomaly_rows.iterrows():
                product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
                store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data
                product_name = product[0]["name"] if product else f"Product ID: {product_id}"
                shop_name = store[0]["shop_name"] if store else f"Store ID: {store_id}"

                anomalies.append({
                    "dynamic_product_id": int(product_id),
                    "store_id": int(store_id),
                    "quantity": int(row["quantity"]),
                    "sold_at": row["sold_at"].isoformat(),
                    "anomaly_type": "Potential theft or error",
                    "product_name": product_name,
                    "shop_name": shop_name,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

        if anomalies:
            logger.info(f"Inserting {len(anomalies)} anomalies into Supabase")
            supabase.table("anomalies").insert(anomalies).execute()
        return anomalies
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
        raise

# Sales trends analysis
def sales_trends():
    try:
        sales = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").execute().data
        if not sales:
            logger.info("No sales data found for trends")
            return {}
        sales_df = pd.DataFrame(sales)
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        sales_df["month"] = sales_df["sold_at"].dt.to_period("M").astype(str)

        product_names = {p["id"]: p["name"] for p in supabase.table("dynamic_product").select("id, name").execute().data}
        store_names = {s["id"]: s["shop_name"] for s in supabase.table("stores").select("id, shop_name").execute().data}

        trends = {
            "top_products": sales_df.groupby("dynamic_product_id")["quantity"].sum().nlargest(5).index.map(product_names).to_dict(),
            "top_stores": sales_df.groupby("store_id")["quantity"].sum().nlargest(5).index.map(store_names).to_dict(),
            "monthly_trends": sales_df.groupby("month")["quantity"].sum().to_dict(),
            "product_performance": sales_df.groupby(["dynamic_product_id", "month"])["quantity"].sum().unstack().fillna(0).to_dict()
        }
        return trends
    except Exception as e:
        logger.error(f"Error in sales_trends: {str(e)}")
        raise

# Enhanced inquiry processing with NLP
def process_inquiry(inquiry_text):
    try:
        doc = nlp(inquiry_text.lower())
        intents = {
            "stock": ["stock", "inventory", "available"],
            "availability": ["available", "in stock", "stock status"],
            "order": ["order", "purchase", "buy"],
            "delivery": ["delivery", "shipping", "arrival"],
            "price": ["price", "cost", "pricing"]
        }
        for intent, keywords in intents.items():
            if any(token.text in keywords for token in doc):
                responses = {
                    "stock": "Check the inventory dashboard for real-time stock levels.",
                    "availability": "Product availability is updated live on the platform.",
                    "order": "Place orders directly via the platform or contact support.",
                    "delivery": "Delivery timelines vary by location. Please provide your details.",
                    "price": "View pricing in the product catalog on the platform."
                }
                return responses[intent]
        return "Thank you for your inquiry. Please provide more details or contact support."
    except Exception as e:
        logger.error(f"Error in process_inquiry: {str(e)}")
        raise

# Handle customer inquiries
def handle_inquiries():
    try:
        inquiries = supabase.table("customer_inquiries").select("id, inquiry_text").eq("status", "pending").execute().data
        logger.info(f"Found {len(inquiries)} pending inquiries")
        for inquiry in inquiries:
            response = process_inquiry(inquiry["inquiry_text"])
            logger.info(f"Processing inquiry {inquiry['id']}: {inquiry['inquiry_text']} -> {response}")
            supabase.table("customer_inquiries").update({
                "response_text": response,
                "status": "responded",
                "created_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", inquiry["id"]).execute()
        return inquiries
    except Exception as e:
        logger.error(f"Error in handle_inquiries: {str(e)}")
        raise

# API Endpoints
@app.route('/forecast', methods=['GET'])
def forecast_endpoint():
    try:
        forecasts = forecast_demand()
        return jsonify({"forecasts": forecasts}), 200
    except Exception as e:
        logger.error(f"Error in /forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/anomalies', methods=['GET'])
def anomalies_endpoint():
    try:
        anomalies = detect_anomalies()
        return jsonify({"anomalies": anomalies}), 200
    except Exception as e:
        logger.error(f"Error in /anomalies: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/trends', methods=['GET'])
def trends_endpoint():
    try:
        trends = sales_trends()
        return jsonify({"trends": trends}), 200
    except Exception as e:
        logger.error(f"Error in /trends: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/inquiries', methods=['GET'])
def inquiries_endpoint():
    try:
        inquiries = handle_inquiries()
        return jsonify({"inquiries": inquiries}), 200
    except Exception as e:
        logger.error(f"Error in /inquiries: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root_endpoint():
    return jsonify({"message": "Sellytics AI Agent Backend"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))