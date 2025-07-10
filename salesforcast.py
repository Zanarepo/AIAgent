from supabaseClient import supabase
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta, timezone
import numpy as np

def forecast_demand():
    sales = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").execute().data
    sales_df = pd.DataFrame(sales)
    sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
    sales_df["month"] = sales_df["sold_at"].dt.to_period("M").astype(str)

    monthly_sales = sales_df.groupby(["dynamic_product_id", "store_id", "month"])["quantity"].sum().reset_index()
    forecasts = []
    for (product_id, store_id) in monthly_sales[["dynamic_product_id", "store_id"]].drop_duplicates().values:
        product_data = monthly_sales[(monthly_sales["dynamic_product_id"] == product_id) & (monthly_sales["store_id"] == store_id)].sort_values("month")
        if len(product_data) < 2:
            continue

        X = [[i] for i in range(len(product_data))]
        y = product_data["quantity"].values
        model = LinearRegression()
        model.fit(X, y)
        predicted_demand = max(0, model.predict([[len(product_data)]])[0])

        inventory = supabase.table("dynamic_inventory").select("available_qty, reorder_level").eq("dynamic_product_id", product_id).eq("store_id", store_id).execute().data
        product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
        store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data

        current_stock = int(inventory[0]["available_qty"]) if inventory else 0
        reorder_level = int(inventory[0]["reorder_level"]) if inventory and inventory[0]["reorder_level"] is not None else 0
        product_name = product[0]["name"] if product else f"Product ID: {product_id}"
        shop_name = store[0]["shop_name"] if store else f"Store ID: {store_id}"

        recommendation = "Restock recommended" if predicted_demand > current_stock + reorder_level else "No restock needed"

        forecast = {
            "dynamic_product_id": int(product_id),  # Convert np.int64 to int
            "store_id": int(store_id),  # Convert np.int64 to int
            "predicted_demand": round(float(predicted_demand), 2),  # Convert np.float64 to float
            "current_stock": current_stock,
            "product_name": product_name,
            "shop_name": shop_name,
            "recommendation": recommendation,
            "forecast_period": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m"),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        forecasts.append(forecast)

    if forecasts:
        print(f"Inserting {len(forecasts)} forecasts into Supabase")
        supabase.table("forecasts").insert(forecasts).execute()
        print("Forecasts inserted successfully")
    else:
        print("No forecasts to insert")

    return forecasts

def detect_anomalies():
    sales = supabase.table("dynamic_sales").select("id, dynamic_product_id, store_id, quantity, sold_at").execute().data
    sales_df = pd.DataFrame(sales)
    sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)

    anomalies = []
    for (product_id, store_id) in sales_df[["dynamic_product_id", "store_id"]].drop_duplicates().values:
        product_sales = sales_df[(sales_df["dynamic_product_id"] == product_id) & (sales_df["store_id"] == store_id)]
        quantities = product_sales["quantity"].values
        if len(quantities) < 3:
            continue

        mean = np.mean(quantities)
        std = np.std(quantities)
        z_scores = [(q - mean) / std if std > 0 else 0 for q in quantities]
        anomaly_indices = [i for i, z in enumerate(z_scores) if abs(z) > 3]

        for idx in anomaly_indices:
            row = product_sales.iloc[idx]
            anomalies.append({
                "dynamic_product_id": int(row["dynamic_product_id"]),  # Convert np.int64 to int
                "store_id": int(row["store_id"]),  # Convert np.int64 to int
                "quantity": int(row["quantity"]),  # Convert np.int64 to int
                "sold_at": row["sold_at"].isoformat(),
                "anomaly_type": "High" if z_scores[idx] > 0 else "Low",
                "created_at": datetime.now(timezone.utc).isoformat()
            })

    if anomalies:
        print(f"Inserting {len(anomalies)} anomalies into Supabase")
        supabase.table("anomalies").insert(anomalies).execute()
        print("Anomalies inserted successfully")
    else:
        print("No anomalies to insert")

    return anomalies

def sales_trends():
    sales = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").execute().data
    sales_df = pd.DataFrame(sales)
    sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
    sales_df["month"] = sales_df["sold_at"].dt.to_period("M").astype(str)

    trends = {
        "top_products": sales_df.groupby("dynamic_product_id")["quantity"].sum().nlargest(5).to_dict(),
        "top_stores": sales_df.groupby("store_id")["quantity"].sum().nlargest(5).to_dict(),
        "monthly_trends": sales_df.groupby("month")["quantity"].sum().to_dict()
    }
    print("Sales trends computed:", trends)
    return trends

def handler(request):
    try:
        forecasts = forecast_demand()
        anomalies = detect_anomalies()
        trends = sales_trends()
        return {
            "statusCode": 200,
            "body": {
                "forecasts": forecasts,
                "anomalies": anomalies,
                "trends": trends
            }
        }
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"statusCode": 500, "body": str(e)}

if __name__ == "__main__":
    forecast_demand()
    detect_anomalies()
    sales_trends()