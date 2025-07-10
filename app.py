from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import logging
import requests
from datetime import datetime, timezone
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust origins in production

# Initialize Supabase client
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
xai_api_key = os.getenv("XAI_API_KEY")
if not supabase_url or not supabase_key:
    logger.error("Missing SUPABASE_URL or SUPABASE_KEY")
    raise ValueError("Supabase credentials not found")
if not xai_api_key:
    logger.error("Missing XAI_API_KEY")
    raise ValueError("xAI API key not found")
supabase: Client = create_client(supabase_url, supabase_key)

# FAQ data
FAQS = [
    {"question": "How do I begin to use Sellytics?", "answer": "As a new user, register your store with the necessary details to start using Sellytics."},
    {"question": "How do I add products and prices?", "answer": "Go to Products & Pricing, click 'Add,' and input the product name, description, total purchase price, and quantity purchased."},
    {"question": "How can I use Sales Tracker?", "answer": "In Sales Tracker, click on Sales to add Product Sold, Quantity, Unit Price, and Payment Method, then click Save Sale."},
    {"question": "Can I filter or search past sales?", "answer": "Yes, use the built-in search box to find transactions by date, product name, or payment method."},
    {"question": "What is inventory for?", "answer": "Manage Inventory lets you track items sold and the number of items available."},
    {"question": "Will I be notified when my stock is running low?", "answer": "Yes, you’ll get automatic alerts when any product hits its minimum stock level, so you never run out unexpectedly."},
    {"question": "Can I create a receipt for every good sold?", "answer": "Yes, Sellytics generates clean, professional receipts for every item sold, including customer details, which can be printed or emailed."},
    {"question": "How do I manage returned items or goods?", "answer": "Go to the Returns Tracker, select the items being returned, and capture the goods returned and their details."},
    {"question": "Can I keep track of my business expenses?", "answer": "Yes, Sellytics allows you to log all expenses like rent and utilities. Click 'Add Expense' and input the details."},
    {"question": "Can I track customers who owe me money?", "answer": "Yes, the Debt Manager lets you log customers who purchase on credit, tracking who owes what after registering them."},
    {"question": "How do I manage multiple stores?", "answer": "Create an account for each store. Sellytics links them after verification, allowing you to manage all locations from a centralized dashboard."},
    {"question": "How do I manage attendants (sellers) working for me?", "answer": "Invite attendants individually and assign them to specific stores. Each attendant manages their account and daily sales within their store."},
    {"question": "How do I use unpaid supplies?", "answer": "Unpaid Supplies records third-party sellers who take goods to sell and return or pay after sales."},
    {"question": "How can I see how my business is performing each day?", "answer": "Your dashboard shows a daily summary of total sales, top products, and performance trends at a glance."},
    {"question": "Can I store my customers’ details?", "answer": "Yes, in the Customer Hub, save names, phone numbers, emails, and addresses to enhance customer service and follow-up."},
    {"question": "Will I get reports that help me make smarter business decisions?", "answer": "Yes, Sellytics provides insightful reports on sales performance, profit margins, and top-selling items to guide decisions."},
    {"question": "Can I export or share my reports?", "answer": "Yes, download reports instantly as PDF or CSV files for bookkeeping or sharing with your team."},
    {"question": "How do I create a receipt for a customer?", "answer": "Go to Quick Receipts, select purchased products, enter quantities, and print or share the receipt instantly."},
    {"question": "How can I update my product prices?", "answer": "Navigate to Products & Pricing, find the product, click Edit, adjust the price, and click Save."},
    {"question": "How can I manage unpaid supplier bills?", "answer": "In the Unpaid Supplies section, add the supplier name and owed amount, and update the record when payment is completed."},
    {"question": "Can I manage more than one store in Sellytics?", "answer": "Yes, in the Multi-Store View, add or select stores to monitor sales, inventory, and staff activity for each location."},
    {"question": "How do I record business expenses?", "answer": "Open the Expense Log, enter expense details (type, amount, date), categorize it, and click Save to record the transaction."},
    {"question": "How do I manage customer information?", "answer": "In the Customer Hub, click 'Add New Customer,' enter their details, and update or view their information anytime."},
    {"question": "Where can I access performance reports?", "answer": "Visit the Reports section to view clear tables showing sales, stock levels, and business trends to support informed decisions."},
    {"question": "How do I handle product returns?", "answer": "Go to the Returns Tracker, click 'Add Return,' select the item, enter the reason, and confirm. Your inventory will update automatically."},
]

def process_inquiry(inquiry_text, user_id=None):
    try:
        # Fetch user history if user_id is provided
        history = []
        if user_id:
            history_data = supabase.table("chat_history").select("inquiry_text, response_text").eq("user_id", user_id).order("created_at", desc=True).limit(5).execute().data
            history = [f"User: {h['inquiry_text']}\nBot: {h['response_text']}" for h in history_data]

        # Prepare the prompt with FAQ context and history
        faq_context = "\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in FAQS])
        history_context = "\n".join(history) if history else "No previous conversation."
        prompt = (
            f"You are a helpful assistant for Sellytics, a platform that helps retail stores manage inventory, sales, and customer interactions. "
            f"Use the following FAQ data to answer the user's inquiry accurately and concisely. If the inquiry is beyond the scope of the FAQs or requires human intervention, respond with: "
            f"'This query requires assistance from our customer service team. A representative will assist you shortly.' and set the escalation flag. "
            f"Use the conversation history to provide personalized responses if relevant.\n\n"
            f"FAQ Data:\n{faq_context}\n\n"
            f"Conversation History:\n{history_context}\n\n"
            f"User Inquiry: {inquiry_text}\n\n"
            f"Provide a clear, professional, and concise response:"
        )

        # Call xAI API (Grok 3)
        headers = {
            "Authorization": f"Bearer {xai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"].strip()

        # Check for escalation
        escalation_needed = "requires assistance from our customer service team" in ai_response.lower()
        return ai_response, escalation_needed
    except Exception as e:
        logger.error(f"Error in process_inquiry with AI: {str(e)}")
        # Fallback response
        inquiry_text = inquiry_text.lower()
        responses = {
            "stock": "Please check the inventory dashboard or contact support for stock details.",
            "availability": "Availability can be checked in real-time on the platform.",
            "order": "Orders can be placed through the platform or by contacting support.",
            "delivery": "Delivery timelines depend on your location. Please provide more details.",
            "price": "Pricing details are available in the product catalog."
        }
        for key, response in responses.items():
            if key in inquiry_text:
                return response, False
        return "Thank you for your inquiry. Please provide more details or contact support.", False

def handle_inquiries():
    try:
        inquiries = supabase.table("customer_inquiries").select("id, inquiry_text, user_id").eq("status", "pending").execute().data
        logger.info(f"Found {len(inquiries)} pending inquiries")
        for inquiry in inquiries:
            response, escalation_needed = process_inquiry(inquiry["inquiry_text"], inquiry["user_id"])
            logger.info(f"Processing inquiry {inquiry['id']}: {inquiry['inquiry_text']} -> {response}")
            
            # Update customer_inquiries table
            supabase.table("customer_inquiries").update({
                "response_text": response,
                "status": "escalated" if escalation_needed else "responded",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", inquiry["id"]).execute()
            
            # Log to chat_history
            if inquiry["user_id"]:
                supabase.table("chat_history").insert({
                    "user_id": inquiry["user_id"],
                    "inquiry_text": inquiry["inquiry_text"],
                    "response_text": response,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }).execute()
            
            # Notify customer service for escalated queries
            if escalation_needed:
                supabase.table("escalated_queries").insert({
                    "inquiry_id": inquiry["id"],
                    "inquiry_text": inquiry["inquiry_text"],
                    "user_id": inquiry["user_id"],
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }).execute()
        
        return inquiries
    except Exception as e:
        logger.error(f"Error in handle_inquiries: {str(e)}")
        raise

@app.route('/inquiry', methods=['POST'])
def inquiry_endpoint():
    try:
        data = request.get_json()
        inquiry_text = data.get("inquiry_text")
        user_id = data.get("user_id")  # Optional user identifier
        if not inquiry_text:
            return jsonify({"error": "Inquiry text is required"}), 400
        
        response, escalation_needed = process_inquiry(inquiry_text, user_id)
        
        # Store in customer_inquiries
        inquiry_data = {
            "inquiry_text": inquiry_text,
            "response_text": response,
            "status": "escalated" if escalation_needed else "responded",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if user_id:
            inquiry_data["user_id"] = user_id
        
        supabase.table("customer_inquiries").insert(inquiry_data).execute()
        
        # Store in chat_history if user_id is provided
        if user_id:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "inquiry_text": inquiry_text,
                "response_text": response,
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        
        # Notify customer service for escalated queries
        if escalation_needed:
            supabase.table("escalated_queries").insert({
                "inquiry_id": supabase.table("customer_inquiries").select("id").eq("inquiry_text", inquiry_text).order("created_at", desc=True).limit(1).execute().data[0]["id"],
                "inquiry_text": inquiry_text,
                "user_id": user_id,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        
        return jsonify({"response": response}), 200
    except Exception as e:
        logger.error(f"Error in /inquiry: {str(e)}")
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
    return jsonify({"message": "Sellytics Chatbot AI Agent Backend"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))