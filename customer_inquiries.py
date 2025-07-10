from supabaseClient import supabase
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime

nltk.download('punkt_tab', quiet=True)

def process_inquiry(inquiry_text):
    tokens = word_tokenize(inquiry_text.lower())
    responses = {
        "stock": "Please check the inventory dashboard or contact support for stock details.",
        "availability": "Availability can be checked in real-time on the platform.",
        "order": "Orders can be placed through the platform or by contacting support.",
        "delivery": "Delivery timelines depend on your location. Please provide more details.",
        "price": "Pricing details are available in the product catalog."
    }
    for key, response in responses.items():
        if key in tokens:
            return response
    return "Thank you for your inquiry. Please provide more details or contact support."

def handle_inquiries():
    inquiries = supabase.table("customer_inquiries").select("id, inquiry_text").eq("status", "pending").execute().data
    print(f"Found {len(inquiries)} pending inquiries")
    for inquiry in inquiries:
        response = process_inquiry(inquiry["inquiry_text"])
        print(f"Processing inquiry {inquiry['id']}: {inquiry['inquiry_text']} -> {response}")
        supabase.table("customer_inquiries").update({
            "response_text": response,
            "status": "responded",
            "created_at": datetime.utcnow().isoformat()
        }).eq("id", inquiry["id"]).execute()

def handler(request):
    try:
        handle_inquiries()
        return {"statusCode": 200, "body": "Inquiries processed"}
    except Exception as e:
        return {"statusCode": 500, "body": str(e)}

if __name__ == "__main__":
    handle_inquiries()