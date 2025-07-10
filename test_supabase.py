from supabaseClient import supabase

def test_connection():
    try:
        result = supabase.table("dynamic_sales").select("id").limit(1).execute()
        print("Connection successful:", result.data)
    except Exception as e:
        print("Connection failed:", str(e))

if __name__ == "__main__":
    test_connection()

