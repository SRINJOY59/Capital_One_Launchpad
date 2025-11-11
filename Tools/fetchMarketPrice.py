import requests
import sys
import json

API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001a52cfa0cf9df446369ab0b90dbcd0df1"

def fetch_market_price(state_name="Karnataka"):
    print(f"[INFO] Fetching market prices for state: {state_name}")
    params = {
        "api-key": API_KEY,
        "format": "json",
        "offset": "0",
        "limit": "4000",
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
    except requests.RequestException as e:
        error_msg = f"Failed to fetch data: {e}"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg, "data": []}
    
    records = data.get("records", [])
    filtered = []
    
    for record in records:
        if record.get("state", "").strip().lower() == state_name.strip().lower():
            filtered.append({
                "state": record.get("state"),
                "district": record.get("district"),
                "market": record.get("market"),
                "commodity": record.get("commodity"),
                "variety": record.get("variety"),
                "min_price": record.get("min_price"),
                "max_price": record.get("max_price"),
                "modal_price": record.get("modal_price"),
                "arrival_date": record.get("arrival_date", "")
            })
    
    if filtered:
        print(f"[INFO] Found {len(filtered)} records for state: {state_name}")
        result = {"success": True, "state": state_name, "count": len(filtered), "data": filtered}
    else:
        print(f"[INFO] No records found for state: {state_name}")
        result = {"success": False, "state": state_name, "count": 0, "data": [], "message": "No records found"}

    return result

if __name__ == "__main__":
    state = 'Karnataka' if len(sys.argv) != 2 else sys.argv[1]
    fetch_market_price(state)
