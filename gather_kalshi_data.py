import requests
import csv
from datetime import date
import time

# Now Gather historical Kalshi Data for the weather High of Tomorrow or Today in Chicago.
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def get_chicago_weather_history():
    """
    Fetch all historical KXHIGHCHI (Chicago high temp) markets from Kalshi.
    Writes to CSV as it goes.
    """
    csv_file = "kalshi_chicago_weather.csv"
    is_first_batch = True
    cursor = None
    total_records = 0
    
    while True:
        params = {
            "series_ticker": "KXHIGHCHI",
            "status": "settled",
            "limit": 1000
        }
        
        if cursor:
            params["cursor"] = cursor
        
        response = requests.get(f"{BASE_URL}/markets", params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
            break
        
        data = response.json()
        markets = data.get("markets", [])
        
        if not markets:
            print("No more markets found")
            break
        
        # Parse markets into clean records
        records = []
        for m in markets:
            record = {
                "ticker": m.get("ticker"),
                "title": m.get("title"),
                "subtitle": m.get("subtitle"),
                "event_ticker": m.get("event_ticker"),
                "close_time": m.get("close_time"),
                "expiration_time": m.get("expiration_time"),
                "result": m.get("result"),
                "last_price": m.get("last_price"),
                "volume": m.get("volume"),
                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
            }
            records.append(record)
        
        # Write to CSV
        mode = 'w' if is_first_batch else 'a'
        with open(csv_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            if is_first_batch:
                writer.writeheader()
                is_first_batch = False
            writer.writerows(records)
        
        total_records += len(records)
        print(f"Wrote {len(records)} records, total: {total_records}")
        
        # Get cursor for next page
        cursor = data.get("cursor")
        if not cursor:
            print("No more pages")
            break
        
        time.sleep(0.25)  # Rate limiting
    
    print(f"\nComplete! Total records written: {total_records}")
    print(f"Saved to {csv_file}")


if __name__ == "__main__":
    print("Fetching KXHIGHCHI historical markets...")
    get_chicago_weather_history()
