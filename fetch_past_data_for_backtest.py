"""
KALSHI HISTORICAL DATA FETCHER
==============================
Fetches ALL settled weather markets from Kalshi API for backtesting.

This gives us REAL data:
- What contracts existed each day
- What prices they traded at (last_price)
- Which bin actually won (result = 'yes' or 'no')
- The actual settlement value

No authentication required for public market data!
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import re
import os

# ============ CONFIGURATION ============
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Weather series we're interested in
SERIES = {
    "KXHIGHCHI": "Chicago (Midway)",
    "KXHIGHNY": "NYC (Central Park)", 
    "KXHIGHMIA": "Miami (MIA)"
}

# ============ FUNCTIONS ============

def fetch_all_settled_markets(series_ticker, limit=200):
    """
    Fetch ALL settled markets for a given series.
    Uses pagination to get complete history.
    """
    all_markets = []
    cursor = None
    page = 1
    
    print(f"\nFetching settled markets for {series_ticker}...")
    
    while True:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": limit
        }
        if cursor:
            params["cursor"] = cursor
        
        response = requests.get(f"{BASE_URL}/markets", params=params)
        
        if response.status_code != 200:
            print(f"  Error: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            break
        
        data = response.json()
        markets = data.get("markets", [])
        
        if not markets:
            break
        
        all_markets.extend(markets)
        print(f"  Page {page}: Retrieved {len(markets)} markets (total: {len(all_markets)})")
        
        # Check for pagination cursor
        cursor = data.get("cursor")
        if not cursor:
            break
        
        page += 1
        time.sleep(0.2)  # Rate limiting
    
    print(f"  ✅ Total settled markets: {len(all_markets)}")
    return all_markets


def parse_market_data(market):
    """
    Parse a market response into useful fields.
    """
    subtitle = market.get("subtitle", "")
    
    temp_low = None
    temp_high = None
    market_type = "range" # Default assumption
    
    # 1. Check for "or below" / "or above"
    if "or below" in subtitle:
        market_type = "below"
        # Look for digits, allowing for optional degree symbol
        match = re.search(r'(\d+)', subtitle)
        if match:
            temp_high = int(match.group(1))
            temp_low = temp_high - 100
    elif "or above" in subtitle:
        market_type = "above"
        match = re.search(r'(\d+)', subtitle)
        if match:
            temp_low = int(match.group(1))
            temp_high = temp_low + 100
            
    else:
        # 2. Handle ranges like "30° to 31°" or "30 to 31"
        # This regex finds all distinct groups of digits
        numbers = re.findall(r'(\d+)', subtitle)
        
        # We need at least 2 numbers for a range (low and high)
        if len(numbers) >= 2:
            market_type = "range"
            temp_low = int(numbers[0])
            temp_high = int(numbers[1])
        else:
            # If we couldn't parse 2 numbers, mark as unknown so we don't do math on it later
            market_type = "unknown"

    # ... (Rest of the function remains exactly the same as your original)
    
    # Parse dates
    close_time = market.get("close_time", "")
    event_ticker = market.get("event_ticker", "")
    target_date = None
    
    # Parse event ticker like "KXHIGHNY-25DEC04"
    date_match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})', event_ticker)
    if date_match:
        year = int("20" + date_match.group(1))
        month_str = date_match.group(2)
        day = int(date_match.group(3))
        
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        month = month_map.get(month_str, 1)
        
        try:
            target_date = datetime(year, month, day).strftime("%Y-%m-%d")
        except:
            pass
    
    return {
        "ticker": market.get("ticker"),
        "event_ticker": event_ticker,
        "target_date": target_date,
        "subtitle": subtitle,
        "market_type": market_type,
        "temp_low": temp_low,
        "temp_high": temp_high,
        "result": market.get("result"),
        "last_price": market.get("last_price"),
        "volume": market.get("volume"),
        "close_time": close_time,
        "settlement_value": market.get("settlement_value"),
    }


def find_winning_bin(markets_for_date):
    """
    Given all markets for a single date, find which bin won (result = 'yes').
    Returns the actual temperature range that occurred.
    """
    for m in markets_for_date:
        if m["result"] == "yes":
            return m
    return None


def reconstruct_actual_temp(winning_market):
    """
    Reconstruct the actual temperature from the winning market.
    """
    if not winning_market:
        return None
    
    # SAFETY CHECK: If parsing failed, return None instead of crashing
    if winning_market["temp_low"] is None and winning_market["temp_high"] is None:
        return None
        
    if winning_market["market_type"] == "range":
        # Ensure we have both numbers before doing math
        if winning_market["temp_low"] is not None and winning_market["temp_high"] is not None:
            return (winning_market["temp_low"] + winning_market["temp_high"]) / 2
        return None
        
    elif winning_market["market_type"] == "below":
        if winning_market["temp_high"] is not None:
            return winning_market["temp_high"] - 1
            
    elif winning_market["market_type"] == "above":
        if winning_market["temp_low"] is not None:
            return winning_market["temp_low"] + 1
    
    return None


def fetch_and_process_series(series_ticker, series_name):
    """
    Fetch all data for a series and organize by date.
    """
    # Fetch all settled markets
    raw_markets = fetch_all_settled_markets(series_ticker)
    
    if not raw_markets:
        return None
    
    # Parse all markets
    parsed = [parse_market_data(m) for m in raw_markets]
    
    # Group by target date
    by_date = {}
    for m in parsed:
        if m["target_date"]:
            if m["target_date"] not in by_date:
                by_date[m["target_date"]] = []
            by_date[m["target_date"]].append(m)
    
    # Build daily summaries
    daily_data = []
    
    for date, markets in sorted(by_date.items()):
        winning_market = find_winning_bin(markets)
        actual_temp = reconstruct_actual_temp(winning_market)
        
        # Find the market with highest last_price (market's top pick)
        top_market = max(markets, key=lambda x: x["last_price"] or 0)
        
        # Calculate what the "implied" market temperature was
        if top_market["market_type"] == "range":
            market_implied_temp = (top_market["temp_low"] + top_market["temp_high"]) / 2
        else:
            market_implied_temp = top_market["temp_low"] or top_market["temp_high"]
        
        daily_data.append({
            "date": date,
            "series": series_ticker,
            "city": series_name,
            "actual_temp_approx": actual_temp,
            "winning_bin": winning_market["subtitle"] if winning_market else None,
            "winning_bin_low": winning_market["temp_low"] if winning_market else None,
            "winning_bin_high": winning_market["temp_high"] if winning_market else None,
            "market_top_pick": top_market["subtitle"],
            "market_top_price": top_market["last_price"],
            "market_implied_temp": market_implied_temp,
            "total_contracts": len(markets),
            "total_volume": sum(m["volume"] or 0 for m in markets),
            # Include all contract data for detailed analysis
            "contracts": markets
        })
    
    return daily_data


def get_script_dir():
    """Get the directory where this script is located."""
    import os
    return os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


def save_data(all_data, filename="kalshi_weather_history.json"):
    """Save to JSON for later analysis."""
    # Save to same directory as script
    script_dir = get_script_dir()
    filepath = os.path.join(script_dir, filename)
    
    try:
        with open(filepath, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"\n✅ Saved JSON to {filepath}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        # Try current directory as fallback
        with open(filename, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"✅ Saved JSON to {os.path.abspath(filename)}")


def create_backtest_csv(all_data, filename="kalshi_backtest_data.csv"):
    """
    Create a CSV suitable for backtesting with key fields.
    """
    rows = []
    
    for day in all_data:
        # For each contract on this day
        for contract in day["contracts"]:
            rows.append({
                "date": day["date"],
                "city": day["city"],
                "series": day["series"],
                "contract_subtitle": contract["subtitle"],
                "market_type": contract["market_type"],
                "temp_low": contract["temp_low"],
                "temp_high": contract["temp_high"],
                "last_price_cents": contract["last_price"],
                "volume": contract["volume"],
                "result": contract["result"],  # "yes" or "no" - THE KEY FIELD
                "actual_temp_approx": day["actual_temp_approx"],
            })
    
    if not rows:
        print("❌ No data to save!")
        return None
    
    df = pd.DataFrame(rows)
    
    # Save to same directory as script
    script_dir = get_script_dir()
    filepath = os.path.join(script_dir, filename)
    
    try:
        df.to_csv(filepath, index=False)
        print(f"✅ Saved CSV to {filepath}")
    except Exception as e:
        print(f"⚠️ Could not save to script dir: {e}")
        # Try current directory as fallback
        df.to_csv(filename, index=False)
        filepath = os.path.abspath(filename)
        print(f"✅ Saved CSV to {filepath}")
    
    print(f"   Total rows: {len(df)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def print_summary(all_data):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("KALSHI WEATHER DATA SUMMARY")
    print("="*70)
    
    # Group by city
    by_city = {}
    for day in all_data:
        city = day["city"]
        if city not in by_city:
            by_city[city] = []
        by_city[city].append(day)
    
    for city, days in by_city.items():
        print(f"\n{city}:")
        print(f"  Days with data: {len(days)}")
        
        if days:
            dates = [d["date"] for d in days]
            print(f"  Date range: {min(dates)} to {max(dates)}")
            
            # How often did market's top pick win?
            correct = sum(1 for d in days if d["winning_bin"] == d["market_top_pick"])
            print(f"  Market accuracy (top pick won): {correct}/{len(days)} ({correct/len(days)*100:.1f}%)")
            
            # Average top pick price
            avg_price = sum(d["market_top_price"] or 0 for d in days) / len(days)
            print(f"  Avg top contract price: {avg_price:.0f}¢")


# ============ MAIN ============

if __name__ == "__main__":
    print("="*70)
    print("KALSHI WEATHER HISTORICAL DATA FETCHER")
    print("="*70)
    
    all_data = []
    
    for series_ticker, series_name in SERIES.items():
        print(f"\n{'='*50}")
        print(f"Processing: {series_name} ({series_ticker})")
        print("="*50)
        
        data = fetch_and_process_series(series_ticker, series_name)
        
        if data:
            all_data.extend(data)
            print(f"  Retrieved {len(data)} days of data")
    
    if all_data:
        # Save raw data
        save_data(all_data, "kalshi_weather_history.json")
        
        # Create backtest CSV
        df = create_backtest_csv(all_data, "kalshi_backtest_data.csv")
        
        # Print summary
        print_summary(all_data)
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Run the backtester with: python backtest_with_kalshi_data.py")
        print("2. The CSV contains REAL Kalshi odds and outcomes")
        print("3. We can now test our model against ACTUAL market prices!")
    else:
        print("\n❌ No data retrieved. Check your internet connection.")