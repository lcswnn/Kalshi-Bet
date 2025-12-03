import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Get tomorrow's open markets for Chicago weather
params = {
    "series_ticker": "KXHIGHCHI",
    "status": "open",
    "limit": 100
}

response = requests.get(f"{BASE_URL}/markets", params=params)
markets = response.json().get("markets", [])

for m in markets:
    print(f"{m['ticker']}: {m['subtitle']}")
    print(f"  Yes price: {m['yes_bid']}¢ - {m['yes_ask']}¢")
    print(f"  Implied prob: ~{m['last_price']}%")
    print()