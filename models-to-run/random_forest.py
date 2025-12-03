import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
from datetime import datetime, timedelta
import re

# ============ TRAIN THE MODEL ============
df = pd.read_csv("weather_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df[["date", "value"]].copy()
df = df.rename(columns={"value": "temp"})

# ============ GET RECENT TEMPS FROM OPEN-METEO ============
print("Fetching recent weather from Open-Meteo...")

lat = 41.8781
lon = -87.6298

open_meteo_url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "daily": ["temperature_2m_max"],
    "past_days": 7,
    "forecast_days": 3,
    "temperature_unit": "fahrenheit",
    "timezone": "America/Chicago"
}

response = requests.get(open_meteo_url, params=params)
meteo_data = response.json()

meteo_dates = meteo_data["daily"]["time"]
meteo_temps = meteo_data["daily"]["temperature_2m_max"]

print(f"Open-Meteo data retrieved:")
for d, t in zip(meteo_dates, meteo_temps):
    print(f"  {d}: {t}°F")

today = datetime.now().date()
for date_str, temp in zip(meteo_dates, meteo_temps):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    if temp is not None and date_obj < today:
        new_row = pd.DataFrame({"date": [pd.to_datetime(date_str)], "temp": [temp]})
        df = pd.concat([df, new_row], ignore_index=True)

df = df.drop_duplicates(subset=["date"], keep="last")
df = df.sort_values("date").reset_index(drop=True)

df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["temp_lag1"] = df["temp"].shift(1)
df["temp_lag2"] = df["temp"].shift(2)
df["temp_lag3"] = df["temp"].shift(3)
df["temp_rolling3"] = df["temp"].shift(1).rolling(window=3).mean()
df["temp_rolling7"] = df["temp"].shift(1).rolling(window=7).mean()
df["temp_change"] = df["temp_lag1"] - df["temp_lag2"]

df = df.dropna().reset_index(drop=True)

df["temp_bin"] = pd.cut(
    df["temp"],
    bins=range(-21, 121, 2),
    labels=[f"{i}-{i+1}" for i in range(-21, 119, 2)]
)

feature_cols = ["day_of_year", "month", "day", "temp_lag1", "temp_lag2", 
                "temp_lag3", "temp_rolling3", "temp_rolling7", "temp_change"]
X = df[feature_cols]
y = df["temp_bin"]

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

print("\nModel trained.\n")

# ============ TARGET DATE ============
target_date = datetime(2025, 12, 3)
target_str = target_date.strftime("%Y-%m-%d")

forecast_temp = None
for d, t in zip(meteo_dates, meteo_temps):
    if d == target_str:
        forecast_temp = t
        break

print(f"Open-Meteo forecast for {target_str}: {forecast_temp}°F")

# ============ PREDICT TARGET DATE ============
recent = df.tail(7).copy()
last_row = recent.iloc[-1]

target_features = {
    "day_of_year": target_date.timetuple().tm_yday,
    "month": target_date.month,
    "day": target_date.day,
    "temp_lag1": last_row["temp"],
    "temp_lag2": recent.iloc[-2]["temp"],
    "temp_lag3": recent.iloc[-3]["temp"],
    "temp_rolling3": recent.tail(3)["temp"].mean(),
    "temp_rolling7": recent["temp"].mean(),
    "temp_change": last_row["temp"] - recent.iloc[-2]["temp"]
}

X_target = pd.DataFrame([target_features])

print(f"\nPredicting for: {target_date.date()}")
print(f"Yesterday's temp: {target_features['temp_lag1']}°")
print(f"3-day avg: {target_features['temp_rolling3']:.1f}°")
print(f"7-day avg: {target_features['temp_rolling7']:.1f}°\n")

probs = model.predict_proba(X_target)[0]
classes = model.classes_
model_probs = dict(zip(classes, probs))

# Show top predictions
prob_df = pd.DataFrame({"bin": classes, "prob": probs})
prob_df = prob_df.sort_values("prob", ascending=False).head(10)

print("Model's top 10 predictions:")
for _, row in prob_df.iterrows():
    print(f"  {row['bin']}°: {row['prob']:.1%}")

# ============ GET KALSHI MARKETS ============
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
params = {"series_ticker": "KXHIGHCHI", "status": "open", "limit": 100}
response = requests.get(f"{BASE_URL}/markets", params=params)
markets = response.json().get("markets", [])

target_markets = [m for m in markets if "25DEC03" in m["ticker"]]

# ============ KALSHI COMPARISON ============
print("\n" + "="*50)
print("KALSHI COMPARISON")
print("="*50)
print(f"Open-Meteo Forecast: {forecast_temp}°F")
print("="*50 + "\n")

print(f"{'Contract':<20} {'Type':<10} {'Kalshi':>8} {'Forecast':>10}")
print("-" * 55)

for m in sorted(target_markets, key=lambda x: x["last_price"], reverse=True):
    subtitle = m["subtitle"]
    kalshi_prob = m["last_price"] / 100
    numbers = re.findall(r'\d+', subtitle)
    
    if "to" in subtitle and len(numbers) >= 2:
        # Band contract: "31° to 32°"
        low = int(numbers[0])
        high = int(numbers[1])
        contract_type = "BAND"
        forecast_result = "YES" if forecast_temp and low <= forecast_temp <= high else "NO"
        
    elif "or below" in subtitle and len(numbers) >= 1:
        # Threshold contract: "30° or below"
        threshold = int(numbers[0])
        contract_type = "BELOW"
        forecast_result = "YES" if forecast_temp and forecast_temp <= threshold else "NO"
        
    elif "or above" in subtitle and len(numbers) >= 1:
        # Threshold contract: "39° or above"
        threshold = int(numbers[0])
        contract_type = "ABOVE"
        forecast_result = "YES" if forecast_temp and forecast_temp >= threshold else "NO"
    else:
        continue
    
    print(f"{subtitle:<20} {contract_type:<10} {kalshi_prob:>7.0%} {forecast_result:>10}")

# ============ BETTING OPPORTUNITIES ============
print("\n" + "="*50)
print("BETTING OPPORTUNITIES (Based on Forecast)")
print("="*50 + "\n")

for m in sorted(target_markets, key=lambda x: x["last_price"], reverse=True):
    subtitle = m["subtitle"]
    kalshi_yes = m["last_price"] / 100
    kalshi_no = 1 - kalshi_yes
    numbers = re.findall(r'\d+', subtitle)
    
    forecast_says_yes = False
    
    if "to" in subtitle and len(numbers) >= 2:
        low = int(numbers[0])
        high = int(numbers[1])
        forecast_says_yes = forecast_temp and low <= forecast_temp <= high
        
    elif "or below" in subtitle and len(numbers) >= 1:
        threshold = int(numbers[0])
        forecast_says_yes = forecast_temp and forecast_temp <= threshold
        
    elif "or above" in subtitle and len(numbers) >= 1:
        threshold = int(numbers[0])
        forecast_says_yes = forecast_temp and forecast_temp >= threshold
    else:
        continue
    
    # Calculate edge
    if forecast_says_yes:
        # Bet YES - forecast says this will happen
        your_price = kalshi_yes
        edge = 1.0 - kalshi_yes  # If forecast is right, true prob is ~100%
        bet_side = "YES"
    else:
        # Bet NO - forecast says this won't happen
        your_price = kalshi_no
        edge = 1.0 - kalshi_no
        bet_side = "NO"
    
    if edge > 0.1:  # Only show if edge > 10%
        print(f"{subtitle}:")
        print(f"  Forecast says: {bet_side}")
        print(f"  Kalshi {bet_side} price: {your_price:.0%}")
        print(f"  Edge (if forecast correct): {edge:.0%}")
        print()

# ============ KELLY CRITERION ============
print("="*50)
print("KELLY CRITERION BET SIZING")
print("="*50 + "\n")

bankroll = 100  # Your Kalshi balance in dollars
forecast_confidence = 0.85  # How much you trust Open-Meteo (0-1)

for m in sorted(target_markets, key=lambda x: x["last_price"], reverse=True):
    subtitle = m["subtitle"]
    kalshi_yes = m["last_price"] / 100
    kalshi_no = 1 - kalshi_yes
    numbers = re.findall(r'\d+', subtitle)
    
    forecast_says_yes = False
    
    if "to" in subtitle and len(numbers) >= 2:
        low = int(numbers[0])
        high = int(numbers[1])
        forecast_says_yes = forecast_temp and low <= forecast_temp <= high
        
    elif "or below" in subtitle and len(numbers) >= 1:
        threshold = int(numbers[0])
        forecast_says_yes = forecast_temp and forecast_temp <= threshold
        
    elif "or above" in subtitle and len(numbers) >= 1:
        threshold = int(numbers[0])
        forecast_says_yes = forecast_temp and forecast_temp >= threshold
    else:
        continue
    
    if forecast_says_yes:
        bet_side = "YES"
        market_price = kalshi_yes
        our_prob = forecast_confidence
    else:
        bet_side = "NO"
        market_price = kalshi_no
        our_prob = forecast_confidence
    
    # Odds (profit per dollar if you win)
    odds = (1 / market_price) - 1
    
    # Kelly: f = (p * b - q) / b
    p = our_prob
    q = 1 - our_prob
    b = odds
    
    kelly_fraction = (p * b - q) / b
    half_kelly = kelly_fraction / 2
    
    if kelly_fraction > 0.05:  # Only show if Kelly > 5%
        bet_amount = bankroll * half_kelly
        potential_profit = bet_amount * odds
        
        print(f"{subtitle}:")
        print(f"  Bet: {bet_side} at {market_price:.0%}")
        print(f"  Your confidence: {our_prob:.0%}")
        print(f"  Odds: {odds:.2f}:1")
        print(f"  Half Kelly: {half_kelly:.1%} of bankroll")
        print(f"  Suggested bet: ${bet_amount:.2f}")
        print(f"  Potential profit: ${potential_profit:.2f}")
        print()