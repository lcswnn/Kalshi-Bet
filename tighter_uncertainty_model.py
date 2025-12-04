import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import requests
from datetime import datetime, timedelta
import re
from scipy import stats

# ============ CITY CONFIGURATION ============
cities = {
    "chicago": {
        "name": "Chicago",
        "csv_file": "weather_data_chicago.csv",
        "lat": 41.8781,
        "lon": -87.6298,
        "kalshi_series": "KXHIGHCHI"
    },
    "nyc": {
        "name": "New York City",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.7128,
        "lon": -74.0060,
        "kalshi_series": "KXHIGHNY"
    }
}

# ============ SELECT CITY ============
print("=" * 60)
print("KALSHI WEATHER BETTING MODEL v2")
print("(Tighter Uncertainty Based on Open-Meteo Accuracy)")
print("=" * 60)
print("\nAvailable cities:")
print("  1. Chicago")
print("  2. New York City")
print("  3. Both")

choice = input("\nSelect city (1/2/3): ").strip()

if choice == "1":
    selected_cities = ["chicago"]
elif choice == "2":
    selected_cities = ["nyc"]
elif choice == "3":
    selected_cities = ["chicago", "nyc"]
else:
    print("Invalid choice, defaulting to Chicago")
    selected_cities = ["chicago"]

# ============ BACKTEST MODE ============
backtest_mode = input("\nRun backtest for today (Dec 3)? (y/n): ").strip().lower() == "y"

# ============ FUNCTIONS ============
def load_and_prepare_data(city_config):
    df = pd.read_csv(city_config["csv_file"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "value"]].copy()
    df = df.rename(columns={"value": "temp"})
    return df


def fetch_open_meteo(lat, lon):
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "precipitation_sum",
            "pressure_msl_mean",
        ],
        "past_days": 7,
        "forecast_days": 3,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Chicago"
    }
    response = requests.get(open_meteo_url, params=params)
    return response.json()


def build_features(df):
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["temp_lag1"] = df["temp"].shift(1)
    df["temp_lag2"] = df["temp"].shift(2)
    df["temp_lag3"] = df["temp"].shift(3)
    df["temp_rolling3"] = df["temp"].shift(1).rolling(window=3).mean()
    df["temp_rolling7"] = df["temp"].shift(1).rolling(window=7).mean()
    df["temp_change"] = df["temp_lag1"] - df["temp_lag2"]
    
    seasonal_avg = df.groupby("day_of_year")["temp"].mean().to_dict()
    df["seasonal_avg"] = df["day_of_year"].map(seasonal_avg)
    df["simulated_forecast"] = (df["temp_lag1"] * 0.6) + (df["seasonal_avg"] * 0.4)
    
    return df.dropna().reset_index(drop=True)


def train_model(df):
    feature_cols = [
        "day_of_year", "month", "day", 
        "temp_lag1", "temp_lag2", "temp_lag3", 
        "temp_rolling3", "temp_rolling7", "temp_change",
        "simulated_forecast"
    ]
    
    X = df[feature_cols]
    y = df["temp"]
    
    split_idx = int(len(df) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    val_predictions = model.predict(X_val)
    model_error_std = np.std(y_val.values - val_predictions)
    mae = mean_absolute_error(y_val, val_predictions)
    
    model.fit(X, y)
    
    return model, model_error_std, mae, feature_cols


def calc_bin_probability(low, high, mean, std):
    return stats.norm.cdf(high, mean, std) - stats.norm.cdf(low, mean, std)

def calc_below_probability(threshold, mean, std):
    return stats.norm.cdf(threshold + 0.5, mean, std)

def calc_above_probability(threshold, mean, std):
    return 1 - stats.norm.cdf(threshold - 0.5, mean, std)


def analyze_city(city_key, backtest=False):
    city = cities[city_key]
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {city['name'].upper()}")
    print(f"{'='*60}")
    
    # Load data
    try:
        df = load_and_prepare_data(city)
        print(f"Loaded {len(df)} historical records")
    except FileNotFoundError:
        print(f"❌ Error: {city['csv_file']} not found.")
        return
    
    # Fetch Open-Meteo data
    print("Fetching Open-Meteo data...")
    meteo_data = fetch_open_meteo(city["lat"], city["lon"])
    
    meteo_dates = meteo_data["daily"]["time"]
    meteo_temps = meteo_data["daily"]["temperature_2m_max"]
    meteo_temps_min = meteo_data["daily"]["temperature_2m_min"]
    meteo_wind = meteo_data["daily"]["wind_speed_10m_max"]
    meteo_wind_dir = meteo_data["daily"]["wind_direction_10m_dominant"]
    meteo_pressure = meteo_data["daily"]["pressure_msl_mean"]
    
    # Display weather data
    print(f"\nOpen-Meteo Data:")
    print(f"{'Date':<12} {'High':>6} {'Low':>6} {'Wind':>6} {'Dir':>5}")
    print("-" * 40)
    for i, d in enumerate(meteo_dates):
        print(f"{d:<12} {meteo_temps[i]:>5.1f}° {meteo_temps_min[i]:>5.1f}° {meteo_wind[i]:>5.1f} {meteo_wind_dir[i]:>5.0f}°")
    
    # Set target date
    today = datetime.now().date()
    
    if backtest:
        # For backtest, predict TODAY (Dec 3) as if we were yesterday
        target_date = today
        target_str = target_date.strftime("%Y-%m-%d")
        print(f"\n🔄 BACKTEST MODE: Predicting for {target_str} (today)")
    else:
        # Normal mode: predict tomorrow
        target_date = today + timedelta(days=1)
        target_str = target_date.strftime("%Y-%m-%d")
    
    # Add recent data (excluding target date for backtest)
    for i, date_str in enumerate(meteo_dates):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        if meteo_temps[i] is not None and date_obj < target_date:
            new_row = pd.DataFrame({"date": [pd.to_datetime(date_str)], "temp": [meteo_temps[i]]})
            df = pd.concat([df, new_row], ignore_index=True)
    
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Build features and train
    df = build_features(df)
    model, model_error_std, mae, feature_cols = train_model(df)
    
    print(f"\nModel trained: MAE = {mae:.2f}°F, Model Std = {model_error_std:.2f}°F")
    
    # Get indices
    today_idx = None
    target_idx = None
    yesterday_idx = None
    
    for i, d in enumerate(meteo_dates):
        if d == today.strftime("%Y-%m-%d"):
            today_idx = i
        if d == target_str:
            target_idx = i
        if d == (target_date - timedelta(days=1)).strftime("%Y-%m-%d"):
            yesterday_idx = i
    
    # Get forecast for target date
    forecast_temp = meteo_temps[target_idx] if target_idx is not None else None
    
    # Weather change indicators
    if yesterday_idx is not None and target_idx is not None:
        temp_drop = meteo_temps[yesterday_idx] - meteo_temps[target_idx]
        wind_dir_target = meteo_wind_dir[target_idx]
        cold_wind = 1 if (wind_dir_target > 270 or wind_dir_target < 90) else 0
        
        print(f"\n{'='*60}")
        print("WEATHER CHANGE INDICATORS")
        print(f"{'='*60}")
        print(f"Day before: {meteo_temps[yesterday_idx]}°F → Target: {meteo_temps[target_idx]}°F")
        print(f"Temperature change: {temp_drop:+.1f}°F {'🥶 COLD FRONT' if temp_drop > 5 else '🌡️ WARMING' if temp_drop < -5 else ''}")
        print(f"Wind direction: {'❄️ Cold (N/NW)' if cold_wind else '🌡️ Warm (S/SW)'}")
        
        cold_front_detected = temp_drop > 5
    else:
        cold_front_detected = False
    
    # ============ KEY CHANGE: TIGHTER UNCERTAINTY ============
    # Open-Meteo is typically accurate within 1-3°F
    # Today it was off by only 0.3°F (28.3° forecast vs 28° actual)
    # Use Open-Meteo's historical accuracy, not our model's error
    
    OPEN_METEO_ACCURACY = 2.5  # Based on typical forecast accuracy (±2.5°F)
    
    # For cold fronts, forecasts can be slightly less accurate
    if cold_front_detected:
        uncertainty = 3.5  # Slightly wider for cold fronts
    else:
        uncertainty = OPEN_METEO_ACCURACY
    
    print(f"\n{'='*60}")
    print("UNCERTAINTY MODEL (v2 - Open-Meteo Based)")
    print(f"{'='*60}")
    print(f"Old model uncertainty: ±{model_error_std:.1f}°F (too wide)")
    print(f"New uncertainty: ±{uncertainty:.1f}°F (based on Open-Meteo accuracy)")
    
    # Make prediction - trust Open-Meteo heavily
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
        "temp_change": last_row["temp"] - recent.iloc[-2]["temp"],
        "simulated_forecast": forecast_temp
    }
    
    X_target = pd.DataFrame([target_features])
    model_prediction = model.predict(X_target)[0]
    
    # Final prediction: heavily weight Open-Meteo (90%)
    forecast_weight = 0.90
    predicted_temp = (forecast_temp * forecast_weight) + (model_prediction * (1 - forecast_weight))
    
    print(f"\n{'='*60}")
    print("PREDICTION")
    print(f"{'='*60}")
    if cold_front_detected:
        print("🥶 COLD FRONT DETECTED")
    print(f"Open-Meteo Forecast: {forecast_temp}°F")
    print(f"Model Prediction: {model_prediction:.1f}°F")
    print(f"Final Prediction: {predicted_temp:.1f}°F")
    print(f"Tight Uncertainty: ±{uncertainty}°F")
    print(f"68% confidence range: {predicted_temp - uncertainty:.1f}°F to {predicted_temp + uncertainty:.1f}°F")
    print(f"95% confidence range: {predicted_temp - 2*uncertainty:.1f}°F to {predicted_temp + 2*uncertainty:.1f}°F")
    
    # Backtest: show actual result
    if backtest and city_key == "chicago":
        actual_temp = 28  # From NOAA
        print(f"\n🎯 ACTUAL RESULT: {actual_temp}°F")
        print(f"   Forecast error: {abs(forecast_temp - actual_temp):.1f}°F")
        print(f"   Within 68% range: {'✅ YES' if abs(predicted_temp - actual_temp) <= uncertainty else '❌ NO'}")
    
    # Get Kalshi markets
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"series_ticker": city["kalshi_series"], "status": "open", "limit": 100}
    response = requests.get(f"{BASE_URL}/markets", params=params)
    markets = response.json().get("markets", [])
    
    target_kalshi = target_date.strftime("%y%b%d").upper()
    target_markets = [m for m in markets if target_kalshi in m["ticker"]]
    
    # For backtest, we need to simulate what Kalshi prices were
    # We'll use current prices as approximation (not perfect but illustrative)
    
    # Find Kalshi's top prediction
    kalshi_top_contract = None
    kalshi_top_prob = 0
    for m in target_markets:
        if m["last_price"] > kalshi_top_prob:
            kalshi_top_prob = m["last_price"]
            kalshi_top_contract = m["subtitle"]
    
    kalshi_implied_temp = None
    if kalshi_top_contract:
        numbers = re.findall(r'\d+', kalshi_top_contract)
        if "to" in kalshi_top_contract and len(numbers) >= 2:
            kalshi_implied_temp = (int(numbers[0]) + int(numbers[1])) / 2
    
    print(f"\n{'='*60}")
    print("MARKET EFFICIENCY CHECK")
    print(f"{'='*60}")
    print(f"Open-Meteo Forecast: {forecast_temp}°F")
    print(f"Kalshi Top Contract: {kalshi_top_contract} ({kalshi_top_prob}%)")
    
    market_efficient = False
    if kalshi_implied_temp:
        print(f"Kalshi Implied Temp: ~{kalshi_implied_temp}°F")
        forecast_vs_kalshi_diff = abs(forecast_temp - kalshi_implied_temp)
        print(f"Forecast vs Kalshi Diff: {forecast_vs_kalshi_diff:.1f}°F")
        
        if forecast_vs_kalshi_diff <= 2:
            print("\n✅ MARKET IS EFFICIENTLY PRICED")
            market_efficient = True
        else:
            print(f"\n🎯 POTENTIAL MISPRICING: {forecast_vs_kalshi_diff:.1f}°F difference!")
    
    # Compare with Kalshi using TIGHT uncertainty
    print(f"\n{'='*60}")
    print("MODEL vs KALSHI COMPARISON (Tight Uncertainty)")
    print(f"{'='*60}\n")
    
    print(f"{'Contract':<20} {'Model':>10} {'Kalshi':>10} {'Edge':>10}")
    print("-" * 55)
    
    results = []
    
    for m in sorted(target_markets, key=lambda x: x["last_price"], reverse=True):
        subtitle = m["subtitle"]
        kalshi_prob = m["last_price"] / 100
        numbers = re.findall(r'\d+', subtitle)
        
        model_prob = 0
        contract_temp = None
        
        if "to" in subtitle and len(numbers) >= 2:
            low = int(numbers[0])
            high = int(numbers[1])
            contract_temp = (low + high) / 2
            model_prob = calc_bin_probability(low, high + 1, predicted_temp, uncertainty)
            
        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            contract_temp = threshold - 2
            model_prob = calc_below_probability(threshold, predicted_temp, uncertainty)
            
        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            contract_temp = threshold + 2
            model_prob = calc_above_probability(threshold, predicted_temp, uncertainty)
        else:
            continue
        
        edge = model_prob - kalshi_prob
        
        # Check if contradicts forecast
        contradicts_forecast = False
        if "above" in subtitle:
            threshold = int(re.findall(r'\d+', subtitle)[0])
            if forecast_temp < (threshold - 5):
                contradicts_forecast = True
        elif "below" in subtitle:
            threshold = int(re.findall(r'\d+', subtitle)[0])
            if forecast_temp > (threshold + 5):
                contradicts_forecast = True
        
        results.append({
            "subtitle": subtitle,
            "model_prob": model_prob,
            "kalshi_prob": kalshi_prob,
            "edge": edge,
            "contradicts_forecast": contradicts_forecast
        })
        
        print(f"{subtitle:<20} {model_prob:>9.1%} {kalshi_prob:>9.0%} {edge:>+9.1%}")
    
    # Betting recommendations
    print(f"\n{'='*60}")
    print("BETTING RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    bankroll = 20
    model_confidence = 0.85  # Higher confidence with tight uncertainty
    good_bets = []
    
    for r in sorted(results, key=lambda x: abs(x["edge"]), reverse=True):
        edge = r["edge"]
        
        # Filters
        if abs(edge) <= 0.10:  # Require bigger edge now (10%)
            continue
        if r["contradicts_forecast"]:
            continue
        if "above" in r["subtitle"]:
            threshold = int(re.findall(r'\d+', r["subtitle"])[0])
            if forecast_temp < threshold - 3:
                continue
        if "below" in r["subtitle"]:
            threshold = int(re.findall(r'\d+', r["subtitle"])[0])
            if forecast_temp > threshold + 3:
                continue
        
        if edge > 0:
            bet_side = "YES"
            our_prob = r["model_prob"] * model_confidence + r["kalshi_prob"] * (1 - model_confidence)
            market_price = r["kalshi_prob"]
        else:
            bet_side = "NO"
            our_prob = (1 - r["model_prob"]) * model_confidence + (1 - r["kalshi_prob"]) * (1 - model_confidence)
            market_price = 1 - r["kalshi_prob"]
        
        if market_price > 0:
            odds = (1 / market_price) - 1
            p = our_prob
            q = 1 - our_prob
            b = odds
            
            kelly_fraction = (p * b - q) / b if b > 0 else 0
            half_kelly = max(kelly_fraction / 2, 0)
            
            if half_kelly > 0.02:
                bet_amount = bankroll * half_kelly
                potential_profit = bet_amount * odds
                
                good_bets.append({
                    "subtitle": r["subtitle"],
                    "bet_side": bet_side,
                    "market_price": market_price,
                    "model_prob": r["model_prob"],
                    "our_prob": our_prob,
                    "odds": odds,
                    "half_kelly": half_kelly,
                    "bet_amount": bet_amount,
                    "potential_profit": potential_profit,
                    "edge": abs(edge)
                })
    
    if len(good_bets) == 0:
        if market_efficient:
            print("📊 NO CLEAR EDGE TODAY")
            print(f"   Market matches forecast within 2°F")
        else:
            print("⚠️ NO RECOMMENDED BETS")
            print("   Edges too small or contradict forecast")
    else:
        for bet in good_bets:
            print(f"{bet['subtitle']}:")
            print(f"  Bet: {bet['bet_side']} at {bet['market_price']:.0%}")
            print(f"  Model prob: {bet['model_prob']:.1%}")
            print(f"  Edge: {bet['edge']:.1%}")
            print(f"  Odds: {bet['odds']:.2f}:1")
            print(f"  Half Kelly: {bet['half_kelly']:.1%}")
            print(f"  Suggested bet: ${bet['bet_amount']:.2f}")
            print(f"  Potential profit: ${bet['potential_profit']:.2f}")
            print()
    
    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY - {city['name'].upper()}")
    print(f"{'='*60}")
    print(f"Target Date: {target_str}")
    print(f"Open-Meteo Forecast: {forecast_temp}°F")
    print(f"Final Prediction: {predicted_temp:.1f}°F (±{uncertainty}°F)")
    print(f"Kalshi Top Pick: {kalshi_top_contract} ({kalshi_top_prob}%)")
    print(f"Market Efficiency: {'✅ Efficient' if market_efficient else '🎯 Potential Edge'}")
    print(f"Recommended Bets: {len(good_bets)}")
    
    if backtest and city_key == "chicago":
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        actual = 28
        print(f"Actual High: {actual}°F")
        print(f"Forecast was: {forecast_temp}°F (error: {abs(forecast_temp - actual):.1f}°F)")
        print(f"Would '30° or below' YES have won? {'✅ YES' if actual <= 30 else '❌ NO'}")
        

# ============ RUN ANALYSIS ============
if __name__ == "__main__":
    for city_key in selected_cities:
        analyze_city(city_key, backtest=backtest_mode)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")