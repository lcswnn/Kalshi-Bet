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
print("KALSHI WEATHER BETTING MODEL")
print("=" * 60)
print("\nAvailable cities:")
print("  1. Chicago")
print("  2. New York City")
print("  3. Both (run analysis for both)")

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

# ============ FUNCTIONS ============
def load_and_prepare_data(city_config):
    """Load historical data and add features."""
    df = pd.read_csv(city_config["csv_file"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "value"]].copy()
    df = df.rename(columns={"value": "temp"})
    return df


def fetch_open_meteo(lat, lon):
    """Fetch weather data from Open-Meteo."""
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
    """Add all features to dataframe."""
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
    """Train the regression model."""
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
    error_std = np.std(y_val.values - val_predictions)
    mae = mean_absolute_error(y_val, val_predictions)
    
    # Retrain on full data
    model.fit(X, y)
    
    return model, error_std, mae, feature_cols


def calc_bin_probability(low, high, mean, std):
    return stats.norm.cdf(high, mean, std) - stats.norm.cdf(low, mean, std)

def calc_below_probability(threshold, mean, std):
    return stats.norm.cdf(threshold + 0.5, mean, std)

def calc_above_probability(threshold, mean, std):
    return 1 - stats.norm.cdf(threshold - 0.5, mean, std)


def analyze_city(city_key):
    """Run full analysis for a city."""
    city = cities[city_key]
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {city['name'].upper()}")
    print(f"{'='*60}")
    
    # Load data
    try:
        df = load_and_prepare_data(city)
        print(f"Loaded {len(df)} historical records")
    except FileNotFoundError:
        print(f"❌ Error: {city['csv_file']} not found. Run the weather fetch script first.")
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
    print(f"\nOpen-Meteo Extended Data:")
    print(f"{'Date':<12} {'High':>6} {'Low':>6} {'Wind':>6} {'Dir':>5} {'Pressure':>10}")
    print("-" * 50)
    for i, d in enumerate(meteo_dates):
        print(f"{d:<12} {meteo_temps[i]:>5.1f}° {meteo_temps_min[i]:>5.1f}° {meteo_wind[i]:>5.1f} {meteo_wind_dir[i]:>5.0f}° {meteo_pressure[i]:>9.1f}")
    
    # Add recent data to dataframe
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    today_str = today.strftime("%Y-%m-%d")
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    
    for i, date_str in enumerate(meteo_dates):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        if meteo_temps[i] is not None and date_obj < today:
            new_row = pd.DataFrame({"date": [pd.to_datetime(date_str)], "temp": [meteo_temps[i]]})
            df = pd.concat([df, new_row], ignore_index=True)
    
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Build features and train
    df = build_features(df)
    model, error_std, mae, feature_cols = train_model(df)
    
    print(f"\nModel trained: MAE = {mae:.2f}°F, Std = {error_std:.2f}°F")
    
    # Get today and tomorrow indices
    today_idx = None
    tomorrow_idx = None
    for i, d in enumerate(meteo_dates):
        if d == today_str:
            today_idx = i
        if d == tomorrow_str:
            tomorrow_idx = i
    
    # Weather change indicators
    if today_idx is not None and tomorrow_idx is not None:
        temp_drop = meteo_temps[today_idx] - meteo_temps[tomorrow_idx]
        wind_dir_tomorrow = meteo_wind_dir[tomorrow_idx]
        cold_wind = 1 if (wind_dir_tomorrow > 270 or wind_dir_tomorrow < 90) else 0
        wind_change = meteo_wind[tomorrow_idx] - meteo_wind[today_idx]
        
        print(f"\n{'='*60}")
        print("WEATHER CHANGE INDICATORS")
        print(f"{'='*60}")
        print(f"Today: {meteo_temps[today_idx]}°F → Tomorrow: {meteo_temps[tomorrow_idx]}°F")
        print(f"Temperature change: {temp_drop:+.1f}°F {'🥶 COLD FRONT' if temp_drop > 5 else '🌡️ WARMING' if temp_drop < -5 else ''}")
        print(f"Wind direction: {'❄️ Cold (N/NW)' if cold_wind else '🌡️ Warm (S/SW)'}")
    
    # Make prediction
    forecast_temp = meteo_temps[tomorrow_idx] if tomorrow_idx else None
    
    recent = df.tail(7).copy()
    last_row = recent.iloc[-1]
    
    target_features = {
        "day_of_year": tomorrow.timetuple().tm_yday,
        "month": tomorrow.month,
        "day": tomorrow.day,
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
    
    # Cold front adjustment
    cold_front_detected = temp_drop > 5 if today_idx and tomorrow_idx else False
    forecast_weight = 0.85 if cold_front_detected else 0.70
    predicted_temp = (forecast_temp * forecast_weight) + (model_prediction * (1 - forecast_weight))
    
    print(f"\n{'='*60}")
    print("PREDICTION")
    print(f"{'='*60}")
    if cold_front_detected:
        print("🥶 COLD FRONT DETECTED")
    print(f"Open-Meteo Forecast: {forecast_temp}°F")
    print(f"Model Prediction: {model_prediction:.1f}°F")
    print(f"Final Prediction: {predicted_temp:.1f}°F (±{error_std:.1f}°F)")
    
    # Get Kalshi markets
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"series_ticker": city["kalshi_series"], "status": "open", "limit": 100}
    response = requests.get(f"{BASE_URL}/markets", params=params)
    markets = response.json().get("markets", [])
    
    target_kalshi = tomorrow.strftime("%y%b%d").upper()
    target_markets = [m for m in markets if target_kalshi in m["ticker"]]
    
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
        
        if forecast_vs_kalshi_diff <= 3:
            print("\n✅ MARKET IS EFFICIENTLY PRICED")
            market_efficient = True
        else:
            print("\n🎯 POTENTIAL MISPRICING DETECTED")
    
    # Compare with Kalshi
    print(f"\n{'='*60}")
    print("MODEL vs KALSHI COMPARISON")
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
            model_prob = calc_bin_probability(low, high + 1, predicted_temp, error_std)
            
        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            contract_temp = threshold - 2
            model_prob = calc_below_probability(threshold, predicted_temp, error_std)
            
        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            contract_temp = threshold + 2
            model_prob = calc_above_probability(threshold, predicted_temp, error_std)
        else:
            continue
        
        edge = model_prob - kalshi_prob
        
        contradicts_forecast = False
        if contract_temp is not None:
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
    model_confidence = 0.75
    good_bets = []
    
    for r in sorted(results, key=lambda x: abs(x["edge"]), reverse=True):
        edge = r["edge"]
        
        if abs(edge) <= 0.05:
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
                    "potential_profit": potential_profit
                })
    
    if market_efficient and len(good_bets) == 0:
        print("📊 NO CLEAR EDGE TODAY")
        print(f"   The market is pricing around {kalshi_implied_temp}°F")
        print(f"   Open-Meteo forecasts {forecast_temp}°F")
        print("   These are close enough that there's no obvious mispricing.")
    elif len(good_bets) == 0:
        print("⚠️ NO RECOMMENDED BETS")
        print("   All potential edges were filtered out.")
    else:
        for bet in good_bets:
            print(f"{bet['subtitle']}:")
            print(f"  Bet: {bet['bet_side']} at {bet['market_price']:.0%}")
            print(f"  Model prob: {bet['model_prob']:.1%}")
            print(f"  Blended prob: {bet['our_prob']:.1%}")
            print(f"  Odds: {bet['odds']:.2f}:1")
            print(f"  Half Kelly: {bet['half_kelly']:.1%}")
            print(f"  Suggested bet: ${bet['bet_amount']:.2f}")
            print(f"  Potential profit: ${bet['potential_profit']:.2f}")
            print()
    
    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY - {city['name'].upper()}")
    print(f"{'='*60}")
    print(f"Tomorrow's Date: {tomorrow_str}")
    print(f"Open-Meteo Forecast: {forecast_temp}°F")
    print(f"Model Prediction: {predicted_temp:.1f}°F")
    print(f"Kalshi Top Pick: {kalshi_top_contract} ({kalshi_top_prob}%)")
    print(f"Market Efficiency: {'✅ Efficient' if market_efficient else '🎯 Potential Edge'}")
    print(f"Recommended Bets: {len(good_bets)}")


# ============ RUN ANALYSIS ============
if __name__ == "__main__":
    for city_key in selected_cities:
        analyze_city(city_key)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")