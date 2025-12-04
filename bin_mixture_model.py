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
    },
    "la": {
        "name": "Los Angeles",
        "csv_file": "weather_data_la.csv",
        "lat": 34.0522,
        "lon": -118.2437,
        "kalshi_series": "KXHIGHLA"
    }
}
all_bets = []

# Default to all cities
selected_cities = ["chicago", "nyc", "la"]

# ============ RUN ALL CITIES ============
print("=" * 60)
print("KALSHI WEATHER BETTING MODEL v3")
print("(Mixture Model: Spike + Uncertainty)")
print("=" * 60)
print(f"\nAnalyzing: Chicago, New York City, Los Angeles")

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


def get_bin_for_temp(temp):
    """Get the Kalshi bin that contains this temperature."""
    # Kalshi uses 2-degree bins: 16-17, 18-19, 20-21, etc.
    # Bins start at odd numbers
    lower = int(temp) if int(temp) % 2 == 0 else int(temp) - 1
    if temp < 0:
        lower = int(temp) - 1 if int(temp) % 2 == 0 else int(temp)
    
    # Adjust for Kalshi's bin structure (odd-start)
    if lower % 2 == 0:
        lower -= 1
    
    return (lower, lower + 1)


def mixture_probability(contract_low, contract_high, forecast_temp, uncertainty_std, 
                        spike_weight=0.55, cold_front=False):
    """
    Mixture Model: Combines a "spike" at the forecast bin with normal distribution uncertainty.
    
    - spike_weight: probability mass on the exact forecast bin (default 55%)
    - remaining weight: spread via normal distribution for uncertainty
    - cold_front: if True, reduce spike weight (more uncertainty)
    """
    
    if cold_front:
        spike_weight = 0.45  # Less confident during cold fronts
        uncertainty_std = uncertainty_std * 1.3  # Wider uncertainty
    
    spread_weight = 1 - spike_weight
    
    # Get the forecast bin
    forecast_bin = get_bin_for_temp(forecast_temp)
    
    # Check if this contract IS the forecast bin
    is_forecast_bin = (contract_low == forecast_bin[0] or 
                       contract_low == forecast_bin[0] - 1 or
                       contract_low == forecast_bin[0] + 1)
    
    # More precise check: does the forecast fall in this bin?
    forecast_in_this_bin = contract_low <= forecast_temp <= contract_high
    
    # Calculate spread probability (normal distribution)
    spread_prob = stats.norm.cdf(contract_high, forecast_temp, uncertainty_std) - \
                  stats.norm.cdf(contract_low, forecast_temp, uncertainty_std)
    
    if forecast_in_this_bin:
        # This IS the forecast bin - gets the spike
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        # Not the forecast bin - only gets spread probability
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)  # Cap at 99%


def mixture_below_probability(threshold, forecast_temp, uncertainty_std, 
                              spike_weight=0.55, cold_front=False):
    """Mixture probability for 'X or below' contracts."""
    
    if cold_front:
        spike_weight = 0.45
        uncertainty_std = uncertainty_std * 1.3
    
    spread_weight = 1 - spike_weight
    
    # Spread probability
    spread_prob = stats.norm.cdf(threshold + 0.5, forecast_temp, uncertainty_std)
    
    # Does forecast fall below threshold?
    if forecast_temp <= threshold:
        # Forecast is below - spike goes here
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        # Forecast is above - no spike, just spread
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def mixture_above_probability(threshold, forecast_temp, uncertainty_std, 
                              spike_weight=0.55, cold_front=False):
    """Mixture probability for 'X or above' contracts."""
    
    if cold_front:
        spike_weight = 0.45
        uncertainty_std = uncertainty_std * 1.3
    
    spread_weight = 1 - spike_weight
    
    # Spread probability
    spread_prob = 1 - stats.norm.cdf(threshold - 0.5, forecast_temp, uncertainty_std)
    
    # Does forecast fall above threshold?
    if forecast_temp >= threshold:
        # Forecast is above - spike goes here
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        # Forecast is below - no spike, just spread
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def analyze_city(city_key):
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
        print(f"   Run the weather fetch script for {city['name']} first.")
        return
    
    # Fetch Open-Meteo data
    print("Fetching Open-Meteo data...")
    meteo_data = fetch_open_meteo(city["lat"], city["lon"])
    
    meteo_dates = meteo_data["daily"]["time"]
    meteo_temps = meteo_data["daily"]["temperature_2m_max"]
    meteo_temps_min = meteo_data["daily"]["temperature_2m_min"]
    meteo_wind = meteo_data["daily"]["wind_speed_10m_max"]
    meteo_wind_dir = meteo_data["daily"]["wind_direction_10m_dominant"]
    
    # Display weather data
    print(f"\nOpen-Meteo Data:")
    print(f"{'Date':<12} {'High':>6} {'Low':>6} {'Wind':>6} {'Dir':>5}")
    print("-" * 40)
    for i, d in enumerate(meteo_dates):
        print(f"{d:<12} {meteo_temps[i]:>5.1f}° {meteo_temps_min[i]:>5.1f}° {meteo_wind[i]:>5.1f} {meteo_wind_dir[i]:>5.0f}°")
    
    # Set target date (tomorrow)
    today = datetime.now().date()
    target_date = today + timedelta(days=1)
    target_str = target_date.strftime("%Y-%m-%d")
    
    # Add recent data
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
    
    print(f"\nModel trained: MAE = {mae:.2f}°F")
    
    # Get indices
    today_idx = None
    target_idx = None
    
    for i, d in enumerate(meteo_dates):
        if d == today.strftime("%Y-%m-%d"):
            today_idx = i
        if d == target_str:
            target_idx = i
    
    # Get forecast
    forecast_temp = meteo_temps[target_idx] if target_idx is not None else None
    
    # Weather change indicators
    cold_front_detected = False
    if today_idx is not None and target_idx is not None:
        temp_drop = meteo_temps[today_idx] - meteo_temps[target_idx]
        wind_dir_target = meteo_wind_dir[target_idx]
        cold_wind = 1 if (wind_dir_target > 270 or wind_dir_target < 90) else 0
        
        print(f"\n{'='*60}")
        print("WEATHER CHANGE INDICATORS")
        print(f"{'='*60}")
        print(f"Today: {meteo_temps[today_idx]}°F → Tomorrow: {meteo_temps[target_idx]}°F")
        print(f"Temperature change: {temp_drop:+.1f}°F {'🥶 COLD FRONT' if temp_drop > 5 else '🌡️ WARMING' if temp_drop < -5 else ''}")
        print(f"Wind direction: {'❄️ Cold (N/NW)' if cold_wind else '🌡️ Warm (S/SW)'}")
        
        cold_front_detected = temp_drop > 5
    
    # ============ MIXTURE MODEL PARAMETERS ============
    UNCERTAINTY_STD = 2.5  # Base uncertainty
    SPIKE_WEIGHT = 0.55    # 55% weight on forecast bin
    
    forecast_bin = get_bin_for_temp(forecast_temp)
    
    print(f"\n{'='*60}")
    print("MIXTURE MODEL (v3)")
    print(f"{'='*60}")
    print(f"Open-Meteo Forecast: {forecast_temp}°F")
    print(f"Forecast Bin: {forecast_bin[0]}° to {forecast_bin[1]}°")
    print(f"Spike Weight: {SPIKE_WEIGHT*100:.0f}% on forecast bin")
    print(f"Spread Weight: {(1-SPIKE_WEIGHT)*100:.0f}% distributed by uncertainty (±{UNCERTAINTY_STD}°F)")
    if cold_front_detected:
        print(f"🥶 Cold front adjustment: spike reduced to 45%, wider uncertainty")
    
    # Get Kalshi markets
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"series_ticker": city["kalshi_series"], "status": "open", "limit": 100}
    response = requests.get(f"{BASE_URL}/markets", params=params)
    markets = response.json().get("markets", [])
    
    target_kalshi = target_date.strftime("%y%b%d").upper()
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
        
        if forecast_vs_kalshi_diff <= 2:
            print("\n✅ MARKET IS EFFICIENTLY PRICED")
            market_efficient = True
        else:
            print(f"\n🎯 POTENTIAL MISPRICING: {forecast_vs_kalshi_diff:.1f}°F difference!")
    
    # Compare with Kalshi using MIXTURE MODEL
    print(f"\n{'='*60}")
    print("MODEL vs KALSHI COMPARISON (Mixture Model)")
    print(f"{'='*60}\n")
    
    print(f"{'Contract':<20} {'Model':>10} {'Kalshi':>10} {'Edge':>10}")
    print("-" * 55)
    
    results = []
    
    for m in sorted(target_markets, key=lambda x: x["last_price"], reverse=True):
        subtitle = m["subtitle"]
        kalshi_prob = m["last_price"] / 100
        numbers = re.findall(r'\d+', subtitle)
        
        model_prob = 0
        is_forecast_bin = False
        
        if "to" in subtitle and len(numbers) >= 2:
            low = int(numbers[0])
            high = int(numbers[1])
            model_prob = mixture_probability(low, high + 1, forecast_temp, UNCERTAINTY_STD, 
                                            SPIKE_WEIGHT, cold_front_detected)
            # Check if this is the forecast bin
            is_forecast_bin = low <= forecast_temp <= high + 1
            
        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = mixture_below_probability(threshold, forecast_temp, UNCERTAINTY_STD,
                                                   SPIKE_WEIGHT, cold_front_detected)
            is_forecast_bin = forecast_temp <= threshold
            
        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = mixture_above_probability(threshold, forecast_temp, UNCERTAINTY_STD,
                                                   SPIKE_WEIGHT, cold_front_detected)
            is_forecast_bin = forecast_temp >= threshold
        else:
            continue
        
        edge = model_prob - kalshi_prob
        
        results.append({
            "subtitle": subtitle,
            "model_prob": model_prob,
            "kalshi_prob": kalshi_prob,
            "edge": edge,
            "is_forecast_bin": is_forecast_bin
        })
        
        marker = "📍" if is_forecast_bin else "  "
        print(f"{marker}{subtitle:<18} {model_prob:>9.1%} {kalshi_prob:>9.0%} {edge:>+9.1%}")
    
    # Betting recommendations
    print(f"\n{'='*60}")
    print("BETTING RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    bankroll = 20
    model_confidence = 0.80
    good_bets = []
    
    for r in sorted(results, key=lambda x: abs(x["edge"]), reverse=True):
        edge = r["edge"]
        
        # CRITICAL FILTER: Never bet NO on the forecast bin
        if edge < 0 and r["is_forecast_bin"]:
            continue
        
        # Require meaningful edge
        if abs(edge) < 0.08:
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
                    "city": city["name"],
                    "subtitle": r["subtitle"],
                    "bet_side": bet_side,
                    "market_price": market_price,
                    "model_prob": r["model_prob"],
                    "our_prob": our_prob,
                    "odds": odds,
                    "half_kelly": half_kelly,
                    "bet_amount": bet_amount,
                    "potential_profit": potential_profit,
                    "edge": edge,
                    "is_forecast_bin": r["is_forecast_bin"]
                })
    
    if len(good_bets) == 0:
        if market_efficient:
            print("📊 NO CLEAR EDGE TODAY")
            print(f"   Market matches forecast (both ~{forecast_temp:.0f}°F)")
            print("   Consider skipping or small hedge bet")
        else:
            print("⚠️ NO RECOMMENDED BETS")
            print("   Edges too small or contradict forecast")
    else:
        for bet in good_bets:
            forecast_marker = " 📍 (FORECAST BIN)" if bet["is_forecast_bin"] else ""
            print(f"{bet['subtitle']}:{forecast_marker}")
            print(f"  Bet: {bet['bet_side']} at {bet['market_price']:.0%}")
            print(f"  Model prob: {bet['model_prob']:.1%}")
            print(f"  Edge: {bet['edge']:+.1%}")
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
    print(f"Open-Meteo Forecast: {forecast_temp}°F → Bin: {forecast_bin[0]}-{forecast_bin[1]}°")
    print(f"Kalshi Top Pick: {kalshi_top_contract} ({kalshi_top_prob}%)")
    print(f"Market Efficiency: {'✅ Efficient' if market_efficient else '🎯 Potential Edge'}")
    print(f"Recommended Bets: {len(good_bets)}")
    
    all_bets.extend(good_bets)


# ============ RUN ANALYSIS ============
if __name__ == "__main__":
    for city_key in selected_cities:
        analyze_city(city_key)
    
    # ============ BETTING SUMMARY ============
    print(f"\n{'='*60}")
    print("💰 BETTING SUMMARY - ALL CITIES")
    print(f"{'='*60}\n")

    if len(all_bets) == 0:
        print("No recommended bets today. Consider skipping.")
    else:
        # Sort by edge (best opportunities first)
        all_bets_sorted = sorted(all_bets, key=lambda x: abs(x["edge"]), reverse=True)
        
        total_suggested = 0
        total_potential_profit = 0
        
        print(f"{'City':<12} {'Contract':<15} {'Bet':<6} {'Price':<8} {'Edge':<8} {'Wager':<10} {'Profit':<10}")
        print("-" * 80)
        
        for bet in all_bets_sorted:
            forecast_marker = "📍" if bet.get("is_forecast_bin") else "  "
            print(f"{bet['city']:<12} {forecast_marker}{bet['subtitle']:<13} {bet['bet_side']:<6} {bet['market_price']*100:>5.0f}¢    {bet['edge']:>+5.1%}   ${bet['bet_amount']:>6.2f}    ${bet['potential_profit']:>7.2f}")
            total_suggested += bet["bet_amount"]
            total_potential_profit += bet["potential_profit"]
        
        print("-" * 80)
        print(f"{'TOTAL':<47} ${total_suggested:>6.2f}    ${total_potential_profit:>7.2f}")
        
        print(f"\n📍 = Forecast bin (highest confidence)")
        
        # Top picks (positive edge only)
        positive_edge_bets = [bet for bet in all_bets_sorted if bet["edge"] > 0]
        
        if len(positive_edge_bets) > 0:
            print(f"\n{'='*60}")
            print("⭐ TOP PICKS (Positive Edge)")
            print(f"{'='*60}")
            for bet in positive_edge_bets:
                forecast_marker = " 📍" if bet.get("is_forecast_bin") else ""
                print(f"\n   {bet['city']}: {bet['subtitle']} {bet['bet_side']}{forecast_marker}")
                print(f"   Price: {bet['market_price']*100:.0f}¢ | Edge: {bet['edge']:+.1%}")
                print(f"   Suggested: ${bet['bet_amount']:.2f} → Potential: ${bet['potential_profit']:.2f}")
        else:
            print(f"\n⚠️ No positive edge bets found. All recommendations are NO bets.")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")