import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import requests
from datetime import datetime, timedelta
import re
from scipy import stats

# ============ HRRR/HERBIE IMPORTS ============
try:
    from herbie import Herbie
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False
    print("⚠️ Herbie not installed. Install with: pip install herbie-data")
    print("   Also need: pip install xarray cfgrib")

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
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.7959,
        "lon": -80.2870,
        "kalshi_series": "KXHIGHMIA"
    }
}
all_bets = []

# Default to all cities
selected_cities = ["chicago", "nyc", "miami"]

# ============ RUN ALL CITIES ============
print("=" * 60)
print("KALSHI WEATHER BETTING MODEL v4")
print("(HRRR via NOAA Open Data Dissemination)")
print("=" * 60)
print(f"\nAnalyzing: Chicago, New York City, Miami")
print(f"Data Source: NOAA HRRR (3km resolution, hourly updates)")

# ============ FUNCTIONS ============
def load_and_prepare_data(city_config):
    """Load historical temperature data from CSV."""
    df = pd.read_csv(city_config["csv_file"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "value"]].copy()
    df = df.rename(columns={"value": "temp"})
    return df


def fetch_hrrr_forecast(lat, lon, target_date):
    """
    Fetch HRRR forecast data using Herbie.
    Returns forecasted daily high temperature for the target date.
    
    HRRR runs every hour with forecasts out to 18h (most runs) or 48h (00, 06, 12, 18 UTC).
    We'll use the latest available 12Z or 00Z run for best coverage.
    """
    if not HERBIE_AVAILABLE:
        return None, None
    
    today = datetime.now().date()
    
    # Determine which model run to use
    # For tomorrow's forecast, use today's 12Z run (gives us ~36h forecast)
    # or today's 00Z run if 12Z isn't available yet
    current_hour = datetime.now().hour
    
    if current_hour >= 14:  # 12Z run should be available by 14:00 local
        model_run_date = today
        model_run_hour = 12
    elif current_hour >= 2:  # 00Z run available by 02:00 local
        model_run_date = today
        model_run_hour = 0
    else:  # Use yesterday's 12Z
        model_run_date = today - timedelta(days=1)
        model_run_hour = 12
    
    model_run_str = f"{model_run_date.strftime('%Y-%m-%d')} {model_run_hour:02d}:00"
    
    print(f"\n  📡 Fetching HRRR data...")
    print(f"     Model run: {model_run_str} UTC")
    
    # Calculate forecast hours needed to cover the target date
    # We want to sample temperatures throughout the day (local time)
    # Target date high temp typically occurs between 12:00-18:00 local
    
    # Hours from model run to target date afternoon
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    target_noon = datetime.combine(target_date, datetime.min.time().replace(hour=12))
    target_evening = datetime.combine(target_date, datetime.min.time().replace(hour=18))
    
    # Forecast hours to check (covering the warmest part of the day)
    hours_to_noon = int((target_noon - model_run_datetime).total_seconds() / 3600)
    hours_to_evening = int((target_evening - model_run_datetime).total_seconds() / 3600)
    
    # Sample every 3 hours during the day for efficiency
    forecast_hours = list(range(max(1, hours_to_noon - 6), min(48, hours_to_evening + 1), 3))
    
    print(f"     Forecast hours: {forecast_hours[0]}h to {forecast_hours[-1]}h")
    
    temperatures = []
    wind_speeds = []
    wind_dirs = []
    
    for fxx in forecast_hours:
        try:
            H = Herbie(
                model_run_str,
                model='hrrr',
                product='sfc',
                fxx=fxx
            )
            
            # Get 2-meter temperature
            ds = H.xarray("TMP:2 m", remove_grib=True)
            
            # Extract temperature at the specific lat/lon
            # HRRR uses Lambert Conformal projection, need to find nearest point
            temp_data = ds['t2m']
            
            # Get lat/lon arrays
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            
            # Convert longitude to match HRRR convention (0-360)
            target_lon = lon if lon > 0 else lon + 360
            
            # Find nearest grid point
            dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            # Get temperature in Kelvin, convert to Fahrenheit
            temp_k = float(temp_data.values[min_idx])
            temp_f = (temp_k - 273.15) * 9/5 + 32
            temperatures.append(temp_f)
            
            # Try to get wind data too
            try:
                ds_wind = H.xarray("UGRD:10 m|VGRD:10 m", remove_grib=True)
                u = float(ds_wind['u10'].values[min_idx])
                v = float(ds_wind['v10'].values[min_idx])
                wind_speed = np.sqrt(u**2 + v**2) * 2.237  # m/s to mph
                wind_dir = (np.arctan2(-u, -v) * 180 / np.pi) % 360
                wind_speeds.append(wind_speed)
                wind_dirs.append(wind_dir)
            except:
                pass
                
        except Exception as e:
            print(f"     ⚠️ Error fetching hour {fxx}: {str(e)[:50]}")
            continue
    
    if not temperatures:
        print("     ❌ No HRRR data retrieved")
        return None, None
    
    # Calculate daily high from sampled temperatures
    forecast_high = max(temperatures)
    avg_wind_dir = np.mean(wind_dirs) if wind_dirs else None
    avg_wind_speed = np.mean(wind_speeds) if wind_speeds else None
    
    print(f"     ✅ Retrieved {len(temperatures)} temperature samples")
    print(f"     📊 Temperature range: {min(temperatures):.1f}°F to {max(temperatures):.1f}°F")
    
    weather_data = {
        'forecast_high': forecast_high,
        'forecast_temps': temperatures,
        'wind_speed': avg_wind_speed,
        'wind_dir': avg_wind_dir,
        'model_run': model_run_str
    }
    
    return forecast_high, weather_data


def fetch_hrrr_recent_temps(lat, lon, num_days=7):
    """
    Fetch recent actual high temperatures from HRRR analysis (F00) files.
    These are essentially observed temperatures.
    """
    if not HERBIE_AVAILABLE:
        return []
    
    recent_temps = []
    today = datetime.now().date()
    
    print(f"\n  📡 Fetching recent HRRR observations...")
    
    for days_ago in range(1, num_days + 1):
        date = today - timedelta(days=days_ago)
        date_str = date.strftime('%Y-%m-%d')
        
        daily_temps = []
        
        # Sample temperatures at peak heating hours (18Z-00Z typically warmest for US)
        for hour in [15, 18, 21]:  # 3pm, 6pm, 9pm UTC
            try:
                H = Herbie(
                    f"{date_str} {hour:02d}:00",
                    model='hrrr',
                    product='sfc',
                    fxx=0  # Analysis (observed)
                )
                
                ds = H.xarray("TMP:2 m", remove_grib=True)
                temp_data = ds['t2m']
                lats = ds['latitude'].values
                lons = ds['longitude'].values
                
                target_lon = lon if lon > 0 else lon + 360
                dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
                min_idx = np.unravel_index(np.argmin(dist), dist.shape)
                
                temp_k = float(temp_data.values[min_idx])
                temp_f = (temp_k - 273.15) * 9/5 + 32
                daily_temps.append(temp_f)
                
            except Exception as e:
                continue
        
        if daily_temps:
            daily_high = max(daily_temps)
            recent_temps.append({
                'date': date,
                'temp': daily_high
            })
            print(f"     {date_str}: {daily_high:.1f}°F")
    
    return recent_temps


def build_features(df):
    """Build ML features from historical data."""
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
    """Train the gradient boosting model."""
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
    lower = int(temp) if int(temp) % 2 == 0 else int(temp) - 1
    if temp < 0:
        lower = int(temp) - 1 if int(temp) % 2 == 0 else int(temp)
    
    if lower % 2 == 0:
        lower -= 1
    
    return (lower, lower + 1)


def mixture_probability(contract_low, contract_high, forecast_temp, uncertainty_std, 
                        spike_weight=0.55, cold_front=False):
    """
    Mixture Model: Combines a "spike" at the forecast bin with normal distribution uncertainty.
    """
    if cold_front:
        spike_weight = 0.45
        uncertainty_std = uncertainty_std * 1.3
    
    spread_weight = 1 - spike_weight
    forecast_bin = get_bin_for_temp(forecast_temp)
    
    is_forecast_bin = (contract_low == forecast_bin[0] or 
                       contract_low == forecast_bin[0] - 1 or
                       contract_low == forecast_bin[0] + 1)
    
    forecast_in_this_bin = contract_low <= forecast_temp <= contract_high
    
    spread_prob = stats.norm.cdf(contract_high, forecast_temp, uncertainty_std) - \
                  stats.norm.cdf(contract_low, forecast_temp, uncertainty_std)
    
    if forecast_in_this_bin:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def mixture_below_probability(threshold, forecast_temp, uncertainty_std, 
                              spike_weight=0.55, cold_front=False):
    """Mixture probability for 'X or below' contracts."""
    if cold_front:
        spike_weight = 0.45
        uncertainty_std = uncertainty_std * 1.3
    
    spread_weight = 1 - spike_weight
    spread_prob = stats.norm.cdf(threshold + 0.5, forecast_temp, uncertainty_std)
    
    if forecast_temp <= threshold:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def mixture_above_probability(threshold, forecast_temp, uncertainty_std, 
                              spike_weight=0.55, cold_front=False):
    """Mixture probability for 'X or above' contracts."""
    if cold_front:
        spike_weight = 0.45
        uncertainty_std = uncertainty_std * 1.3
    
    spread_weight = 1 - spike_weight
    spread_prob = 1 - stats.norm.cdf(threshold - 0.5, forecast_temp, uncertainty_std)
    
    if forecast_temp >= threshold:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def analyze_city(city_key):
    """Main analysis function for a single city."""
    city = cities[city_key]
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {city['name'].upper()}")
    print(f"{'='*60}")
    
    # Load historical data
    try:
        df = load_and_prepare_data(city)
        print(f"Loaded {len(df)} historical records")
    except FileNotFoundError:
        print(f"❌ Error: {city['csv_file']} not found.")
        print(f"   Run the weather fetch script for {city['name']} first.")
        return
    
    # Set target date (tomorrow)
    today = datetime.now().date()
    target_date = today + timedelta(days=1)
    target_str = target_date.strftime("%Y-%m-%d")
    
    # Fetch HRRR forecast
    forecast_temp, weather_data = fetch_hrrr_forecast(
        city["lat"], city["lon"], target_date
    )
    
    if forecast_temp is None:
        print("❌ Could not retrieve HRRR forecast. Skipping city.")
        return
    
    # Display HRRR data
    print(f"\n  HRRR Forecast Data:")
    print(f"  {'='*50}")
    print(f"  Model Run: {weather_data['model_run']} UTC")
    print(f"  Target Date: {target_str}")
    print(f"  Forecast High: {forecast_temp:.1f}°F")
    if weather_data['wind_speed']:
        print(f"  Avg Wind: {weather_data['wind_speed']:.1f} mph @ {weather_data['wind_dir']:.0f}°")
    
    # Fetch recent temps to update our historical data
    recent_temps = fetch_hrrr_recent_temps(city["lat"], city["lon"], num_days=5)
    
    # Add recent data to dataframe
    for rt in recent_temps:
        new_row = pd.DataFrame({
            "date": [pd.to_datetime(rt['date'])], 
            "temp": [rt['temp']]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Build features and train
    df = build_features(df)
    model, model_error_std, mae, feature_cols = train_model(df)
    
    print(f"\n  Model trained: MAE = {mae:.2f}°F")
    
    # Weather change indicators
    cold_front_detected = False
    if recent_temps and len(recent_temps) >= 2:
        yesterday_temp = recent_temps[0]['temp']  # Most recent
        temp_change = yesterday_temp - forecast_temp
        
        print(f"\n  {'='*50}")
        print("  WEATHER CHANGE INDICATORS")
        print(f"  {'='*50}")
        print(f"  Yesterday: {yesterday_temp:.1f}°F → Tomorrow: {forecast_temp:.1f}°F")
        print(f"  Temperature change: {-temp_change:+.1f}°F", end="")
        
        if temp_change > 10:
            print(" 🥶 COLD FRONT")
            cold_front_detected = True
        elif temp_change < -10:
            print(" 🌡️ WARMING")
        else:
            print()
        
        # Check wind direction for cold air advection
        if weather_data['wind_dir']:
            wind_dir = weather_data['wind_dir']
            cold_wind = wind_dir > 270 or wind_dir < 90
            print(f"  Wind direction: {'❄️ Cold (N/NW)' if cold_wind else '🌡️ Warm (S/SW)'}")
            if cold_wind and temp_change > 5:
                cold_front_detected = True
    
    # ============ MIXTURE MODEL PARAMETERS ============
    UNCERTAINTY_STD = 2.5  # Base uncertainty
    SPIKE_WEIGHT = 0.55    # 55% weight on forecast bin
    
    # HRRR is more accurate than Open-Meteo, so we can be slightly more confident
    # But still account for uncertainty
    SPIKE_WEIGHT = 0.60  # Increased confidence for HRRR
    UNCERTAINTY_STD = 2.2  # Slightly tighter uncertainty
    
    forecast_bin = get_bin_for_temp(forecast_temp)
    
    print(f"\n  {'='*50}")
    print("  MIXTURE MODEL (v4 - HRRR)")
    print(f"  {'='*50}")
    print(f"  HRRR Forecast: {forecast_temp:.1f}°F")
    print(f"  Forecast Bin: {forecast_bin[0]}° to {forecast_bin[1]}°")
    print(f"  Spike Weight: {SPIKE_WEIGHT*100:.0f}% on forecast bin")
    print(f"  Spread Weight: {(1-SPIKE_WEIGHT)*100:.0f}% distributed (±{UNCERTAINTY_STD}°F)")
    if cold_front_detected:
        print(f"  🥶 Cold front adjustment: spike reduced to 50%, wider uncertainty")
    
    # Get Kalshi markets
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"series_ticker": city["kalshi_series"], "status": "open", "limit": 100}
    response = requests.get(f"{BASE_URL}/markets", params=params)
    markets = response.json().get("markets", [])
    
    target_kalshi = target_date.strftime("%y%b%d").upper()
    target_markets = [m for m in markets if target_kalshi in m["ticker"]]
    
    if not target_markets:
        print(f"\n  ⚠️ No Kalshi markets found for {target_str}")
        return
    
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
    
    print(f"\n  {'='*50}")
    print("  MARKET EFFICIENCY CHECK")
    print(f"  {'='*50}")
    print(f"  HRRR Forecast: {forecast_temp:.1f}°F")
    print(f"  Kalshi Top Contract: {kalshi_top_contract} ({kalshi_top_prob}%)")
    
    market_efficient = False
    if kalshi_implied_temp:
        print(f"  Kalshi Implied Temp: ~{kalshi_implied_temp}°F")
        forecast_vs_kalshi_diff = abs(forecast_temp - kalshi_implied_temp)
        print(f"  Forecast vs Kalshi Diff: {forecast_vs_kalshi_diff:.1f}°F")
        
        if forecast_vs_kalshi_diff <= 2:
            print("\n  ✅ MARKET IS EFFICIENTLY PRICED")
            market_efficient = True
        else:
            print(f"\n  🎯 POTENTIAL MISPRICING: {forecast_vs_kalshi_diff:.1f}°F difference!")
    
    # Compare with Kalshi
    print(f"\n  {'='*50}")
    print("  MODEL vs KALSHI COMPARISON")
    print(f"  {'='*50}\n")
    
    print(f"  {'Contract':<20} {'Model':>10} {'Kalshi':>10} {'Edge':>10}")
    print("  " + "-" * 55)
    
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
        print(f"  {marker}{subtitle:<18} {model_prob:>9.1%} {kalshi_prob:>9.0%} {edge:>+9.1%}")
    
    # Betting recommendations
    print(f"\n  {'='*50}")
    print("  BETTING RECOMMENDATIONS")
    print(f"  {'='*50}\n")
    
    bankroll = 20
    model_confidence = 0.85  # Higher confidence with HRRR
    good_bets = []
    
    for r in sorted(results, key=lambda x: abs(x["edge"]), reverse=True):
        edge = r["edge"]
        
        # Never bet NO on the forecast bin
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
            print("  📊 NO CLEAR EDGE TODAY")
            print(f"     Market matches forecast (both ~{forecast_temp:.0f}°F)")
            print("     Consider skipping or small hedge bet")
        else:
            print("  ⚠️ NO RECOMMENDED BETS")
            print("     Edges too small or contradict forecast")
    else:
        for bet in good_bets:
            forecast_marker = " 📍 (FORECAST BIN)" if bet["is_forecast_bin"] else ""
            print(f"  {bet['subtitle']}:{forecast_marker}")
            print(f"    Bet: {bet['bet_side']} at {bet['market_price']:.0%}")
            print(f"    Model prob: {bet['model_prob']:.1%}")
            print(f"    Edge: {bet['edge']:+.1%}")
            print(f"    Odds: {bet['odds']:.2f}:1")
            print(f"    Half Kelly: {bet['half_kelly']:.1%}")
            print(f"    Suggested bet: ${bet['bet_amount']:.2f}")
            print(f"    Potential profit: ${bet['potential_profit']:.2f}")
            print()
    
    # Summary
    print(f"  {'='*50}")
    print(f"  SUMMARY - {city['name'].upper()}")
    print(f"  {'='*50}")
    print(f"  Target Date: {target_str}")
    print(f"  HRRR Forecast: {forecast_temp:.1f}°F → Bin: {forecast_bin[0]}-{forecast_bin[1]}°")
    print(f"  Kalshi Top Pick: {kalshi_top_contract} ({kalshi_top_prob}%)")
    print(f"  Market Efficiency: {'✅ Efficient' if market_efficient else '🎯 Potential Edge'}")
    print(f"  Recommended Bets: {len(good_bets)}")
    
    all_bets.extend(good_bets)


# ============ RUN ANALYSIS ============
if __name__ == "__main__":
    if not HERBIE_AVAILABLE:
        print("\n❌ Cannot run without Herbie. Please install:")
        print("   pip install herbie-data xarray cfgrib eccodes")
        exit(1)
    
    for city_key in selected_cities:
        analyze_city(city_key)
    
    # ============ BETTING SUMMARY ============
    target_date = (datetime.now().date() + timedelta(days=1)).strftime("%A, %B %d, %Y")
    print(f"\n{'='*60}")
    print(f"💰 BETTING SUMMARY - ALL CITIES")
    print(f"📅 Target Date: {target_date}")
    print(f"📡 Data Source: NOAA HRRR (3km resolution)")
    print(f"{'='*60}\n")

    if len(all_bets) == 0:
        print("No recommended bets today. Consider skipping.")
    else:
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