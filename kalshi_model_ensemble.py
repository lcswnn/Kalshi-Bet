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
    print("⚠️ Herbie not installed. Will use Open-Meteo only.")
    print("   Install with: pip install herbie-data xarray cfgrib")

# ============ CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - only bet if models agree within this range
CONFIDENCE_BOOST_THRESHOLD = 2.0   # °F - boost confidence if within this range

# ============ CITY CONFIGURATION ============
cities = {
    "chicago": {
        "name": "Chicago",
        "csv_file": "weather_data_chicago.csv",
        "lat": 41.8781,
        "lon": -87.6298,
        "kalshi_series": "KXHIGHCHI",
        "timezone": "America/Chicago",
        "utc_offset": -6  # CST
    },
    "nyc": {
        "name": "New York City",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.7128,
        "lon": -74.0060,
        "kalshi_series": "KXHIGHNY",
        "timezone": "America/New_York",
        "utc_offset": -5  # EST
    },
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.7959,
        "lon": -80.2870,
        "kalshi_series": "KXHIGHMIA",
        "timezone": "America/New_York",
        "utc_offset": -5  # EST
    }
}
all_bets = []

# Default to all cities
selected_cities = ["chicago", "nyc", "miami"]

# ============ HEADER ============
print("=" * 70)
print("KALSHI WEATHER BETTING MODEL v6")
print("(Ensemble: HRRR + Open-Meteo Agreement)")
print("=" * 70)
print(f"\nAnalyzing: Chicago, New York City, Miami")
print(f"Agreement Threshold: ±{ENSEMBLE_AGREEMENT_THRESHOLD}°F")
print(f"Data Sources: NOAA HRRR (3km) + Open-Meteo")

# ============ FUNCTIONS ============

def load_and_prepare_data(city_config):
    """Load historical temperature data from CSV."""
    df = pd.read_csv(city_config["csv_file"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "value"]].copy()
    df = df.rename(columns={"value": "temp"})
    return df


def fetch_open_meteo(lat, lon, timezone):
    """Fetch forecast from Open-Meteo API."""
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
        ],
        "past_days": 7,
        "forecast_days": 3,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": timezone
    }
    response = requests.get(open_meteo_url, params=params)
    return response.json()


def fetch_hrrr_forecast_fixed(lat, lon, target_date, utc_offset):
    """
    Fetch HRRR forecast with FIXED sampling to get actual daily high.
    
    The key fix: Sample during LOCAL afternoon hours (12pm-6pm local time)
    when the daily high actually occurs.
    """
    if not HERBIE_AVAILABLE:
        return None, None
    
    today = datetime.now().date()
    
    # Determine which model run to use
    current_hour = datetime.now().hour
    
    if current_hour >= 14:
        model_run_date = today
        model_run_hour = 12
    elif current_hour >= 2:
        model_run_date = today
        model_run_hour = 0
    else:
        model_run_date = today - timedelta(days=1)
        model_run_hour = 12
    
    model_run_str = f"{model_run_date.strftime('%Y-%m-%d')} {model_run_hour:02d}:00"
    
    # Calculate forecast hours for LOCAL afternoon (12pm-6pm local)
    # Convert local afternoon to UTC, then to forecast hours
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    
    # Target afternoon in LOCAL time, converted to UTC
    # Local 12pm = UTC (12 - utc_offset), Local 6pm = UTC (18 - utc_offset)
    target_12_local_utc = 12 - utc_offset  # e.g., Chicago 12pm = 18 UTC
    target_18_local_utc = 18 - utc_offset  # e.g., Chicago 6pm = 00 UTC next day
    
    # Calculate forecast hours from model run to target afternoon
    target_start = datetime.combine(target_date, datetime.min.time().replace(hour=target_12_local_utc % 24))
    if target_12_local_utc >= 24:
        target_start += timedelta(days=1)
    
    # Hours from model run to target afternoon start
    hours_to_start = int((target_start - model_run_datetime).total_seconds() / 3600)
    
    # Sample every hour during the 6-hour afternoon window
    forecast_hours = list(range(max(1, hours_to_start), min(48, hours_to_start + 7)))
    
    print(f"     HRRR Model run: {model_run_str} UTC")
    print(f"     Sampling hours: {forecast_hours} (local afternoon)")
    
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
            
            temp_data = ds['t2m']
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
            
            # Try to get wind data
            try:
                ds_u = H.xarray("UGRD:10 m", remove_grib=True)
                ds_v = H.xarray("VGRD:10 m", remove_grib=True)
                u = float(ds_u['u10'].values[min_idx]) if 'u10' in ds_u else float(ds_u['u'].values[min_idx])
                v = float(ds_v['v10'].values[min_idx]) if 'v10' in ds_v else float(ds_v['v'].values[min_idx])
                wind_speed = np.sqrt(u**2 + v**2) * 2.237  # m/s to mph
                wind_dir = (np.arctan2(-u, -v) * 180 / np.pi) % 360
                wind_speeds.append(wind_speed)
                wind_dirs.append(wind_dir)
            except:
                pass
                
        except Exception as e:
            continue
    
    if not temperatures:
        print("     ❌ No HRRR data retrieved")
        return None, None
    
    # Daily high is the MAX of afternoon temperatures
    forecast_high = max(temperatures)
    
    print(f"     ✅ Retrieved {len(temperatures)} samples")
    print(f"     📊 Afternoon temps: {min(temperatures):.1f}°F to {max(temperatures):.1f}°F")
    
    avg_wind_speed = np.mean(wind_speeds) if wind_speeds else None
    avg_wind_dir = np.mean(wind_dirs) if wind_dirs else None
    
    weather_data = {
        'forecast_high': forecast_high,
        'all_temps': temperatures,
        'wind_speed': avg_wind_speed,
        'wind_dir': avg_wind_dir,
        'model_run': model_run_str,
        'source': 'HRRR'
    }
    
    return forecast_high, weather_data


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


def ensemble_probability(contract_low, contract_high, ensemble_forecast, uncertainty_std, 
                         spike_weight=0.60, agreement_level='high'):
    """
    Calculate probability using ensemble forecast with agreement-based confidence.
    
    agreement_level: 'high' (within 2°F), 'medium' (within 3°F), 'low' (>3°F disagreement)
    """
    # Adjust parameters based on agreement level
    if agreement_level == 'high':
        spike_weight = 0.65  # Very confident
        uncertainty_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        spike_weight = 0.55  # Moderate confidence
        uncertainty_std = uncertainty_std * 1.0
    else:  # low
        spike_weight = 0.40  # Low confidence
        uncertainty_std = uncertainty_std * 1.4
    
    spread_weight = 1 - spike_weight
    forecast_bin = get_bin_for_temp(ensemble_forecast)
    
    forecast_in_this_bin = contract_low <= ensemble_forecast <= contract_high
    
    spread_prob = stats.norm.cdf(contract_high, ensemble_forecast, uncertainty_std) - \
                  stats.norm.cdf(contract_low, ensemble_forecast, uncertainty_std)
    
    if forecast_in_this_bin:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def ensemble_below_probability(threshold, ensemble_forecast, uncertainty_std, 
                               spike_weight=0.60, agreement_level='high'):
    """Ensemble probability for 'X or below' contracts."""
    if agreement_level == 'high':
        spike_weight = 0.65
        uncertainty_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        spike_weight = 0.55
    else:
        spike_weight = 0.40
        uncertainty_std = uncertainty_std * 1.4
    
    spread_weight = 1 - spike_weight
    spread_prob = stats.norm.cdf(threshold + 0.5, ensemble_forecast, uncertainty_std)
    
    if ensemble_forecast <= threshold:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def ensemble_above_probability(threshold, ensemble_forecast, uncertainty_std, 
                               spike_weight=0.60, agreement_level='high'):
    """Ensemble probability for 'X or above' contracts."""
    if agreement_level == 'high':
        spike_weight = 0.65
        uncertainty_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        spike_weight = 0.55
    else:
        spike_weight = 0.40
        uncertainty_std = uncertainty_std * 1.4
    
    spread_weight = 1 - spike_weight
    spread_prob = 1 - stats.norm.cdf(threshold - 0.5, ensemble_forecast, uncertainty_std)
    
    if ensemble_forecast >= threshold:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def analyze_city(city_key):
    """Main analysis function for a single city using ensemble approach."""
    city = cities[city_key]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {city['name'].upper()}")
    print(f"{'='*70}")
    
    # Load historical data
    try:
        df = load_and_prepare_data(city)
        print(f"Loaded {len(df)} historical records")
    except FileNotFoundError:
        print(f"❌ Error: {city['csv_file']} not found.")
        return
    
    # Set target date
    today = datetime.now().date()
    target_date = today + timedelta(days=1)
    target_str = target_date.strftime("%Y-%m-%d")
    
    # ============ FETCH BOTH FORECASTS ============
    print(f"\n  📡 Fetching forecasts for {target_str}...")
    
    # 1. Open-Meteo forecast
    print(f"\n  [1] Open-Meteo:")
    meteo_data = fetch_open_meteo(city["lat"], city["lon"], city["timezone"])
    meteo_dates = meteo_data["daily"]["time"]
    meteo_temps = meteo_data["daily"]["temperature_2m_max"]
    meteo_wind = meteo_data["daily"]["wind_speed_10m_max"]
    meteo_wind_dir = meteo_data["daily"]["wind_direction_10m_dominant"]
    
    # Find target date in Open-Meteo data
    open_meteo_forecast = None
    open_meteo_wind = None
    open_meteo_wind_dir = None
    for i, d in enumerate(meteo_dates):
        if d == target_str:
            open_meteo_forecast = meteo_temps[i]
            open_meteo_wind = meteo_wind[i]
            open_meteo_wind_dir = meteo_wind_dir[i]
            break
    
    if open_meteo_forecast:
        print(f"     Forecast High: {open_meteo_forecast:.1f}°F")
        print(f"     Wind: {open_meteo_wind:.1f} mph @ {open_meteo_wind_dir:.0f}°")
    else:
        print(f"     ❌ No forecast available")
    
    # 2. HRRR forecast (with fixed sampling)
    print(f"\n  [2] HRRR (NOAA):")
    hrrr_forecast, hrrr_data = fetch_hrrr_forecast_fixed(
        city["lat"], city["lon"], target_date, city["utc_offset"]
    )
    
    if hrrr_forecast:
        print(f"     Forecast High: {hrrr_forecast:.1f}°F")
    
    # ============ ENSEMBLE ANALYSIS ============
    print(f"\n  {'='*60}")
    print("  ENSEMBLE ANALYSIS")
    print(f"  {'='*60}")
    
    if open_meteo_forecast is None and hrrr_forecast is None:
        print("  ❌ No forecasts available. Skipping city.")
        return
    
    # Calculate ensemble forecast and agreement
    if open_meteo_forecast and hrrr_forecast:
        forecast_diff = abs(open_meteo_forecast - hrrr_forecast)
        ensemble_forecast = (open_meteo_forecast + hrrr_forecast) / 2
        
        print(f"\n  Open-Meteo:  {open_meteo_forecast:.1f}°F")
        print(f"  HRRR:        {hrrr_forecast:.1f}°F")
        print(f"  Difference:  {forecast_diff:.1f}°F")
        print(f"  Ensemble:    {ensemble_forecast:.1f}°F (average)")
        
        if forecast_diff <= CONFIDENCE_BOOST_THRESHOLD:
            agreement_level = 'high'
            print(f"\n  ✅ HIGH AGREEMENT (within {CONFIDENCE_BOOST_THRESHOLD}°F)")
            print(f"     → Using boosted confidence parameters")
        elif forecast_diff <= ENSEMBLE_AGREEMENT_THRESHOLD:
            agreement_level = 'medium'
            print(f"\n  ⚠️ MEDIUM AGREEMENT (within {ENSEMBLE_AGREEMENT_THRESHOLD}°F)")
            print(f"     → Using standard confidence parameters")
        else:
            agreement_level = 'low'
            print(f"\n  🚨 LOW AGREEMENT (>{ENSEMBLE_AGREEMENT_THRESHOLD}°F difference)")
            print(f"     → CAUTION: Models disagree significantly")
            print(f"     → Using conservative parameters, smaller bets recommended")
    
    elif open_meteo_forecast:
        ensemble_forecast = open_meteo_forecast
        agreement_level = 'medium'
        print(f"\n  Using Open-Meteo only: {ensemble_forecast:.1f}°F")
        print(f"  ⚠️ Single source - medium confidence")
    else:
        ensemble_forecast = hrrr_forecast
        agreement_level = 'medium'
        print(f"\n  Using HRRR only: {ensemble_forecast:.1f}°F")
        print(f"  ⚠️ Single source - medium confidence")
    
    # Build features and train ML model
    df = build_features(df)
    model, model_error_std, mae, feature_cols = train_model(df)
    print(f"\n  ML Model trained: MAE = {mae:.2f}°F")
    
    # ============ MODEL PARAMETERS ============
    BASE_UNCERTAINTY = 2.5
    BASE_SPIKE = 0.60
    
    forecast_bin = get_bin_for_temp(ensemble_forecast)
    
    print(f"\n  {'='*60}")
    print("  MIXTURE MODEL (v6 - Ensemble)")
    print(f"  {'='*60}")
    print(f"  Ensemble Forecast: {ensemble_forecast:.1f}°F")
    print(f"  Forecast Bin: {forecast_bin[0]}° to {forecast_bin[1]}°")
    print(f"  Agreement Level: {agreement_level.upper()}")
    
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
    
    print(f"\n  {'='*60}")
    print("  MARKET COMPARISON")
    print(f"  {'='*60}")
    print(f"  Ensemble Forecast: {ensemble_forecast:.1f}°F")
    print(f"  Kalshi Top Contract: {kalshi_top_contract} ({kalshi_top_prob}%)")
    
    market_efficient = False
    if kalshi_implied_temp:
        print(f"  Kalshi Implied Temp: ~{kalshi_implied_temp}°F")
        forecast_vs_kalshi_diff = abs(ensemble_forecast - kalshi_implied_temp)
        print(f"  Ensemble vs Kalshi Diff: {forecast_vs_kalshi_diff:.1f}°F")
        
        if forecast_vs_kalshi_diff <= 2:
            print("\n  ✅ MARKET IS EFFICIENTLY PRICED")
            market_efficient = True
        else:
            print(f"\n  🎯 POTENTIAL MISPRICING: {forecast_vs_kalshi_diff:.1f}°F difference!")
    
    # Compare with Kalshi
    print(f"\n  {'='*60}")
    print("  MODEL vs KALSHI COMPARISON")
    print(f"  {'='*60}\n")
    
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
            model_prob = ensemble_probability(low, high + 1, ensemble_forecast, BASE_UNCERTAINTY, 
                                             BASE_SPIKE, agreement_level)
            is_forecast_bin = low <= ensemble_forecast <= high + 1
            
        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = ensemble_below_probability(threshold, ensemble_forecast, BASE_UNCERTAINTY,
                                                    BASE_SPIKE, agreement_level)
            is_forecast_bin = ensemble_forecast <= threshold
            
        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = ensemble_above_probability(threshold, ensemble_forecast, BASE_UNCERTAINTY,
                                                    BASE_SPIKE, agreement_level)
            is_forecast_bin = ensemble_forecast >= threshold
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
    print(f"\n  {'='*60}")
    print("  BETTING RECOMMENDATIONS")
    print(f"  {'='*60}\n")
    
    # Adjust bankroll and confidence based on agreement level
    base_bankroll = 20
    if agreement_level == 'high':
        bankroll = base_bankroll
        model_confidence = 0.85
        min_edge = 0.08
    elif agreement_level == 'medium':
        bankroll = base_bankroll * 0.75
        model_confidence = 0.75
        min_edge = 0.10
    else:  # low
        bankroll = base_bankroll * 0.5
        model_confidence = 0.60
        min_edge = 0.15
        print(f"  ⚠️ LOW AGREEMENT: Reduced bet sizes and higher edge threshold")
        print(f"     Bankroll: ${bankroll:.2f} (reduced)")
        print(f"     Min edge required: {min_edge:.0%}\n")
    
    good_bets = []
    
    for r in sorted(results, key=lambda x: abs(x["edge"]), reverse=True):
        edge = r["edge"]
        
        # Never bet NO on the forecast bin
        if edge < 0 and r["is_forecast_bin"]:
            continue
        
        # Require meaningful edge (adjusted by agreement level)
        if abs(edge) < min_edge:
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
                    "is_forecast_bin": r["is_forecast_bin"],
                    "agreement_level": agreement_level
                })
    
    if len(good_bets) == 0:
        if agreement_level == 'low':
            print("  🛑 NO BETS RECOMMENDED")
            print("     Models disagree significantly - waiting for convergence")
        elif market_efficient:
            print("  📊 NO CLEAR EDGE TODAY")
            print(f"     Market matches ensemble forecast (~{ensemble_forecast:.0f}°F)")
        else:
            print("  ⚠️ NO RECOMMENDED BETS")
            print("     Edges too small or contradict forecast")
    else:
        for bet in good_bets:
            forecast_marker = " 📍 (FORECAST BIN)" if bet["is_forecast_bin"] else ""
            agreement_marker = f" [{bet['agreement_level'].upper()}]"
            print(f"  {bet['subtitle']}:{forecast_marker}{agreement_marker}")
            print(f"    Bet: {bet['bet_side']} at {bet['market_price']:.0%}")
            print(f"    Model prob: {bet['model_prob']:.1%}")
            print(f"    Edge: {bet['edge']:+.1%}")
            print(f"    Half Kelly: {bet['half_kelly']:.1%}")
            print(f"    Suggested bet: ${bet['bet_amount']:.2f}")
            print(f"    Potential profit: ${bet['potential_profit']:.2f}")
            print()
    
    # Summary
    print(f"  {'='*60}")
    print(f"  SUMMARY - {city['name'].upper()}")
    print(f"  {'='*60}")
    print(f"  Target Date: {target_str}")
    if open_meteo_forecast and hrrr_forecast:
        print(f"  Open-Meteo: {open_meteo_forecast:.1f}°F | HRRR: {hrrr_forecast:.1f}°F")
    print(f"  Ensemble Forecast: {ensemble_forecast:.1f}°F → Bin: {forecast_bin[0]}-{forecast_bin[1]}°")
    print(f"  Agreement Level: {agreement_level.upper()}")
    print(f"  Kalshi Top Pick: {kalshi_top_contract} ({kalshi_top_prob}%)")
    print(f"  Market Status: {'✅ Efficient' if market_efficient else '🎯 Potential Edge'}")
    print(f"  Recommended Bets: {len(good_bets)}")
    
    all_bets.extend(good_bets)


# ============ RUN ANALYSIS ============
if __name__ == "__main__":
    for city_key in selected_cities:
        analyze_city(city_key)
    
    # ============ BETTING SUMMARY ============
    target_date = (datetime.now().date() + timedelta(days=1)).strftime("%A, %B %d, %Y")
    print(f"\n{'='*70}")
    print(f"💰 BETTING SUMMARY - ALL CITIES")
    print(f"📅 Target Date: {target_date}")
    print(f"📡 Data Source: Ensemble (HRRR + Open-Meteo)")
    print(f"{'='*70}\n")

    if len(all_bets) == 0:
        print("No recommended bets today.")
        print("This could mean:")
        print("  • Models disagree significantly (waiting for convergence)")
        print("  • Market is efficiently priced")
        print("  • Edges are too small to bet on")
    else:
        # Group by agreement level
        high_conf_bets = [b for b in all_bets if b['agreement_level'] == 'high']
        med_conf_bets = [b for b in all_bets if b['agreement_level'] == 'medium']
        low_conf_bets = [b for b in all_bets if b['agreement_level'] == 'low']
        
        all_bets_sorted = sorted(all_bets, key=lambda x: (-['high', 'medium', 'low'].index(x['agreement_level']), -abs(x["edge"])))
        
        total_suggested = 0
        total_potential_profit = 0
        
        print(f"{'City':<12} {'Contract':<15} {'Bet':<6} {'Price':<8} {'Edge':<8} {'Conf':<8} {'Wager':<10} {'Profit':<10}")
        print("-" * 90)
        
        for bet in all_bets_sorted:
            forecast_marker = "📍" if bet.get("is_forecast_bin") else "  "
            conf_marker = "🟢" if bet['agreement_level'] == 'high' else "🟡" if bet['agreement_level'] == 'medium' else "🔴"
            print(f"{bet['city']:<12} {forecast_marker}{bet['subtitle']:<13} {bet['bet_side']:<6} {bet['market_price']*100:>5.0f}¢    {bet['edge']:>+5.1%}   {conf_marker:<8} ${bet['bet_amount']:>6.2f}    ${bet['potential_profit']:>7.2f}")
            total_suggested += bet["bet_amount"]
            total_potential_profit += bet["potential_profit"]
        
        print("-" * 90)
        print(f"{'TOTAL':<56} ${total_suggested:>6.2f}    ${total_potential_profit:>7.2f}")
        
        print(f"\n📍 = Forecast bin | 🟢 = High confidence | 🟡 = Medium | 🔴 = Low (caution)")
        
        # Top picks (high confidence only)
        if high_conf_bets:
            print(f"\n{'='*70}")
            print("⭐ TOP PICKS (High Confidence - Models Agree)")
            print(f"{'='*70}")
            for bet in sorted(high_conf_bets, key=lambda x: abs(x["edge"]), reverse=True)[:3]:
                forecast_marker = " 📍" if bet.get("is_forecast_bin") else ""
                print(f"\n   {bet['city']}: {bet['subtitle']} {bet['bet_side']}{forecast_marker}")
                print(f"   Price: {bet['market_price']*100:.0f}¢ | Edge: {bet['edge']:+.1%}")
                print(f"   Suggested: ${bet['bet_amount']:.2f} → Potential: ${bet['potential_profit']:.2f}")
        
        if low_conf_bets:
            print(f"\n{'='*70}")
            print("⚠️ CAUTION BETS (Low Confidence - Models Disagree)")
            print(f"{'='*70}")
            print("These bets have significant model disagreement. Consider:")
            print("  • Waiting for the 12Z HRRR run for updated forecasts")
            print("  • Reducing bet sizes further")
            print("  • Skipping these entirely")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")