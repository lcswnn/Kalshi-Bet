"""
KALSHI WEATHER BETTING MODEL v9.5
=================================
KEY IMPROVEMENT: MIXTURE DISTRIBUTION PROBABILITY MODEL

The core insight: When HRRR says 72°F and NWS says 76°F, the truth isn't
necessarily at 73.35°F (the weighted average). There's a 45% chance HRRR
is "right" (truth near 72°F) and a 30% chance NWS is "right" (truth near 76°F).

v9.4 collapsed these to a single point, then applied uncertainty around that point.
v9.5 treats the ensemble as a MIXTURE of distributions - each model contributes
its own probability distribution centered on ITS forecast.

This matters most when:
- Models disagree substantially (4°F+ spread)
- Evaluating bins at the tails (near one model's forecast, far from average)
- One high-weight model is an outlier

Example: If bins are 71-72, 73-74, 75-76:
- v9.4 (point estimate at 73.35°F): 71-72 gets ~15%, 73-74 gets ~45%
- v9.5 (mixture): 71-72 gets ~25% (HRRR's peak), 73-74 gets ~35%

The mixture approach gives more probability to bins where individual models peak,
rather than concentrating everything at an artificial middle point.

What we KEPT from v9.4:
- Full 24h HRRR sampling (cold fronts)
- Student's t-distribution for fat tails
- Dynamic uncertainty
- Flat edge requirement
- Smart bet selection
- Kelly Criterion with fees
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import requests
from datetime import datetime, timedelta
import re
from scipy import stats
import argparse

# ============ HRRR/HERBIE IMPORTS ============
try:
    from herbie import Herbie  # type: ignore
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False
    print("⚠️ Herbie not installed. Will use Open-Meteo only.")
    print("   Install with: pip install herbie-data xarray cfgrib")

# ============ PRICE FILTER CONFIGURATION ============
MIN_CONTRACT_PRICE = 0.15   # Never bet on contracts below 15¢
MAX_CONTRACT_PRICE = 0.90   # Never bet on contracts above 90¢
SWEET_SPOT_LOW = 0.15       # Preferred range lower bound
SWEET_SPOT_HIGH = 0.50      # Preferred range upper bound

# ============ SMART BET SELECTION CONFIGURATION ============
SAME_CITY_MULTI_BET_THRESHOLD = 2.0   # Need 2x minimum edge to stack same-city bets
MAX_BETS_PER_CITY = 2                  # Never more than 2 bets in same city
MAX_TOTAL_BETS_PER_DAY = 6             # Cap total daily bets across all cities
SUPER_CONFIDENT_EDGE_RATIO = 2.5       # What counts as "super confident"

# ============ FEE CONFIGURATION ============
KALSHI_FEE_RATE = 0.04        # ~4% fee on entry

# ============ KELLY CRITERION CONFIGURATION ============
STARTING_BANKROLL = 70        # Your bankroll
KELLY_FRACTION = 0.50         # Half Kelly (balanced risk/reward)
MIN_BET_SIZE = 0.50           # Don't bet less than 50¢
MAX_BET_FRACTION = 0.15       # Never bet more than 15% of bankroll

# ============ ENSEMBLE CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - models should agree within this
CONFIDENCE_BOOST_THRESHOLD = 2.0    # °F - boost confidence if within this

# ============ V9.5: MIXTURE DISTRIBUTION CONFIG ============
# Base edge requirement - FLAT (no agreement penalty - that's handled by σ)
BASE_EDGE_REQUIREMENT = 0.16  # 16% edge required for all bets

# Student's t degrees of freedom (lower = fatter tails)
T_DISTRIBUTION_DF = 5

# Model weights for mixture distribution
# These represent the probability that each model is "correct"
MODEL_WEIGHTS = {
    "HRRR": 0.45,
    "NWS": 0.30,
    "Open-Meteo": 0.25
}

# City-specific uncertainty MULTIPLIERS (applied to dynamic base)
CITY_UNCERTAINTY_MULTIPLIER = {
    "chicago": 1.15,    # Lake effect + fronts → 15% more uncertain
    "nyc": 1.15,        # Coastal + urban effects → 15% more uncertain  
    "miami": 0.90,      # Stable subtropical → 10% less uncertain
    "austin": 1.00,     # Continental, baseline
    "la": 0.90,         # Mediterranean climate → stable
}


# ============ DYNAMIC UNCERTAINTY ============

def calculate_dynamic_uncertainty(hrrr_temp, nws_temp, meteo_temp, city_key, hours_until_target=18):
    """
    Calculate uncertainty dynamically based on model spread, lead time, and city.
    
    Returns:
        uncertainty_std (float): Standard deviation in °F
        model_spread (float): Spread between models in °F
    """
    temps = [t for t in [hrrr_temp, nws_temp, meteo_temp] if t is not None]
    
    if len(temps) < 2:
        model_spread = 0
        base_std = 3.75  # Single model - higher baseline
    else:
        model_spread = max(temps) - min(temps)
        
        # Base uncertainty from model spread (INFLATED 25% from v13)
        if model_spread < 2.0:
            base_std = 2.25   # High agreement
        elif model_spread < 4.0:
            base_std = 3.1    # Medium agreement
        elif model_spread < 6.0:
            base_std = 4.4    # Low agreement
        else:
            base_std = 5.6    # Major disagreement
    
    # Lead-time adjustment
    if hours_until_target <= 6:
        lead_time_mult = 0.75
    elif hours_until_target <= 12:
        lead_time_mult = 0.85
    elif hours_until_target <= 18:
        lead_time_mult = 1.0
    else:
        lead_time_mult = 1.15
    
    # City-specific adjustment
    city_mult = CITY_UNCERTAINTY_MULTIPLIER.get(city_key, 1.0)
    
    final_std = base_std * lead_time_mult * city_mult
    
    return final_std, model_spread


def get_min_edge(price, agreement_level='high'):
    """Flat edge requirement regardless of agreement level."""
    return BASE_EDGE_REQUIREMENT


def get_price_bucket(price):
    """Categorize price into buckets."""
    return "sweet_spot" if price <= SWEET_SPOT_HIGH else "high_price"


# ============ KELLY CRITERION ============

def calculate_kelly_bet(bankroll, our_prob, bet_price, kelly_fraction=KELLY_FRACTION):
    """Calculate optimal bet size using Kelly Criterion, accounting for Kalshi fees."""
    if bet_price <= 0 or bet_price >= 1:
        return 0, {}
    
    b_gross = (1 / bet_price) - 1
    b_net = b_gross - KALSHI_FEE_RATE
    
    p = our_prob
    q = 1 - p
    
    full_kelly = (p * b_net - q) / b_net if b_net > 0 else 0
    fractional_kelly = full_kelly * kelly_fraction
    
    fractional_kelly = max(0, fractional_kelly)
    fractional_kelly = min(fractional_kelly, MAX_BET_FRACTION)
    
    bet_size = bankroll * fractional_kelly
    
    if bet_size < MIN_BET_SIZE:
        bet_size = 0
    
    kelly_info = {
        "full_kelly_pct": full_kelly * 100,
        "fractional_kelly_pct": fractional_kelly * 100,
        "odds_gross": b_gross,
        "odds_net": b_net,
        "fee_impact": KALSHI_FEE_RATE,
    }
    
    return bet_size, kelly_info


# ============ CITY CONFIGURATION ============
cities = {
    "chicago": {
        "name": "Chicago",
        "csv_file": "weather_data_chicago.csv",
        "lat": 41.79,
        "lon": -87.74,
        "kalshi_series": "KXHIGHCHI",
        "timezone": "America/Chicago",
        "utc_offset": -6
    },
    "nyc": {
        "name": "New York City",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.78,
        "lon": -73.97,
        "kalshi_series": "KXHIGHNY",
        "timezone": "America/New_York",
        "utc_offset": -5
    },
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.79,
        "lon": -80.31,
        "kalshi_series": "KXHIGHMIA",
        "timezone": "America/New_York",
        "utc_offset": -5
    },
    "austin": {
        "name": "Austin",
        "csv_file": "weather_data_austin.csv",
        "lat": 30.2,
        "lon": -97.68,
        "kalshi_series": "KXHIGHAUS",
        "timezone": "America/Chicago",
        "utc_offset": -6
    },
    "la": {
        "name": "Los Angeles",
        "csv_file": "weather_data_la.csv",
        "lat": 33.94,
        "lon": -118.42,
        "kalshi_series": "KXHIGHLAX",
        "timezone": "America/Los_Angeles",
        "utc_offset": -8
    }
}

# Default to all cities
selected_cities = ["chicago", "nyc", "miami", "austin", "la"]


# ============ HEADER ============
def print_header():
    print("=" * 70)
    print("KALSHI WEATHER BETTING MODEL v9.5")
    print("(MIXTURE DISTRIBUTION + Full 24h HRRR + T-Distribution)")
    print("=" * 70)
    print(f"\nAnalyzing: Chicago, New York City, Miami, Austin, Los Angeles")
    print(f"Agreement Threshold: ±{ENSEMBLE_AGREEMENT_THRESHOLD}°F")
    print(f"Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")
    print(f"Sweet Spot: {SWEET_SPOT_LOW*100:.0f}¢ - {SWEET_SPOT_HIGH*100:.0f}¢")
    print(f"⚠️ FEE ADJUSTMENT: {KALSHI_FEE_RATE*100:.0f}% entry fee built into odds")
    print(f"\n🎯 V9.5 KEY IMPROVEMENT:")
    print(f"   • MIXTURE DISTRIBUTION: Each model contributes its own probability")
    print(f"     distribution, not collapsed to a single point estimate")
    print(f"   • Better tail probability estimation when models disagree")
    print(f"\n🎯 KEPT FROM V9.4:")
    print(f"   • Full 24h calendar day HRRR sampling (cold fronts)")
    print(f"   • Student's t-distribution (df={T_DISTRIBUTION_DF}) for fat tails")
    print(f"   • Dynamic uncertainty scaling")
    print(f"   • Flat {BASE_EDGE_REQUIREMENT*100:.0f}% edge requirement")
    print(f"\n🎯 SMART BET SELECTION:")
    print(f"   • Different cities: Bet freely (uncorrelated)")
    print(f"   • Same city: Only stack if edge ratio ≥ {SAME_CITY_MULTI_BET_THRESHOLD}x")
    print(f"   • Max {MAX_BETS_PER_CITY} bets/city, {MAX_TOTAL_BETS_PER_DAY} total/day")
    print(f"\n💰 BANKROLL: ${STARTING_BANKROLL:.2f} | {KELLY_FRACTION:.0%} Kelly")
    print(f"Model Weights: HRRR ({MODEL_WEIGHTS['HRRR']*100:.0f}%) + NWS ({MODEL_WEIGHTS['NWS']*100:.0f}%) + Open-Meteo ({MODEL_WEIGHTS['Open-Meteo']*100:.0f}%)")


# ============ DATA FUNCTIONS ============

def load_and_prepare_data(city_config):
    """Load historical temperature data from CSV."""
    df = pd.read_csv(city_config["csv_file"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "value"]].copy()
    df = df.rename(columns={"value": "temp"})
    return df


def fetch_open_meteo(lat, lon, timezone):
    """Fetch forecast from Open-Meteo API using HOURLY data for true 24h max."""
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
        "hourly": "temperature_2m",
        "past_days": 7,
        "forecast_days": 3,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": timezone
    }
    response = requests.get(open_meteo_url, params=params)
    return response.json()


def get_open_meteo_24h_max(meteo_data, target_date_str):
    """Extract the true 24-hour max from Open-Meteo hourly data."""
    if "hourly" in meteo_data and "time" in meteo_data["hourly"]:
        hourly_times = meteo_data["hourly"]["time"]
        hourly_temps = meteo_data["hourly"]["temperature_2m"]
        
        target_temps = []
        for i, time_str in enumerate(hourly_times):
            if target_date_str in time_str and hourly_temps[i] is not None:
                target_temps.append(hourly_temps[i])
        
        if target_temps:
            return max(target_temps), True
    
    if "daily" in meteo_data:
        daily_dates = meteo_data["daily"]["time"]
        daily_maxes = meteo_data["daily"]["temperature_2m_max"]
        
        for i, d in enumerate(daily_dates):
            if d == target_date_str:
                return daily_maxes[i], False
    
    return None, False


def fetch_nws_forecast(lat, lon, target_date):
    """Fetch forecast from National Weather Service API."""
    try:
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        points_response = requests.get(
            points_url,
            headers={"User-Agent": "(KalshiWeatherModel, contact@example.com)"},
            timeout=10
        )

        if points_response.status_code != 200:
            return None

        points_data = points_response.json()
        forecast_url = points_data["properties"]["forecast"]

        forecast_response = requests.get(
            forecast_url,
            headers={"User-Agent": "(KalshiWeatherModel, contact@example.com)"},
            timeout=10
        )

        if forecast_response.status_code != 200:
            return None

        forecast_data = forecast_response.json()
        periods = forecast_data["properties"]["periods"]

        target_str = target_date.strftime("%Y-%m-%d")

        for period in periods:
            period_start = period.get("startTime", "")
            if target_str in period_start and period.get("isDaytime", False):
                temp = period.get("temperature")
                if temp:
                    return float(temp)

        return None

    except Exception as e:
        print(f"     ⚠️ NWS API error: {e}")
        return None


def fetch_hrrr_forecast_full_day(lat, lon, target_date, utc_offset):
    """Fetch HRRR forecast sampling the FULL 24-hour calendar day."""
    if not HERBIE_AVAILABLE:
        return None, None
    
    today = datetime.now().date()
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
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    
    midnight_local_utc = (0 - utc_offset) % 24
    
    target_start_utc = datetime.combine(target_date, datetime.min.time().replace(hour=midnight_local_utc))
    if midnight_local_utc > 12:
        target_start_utc -= timedelta(days=1)
    
    target_end_utc = target_start_utc + timedelta(hours=24)
    
    hours_to_day_start = int((target_start_utc - model_run_datetime).total_seconds() / 3600)
    hours_to_day_end = int((target_end_utc - model_run_datetime).total_seconds() / 3600)
    
    key_local_hours = [0, 2, 4, 6, 9, 12, 14, 16, 18, 21, 23]
    
    forecast_hours = []
    for local_hour in key_local_hours:
        utc_hour_offset = hours_to_day_start + local_hour
        if 1 <= utc_hour_offset <= 48:
            forecast_hours.append(utc_hour_offset)
    
    forecast_hours = sorted(set(forecast_hours))
    
    print(f"     HRRR Model run: {model_run_str} UTC")
    print(f"     Sampling FULL 24h calendar day (fxx: {forecast_hours[:3]}...{forecast_hours[-3:] if len(forecast_hours) > 3 else forecast_hours})")
    print(f"     Target: {target_date} local midnight-to-midnight")
    
    temperatures = []
    temp_by_hour = {}
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
            
            ds = H.xarray("TMP:2 m", remove_grib=True)
            temp_data = ds['t2m']
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            
            target_lon = lon if lon > 0 else lon + 360
            dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            temp_k = float(temp_data.values[min_idx])
            temp_f = (temp_k - 273.15) * 9/5 + 32
            temperatures.append(temp_f)
            
            local_hour = (fxx - hours_to_day_start) % 24
            temp_by_hour[local_hour] = temp_f
            
            try:
                ds_u = H.xarray("UGRD:10 m", remove_grib=True)
                ds_v = H.xarray("VGRD:10 m", remove_grib=True)
                u = float(ds_u['u10'].values[min_idx]) if 'u10' in ds_u else float(ds_u['u'].values[min_idx])
                v = float(ds_v['v10'].values[min_idx]) if 'v10' in ds_v else float(ds_v['v'].values[min_idx])
                wind_speed = np.sqrt(u**2 + v**2) * 2.237
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
    
    forecast_high = max(temperatures)
    
    max_temp_hour = max(temp_by_hour.keys(), key=lambda h: temp_by_hour[h]) if temp_by_hour else None
    high_time_str = f"{max_temp_hour}:00 local" if max_temp_hour is not None else "unknown"
    
    is_midnight_high = max_temp_hour is not None and (max_temp_hour <= 2 or max_temp_hour >= 22)
    
    print(f"     ✅ Retrieved {len(temperatures)} samples across 24h")
    print(f"     📊 24h temp range: {min(temperatures):.1f}°F to {max(temperatures):.1f}°F")
    print(f"     🎯 High of {forecast_high:.1f}°F expected around {high_time_str}")
    if is_midnight_high:
        print(f"     ⚠️  COLD FRONT DETECTED: High occurs near midnight!")
    
    avg_wind_speed = np.mean(wind_speeds) if wind_speeds else None
    avg_wind_dir = np.mean(wind_dirs) if wind_dirs else None
    
    hours_until_target = hours_to_day_start + 12
    
    weather_data = {
        'forecast_high': forecast_high,
        'all_temps': temperatures,
        'temp_by_hour': temp_by_hour,
        'high_occurs_at': high_time_str,
        'is_midnight_high': is_midnight_high,
        'wind_speed': avg_wind_speed,
        'wind_dir': avg_wind_dir,
        'model_run': model_run_str,
        'source': 'HRRR',
        'hours_until_target': hours_until_target
    }
    
    return forecast_high, weather_data


def get_bin_for_temp(temp):
    """Get the Kalshi bin that contains this temperature."""
    temp_rounded = int(np.round(temp))
    
    if temp_rounded % 2 == 1:
        lower = temp_rounded
    else:
        lower = temp_rounded - 1
    
    return (lower, lower + 1)


# ============ V9.5: MIXTURE DISTRIBUTION PROBABILITY MODELS ============

def mixture_probability_range(contract_low, contract_high, forecasts_with_weights, uncertainty_std, agreement_level='high'):
    """
    V9.5: Calculate probability using MIXTURE of t-distributions.
    
    Instead of collapsing forecasts to a single point, each model contributes
    its own distribution centered on ITS forecast. The final probability is
    the weighted sum of individual model probabilities.
    
    Args:
        contract_low: Lower bound of temperature bin
        contract_high: Upper bound of temperature bin
        forecasts_with_weights: List of (model_name, temperature, weight) tuples
        uncertainty_std: Base uncertainty standard deviation
        agreement_level: 'high', 'medium', or 'low' - adjusts uncertainty scale
    
    Returns:
        Probability that temperature falls in [contract_low, contract_high]
    """
    # Adjust uncertainty based on agreement level
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:
        adj_std = uncertainty_std * 1.3
    
    total_prob = 0.0
    total_weight = 0.0
    
    for model_name, temp, weight in forecasts_with_weights:
        if temp is None:
            continue
            
        # Each model contributes its own t-distribution centered on its forecast
        model_prob = stats.t.cdf(contract_high + 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=adj_std) - \
                     stats.t.cdf(contract_low - 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=adj_std)
        
        total_prob += weight * model_prob
        total_weight += weight
    
    # Normalize by actual weight (in case some models are missing)
    if total_weight > 0:
        total_prob = total_prob / total_weight
    
    return max(min(total_prob, 0.99), 0.01)


def mixture_probability_below(threshold, forecasts_with_weights, uncertainty_std, agreement_level='high'):
    """V9.5: Mixture t-distribution probability for 'X or below' contracts."""
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:
        adj_std = uncertainty_std * 1.3
    
    total_prob = 0.0
    total_weight = 0.0
    
    for model_name, temp, weight in forecasts_with_weights:
        if temp is None:
            continue
            
        model_prob = stats.t.cdf(threshold + 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=adj_std)
        total_prob += weight * model_prob
        total_weight += weight
    
    if total_weight > 0:
        total_prob = total_prob / total_weight
    
    return max(min(total_prob, 0.99), 0.01)


def mixture_probability_above(threshold, forecasts_with_weights, uncertainty_std, agreement_level='high'):
    """V9.5: Mixture t-distribution probability for 'X or above' contracts."""
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:
        adj_std = uncertainty_std * 1.3
    
    total_prob = 0.0
    total_weight = 0.0
    
    for model_name, temp, weight in forecasts_with_weights:
        if temp is None:
            continue
            
        model_prob = 1 - stats.t.cdf(threshold - 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=adj_std)
        total_prob += weight * model_prob
        total_weight += weight
    
    if total_weight > 0:
        total_prob = total_prob / total_weight
    
    return max(min(total_prob, 0.99), 0.01)


# Keep the old functions for comparison/fallback
def calibrated_probability_t(contract_low, contract_high, forecast, uncertainty_std, agreement_level='high'):
    """V9.4 style: Single point estimate probability (kept for reference)."""
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:
        adj_std = uncertainty_std * 1.3
    
    prob = stats.t.cdf(contract_high + 0.5, df=T_DISTRIBUTION_DF, loc=forecast, scale=adj_std) - \
           stats.t.cdf(contract_low - 0.5, df=T_DISTRIBUTION_DF, loc=forecast, scale=adj_std)
    
    return max(min(prob, 0.99), 0.01)


# ============ CITY ANALYSIS ============

def analyze_city(city_key, bankroll):
    """
    Analyze a single city and return all qualifying bets.
    
    V9.5 changes:
    - Uses MIXTURE distribution for probability calculation
    - Each model contributes its own distribution, not collapsed to point
    """
    city = cities[city_key]
    qualifying_bets = []

    print(f"\n{'='*70}")
    print(f"ANALYZING: {city['name'].upper()}")
    print(f"{'='*70}")

    # Load historical data
    try:
        df = load_and_prepare_data(city)
        print(f"Loaded {len(df)} historical records")
    except FileNotFoundError:
        print(f"❌ Error: {city['csv_file']} not found.")
        return qualifying_bets, None

    # Set target date
    now = datetime.now()
    today = now.date()
    current_hour = now.hour

    if current_hour < 6:
        target_date = today
        print(f"  ⏰ Running at {now.strftime('%I:%M %p')} - targeting TODAY ({today})")
    else:
        target_date = today + timedelta(days=1)

    target_str = target_date.strftime("%Y-%m-%d")

    # ============ FETCH FORECASTS ============
    print(f"\n  📡 Fetching forecasts for {target_str}...")

    # 1. Open-Meteo forecast
    print(f"\n  [1] Open-Meteo (24h max):")
    meteo_data = fetch_open_meteo(city["lat"], city["lon"], city["timezone"])
    open_meteo_forecast, used_hourly = get_open_meteo_24h_max(meteo_data, target_str)
    
    if open_meteo_forecast:
        source_type = "hourly 24h" if used_hourly else "daily"
        print(f"     ✅ Forecast: {open_meteo_forecast:.1f}°F ({source_type})")
    else:
        print(f"     ❌ No forecast found for {target_str}")

    # 2. HRRR forecast
    print(f"\n  [2] HRRR (3km, FULL 24h):")
    hrrr_forecast, hrrr_data = fetch_hrrr_forecast_full_day(
        city["lat"], city["lon"], target_date, city["utc_offset"]
    )

    if hrrr_forecast:
        print(f"     ✅ Forecast: {hrrr_forecast:.1f}°F")
    else:
        print("     ⚠️ HRRR unavailable")

    # 3. NWS forecast
    print(f"\n  [3] NWS (human-adjusted):")
    nws_forecast = fetch_nws_forecast(city["lat"], city["lon"], target_date)

    if nws_forecast:
        print(f"     ✅ Forecast: {nws_forecast:.1f}°F")
    else:
        print("     ⚠️ NWS unavailable")

    # ============ BUILD FORECASTS WITH WEIGHTS FOR MIXTURE ============
    # V9.5: Instead of collapsing to ensemble, we keep individual forecasts
    forecasts_with_weights = []
    forecasts_display = []
    
    if hrrr_forecast:
        forecasts_with_weights.append(("HRRR", hrrr_forecast, MODEL_WEIGHTS["HRRR"]))
        forecasts_display.append(("HRRR", hrrr_forecast))
    if open_meteo_forecast:
        forecasts_with_weights.append(("Open-Meteo", open_meteo_forecast, MODEL_WEIGHTS["Open-Meteo"]))
        forecasts_display.append(("Open-Meteo", open_meteo_forecast))
    if nws_forecast:
        forecasts_with_weights.append(("NWS", nws_forecast, MODEL_WEIGHTS["NWS"]))
        forecasts_display.append(("NWS", nws_forecast))

    if len(forecasts_with_weights) == 0:
        print(f"\n  ❌ No forecasts available!")
        return qualifying_bets, None

    # Still calculate ensemble for display and bin determination
    # But probabilities will use the mixture approach
    temps = [f[1] for f in forecasts_display]
    total_weight = sum(fw[2] for fw in forecasts_with_weights)
    ensemble_forecast = sum(fw[1] * fw[2] for fw in forecasts_with_weights) / total_weight

    # Calculate model spread for agreement level
    model_diff = max(temps) - min(temps) if len(temps) > 1 else 0

    if model_diff <= CONFIDENCE_BOOST_THRESHOLD:
        agreement_level = 'high'
    elif model_diff <= ENSEMBLE_AGREEMENT_THRESHOLD:
        agreement_level = 'medium'
    else:
        agreement_level = 'low'

    print(f"\n  📊 Model Comparison:")
    for name, temp in forecasts_display:
        weight = MODEL_WEIGHTS.get(name, 0)
        print(f"     {name}: {temp:.1f}°F (weight: {weight*100:.0f}%)")
    if len(forecasts_display) > 1:
        print(f"     Spread: {model_diff:.1f}°F → Agreement: {agreement_level.upper()}")

    print(f"\n  🎯 WEIGHTED MEAN: {ensemble_forecast:.1f}°F (for reference)")
    print(f"     ⚡ V9.5: Using MIXTURE distribution, not point estimate!")

    # Calculate dynamic uncertainty
    hours_until = hrrr_data.get('hours_until_target', 18) if hrrr_data else 18
    dynamic_uncertainty, _ = calculate_dynamic_uncertainty(
        hrrr_forecast, nws_forecast, open_meteo_forecast, city_key, hours_until
    )
    print(f"     Uncertainty (σ): {dynamic_uncertainty:.2f}°F per model (t-dist df={T_DISTRIBUTION_DF})")

    # Cold front warning
    is_midnight_high = hrrr_data.get('is_midnight_high', False) if hrrr_data else False
    high_occurs_at = hrrr_data.get('high_occurs_at', 'afternoon') if hrrr_data else 'afternoon'
    if is_midnight_high:
        print(f"     ⚠️ COLD FRONT: High expected around {high_occurs_at}")

    forecast_bin = get_bin_for_temp(ensemble_forecast)
    print(f"     Primary bin (from mean): {forecast_bin[0]}°-{forecast_bin[1]}°F")

    # Store city summary
    city_summary = {
        "city": city["name"],
        "ensemble_forecast": ensemble_forecast,
        "forecast_bin": forecast_bin,
        "agreement_level": agreement_level,
        "dynamic_uncertainty": dynamic_uncertainty,
        "hrrr_forecast": hrrr_forecast,
        "open_meteo_forecast": open_meteo_forecast,
        "nws_forecast": nws_forecast,
        "is_midnight_high": is_midnight_high,
        "high_occurs_at": high_occurs_at,
        "model_spread": model_diff,
        "forecasts_with_weights": forecasts_with_weights,
    }

    # ============ FETCH KALSHI DATA ============
    print(f"\n  💰 Fetching Kalshi market data...")

    kalshi_base = "https://api.elections.kalshi.com/trade-api/v2"
    params = {
        "series_ticker": city["kalshi_series"],
        "status": "open",
        "limit": 100
    }

    try:
        markets_response = requests.get(
            f"{kalshi_base}/markets",
            params=params,
            headers={"Accept": "application/json"}
        )

        if markets_response.status_code != 200:
            print(f"     ❌ API error: {markets_response.status_code}")
            return qualifying_bets, city_summary

        all_markets = markets_response.json().get("markets", [])

        target_kalshi = target_date.strftime("%y%b%d").upper()
        markets = [m for m in all_markets if target_kalshi in m.get("ticker", "")]

        if not markets:
            print(f"     ❌ No markets found for {target_str}")
            available_dates = set()
            for m in all_markets[:20]:
                ticker = m.get("ticker", "")
                parts = ticker.split("-")
                if len(parts) >= 2:
                    available_dates.add(parts[1])
            if available_dates:
                print(f"        Available dates: {', '.join(sorted(available_dates)[:5])}")
            return qualifying_bets, city_summary

        print(f"     ✅ Found {len(markets)} contracts for {target_str}")

    except Exception as e:
        print(f"     ❌ API error: {e}")
        return qualifying_bets, city_summary

    # ============ ANALYZE CONTRACTS ============
    print(f"\n  {'='*60}")
    print(f"  CONTRACT ANALYSIS (Filtered: {MIN_CONTRACT_PRICE*100:.0f}¢-{MAX_CONTRACT_PRICE*100:.0f}¢)")
    print(f"  Using MIXTURE distribution for probability calculation")
    print(f"  {'='*60}")

    skipped_cheap = 0
    skipped_expensive = 0

    for market in markets:
        ticker = market.get("ticker", "")
        subtitle = market.get("subtitle", "")

        yes_bid = market.get("yes_bid", 0) or 0
        yes_ask = market.get("yes_ask", 100) or 100
        last_price = market.get("last_price", 0) or 0

        if yes_bid > 0 and yes_ask < 100:
            kalshi_prob = (yes_bid + yes_ask) / 200
        elif last_price > 0:
            kalshi_prob = last_price / 100
        else:
            continue

        # === PRICE FILTERS ===
        if kalshi_prob < MIN_CONTRACT_PRICE:
            skipped_cheap += 1
            continue
        if kalshi_prob > MAX_CONTRACT_PRICE:
            skipped_expensive += 1
            continue

        # Parse contract
        numbers = re.findall(r'-?\d+', subtitle)
        contract_type = None

        if "to" in subtitle and len(numbers) >= 2:
            low = int(numbers[0])
            high = int(numbers[1])
            # V9.5: Use MIXTURE distribution
            model_prob = mixture_probability_range(
                low, high, forecasts_with_weights, dynamic_uncertainty, agreement_level
            )
            is_forecast_bin = (low == forecast_bin[0])
            contract_type = "range"

        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = mixture_probability_below(
                threshold, forecasts_with_weights, dynamic_uncertainty, agreement_level
            )
            is_forecast_bin = ensemble_forecast <= threshold
            contract_type = "below"

        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = mixture_probability_above(
                threshold, forecasts_with_weights, dynamic_uncertainty, agreement_level
            )
            is_forecast_bin = ensemble_forecast >= threshold
            contract_type = "above"
        else:
            continue

        # Evaluate YES bet
        yes_edge = model_prob - kalshi_prob
        min_edge = get_min_edge(kalshi_prob, agreement_level)
        
        if yes_edge > min_edge:
            edge_ratio = yes_edge / min_edge
            our_prob_win = model_prob
            bet_size, kelly_info = calculate_kelly_bet(bankroll, our_prob_win, kalshi_prob)
            
            if bet_size >= MIN_BET_SIZE:
                qualifying_bets.append({
                    "city": city["name"],
                    "city_key": city_key,
                    "subtitle": subtitle,
                    "side": "YES",
                    "bet_price": kalshi_prob,
                    "model_prob": model_prob,
                    "our_prob_win": our_prob_win,
                    "edge": yes_edge,
                    "edge_ratio": edge_ratio,
                    "min_edge": min_edge,
                    "bet_size": bet_size,
                    "kelly_info": kelly_info,
                    "price_bucket": get_price_bucket(kalshi_prob),
                    "contract_type": contract_type,
                    "is_forecast_bin": is_forecast_bin,
                    "agreement_level": agreement_level,
                    "ensemble_forecast": ensemble_forecast,
                    "dynamic_uncertainty": dynamic_uncertainty,
                    "prob_method": "mixture",
                })

        # Evaluate NO bet (don't bet NO on forecast bin)
        if not is_forecast_bin:
            no_prob = 1 - model_prob
            no_market = 1 - kalshi_prob
            no_edge = no_prob - no_market
            no_min_edge = get_min_edge(no_market, agreement_level)
            
            # Skip NO range bets in sweet spot
            no_bucket = get_price_bucket(no_market)
            skip_no_range = (contract_type == "range" and no_bucket == "sweet_spot")
            
            if no_edge > no_min_edge and not skip_no_range:
                edge_ratio = no_edge / no_min_edge
                our_prob_win = no_prob
                bet_size, kelly_info = calculate_kelly_bet(bankroll, our_prob_win, no_market)
                
                if bet_size >= MIN_BET_SIZE:
                    qualifying_bets.append({
                        "city": city["name"],
                        "city_key": city_key,
                        "subtitle": subtitle,
                        "side": "NO",
                        "bet_price": no_market,
                        "model_prob": model_prob,
                        "our_prob_win": our_prob_win,
                        "edge": no_edge,
                        "edge_ratio": edge_ratio,
                        "min_edge": no_min_edge,
                        "bet_size": bet_size,
                        "kelly_info": kelly_info,
                        "price_bucket": no_bucket,
                        "contract_type": contract_type,
                        "is_forecast_bin": False,
                        "agreement_level": agreement_level,
                        "ensemble_forecast": ensemble_forecast,
                        "dynamic_uncertainty": dynamic_uncertainty,
                        "prob_method": "mixture",
                    })

    print(f"\n  Found {len(qualifying_bets)} qualifying bets")
    print(f"  Filtered out: {skipped_cheap} cheap, {skipped_expensive} expensive")

    return qualifying_bets, city_summary


# ============ SMART BET SELECTION ============

def smart_select_bets(all_bets):
    """Select the top 6 bets by edge ratio."""
    if not all_bets:
        return []

    sorted_bets = sorted(all_bets, key=lambda x: -x["edge_ratio"])
    selected_bets = sorted_bets[:MAX_TOTAL_BETS_PER_DAY]

    for i, bet in enumerate(selected_bets):
        bet["selection_reason"] = f"top_{i+1}_by_edge"

    return selected_bets


def print_recommendations(selected_bets, all_bets, city_summaries, bankroll):
    """Print the final betting recommendations in a clear format."""

    now = datetime.now()
    if now.hour < 6:
        target_date = now.date().strftime("%A, %B %d, %Y")
    else:
        target_date = (now.date() + timedelta(days=1)).strftime("%A, %B %d, %Y")
    
    print(f"\n{'='*70}")
    print("🎯 TODAY'S BETTING RECOMMENDATIONS")
    print(f"📅 Target Date: {target_date}")
    print(f"💰 Bankroll: ${bankroll:.2f}")
    print(f"{'='*70}")
    
    if not selected_bets:
        print("\n🛑 NO BETS RECOMMENDED TODAY")
        print("\nThis could mean:")
        print("  • No contracts with sufficient edge after filtering")
        print("  • Models disagree significantly")
        print("  • Market is efficiently priced")
        print("\n💡 Sitting out is a valid strategy!")
        return
    
    cities_with_bets = set(b["city"] for b in selected_bets)
    num_cities = len(cities_with_bets)
    
    if num_cities > 1:
        print(f"\n✅ Found bets in {num_cities} DIFFERENT CITIES (uncorrelated):")
        print("   → Safe to bet on all of these - different weather = independent outcomes!")
    elif len(selected_bets) > 1:
        city_name = list(cities_with_bets)[0]
        print(f"\n🔥 Found {len(selected_bets)} bets in {city_name} - SUPER CONFIDENT:")
        print(f"   → Stacking same-city bets because edge ratio ≥ {SAME_CITY_MULTI_BET_THRESHOLD}x!")
    else:
        city_name = list(cities_with_bets)[0]
        print(f"\n⭐ Found 1 great bet in {city_name}:")
        print("   → This is your best opportunity today.")
    
    print()
    
    total_wager = 0
    total_potential = 0

    for i, bet in enumerate(selected_bets, 1):
        if bet["edge_ratio"] >= SUPER_CONFIDENT_EDGE_RATIO:
            confidence = "🔥 SUPER CONFIDENT"
        elif bet["edge_ratio"] >= SAME_CITY_MULTI_BET_THRESHOLD:
            confidence = "✓ High confidence"
        else:
            confidence = "⭐ Good opportunity"

        rank_label = f"(#{i} by edge)"

        bucket_emoji = "🎯" if bet["price_bucket"] == "sweet_spot" else "📊"
        forecast_marker = "📍" if bet.get("is_forecast_bin") else ""

        odds = bet["kelly_info"].get("odds_net", 0)
        potential_profit = bet["bet_size"] * odds

        print(f"   ┌─ BET #{i}: {bet['city']} {rank_label}")
        print(f"   │  {confidence}")
        print(f"   │  Contract: {bet['subtitle']} {forecast_marker}")
        print(f"   │  Side: {bet['side']} at {bet['bet_price']*100:.0f}¢ {bucket_emoji}")
        print(f"   │  Your probability: {bet['our_prob_win']*100:.1f}% (MIXTURE)")
        print(f"   │  Edge: {bet['edge']*100:+.1f}% ({bet['edge_ratio']:.1f}x minimum)")
        print(f"   │  Kelly: {bet['kelly_info'].get('fractional_kelly_pct', 0):.1f}% of bankroll")
        print(f"   │  Uncertainty: {bet.get('dynamic_uncertainty', 2.8):.2f}°F (t-dist)")
        print(f"   │")
        print(f"   │  💰 BET: ${bet['bet_size']:.2f}")
        print(f"   │  📈 Potential profit: ${potential_profit:.2f}")
        print(f"   └{'─'*50}")
        print()

        total_wager += bet["bet_size"]
        total_potential += potential_profit

    print(f"   {'='*55}")
    print(f"   TOTAL WAGER: ${total_wager:.2f} ({100*total_wager/bankroll:.1f}% of bankroll)")
    print(f"   POTENTIAL PROFIT: ${total_potential:.2f}")
    print(f"   {'='*55}")

    # City forecasts summary
    print(f"\n📊 FORECAST SUMMARY:")
    for summary in city_summaries:
        if summary:
            agreement_emoji = "✅" if summary["agreement_level"] == "high" else "⚠️" if summary["agreement_level"] == "medium" else "❌"
            midnight_marker = "🌙" if summary.get("is_midnight_high") else ""
            high_time = f" (high @ {summary.get('high_occurs_at', 'afternoon')})" if summary.get("is_midnight_high") else ""
            spread_info = f" spread={summary.get('model_spread', 0):.1f}°F"
            print(f"   {summary['city']}: {summary['ensemble_forecast']:.1f}°F → Bin {summary['forecast_bin'][0]}-{summary['forecast_bin'][1]}° {agreement_emoji} {summary['agreement_level'].upper()} (σ={summary.get('dynamic_uncertainty', 2.8):.1f}°F{spread_info}) {midnight_marker}{high_time}")

    not_selected = [b for b in all_bets if b not in selected_bets]
    if not_selected:
        print(f"\n📋 Also considered but not selected: {len(not_selected)} other opportunities")
        print("   (Beyond top 6 by edge ratio)")


# ============ ARGUMENT PARSING ============

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Kalshi Weather Betting Model v9.5")

    parser.add_argument("--kelly", type=float, default=KELLY_FRACTION,
                        help=f"Kelly fraction (default: {KELLY_FRACTION})")
    parser.add_argument("--bankroll", type=float, default=STARTING_BANKROLL,
                        help=f"Starting bankroll (default: ${STARTING_BANKROLL})")
    parser.add_argument("--cities", type=str, default="all",
                        help="Comma-separated city list or 'all' (default: all)")
    parser.add_argument("--today", action="store_true",
                        help="Analyze today's weather")
    parser.add_argument("--tomorrow", action="store_true",
                        help="Analyze tomorrow's weather")
    parser.add_argument("--date", type=str, default=None,
                        help="Specific date to analyze (YYYY-MM-DD)")

    return parser.parse_args()


# ============ MAIN ============

def main():
    global KELLY_FRACTION, STARTING_BANKROLL, selected_cities

    args = parse_arguments()

    KELLY_FRACTION = args.kelly
    STARTING_BANKROLL = args.bankroll

    if args.cities and args.cities.lower() != "all":
        city_input = args.cities.lower().strip()
        if "," in city_input:
            selected_cities = [c.strip() for c in city_input.split(",")]
        else:
            selected_cities = [city_input]
        valid_cities = []
        for city in selected_cities:
            if city in cities:
                valid_cities.append(city)
            else:
                print(f"⚠️ Unknown city '{city}', skipping...")
        selected_cities = valid_cities if valid_cities else ["chicago", "nyc", "miami", "austin", "la"]

    if args.today:
        print("📅 Mode: Analyzing TODAY's weather")
    elif args.tomorrow:
        print("📅 Mode: Analyzing TOMORROW's weather")
    elif args.date:
        print(f"📅 Mode: Analyzing weather for {args.date}")

    print_header()

    bankroll = STARTING_BANKROLL
    all_bets = []
    city_summaries = []

    for city_key in selected_cities:
        city_bets, city_summary = analyze_city(city_key, bankroll)
        all_bets.extend(city_bets)
        city_summaries.append(city_summary)

    selected_bets = smart_select_bets(all_bets)

    print_recommendations(selected_bets, all_bets, city_summaries, bankroll)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE (v9.5 - MIXTURE DISTRIBUTION)")
    print(f"{'='*70}")
    print("\n💡 V9.5 KEY CHANGE:")
    print("   • MIXTURE DISTRIBUTION: Each model (HRRR/NWS/Open-Meteo) contributes")
    print("     its own probability distribution centered on ITS forecast")
    print("   • No longer collapses to single point estimate before calculating probs")
    print("   • Better captures tail probabilities when models disagree")
    print("\n   Example: If HRRR=72°F, NWS=76°F, Open-Meteo=74°F:")
    print("   • v9.4 would center probability at 73.35°F")
    print("   • v9.5 weights three separate distributions at 72°F, 76°F, and 74°F")
    print("   • Bins near individual model forecasts get more probability")


if __name__ == "__main__":
    main()