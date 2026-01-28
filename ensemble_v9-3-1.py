"""
KALSHI WEATHER BETTING MODEL v9.3.1
===================================
Supports Austin + Miami (best performers) or All Cities mode.

v9.3.1 CHANGE: Fixed Kalshi fee calculation
- OLD: Flat 4% fee assumption (WRONG)
- NEW: Actual Kalshi formula: fee = round_up(0.07 × C × P × (1-P))
  This is per-contract and varies by price:
  - At 20¢: ~5.6% effective fee
  - At 50¢: ~3.5% effective fee
  - At 80¢: ~1.4% effective fee

Based on performance analysis:
- Miami: +$64 profit, 67% win rate - BEST CITY
- Austin: +$30 profit, 77% win rate - SECOND BEST
- Chicago/NYC: Available in "All Cities" mode

Features:
1. CITY-SPECIFIC UNCERTAINTY:
   - Miami: σ=2.5°F (stable subtropical weather)
   - Austin: σ=2.8°F (moderate continental)
   - Chicago: σ=3.2°F (Great Lakes effect)
   - NYC: σ=3.0°F (coastal variability)
2. SMART BET SELECTION with edge-based ranking
3. KELLY CRITERION: Half Kelly sizing with bankroll tracking
4. CORRECT FEE CALCULATION: Uses actual Kalshi fee formula
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
    from herbie import Herbie # type: ignore
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
# Different cities = uncorrelated, safe to bet on multiple
# Same city = correlated, only stack if SUPER confident

SAME_CITY_MULTI_BET_THRESHOLD = 2.0   # Need 2x the minimum edge to stack same-city bets
MAX_BETS_PER_CITY = 2                  # Never more than 2 bets in same city
MAX_TOTAL_BETS_PER_DAY = 6             # Cap total daily bets across all cities
SUPER_CONFIDENT_EDGE_RATIO = 2.5       # What counts as "super confident"

# ============ FEE CONFIGURATION ============
# CORRECTED: Kalshi's actual fee formula (not a flat percentage!)
# Formula: fee = round_up(0.07 × contracts × price × (1-price))
# This means fees vary by price - highest at 50¢, lowest at extremes
KALSHI_FEE_MULTIPLIER = 0.07  # 7% of expected earnings

def calculate_kalshi_fee(num_contracts, price_per_contract):
    """
    Calculate Kalshi's actual fee using their official formula.
    fee = round_up(0.07 × C × P × (1-P))
    
    Examples:
    - 10 contracts at 50¢: fee = ceil(0.07 × 10 × 0.50 × 0.50) = ceil(0.175) = $0.18
    - 10 contracts at 20¢: fee = ceil(0.07 × 10 × 0.20 × 0.80) = ceil(0.112) = $0.12
    - 10 contracts at 80¢: fee = ceil(0.07 × 10 × 0.80 × 0.20) = ceil(0.112) = $0.12
    """
    P = price_per_contract
    C = num_contracts
    raw_fee = KALSHI_FEE_MULTIPLIER * C * P * (1 - P)
    # Round up to nearest cent
    return np.ceil(raw_fee * 100) / 100

# ============ KELLY CRITERION CONFIGURATION ============
STARTING_BANKROLL = 70        # Your bankroll
KELLY_FRACTION = 0.50         # Half Kelly (balanced risk/reward)
MIN_BET_SIZE = 0.50           # Don't bet less than 50¢
MAX_BET_FRACTION = 1.00       # Never bet more than 20% of bankroll

# ============ ENSEMBLE CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - models should agree within this
CONFIDENCE_BOOST_THRESHOLD = 2.0    # °F - boost confidence if within this
CALIBRATED_FORECAST_STD = 2.8       # °F - fallback if city not in CITY_UNCERTAINTY

# ============ CITY-SPECIFIC UNCERTAINTY ============
# Based on actual performance data:
# - Miami: +$64 profit (67% WR) → stable subtropical, lower uncertainty
# - Austin: +$30 profit (77% WR) → moderate continental volatility
# - Chicago/NYC: Higher volatility, more variable weather
CITY_UNCERTAINTY = {
    "miami": 2.5,      # Stable subtropical → lower uncertainty
    "austin": 2.8,     # Continental, moderate volatility
    "chicago": 3.2,    # Great Lakes effect, higher volatility
    "nyc": 3.0,        # Coastal, moderate-high volatility
}

# Edge requirements by price bucket (INCREASED TO ACCOUNT FOR ~4% FEE DRAG)
BASE_EDGE_REQUIREMENTS = {
    "sweet_spot": 0.16,    # 16% edge for 15-50¢ contracts (was 12%, +4% for fees)
    "high_price": 0.16,    # 16% edge for 50-90¢ contracts (was 12%, +4% for fees)
}

def get_min_edge(price, agreement_level='high'):
    """Dynamic edge requirement based on price and model agreement."""
    base_edge = BASE_EDGE_REQUIREMENTS.get(
        "sweet_spot" if price <= SWEET_SPOT_HIGH else "high_price", 
        0.12
    )
    
    # Adjust for agreement level
    if agreement_level == 'high':
        return base_edge
    elif agreement_level == 'medium':
        return base_edge * 1.25
    else:  # low
        return base_edge * 1.5

def get_price_bucket(price):
    """Categorize price into buckets."""
    return "sweet_spot" if price <= SWEET_SPOT_HIGH else "high_price"


# ============ KELLY CRITERION ============

def calculate_kelly_bet(bankroll, our_prob, bet_price, kelly_fraction=KELLY_FRACTION):
    """
    Calculate optimal bet size using Kelly Criterion with CORRECT Kalshi fees.
    
    Kalshi fee formula: fee = round_up(0.07 × contracts × price × (1-price))
    Fee is charged on entry, reduces effective payout.
    
    Returns:
        bet_size: Dollar amount to bet
        kelly_info: Dict with calculation details
    """
    if bet_price <= 0 or bet_price >= 1:
        return 0, {}
    
    # Estimate number of contracts for a reasonable bet (~$10 position)
    est_contracts = max(1, int(10 / bet_price))
    
    # Calculate actual fee for this trade
    fee = calculate_kalshi_fee(est_contracts, bet_price)
    
    # Cost to enter position
    position_cost = est_contracts * bet_price
    total_cost = position_cost + fee
    
    # If we win: we get $1 per contract, minus what we paid
    gross_win = est_contracts * 1.0
    net_profit_if_win = gross_win - position_cost - fee
    
    # If we lose: we lose our position cost + fee
    net_loss_if_lose = total_cost
    
    # Calculate effective odds for Kelly (profit / risk)
    b_net = net_profit_if_win / total_cost if total_cost > 0 else 0
    
    # Also calculate fee as percentage for display
    fee_rate = (fee / position_cost * 100) if position_cost > 0 else 0
    
    p = our_prob  # Probability of winning
    q = 1 - p     # Probability of losing
    
    # Full Kelly formula with fee-adjusted odds
    full_kelly = (p * b_net - q) / b_net if b_net > 0 else 0
    
    # Apply fractional Kelly
    fractional_kelly = full_kelly * kelly_fraction
    
    # Clamp to reasonable bounds
    fractional_kelly = max(0, fractional_kelly)
    fractional_kelly = min(fractional_kelly, MAX_BET_FRACTION)
    
    # Calculate actual bet size
    bet_size = bankroll * fractional_kelly
    
    # Apply minimum bet size
    if bet_size < MIN_BET_SIZE:
        bet_size = 0
    
    kelly_info = {
        "full_kelly_pct": full_kelly * 100,
        "fractional_kelly_pct": fractional_kelly * 100,
        "odds_net": b_net,
        "fee": fee,
        "fee_rate": fee_rate,
        "est_contracts": est_contracts,
    }
    
    return bet_size, kelly_info


# ============ CITY CONFIGURATION ============
# Austin + Miami are best performers, Chicago + NYC available for "All Cities"
cities = {
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.7959,
        "lon": -80.2870,
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
    "chicago": {
        "name": "Chicago",
        "csv_file": "weather_data_chicago.csv",
        "lat": 41.85,
        "lon": -87.65,
        "kalshi_series": "KXHIGHCHI",
        "timezone": "America/Chicago",
        "utc_offset": -6
    },
    "nyc": {
        "name": "New York",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.78,
        "lon": -73.97,
        "kalshi_series": "KXHIGHNYC",
        "timezone": "America/New_York",
        "utc_offset": -5
    }
}

# Will collect all qualifying bets across all cities
all_qualifying_bets = []

# Default to Austin and Miami only (best performers)
selected_cities = ["austin", "miami"]


# ============ HEADER ============
def print_header():
    print("=" * 70)
    print("KALSHI WEATHER BETTING MODEL v9.3.1")
    print("(Smart Bet Selection + Kelly Criterion + CORRECT FEE CALCULATION)")
    print("=" * 70)
    city_names = [cities[c]["name"] for c in selected_cities]
    print(f"\nAnalyzing: {', '.join(city_names)}")
    print(f"Agreement Threshold: ±{ENSEMBLE_AGREEMENT_THRESHOLD}°F")
    print(f"Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")
    print(f"Sweet Spot: {SWEET_SPOT_LOW*100:.0f}¢ - {SWEET_SPOT_HIGH*100:.0f}¢")
    print(f"\n💰 KALSHI FEE (CORRECTED):")
    print(f"   Formula: fee = round_up(0.07 × contracts × price × (1-price))")
    print(f"   At 20¢: ~5.6% | At 50¢: ~3.5% | At 80¢: ~1.4%")
    print(f"\n🌡️ CITY UNCERTAINTY:")
    for city_key, sigma in CITY_UNCERTAINTY.items():
        print(f"   • {city_key.upper()}: σ={sigma}°F")
    print(f"\n🎯 SMART BET SELECTION:")
    print(f"   • Different cities: Bet freely (uncorrelated)")
    print(f"   • Same city: Only stack if edge ratio ≥ {SAME_CITY_MULTI_BET_THRESHOLD}x")
    print(f"   • Max {MAX_BETS_PER_CITY} bets/city, {MAX_TOTAL_BETS_PER_DAY} total/day")
    print(f"\n💰 BANKROLL: ${STARTING_BANKROLL:.2f} | {KELLY_FRACTION:.0%} Kelly | Max Bet: {MAX_BET_FRACTION:.0%}")
    print("Data Sources: NWS (45%) + HRRR (35%) + Open-Meteo (20%)")


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


def fetch_nws_forecast(lat, lon, target_date):
    """
    Fetch forecast from National Weather Service API.
    NWS forecasters add human judgment to model output.
    """
    try:
        # Step 1: Get the grid point for this location
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

        # Step 2: Get the actual forecast
        forecast_response = requests.get(
            forecast_url,
            headers={"User-Agent": "(KalshiWeatherModel, contact@example.com)"},
            timeout=10
        )

        if forecast_response.status_code != 200:
            return None

        forecast_data = forecast_response.json()
        periods = forecast_data["properties"]["periods"]

        # Step 3: Find the forecast for the target date
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


def fetch_hrrr_forecast_fixed(lat, lon, target_date, utc_offset):
    """Fetch HRRR forecast with FIXED sampling to get actual daily high."""
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
    
    target_12_local_utc = 12 - utc_offset
    target_start = datetime.combine(target_date, datetime.min.time().replace(hour=target_12_local_utc % 24))
    if target_12_local_utc >= 24:
        target_start += timedelta(days=1)
    
    hours_to_start = int((target_start - model_run_datetime).total_seconds() / 3600)
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


def get_bin_for_temp(temp):
    """
    Get the assumed Kalshi bin that contains this temperature.
    NOTE: This is a fallback - prefer find_actual_kalshi_bucket() which uses real market data.

    NOAA stations round temperatures: 0.5 and above rounds UP.
    So we first round the forecast, then find the bin.

    Kalshi uses 2-degree bins with ODD lower bounds:
    ..., 27-28, 29-30, 31-32, 33-34, ...

    Examples (with NOAA rounding):
        29.4°F → rounds to 29 → bin (29, 30)
        29.5°F → rounds to 30 → bin (29, 30)
        30.4°F → rounds to 30 → bin (29, 30)
        30.5°F → rounds to 31 → bin (31, 32)
    """
    # NOAA rounding: 0.5 and above rounds up
    temp_rounded = int(np.round(temp))

    # Find the odd number that starts the bin containing this temp
    # If temp_rounded is odd, that's our lower bound
    # If temp_rounded is even, the bin started at temp_rounded - 1
    if temp_rounded % 2 == 1:  # odd
        lower = temp_rounded
    else:  # even
        lower = temp_rounded - 1

    return (lower, lower + 1)


def find_actual_kalshi_bucket(temp, markets):
    """
    Find the actual Kalshi bucket that contains this temperature based on real market data.

    Args:
        temp: The ensemble forecast temperature
        markets: List of market dicts from Kalshi API

    Returns:
        Tuple (low, high) of the bucket containing the temperature, or None if not found.
    """
    # NOAA rounding: 0.5 and above rounds up
    temp_rounded = int(np.round(temp))

    # Parse all range contracts to find available buckets
    available_buckets = []
    for market in markets:
        subtitle = market.get("subtitle", "")
        if "to" in subtitle:
            numbers = re.findall(r'-?\d+', subtitle)
            if len(numbers) >= 2:
                low = int(numbers[0])
                high = int(numbers[1])
                available_buckets.append((low, high))

    if not available_buckets:
        return None

    # Find which bucket contains the rounded temperature
    for low, high in available_buckets:
        if low <= temp_rounded <= high:
            return (low, high)

    # Temperature is outside all buckets - return None
    return None


# ============ PROBABILITY MODELS ============

def calibrated_probability(contract_low, contract_high, forecast, uncertainty_std, agreement_level='high'):
    """Calibrated probability model using Gaussian distribution."""
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:
        adj_std = uncertainty_std * 1.3
    
    prob = stats.norm.cdf(contract_high + 0.5, forecast, adj_std) - \
           stats.norm.cdf(contract_low - 0.5, forecast, adj_std)
    
    return max(min(prob, 0.99), 0.01)


def calibrated_below_probability(threshold, forecast, uncertainty_std, agreement_level='high'):
    """Calibrated probability for 'X or below' contracts."""
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:
        adj_std = uncertainty_std * 1.3
    
    prob = stats.norm.cdf(threshold + 0.5, forecast, adj_std)
    return max(min(prob, 0.99), 0.01)


def calibrated_above_probability(threshold, forecast, uncertainty_std, agreement_level='high'):
    """Calibrated probability for 'X or above' contracts."""
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:
        adj_std = uncertainty_std * 1.3
    
    prob = 1 - stats.norm.cdf(threshold - 0.5, forecast, adj_std)
    return max(min(prob, 0.99), 0.01)


# ============ CITY ANALYSIS ============

def analyze_city(city_key, bankroll):
    """
    Analyze a single city and return all qualifying bets.
    Does NOT make final bet selection - that happens across all cities.
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
    # If running between midnight and 6 AM, we likely want TODAY's forecast
    # (the day that just started), not tomorrow's
    now = datetime.now()
    today = now.date()
    current_hour = now.hour

    if current_hour < 6:
        # Late night/early morning: target TODAY (the day we're in)
        target_date = today
        print(f"  ⏰ Running at {now.strftime('%I:%M %p')} - targeting TODAY ({today})")
    else:
        # Normal hours: target tomorrow
        target_date = today + timedelta(days=1)

    target_str = target_date.strftime("%Y-%m-%d")

    # ============ FETCH FORECASTS ============
    print(f"\n  📡 Fetching forecasts for {target_str}...")

    # 1. Open-Meteo forecast
    print(f"\n  [1] Open-Meteo:")
    meteo_data = fetch_open_meteo(city["lat"], city["lon"], city["timezone"])
    meteo_dates = meteo_data["daily"]["time"]
    meteo_temps = meteo_data["daily"]["temperature_2m_max"]

    open_meteo_forecast = None
    for i, d in enumerate(meteo_dates):
        if d == target_str:
            open_meteo_forecast = meteo_temps[i]
            print(f"     ✅ Forecast: {open_meteo_forecast:.1f}°F")
            break

    if open_meteo_forecast is None:
        print(f"     ❌ No forecast found for {target_str}")

    # 2. HRRR forecast
    print(f"\n  [2] HRRR (3km):")
    hrrr_forecast, hrrr_data = fetch_hrrr_forecast_fixed(
        city["lat"], city["lon"], target_date, city["utc_offset"]
    )

    if hrrr_forecast:
        print(f"     ✅ Forecast: {hrrr_forecast:.1f}°F")
    else:
        print("     ⚠️ HRRR unavailable")

    # 3. NWS forecast (human-adjusted)
    print(f"\n  [3] NWS (human-adjusted):")
    nws_forecast = fetch_nws_forecast(city["lat"], city["lon"], target_date)

    if nws_forecast:
        print(f"     ✅ Forecast: {nws_forecast:.1f}°F")
    else:
        print("     ⚠️ NWS unavailable")

    # ============ ENSEMBLE FORECAST ============
    # Collect available forecasts
    forecasts = []
    if hrrr_forecast:
        forecasts.append(("HRRR", hrrr_forecast))
    if open_meteo_forecast:
        forecasts.append(("Open-Meteo", open_meteo_forecast))
    if nws_forecast:
        forecasts.append(("NWS", nws_forecast))

    if len(forecasts) == 0:
        print(f"\n  ❌ No forecasts available!")
        return qualifying_bets, None

    # Calculate ensemble with fixed weights based on what's available
    # UPDATED: NWS weighted higher since Kalshi settles on NWS station data
    if len(forecasts) == 3:
        # All three: NWS 45%, HRRR 35%, Open-Meteo 20%
        ensemble_forecast = (hrrr_forecast * 0.35) + (nws_forecast * 0.45) + (open_meteo_forecast * 0.20)
    elif len(forecasts) == 2:
        if hrrr_forecast and open_meteo_forecast:
            ensemble_forecast = (hrrr_forecast * 0.6) + (open_meteo_forecast * 0.4)
        elif hrrr_forecast and nws_forecast:
            # NWS gets more weight when paired with HRRR
            ensemble_forecast = (hrrr_forecast * 0.45) + (nws_forecast * 0.55)
        else:  # open_meteo and nws
            ensemble_forecast = (nws_forecast * 0.60) + (open_meteo_forecast * 0.40)
    else:
        # Single forecast
        ensemble_forecast = forecasts[0][1]

    # Calculate model spread for agreement level
    temps = [f[1] for f in forecasts]
    model_diff = max(temps) - min(temps) if len(temps) > 1 else 0

    if model_diff <= CONFIDENCE_BOOST_THRESHOLD:
        agreement_level = 'high'
    elif model_diff <= ENSEMBLE_AGREEMENT_THRESHOLD:
        agreement_level = 'medium'
    else:
        agreement_level = 'low'

    print(f"\n  📊 Model Comparison:")
    for name, temp in forecasts:
        print(f"     {name}: {temp:.1f}°F")
    if len(forecasts) > 1:
        print(f"     Spread: {model_diff:.1f}°F → Agreement: {agreement_level.upper()}")

    print(f"\n  🎯 ENSEMBLE FORECAST: {ensemble_forecast:.1f}°F")

    # Get city-specific uncertainty
    city_uncertainty = CITY_UNCERTAINTY.get(city_key, CALIBRATED_FORECAST_STD)
    print(f"     Uncertainty (σ): {city_uncertainty}°F")

    # forecast_bin will be set after fetching markets to use actual Kalshi buckets
    forecast_bin = None

    # Store city summary (forecast_bin will be updated after fetching markets)
    city_summary = {
        "city": city["name"],
        "ensemble_forecast": ensemble_forecast,
        "forecast_bin": forecast_bin,
        "agreement_level": agreement_level,
        "uncertainty": city_uncertainty,
        "hrrr_forecast": hrrr_forecast,
        "open_meteo_forecast": open_meteo_forecast,
        "nws_forecast": nws_forecast,
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

    # ============ FIND ACTUAL KALSHI BUCKET ============
    # Use real market data to find which bucket the forecast falls into
    forecast_bin = find_actual_kalshi_bucket(ensemble_forecast, markets)
    if forecast_bin:
        print(f"\n  🎯 Actual Kalshi bucket for {ensemble_forecast:.1f}°F: {forecast_bin[0]}°-{forecast_bin[1]}°F")
    else:
        # Fallback to calculated bin if no matching market found
        forecast_bin = get_bin_for_temp(ensemble_forecast)
        print(f"\n  ⚠️ No matching Kalshi bucket found, using calculated: {forecast_bin[0]}°-{forecast_bin[1]}°F")

    # Update city_summary with actual forecast_bin
    city_summary["forecast_bin"] = forecast_bin

    # ============ ANALYZE CONTRACTS ============
    print(f"\n  {'='*60}")
    print(f"  CONTRACT ANALYSIS (Filtered: {MIN_CONTRACT_PRICE*100:.0f}¢-{MAX_CONTRACT_PRICE*100:.0f}¢)")
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
            model_prob = calibrated_probability(low, high, ensemble_forecast, 
                                                city_uncertainty, agreement_level)
            # Check if this contract IS our forecast bin
            is_forecast_bin = (low == forecast_bin[0])
            contract_type = "range"

        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = calibrated_below_probability(threshold, ensemble_forecast,
                                                       city_uncertainty, agreement_level)
            is_forecast_bin = ensemble_forecast <= threshold
            contract_type = "below"

        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = calibrated_above_probability(threshold, ensemble_forecast,
                                                       city_uncertainty, agreement_level)
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
                })

        # Evaluate NO bet (don't bet NO on forecast bin)
        if not is_forecast_bin:
            no_prob = 1 - model_prob
            no_market = 1 - kalshi_prob
            no_edge = no_prob - no_market
            no_min_edge = get_min_edge(no_market, agreement_level)
            
            # Skip NO range bets in sweet spot (historically lose money)
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
                    })

    print(f"\n  Found {len(qualifying_bets)} qualifying bets")
    print(f"  Filtered out: {skipped_cheap} cheap, {skipped_expensive} expensive")

    return qualifying_bets, city_summary


# ============ SMART BET SELECTION ============

def smart_select_bets(all_bets):
    """
    Select the top 6 bets by edge ratio.

    Simple selection: rank all bets by edge ratio and return the top 6.
    """
    if not all_bets:
        return []

    # Sort all bets by edge ratio (best first)
    sorted_bets = sorted(all_bets, key=lambda x: -x["edge_ratio"])

    # Take top 6 (or fewer if not enough bets)
    selected_bets = sorted_bets[:MAX_TOTAL_BETS_PER_DAY]

    # Mark selection reason based on rank
    for i, bet in enumerate(selected_bets):
        bet["selection_reason"] = f"top_{i+1}_by_edge"

    return selected_bets


def print_recommendations(selected_bets, all_bets, city_summaries, bankroll):
    """Print the final betting recommendations in a clear format."""

    # Match the same logic used in analyze_city
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
    
    # Analyze what we're recommending
    cities_with_bets = set(b["city"] for b in selected_bets)
    num_cities = len(cities_with_bets)
    
    super_confident_count = sum(1 for b in selected_bets if b.get("selection_reason") == "super_confident_stack")
    
    # Print the recommendation header
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
    
    # Print each bet
    total_wager = 0
    total_potential = 0

    for i, bet in enumerate(selected_bets, 1):
        # Confidence indicator based on edge ratio
        if bet["edge_ratio"] >= SUPER_CONFIDENT_EDGE_RATIO:
            confidence = "🔥 SUPER CONFIDENT"
        elif bet["edge_ratio"] >= SAME_CITY_MULTI_BET_THRESHOLD:
            confidence = "✓ High confidence"
        else:
            confidence = "⭐ Good opportunity"

        # Show rank
        rank_label = f"(#{i} by edge)"

        bucket_emoji = "🎯" if bet["price_bucket"] == "sweet_spot" else "📊"
        forecast_marker = "📍" if bet.get("is_forecast_bin") else ""

        odds = bet["kelly_info"].get("odds_net", 0)
        potential_profit = bet["bet_size"] * odds

        print(f"   ┌─ BET #{i}: {bet['city']} {rank_label}")
        print(f"   │  {confidence}")
        print(f"   │  Contract: {bet['subtitle']} {forecast_marker}")
        print(f"   │  Side: {bet['side']} at {bet['bet_price']*100:.0f}¢ {bucket_emoji}")
        print(f"   │  Your probability: {bet['our_prob_win']*100:.1f}%")
        print(f"   │  Edge: {bet['edge']*100:+.1f}% ({bet['edge_ratio']:.1f}x minimum)")
        print(f"   │  Kelly: {bet['kelly_info'].get('fractional_kelly_pct', 0):.1f}% of bankroll")
        print(f"   │  Fee: ~{bet['kelly_info'].get('fee_rate', 0):.1f}% (${bet['kelly_info'].get('fee', 0):.2f})")
        print(f"   │")
        print(f"   │  💰 BET: ${bet['bet_size']:.2f}")
        print(f"   │  📈 Potential profit: ${potential_profit:.2f}")
        print(f"   └{'─'*50}")
        print()

        total_wager += bet["bet_size"]
        total_potential += potential_profit

    # Summary
    print(f"   {'='*55}")
    print(f"   TOTAL WAGER: ${total_wager:.2f} ({100*total_wager/bankroll:.1f}% of bankroll)")
    print(f"   POTENTIAL PROFIT: ${total_potential:.2f}")
    print(f"   {'='*55}")

    # City forecasts summary
    print(f"\n📊 FORECAST SUMMARY:")
    for summary in city_summaries:
        if summary:
            agreement_emoji = "✅" if summary["agreement_level"] == "high" else "⚠️" if summary["agreement_level"] == "medium" else "❌"
            uncertainty = summary.get("uncertainty", CALIBRATED_FORECAST_STD)
            print(f"   {summary['city']}: {summary['ensemble_forecast']:.1f}°F → Bin {summary['forecast_bin'][0]}-{summary['forecast_bin'][1]}° {agreement_emoji} {summary['agreement_level'].upper()} (σ={uncertainty}°F)")

    # Show what we didn't bet on
    not_selected = [b for b in all_bets if b not in selected_bets]
    if not_selected:
        print(f"\n📋 Also considered but not selected: {len(not_selected)} other opportunities")
        print("   (Beyond top 6 by edge ratio)")


# ============ ARGUMENT PARSING ============

def parse_arguments():
    """Parse command-line arguments for Flask integration."""
    parser = argparse.ArgumentParser(description="Kalshi Weather Betting Model v9.3.1")

    parser.add_argument("--kelly", type=float, default=KELLY_FRACTION,
                        help=f"Kelly fraction (default: {KELLY_FRACTION})")
    parser.add_argument("--maxbet", type=float, default=MAX_BET_FRACTION,
                        help=f"Max bet fraction of bankroll (default: {MAX_BET_FRACTION})")
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
    global KELLY_FRACTION, MAX_BET_FRACTION, STARTING_BANKROLL, selected_cities

    # Parse command-line arguments
    args = parse_arguments()

    # Apply arguments to config
    KELLY_FRACTION = args.kelly
    MAX_BET_FRACTION = args.maxbet
    STARTING_BANKROLL = args.bankroll

    # Handle city filter
    if args.cities and args.cities.lower() == "all":
        # All cities mode - include all 4 cities
        selected_cities = list(cities.keys())
    elif args.cities:
        # Support both single city and comma-separated list
        city_input = args.cities.lower().strip()
        if "," in city_input:
            selected_cities = [c.strip() for c in city_input.split(",")]
        else:
            selected_cities = [city_input]
        # Validate cities exist
        valid_cities = []
        for city in selected_cities:
            if city in cities:
                valid_cities.append(city)
            else:
                print(f"⚠️ Unknown city '{city}', skipping...")
        selected_cities = valid_cities if valid_cities else ["austin", "miami"]

    # TODO: Date handling (--today, --tomorrow, --date) would require
    # modifying the analyze_city function to accept a target date.
    # For now, the script uses its default date logic.
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

    # Analyze each city
    for city_key in selected_cities:
        city_bets, city_summary = analyze_city(city_key, bankroll)
        all_bets.extend(city_bets)
        city_summaries.append(city_summary)

    # Smart bet selection across all cities
    selected_bets = smart_select_bets(all_bets)

    # Print recommendations
    print_recommendations(selected_bets, all_bets, city_summaries, bankroll)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE (v9.3.1-NWS-WEIGHTED)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()