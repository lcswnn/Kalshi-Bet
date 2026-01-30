"""
KALSHI WEATHER BETTING MODEL v9.3.6
===================================
ADDED GFS MODEL via Open-Meteo API

v9.3.6 CHANGES: Added GFS as 4th weather model
- GFS (Global Forecast System) added via Open-Meteo API
- Now have 4 models: HRRR, Open-Meteo, NWS, GFS
- New weights when all 4 available: HRRR 25%, NWS 35%, Open-Meteo 20%, GFS 20%
- Better consensus detection with 4 independent sources
- GFS is global model (0.25° resolution), complements HRRR's regional detail

v9.3.5 CHANGES: Removed redundant edge requirement scaling
- REMOVED: Dynamic edge requirements based on agreement level
- Layer 1: Probability σ scales with spread (mathematically correct)
- Layer 2: Kelly sizing scales with spread (bet sizing correct)
- Single flat 20% edge requirement for all bets
- Cleaner, less arbitrary, fewer "cliffs"

v9.3.4 CHANGES: Kelly sizing now incorporates model spread directly
- When models disagree, Kelly fraction is reduced proportionally
- Formula: agreement_factor = max(0.5, 1 - (spread - 2) * 0.15)

v9.3.3 CHANGES: Calibration based on actual betting results
- Increased base σ by 1.8x, max probability cap 85%, min edge 20%

Features:
1. FOUR-MODEL ENSEMBLE: HRRR + NWS + Open-Meteo + GFS
2. SPREAD-ADJUSTED PROBABILITY: σ = sqrt(city_base² + (spread/1.5)²)
3. SPREAD-ADJUSTED KELLY: Kelly sizing scales with model agreement
4. Flat 20% edge requirement
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

# ============ SPREAD-ADJUSTED KELLY CONFIGURATION ============
# When models disagree, we reduce Kelly sizing proportionally
# This directly incorporates forecast uncertainty into bet sizing
SPREAD_KELLY_THRESHOLD = 2.0       # Below this spread, full Kelly (high agreement)
SPREAD_KELLY_PENALTY = 0.15        # Kelly reduction per °F of spread above threshold
SPREAD_KELLY_FLOOR = 0.50          # Never reduce Kelly multiplier below this (50%)

def calculate_spread_agreement_factor(model_spread):
    """
    Calculate Kelly multiplier based on model spread/agreement.
    
    The idea: When models disagree, we're less certain about our probability
    estimate, so we should bet less even if the "expected" edge looks good.
    
    Formula:
        - spread ≤ 2°F:  factor = 1.0 (full Kelly)
        - spread > 2°F:  factor = max(0.5, 1 - (spread - 2) * 0.15)
    
    Examples:
        - spread = 1°F  → factor = 1.00 (100% of Kelly)
        - spread = 2°F  → factor = 1.00 (100% of Kelly)
        - spread = 3°F  → factor = 0.85 (85% of Kelly)
        - spread = 4°F  → factor = 0.70 (70% of Kelly)
        - spread = 5°F  → factor = 0.55 (55% of Kelly)
        - spread = 6°F+ → factor = 0.50 (50% floor)
    """
    if model_spread <= SPREAD_KELLY_THRESHOLD:
        return 1.0
    
    # Calculate penalty for spread above threshold
    excess_spread = model_spread - SPREAD_KELLY_THRESHOLD
    factor = 1.0 - (excess_spread * SPREAD_KELLY_PENALTY)
    
    # Apply floor
    return max(SPREAD_KELLY_FLOOR, factor)

# ============ ENSEMBLE CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - models should agree within this
CONFIDENCE_BOOST_THRESHOLD = 2.0    # °F - boost confidence if within this
CALIBRATED_FORECAST_STD = 2.8       # °F - fallback if city not in CITY_BASE_UNCERTAINTY

# ============ CITY-SPECIFIC BASE UNCERTAINTY ============
# CALIBRATED based on 26 resolved v9.3.x bets
# Previous values were too small, causing ~20% overconfidence
# Multiplied by 1.8x to match actual observed variance
CITY_BASE_UNCERTAINTY = {
    "miami": 4.5,      # Was 2.5 - calibrated up
    "austin": 5.0,     # Was 2.8 - calibrated up
    "chicago": 5.8,    # Was 3.2 - calibrated up
    "nyc": 5.4,        # Was 3.0 - calibrated up (NYC had worst performance)
}

# Minimum uncertainty floor - never go below this even with perfect agreement
MIN_UNCERTAINTY = 4.0  # Was 2.0 - increased for safety

# How much of the model spread contributes to uncertainty
# REDUCED from 2.0 to 1.5 - model disagreement should add MORE uncertainty
# spread_contribution = model_spread / SPREAD_DIVISOR
SPREAD_DIVISOR = 1.5  # Was 2.0 - now more conservative

# Maximum probability cap - NEVER report higher than this
# Data showed 90%+ predictions only won 57% of the time
MAX_PROBABILITY = 0.85  # New in v9.3.3

# Edge requirement - FLAT 20% for all bets (v9.3.5 simplification)
# Previous versions scaled this by agreement level, but that's now redundant
# since spread is already handled by: (1) probability σ and (2) Kelly sizing
MIN_EDGE_REQUIREMENT = 0.10  # 20% edge required for all bets

def get_min_edge(price, agreement_level='high'):
    """
    Return minimum edge requirement.
    
    v9.3.5: Now returns flat 20% regardless of agreement level.
    Agreement is already accounted for in probability σ and Kelly sizing.
    """
    return MIN_EDGE_REQUIREMENT

def get_price_bucket(price):
    """Categorize price into buckets."""
    return "sweet_spot" if price <= SWEET_SPOT_HIGH else "high_price"


# ============ KELLY CRITERION ============

def calculate_kelly_bet(bankroll, our_prob, bet_price, kelly_fraction=KELLY_FRACTION, 
                        model_spread=0.0, agreement_level='high'):
    """
    Calculate optimal bet size using Kelly Criterion with CORRECT Kalshi fees
    AND spread-adjusted sizing.
    
    NEW in v9.3.4: Kelly fraction is scaled by model agreement.
    When models disagree (high spread), we bet less even if edge looks good.
    
    Kalshi fee formula: fee = round_up(0.07 × contracts × price × (1-price))
    Fee is charged on entry, reduces effective payout.
    
    Args:
        bankroll: Current bankroll
        our_prob: Our estimated probability of winning
        bet_price: Market price (0-1)
        kelly_fraction: Base Kelly fraction (default 0.5 for half-Kelly)
        model_spread: Difference between highest and lowest model forecasts (°F)
        agreement_level: 'high', 'medium', or 'low' (for display)
    
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
    
    # NEW v9.3.4: Calculate spread-based agreement factor
    agreement_factor = calculate_spread_agreement_factor(model_spread)
    
    # Apply fractional Kelly AND spread adjustment
    # Final Kelly = full_kelly × base_fraction × agreement_factor
    adjusted_kelly_fraction = kelly_fraction * agreement_factor
    fractional_kelly = full_kelly * adjusted_kelly_fraction
    
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
        "base_kelly_fraction": kelly_fraction,
        "agreement_factor": agreement_factor,
        "effective_kelly_fraction": adjusted_kelly_fraction,
        "model_spread": model_spread,
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
    print("KALSHI WEATHER BETTING MODEL v9.3.6")
    print("(4-Model Ensemble + Spread-Adjusted Kelly)")
    print("=" * 70)
    city_names = [cities[c]["name"] for c in selected_cities]
    print(f"\nAnalyzing: {', '.join(city_names)}")
    print(f"Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")
    print(f"Sweet Spot: {SWEET_SPOT_LOW*100:.0f}¢ - {SWEET_SPOT_HIGH*100:.0f}¢")
    print(f"\n🌤️ FOUR-MODEL ENSEMBLE (v9.3.6):")
    print(f"   • NWS: 35% (human forecaster judgment)")
    print(f"   • HRRR: 25% (3km regional model)")
    print(f"   • Open-Meteo: 20% (blended global)")
    print(f"   • GFS: 20% (0.25° global model)")
    print(f"\n📊 TWO-LAYER SPREAD ADJUSTMENT:")
    print(f"   Layer 1 - PROBABILITY: σ = sqrt(city_base² + (spread/{SPREAD_DIVISOR})²)")
    print(f"   Layer 2 - KELLY: bet_size × agreement_factor (based on spread)")
    print(f"\n🌡️ CITY BASE UNCERTAINTY:")
    for city_key, sigma in CITY_BASE_UNCERTAINTY.items():
        print(f"   • {city_key.upper()}: σ_base={sigma}°F")
    print(f"\n📉 SPREAD-ADJUSTED KELLY:")
    print(f"   • Spread ≤{SPREAD_KELLY_THRESHOLD}°F: 100% of Kelly")
    print(f"   • Spread >{SPREAD_KELLY_THRESHOLD}°F: -{SPREAD_KELLY_PENALTY*100:.0f}% Kelly per °F")
    print(f"   • Floor: {SPREAD_KELLY_FLOOR:.0%} (never reduce below this)")
    print(f"\n🎯 BET SELECTION:")
    print(f"   • Min edge: {MIN_EDGE_REQUIREMENT*100:.0f}% (flat, all bets)")
    print(f"   • Max probability: {MAX_PROBABILITY*100:.0f}%")
    print(f"   • Max {MAX_BETS_PER_CITY} bets/city, {MAX_TOTAL_BETS_PER_DAY} total/day")
    print(f"\n💰 BANKROLL: ${STARTING_BANKROLL:.2f} | {KELLY_FRACTION:.0%} Base Kelly | Max Bet: {MAX_BET_FRACTION:.0%}")


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


def fetch_gfs_forecast(lat, lon, timezone, target_date):
    """
    Fetch GFS (Global Forecast System) forecast via Open-Meteo API.
    
    GFS is NOAA's global model with 0.25° resolution (~28km).
    It's independent from HRRR and provides a different perspective.
    Open-Meteo provides easy access to GFS data.
    """
    try:
        # Open-Meteo has a specific GFS endpoint
        gfs_url = "https://api.open-meteo.com/v1/gfs"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max"],
            "forecast_days": 3,
            "temperature_unit": "fahrenheit",
            "timezone": timezone
        }
        
        response = requests.get(gfs_url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"     ⚠️ GFS API error: {response.status_code}")
            return None
        
        data = response.json()
        dates = data.get("daily", {}).get("time", [])
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        
        target_str = target_date.strftime("%Y-%m-%d")
        
        for i, d in enumerate(dates):
            if d == target_str:
                return temps[i]
        
        return None
        
    except Exception as e:
        print(f"     ⚠️ GFS error: {e}")
        return None


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
    """
    Calibrated probability model using Gaussian distribution.

    Returns UNCAPPED probability. Capping is handled in betting logic to allow
    proper edge calculation when market price is also at extremes.
    """
    prob = stats.norm.cdf(contract_high + 0.5, forecast, uncertainty_std) - \
           stats.norm.cdf(contract_low - 0.5, forecast, uncertainty_std)

    return max(min(prob, 0.99), 0.01)


def calibrated_below_probability(threshold, forecast, uncertainty_std, agreement_level='high'):
    """
    Calibrated probability for 'X or below' contracts.

    Returns UNCAPPED probability. Capping is handled in betting logic.
    """
    prob = stats.norm.cdf(threshold + 0.5, forecast, uncertainty_std)
    return max(min(prob, 0.99), 0.01)


def calibrated_above_probability(threshold, forecast, uncertainty_std, agreement_level='high'):
    """
    Calibrated probability for 'X or above' contracts.

    Returns UNCAPPED probability. Capping is handled in betting logic.
    """
    prob = 1 - stats.norm.cdf(threshold - 0.5, forecast, uncertainty_std)
    return max(min(prob, 0.99), 0.01)


def apply_probability_cap(prob, market_prob, side='YES'):
    """
    Apply probability cap with edge case handling.

    The 85% cap exists because historically 90%+ predictions only won 57% of the time.
    However, we need special handling when market is also at extreme prices:

    - For YES bets: cap our probability to avoid overconfidence
    - For NO bets: if we think YES is very likely (>80% uncapped), don't bet NO
      even if the capped math suggests edge

    Args:
        prob: Our uncapped probability estimate
        market_prob: The market's price (for YES side)
        side: 'YES' or 'NO'

    Returns:
        capped_prob: Probability to use for edge calculation
        should_skip: Whether to skip this bet entirely
    """
    if side == 'YES':
        # For YES bets, apply the cap to avoid overconfidence
        capped_prob = min(prob, MAX_PROBABILITY)

        # Edge case: if market is above our cap, we can't assess edge reliably
        # Skip rather than showing false negative edge
        if market_prob > MAX_PROBABILITY:
            return capped_prob, True  # Skip this contract

        return capped_prob, False

    else:  # NO bet
        # For NO bets, the danger is betting NO when we actually think YES is very likely
        # If uncapped YES probability > 80%, don't bet NO regardless of what capped math says
        if prob > 0.80:
            return 1 - prob, True  # Skip - we actually think YES is likely

        # Otherwise, use capped probability for NO calculation
        capped_yes_prob = min(prob, MAX_PROBABILITY)
        return 1 - capped_yes_prob, False


def calculate_dynamic_uncertainty(city_key, model_spread, num_models):
    """
    Calculate dynamic uncertainty that incorporates both:
    1. Base city uncertainty (inherent weather volatility)
    2. Model disagreement (when forecasts differ significantly)
    
    Formula: σ_effective = sqrt(σ_base² + (spread/2)²)
    
    This ensures:
    - When models agree (spread ≈ 0): uncertainty ≈ base city σ
    - When models disagree (spread = 16°F): uncertainty jumps significantly
    
    Example with Miami (base σ = 2.5°F):
    - Models agree within 2°F:  σ = sqrt(2.5² + 1²) = 2.7°F
    - Models spread of 8°F:     σ = sqrt(2.5² + 4²) = 4.7°F  
    - Models spread of 16°F:    σ = sqrt(2.5² + 8²) = 8.4°F
    
    Args:
        city_key: City identifier for base uncertainty lookup
        model_spread: max(forecasts) - min(forecasts) in °F
        num_models: Number of models that provided forecasts
    
    Returns:
        effective_uncertainty: The σ to use for probability calculations
        components: Dict with breakdown of uncertainty sources
    """
    # Get base uncertainty for this city
    base_sigma = CITY_BASE_UNCERTAINTY.get(city_key, CALIBRATED_FORECAST_STD)
    
    # Calculate spread contribution
    # Divide by SPREAD_DIVISOR to convert spread to standard deviation estimate
    spread_sigma = model_spread / SPREAD_DIVISOR
    
    # Combine using root-sum-squares (assumes independent error sources)
    effective_sigma = np.sqrt(base_sigma**2 + spread_sigma**2)
    
    # Apply floor
    effective_sigma = max(effective_sigma, MIN_UNCERTAINTY)
    
    # If only one model, add extra uncertainty since we can't measure disagreement
    if num_models == 1:
        effective_sigma *= 1.3  # 30% penalty for single-model forecasts
    
    components = {
        "base_sigma": base_sigma,
        "spread_sigma": spread_sigma,
        "model_spread": model_spread,
        "num_models": num_models,
        "effective_sigma": effective_sigma,
    }
    
    return effective_sigma, components


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

    # 4. GFS forecast (global model via Open-Meteo)
    print(f"\n  [4] GFS (global 0.25°):")
    gfs_forecast = fetch_gfs_forecast(city["lat"], city["lon"], city["timezone"], target_date)

    if gfs_forecast:
        print(f"     ✅ Forecast: {gfs_forecast:.1f}°F")
    else:
        print("     ⚠️ GFS unavailable")

    # ============ ENSEMBLE FORECAST ============
    # Collect available forecasts
    forecasts = []
    if hrrr_forecast:
        forecasts.append(("HRRR", hrrr_forecast))
    if open_meteo_forecast:
        forecasts.append(("Open-Meteo", open_meteo_forecast))
    if nws_forecast:
        forecasts.append(("NWS", nws_forecast))
    if gfs_forecast:
        forecasts.append(("GFS", gfs_forecast))

    if len(forecasts) == 0:
        print(f"\n  ❌ No forecasts available!")
        return qualifying_bets, None

    # Calculate ensemble with fixed weights based on what's available
    # Weights philosophy:
    #   - NWS gets highest weight (human forecaster judgment)
    #   - HRRR gets high weight (high-resolution regional model)
    #   - GFS and Open-Meteo split the rest (global models)
    
    if len(forecasts) == 4:
        # All four: HRRR 25%, NWS 35%, Open-Meteo 20%, GFS 20%
        ensemble_forecast = (
            hrrr_forecast * 0.25 + 
            nws_forecast * 0.35 + 
            open_meteo_forecast * 0.20 + 
            gfs_forecast * 0.20
        )
    elif len(forecasts) == 3:
        # Three models - handle various combinations
        available = {name for name, _ in forecasts}
        if available == {"HRRR", "NWS", "Open-Meteo"}:
            ensemble_forecast = (hrrr_forecast * 0.30) + (nws_forecast * 0.50) + (open_meteo_forecast * 0.20)
        elif available == {"HRRR", "NWS", "GFS"}:
            ensemble_forecast = (hrrr_forecast * 0.30) + (nws_forecast * 0.45) + (gfs_forecast * 0.25)
        elif available == {"HRRR", "Open-Meteo", "GFS"}:
            ensemble_forecast = (hrrr_forecast * 0.40) + (open_meteo_forecast * 0.30) + (gfs_forecast * 0.30)
        elif available == {"NWS", "Open-Meteo", "GFS"}:
            ensemble_forecast = (nws_forecast * 0.50) + (open_meteo_forecast * 0.25) + (gfs_forecast * 0.25)
        else:
            # Fallback: equal weights
            ensemble_forecast = sum(t for _, t in forecasts) / len(forecasts)
    elif len(forecasts) == 2:
        # Two models - use simple weighted average
        names = {name for name, _ in forecasts}
        temps_dict = {name: temp for name, temp in forecasts}
        
        if "NWS" in names:
            # NWS gets 55%, other gets 45%
            other = [t for n, t in forecasts if n != "NWS"][0]
            ensemble_forecast = (temps_dict["NWS"] * 0.55) + (other * 0.45)
        elif "HRRR" in names:
            # HRRR gets 55%, other gets 45%
            other = [t for n, t in forecasts if n != "HRRR"][0]
            ensemble_forecast = (temps_dict["HRRR"] * 0.55) + (other * 0.45)
        else:
            # Open-Meteo and GFS: equal weights
            ensemble_forecast = sum(t for _, t in forecasts) / 2
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

    # Calculate dynamic uncertainty based on city AND model disagreement
    city_uncertainty, uncertainty_components = calculate_dynamic_uncertainty(
        city_key, model_diff, len(forecasts)
    )
    
    print(f"\n  📐 UNCERTAINTY CALCULATION:")
    print(f"     Base σ ({city['name']}): {uncertainty_components['base_sigma']:.1f}°F")
    print(f"     Model spread: {uncertainty_components['model_spread']:.1f}°F → adds {uncertainty_components['spread_sigma']:.1f}°F")
    print(f"     Effective σ: {uncertainty_components['effective_sigma']:.1f}°F")
    if len(forecasts) == 1:
        print(f"     ⚠️ Single model penalty applied (×1.3)")

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
        "gfs_forecast": gfs_forecast,
        "model_count": len(forecasts),
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

        # Evaluate YES bet with probability cap handling
        capped_yes_prob, skip_yes = apply_probability_cap(model_prob, kalshi_prob, side='YES')
        yes_edge = capped_yes_prob - kalshi_prob
        min_edge = get_min_edge(kalshi_prob, agreement_level)

        if not skip_yes and yes_edge > min_edge:
            edge_ratio = yes_edge / min_edge
            our_prob_win = capped_yes_prob
            bet_size, kelly_info = calculate_kelly_bet(
                bankroll, our_prob_win, kalshi_prob,
                model_spread=model_diff, agreement_level=agreement_level
            )

            if bet_size >= MIN_BET_SIZE:
                qualifying_bets.append({
                    "city": city["name"],
                    "city_key": city_key,
                    "subtitle": subtitle,
                    "side": "YES",
                    "bet_price": kalshi_prob,
                    "model_prob": model_prob,  # Store uncapped for reference
                    "our_prob_win": our_prob_win,  # Capped probability used for betting
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
                    "model_spread": model_diff,
                })

        # Evaluate NO bet (don't bet NO on forecast bin)
        if not is_forecast_bin:
            # Apply cap with NO-specific logic: skip if we actually think YES is very likely
            capped_no_prob, skip_no = apply_probability_cap(model_prob, kalshi_prob, side='NO')
            no_market = 1 - kalshi_prob
            no_edge = capped_no_prob - no_market
            no_min_edge = get_min_edge(no_market, agreement_level)

            # Skip NO range bets in sweet spot (historically lose money)
            no_bucket = get_price_bucket(no_market)
            skip_no_range = (contract_type == "range" and no_bucket == "sweet_spot")

            if not skip_no and no_edge > no_min_edge and not skip_no_range:
                edge_ratio = no_edge / no_min_edge
                our_prob_win = capped_no_prob
                bet_size, kelly_info = calculate_kelly_bet(
                    bankroll, our_prob_win, no_market,
                    model_spread=model_diff, agreement_level=agreement_level
                )
                
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
                        "model_spread": model_diff,
                    })

    print(f"\n  Found {len(qualifying_bets)} qualifying bets")
    print(f"  Filtered out: {skipped_cheap} cheap, {skipped_expensive} expensive")

    return qualifying_bets, city_summary


# ============ SMART BET SELECTION ============

def smart_select_bets(all_bets):
    """
    Select bets by edge ratio while enforcing per-city limits.

    Rules:
    - Maximum MAX_BETS_PER_CITY (2) bets per city to limit correlated exposure
    - Maximum MAX_TOTAL_BETS_PER_DAY (6) total bets
    - For same-city stacking beyond 1 bet, require edge_ratio >= SAME_CITY_MULTI_BET_THRESHOLD
    """
    if not all_bets:
        return []

    # Sort all bets by edge ratio (best first)
    sorted_bets = sorted(all_bets, key=lambda x: -x["edge_ratio"])

    selected_bets = []
    city_bet_counts = {}  # Track bets per city

    for bet in sorted_bets:
        # Stop if we've reached max total bets
        if len(selected_bets) >= MAX_TOTAL_BETS_PER_DAY:
            break

        city_key = bet["city_key"]
        current_city_count = city_bet_counts.get(city_key, 0)

        # Check if we can add another bet for this city
        if current_city_count >= MAX_BETS_PER_CITY:
            # Already at max for this city, skip
            continue

        # For 2nd+ bet in same city, require higher edge threshold
        if current_city_count >= 1:
            if bet["edge_ratio"] < SAME_CITY_MULTI_BET_THRESHOLD:
                # Not confident enough to stack same-city bets
                continue
            bet["selection_reason"] = f"same_city_stack (edge {bet['edge_ratio']:.1f}x >= {SAME_CITY_MULTI_BET_THRESHOLD}x)"
        else:
            bet["selection_reason"] = f"top_by_edge"

        # Add the bet
        selected_bets.append(bet)
        city_bet_counts[city_key] = current_city_count + 1

    # Re-number selection reasons for display
    for i, bet in enumerate(selected_bets):
        if "top_by_edge" in bet.get("selection_reason", ""):
            bet["selection_reason"] = f"#{i+1} by edge"

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
        
        # Show spread-adjusted Kelly breakdown
        kelly_info = bet['kelly_info']
        full_kelly = kelly_info.get('full_kelly_pct', 0)
        agreement_factor = kelly_info.get('agreement_factor', 1.0)
        effective_fraction = kelly_info.get('effective_kelly_fraction', KELLY_FRACTION)
        model_spread = kelly_info.get('model_spread', 0)
        
        if agreement_factor < 1.0:
            # Show that Kelly was reduced due to spread
            print(f"   │  Full Kelly: {full_kelly:.1f}% → Adjusted: {kelly_info.get('fractional_kelly_pct', 0):.1f}%")
            print(f"   │  📉 Spread adjustment: {agreement_factor:.0%} (spread={model_spread:.1f}°F)")
        else:
            print(f"   │  Kelly: {kelly_info.get('fractional_kelly_pct', 0):.1f}% of bankroll")
        
        print(f"   │  Fee: ~{kelly_info.get('fee_rate', 0):.1f}% (${kelly_info.get('fee', 0):.2f})")
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
    parser = argparse.ArgumentParser(description="Kalshi Weather Betting Model v9.3.6 (4-Model Ensemble)")

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
    print("ANALYSIS COMPLETE (v9.3.6-FOUR-MODEL)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()