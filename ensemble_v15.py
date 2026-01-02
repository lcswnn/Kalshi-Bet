"""
KALSHI WEATHER BETTING MODEL v15
================================
DEFENSIVE REBUILD based on actual performance data analysis.

KEY CHANGES FROM v9:
1. ASYMMETRIC EDGE REQUIREMENTS:
   - YES bets at low prices: 14% edge (favorable risk/reward)
   - NO bets at high prices: 22-28% edge (unfavorable risk/reward)
   
2. CITY-SPECIFIC UNCERTAINTY:
   - Miami/Austin: σ=2.2-2.5°F (stable weather, you're profitable here)
   - Chicago/NYC: σ=3.5-4.0°F (volatile weather, you were losing here)
   
3. POSITION-AWARE KELLY SIZING:
   - Full Kelly fraction for YES bets at low prices
   - Reduced Kelly (40-60%) for NO bets at high prices
   
4. IMPROVED FEE TRACKING:
   - Tracks estimated fees per bet
   - Shows fee impact in recommendations
   
5. CONSERVATIVE DEFAULTS:
   - Lower max bet fraction (12% vs 15%)
   - Higher minimum edge overall
   - Warns on volatile city bets

WHAT WE KEPT FROM v9:
- Clean ensemble weighting: 45% HRRR, 30% NWS, 25% Open-Meteo
- Smart bet selection across cities
- Price filtering (15¢-90¢)
- Agreement threshold system

PERFORMANCE DATA THAT DROVE THESE CHANGES:
- Your YES bets: +$3.59 (20.8% WR, but good risk/reward)
- Your NO bets: -$16.88 (62.9% WR, but bad risk/reward)
- Miami/Austin: +$48 combined
- Chicago/NYC: -$61 combined
- Every NO loss hit the EXACT bin you bet against (1 bin off)
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
import json
from pathlib import Path

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
MAX_CONTRACT_PRICE = 0.85   # Reduced from 90¢ - extreme prices are traps
SWEET_SPOT_LOW = 0.15       # Preferred range lower bound
SWEET_SPOT_HIGH = 0.45      # Reduced from 50¢ - be more conservative

# ============ SMART BET SELECTION CONFIGURATION ============
SAME_CITY_MULTI_BET_THRESHOLD = 2.0
MAX_BETS_PER_CITY = 2
MAX_TOTAL_BETS_PER_DAY = 5   # Reduced from 6 - quality over quantity
SUPER_CONFIDENT_EDGE_RATIO = 2.5

# ============ FEE CONFIGURATION ============
KALSHI_FEE_RATE = 0.035     # ~3.5% fee (based on your actual $28 fees on $790 wagered)

# ============ KELLY CRITERION CONFIGURATION ============
STARTING_BANKROLL = 70
KELLY_FRACTION = 0.40        # Reduced from 0.50 - more conservative base
MIN_BET_SIZE = 0.50
MAX_BET_FRACTION = 0.12      # Reduced from 0.15 - smaller max position

# ============ ENSEMBLE CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0
CONFIDENCE_BOOST_THRESHOLD = 2.0

# ============ NEW IN V15: CITY-SPECIFIC UNCERTAINTY ============
# Based on your actual performance data:
# - Miami: +$32.35 (62.5% WR) → model works well, lower uncertainty
# - Austin: +$15.61 (57.1% WR) → model works well, moderate uncertainty  
# - Chicago: -$18.80 (50% WR) → model overconfident, higher uncertainty
# - NYC: -$42.45 (40.7% WR) → model very overconfident, highest uncertainty

CITY_UNCERTAINTY = {
    "chicago": 3.8,    # Lake effect + fronts → high uncertainty (was 2.8)
    "nyc": 4.0,        # Coastal + urban effects → highest uncertainty (was 2.8)
    "miami": 2.2,      # Stable subtropical → low uncertainty
    "austin": 2.5,     # Continental but less volatile than north
    "la": 2.3,         # Mediterranean climate → stable
}

# Fallback for unknown cities
DEFAULT_UNCERTAINTY = 3.0

# ============ NEW IN V15: ASYMMETRIC EDGE REQUIREMENTS ============
# Your data showed:
# - YES bets at low prices: Good risk/reward (lose small, win big)
# - NO bets at high prices: Bad risk/reward (win small, lose big)
#
# Every single NO loss hit the EXACT bin you bet against.
# This means you need MORE edge on high-priced NOs to be profitable.

def get_min_edge(price, side, agreement_level='high', city_key=None):
    """
    Dynamic edge requirement based on price, position side, agreement, and city.
    
    NEW IN V15: Asymmetric requirements for YES vs NO.
    """
    # Base edge by position type and price
    if side == "YES":
        if price <= 0.35:
            base_edge = 0.14   # Low-priced YES: favorable risk/reward
        elif price <= 0.50:
            base_edge = 0.16   # Mid-priced YES: moderate
        else:
            base_edge = 0.18   # High-priced YES: less favorable
    else:  # NO
        if price <= 0.40:
            base_edge = 0.16   # Low-priced NO: okay risk/reward
        elif price <= 0.55:
            base_edge = 0.18   # Mid-priced NO: moderate
        elif price <= 0.65:
            base_edge = 0.22   # High-priced NO: unfavorable (was 0.16)
        elif price <= 0.75:
            base_edge = 0.25   # Very high-priced NO: very unfavorable
        else:
            base_edge = 0.30   # Extreme NO: basically don't bet
    
    # Adjust for agreement level
    if agreement_level == 'high':
        agreement_mult = 1.0
    elif agreement_level == 'medium':
        agreement_mult = 1.25
    else:  # low
        agreement_mult = 1.5
    
    # NEW: City risk adjustment for volatile cities
    city_mult = 1.0
    if city_key in ['chicago', 'nyc']:
        city_mult = 1.15  # 15% higher edge required for volatile cities
    
    return base_edge * agreement_mult * city_mult


def get_price_bucket(price):
    """Categorize price into buckets."""
    if price <= 0.35:
        return "low"
    elif price <= SWEET_SPOT_HIGH:
        return "sweet_spot"
    elif price <= 0.65:
        return "mid_high"
    else:
        return "high"


# ============ NEW IN V15: POSITION-AWARE KELLY ============

def calculate_kelly_bet(bankroll, our_prob, bet_price, side, city_key=None, kelly_fraction=None):
    """
    Calculate optimal bet size using Kelly Criterion with position-aware adjustments.
    
    NEW IN V15:
    - Reduces Kelly fraction for high-priced NO bets (bad risk/reward)
    - Increases slightly for low-priced YES bets (good risk/reward)
    - City-specific adjustments for volatile markets
    """
    if kelly_fraction is None:
        kelly_fraction = KELLY_FRACTION
        
    if bet_price <= 0 or bet_price >= 1:
        return 0, {}
    
    # Odds received before fees
    b_gross = (1 / bet_price) - 1
    b_net = b_gross - KALSHI_FEE_RATE
    
    p = our_prob
    q = 1 - p
    
    # Full Kelly formula with fee-adjusted odds
    full_kelly = (p * b_net - q) / b_net if b_net > 0 else 0
    
    # Apply base fractional Kelly
    fractional_kelly = full_kelly * kelly_fraction
    
    # ============ NEW: Position-aware adjustments ============
    position_mult = 1.0
    position_reason = "standard"
    
    if side == "YES":
        if bet_price <= 0.35:
            position_mult = 1.15  # Slightly larger on favorable YES bets
            position_reason = "favorable_yes"
    else:  # NO
        if bet_price >= 0.70:
            position_mult = 0.40  # Much smaller on high-priced NOs
            position_reason = "high_price_no_reduction"
        elif bet_price >= 0.60:
            position_mult = 0.60  # Smaller on mid-high NOs
            position_reason = "mid_high_no_reduction"
        elif bet_price >= 0.50:
            position_mult = 0.80  # Slightly smaller on mid NOs
            position_reason = "mid_no_reduction"
    
    # City volatility adjustment
    city_mult = 1.0
    if city_key in ['chicago', 'nyc']:
        city_mult = 0.75  # 25% smaller positions in volatile cities
        position_reason += "_volatile_city"
    
    fractional_kelly *= position_mult * city_mult
    
    # Clamp to bounds
    fractional_kelly = max(0, fractional_kelly)
    fractional_kelly = min(fractional_kelly, MAX_BET_FRACTION)
    
    # Calculate actual bet size
    bet_size = bankroll * fractional_kelly
    
    # Apply minimum bet size
    if bet_size < MIN_BET_SIZE:
        bet_size = 0
    
    # Calculate estimated fee for this bet
    estimated_fee = bet_size * KALSHI_FEE_RATE
    
    kelly_info = {
        "full_kelly_pct": full_kelly * 100,
        "fractional_kelly_pct": fractional_kelly * 100,
        "odds_gross": b_gross,
        "odds_net": b_net,
        "fee_rate": KALSHI_FEE_RATE,
        "estimated_fee": estimated_fee,
        "position_mult": position_mult,
        "city_mult": city_mult,
        "position_reason": position_reason,
    }
    
    return bet_size, kelly_info


# ============ CITY CONFIGURATION ============
cities = {
    "chicago": {
        "name": "Chicago",
        "csv_file": "weather_data_chicago.csv",
        "lat": 41.8781,
        "lon": -87.6298,
        "kalshi_series": "KXHIGHCHI",
        "timezone": "America/Chicago",
        "utc_offset": -6,
        "volatility": "high",  # NEW: volatility flag
    },
    "nyc": {
        "name": "New York City",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.7128,
        "lon": -74.0060,
        "kalshi_series": "KXHIGHNY",
        "timezone": "America/New_York",
        "utc_offset": -5,
        "volatility": "high",
    },
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.7959,
        "lon": -80.2870,
        "kalshi_series": "KXHIGHMIA",
        "timezone": "America/New_York",
        "utc_offset": -5,
        "volatility": "low",
    },
    "austin": {
        "name": "Austin",
        "csv_file": "weather_data_austin.csv",
        "lat": 30.2,
        "lon": -97.68,
        "kalshi_series": "KXHIGHAUS",
        "timezone": "America/Chicago",
        "utc_offset": -6,
        "volatility": "medium",
    },
    "la": {
        "name": "Los Angeles",
        "csv_file": "weather_data_la.csv",
        "lat": 33.94,
        "lon": -118.42,
        "kalshi_series": "KXHIGHLAX",
        "timezone": "America/Los_Angeles",
        "utc_offset": -8,
        "volatility": "low",
    }
}

# Default to all cities
selected_cities = ["chicago", "nyc", "miami", "austin", "la"]


# ============ HEADER ============
def print_header():
    print("=" * 70)
    print("KALSHI WEATHER BETTING MODEL v15 - DEFENSIVE REBUILD")
    print("(Asymmetric Edge + City Uncertainty + Position-Aware Kelly)")
    print("=" * 70)
    print(f"\nAnalyzing: {', '.join(cities[c]['name'] for c in selected_cities)}")
    print(f"\n🛡️ DEFENSIVE FEATURES:")
    print(f"   • City-specific uncertainty (Miami/Austin: ~2.3°F, Chi/NYC: ~3.9°F)")
    print(f"   • Asymmetric edge requirements (YES: 14-18%, NO: 16-30%)")
    print(f"   • Position-aware Kelly (reduced sizing on high-priced NOs)")
    print(f"   • Fee tracking: {KALSHI_FEE_RATE*100:.1f}% built into all calculations")
    print(f"\n💰 BANKROLL: ${STARTING_BANKROLL:.2f} | {KELLY_FRACTION:.0%} Base Kelly")
    print(f"📊 Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")


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


def fetch_hrrr_forecast_fixed(lat, lon, target_date, utc_offset):
    """Fetch HRRR forecast with proper sampling for daily high."""
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
                
        except Exception as e:
            continue
    
    if not temperatures:
        print("     ❌ No HRRR data retrieved")
        return None, None
    
    forecast_high = max(temperatures)
    
    print(f"     ✅ Retrieved {len(temperatures)} samples")
    print(f"     📊 Afternoon temps: {min(temperatures):.1f}°F to {max(temperatures):.1f}°F")
    
    weather_data = {
        'forecast_high': forecast_high,
        'all_temps': temperatures,
        'model_run': model_run_str,
        'source': 'HRRR'
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


# ============ PROBABILITY MODELS ============

def get_city_uncertainty(city_key):
    """Get the uncertainty (std dev) for a specific city."""
    return CITY_UNCERTAINTY.get(city_key, DEFAULT_UNCERTAINTY)


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
    """Analyze a single city and return all qualifying bets."""
    city = cities[city_key]
    qualifying_bets = []

    # Get city-specific uncertainty
    city_uncertainty = get_city_uncertainty(city_key)
    volatility = city.get("volatility", "medium")

    print(f"\n{'='*70}")
    print(f"ANALYZING: {city['name'].upper()}")
    if volatility == "high":
        print(f"⚠️ HIGH VOLATILITY CITY - Using σ={city_uncertainty}°F, stricter requirements")
    else:
        print(f"📊 Uncertainty: σ={city_uncertainty}°F")
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

    # 3. NWS forecast
    print(f"\n  [3] NWS (human-adjusted):")
    nws_forecast = fetch_nws_forecast(city["lat"], city["lon"], target_date)

    if nws_forecast:
        print(f"     ✅ Forecast: {nws_forecast:.1f}°F")
    else:
        print("     ⚠️ NWS unavailable")

    # ============ ENSEMBLE FORECAST ============
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

    # Calculate ensemble with fixed weights
    if len(forecasts) == 3:
        ensemble_forecast = (hrrr_forecast * 0.45) + (nws_forecast * 0.30) + (open_meteo_forecast * 0.25)
    elif len(forecasts) == 2:
        if hrrr_forecast and open_meteo_forecast:
            ensemble_forecast = (hrrr_forecast * 0.6) + (open_meteo_forecast * 0.4)
        elif hrrr_forecast and nws_forecast:
            ensemble_forecast = (hrrr_forecast * 0.55) + (nws_forecast * 0.45)
        else:
            ensemble_forecast = (nws_forecast * 0.55) + (open_meteo_forecast * 0.45)
    else:
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

    print(f"\n  🎯 ENSEMBLE FORECAST: {ensemble_forecast:.1f}°F (σ={city_uncertainty}°F)")

    forecast_bin = get_bin_for_temp(ensemble_forecast)
    print(f"     Primary bin: {forecast_bin[0]}°-{forecast_bin[1]}°F")

    # Store city summary
    city_summary = {
        "city": city["name"],
        "city_key": city_key,
        "ensemble_forecast": ensemble_forecast,
        "forecast_bin": forecast_bin,
        "agreement_level": agreement_level,
        "uncertainty": city_uncertainty,
        "volatility": volatility,
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
            return qualifying_bets, city_summary

        print(f"     ✅ Found {len(markets)} contracts for {target_str}")

    except Exception as e:
        print(f"     ❌ API error: {e}")
        return qualifying_bets, city_summary

    # ============ ANALYZE CONTRACTS ============
    print(f"\n  {'='*60}")
    print(f"  CONTRACT ANALYSIS (Price: {MIN_CONTRACT_PRICE*100:.0f}¢-{MAX_CONTRACT_PRICE*100:.0f}¢)")
    print(f"  {'='*60}")

    skipped_cheap = 0
    skipped_expensive = 0
    skipped_low_edge = 0

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

        # Price filters
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

        # ============ EVALUATE YES BET ============
        yes_edge = model_prob - kalshi_prob
        min_edge_yes = get_min_edge(kalshi_prob, "YES", agreement_level, city_key)
        
        if yes_edge > min_edge_yes:
            edge_ratio = yes_edge / min_edge_yes
            our_prob_win = model_prob
            bet_size, kelly_info = calculate_kelly_bet(
                bankroll, our_prob_win, kalshi_prob, "YES", city_key
            )
            
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
                    "min_edge": min_edge_yes,
                    "bet_size": bet_size,
                    "kelly_info": kelly_info,
                    "price_bucket": get_price_bucket(kalshi_prob),
                    "contract_type": contract_type,
                    "is_forecast_bin": is_forecast_bin,
                    "agreement_level": agreement_level,
                    "ensemble_forecast": ensemble_forecast,
                    "city_uncertainty": city_uncertainty,
                    "volatility": volatility,
                })
        else:
            if yes_edge > 0:
                skipped_low_edge += 1

        # ============ EVALUATE NO BET ============
        # Don't bet NO on forecast bin (that's betting against our prediction)
        if not is_forecast_bin:
            no_prob = 1 - model_prob
            no_market = 1 - kalshi_prob
            no_edge = no_prob - no_market
            min_edge_no = get_min_edge(no_market, "NO", agreement_level, city_key)
            
            if no_edge > min_edge_no:
                edge_ratio = no_edge / min_edge_no
                our_prob_win = no_prob
                bet_size, kelly_info = calculate_kelly_bet(
                    bankroll, our_prob_win, no_market, "NO", city_key
                )
                
                if bet_size >= MIN_BET_SIZE:
                    # Additional warning for high-priced NOs
                    risk_warning = None
                    if no_market >= 0.65:
                        risk_warning = "⚠️ HIGH-PRICE NO: Reduced position due to unfavorable risk/reward"
                    
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
                        "min_edge": min_edge_no,
                        "bet_size": bet_size,
                        "kelly_info": kelly_info,
                        "price_bucket": get_price_bucket(no_market),
                        "contract_type": contract_type,
                        "is_forecast_bin": False,
                        "agreement_level": agreement_level,
                        "ensemble_forecast": ensemble_forecast,
                        "city_uncertainty": city_uncertainty,
                        "volatility": volatility,
                        "risk_warning": risk_warning,
                    })
            else:
                if no_edge > 0:
                    skipped_low_edge += 1

    print(f"\n  Found {len(qualifying_bets)} qualifying bets")
    print(f"  Filtered: {skipped_cheap} cheap, {skipped_expensive} expensive, {skipped_low_edge} low edge")

    return qualifying_bets, city_summary


# ============ SMART BET SELECTION ============

def smart_select_bets(all_bets):
    """Select the best bets, prioritizing low-volatility cities and YES bets."""
    if not all_bets:
        return []

    # Score each bet (higher is better)
    for bet in all_bets:
        score = bet["edge_ratio"]
        
        # Bonus for low-volatility cities (Miami, Austin, LA)
        if bet.get("volatility") == "low":
            score *= 1.2
        elif bet.get("volatility") == "medium":
            score *= 1.1
        # No bonus for high volatility
        
        # Bonus for YES bets (better risk/reward historically)
        if bet["side"] == "YES":
            score *= 1.15
        
        # Penalty for high-priced NOs
        if bet["side"] == "NO" and bet["bet_price"] >= 0.65:
            score *= 0.8
        
        bet["selection_score"] = score

    # Sort by selection score
    sorted_bets = sorted(all_bets, key=lambda x: -x["selection_score"])

    # Take top bets
    selected_bets = sorted_bets[:MAX_TOTAL_BETS_PER_DAY]

    return selected_bets


def print_recommendations(selected_bets, all_bets, city_summaries, bankroll):
    """Print the final betting recommendations."""
    
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
    else:
        city_name = list(cities_with_bets)[0]
        print(f"\n⭐ Found {len(selected_bets)} bet(s) in {city_name}:")
    
    print()
    
    total_wager = 0
    total_potential = 0
    total_fees = 0

    for i, bet in enumerate(selected_bets, 1):
        # Confidence indicator
        if bet["edge_ratio"] >= SUPER_CONFIDENT_EDGE_RATIO:
            confidence = "🔥 SUPER CONFIDENT"
        elif bet["edge_ratio"] >= 2.0:
            confidence = "✓ High confidence"
        else:
            confidence = "⭐ Good opportunity"

        # Volatility warning
        vol_warning = ""
        if bet.get("volatility") == "high":
            vol_warning = " ⚠️ VOLATILE"

        bucket_emoji = "🎯" if bet["price_bucket"] in ["low", "sweet_spot"] else "📊"
        forecast_marker = "📍" if bet.get("is_forecast_bin") else ""

        odds = bet["kelly_info"].get("odds_net", 0)
        potential_profit = bet["bet_size"] * odds
        est_fee = bet["kelly_info"].get("estimated_fee", 0)

        print(f"   ┌─ BET #{i}: {bet['city']}{vol_warning}")
        print(f"   │  {confidence}")
        print(f"   │  Contract: {bet['subtitle']} {forecast_marker}")
        print(f"   │  Side: {bet['side']} at {bet['bet_price']*100:.0f}¢ {bucket_emoji}")
        print(f"   │  Your probability: {bet['our_prob_win']*100:.1f}%")
        print(f"   │  Edge: {bet['edge']*100:+.1f}% (need {bet['min_edge']*100:.0f}%, have {bet['edge_ratio']:.1f}x)")
        print(f"   │  Kelly: {bet['kelly_info'].get('fractional_kelly_pct', 0):.1f}% of bankroll")
        
        # Show position adjustment reason if any
        position_reason = bet["kelly_info"].get("position_reason", "")
        if "reduction" in position_reason:
            print(f"   │  🛡️ Position reduced: {position_reason}")
        
        print(f"   │")
        print(f"   │  💰 BET: ${bet['bet_size']:.2f} (est. fee: ${est_fee:.2f})")
        print(f"   │  📈 Potential profit: ${potential_profit:.2f}")
        
        # Risk warning for high-priced NOs
        if bet.get("risk_warning"):
            print(f"   │  {bet['risk_warning']}")
        
        print(f"   └{'─'*50}")
        print()

        total_wager += bet["bet_size"]
        total_potential += potential_profit
        total_fees += est_fee

    # Summary
    print(f"   {'='*55}")
    print(f"   TOTAL WAGER: ${total_wager:.2f} ({100*total_wager/bankroll:.1f}% of bankroll)")
    print(f"   ESTIMATED FEES: ${total_fees:.2f}")
    print(f"   POTENTIAL PROFIT: ${total_potential:.2f}")
    print(f"   {'='*55}")

    # City forecasts summary
    print(f"\n📊 FORECAST SUMMARY:")
    for summary in city_summaries:
        if summary:
            agreement_emoji = "✅" if summary["agreement_level"] == "high" else "⚠️" if summary["agreement_level"] == "medium" else "❌"
            vol_marker = "🔥" if summary.get("volatility") == "high" else ""
            print(f"   {summary['city']}: {summary['ensemble_forecast']:.1f}°F → Bin {summary['forecast_bin'][0]}-{summary['forecast_bin'][1]}° {agreement_emoji} σ={summary['uncertainty']}°F {vol_marker}")

    # Show what we didn't bet on
    not_selected = [b for b in all_bets if b not in selected_bets]
    if not_selected:
        print(f"\n📋 Also considered but not selected: {len(not_selected)} other opportunities")
        
        # Show why some weren't selected
        high_price_nos = [b for b in not_selected if b["side"] == "NO" and b["bet_price"] >= 0.65]
        if high_price_nos:
            print(f"   • {len(high_price_nos)} high-priced NO bets (reduced priority)")
        
        volatile_city = [b for b in not_selected if b.get("volatility") == "high"]
        if volatile_city:
            print(f"   • {len(volatile_city)} bets in volatile cities (reduced priority)")


# ============ ARGUMENT PARSING ============

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Kalshi Weather Betting Model v15")

    parser.add_argument("--kelly", type=float, default=KELLY_FRACTION,
                        help=f"Kelly fraction (default: {KELLY_FRACTION})")
    parser.add_argument("--bankroll", type=float, default=STARTING_BANKROLL,
                        help=f"Starting bankroll (default: ${STARTING_BANKROLL})")
    parser.add_argument("--cities", type=str, default="all",
                        help="Comma-separated city list or 'all' (default: all)")
    parser.add_argument("--conservative", action="store_true",
                        help="Use extra-conservative settings (skip Chi/NYC, lower Kelly)")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use less conservative settings (higher Kelly, lower edge req)")

    return parser.parse_args()


# ============ MAIN ============

def main():
    global KELLY_FRACTION, STARTING_BANKROLL, selected_cities

    args = parse_arguments()

    KELLY_FRACTION = args.kelly
    STARTING_BANKROLL = args.bankroll

    # Handle city filter
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
        selected_cities = valid_cities if valid_cities else list(cities.keys())

    # Conservative mode: skip volatile cities
    if args.conservative:
        print("🛡️ CONSERVATIVE MODE: Skipping Chicago and NYC")
        selected_cities = [c for c in selected_cities if c not in ['chicago', 'nyc']]
        KELLY_FRACTION = 0.30
    
    # Aggressive mode
    if args.aggressive:
        print("⚡ AGGRESSIVE MODE: Higher Kelly, standard edge requirements")
        KELLY_FRACTION = 0.60

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
    print("ANALYSIS COMPLETE (v15 - Defensive Rebuild)")
    print(f"{'='*70}")
    print("\n💡 TIPS:")
    print("   --conservative    Skip Chi/NYC, lower Kelly (safest)")
    print("   --aggressive      Higher Kelly (more risk)")
    print("   --cities miami,austin   Only analyze specific cities")


if __name__ == "__main__":
    main()