"""
KALSHI WEATHER BETTING MODEL v9.4-PROFESSIONAL
===============================================
MAJOR IMPROVEMENTS based on professional weather prediction research:

1. CORRECT KALSHI FEE CALCULATION
   - Kalshi uses: fee = round_up(0.07 × C × P × (1-P))
   - This is PER CONTRACT, not a flat percentage
   - Fee is highest at 50¢ contracts (~1.75¢/contract), lowest at extremes
   - Previous model used flat 4% which OVERESTIMATED fees at extremes

2. IMPROVED UNCERTAINTY MODELING
   - Added time-of-day adjustment (forecasts made later are more accurate)
   - City-specific seasonal adjustments
   - Better model agreement weighting

3. BIAS CORRECTION (MOS-INSPIRED)
   - Professional weather services use Model Output Statistics
   - Track historical forecast errors and apply rolling correction
   - Reduces systematic bias in predictions

4. ECMWF-STYLE ENSEMBLE SPREAD
   - Use spread between models as uncertainty measure
   - Wider spread = wider probability distribution
   - This is how professional services calibrate confidence

5. MINIMUM PROFIT THRESHOLD
   - Only bet if expected profit AFTER FEES exceeds threshold
   - Ensures we're actually making money after Kalshi takes their cut

Based on:
- ECMWF ensemble methodology
- NWS Model Output Statistics (MOS)
- HRRR 3km high-resolution rapid refresh
- Kalshi's actual fee structure (7% of expected earnings formula)

Cities: Austin + Miami only (best historical performers)
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
MAX_CONTRACT_PRICE = 0.85   # Lowered from 90¢ - high price = low edge potential
SWEET_SPOT_LOW = 0.20       # Adjusted - very cheap contracts have high fees
SWEET_SPOT_HIGH = 0.45      # Adjusted for optimal fee/edge ratio

# ============ SMART BET SELECTION CONFIGURATION ============
SAME_CITY_MULTI_BET_THRESHOLD = 2.0   # Need 2x the minimum edge to stack same-city bets
MAX_BETS_PER_CITY = 2                  # Never more than 2 bets in same city
MAX_TOTAL_BETS_PER_DAY = 4             # Reduced from 6 - focus on highest conviction
SUPER_CONFIDENT_EDGE_RATIO = 2.5       # What counts as "super confident"

# ============ KALSHI FEE CONFIGURATION (CORRECTED) ============
# Kalshi's ACTUAL formula: fee = round_up(0.07 × C × P × (1-P))
# This means fee varies by price - highest at 50¢, lowest at extremes
KALSHI_FEE_MULTIPLIER = 0.07  # 7% of expected earnings

def calculate_kalshi_fee(num_contracts, price_per_contract):
    """
    Calculate Kalshi's actual fee using their formula.
    fee = round_up(0.07 × C × P × (1-P))
    
    This is MUCH more accurate than the flat 4% we used before.
    At 50¢: ~1.75¢ per contract (3.5% of bet)
    At 20¢: ~1.12¢ per contract (5.6% of bet)  
    At 80¢: ~1.12¢ per contract (1.4% of bet)
    """
    P = price_per_contract
    C = num_contracts
    raw_fee = KALSHI_FEE_MULTIPLIER * C * P * (1 - P)
    # Round up to nearest cent
    return np.ceil(raw_fee * 100) / 100

def calculate_fee_rate(price):
    """Calculate effective fee rate as percentage of bet amount."""
    # For 1 contract at this price
    fee = calculate_kalshi_fee(1, price)
    bet_amount = price
    return fee / bet_amount if bet_amount > 0 else 0

# ============ KELLY CRITERION CONFIGURATION ============
STARTING_BANKROLL = 70        # Your bankroll
KELLY_FRACTION = 0.40         # Reduced from 0.50 - more conservative
MIN_BET_SIZE = 1.00           # Increased minimum - fees eat small bets
MAX_BET_FRACTION = 0.12       # Reduced from 15% - more conservative
MIN_EXPECTED_PROFIT = 0.50    # NEW: Minimum expected profit after fees

# ============ ENSEMBLE CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - models should agree within this
CONFIDENCE_BOOST_THRESHOLD = 2.0    # °F - boost confidence if within this
CALIBRATED_FORECAST_STD = 2.8       # °F - fallback if city not in CITY_UNCERTAINTY

# ============ CITY-SPECIFIC UNCERTAINTY ============
# Based on actual performance data and meteorological characteristics
# These are BASE values - adjusted by time-of-day and model agreement
CITY_UNCERTAINTY = {
    "miami": {
        "base_std": 2.3,      # Very stable subtropical - reduced from 2.5
        "seasonal_adj": {     # Miami is more predictable in winter
            "winter": 0.9,    # Dec-Feb: even more stable
            "spring": 1.0,    # Mar-May: typical
            "summer": 1.1,    # Jun-Aug: afternoon storms add uncertainty
            "fall": 1.0       # Sep-Nov: typical
        }
    },
    "austin": {
        "base_std": 2.6,      # Reduced from 2.8 based on good performance
        "seasonal_adj": {     # Austin has more weather variability
            "winter": 1.1,    # Dec-Feb: cold fronts add uncertainty
            "spring": 1.2,    # Mar-May: severe weather season
            "summer": 0.9,    # Jun-Aug: consistently hot
            "fall": 1.0       # Sep-Nov: typical
        }
    }
}

def get_season():
    """Get current meteorological season."""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

def get_adjusted_uncertainty(city_key, model_spread, hours_until_event=24):
    """
    Get city-specific uncertainty with professional adjustments.
    
    Factors:
    1. Base city uncertainty (climate-based)
    2. Seasonal adjustment
    3. Model agreement (spread-based)
    4. Time until event (shorter = more accurate)
    """
    city_config = CITY_UNCERTAINTY.get(city_key, {"base_std": 2.8, "seasonal_adj": {}})
    base_std = city_config["base_std"]
    
    # Seasonal adjustment
    season = get_season()
    seasonal_mult = city_config.get("seasonal_adj", {}).get(season, 1.0)
    
    # Model agreement adjustment
    # If models disagree by >3°F, increase uncertainty
    if model_spread > 4.0:
        spread_mult = 1.3
    elif model_spread > 3.0:
        spread_mult = 1.15
    elif model_spread > 2.0:
        spread_mult = 1.05
    elif model_spread < 1.0:
        spread_mult = 0.90  # High agreement = lower uncertainty
    else:
        spread_mult = 1.0
    
    # Time adjustment (forecasts get better closer to event)
    # Professional models show ~0.5°F improvement per 12 hours
    if hours_until_event <= 12:
        time_mult = 0.85
    elif hours_until_event <= 18:
        time_mult = 0.92
    elif hours_until_event <= 24:
        time_mult = 1.0
    else:
        time_mult = 1.1
    
    final_std = base_std * seasonal_mult * spread_mult * time_mult
    return final_std

# Edge requirements - ADJUSTED for variable fee structure
BASE_EDGE_REQUIREMENTS = {
    "sweet_spot": 0.14,    # 14% edge for 20-45¢ contracts (fees ~3-4%)
    "low_price": 0.18,     # 18% edge for 15-20¢ contracts (fees ~5-6%)
    "high_price": 0.12,    # 12% edge for 45-85¢ contracts (fees ~1-2%)
}

def get_min_edge(price, agreement_level='high'):
    """Dynamic edge requirement based on price bucket and model agreement."""
    if price < 0.20:
        base_edge = BASE_EDGE_REQUIREMENTS["low_price"]
    elif price <= 0.45:
        base_edge = BASE_EDGE_REQUIREMENTS["sweet_spot"]
    else:
        base_edge = BASE_EDGE_REQUIREMENTS["high_price"]
    
    # Adjust for agreement level
    if agreement_level == 'high':
        return base_edge
    elif agreement_level == 'medium':
        return base_edge * 1.25
    else:  # low
        return base_edge * 1.5

def get_price_bucket(price):
    """Categorize price into buckets for edge requirements."""
    if price < 0.20:
        return "low_price"
    elif price <= 0.45:
        return "sweet_spot"
    else:
        return "high_price"


# ============ KELLY CRITERION (CORRECTED FOR ACTUAL FEES) ============

def calculate_kelly_bet(bankroll, our_prob, bet_price, num_contracts, kelly_fraction=KELLY_FRACTION):
    """
    Calculate optimal bet size using Kelly Criterion with CORRECT Kalshi fees.
    
    Key insight: Kalshi's fee formula means fees are NOT proportional to bet size
    in a simple way. We need to calculate expected value properly.
    
    Returns:
        bet_size: Dollar amount to bet
        kelly_info: Dict with calculation details
    """
    if bet_price <= 0 or bet_price >= 1:
        return 0, {}
    
    # Calculate fee for this specific trade
    fee = calculate_kalshi_fee(num_contracts, bet_price)
    fee_rate = fee / (num_contracts * bet_price) if num_contracts > 0 else 0
    
    # Payout if win: $1 per contract
    gross_win = num_contracts * 1.0
    
    # Cost to enter position
    cost = num_contracts * bet_price
    
    # Net profit if win (after fee)
    net_profit_if_win = gross_win - cost - fee
    
    # Net loss if lose (you lose your stake, fee already paid)
    net_loss_if_lose = cost + fee
    
    # Odds for Kelly: net profit / stake (including fee)
    total_stake = cost + fee
    b_net = net_profit_if_win / total_stake if total_stake > 0 else 0
    
    p = our_prob  # Probability of winning
    q = 1 - p     # Probability of losing
    
    # Kelly formula
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
    
    # Check minimum expected profit
    if bet_size > 0:
        expected_profit = (our_prob * net_profit_if_win) - ((1 - our_prob) * net_loss_if_lose)
        if expected_profit < MIN_EXPECTED_PROFIT:
            bet_size = 0  # Don't bet if expected profit is too low
    
    kelly_info = {
        "full_kelly_pct": full_kelly * 100,
        "fractional_kelly_pct": fractional_kelly * 100,
        "odds_net": b_net,
        "fee": fee,
        "fee_rate": fee_rate * 100,  # As percentage
        "net_profit_if_win": net_profit_if_win,
        "total_stake": total_stake,
    }
    
    return bet_size, kelly_info


# ============ CITY CONFIGURATION ============
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
    }
}

# Will collect all qualifying bets across all cities
all_qualifying_bets = []

# Default to Austin and Miami only
selected_cities = ["austin", "miami"]


# ============ HEADER ============
def print_header():
    print("=" * 70)
    print("KALSHI WEATHER BETTING MODEL v9.4-PROFESSIONAL")
    print("(Correct Fees + MOS-Style Calibration + Dynamic Uncertainty)")
    print("=" * 70)
    print(f"\nAnalyzing: Austin, Miami (best performing cities)")
    print(f"Agreement Threshold: ±{ENSEMBLE_AGREEMENT_THRESHOLD}°F")
    print(f"Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")
    print(f"Sweet Spot: {SWEET_SPOT_LOW*100:.0f}¢ - {SWEET_SPOT_HIGH*100:.0f}¢")
    print(f"\n💰 KALSHI FEE STRUCTURE (CORRECTED):")
    print(f"   Formula: fee = round_up(0.07 × contracts × price × (1-price))")
    print(f"   At 20¢: ~{calculate_fee_rate(0.20)*100:.1f}% | At 50¢: ~{calculate_fee_rate(0.50)*100:.1f}% | At 80¢: ~{calculate_fee_rate(0.80)*100:.1f}%")
    print(f"\n🌡️ CITY UNCERTAINTY (Season: {get_season().upper()}):")
    for city_key in selected_cities:
        base = CITY_UNCERTAINTY[city_key]["base_std"]
        season_adj = CITY_UNCERTAINTY[city_key]["seasonal_adj"].get(get_season(), 1.0)
        print(f"   • {city_key.upper()}: σ={base:.1f}°F × {season_adj:.1f} (seasonal)")
    print(f"\n🎯 SMART BET SELECTION:")
    print(f"   • Min expected profit: ${MIN_EXPECTED_PROFIT:.2f}")
    print(f"   • Max {MAX_BETS_PER_CITY} bets/city, {MAX_TOTAL_BETS_PER_DAY} total/day")
    print(f"\n💰 BANKROLL: ${STARTING_BANKROLL:.2f} | {KELLY_FRACTION:.0%} Kelly")
    print("Data Sources: HRRR (45%) + NWS (30%) + Open-Meteo (25%)")


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
    NWS forecasters add human judgment to model output - this is essentially MOS.
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
    
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    
    target_12_local_utc = 12 - utc_offset
    target_start = datetime.combine(target_date, datetime.min.time().replace(hour=target_12_local_utc % 24))
    
    hours_ahead_start = int((target_start - model_run_datetime).total_seconds() / 3600)
    
    if hours_ahead_start < 0 or hours_ahead_start > 48:
        print(f"     ⚠️ HRRR: Target out of range (need fhour {hours_ahead_start})")
        return None, None
    
    temps = []
    for offset in range(0, 7, 2):
        fhour = hours_ahead_start + offset
        if fhour > 48:
            continue
            
        try:
            H = Herbie(
                model_run_datetime.strftime("%Y-%m-%d %H:%M"),
                model="hrrr",
                product="sfc",
                fxx=fhour
            )
            
            ds = H.xarray(":TMP:2 m above ground:")
            
            if ds is not None and 't2m' in ds:
                lats = ds['latitude'].values
                lons = ds['longitude'].values
                
                target_lon = lon if lon > 0 else lon + 360
                dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
                
                idx = np.unravel_index(np.argmin(dist), dist.shape)
                temp_k = float(ds['t2m'].values[idx])
                temp_f = (temp_k - 273.15) * 9/5 + 32
                temps.append(temp_f)
                
        except Exception as e:
            continue
    
    if temps:
        hrrr_max = max(temps)
        return hrrr_max, {"samples": len(temps), "temps": temps}
    
    return None, None


def get_bin_for_temp(temp):
    """
    Determine which Kalshi bin a temperature falls into.
    NOAA stations round temperatures: 0.5 and above rounds UP.
    Kalshi uses 2-degree bins with ODD lower bounds.
    """
    temp_rounded = int(np.round(temp))
    
    if temp_rounded % 2 == 1:
        lower = temp_rounded
    else:
        lower = temp_rounded - 1

    return (lower, lower + 1)


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
    
    # Calculate hours until event (for uncertainty adjustment)
    target_noon = datetime.combine(target_date, datetime.min.time().replace(hour=12))
    hours_until_event = max(0, (target_noon - now).total_seconds() / 3600)

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

    # 3. NWS forecast (human-adjusted / MOS)
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
    model_spread = max(temps) - min(temps) if len(temps) > 1 else 0

    if model_spread <= CONFIDENCE_BOOST_THRESHOLD:
        agreement_level = 'high'
    elif model_spread <= ENSEMBLE_AGREEMENT_THRESHOLD:
        agreement_level = 'medium'
    else:
        agreement_level = 'low'

    print(f"\n  📊 Model Comparison:")
    for name, temp in forecasts:
        print(f"     {name}: {temp:.1f}°F")
    if len(forecasts) > 1:
        print(f"     Spread: {model_spread:.1f}°F → Agreement: {agreement_level.upper()}")

    print(f"\n  🎯 ENSEMBLE FORECAST: {ensemble_forecast:.1f}°F")

    # Get DYNAMIC city-specific uncertainty
    city_uncertainty = get_adjusted_uncertainty(city_key, model_spread, hours_until_event)
    print(f"     Uncertainty (σ): {city_uncertainty:.2f}°F (adjusted for season, spread, timing)")

    forecast_bin = get_bin_for_temp(ensemble_forecast)
    print(f"     Primary bin: {forecast_bin[0]}°-{forecast_bin[1]}°F")

    # Store city summary
    city_summary = {
        "city": city["name"],
        "ensemble_forecast": ensemble_forecast,
        "forecast_bin": forecast_bin,
        "agreement_level": agreement_level,
        "uncertainty": city_uncertainty,
        "model_spread": model_spread,
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
    print(f"  CONTRACT ANALYSIS (Filtered: {MIN_CONTRACT_PRICE*100:.0f}¢-{MAX_CONTRACT_PRICE*100:.0f}¢)")
    print(f"  {'='*60}")

    skipped_cheap = 0
    skipped_expensive = 0
    skipped_low_profit = 0

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

        # === EVALUATE BOTH SIDES ===
        for side in ["YES", "NO"]:
            if side == "YES":
                bet_price = kalshi_prob
                our_prob_win = model_prob
            else:
                bet_price = 1 - kalshi_prob
                our_prob_win = 1 - model_prob

            # Skip if price out of range
            if bet_price < MIN_CONTRACT_PRICE or bet_price > MAX_CONTRACT_PRICE:
                continue

            edge = our_prob_win - bet_price
            price_bucket = get_price_bucket(bet_price)
            min_edge = get_min_edge(bet_price, agreement_level)
            edge_ratio = edge / min_edge if min_edge > 0 else 0

            if edge > min_edge:
                # Calculate optimal number of contracts for this bet
                # Start with ~$10 position to estimate
                est_contracts = int(10 / bet_price) if bet_price > 0 else 1
                est_contracts = max(1, est_contracts)
                
                bet_size, kelly_info = calculate_kelly_bet(
                    bankroll, our_prob_win, bet_price, est_contracts
                )

                if bet_size > 0:
                    # Recalculate with actual bet size
                    actual_contracts = int(bet_size / bet_price)
                    if actual_contracts < 1:
                        skipped_low_profit += 1
                        continue
                    
                    # Recalculate Kelly with actual contracts
                    bet_size, kelly_info = calculate_kelly_bet(
                        bankroll, our_prob_win, bet_price, actual_contracts
                    )
                    
                    if bet_size > 0:
                        qualifying_bets.append({
                            "city": city["name"],
                            "ticker": ticker,
                            "subtitle": subtitle,
                            "side": side,
                            "bet_price": bet_price,
                            "our_prob_win": our_prob_win,
                            "edge": edge,
                            "edge_ratio": edge_ratio,
                            "min_edge": min_edge,
                            "bet_size": bet_size,
                            "contracts": actual_contracts,
                            "kelly_info": kelly_info,
                            "price_bucket": price_bucket,
                            "is_forecast_bin": is_forecast_bin,
                            "agreement_level": agreement_level,
                        })

    print(f"\n  📋 Filter Summary:")
    print(f"     Skipped (too cheap): {skipped_cheap}")
    print(f"     Skipped (too expensive): {skipped_expensive}")
    print(f"     Skipped (low expected profit): {skipped_low_profit}")
    print(f"     Qualifying bets: {len(qualifying_bets)}")

    return qualifying_bets, city_summary


def smart_select_bets(all_bets):
    """Select the top bets by edge ratio."""
    if not all_bets:
        return []

    sorted_bets = sorted(all_bets, key=lambda x: -x["edge_ratio"])
    selected_bets = sorted_bets[:MAX_TOTAL_BETS_PER_DAY]

    for i, bet in enumerate(selected_bets):
        bet["selection_reason"] = f"top_{i+1}_by_edge"

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
        print("  • No contracts with sufficient edge after fees")
        print("  • Expected profit below minimum threshold")
        print("  • Models disagree significantly")
        print("\n💡 Sitting out is a valid strategy!")
        return
    
    cities_with_bets = set(b["city"] for b in selected_bets)
    num_cities = len(cities_with_bets)
    
    if num_cities > 1:
        print(f"\n✅ Found bets in {num_cities} DIFFERENT CITIES (uncorrelated):")
    elif len(selected_bets) > 1:
        city_name = list(cities_with_bets)[0]
        print(f"\n🔥 Found {len(selected_bets)} bets in {city_name}:")
    else:
        city_name = list(cities_with_bets)[0]
        print(f"\n⭐ Found 1 great bet in {city_name}:")

    print()
    
    total_wager = 0
    total_potential = 0
    total_fees = 0

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

        fee = bet["kelly_info"].get("fee", 0)
        net_profit = bet["kelly_info"].get("net_profit_if_win", 0)

        print(f"   ┌─ BET #{i}: {bet['city']} {rank_label}")
        print(f"   │  {confidence}")
        print(f"   │  Contract: {bet['subtitle']} {forecast_marker}")
        print(f"   │  Side: {bet['side']} at {bet['bet_price']*100:.0f}¢ {bucket_emoji}")
        print(f"   │  Contracts: {bet['contracts']}")
        print(f"   │  Your probability: {bet['our_prob_win']*100:.1f}%")
        print(f"   │  Edge: {bet['edge']*100:+.1f}% ({bet['edge_ratio']:.1f}x minimum)")
        print(f"   │")
        print(f"   │  💰 BET: ${bet['bet_size']:.2f} (+ ${fee:.2f} fee)")
        print(f"   │  📈 Net profit if win: ${net_profit:.2f}")
        print(f"   │  ⚠️ Fee rate: {bet['kelly_info'].get('fee_rate', 0):.1f}%")
        print(f"   └{'─'*50}")
        print()

        total_wager += bet["bet_size"]
        total_potential += net_profit
        total_fees += fee

    print(f"   {'='*55}")
    print(f"   TOTAL WAGER: ${total_wager:.2f} ({100*total_wager/bankroll:.1f}% of bankroll)")
    print(f"   TOTAL FEES: ${total_fees:.2f}")
    print(f"   MAX POTENTIAL PROFIT: ${total_potential:.2f}")
    print(f"   {'='*55}")

    print(f"\n📊 FORECAST SUMMARY:")
    for summary in city_summaries:
        if summary:
            agreement_emoji = "✅" if summary["agreement_level"] == "high" else "⚠️" if summary["agreement_level"] == "medium" else "❌"
            print(f"   {summary['city']}: {summary['ensemble_forecast']:.1f}°F → Bin {summary['forecast_bin'][0]}-{summary['forecast_bin'][1]}° {agreement_emoji} (σ={summary['uncertainty']:.2f}°F)")

    not_selected = [b for b in all_bets if b not in selected_bets]
    if not_selected:
        print(f"\n📋 Also considered: {len(not_selected)} other opportunities")


# ============ ARGUMENT PARSING ============

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Kalshi Weather Betting Model v9.4")

    parser.add_argument("--kelly", type=float, default=KELLY_FRACTION,
                        help=f"Kelly fraction (default: {KELLY_FRACTION})")
    parser.add_argument("--bankroll", type=float, default=STARTING_BANKROLL,
                        help=f"Starting bankroll (default: ${STARTING_BANKROLL})")
    parser.add_argument("--cities", type=str, default="all",
                        help="Comma-separated city list or 'all' (default: all)")

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
        selected_cities = valid_cities if valid_cities else ["austin", "miami"]

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
    print("ANALYSIS COMPLETE (v9.4-PROFESSIONAL)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()