"""
KALSHI WEATHER BETTING MODEL v9.3.6 (FULL PRODUCTION VERSION)
=============================================================
MERGE: v9.3.5 Infrastructure + v9.3.6 Mathematical Improvements

v9.3.6 CHANGES:
- NEW: Station Bias Correction (Grid-to-Station offset)
- NEW: Skewed Normal Distribution (handles hard temp ceilings/floors)
- NEW: Configurable Model Weights (no more hardcoded magic numbers)
- NEW: "Too Good To Be True" Guardrails (flags suspicious edges >25%)
- RESTORED: Full NWS, HRRR, and Kalshi API connectivity from v9.3.5

CRITICAL: YOU MUST CALIBRATE 'STATION_BIAS' FOR THIS TO WORK OPTIMALLY.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
from scipy import stats
from scipy.stats import skewnorm  # NEW: For non-normal distributions
import argparse

# ============ HRRR/HERBIE IMPORTS ============
try:
    from herbie import Herbie # type: ignore
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False
    print("⚠️ Herbie not installed. Will use Open-Meteo only.")
    print("   Install with: pip install herbie-data xarray cfgrib")

# ============ 1. MODEL WEIGHTS CONFIGURATION ============
# Adjust these based on your backtesting results.
ENSEMBLE_WEIGHTS = {
    "NWS": 0.35,        # Human-adjusted, usually stable
    "HRRR": 0.25,       # High-res, good for local phenomena
    "Open-Meteo": 0.20, # Blended global
    "GFS": 0.20         # Global macro trends
}

# ============ 2. STATION BIAS CONFIGURATION (THE #1 FIX) ============
# Grid models predict the average temp of a 3km square.
# Kalshi settles on specific sensors. 
# POSITIVE value = Station is WARMER than model (e.g., Heat Island).
# NEGATIVE value = Station is COOLER than model (e.g., near water/park).
STATION_BIAS = {
    "miami": 0.0,       # TODO: Monitor and adjust (e.g., +1.2 if MIA consistently hotter)
    "austin": 0.0,      # TODO: Monitor and adjust
    "chicago": 1.5,     # Example: Midway is often warmer than the grid
    "nyc": -0.5,        # Example: Central Park can be cooler than concrete jungle
}

# ============ 3. SKEWED DISTRIBUTION PARAMS ============
# Skew = 0: Normal Distribution (Bell Curve)
# Skew < 0: Long tail to the left (Hard ceiling on heat, e.g., coastal sea breeze)
# Skew > 0: Long tail to the right (Hard floor on cold)
CITY_SKEW_PARAMS = {
    "miami": -2.0,      # Hard ceiling due to ocean (neg skew)
    "austin": 0.0,      # Inland, fairly normal
    "chicago": -1.0,    # Lake effect can create ceilings
    "nyc": -1.0,
}

# ============ PRICE & EDGE CONFIGURATION ============
MIN_CONTRACT_PRICE = 0.15   # Never bet on contracts below 15¢
MAX_CONTRACT_PRICE = 0.90   # Never bet on contracts above 90¢
SWEET_SPOT_LOW = 0.15       # Preferred range lower bound
SWEET_SPOT_HIGH = 0.50      # Preferred range upper bound

# EDGE GUARDRAILS
MIN_EDGE_REQUIREMENT = 0.12    # Lowered to 8% (10% was too aggressive/rare)
MAX_EDGE_WARNING = 0.25        # Warn if edge > 25% (likely data error or station bias)
SUSPICIOUS_PROB_CAP = 0.92     # Cap internal confidence even lower for safety
MAX_PROBABILITY = 0.85         # General cap

# ============ BETTING CONFIGURATION ============
SAME_CITY_MULTI_BET_THRESHOLD = 2.0   # Need 2x the minimum edge to stack same-city bets
MAX_BETS_PER_CITY = 2                  # Never more than 2 bets in same city
MAX_TOTAL_BETS_PER_DAY = 6             # Cap total daily bets across all cities
SUPER_CONFIDENT_EDGE_RATIO = 2.5       # What counts as "super confident"

# Fee Config
KALSHI_FEE_MULTIPLIER = 0.07  # 7% of expected earnings

# Kelly Config
STARTING_BANKROLL = 70        # Your bankroll
KELLY_FRACTION = 0.50         # Half Kelly (balanced risk/reward)
MIN_BET_SIZE = 0.50           # Don't bet less than 50¢
MAX_BET_FRACTION = 0.20       # Safety cap: 20% of bankroll max per bet

# Spread Adjustment Constants
SPREAD_KELLY_THRESHOLD = 2.0       # Below this spread, full Kelly
SPREAD_KELLY_PENALTY = 0.15        # Kelly reduction per °F of spread above threshold
SPREAD_KELLY_FLOOR = 0.50          # Never reduce Kelly multiplier below this
SPREAD_UNCERTAINTY_DIVISOR = 1.5   # Divisor for converting spread to sigma

# Base Uncertainty (Calibrated)
CITY_BASE_UNCERTAINTY = {
    "miami": 4.5,
    "austin": 5.0,
    "chicago": 5.8,
    "nyc": 5.4,
}
MIN_UNCERTAINTY = 4.0

# ============ CITY CONFIG ============
cities = {
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv", # Keep for historical ref if needed
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

# Selected cities default
selected_cities = ["austin", "miami"]

# ============ HEADER ============
def print_header():
    print("=" * 70)
    print("KALSHI WEATHER BETTING MODEL v9.3.6")
    print("(Station Bias + Skewed Dist + Configurable Weights)")
    print("=" * 70)
    city_names = [cities[c]["name"] for c in selected_cities]
    print(f"\nAnalyzing: {', '.join(city_names)}")
    print(f"Weights: {ENSEMBLE_WEIGHTS}")
    print(f"Station Bias: {STATION_BIAS}")
    print(f"Skew Params: {CITY_SKEW_PARAMS}")

# ============ CALCULATIONS ============

def calculate_kalshi_fee(num_contracts, price_per_contract):
    """fee = round_up(0.07 * C * P * (1-P))"""
    P = price_per_contract
    C = num_contracts
    raw_fee = KALSHI_FEE_MULTIPLIER * C * P * (1 - P)
    return np.ceil(raw_fee * 100) / 100

def get_min_edge(price, agreement_level='high'):
    return MIN_EDGE_REQUIREMENT

def get_price_bucket(price):
    return "sweet_spot" if price <= SWEET_SPOT_HIGH else "high_price"

def calculate_spread_agreement_factor(model_spread):
    if model_spread <= SPREAD_KELLY_THRESHOLD:
        return 1.0
    excess_spread = model_spread - SPREAD_KELLY_THRESHOLD
    factor = 1.0 - (excess_spread * SPREAD_KELLY_PENALTY)
    return max(SPREAD_KELLY_FLOOR, factor)

def calculate_kelly_bet(bankroll, our_prob, bet_price, kelly_fraction=KELLY_FRACTION, 
                        model_spread=0.0, agreement_level='high'):
    if bet_price <= 0 or bet_price >= 1: return 0, {}
    
    est_contracts = max(1, int(10 / bet_price))
    fee = calculate_kalshi_fee(est_contracts, bet_price)
    position_cost = est_contracts * bet_price
    total_cost = position_cost + fee
    
    net_profit_if_win = (est_contracts * 1.0) - total_cost
    b_net = net_profit_if_win / total_cost if total_cost > 0 else 0
    
    # Also calculate fee as percentage for display
    fee_rate = (fee / position_cost * 100) if position_cost > 0 else 0

    p = our_prob
    q = 1 - p
    
    # Kelly Formula
    full_kelly = (p * b_net - q) / b_net if b_net > 0 else 0
    
    # Spread Adjustment
    agreement_factor = calculate_spread_agreement_factor(model_spread)
    adjusted_kelly_fraction = kelly_fraction * agreement_factor
    fractional_kelly = full_kelly * adjusted_kelly_fraction
    
    fractional_kelly = max(0, min(fractional_kelly, MAX_BET_FRACTION))
    bet_size = bankroll * fractional_kelly
    
    if bet_size < MIN_BET_SIZE: bet_size = 0
    
    return bet_size, {
        "full_kelly_pct": full_kelly * 100,
        "fractional_kelly_pct": fractional_kelly * 100,
        "agreement_factor": agreement_factor,
        "effective_kelly_fraction": adjusted_kelly_fraction,
        "model_spread": model_spread,
        "odds_net": b_net,
        "fee": fee,
        "fee_rate": fee_rate
    }

# ============ PROBABILITY MODELS (UPDATED FOR SKEW) ============

def get_skew_parameter(city_key):
    """Return the skew parameter (alpha) for the city."""
    return CITY_SKEW_PARAMS.get(city_key, 0.0)

def calibrated_probability(contract_low, contract_high, forecast, uncertainty_std, city_key, agreement_level='high'):
    """
    v9.3.6: Uses Skewed Normal Distribution.
    """
    skew_param = get_skew_parameter(city_key)
    
    # CDF of high bound - CDF of low bound
    # Note: skewnorm.cdf(x, a, loc, scale)
    prob = skewnorm.cdf(contract_high + 0.5, skew_param, loc=forecast, scale=uncertainty_std) - \
           skewnorm.cdf(contract_low - 0.5, skew_param, loc=forecast, scale=uncertainty_std)
    
    return max(min(prob, 0.99), 0.01)

def calibrated_below_probability(threshold, forecast, uncertainty_std, city_key, agreement_level='high'):
    skew_param = get_skew_parameter(city_key)
    prob = skewnorm.cdf(threshold + 0.5, skew_param, loc=forecast, scale=uncertainty_std)
    return max(min(prob, 0.99), 0.01)

def calibrated_above_probability(threshold, forecast, uncertainty_std, city_key, agreement_level='high'):
    skew_param = get_skew_parameter(city_key)
    prob = 1 - skewnorm.cdf(threshold - 0.5, skew_param, loc=forecast, scale=uncertainty_std)
    return max(min(prob, 0.99), 0.01)

def apply_probability_cap(prob, market_prob, side='YES'):
    if side == 'YES':
        capped_prob = min(prob, MAX_PROBABILITY)
        if market_prob > MAX_PROBABILITY:
            return capped_prob, True
        return capped_prob, False
    else:
        if prob > 0.80:
            return 1 - prob, True
        capped_yes_prob = min(prob, MAX_PROBABILITY)
        return 1 - capped_yes_prob, False

def calculate_dynamic_uncertainty(city_key, model_spread, num_models):
    base_sigma = CITY_BASE_UNCERTAINTY.get(city_key, 2.8)
    spread_sigma = model_spread / SPREAD_UNCERTAINTY_DIVISOR
    effective_sigma = np.sqrt(base_sigma**2 + spread_sigma**2)
    effective_sigma = max(effective_sigma, MIN_UNCERTAINTY)
    
    if num_models == 1:
        effective_sigma *= 1.3
    
    components = {
        "base_sigma": base_sigma,
        "spread_sigma": spread_sigma,
        "model_spread": model_spread,
        "effective_sigma": effective_sigma
    }
        
    return effective_sigma, components

def get_bin_for_temp(temp):
    temp_rounded = int(np.round(temp))
    if temp_rounded % 2 == 1: lower = temp_rounded
    else: lower = temp_rounded - 1
    return (lower, lower + 1)

def find_actual_kalshi_bucket(temp, markets):
    temp_rounded = int(np.round(temp))
    available_buckets = []
    for market in markets:
        subtitle = market.get("subtitle", "")
        if "to" in subtitle:
            numbers = re.findall(r'-?\d+', subtitle)
            if len(numbers) >= 2:
                available_buckets.append((int(numbers[0]), int(numbers[1])))
    if not available_buckets: return None
    for low, high in available_buckets:
        if low <= temp_rounded <= high: return (low, high)
    return None

# ============ DATA FETCHING (RESTORED FROM v9.3.5) ============

def fetch_open_meteo(lat, lon, timezone):
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ["temperature_2m_max"],
        "past_days": 7, "forecast_days": 3,
        "temperature_unit": "fahrenheit", "timezone": timezone
    }
    return requests.get(open_meteo_url, params=params).json()

def fetch_gfs_forecast(lat, lon, timezone, target_date):
    try:
        gfs_url = "https://api.open-meteo.com/v1/gfs"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": ["temperature_2m_max"],
            "forecast_days": 3,
            "temperature_unit": "fahrenheit", "timezone": timezone
        }
        response = requests.get(gfs_url, params=params, timeout=10)
        if response.status_code != 200: return None
        
        data = response.json()
        dates = data.get("daily", {}).get("time", [])
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        target_str = target_date.strftime("%Y-%m-%d")
        
        for i, d in enumerate(dates):
            if d == target_str: return temps[i]
        return None
    except Exception as e:
        print(f"     ⚠️ GFS error: {e}")
        return None

def fetch_nws_forecast(lat, lon, target_date):
    """Fetch forecast from National Weather Service API."""
    try:
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        points_response = requests.get(
            points_url,
            headers={"User-Agent": "(KalshiWeatherModel, contact@example.com)"},
            timeout=10
        )
        if points_response.status_code != 200: return None

        forecast_url = points_response.json()["properties"]["forecast"]
        forecast_response = requests.get(
            forecast_url,
            headers={"User-Agent": "(KalshiWeatherModel, contact@example.com)"},
            timeout=10
        )
        if forecast_response.status_code != 200: return None

        periods = forecast_response.json()["properties"]["periods"]
        target_str = target_date.strftime("%Y-%m-%d")

        for period in periods:
            period_start = period.get("startTime", "")
            if target_str in period_start and period.get("isDaytime", False):
                temp = period.get("temperature")
                if temp: return float(temp)
        return None
    except Exception as e:
        print(f"     ⚠️ NWS API error: {e}")
        return None

def fetch_hrrr_forecast_fixed(lat, lon, target_date, utc_offset):
    """Fetch HRRR forecast with FIXED sampling."""
    if not HERBIE_AVAILABLE: return None, None
    
    today = datetime.now().date()
    current_hour = datetime.now().hour
    
    if current_hour >= 14:
        model_run_date = today; model_run_hour = 12
    elif current_hour >= 2:
        model_run_date = today; model_run_hour = 0
    else:
        model_run_date = today - timedelta(days=1); model_run_hour = 12
    
    model_run_str = f"{model_run_date.strftime('%Y-%m-%d')} {model_run_hour:02d}:00"
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    
    target_12_local_utc = 12 - utc_offset
    target_start = datetime.combine(target_date, datetime.min.time().replace(hour=target_12_local_utc % 24))
    if target_12_local_utc >= 24: target_start += timedelta(days=1)
    
    hours_to_start = int((target_start - model_run_datetime).total_seconds() / 3600)
    forecast_hours = list(range(max(1, hours_to_start), min(48, hours_to_start + 7)))
    
    print(f"     HRRR Model run: {model_run_str} UTC")
    
    temperatures = []
    
    for fxx in forecast_hours:
        try:
            H = Herbie(model_run_str, model='hrrr', product='sfc', fxx=fxx)
            ds = H.xarray("TMP:2 m", remove_grib=True)
            temp_data = ds['t2m']
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            
            target_lon = lon if lon > 0 else lon + 360
            dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            temp_k = float(temp_data.values[min_idx])
            temperatures.append((temp_k - 273.15) * 9/5 + 32)
        except: continue
    
    if not temperatures: return None, None
    return max(temperatures), {'all_temps': temperatures}

# ============ MAIN ANALYSIS LOGIC ============

def analyze_city(city_key, bankroll, target_date_override=None):
    city = cities[city_key]
    qualifying_bets = []

    print(f"\n{'='*70}")
    print(f"📍 ANALYZING: {city['name'].upper()}")

    # Date Logic
    if target_date_override:
        target_date = target_date_override
    else:
        now = datetime.now()
        if now.hour < 6:
            target_date = now.date()
        else:
            target_date = now.date() + timedelta(days=1)
    target_str = target_date.strftime("%Y-%m-%d")
    
    print(f"   Date: {target_str}")
    
    # 1. Fetch Forecasts
    print(f"\n  📡 Fetching forecasts...")
    
    # Open-Meteo
    om_data = fetch_open_meteo(city["lat"], city["lon"], city["timezone"])
    om_val = None
    if om_data:
        for i, d in enumerate(om_data["daily"]["time"]):
            if d == target_str:
                om_val = om_data["daily"]["temperature_2m_max"][i]
                print(f"     [1] Open-Meteo: {om_val:.1f}°F")
                break
    
    # GFS
    gfs_val = fetch_gfs_forecast(city["lat"], city["lon"], city["timezone"], target_date)
    if gfs_val: print(f"     [2] GFS: {gfs_val:.1f}°F")
    
    # NWS
    nws_val = fetch_nws_forecast(city["lat"], city["lon"], target_date)
    if nws_val: print(f"     [3] NWS: {nws_val:.1f}°F")
    
    # HRRR
    hrrr_val, _ = fetch_hrrr_forecast_fixed(city["lat"], city["lon"], target_date, city["utc_offset"])
    if hrrr_val: print(f"     [4] HRRR: {hrrr_val:.1f}°F")
    
    forecasts = []
    if om_val: forecasts.append(("Open-Meteo", om_val))
    if gfs_val: forecasts.append(("GFS", gfs_val))
    if nws_val: forecasts.append(("NWS", nws_val))
    if hrrr_val: forecasts.append(("HRRR", hrrr_val))
    
    if not forecasts:
        print("   ❌ No forecasts available.")
        return [], None
        
    # 2. Weighted Ensemble (Configurable)
    total_weight = 0
    weighted_sum = 0
    temps = []
    
    # Normalize weights if some models missing
    available_models = [f[0] for f in forecasts]
    
    print("\n   📊 Ensemble Calculation:")
    for name, val in forecasts:
        w = ENSEMBLE_WEIGHTS.get(name, 0.25)
        weighted_sum += val * w
        total_weight += w
        temps.append(val)
        
    if total_weight > 0:
        raw_ensemble = weighted_sum / total_weight
    else:
        raw_ensemble = sum(temps) / len(temps)
    
    # 3. Apply Station Bias (CRITICAL STEP)
    bias = STATION_BIAS.get(city_key, 0.0)
    final_forecast = raw_ensemble + bias
    
    print(f"   ⚖️  Raw Ensemble: {raw_ensemble:.2f}°F")
    print(f"   🔧 Station Bias: {bias:+.2f}°F")
    print(f"   🎯 FINAL FORECAST: {final_forecast:.2f}°F")
    
    if bias == 0.0:
        print("      ⚠️  WARNING: Bias is 0.0. Verify this city's station offset!")

    # 4. Uncertainty & Spread
    model_spread = max(temps) - min(temps)
    uncertainty, unc_components = calculate_dynamic_uncertainty(city_key, model_spread, len(forecasts))
    skew = get_skew_parameter(city_key)
    agreement_level = 'high' if model_spread < 2.0 else 'low'
    
    print(f"   📉 Uncertainty (σ): {uncertainty:.2f}°F (Base: {unc_components['base_sigma']}, Spread: {model_spread:.1f})")
    print(f"   〰️  Skew Param (α): {skew} ({'Normal' if skew==0 else 'Skewed'})")

    # 5. Fetch Markets (RESTORED)
    print(f"\n  💰 Fetching Kalshi market data...")
    kalshi_base = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"series_ticker": city["kalshi_series"], "status": "open", "limit": 100}
    
    try:
        markets_response = requests.get(f"{kalshi_base}/markets", params=params)
        all_markets = markets_response.json().get("markets", [])
        target_kalshi = target_date.strftime("%y%b%d").upper()
        markets = [m for m in all_markets if target_kalshi in m.get("ticker", "")]
        
        if not markets:
            print(f"     ❌ No markets found for {target_str}")
            return [], None
            
        print(f"     ✅ Found {len(markets)} contracts")
        
        # Find bucket
        forecast_bin = find_actual_kalshi_bucket(final_forecast, markets)
        if not forecast_bin: forecast_bin = get_bin_for_temp(final_forecast)
        print(f"     🎯 Target Bin: {forecast_bin[0]}°-{forecast_bin[1]}°F")
        
        city_summary = {
            "city": city["name"],
            "ensemble_forecast": final_forecast,
            "forecast_bin": forecast_bin,
            "agreement_level": agreement_level,
            "uncertainty": uncertainty
        }
        
        # 6. Analyze Contracts
        for market in markets:
            subtitle = market.get("subtitle", "")
            kalshi_prob = ((market.get("yes_bid", 0) + market.get("yes_ask", 100))/200) if market.get("yes_bid") else 0
            if kalshi_prob == 0: continue
            
            # Price Filter
            if kalshi_prob < MIN_CONTRACT_PRICE or kalshi_prob > MAX_CONTRACT_PRICE: continue
            
            # Parse Contract
            numbers = re.findall(r'-?\d+', subtitle)
            contract_type = None
            model_prob = 0
            is_forecast_bin = False
            
            if "to" in subtitle and len(numbers) >= 2:
                low, high = int(numbers[0]), int(numbers[1])
                model_prob = calibrated_probability(low, high, final_forecast, uncertainty, city_key)
                is_forecast_bin = (low == forecast_bin[0])
                contract_type = "range"
            elif "or below" in subtitle and len(numbers) >= 1:
                thresh = int(numbers[0])
                model_prob = calibrated_below_probability(thresh, final_forecast, uncertainty, city_key)
                contract_type = "below"
            elif "or above" in subtitle and len(numbers) >= 1:
                thresh = int(numbers[0])
                model_prob = calibrated_above_probability(thresh, final_forecast, uncertainty, city_key)
                contract_type = "above"
            else: continue
            
            # Evaluate YES
            capped_yes, skip_yes = apply_probability_cap(model_prob, kalshi_prob, 'YES')
            yes_edge = capped_yes - kalshi_prob
            
            is_suspicious = yes_edge > MAX_EDGE_WARNING
            
            if not skip_yes and yes_edge > MIN_EDGE_REQUIREMENT and not is_suspicious:
                bet_size, kelly_info = calculate_kelly_bet(bankroll, capped_yes, kalshi_prob, 
                                                         model_spread=model_spread)
                if bet_size >= MIN_BET_SIZE:
                    qualifying_bets.append({
                        "city": city["name"], "city_key": city_key,
                        "subtitle": subtitle, "side": "YES",
                        "bet_price": kalshi_prob, "our_prob_win": capped_yes,
                        "edge": yes_edge, "bet_size": bet_size,
                        "kelly_info": kelly_info, "price_bucket": get_price_bucket(kalshi_prob),
                        "is_forecast_bin": is_forecast_bin,
                        "edge_ratio": yes_edge / MIN_EDGE_REQUIREMENT
                    })
                    
            # Evaluate NO (Simplified)
            if not is_forecast_bin:
                capped_no, skip_no = apply_probability_cap(model_prob, kalshi_prob, 'NO')
                no_market = 1 - kalshi_prob
                no_edge = capped_no - no_market
                
                if not skip_no and no_edge > MIN_EDGE_REQUIREMENT and (no_edge <= MAX_EDGE_WARNING):
                     bet_size, kelly_info = calculate_kelly_bet(bankroll, capped_no, no_market, 
                                                              model_spread=model_spread)
                     if bet_size >= MIN_BET_SIZE:
                        qualifying_bets.append({
                            "city": city["name"], "city_key": city_key,
                            "subtitle": subtitle, "side": "NO",
                            "bet_price": no_market, "our_prob_win": capped_no,
                            "edge": no_edge, "bet_size": bet_size,
                            "kelly_info": kelly_info, "price_bucket": get_price_bucket(no_market),
                            "is_forecast_bin": False,
                            "edge_ratio": no_edge / MIN_EDGE_REQUIREMENT
                        })

        return qualifying_bets, city_summary
        
    except Exception as e:
        print(f"     ❌ Market Fetch Error: {e}")
        return [], None

# ============ SMART SELECTION ============
def smart_select_bets(all_bets):
    if not all_bets: return []
    sorted_bets = sorted(all_bets, key=lambda x: -x["edge_ratio"])
    selected = []
    city_counts = {}
    
    for bet in sorted_bets:
        if len(selected) >= MAX_TOTAL_BETS_PER_DAY: break
        city = bet["city_key"]
        if city_counts.get(city, 0) >= MAX_BETS_PER_CITY: continue
        
        if city_counts.get(city, 0) >= 1:
            if bet["edge_ratio"] < SAME_CITY_MULTI_BET_THRESHOLD: continue
            
        selected.append(bet)
        city_counts[city] = city_counts.get(city, 0) + 1
        
    return selected

def print_recommendations(selected_bets, city_summaries, bankroll):
    print(f"\n{'='*70}")
    print("🎯 BETTING RECOMMENDATIONS (v9.3.6)")
    print(f"{'='*70}")
    
    if not selected_bets:
        print("\n🛑 NO BETS RECOMMENDED TODAY")
        return

    for i, bet in enumerate(selected_bets, 1):
        print(f"   ┌─ BET #{i}: {bet['city']} ({bet['side']})")
        print(f"   │  Contract: {bet['subtitle']}")
        print(f"   │  Price: {bet['bet_price']*100:.0f}¢ | Prob: {bet['our_prob_win']*100:.1f}% | Edge: {bet['edge']*100:.1f}%")
        print(f"   │  💰 BET: ${bet['bet_size']:.2f}")
        print(f"   └{'─'*50}")

# ============ ARGUMENT PARSING ============
def parse_arguments():
    parser = argparse.ArgumentParser(description="Kalshi Weather Betting Model v9.3.6")
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

    args = parse_arguments()
    KELLY_FRACTION = args.kelly
    MAX_BET_FRACTION = args.maxbet
    STARTING_BANKROLL = args.bankroll

    # Handle city filter
    if args.cities and args.cities.lower() == "all":
        selected_cities = list(cities.keys())
    elif args.cities:
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
                print(f"Warning: Unknown city '{city}', skipping...")
        selected_cities = valid_cities if valid_cities else ["austin", "miami"]

    # Handle date options
    target_date = None
    now = datetime.now()

    if args.today:
        target_date = now.date()
        print(f"Mode: Analyzing TODAY's weather ({target_date})")
    elif args.tomorrow:
        target_date = now.date() + timedelta(days=1)
        print(f"Mode: Analyzing TOMORROW's weather ({target_date})")
    elif args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            print(f"Mode: Analyzing weather for {target_date}")
        except ValueError:
            print(f"Warning: Invalid date format '{args.date}', using default")
            target_date = None
    # If no date option specified, analyze_city will use its default logic

    print_header()

    all_bets = []
    summaries = []

    for city_key in selected_cities:
        bets, summary = analyze_city(city_key, STARTING_BANKROLL, target_date)
        all_bets.extend(bets)
        summaries.append(summary)

    final_bets = smart_select_bets(all_bets)
    print_recommendations(final_bets, summaries, STARTING_BANKROLL)

if __name__ == "__main__":
    main()