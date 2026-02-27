"""
KALSHI BIN-BASED PREDICTION MODEL v1.5
======================================
STATION PRECISION + ADJACENT BIN FLOOR + C→F ROUNDING AWARENESS

v1.5 Changes (from v1.4):

1. ✅ CORRECTED NWS STATION COORDINATES
   - Miami (KMIA): 25.7959,-80.2870 → 25.7954,-80.2901 (minor)
   - Austin (KAUS): 30.2,-97.68 → 30.1945,-97.6699 (minor)
   - Chicago (KMDW): 41.85,-87.65 → 41.7856,-87.7527 (BIG fix — was ~7mi off, near Loop not Midway)
   - NYC (KNYC): 40.78,-73.97 → 40.7790,-73.9692 (minor, Central Park station)
   - Added NWS station IDs for reference (KMIA, KAUS, KMDW, KNYC)

2. ✅ ADJACENT BIN FLOOR + MARKET PRIOR BLEND
   - Model was assigning near-zero (1.4%) to bins only 2-3°F from forecast
   - Part A: Raised floors significantly — ±1 bin=12%, ±2 bins=5%, ±3 bins=2%
   - Part B: Bayesian shrinkage toward market when disagreement is large
     - When model says 4% and market says 48%, model is likely missing something
     - Blends model with market prior: blend = (1-s)*model + s*market
     - Shrinkage s scales with disagreement magnitude AND bin proximity
     - Base s=15%, max s=40% (model always has at least 60% weight)
   - Together these prevent absurd 96%+ NO confidence on nearby bins

3. ✅ C→F ROUNDING AWARENESS
   - NWS records in Celsius and rounds to Fahrenheit
   - Some °F values map to 1 C reading (exact), others to 2 (rounded)
   - Bins at exact conversion points (32,41,50,59,68,77,86,95°F) are
     effectively narrower — model now adjusts bin width accordingly
   - Adjacent bins that straddle a rounding boundary get a slight boost

4. ✅ STATION MICROCLIMATE BIAS (ENHANCED)
   - Airport stations (KMIA, KAUS, KMDW) have tarmac heat island effect
   - Added warm_bias_sunny parameter: extra +0.5°F for clear/partly cloudy days
   - NYC Central Park (KNYC) has park cooling effect, already reflected in bias

All v1.4 features retained:
- Fixed calibration, single spread penalty, Kelly diagnostics
- 4-source independent ensemble (NWS, HRRR, ECMWF, ICON)
- Bin probability renormalization
- City-conditional entropy thresholds
- Bid-ask spread filtering

Data Sources: NWS, HRRR, ECMWF (via Open-Meteo), ICON (via Open-Meteo)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
from scipy import stats
from scipy.stats import skewnorm, entropy as scipy_entropy
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ============ HRRR/HERBIE IMPORTS ============
try:
    from herbie import Herbie
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False
    print("⚠️ Herbie not installed. HRRR data unavailable.")

# ============================================================================
#                           CONFIGURATION
# ============================================================================

# Model Weights - UPDATED for new sources
MODEL_WEIGHTS = {
    "NWS": 0.40,      # Official source - highest weight
    "HRRR": 0.25,     # High-res short-term
    "ECMWF": 0.20,    # Best global model (was GFS)
    "ICON": 0.15      # German model for independence (was Open-Meteo)
}

# Station Bias — base offset (model grid vs actual station thermometer)
STATION_BIAS = {
    "miami": -0.2,
    "austin": 0.3,
    "chicago": 1.5,
    "nyc": -0.5,
}

# Airport tarmac warm bias — extra boost for airport stations on sunny/partly sunny days
# NYC (Central Park) is NOT an airport station, so no tarmac effect
STATION_WARM_BIAS_SUNNY = {
    "miami": 0.5,       # KMIA — large airport, concrete runways
    "austin": 0.4,      # KAUS — large airport
    "chicago": 0.3,     # KMDW — smaller footprint but still tarmac
    "nyc": 0.0,         # KNYC — Central Park, no tarmac effect (park cooling)
}

# ============ C→F ROUNDING TABLE ============
# NWS records in Celsius and converts to Fahrenheit with rounding.
# Every 5°C is exact (e.g., 25°C = 77°F exactly).
# Other values round to a 2°F range (e.g., 27°C = 80.6°F → reports as 80 or 81).
# This means some Kalshi bins are effectively "wider" or "narrower" depending on
# whether they contain exact conversion points.
#
# Exact °F values (only 1 possible C reading): 
#   ...32, 41, 50, 59, 68, 77, 86, 95, 104...
# These are "pinch points" — harder to land on.
EXACT_F_VALUES = {32, 41, 50, 59, 68, 77, 86, 95, 104, 23, 14, 5}

# C→F mapping: celsius -> (exact_f, reported_f_low, reported_f_high)
CF_ROUNDING = {}
for c in range(-18, 41):
    exact_f = c * 9.0 / 5.0 + 32.0
    rounded = round(exact_f)
    if exact_f == int(exact_f):  # Exact conversion (multiple of 5°C)
        CF_ROUNDING[c] = (exact_f, int(exact_f), int(exact_f))
    else:
        low_f = int(exact_f) if exact_f - int(exact_f) < 0.5 else int(exact_f) + 1
        # More precisely: NWS rounds to nearest int
        low_f = rounded
        CF_ROUNDING[c] = (exact_f, rounded, rounded)

# City confidence multipliers (from backtest — v1.4: reduced MIA to limit feedback loop)
CITY_CONFIDENCE_MULTIPLIER = {
    "miami": 1.08,      # Was 1.15 — still best performer but less aggressive
    "austin": 1.00,
    "chicago": 0.85,
    "nyc": 0.50,        # Worst performer
}

# City base uncertainty
CITY_BASE_UNCERTAINTY = {
    "miami": 2.0,
    "austin": 2.1,
    "chicago": 4.0,
    "nyc": 5.0,
}

# Skew parameters
CITY_SKEW = {
    "miami": -1.5,
    "austin": 0.0,
    "chicago": -0.5,
    "nyc": -0.5,
}

# ============ CITY-CONDITIONAL ENTROPY THRESHOLDS ============
CITY_ENTROPY_THRESHOLD = {
    "miami": 1.8,
    "austin": 2.0,
    "chicago": 2.3,
    "nyc": 2.2,
}
DEFAULT_ENTROPY_THRESHOLD = 2.0

# ============ BID-ASK SPREAD CONFIGURATION ============
MAX_SPREAD_THRESHOLD = 0.15
SPREAD_PENALTY_FACTOR = 0.5

# ============ TRADING CONFIGURATION ============
MIN_EDGE_THRESHOLD = 0.10
MIN_EDGE_YES_BET = 0.15
MIN_CONFIDENCE_GAP = 0.08

# Price Filters
MIN_CONTRACT_PRICE = 0.27
MAX_CONTRACT_PRICE = 0.80

# Kelly Criterion — v1.4: tightened from 0.20/0.10 to 0.15/0.08
KELLY_FRACTION = 0.15       # Was 0.20 — your actual bets were overshooting model recs
MIN_BET_SIZE = 1.00
MAX_BET_FRACTION = 0.08     # Was 0.10 — caps any single bet at 8% of bankroll
STARTING_BANKROLL = 100.0   # ⚠️ UPDATE THIS or use --bankroll flag with your ACTUAL balance

# ============ AGGREGATE EXPOSURE LIMITS ============
MAX_CITY_EXPOSURE_FRACTION = 0.25
MAX_TOTAL_BETS = 5
MAX_BETS_PER_CITY = 2

# YES/NO preferences (from backtest)
NO_BET_PREFERENCE_MULTIPLIER = 1.35
YES_BET_PENALTY_MULTIPLIER = 0.60

# Bin type multipliers (from backtest — v1.4: raised below/above, were too harsh)
BIN_TYPE_MULTIPLIER = {
    "range": 1.15,
    "below": 0.40,      # Was 0.25 — still penalized but you take these bets anyway
    "above": 0.35,      # Was 0.20 — same reasoning
}

# T-distribution DF
T_DISTRIBUTION_DF = 5

# C→F Rounding Width Adjustments (v1.5)
CF_WIDTH_BOOST = 1.06    # Boost for "all-rounded" bins (effectively wider target)
CF_WIDTH_PENALTY = 0.94  # Penalty for bins containing exact conversion points (narrower)

# ============ CITY CONFIGURATION ============
# Coordinates are the EXACT NWS station locations that Kalshi uses for settlement
CITIES = {
    "miami": {
        "name": "Miami",
        "lat": 25.7954,         # KMIA — Miami International Airport (was 25.7959)
        "lon": -80.2901,        # KMIA — corrected from -80.2870
        "nws_station": "KMIA",
        "kalshi_series": "KXHIGHMIA",
        "timezone": "America/New_York",
        "utc_offset": -5,
        "is_airport": True,     # Tarmac heat island applies
    },
    "austin": {
        "name": "Austin",
        "lat": 30.1945,         # KAUS — Austin-Bergstrom Intl (was 30.2)
        "lon": -97.6699,        # KAUS — corrected from -97.68
        "nws_station": "KAUS",
        "kalshi_series": "KXHIGHAUS",
        "timezone": "America/Chicago",
        "utc_offset": -6,
        "is_airport": True,
    },
    "chicago": {
        "name": "Chicago",
        "lat": 41.7856,         # KMDW — Chicago Midway Airport (was 41.85 — ~7mi off!)
        "lon": -87.7527,        # KMDW — corrected from -87.65 (was pointing near the Loop)
        "nws_station": "KMDW",
        "kalshi_series": "KXHIGHCHI",
        "timezone": "America/Chicago",
        "utc_offset": -6,
        "is_airport": True,
    },
    "nyc": {
        "name": "New York",
        "lat": 40.7790,         # KNYC — Central Park (was 40.78)
        "lon": -73.9692,        # KNYC — corrected from -73.97
        "nws_station": "KNYC",
        "kalshi_series": "KXHIGHNYC",
        "timezone": "America/New_York",
        "utc_offset": -5,
        "is_airport": False,    # Central Park — no tarmac effect
    }
}

SELECTED_CITIES = ["miami", "austin"]


# ============================================================================
#                           DATA CLASSES
# ============================================================================

@dataclass
class Forecast:
    source: str
    temperature: float
    weight: float = 0.0


@dataclass
class BinInfo:
    ticker: str
    subtitle: str
    bin_type: str
    low: Optional[int] = None
    high: Optional[int] = None
    threshold: Optional[int] = None
    
    # Market data
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    spread: float = 0.0
    market_prob: float = 0.0
    market_prob_pessimistic: float = 0.0
    
    # Model predictions
    model_prob_raw: float = 0.0
    model_prob: float = 0.0
    calibrated_prob: float = 0.0
    
    # Edge calculations
    edge_raw: float = 0.0
    edge_spread_adjusted: float = 0.0
    adjusted_edge: float = 0.0
    
    rank: int = 0
    spread_warning: bool = False


@dataclass  
class BinDistribution:
    bins: List[BinInfo]
    entropy: float = 0.0
    entropy_threshold: float = 2.0
    confidence_gap: float = 0.0
    top_bin: Optional[BinInfo] = None
    second_bin: Optional[BinInfo] = None
    latent_forecast: float = 0.0
    uncertainty: float = 0.0
    model_spread: float = 0.0
    agreement_level: str = "medium"
    total_prob_before_norm: float = 0.0


@dataclass
class TradeRecommendation:
    city: str
    city_key: str
    bin_info: BinInfo
    side: str
    model_prob: float
    market_prob: float
    raw_edge: float
    adjusted_edge: float
    bet_size: float
    potential_profit: float
    confidence_rank: int
    adjustments_applied: List[str]
    spread_penalty: float = 0.0


# ============================================================================
#                           CALIBRATION
# ============================================================================

def apply_calibration(raw_prob: float) -> float:
    """
    Apply calibration — v1.4 FIXED.
    
    Old version was way too aggressive:
      - 0.55 raw → 0.385 calibrated (30% haircut!)
      - 0.65 raw → 0.325 calibrated (50% haircut!)
      - 0.75 raw → 0.300 calibrated (60% haircut!)
    This destroyed YES edges and artificially inflated NO edges.
    
    New version: gentle pull toward center (0.30) for high probs.
    - Low probs (<0.20): untouched (tail bets, let the model speak)
    - Mid probs (0.20-0.45): slight boost (model tends to underestimate these)
    - High probs (>0.45): gentle dampening, NOT destruction
      - 0.55 → 0.49  (not 0.385)
      - 0.65 → 0.52  (not 0.325)  
      - 0.75 → 0.53  (not 0.300)
    """
    if raw_prob < 0.20:
        return raw_prob
    elif raw_prob < 0.45:
        # Slight boost for mid-range — model underestimates these bins
        return min(raw_prob * 1.12, 0.50)
    else:
        # Gentle sigmoid pull toward 0.30 for high probabilities
        # Dampens overconfidence without destroying the signal
        center = 0.30
        strength = 0.55  # How hard to pull toward center (0=none, 1=full)
        dampened = raw_prob + strength * (center - raw_prob)
        return max(min(dampened, 0.60), 0.15)


# ============================================================================
#                           DATA FETCHING - UPDATED!
# ============================================================================

def fetch_ecmwf(lat: float, lon: float, timezone: str, target_date) -> Optional[Forecast]:
    """
    Fetch ECMWF forecast via Open-Meteo API.
    ECMWF (European Centre for Medium-Range Weather Forecasts) is often 
    considered the most accurate global weather model.
    """
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ["temperature_2m_max"],
        "forecast_days": 10,  # ECMWF has longer range
        "temperature_unit": "fahrenheit", 
        "timezone": timezone
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        target_str = target_date.strftime("%Y-%m-%d")
        for i, d in enumerate(data.get("daily", {}).get("time", [])):
            if d == target_str:
                temp = data["daily"]["temperature_2m_max"][i]
                if temp is not None:
                    return Forecast(source="ECMWF", temperature=temp, weight=MODEL_WEIGHTS["ECMWF"])
        return None
    except Exception as e:
        print(f"     ⚠️ ECMWF error: {e}")
        return None


def fetch_icon(lat: float, lon: float, timezone: str, target_date) -> Optional[Forecast]:
    """
    Fetch ICON forecast via Open-Meteo API.
    ICON (Icosahedral Nonhydrostatic) is the German weather model from DWD.
    Provides independent verification from US and European models.
    """
    url = "https://api.open-meteo.com/v1/dwd-icon"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ["temperature_2m_max"],
        "forecast_days": 7,
        "temperature_unit": "fahrenheit", 
        "timezone": timezone
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        target_str = target_date.strftime("%Y-%m-%d")
        for i, d in enumerate(data.get("daily", {}).get("time", [])):
            if d == target_str:
                temp = data["daily"]["temperature_2m_max"][i]
                if temp is not None:
                    return Forecast(source="ICON", temperature=temp, weight=MODEL_WEIGHTS["ICON"])
        return None
    except Exception as e:
        print(f"     ⚠️ ICON error: {e}")
        return None


def fetch_nws(lat: float, lon: float, target_date) -> Optional[Forecast]:
    """
    Fetch from NWS API - free, unlimited, official US source.
    This is what Kalshi uses for settlement!
    """
    try:
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        headers = {"User-Agent": "(KalshiBinModel/1.3, contact@example.com)"}
        points_response = requests.get(points_url, headers=headers, timeout=10)
        if points_response.status_code != 200:
            return None
        forecast_url = points_response.json()["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        if forecast_response.status_code != 200:
            return None
        periods = forecast_response.json()["properties"]["periods"]
        target_str = target_date.strftime("%Y-%m-%d")
        for period in periods:
            if target_str in period.get("startTime", "") and period.get("isDaytime", False):
                temp = period.get("temperature")
                if temp:
                    return Forecast(source="NWS", temperature=float(temp), weight=MODEL_WEIGHTS["NWS"])
        return None
    except Exception as e:
        print(f"     ⚠️ NWS error: {e}")
        return None


def fetch_hrrr(lat: float, lon: float, target_date, utc_offset: int) -> Optional[Forecast]:
    """
    Fetch HRRR (High-Resolution Rapid Refresh) via Herbie.
    Best for short-term forecasts (< 18 hours).
    """
    if not HERBIE_AVAILABLE:
        return None
    try:
        today = datetime.now().date()
        current_hour = datetime.now().hour
        if current_hour >= 14:
            model_run_date, model_run_hour = today, 12
        elif current_hour >= 2:
            model_run_date, model_run_hour = today, 0
        else:
            model_run_date, model_run_hour = today - timedelta(days=1), 12
        
        model_run_str = f"{model_run_date.strftime('%Y-%m-%d')} {model_run_hour:02d}:00"
        model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
        
        target_12_local_utc = 12 - utc_offset
        target_start = datetime.combine(target_date, datetime.min.time().replace(hour=target_12_local_utc % 24))
        if target_12_local_utc >= 24:
            target_start += timedelta(days=1)
        
        hours_to_start = int((target_start - model_run_datetime).total_seconds() / 3600)
        forecast_hours = list(range(max(1, hours_to_start), min(48, hours_to_start + 7)))
        
        print(f"     HRRR run: {model_run_str} UTC")
        
        temperatures = []
        for fxx in forecast_hours:
            try:
                H = Herbie(model_run_str, model='hrrr', product='sfc', fxx=fxx)
                ds = H.xarray("TMP:2 m", remove_grib=True)
                temp_data = ds['t2m']
                lats, lons = ds['latitude'].values, ds['longitude'].values
                target_lon = lon if lon > 0 else lon + 360
                dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
                min_idx = np.unravel_index(np.argmin(dist), dist.shape)
                temp_k = float(temp_data.values[min_idx])
                temperatures.append((temp_k - 273.15) * 9/5 + 32)
            except:
                continue
        
        if temperatures:
            return Forecast(source="HRRR", temperature=max(temperatures), weight=MODEL_WEIGHTS["HRRR"])
        return None
    except Exception as e:
        print(f"     ⚠️ HRRR error: {e}")
        return None


def fetch_all_forecasts(city_key: str, target_date) -> List[Forecast]:
    """
    Fetch forecasts from 4 INDEPENDENT sources:
    1. NWS - Official US source (what Kalshi uses)
    2. HRRR - High-resolution US model  
    3. ECMWF - European model (best global)
    4. ICON - German model (independent verification)
    """
    city = CITIES[city_key]
    forecasts = []
    
    print("  📡 Fetching forecasts...")
    
    # 1. ECMWF (European model) - was Open-Meteo
    result = fetch_ecmwf(city["lat"], city["lon"], city["timezone"], target_date)
    if result:
        forecasts.append(result)
        print(f"     [1] ECMWF: {result.temperature:.1f}°F")
    else:
        print(f"     [1] ECMWF: ❌")
    
    # 2. ICON (German model) - was GFS
    result = fetch_icon(city["lat"], city["lon"], city["timezone"], target_date)
    if result:
        forecasts.append(result)
        print(f"     [2] ICON: {result.temperature:.1f}°F")
    else:
        print(f"     [2] ICON: ❌")
    
    # 3. NWS (Official US source) - unchanged, highest weight
    result = fetch_nws(city["lat"], city["lon"], target_date)
    if result:
        forecasts.append(result)
        print(f"     [3] NWS: {result.temperature:.1f}°F ⭐")
    else:
        print(f"     [3] NWS: ❌")
    
    # 4. HRRR (High-res US model) - unchanged
    result = fetch_hrrr(city["lat"], city["lon"], target_date, city["utc_offset"])
    if result:
        forecasts.append(result)
        print(f"     [4] HRRR: {result.temperature:.1f}°F")
    else:
        print(f"     [4] HRRR: ❌")
    
    return forecasts


# ============================================================================
#                           KALSHI API
# ============================================================================

def fetch_kalshi_markets(series_ticker: str, target_date) -> List[Dict]:
    kalshi_base = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"series_ticker": series_ticker, "status": "open", "limit": 100}
    try:
        response = requests.get(f"{kalshi_base}/markets", params=params, headers={"Accept": "application/json"}, timeout=10)
        if response.status_code != 200:
            return []
        all_markets = response.json().get("markets", [])
        target_kalshi = target_date.strftime("%y%b%d").upper()
        return [m for m in all_markets if target_kalshi in m.get("ticker", "")]
    except Exception as e:
        print(f"     ❌ Kalshi API error: {e}")
        return []


def parse_kalshi_bins(markets: List[Dict]) -> List[BinInfo]:
    """Parse bins with spread calculation and pessimistic pricing."""
    bins = []
    
    for market in markets:
        subtitle = market.get("subtitle", "")
        ticker = market.get("ticker", "")
        
        yes_bid = market.get("yes_bid", 0) / 100 if market.get("yes_bid") else 0
        yes_ask = market.get("yes_ask", 100) / 100 if market.get("yes_ask") else 1.0
        
        spread = yes_ask - yes_bid if yes_bid > 0 else 0.5
        
        if yes_bid > 0 and yes_ask < 1:
            market_prob = (yes_bid + yes_ask) / 2
            market_prob_pessimistic_yes = yes_ask
            market_prob_pessimistic_no = 1 - yes_bid
        else:
            market_prob = yes_ask if yes_ask < 1 else 0.5
            market_prob_pessimistic_yes = yes_ask
            market_prob_pessimistic_no = 1 - yes_bid if yes_bid > 0 else 0.5
        
        spread_warning = spread > MAX_SPREAD_THRESHOLD
        
        numbers = re.findall(r'-?\d+', subtitle)
        
        if "to" in subtitle.lower() and len(numbers) >= 2:
            bins.append(BinInfo(
                ticker=ticker, subtitle=subtitle, bin_type="range",
                low=int(numbers[0]), high=int(numbers[1]),
                yes_bid=yes_bid, yes_ask=yes_ask, spread=spread,
                market_prob=market_prob,
                market_prob_pessimistic=market_prob_pessimistic_yes,
                spread_warning=spread_warning
            ))
        elif "or lower" in subtitle.lower() and len(numbers) >= 1:
            bins.append(BinInfo(
                ticker=ticker, subtitle=subtitle, bin_type="below",
                threshold=int(numbers[0]),
                yes_bid=yes_bid, yes_ask=yes_ask, spread=spread,
                market_prob=market_prob,
                market_prob_pessimistic=market_prob_pessimistic_yes,
                spread_warning=spread_warning
            ))
        elif "or higher" in subtitle.lower() and len(numbers) >= 1:
            bins.append(BinInfo(
                ticker=ticker, subtitle=subtitle, bin_type="above",
                threshold=int(numbers[0]),
                yes_bid=yes_bid, yes_ask=yes_ask, spread=spread,
                market_prob=market_prob,
                market_prob_pessimistic=market_prob_pessimistic_yes,
                spread_warning=spread_warning
            ))
    
    return bins


# ============================================================================
#                    PROBABILITY CALCULATION
# ============================================================================

def calculate_mixture_probability_range(low: int, high: int, forecasts: List[Forecast], 
                                        base_std: float, skew: float = 0.0) -> float:
    if not forecasts:
        return 0.0
    total_weight = sum(f.weight for f in forecasts)
    if total_weight == 0:
        return 0.0
    
    total_prob = 0.0
    for forecast in forecasts:
        weight = forecast.weight / total_weight
        temp = forecast.temperature
        if skew != 0:
            model_prob = (skewnorm.cdf(high + 0.5, skew, loc=temp, scale=base_std) -
                         skewnorm.cdf(low - 0.5, skew, loc=temp, scale=base_std))
        else:
            model_prob = (stats.t.cdf(high + 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=base_std) -
                         stats.t.cdf(low - 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=base_std))
        total_prob += weight * model_prob
    
    return max(min(total_prob, 0.99), 0.01)


def calculate_mixture_probability_below(threshold: int, forecasts: List[Forecast],
                                        base_std: float, skew: float = 0.0) -> float:
    if not forecasts:
        return 0.0
    total_weight = sum(f.weight for f in forecasts)
    if total_weight == 0:
        return 0.0
    
    total_prob = 0.0
    for forecast in forecasts:
        weight = forecast.weight / total_weight
        temp = forecast.temperature
        if skew != 0:
            model_prob = skewnorm.cdf(threshold + 0.5, skew, loc=temp, scale=base_std)
        else:
            model_prob = stats.t.cdf(threshold + 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=base_std)
        total_prob += weight * model_prob
    
    return max(min(total_prob, 0.99), 0.01)


def calculate_mixture_probability_above(threshold: int, forecasts: List[Forecast],
                                        base_std: float, skew: float = 0.0) -> float:
    if not forecasts:
        return 0.0
    total_weight = sum(f.weight for f in forecasts)
    if total_weight == 0:
        return 0.0
    
    total_prob = 0.0
    for forecast in forecasts:
        weight = forecast.weight / total_weight
        temp = forecast.temperature
        if skew != 0:
            model_prob = 1 - skewnorm.cdf(threshold - 0.5, skew, loc=temp, scale=base_std)
        else:
            model_prob = 1 - stats.t.cdf(threshold - 0.5, df=T_DISTRIBUTION_DF, loc=temp, scale=base_std)
        total_prob += weight * model_prob
    
    return max(min(total_prob, 0.99), 0.01)


# ============================================================================
#              BIN DISTRIBUTION (WITH RENORMALIZATION)
# ============================================================================

def calculate_bin_distribution(city_key: str, forecasts: List[Forecast], bins: List[BinInfo]) -> BinDistribution:
    """
    Calculate distribution with:
    1. Probability renormalization (sum to 1)
    2. City-specific entropy threshold
    3. Spread-aware edge calculation
    """
    
    # Apply station bias + airport tarmac warm bias (v1.5)
    bias = STATION_BIAS.get(city_key, 0.0)
    warm_bias = STATION_WARM_BIAS_SUNNY.get(city_key, 0.0)
    total_bias = bias + warm_bias  # warm_bias applies as a default; ideally conditioned on sky cover
    adjusted_forecasts = [Forecast(source=f.source, temperature=f.temperature + total_bias, weight=f.weight) for f in forecasts]
    
    # Calculate spread and agreement
    if len(adjusted_forecasts) >= 2:
        temps = [f.temperature for f in adjusted_forecasts]
        model_spread = max(temps) - min(temps)
    else:
        model_spread = 0.0
    
    if model_spread <= 2.0:
        agreement_level = "high"
        uncertainty_mult = 0.85
    elif model_spread <= 4.0:
        agreement_level = "medium"
        uncertainty_mult = 1.0
    else:
        agreement_level = "low"
        uncertainty_mult = 1.25
    
    base_sigma = CITY_BASE_UNCERTAINTY.get(city_key, 3.0)
    spread_sigma = model_spread / 2.0
    effective_sigma = np.sqrt(base_sigma**2 + spread_sigma**2) * uncertainty_mult
    
    skew = CITY_SKEW.get(city_key, 0.0)
    
    # Latent forecast
    total_weight = sum(f.weight for f in adjusted_forecasts)
    if total_weight > 0:
        latent_forecast = sum(f.temperature * f.weight for f in adjusted_forecasts) / total_weight
    else:
        latent_forecast = adjusted_forecasts[0].temperature if adjusted_forecasts else 0
    
    # ========== STEP 1: Calculate RAW probabilities for RANGE bins only ==========
    range_bins = [b for b in bins if b.bin_type == "range"]
    below_bins = [b for b in bins if b.bin_type == "below"]
    above_bins = [b for b in bins if b.bin_type == "above"]
    
    for bin_info in range_bins:
        raw_prob = calculate_mixture_probability_range(
            bin_info.low, bin_info.high, adjusted_forecasts, effective_sigma, skew
        )
        bin_info.model_prob_raw = raw_prob
    
    # ========== STEP 2: RENORMALIZE range bins to sum to exactly 1.0 ==========
    total_range_prob = sum(b.model_prob_raw for b in range_bins)
    
    if total_range_prob > 0:
        for b in range_bins:
            b.model_prob = b.model_prob_raw / total_range_prob
    else:
        for b in range_bins:
            b.model_prob = 1.0 / len(range_bins) if range_bins else 0.0
    
    # ========== STEP 2.5: ADJACENT BIN FLOOR + MARKET PRIOR BLEND (v1.5) ==========
    # 
    # TWO-PART FIX for overconfident NO bets on nearby bins:
    #
    # PART A: Adjacent Bin Floor
    #   Model was assigning near-zero to bins only 2-3°F from forecast.
    #   Floor ensures nearby bins get a minimum probability based on distance.
    #   ±1 bin: floor 12%, ±2 bins: floor 5%, ±3 bins: floor 2%
    #
    # PART B: Market Prior Blend (Bayesian shrinkage)
    #   When the model is very confident AND the market strongly disagrees,
    #   the model is likely missing something (station microclimate, C→F rounding,
    #   experienced trader knowledge). We blend model probabilities with market
    #   probabilities using a shrinkage factor.
    #   
    #   blend = (1 - shrinkage) * model_prob + shrinkage * market_prob
    #   
    #   Shrinkage is HIGHER when:
    #     - The model-market disagreement is large (> 20%)
    #     - The bin is close to the forecast (model should be uncertain here)
    #   Shrinkage is LOWER when:
    #     - Model and market roughly agree
    #     - The bin is far from the forecast (tail bin, model can be more confident)
    
    if range_bins:
        # --- PART A: Adjacent Bin Floor ---
        peak_bin = max(range_bins, key=lambda b: b.model_prob)
        peak_center = (peak_bin.low + peak_bin.high) / 2 if peak_bin.low is not None else 0
        
        bin_widths = [(b.high - b.low) for b in range_bins if b.low is not None and b.high is not None]
        typical_width = np.median(bin_widths) if bin_widths else 2
        
        FLOOR_ADJACENT_1 = 0.12  # ±1 bin — very plausible, must be non-trivial
        FLOOR_ADJACENT_2 = 0.05  # ±2 bins — still plausible (this is the 80-81 fix)
        FLOOR_ADJACENT_3 = 0.02  # ±3 bins — tail but not impossible
        
        floors_applied = False
        for b in range_bins:
            if b.low is None or b.high is None:
                continue
            bin_center = (b.low + b.high) / 2
            distance_bins = abs(bin_center - peak_center) / max(typical_width, 1)
            
            if distance_bins <= 1.5 and distance_bins > 0.1:
                if b.model_prob < FLOOR_ADJACENT_1:
                    b.model_prob = FLOOR_ADJACENT_1
                    floors_applied = True
            elif distance_bins <= 2.5:
                if b.model_prob < FLOOR_ADJACENT_2:
                    b.model_prob = FLOOR_ADJACENT_2
                    floors_applied = True
            elif distance_bins <= 3.5:
                if b.model_prob < FLOOR_ADJACENT_3:
                    b.model_prob = FLOOR_ADJACENT_3
                    floors_applied = True
        
        if floors_applied:
            total_after_floor = sum(b.model_prob for b in range_bins)
            if total_after_floor > 0:
                for b in range_bins:
                    b.model_prob = b.model_prob / total_after_floor
        
        # --- PART B: Market Prior Blend (Bayesian shrinkage) ---
        # Base shrinkage: how much we trust the market vs our model
        # 0.0 = trust model entirely, 1.0 = trust market entirely
        BASE_SHRINKAGE = 0.15  # Default: 85% model, 15% market
        MAX_SHRINKAGE = 0.40   # Cap: never go more than 60% model / 40% market
        
        # Increase shrinkage when disagreement is large on nearby bins
        for b in range_bins:
            if b.low is None or b.high is None or b.market_prob <= 0:
                continue
            
            bin_center = (b.low + b.high) / 2
            distance_bins = abs(bin_center - peak_center) / max(typical_width, 1)
            disagreement = abs(b.model_prob - b.market_prob)
            
            # Only blend for bins within ±3 of peak AND with significant disagreement
            if distance_bins <= 3.0 and disagreement > 0.15:
                # Scale shrinkage by disagreement magnitude
                # Bigger disagreement = more we hedge toward market
                disagreement_factor = min(disagreement / 0.40, 1.0)  # Caps at 40% disagreement
                
                # Also scale by proximity: closer bins get more shrinkage
                proximity_factor = max(0, 1.0 - distance_bins / 3.0)
                
                shrinkage = BASE_SHRINKAGE + (MAX_SHRINKAGE - BASE_SHRINKAGE) * disagreement_factor * proximity_factor
                
                b.model_prob = (1.0 - shrinkage) * b.model_prob + shrinkage * b.market_prob
        
        # Re-normalize after blend
        total_after_blend = sum(b.model_prob for b in range_bins)
        if total_after_blend > 0:
            for b in range_bins:
                b.model_prob = b.model_prob / total_after_blend
    
    # ========== STEP 2.7: C→F ROUNDING WIDTH ADJUSTMENT (v1.5) ==========
    # NWS reports in whole °F after converting from °C. Due to rounding:
    #   - Some °F values can only come from 1 °C reading (exact: 32,41,50,59,68,77,86,95)
    #   - Other °F values can come from 2 possible °C readings
    # A Kalshi bin like "76-77°F" that contains 77 (exact) is effectively 
    # narrower than "78-79°F" (both values come from rounding).
    # We adjust by slightly boosting bins where ALL values are "rounded" (2°F effective)
    # and slightly penalizing bins that contain an "exact" pinch point.
    
    for b in range_bins:
        if b.low is None or b.high is None:
            continue
        f_values = list(range(b.low, b.high + 1))
        has_exact = any(f in EXACT_F_VALUES for f in f_values)
        if has_exact:
            b.model_prob *= CF_WIDTH_PENALTY
        else:
            b.model_prob *= CF_WIDTH_BOOST
    
    # Final re-normalize
    total_cf_adj = sum(b.model_prob for b in range_bins)
    if total_cf_adj > 0:
        for b in range_bins:
            b.model_prob = b.model_prob / total_cf_adj
    
    # ========== STEP 3: DERIVE below/above from normalized range bins ==========
    sorted_ranges = sorted(range_bins, key=lambda b: b.low)
    
    for below_bin in below_bins:
        threshold = below_bin.threshold
        derived_prob = sum(
            b.model_prob for b in sorted_ranges 
            if b.high <= threshold
        )
        for b in sorted_ranges:
            if b.low <= threshold < b.high:
                bin_width = b.high - b.low + 1
                below_fraction = (threshold - b.low + 1) / bin_width
                derived_prob += b.model_prob * below_fraction
                break
        
        below_bin.model_prob_raw = derived_prob
        below_bin.model_prob = derived_prob
    
    for above_bin in above_bins:
        threshold = above_bin.threshold
        derived_prob = sum(
            b.model_prob for b in sorted_ranges 
            if b.low >= threshold
        )
        for b in sorted_ranges:
            if b.low < threshold <= b.high:
                bin_width = b.high - b.low + 1
                above_fraction = (b.high - threshold + 1) / bin_width
                derived_prob += b.model_prob * above_fraction
                break
        
        above_bin.model_prob_raw = derived_prob
        above_bin.model_prob = derived_prob
    
    # ========== STEP 4: Apply calibration and city multiplier ==========
    city_conf_mult = CITY_CONFIDENCE_MULTIPLIER.get(city_key, 1.0)
    
    for bin_info in bins:
        calibrated = apply_calibration(bin_info.model_prob)
        calibrated = calibrated * city_conf_mult
        calibrated = max(min(calibrated, 0.95), 0.01)
        bin_info.calibrated_prob = calibrated
        
        # Calculate edge with spread adjustment
        bin_info.edge_raw = bin_info.calibrated_prob - bin_info.market_prob
        spread_penalty = bin_info.spread * SPREAD_PENALTY_FACTOR
        bin_info.edge_spread_adjusted = bin_info.edge_raw - spread_penalty
    
    # ========== STEP 5: Sort and rank ==========
    sorted_bins = sorted(bins, key=lambda b: b.calibrated_prob, reverse=True)
    for i, b in enumerate(sorted_bins):
        b.rank = i + 1
    
    # ========== STEP 6: Calculate distribution metrics ==========
    entropy_threshold = CITY_ENTROPY_THRESHOLD.get(city_key, DEFAULT_ENTROPY_THRESHOLD)
    
    if len(range_bins) >= 2:
        range_probs = [b.model_prob for b in range_bins]
        dist_entropy = scipy_entropy(range_probs)
        
        sorted_range = sorted(range_bins, key=lambda b: b.calibrated_prob, reverse=True)
        top_bin, second_bin = sorted_range[0], sorted_range[1]
        confidence_gap = top_bin.calibrated_prob - second_bin.calibrated_prob
    else:
        dist_entropy = 0.0
        confidence_gap = 0.0
        top_bin = sorted_bins[0] if sorted_bins else None
        second_bin = sorted_bins[1] if len(sorted_bins) > 1 else None
    
    return BinDistribution(
        bins=sorted_bins, 
        entropy=dist_entropy, 
        entropy_threshold=entropy_threshold,
        confidence_gap=confidence_gap,
        top_bin=top_bin, 
        second_bin=second_bin, 
        latent_forecast=latent_forecast,
        uncertainty=effective_sigma, 
        model_spread=model_spread, 
        agreement_level=agreement_level,
        total_prob_before_norm=total_range_prob
    )


# ============================================================================
#              TRADE SELECTION
# ============================================================================

def calculate_kelly_bet(edge: float, model_prob: float, market_prob: float, 
                        bankroll: float, remaining_city_budget: float) -> Tuple[float, dict]:
    """
    Kelly with city exposure limit — v1.4 FIXED.
    Now returns diagnostics so you can see WHY the bet is sized the way it is.
    Uses edge directly instead of re-deriving from model_prob.
    """
    diagnostics = {"full_kelly": 0.0, "fractional": 0.0, "capped": 0.0, "reason": ""}
    
    if edge <= 0 or model_prob <= 0 or model_prob >= 1 or market_prob <= 0 or market_prob >= 1:
        diagnostics["reason"] = "no_edge"
        return 0.0, diagnostics
    
    # Kelly formula: f* = (p*b - q) / b where b = odds, p = win prob, q = 1-p
    b = (1 - market_prob) / market_prob  # decimal odds for YES
    p, q = model_prob, 1 - model_prob
    full_kelly = (p * b - q) / b if b > 0 else 0
    diagnostics["full_kelly"] = full_kelly
    
    fractional_kelly = max(0, full_kelly * KELLY_FRACTION)
    diagnostics["fractional"] = fractional_kelly
    
    # Cap at MAX_BET_FRACTION of bankroll
    capped = min(fractional_kelly, MAX_BET_FRACTION)
    diagnostics["capped"] = capped
    
    if capped < fractional_kelly:
        diagnostics["reason"] = "capped_at_max"
    else:
        diagnostics["reason"] = "kelly_sized"
    
    bet_size = bankroll * capped
    bet_size = min(bet_size, remaining_city_budget)
    
    if bet_size < MIN_BET_SIZE:
        diagnostics["reason"] = "below_minimum"
        return 0.0, diagnostics
    
    return round(bet_size, 2), diagnostics


def select_trades(city_key: str, distribution: BinDistribution, 
                  bankroll: float, city_exposure_used: float = 0.0) -> List[TradeRecommendation]:
    """
    Select trades — v1.4 FIXED.
    - NO bets no longer double-penalized for spread
    - Kelly calls updated to use diagnostics
    """
    
    city = CITIES[city_key]
    trades = []
    
    if distribution.entropy > distribution.entropy_threshold:
        print(f"     ⚠️ Skipping: entropy {distribution.entropy:.2f} > {distribution.entropy_threshold:.2f}")
        return []
    
    max_city_exposure = bankroll * MAX_CITY_EXPOSURE_FRACTION
    remaining_city_budget = max_city_exposure - city_exposure_used
    
    if remaining_city_budget < MIN_BET_SIZE:
        print(f"     ⚠️ City exposure limit reached")
        return []
    
    for bin_info in distribution.bins:
        if bin_info.spread_warning:
            continue
        
        # Spread penalty — calculated ONCE, used for both YES and NO
        spread_penalty = bin_info.spread * SPREAD_PENALTY_FACTOR
        
        # === YES BET EVALUATION ===
        raw_edge = bin_info.calibrated_prob - bin_info.market_prob - spread_penalty
        
        if raw_edge > 0:
            adjustments = []
            adjusted_edge = raw_edge
            
            if spread_penalty > 0.005:
                adjustments.append(f"spread_penalty: -{spread_penalty:.1%}")
            
            bin_mult = BIN_TYPE_MULTIPLIER.get(bin_info.bin_type, 1.0)
            adjusted_edge *= bin_mult
            if bin_mult != 1.0:
                adjustments.append(f"bin_type: {bin_mult:.2f}x")
            
            adjusted_edge *= YES_BET_PENALTY_MULTIPLIER
            adjustments.append(f"yes_penalty: {YES_BET_PENALTY_MULTIPLIER:.2f}x")
            
            bin_info.adjusted_edge = adjusted_edge
            
            if adjusted_edge >= MIN_EDGE_YES_BET:
                if bin_info.market_prob < MIN_CONTRACT_PRICE or bin_info.market_prob > MAX_CONTRACT_PRICE:
                    continue
                if bin_info.rank == 1 and distribution.confidence_gap < MIN_CONFIDENCE_GAP:
                    continue
                
                bet_size, kelly_diag = calculate_kelly_bet(
                    adjusted_edge, bin_info.calibrated_prob, bin_info.market_prob,
                    bankroll, remaining_city_budget
                )
                
                if bet_size > 0:
                    remaining_city_budget -= bet_size
                    potential_profit = bet_size * (1 / bin_info.market_prob - 1)
                    adjustments.append(f"kelly: {kelly_diag['reason']} (full={kelly_diag['full_kelly']:.3f})")
                    trades.append(TradeRecommendation(
                        city=city["name"], city_key=city_key, bin_info=bin_info, side="YES",
                        model_prob=bin_info.calibrated_prob, market_prob=bin_info.market_prob,
                        raw_edge=raw_edge, adjusted_edge=adjusted_edge, bet_size=bet_size,
                        potential_profit=round(potential_profit, 2), confidence_rank=bin_info.rank,
                        adjustments_applied=adjustments,
                        spread_penalty=spread_penalty
                    ))
        
        # === NO BET EVALUATION — v1.4: spread penalty applied ONCE (not double) ===
        no_calibrated_prob = 1 - bin_info.calibrated_prob
        no_market_prob = 1 - bin_info.market_prob
        # Edge = our NO prob - market NO prob - spread (SINGLE application)
        raw_no_edge = no_calibrated_prob - no_market_prob - spread_penalty
        
        if raw_no_edge > 0:
            adjustments = []
            adjusted_no_edge = raw_no_edge
            
            if spread_penalty > 0.005:
                adjustments.append(f"spread_penalty: -{spread_penalty:.1%}")
            
            bin_mult = BIN_TYPE_MULTIPLIER.get(bin_info.bin_type, 1.0)
            adjusted_no_edge *= bin_mult
            if bin_mult != 1.0:
                adjustments.append(f"bin_type: {bin_mult:.2f}x")
            
            adjusted_no_edge *= NO_BET_PREFERENCE_MULTIPLIER
            adjustments.append(f"no_boost: {NO_BET_PREFERENCE_MULTIPLIER:.2f}x")
            
            if adjusted_no_edge >= MIN_EDGE_THRESHOLD:
                if no_market_prob < MIN_CONTRACT_PRICE or no_market_prob > MAX_CONTRACT_PRICE:
                    continue
                if bin_info.rank <= 2:
                    continue
                
                bet_size, kelly_diag = calculate_kelly_bet(
                    adjusted_no_edge, no_calibrated_prob, no_market_prob,
                    bankroll, remaining_city_budget
                )
                
                if bet_size > 0:
                    remaining_city_budget -= bet_size
                    potential_profit = bet_size * (1 / no_market_prob - 1)
                    adjustments.append(f"kelly: {kelly_diag['reason']} (full={kelly_diag['full_kelly']:.3f})")
                    trades.append(TradeRecommendation(
                        city=city["name"], city_key=city_key, bin_info=bin_info, side="NO",
                        model_prob=no_calibrated_prob, market_prob=no_market_prob,
                        raw_edge=raw_no_edge, adjusted_edge=adjusted_no_edge, bet_size=bet_size,
                        potential_profit=round(potential_profit, 2), confidence_rank=bin_info.rank,
                        adjustments_applied=adjustments,
                        spread_penalty=spread_penalty
                    ))
    
    return trades


def smart_select_bets(all_trades: List[TradeRecommendation]) -> List[TradeRecommendation]:
    if not all_trades:
        return []
    
    sorted_trades = sorted(all_trades, key=lambda t: t.adjusted_edge, reverse=True)
    
    selected = []
    city_counts = {}
    city_exposure = {}
    
    for trade in sorted_trades:
        if len(selected) >= MAX_TOTAL_BETS:
            break
        
        city = trade.city_key
        
        if city_counts.get(city, 0) >= MAX_BETS_PER_CITY:
            continue
        
        selected.append(trade)
        city_counts[city] = city_counts.get(city, 0) + 1
        city_exposure[city] = city_exposure.get(city, 0) + trade.bet_size
    
    return selected


# ============================================================================
#                    DISPLAY FUNCTIONS
# ============================================================================

def print_header(target_date):
    print("=" * 70)
    print("🎯 KALSHI BIN-BASED PREDICTION MODEL v1.5")
    print("   (ECMWF + ICON + NWS + HRRR Ensemble)")
    print("   Station Precision + Adjacent Bin Floor + C→F Rounding")
    print("=" * 70)
    print(f"\n📅 Target Date: {target_date.strftime('%A, %B %d, %Y')}")
    print(f"🔧 Key Settings:")
    print(f"   • Min Edge (NO): {MIN_EDGE_THRESHOLD*100:.0f}% | Min Edge (YES): {MIN_EDGE_YES_BET*100:.0f}%")
    print(f"   • Max Spread: {MAX_SPREAD_THRESHOLD*100:.0f}% | Spread Penalty: {SPREAD_PENALTY_FACTOR*100:.0f}%")
    print(f"   • Max City Exposure: {MAX_CITY_EXPOSURE_FRACTION*100:.0f}% of bankroll")
    print(f"   • Model Weights: NWS={MODEL_WEIGHTS['NWS']}, HRRR={MODEL_WEIGHTS['HRRR']}, ECMWF={MODEL_WEIGHTS['ECMWF']}, ICON={MODEL_WEIGHTS['ICON']}")
    print(f"   • Adjacent Bin Floors: ±1=12%, ±2=5%, ±3=2% | Market Blend: 15-40%")
    print(f"   • C→F Rounding: exact pinch={CF_WIDTH_PENALTY:.0%}, rounded boost={CF_WIDTH_BOOST:.0%}")
    print("=" * 70)


def print_forecasts_only(city_key: str, forecasts: List[Forecast]):
    """Print city forecast info without bin distribution (when markets aren't available)."""
    city = CITIES[city_key]
    city_mult = CITY_CONFIDENCE_MULTIPLIER.get(city_key, 1.0)
    bias = STATION_BIAS.get(city_key, 0.0)

    emoji = "✅" if city_mult >= 1.0 else "⚠️"
    print(f"\n{'─'*70}")
    print(f"📍 {city['name'].upper()} {emoji} (conf: {city_mult:.2f}x)")
    print(f"{'─'*70}")

    print(f"\n  📡 FORECASTS (bias: {bias:+.1f}°F + warm: {STATION_WARM_BIAS_SUNNY.get(city_key, 0):+.1f}°F):")
    adjusted_temps = []
    for f in forecasts:
        total_bias = bias + STATION_WARM_BIAS_SUNNY.get(city_key, 0)
        adjusted = f.temperature + total_bias
        adjusted_temps.append((f, adjusted))
        marker = "⭐" if f.source == "NWS" else ""
        print(f"      {f.source:<12} {f.temperature:>5.1f}°F → {adjusted:>5.1f}°F {marker}")

    # Simple weighted average for latent forecast
    total_weight = sum(f.weight for f, _ in adjusted_temps)
    if total_weight > 0:
        latent = sum(adj * f.weight for f, adj in adjusted_temps) / total_weight
    else:
        latent = adjusted_temps[0][1] if adjusted_temps else 0

    temps = [adj for _, adj in adjusted_temps]
    model_spread = max(temps) - min(temps) if len(temps) >= 2 else 0
    base_sigma = CITY_BASE_UNCERTAINTY.get(city_key, 3.0)
    spread_sigma = model_spread / 2.0
    if model_spread <= 2.0:
        uncertainty_mult = 0.85
    elif model_spread <= 4.0:
        uncertainty_mult = 1.0
    else:
        uncertainty_mult = 1.25
    effective_sigma = (base_sigma**2 + spread_sigma**2)**0.5 * uncertainty_mult

    print(f"\n  🎯 LATENT FORECAST: {latent:.1f}°F")
    print(f"     Uncertainty: {effective_sigma:.1f}°F | Model Spread: {model_spread:.1f}°F")
    print(f"\n  ⚠️ No market data available — bin distribution not computed")


def print_distribution(city_key: str, distribution: BinDistribution, forecasts: List[Forecast]):
    city = CITIES[city_key]
    city_mult = CITY_CONFIDENCE_MULTIPLIER.get(city_key, 1.0)
    
    print(f"\n{'─'*70}")
    emoji = "✅" if city_mult >= 1.0 else "⚠️"
    print(f"📍 {city['name'].upper()} {emoji} (conf: {city_mult:.2f}x)")
    print(f"{'─'*70}")
    
    print(f"\n  📡 FORECASTS (bias: {STATION_BIAS.get(city_key, 0):+.1f}°F + warm: {STATION_WARM_BIAS_SUNNY.get(city_key, 0):+.1f}°F):")
    for f in forecasts:
        total_bias = STATION_BIAS.get(city_key, 0) + STATION_WARM_BIAS_SUNNY.get(city_key, 0)
        adjusted = f.temperature + total_bias
        marker = "⭐" if f.source == "NWS" else ""
        print(f"      {f.source:<12} {f.temperature:>5.1f}°F → {adjusted:>5.1f}°F {marker}")
    
    print(f"\n  🎯 LATENT FORECAST: {distribution.latent_forecast:.1f}°F")
    print(f"     Uncertainty: {distribution.uncertainty:.1f}°F | Model Spread: {distribution.model_spread:.1f}°F")
    
    print(f"\n  📊 DISTRIBUTION METRICS:")
    entropy_status = "⚠️ HIGH" if distribution.entropy > distribution.entropy_threshold else "✅"
    print(f"     Entropy: {distribution.entropy:.2f} (threshold: {distribution.entropy_threshold:.1f}) {entropy_status}")
    print(f"     Confidence Gap: {distribution.confidence_gap*100:.1f}%")
    print(f"     Pre-norm total: {distribution.total_prob_before_norm:.2f} → normalized to 1.00")
    
    print(f"\n  📈 BIN DISTRIBUTION (top 10):")
    print(f"     {'Rank':<4} {'Bin':<16} {'Raw':>6} {'Norm':>6} {'Cal':>6} {'Mkt':>6} {'Sprd':>5} {'Edge':>7}")
    print(f"     {'-'*62}")
    
    for bin_info in distribution.bins[:10]:
        if bin_info.bin_type == "range":
            bin_label = f"{bin_info.low}°-{bin_info.high}°F"
        elif bin_info.bin_type == "below":
            bin_label = f"≤{bin_info.threshold}°F"
        else:
            bin_label = f"≥{bin_info.threshold}°F"
        
        spread_warn = "⚠️" if bin_info.spread_warning else ""
        marker = "◀TOP" if bin_info.rank == 1 else ("💰" if bin_info.edge_spread_adjusted >= MIN_EDGE_THRESHOLD else "")
        
        print(f"     {bin_info.rank:<4} {bin_label:<16} {bin_info.model_prob_raw:>5.1%} "
              f"{bin_info.model_prob:>5.1%} {bin_info.calibrated_prob:>5.1%} "
              f"{bin_info.market_prob:>5.1%} {bin_info.spread:>4.0%}{spread_warn} "
              f"{bin_info.edge_spread_adjusted:>+6.1%} {marker}")


def print_recommendations(trades: List[TradeRecommendation], bankroll: float):
    print(f"\n{'='*70}")
    print("💰 BETTING RECOMMENDATIONS (v1.5 — station precision + bin floors + C→F)")
    print(f"{'='*70}")
    
    if not trades:
        print("\n  🛑 NO BETS RECOMMENDED")
        return
    
    total_bet = sum(t.bet_size for t in trades)
    total_potential = sum(t.potential_profit for t in trades)
    
    city_exposure = {}
    for t in trades:
        city_exposure[t.city] = city_exposure.get(t.city, 0) + t.bet_size
    
    print(f"\n  💵 Bankroll: ${bankroll:.2f}")
    print(f"  📊 Total Wagered: ${total_bet:.2f} ({total_bet/bankroll*100:.1f}%)")
    print(f"  🏙️  City Exposure: {', '.join(f'{c}: ${e:.2f}' for c, e in city_exposure.items())}")
    print(f"  🎯 Total Potential: ${total_potential:.2f}")
    
    for i, trade in enumerate(trades, 1):
        bin_info = trade.bin_info
        if bin_info.bin_type == "range":
            bin_label = f"{bin_info.low}°-{bin_info.high}°F"
        elif bin_info.bin_type == "below":
            bin_label = f"≤{bin_info.threshold}°F"
        else:
            bin_label = f"≥{bin_info.threshold}°F"
        
        side_emoji = "🔴" if trade.side == "NO" else "🟢"
        
        print(f"\n  ┌─ BET #{i}: {trade.city} {side_emoji}")
        print(f"  │  Contract: {bin_label}")
        print(f"  │  Side: {trade.side} (rank #{trade.confidence_rank})")
        print(f"  │  Model: {trade.model_prob*100:.1f}% | Market: {trade.market_prob*100:.1f}%")
        print(f"  │  Raw Edge: {trade.raw_edge*100:+.1f}% → Adjusted: {trade.adjusted_edge*100:+.1f}%")
        if trade.spread_penalty > 0.01:
            print(f"  │  Spread Penalty: -{trade.spread_penalty*100:.1f}%")
        print(f"  │  Adjustments: {', '.join(trade.adjustments_applied)}")
        print(f"  │  💵 BET: ${trade.bet_size:.2f} → Potential: ${trade.potential_profit:.2f}")
        print(f"  └{'─'*50}")


# ============================================================================
#                    MAIN
# ============================================================================

def analyze_city(city_key: str, target_date, bankroll: float) -> Tuple[Optional[BinDistribution], List[TradeRecommendation]]:
    city = CITIES[city_key]
    
    forecasts = fetch_all_forecasts(city_key, target_date)
    if not forecasts:
        print(f"  ❌ No forecasts available")
        return None, []
    
    print(f"\n  💰 Fetching Kalshi markets...")
    markets = fetch_kalshi_markets(city["kalshi_series"], target_date)
    if not markets:
        print(f"     ❌ No markets found")
        print_forecasts_only(city_key, forecasts)
        return None, []
    print(f"     ✅ Found {len(markets)} contracts")

    bins = parse_kalshi_bins(markets)
    if not bins:
        print(f"     ❌ Could not parse bins")
        print_forecasts_only(city_key, forecasts)
        return None, []
    
    spread_warnings = sum(1 for b in bins if b.spread_warning)
    if spread_warnings > 0:
        print(f"     ⚠️ {spread_warnings} bins have wide spreads (>{MAX_SPREAD_THRESHOLD*100:.0f}%)")
    
    distribution = calculate_bin_distribution(city_key, forecasts, bins)
    print_distribution(city_key, distribution, forecasts)
    
    trades = select_trades(city_key, distribution, bankroll)
    
    return distribution, trades


def main():
    parser = argparse.ArgumentParser(description="Kalshi Bin Model v1.5")
    parser.add_argument("--bankroll", type=float, default=STARTING_BANKROLL)
    parser.add_argument("--cities", type=str, default=",".join(SELECTED_CITIES))
    parser.add_argument("--today", action="store_true")
    parser.add_argument("--tomorrow", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.cities.lower() == "all":
        selected = list(CITIES.keys())
    else:
        selected = [c.strip().lower() for c in args.cities.split(",")]
        selected = [c for c in selected if c in CITIES]
    if not selected:
        selected = SELECTED_CITIES
    
    now = datetime.now()
    if args.today:
        target_date = now.date()
    elif args.tomorrow:
        target_date = now.date() + timedelta(days=1)
    elif args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            target_date = now.date() + timedelta(days=1)
    else:
        target_date = now.date() + timedelta(days=1) if now.hour >= 6 else now.date()
    
    print_header(target_date)
    
    all_trades = []
    for city_key in selected:
        _, trades = analyze_city(city_key, target_date, args.bankroll)
        all_trades.extend(trades)
    
    final_trades = smart_select_bets(all_trades)
    print_recommendations(final_trades, args.bankroll)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE (v1.5 - Station Precision + Bin Floors + C→F Rounding)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()