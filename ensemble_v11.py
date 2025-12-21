"""
KALSHI WEATHER BETTING MODEL v11
================================
ENHANCED FORECAST VERSION with NWS Integration & Disagreement Detection.

Key improvements over v9:
1. NWS FORECAST: Added National Weather Service as third forecast source
2. RECENT HRRR RUNS: Uses most recent available HRRR run (not just 00Z/12Z)
3. MAJOR DISAGREEMENT DETECTION: Flags when models differ >5°F (frontal systems)
4. SMART WEIGHTING: Adjusts ensemble based on model agreement/disagreement

All v9 features maintained:
- Smart bet selection (different cities vs same city)
- Kelly Criterion sizing with bankroll tracking
- Price filtering and edge requirements
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.model_selection import cross_val_score # type: ignore
import requests # type: ignore
from datetime import datetime, timedelta, timezone
import re
from scipy import stats # type: ignore
import glob
import argparse
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

from pathlib import Path
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
SAME_CITY_MULTI_BET_THRESHOLD = 2.0   # Need 2x the minimum edge to stack same-city bets
MAX_BETS_PER_CITY = 2                  # Never more than 2 bets in same city
MAX_TOTAL_BETS_PER_DAY = 6             # Cap total daily bets across all cities
SUPER_CONFIDENT_EDGE_RATIO = 2.5       # What counts as "super confident"

# ============ TIMEZONE CONFIGURATION ============
KALSHI_TIMEZONE = ZoneInfo("America/New_York")
EVENING_CUTOFF_HOUR = 18  # 6 PM Eastern

# ============ KELLY CRITERION CONFIGURATION ============
STARTING_BANKROLL = 100        # Your bankroll
KELLY_FRACTION = 0.75         # Half Kelly (balanced risk/reward)
MIN_BET_SIZE = 0.50           # Don't bet less than 50¢
MAX_BET_FRACTION = 0.15       # Never bet more than 15% of bankroll

# ============ ENSEMBLE CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - models should agree within this
CONFIDENCE_BOOST_THRESHOLD = 2.0    # °F - boost confidence if within this
CALIBRATED_FORECAST_STD = 2.8       # °F - typical day-ahead forecast error
MAJOR_DISAGREEMENT_THRESHOLD = 5.0  # °F - flag as major disagreement (frontal systems)


# ============ ATMOSPHERIC ANALOG MODEL CONFIGURATION ============
ATMOSPHERIC_MODEL_WEIGHT = 0.15     # How much weight to give atmospheric analog prediction (reduced from 0.25)
MIN_TRAINING_SAMPLES = 30           # Minimum samples needed to train atmospheric model
ATMOSPHERIC_MODEL_PATH = "atmospheric_models"  # Directory to save/load models

# ============ AIRMASS & WIND ANALYSIS CONFIGURATION ============
WARM_ADVECTION_THRESHOLD = 15.0     # °F - significant warm airmass change
COLD_ADVECTION_THRESHOLD = -15.0    # °F - significant cold airmass change
WIND_SHIFT_THRESHOLD = 90.0         # degrees - significant wind direction change

# Edge requirements by price bucket
BASE_EDGE_REQUIREMENTS = {
    "sweet_spot": 0.12,    # 12% edge for 15-50¢ contracts
    "high_price": 0.12,    # 12% edge for 50-90¢ contracts
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
    Calculate optimal bet size using Kelly Criterion.
    
    Returns:
        bet_size: Dollar amount to bet
        kelly_info: Dict with calculation details
    """
    if bet_price <= 0 or bet_price >= 1:
        return 0, {}
    
    # Odds received: profit per $1 wagered if win
    b = (1 / bet_price) - 1
    
    p = our_prob  # Probability of winning
    q = 1 - p     # Probability of losing
    
    # Full Kelly formula
    full_kelly = (p * b - q) / b if b > 0 else 0
    
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
        "odds": b,
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
        "utc_offset": -6
    },
    "nyc": {
        "name": "New York City",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.7128,
        "lon": -74.0060,
        "kalshi_series": "KXHIGHNY",
        "timezone": "America/New_York",
        "utc_offset": -5
    },
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.7959,
        "lon": -80.2870,
        "kalshi_series": "KXHIGHMIA",
        "timezone": "America/New_York",
        "utc_offset": -5
    }
}

# Will collect all qualifying bets across all cities
all_qualifying_bets = []

# Default to all cities
selected_cities = ["chicago", "nyc", "miami"]


# ============ TIMEZONE HELPERS ============
def get_eastern_now():
    """Get the current datetime in US Eastern time."""
    return datetime.now(KALSHI_TIMEZONE)

def get_target_date(force_today=False):
    """
    Get the target date for market lookup.

    By default:
    - Before EVENING_CUTOFF_HOUR Eastern: look at tomorrow's markets
    - After EVENING_CUTOFF_HOUR Eastern: look at today's markets

    Args:
        force_today: If True, always look at today's markets

    Returns:
        target_date: The date to look up markets for
        is_today: Whether we're looking at today's markets
    """
    eastern_now = get_eastern_now()
    eastern_today = eastern_now.date()

    if force_today:
        return eastern_today, True

    # After evening cutoff, look at today's markets for last-minute bets
    if eastern_now.hour >= EVENING_CUTOFF_HOUR:
        return eastern_today, True
    else:
        return eastern_today + timedelta(days=1), False

# ============ HEADER ============
def print_header(target_date, is_today):
    print("=" * 70)
    print("KALSHI WEATHER BETTING MODEL v11")
    print("(Atmospheric Analog-Enhanced Forecasts)")
    print("=" * 70)

    eastern_now = get_eastern_now()
    local_now = datetime.now()
    print(f"\nYour local time: {local_now.strftime('%Y-%m-%d %H:%M')}")
    print(f"US Eastern time: {eastern_now.strftime('%Y-%m-%d %H:%M')}")

    if is_today:
        print(f"Looking at: TODAY's markets ({target_date.strftime('%Y-%m-%d')})")
    else:
        print(f"Looking at: TOMORROW's markets ({target_date.strftime('%Y-%m-%d')})")

    print(f"\nAnalyzing: Chicago, New York City, Miami")
    print(f"Agreement Threshold: ±{ENSEMBLE_AGREEMENT_THRESHOLD}°F")
    print(f"Major Disagreement Flag: >{MAJOR_DISAGREEMENT_THRESHOLD}°F")
    print(f"Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")
    print(f"Sweet Spot: {SWEET_SPOT_LOW*100:.0f}¢ - {SWEET_SPOT_HIGH*100:.0f}¢")
    print(f"\n🎯 SMART BET SELECTION:")
    print("   • Different cities: Bet freely (uncorrelated)")
    print(f"   • Same city: Only stack if edge ratio ≥ {SAME_CITY_MULTI_BET_THRESHOLD}x")
    print(f"   • Max {MAX_BETS_PER_CITY} bets/city, {MAX_TOTAL_BETS_PER_DAY} total/day")
    print(f"\n🤖 ML ENHANCEMENTS:")
    print("   • Wind advection analysis (warm/cold air masses)")
    print("   • 850mb temperature tracking (airmass changes)")
    print("   • Dynamic forecast adjustment based on atmospheric signals")
    print(f"\n💰 BANKROLL: ${STARTING_BANKROLL:.2f} | {KELLY_FRACTION:.0%} Kelly")
    print("Data Sources: NOAA HRRR (3km) + Open-Meteo + NWS")


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
    
    NWS is excellent at synoptic-scale patterns (frontal systems).
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
                # Extract temperature
                temp = period.get("temperature")
                if temp:
                    return float(temp)
        
        return None
        
    except Exception as e:
        print(f"     ⚠️ NWS API error: {e}")
        return None


def get_most_recent_hrrr_run():
    """Get the most recent available HRRR model run using v11's proven logic."""
    if not HERBIE_AVAILABLE:
        return None, None
    
    today = datetime.now().date()
    current_hour = datetime.now().hour
    
    if current_hour >= 14:
        return today, 12
    elif current_hour >= 2:
        return today, 0
    else:
        return today - timedelta(days=1), 12


def fetch_hrrr_forecast_fixed(lat, lon, target_date, utc_offset):
    """Fetch HRRR forecast with FIXED sampling to get actual daily high."""
    if not HERBIE_AVAILABLE:
        return None, None
    
    # Use most recent available run
    model_run_date, model_run_hour = get_most_recent_hrrr_run()
    if not model_run_date:
        return None, None
    
    model_run_str = f"{model_run_date.strftime('%Y-%m-%d')} {model_run_hour:02d}:00"
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    
    target_12_local_utc = 12 - utc_offset
    # Expand window: start at 6 AM local, sample through 8 PM local (14 hours)
    target_start_hour = 6  # 6 AM local
    target_start_utc = (target_start_hour - utc_offset) % 24
    target_start = datetime.combine(target_date, datetime.min.time().replace(hour=target_start_utc))
    if (target_start_hour - utc_offset) >= 24:
        target_start += timedelta(days=1)
    elif (target_start_hour - utc_offset) < 0:
        target_start -= timedelta(days=1)
    
    hours_to_start = int((target_start - model_run_datetime).total_seconds() / 3600)
    # Sample 14 hours (6 AM to 8 PM local) instead of 7 hours
    forecast_hours = list(range(max(1, hours_to_start), min(48, hours_to_start + 14)))
    if not forecast_hours:
        print(f"     ⚠️ HRRR forecast not available yet (need F{hours_to_start}+)")
        return None, None
    
    # Calculate the actual time range being sampled
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    first_sample_time = model_run_datetime + timedelta(hours=forecast_hours[0])
    last_sample_time = model_run_datetime + timedelta(hours=forecast_hours[-1])
    
    # Convert to local time for display (utc_offset is hours behind UTC)
    try:
        from zoneinfo import ZoneInfo
        # Determine timezone based on UTC offset
        if utc_offset == -6:
            local_tz = ZoneInfo("America/Chicago")
            tz_name = "CT"
        elif utc_offset == -5:
            local_tz = ZoneInfo("America/New_York")
            tz_name = "ET"
        else:
            # Default to UTC if unknown
            local_tz = timezone.utc
            tz_name = "UTC"
        
        first_local = first_sample_time.replace(tzinfo=timezone.utc).astimezone(local_tz)
        last_local = last_sample_time.replace(tzinfo=timezone.utc).astimezone(local_tz)
        
        time_frame = f"{first_local.strftime('%I:%M %p')} - {last_local.strftime('%I:%M %p')} {tz_name}"
        date_display = first_local.strftime('%b %d')
        
        print(f"     HRRR Model run: {model_run_str} UTC (most recent)")
        print(f"     📅 HRRR Time Frame: {date_display}, {time_frame} (expanded window)")
        print(f"     Sampling forecast hours: F{forecast_hours[0]}-F{forecast_hours[-1]}")
    except:
        # Fallback if timezone conversion fails
        print(f"     HRRR Model run: {model_run_str} UTC (most recent)")
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
    
    # Fetch 850mb data for airmass analysis
    temps_850mb = fetch_hrrr_850mb_temp(lat, lon, target_date, utc_offset)
    
    weather_data = {
        'forecast_high': forecast_high,
        'all_temps': temperatures,
        'wind_speed': avg_wind_speed,
        'wind_dir': avg_wind_dir,
        'temps_850mb': temps_850mb,
        'model_run': model_run_str,
        'source': 'HRRR'
    }
    
    return forecast_high, weather_data


def get_bin_for_temp(temp):
    """
    Get the Kalshi bin that contains this temperature.
    
    Kalshi uses 2-degree bins with ODD lower bounds:
    ..., 27-28, 29-30, 31-32, 33-34, ...
    
    Examples:
        27.5°F → bin (27, 28)
        29.0°F → bin (29, 30)
        29.9°F → bin (29, 30)
        30.0°F → bin (29, 30)  # 30 is the upper bound of 29-30
        30.1°F → bin (31, 32)  # Just above 30 goes to next bin
    """
    # Floor to get the integer part
    temp_floor = int(np.floor(temp))
    
    # Find the odd number that starts the bin containing this temp
    # If temp_floor is odd, that's our lower bound
    # If temp_floor is even, the bin started at temp_floor - 1
    if temp_floor % 2 == 1:  # odd
        lower = temp_floor
    else:  # even
        lower = temp_floor - 1
    
    return (lower, lower + 1)


# ============ WIND & AIRMASS ANALYSIS ============

def analyze_wind_direction(wind_dir):
    """
    Classify wind direction into cardinal directions and advection type.
    
    Args:
        wind_dir: Wind direction in degrees (0-360, where 0/360 is north)
    
    Returns:
        dict with cardinal direction and advection type
    """
    if wind_dir is None:
        return None

    # Normalize to 0-360
    wind_dir = wind_dir % 360

    # Determine cardinal direction
    if wind_dir >= 337.5 or wind_dir < 22.5:
        cardinal = "N"
        advection_type = "cold"
    elif 22.5 <= wind_dir < 67.5:
        cardinal = "NE"
        advection_type = "cold"
    elif 67.5 <= wind_dir < 112.5:
        cardinal = "E"
        advection_type = "neutral"
    elif 112.5 <= wind_dir < 157.5:
        cardinal = "SE"
        advection_type = "warm"
    elif 157.5 <= wind_dir < 202.5:
        cardinal = "S"
        advection_type = "warm"
    elif 202.5 <= wind_dir < 247.5:
        cardinal = "SW"
        advection_type = "warm"
    elif 247.5 <= wind_dir < 292.5:
        cardinal = "W"
        advection_type = "neutral"
    else:
        cardinal = "NW"
        advection_type = "cold"
    return {
        "degrees": wind_dir,
        "cardinal": cardinal,
        "advection_type": advection_type
    }


def fetch_hrrr_850mb_temp(lat, lon, target_date, utc_offset):
    """
    Fetch 850mb (airmass level) temperature from HRRR.
    
    850mb temperature is a better indicator of the airmass than surface temps.
    A change in 850mb temp indicates a true airmass change (frontal passage).
    
    Returns:
        dict with current 850mb temp and 24hr change
    """
    if not HERBIE_AVAILABLE:
        return None

    model_run_date, model_run_hour = get_most_recent_hrrr_run()
    if not model_run_date:
        return None

    model_run_str = f"{model_run_date.strftime('%Y-%m-%d')} {model_run_hour:02d}:00"
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))

    # Calculate target time (local noon)
    target_12_local_utc = 12 - utc_offset
    target_noon = datetime.combine(target_date, datetime.min.time().replace(hour=target_12_local_utc % 24))
    if target_12_local_utc >= 24:
        target_noon += timedelta(days=1)

    hours_to_noon = int((target_noon - model_run_datetime).total_seconds() / 3600)

    # Skip if out of range
    if hours_to_noon < 1 or hours_to_noon > 47:
        return None

    try:
        return _extracted_from_fetch_hrrr_850mb_temp_35(
            model_run_str, hours_to_noon, lon, lat
        )
    except Exception as e:
        print(f"     ⚠️ 850mb fetch error: {e}")
        return None


# TODO Rename this here and in `fetch_hrrr_850mb_temp`
def _extracted_from_fetch_hrrr_850mb_temp_35(model_run_str, hours_to_noon, lon, lat):
    # Fetch 850mb temp at target time
    H_target = Herbie(
        model_run_str,
        model='hrrr',
        product='prs',  # Pressure levels
        fxx=hours_to_noon
    )

    ds_850 = H_target.xarray("TMP:850 mb", remove_grib=True)
    temp_850_data = ds_850['t']
    lats = ds_850['latitude'].values
    lons = ds_850['longitude'].values

    target_lon = lon if lon > 0 else lon + 360
    dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
    min_idx = np.unravel_index(np.argmin(dist), dist.shape)

    temp_850_k = float(temp_850_data.values[min_idx])
    temp_850_f = (temp_850_k - 273.15) * 9/5 + 32

    # Try to get 24hr ago for comparison
    hours_24hr_ago = hours_to_noon - 24
    temp_850_24hr_f = None

    if hours_24hr_ago >= 1:
        try:
            H_24hr = Herbie(
                model_run_str,
                model='hrrr',
                product='prs',
                fxx=hours_24hr_ago
            )
            ds_850_24hr = H_24hr.xarray("TMP:850 mb", remove_grib=True)
            temp_850_24hr_data = ds_850_24hr['t']
            temp_850_24hr_k = float(temp_850_24hr_data.values[min_idx])
            temp_850_24hr_f = (temp_850_24hr_k - 273.15) * 9/5 + 32
        except:
            pass

    return {
        "temp_850mb_f": temp_850_f,
        "temp_850mb_24hr_ago_f": temp_850_24hr_f,
        "temp_850mb_change_24hr": temp_850_f - temp_850_24hr_f if temp_850_24hr_f else None
    }


def extract_atmospheric_features(hrrr_data):
    """
    Extract atmospheric features for analog prediction.
    
    Returns a dict of features that can be used for ML prediction.
    """
    if not hrrr_data:
        return None

    # Wind features
    wind_speed = hrrr_data.get('wind_speed')
    wind_dir = hrrr_data.get('wind_dir')

    features = {}
    if wind_speed is not None:
        features['wind_speed'] = wind_speed

    if wind_dir is not None:
        _extracted_from_extract_atmospheric_features_(wind_dir, features)
    # 850mb airmass features
    temps_850mb = hrrr_data.get('temps_850mb')
    if temps_850mb:
        temp_850 = temps_850mb.get("temp_850mb_f")
        temp_change = temps_850mb.get("temp_850mb_change_24hr")

        if temp_850 is not None:
            features['temp_850mb'] = temp_850

        if temp_change is not None:
            features['temp_850mb_change_24hr'] = temp_change
            # Categorize airmass change
            features['airmass_warming'] = 1 if temp_change > 5 else 0
            features['airmass_cooling'] = 1 if temp_change < -5 else 0
            features['airmass_stable'] = 1 if abs(temp_change) <= 5 else 0

    return features or None


# TODO Rename this here and in `extract_atmospheric_features`
def _extracted_from_extract_atmospheric_features_(wind_dir, features):
    features['wind_dir'] = wind_dir
    # Convert wind direction to cardinal components (helps ML)
    wind_dir_rad = np.radians(wind_dir)
    features['wind_north_component'] = np.cos(wind_dir_rad)  # -1 to 1
    features['wind_east_component'] = np.sin(wind_dir_rad)   # -1 to 1

    # Wind advection type
    wind_info = analyze_wind_direction(wind_dir)
    if wind_info:
        advection = wind_info["advection_type"]
        features['wind_advection_warm'] = 1 if advection == "warm" else 0
        features['wind_advection_cold'] = 1 if advection == "cold" else 0
        features['wind_advection_neutral'] = 1 if advection == "neutral" else 0



def analyze_airmass_and_wind(hrrr_data, city_name):
    """
    Analyze wind patterns and airmass changes from HRRR data.
    
    Returns:
        dict with wind analysis and airmass insights
    """
    if not hrrr_data:
        return None

    wind_speed = hrrr_data.get('wind_speed')
    wind_dir = hrrr_data.get('wind_dir')
    temps_850mb = hrrr_data.get('temps_850mb')

    analysis = {
        "wind_analysis": None,
        "airmass_analysis": None,
        "forecast_adjustment": None
    }

    # Wind direction analysis
    if wind_dir is not None:
        wind_info = analyze_wind_direction(wind_dir)
        if wind_info:
            _extracted_from_analyze_airmass_and_wind_25(
                wind_info, wind_dir, wind_speed, analysis
            )
    # 850mb airmass analysis
    if temps_850mb:
        temp_850 = temps_850mb.get("temp_850mb_f")
        temp_change = temps_850mb.get("temp_850mb_change_24hr")

        if temp_850 is not None:
            airmass_message = f"850mb temp: {temp_850:.1f}°F"

            if temp_change is not None:
                airmass_message += f" (change: {temp_change:+.1f}°F/24hr)"

                if temp_change >= WARM_ADVECTION_THRESHOLD:
                    airmass_message += f"\n        🔥 SIGNIFICANT WARM AIRMASS MOVING IN"
                    analysis["forecast_adjustment"] = "warmer"
                elif temp_change <= COLD_ADVECTION_THRESHOLD:
                    airmass_message += f"\n        ❄️ SIGNIFICANT COLD AIRMASS MOVING IN"
                    analysis["forecast_adjustment"] = "colder"
                elif abs(temp_change) > 5:
                    airmass_message += f"\n        ⚠️ Notable airmass change detected"

            analysis["airmass_analysis"] = airmass_message

    return analysis


# TODO Rename this here and in `analyze_airmass_and_wind`
def _extracted_from_analyze_airmass_and_wind_25(wind_info, wind_dir, wind_speed, analysis):
    advection = wind_info["advection_type"]
    cardinal = wind_info["cardinal"]

            # Build wind analysis message
    if advection == "cold":
        wind_message = f"❄️ {cardinal} winds ({wind_dir:.0f}°) - COLD ADVECTION"
        wind_message += f"\n        Bringing colder air from the north"
        adjustment = "colder"
    elif advection == "warm":
        wind_message = f"🌡️ {cardinal} winds ({wind_dir:.0f}°) - WARM ADVECTION"
        wind_message += f"\n        Bringing warmer air from the south"
        adjustment = "warmer"
    else:
        wind_message = f"➡️ {cardinal} winds ({wind_dir:.0f}°) - neutral"
        adjustment = None

    if wind_speed:
        wind_message += f" at {wind_speed:.0f} mph"

    analysis["wind_analysis"] = wind_message
    analysis["forecast_adjustment"] = adjustment


class AtmosphericAnalogModel:
    """
    Model that predicts forecast adjustment based on atmospheric conditions.
    
    Learns patterns like:
    - "When wind is SW at 15mph + 10°F warm advection → forecasts are typically 2°F too cold"
    - "When 850mb drops 15°F in 24hr → forecasts are typically 1.5°F too warm"
    """
    
    def __init__(self, city_key):
        self.city_key = city_key
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_samples = 0
        
    def train(self, historical_df, verbose=True):
        """
        Train the atmospheric analog model on historical data.
        
        Args:
            historical_df: DataFrame with columns:
                - date
                - actual_temp (observed temperature)
                - forecast_ensemble (ensemble forecast for that day)
                - atmospheric features (wind_speed, wind_dir, etc.)
        
        Returns:
            success: True if trained successfully
        """
        if verbose:
            print(f"\n   🤖 Training atmospheric analog model for {self.city_key}...")
        
        # Filter to rows that have both actual temps and atmospheric features
        required_cols = ['actual_temp', 'forecast_ensemble']
        feature_cols = [col for col in historical_df.columns if col not in ['date', 'actual_temp', 'forecast_ensemble']]
        
        if not feature_cols:
            if verbose:
                print(f"      ⚠️ No atmospheric features found in training data")
            return False
        
        # Drop rows with missing values
        df_clean = historical_df[required_cols + feature_cols].dropna()
        
        if len(df_clean) < MIN_TRAINING_SAMPLES:
            if verbose:
                print(f"      ⚠️ Insufficient training samples: {len(df_clean)} < {MIN_TRAINING_SAMPLES}")
            return False
        
        # Target: actual - forecast (forecast error)
        y = df_clean['actual_temp'] - df_clean['forecast_ensemble']
        X = df_clean[feature_cols]
        
        self.feature_names = feature_cols
        self.training_samples = len(df_clean)
        
        # Train Random Forest (robust to missing features)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        if verbose:
            print(f"      ✅ Model trained on {self.training_samples} samples")
            print(f"      📊 Cross-validation MAE: {cv_mae:.2f}°F")
            print(f"      🔍 Features used: {', '.join(self.feature_names)}")
            
            # Show feature importance
            importances = self.model.feature_importances_
            top_features = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
            print(f"      🏆 Top features:")
            for feat, importance in top_features:
                print(f"         • {feat}: {importance:.3f}")
        
        return True
    
    def predict_adjustment(self, atmospheric_features):
        """
        Predict the forecast adjustment based on current atmospheric conditions.
        
        Args:
            atmospheric_features: Dict of current atmospheric features
        
        Returns:
            adjustment: Predicted temperature adjustment (°F)
            confidence: Confidence in prediction (0-1)
        """
        if not self.is_trained or not atmospheric_features:
            return 0.0, 0.0
        
        # Build feature vector
        try:
            feature_vector = []
            for feat_name in self.feature_names:
                feature_vector.append(atmospheric_features.get(feat_name, 0.0))
            
            X = np.array(feature_vector).reshape(1, -1)
            adjustment = self.model.predict(X)[0]
            
            # Confidence based on training sample size
            confidence = min(1.0, self.training_samples / 100.0)
            
            return adjustment, confidence
            
        except Exception as e:
            print(f"      ⚠️ Atmospheric model prediction error: {e}")
            return 0.0, 0.0
    
    def save(self, filepath):
        """Save the model to disk."""
        if not self.is_trained:
            return False
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_samples': self.training_samples,
            'city_key': self.city_key
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        return True
    
    def load(self, filepath):
        """Load the model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.training_samples = model_data['training_samples']
            self.city_key = model_data['city_key']
            self.is_trained = True
            
            return True
        except Exception as e:
            print(f"      ⚠️ Failed to load atmospheric model: {e}")
            return False




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

def analyze_city(city_key, bankroll, target_date):
    """
    Analyze a single city and return all qualifying bets.
    Does NOT make final bet selection - that happens across all cities.

    Args:
        city_key: Key for the city in the cities dict
        bankroll: Current bankroll amount
        target_date: The date to analyze markets for (in US Eastern time)
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

    # Use the target date passed in (already in US Eastern time)
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
        
        # Analyze wind and airmass (NEW!)
        if hrrr_data:
            airmass_info = analyze_airmass_and_wind(hrrr_data, city["name"])
            if airmass_info:
                if airmass_info.get("wind_analysis"):
                    print(f"     {airmass_info['wind_analysis']}")
                if airmass_info.get("airmass_analysis"):
                    print(f"     {airmass_info['airmass_analysis']}")
    else:
        print("     ⚠️ HRRR unavailable")
        airmass_info = None

    # 3. NWS forecast (NEW in v10!)
    print(f"\n  [3] National Weather Service:")
    nws_forecast = fetch_nws_forecast(city["lat"], city["lon"], target_date)
    
    if nws_forecast:
        print(f"     ✅ Forecast: {nws_forecast:.1f}°F")
    else:
        print("     ⚠️ NWS unavailable")

    # ============ ENHANCED ENSEMBLE FORECAST ============
    # Collect all available forecasts
    forecasts = []
    forecast_sources = []
    
    if hrrr_forecast:
        forecasts.append(hrrr_forecast)
        forecast_sources.append("HRRR")
    if open_meteo_forecast:
        forecasts.append(open_meteo_forecast)
        forecast_sources.append("Open-Meteo")
    if nws_forecast:
        forecasts.append(nws_forecast)
        forecast_sources.append("NWS")
    
    if not forecasts:
        print(f"\n  ❌ No forecasts available!")
        return qualifying_bets, None
    
    # Calculate ensemble with smart weighting
    if len(forecasts) == 1:
        ensemble_forecast = forecasts[0]
        agreement_level = 'medium'
        model_spread = 0
    else:
        # Calculate spread to detect disagreement
        model_spread = max(forecasts) - min(forecasts)
        
        # Weight by reliability and disagreement detection
        if len(forecasts) == 3:
            # All three sources available - weighted average
            # More balanced weighting - HRRR is good but not dominant
            if model_spread <= CONFIDENCE_BOOST_THRESHOLD:
                # High agreement - slight preference for HRRR
                weights = [0.40, 0.30, 0.30]  # HRRR, Open-Meteo, NWS
                agreement_level = 'high'
            elif model_spread <= ENSEMBLE_AGREEMENT_THRESHOLD:
                # Medium agreement - nearly equal
                weights = [0.35, 0.325, 0.325]
                agreement_level = 'medium'
            else:
                # Disagreement - equal weight, let them average out
                weights = [0.33, 0.33, 0.34]
                agreement_level = 'low'
        else:
            # Two sources - equal weight
            weights = [0.5, 0.5]
            if model_spread <= CONFIDENCE_BOOST_THRESHOLD:
                agreement_level = 'high'
            elif model_spread <= ENSEMBLE_AGREEMENT_THRESHOLD:
                agreement_level = 'medium'
            else:
                agreement_level = 'low'
        
        ensemble_forecast = sum(f * w for f, w in zip(forecasts, weights))
    
    # ============ MACHINE LEARNING ADJUSTMENT (NEW!) ============
    # Apply wind/airmass-based adjustments to ensemble forecast
    ml_adjustment = 0.0
    adjustment_reasons = []
    
    if airmass_info:
        forecast_adjustment = airmass_info.get("forecast_adjustment")
        
        # Get airmass data if available
        if hrrr_data and hrrr_data.get('temps_850mb'):
            temp_850_change = hrrr_data['temps_850mb'].get('temp_850mb_change_24hr')
            
            if temp_850_change is not None:
                # Strong airmass signal = adjust forecast (but more conservatively)
                if temp_850_change >= WARM_ADVECTION_THRESHOLD:
                    # Significant warming airmass
                    # If HRRR is lower than others, trust the warmer forecasts more
                    if hrrr_forecast and hrrr_forecast < ensemble_forecast:
                        ml_adjustment = model_spread * 0.20  # Shift 20% toward warmer (reduced from 30%)
                        adjustment_reasons.append(f"Warm airmass (+{temp_850_change:.1f}°F/24hr)")
                elif temp_850_change <= COLD_ADVECTION_THRESHOLD:
                    # Significant cooling airmass
                    # If HRRR is higher than others, trust the colder forecasts more
                    if hrrr_forecast and hrrr_forecast > ensemble_forecast:
                        ml_adjustment = model_spread * -0.20  # Shift 20% toward colder (reduced from 30%)
                        adjustment_reasons.append(f"Cold airmass ({temp_850_change:.1f}°F/24hr)")
                elif abs(temp_850_change) > 8:
                    # Moderate airmass change
                    if forecast_adjustment == "warmer" and hrrr_forecast and hrrr_forecast < ensemble_forecast:
                        ml_adjustment = model_spread * 0.10  # Shift 10% toward warmer (reduced from 20%)
                        adjustment_reasons.append(f"Moderate warming ({temp_850_change:+.1f}°F/24hr)")
                    elif forecast_adjustment == "colder" and hrrr_forecast and hrrr_forecast > ensemble_forecast:
                        ml_adjustment = model_spread * -0.10  # Shift 10% toward colder (reduced from 20%)
                        adjustment_reasons.append(f"Moderate cooling ({temp_850_change:.1f}°F/24hr)")
        
        # Wind-only signal (if no 850mb data)
        elif forecast_adjustment and model_spread > CONFIDENCE_BOOST_THRESHOLD:
            # Only adjust on wind if there's disagreement and we have a clear signal
            if forecast_adjustment == "warmer":
                # Warm advection detected
                if hrrr_forecast and hrrr_forecast < ensemble_forecast:
                    ml_adjustment = model_spread * 0.08  # Smaller adjustment (8%, reduced from 15%)
                    adjustment_reasons.append("Warm wind advection")
            elif forecast_adjustment == "colder":
                # Cold advection detected
                if hrrr_forecast and hrrr_forecast > ensemble_forecast:
                    ml_adjustment = model_spread * -0.08  # Smaller adjustment (8%, reduced from 15%)
                    adjustment_reasons.append("Cold wind advection")
    
    # Apply the ML adjustment (only if meaningful)
    # Cap at ±1.5°F to prevent extreme adjustments
    ml_adjustment = max(-1.5, min(1.5, ml_adjustment))
    
    if abs(ml_adjustment) > 0.2:  # Ignore tiny adjustments
        original_ensemble = ensemble_forecast
        ensemble_forecast += ml_adjustment
        print(f"\n  🤖 ML ADJUSTMENT: {original_ensemble:.1f}°F → {ensemble_forecast:.1f}°F ({ml_adjustment:+.1f}°F)")
        for reason in adjustment_reasons:
            print(f"     Reason: {reason}")
    
    
    # ============ ATMOSPHERIC ANALOG MODEL PREDICTION (V11!) ============
    atmospheric_adjustment = 0.0
    atmospheric_confidence = 0.0
    
    # Load and use atmospheric model if available
    atmospheric_model = AtmosphericAnalogModel(city_key)
    model_path = Path(ATMOSPHERIC_MODEL_PATH) / f"{city_key}_atmospheric.pkl"
    
    if model_path.exists():
        # Load the trained model
        atmospheric_model.load(str(model_path))
        
        # Extract current atmospheric features from HRRR data
        if hrrr_data:
            atmos_features = extract_atmospheric_features(hrrr_data)
            
            if atmos_features:
                # Get prediction from atmospheric model
                atmospheric_adjustment, atmospheric_confidence = atmospheric_model.predict_adjustment(atmos_features)
                
                # Cap atmospheric predictions at ±1.0°F to prevent extreme adjustments
                atmospheric_adjustment = max(-1.0, min(1.0, atmospheric_adjustment))
                
                # Only apply if adjustment is meaningful (>0.3°F) to avoid noise
                if abs(atmospheric_adjustment) > 0.3:
                    print(f"\n  🌐 ATMOSPHERIC ANALOG PREDICTION:")
                    print(f"     Based on {atmospheric_model.training_samples} historical patterns")
                    print(f"     Predicted adjustment: {atmospheric_adjustment:+.1f}°F (confidence: {atmospheric_confidence:.0%})")
                    print(f"     Current conditions: {atmos_features.get('wind_speed', 0):.0f}mph from {atmos_features.get('wind_dir', 0):.0f}°")
                    if 'temp_850mb_change_24hr' in atmos_features and atmos_features['temp_850mb_change_24hr']:
                        print(f"     850mb change: {atmos_features['temp_850mb_change_24hr']:+.1f}°F/24hr")
                    
                    # Apply weighted atmospheric adjustment to ensemble
                    weighted_adjustment = atmospheric_adjustment * ATMOSPHERIC_MODEL_WEIGHT * atmospheric_confidence
                    original_ensemble = ensemble_forecast
                    ensemble_forecast += weighted_adjustment
                    
                    print(f"     Final adjustment: {weighted_adjustment:+.2f}°F ({ATMOSPHERIC_MODEL_WEIGHT:.0%} weight × {atmospheric_confidence:.0%} confidence)")
                    print(f"     Ensemble: {original_ensemble:.1f}°F → {ensemble_forecast:.1f}°F")
                    
                    # Recalculate forecast bin with adjusted ensemble
                    forecast_bin = get_bin_for_temp(ensemble_forecast)
    
    
    # Major disagreement detection (NEW in v10!)
    major_disagreement = model_spread > MAJOR_DISAGREEMENT_THRESHOLD
    
    print(f"\n  📊 Model Comparison:")
    for source, forecast in zip(forecast_sources, forecasts):
        print(f"     {source}: {forecast:.1f}°F")
    
    if len(forecasts) > 1:
        print(f"     Spread: {model_spread:.1f}°F → Agreement: {agreement_level.upper()}")
        
        if major_disagreement:
            print(f"     ⚠️  MAJOR DISAGREEMENT DETECTED!")
            print(f"        This often indicates frontal systems or airmass changes")
            print(f"        Consider checking radar/surface analysis")
            # Automatically downgrade agreement level
            if agreement_level == 'high':
                agreement_level = 'medium'
            elif agreement_level == 'medium':
                agreement_level = 'low'

    print(f"\n  🎯 ENSEMBLE FORECAST: {ensemble_forecast:.1f}°F")

    forecast_bin = get_bin_for_temp(ensemble_forecast)
    print(f"     Primary bin: {forecast_bin[0]}°-{forecast_bin[1]}°F")

    # Store city summary
    city_summary = {
        "city": city["name"],
        "ensemble_forecast": ensemble_forecast,
        "forecast_bin": forecast_bin,
        "agreement_level": agreement_level,
        "hrrr_forecast": hrrr_forecast,
        "open_meteo_forecast": open_meteo_forecast,
        "nws_forecast": nws_forecast,
        "model_spread": model_spread,
        "major_disagreement": major_disagreement,
        "airmass_info": airmass_info if hrrr_forecast else None,
        "ml_adjustment": ml_adjustment,
        "adjustment_reasons": adjustment_reasons if abs(ml_adjustment) > 0.2 else None,
        "atmospheric_adjustment": atmospheric_adjustment,
        "atmospheric_confidence": atmospheric_confidence,
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
                                                CALIBRATED_FORECAST_STD, agreement_level)
            # Check if our forecast actually falls IN this contract's bin
            is_forecast_bin = (low <= ensemble_forecast <= high)
            contract_type = "range"

        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = calibrated_below_probability(threshold, ensemble_forecast,
                                                       CALIBRATED_FORECAST_STD, agreement_level)
            is_forecast_bin = ensemble_forecast <= threshold
            contract_type = "below"

        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = calibrated_above_probability(threshold, ensemble_forecast,
                                                       CALIBRATED_FORECAST_STD, agreement_level)
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
                    "major_disagreement": major_disagreement,
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
                        "major_disagreement": major_disagreement,
                    })

    print(f"\n  Found {len(qualifying_bets)} qualifying bets")
    print(f"  Filtered out: {skipped_cheap} cheap, {skipped_expensive} expensive")

    return qualifying_bets, city_summary


# ============ SMART BET SELECTION ============

def smart_select_bets(all_bets):
    """
    Apply smart bet selection logic across all cities.
    
    Rules:
    1. Different cities = uncorrelated → bet freely on best bet from each
    2. Same city = correlated → only stack if SUPER confident (edge_ratio >= 2.0x)
    3. Cap total bets per day
    """
    if not all_bets:
        return []
    
    # Group bets by city
    bets_by_city = {}
    for bet in all_bets:
        city = bet["city"]
        if city not in bets_by_city:
            bets_by_city[city] = []
        bets_by_city[city].append(bet)
    
    # Select bets from each city
    selected_bets = []
    
    for city, city_bets in bets_by_city.items():
        # Sort by edge ratio (best first)
        city_bets = sorted(city_bets, key=lambda x: -x["edge_ratio"])
        
        # Always take the best bet from each city
        best_bet = city_bets[0]
        best_bet["selection_reason"] = "best_in_city"
        selected_bets.append(best_bet)
        
        # Add additional same-city bets ONLY if super confident
        for bet in city_bets[1:MAX_BETS_PER_CITY]:
            if bet["edge_ratio"] >= SAME_CITY_MULTI_BET_THRESHOLD:
                bet["selection_reason"] = "super_confident_stack"
                selected_bets.append(bet)
    
    # Sort final selection by edge ratio
    selected_bets = sorted(selected_bets, key=lambda x: -x["edge_ratio"])
    
    # Cap total bets
    selected_bets = selected_bets[:MAX_TOTAL_BETS_PER_DAY]
    
    return selected_bets


def print_recommendations(selected_bets, all_bets, city_summaries, bankroll, target_date):
    """Print the final betting recommendations in a clear format."""

    target_date_str = target_date.strftime("%A, %B %d, %Y")
    
    print(f"\n{'='*70}")
    print("🎯 TODAY'S BETTING RECOMMENDATIONS")
    print(f"📅 Target Date: {target_date_str}")
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
    
    # Check for major disagreements
    has_major_disagreement = any(b.get("major_disagreement", False) for b in selected_bets)
    if has_major_disagreement:
        print("\n⚠️  WARNING: MAJOR MODEL DISAGREEMENT DETECTED")
        print("   Some forecasts differ by >5°F - possible frontal system")
        print("   Proceed with caution or reduce bet sizes")
        print()
    
    # Analyze what we're recommending
    cities_with_bets = set(b["city"] for b in selected_bets)
    num_cities = len(cities_with_bets)
    
    super_confident_count = sum(bool(b.get("selection_reason") == "super_confident_stack")
                            for b in selected_bets)
    
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
        # Confidence indicator
        if bet["edge_ratio"] >= SUPER_CONFIDENT_EDGE_RATIO:
            confidence = "🔥 SUPER CONFIDENT"
        elif bet["edge_ratio"] >= SAME_CITY_MULTI_BET_THRESHOLD:
            confidence = "✓ High confidence"
        else:
            confidence = "⭐ Best opportunity"
        
        # Add disagreement warning if applicable
        if bet.get("major_disagreement", False):
            confidence += " ⚠️ (major disagreement)"
        
        # Selection reason
        if bet.get("selection_reason") == "super_confident_stack":
            reason = "(stacked - high edge)"
        else:
            reason = "(best in city)"
        
        bucket_emoji = "🎯" if bet["price_bucket"] == "sweet_spot" else "📊"
        forecast_marker = "📍" if bet.get("is_forecast_bin") else ""
        
        odds = bet["kelly_info"].get("odds", 0)
        potential_profit = bet["bet_size"] * odds
        
        print(f"   ┌─ BET #{i}: {bet['city']} {reason}")
        print(f"   │  {confidence}")
        print(f"   │  🌡️  Model predicts high: {bet['ensemble_forecast']:.1f}°F")
        print(f"   │  Contract: {bet['subtitle']} {forecast_marker}")
        print(f"   │  Side: {bet['side']} at {bet['bet_price']*100:.0f}¢ {bucket_emoji}")
        print(f"   │  Your probability: {bet['our_prob_win']*100:.1f}%")
        print(f"   │  Edge: {bet['edge']*100:+.1f}% ({bet['edge_ratio']:.1f}x minimum)")
        print(f"   │  Kelly: {bet['kelly_info'].get('fractional_kelly_pct', 0):.1f}% of bankroll")
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
            disagreement_flag = " 🚨 MAJOR DISAGREEMENT" if summary.get("major_disagreement", False) else ""
            
            # Show all three forecasts if available
            sources_str = ""
            if summary.get("hrrr_forecast"):
                sources_str += f"HRRR:{summary['hrrr_forecast']:.1f}° "
            if summary.get("open_meteo_forecast"):
                sources_str += f"OM:{summary['open_meteo_forecast']:.1f}° "
            if summary.get("nws_forecast"):
                sources_str += f"NWS:{summary['nws_forecast']:.1f}° "
            
            ml_adj_display = ""
            if summary.get("ml_adjustment") and summary["ml_adjustment"] != 0.0:
                ml_adj_display = f" 🤖 ML+{summary['ml_adjustment']:+.1f}°F"
            
            print(f"   {summary['city']}: {summary['ensemble_forecast']:.1f}°F{ml_adj_display} → Bin {summary['forecast_bin'][0]}-{summary['forecast_bin'][1]}° {agreement_emoji} {summary['agreement_level'].upper()}{disagreement_flag}")
            if sources_str:
                print(f"      ({sources_str.strip()})")
            
            # Show ML adjustment reasons if available
            if summary.get("adjustment_reasons"):
                for reason in summary["adjustment_reasons"]:
                    print(f"      🤖 ML: {reason}")
            
            # Show airmass/wind info if available
            airmass_info = summary.get("airmass_info")
            if airmass_info:
                if airmass_info.get("wind_analysis"):
                    wind_line = airmass_info['wind_analysis'].split('\n')[0]
                    print(f"      Wind: {wind_line}")
                if airmass_info.get("airmass_analysis"):
                    # Show just the first line
                    airmass_line = airmass_info['airmass_analysis'].split('\n')[0]
                    print(f"      Airmass: {airmass_line}")
    
    # Show what we didn't bet on
    not_selected = [b for b in all_bets if b not in selected_bets]
    if not_selected:
        print(f"\n📋 Also considered but not selected: {len(not_selected)} other opportunities")
        print("   (Either same-city with edge ratio < 2.0x, or beyond daily limit)")


# ============ MAIN ============

def main():
    global STARTING_BANKROLL, KELLY_FRACTION, selected_cities

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Kalshi Weather Betting Model v10')
    parser.add_argument('--kelly', type=float, default=KELLY_FRACTION,
                        help=f'Kelly fraction (default: {KELLY_FRACTION})')
    parser.add_argument('--bankroll', type=float, default=STARTING_BANKROLL,
                        help=f'Starting bankroll in dollars (default: {STARTING_BANKROLL})')
    parser.add_argument('--today', action='store_true',
                        help="Look at today's markets instead of tomorrow's")
    parser.add_argument('--tomorrow', action='store_true',
                        help="Force looking at tomorrow's markets (even if it's evening)")
    parser.add_argument('--date', type=str, default=None,
                        help="Specify a specific date (format: YYYY-MM-DD, e.g., 2025-12-10)")
    parser.add_argument('--train', action='store_true',
                        help="Build and train atmospheric models from historical data")
    
    parser.add_argument('--cities', type=str, default=None,
                        help="Comma-separated list of cities to analyze (default: all). Options: chicago, nyc, miami")
    args = parser.parse_args()

    # Update global settings from command-line args
    KELLY_FRACTION = args.kelly
    STARTING_BANKROLL = args.bankroll

    # Update selected cities if specified
    if args.cities:
        cities_to_analyze = [c.strip().lower() for c in args.cities.split(',')]
        # Validate cities
        cities_to_analyze = [c for c in cities_to_analyze if c in cities]
        if cities_to_analyze:
            selected_cities = cities_to_analyze


    # Training mode
    if args.train:
        print("=" * 70)
        print("ATMOSPHERIC MODEL TRAINING MODE")
        print("=" * 70)
        
        for city_key in selected_cities:
            city_config = cities[city_key]
            print(f"\n🏙️ Training model for {city_config['name']}...")
            
            # Look for training data CSV
            import glob
            training_files = glob.glob(f"training_data_{city_key}_*.csv")
            
            if not training_files:
                print(f"   ❌ No training data found for {city_key}")
                print(f"   💡 Run historical_hrrr_fetcher.py first to generate training data")
                continue
            
            # Use the most recent training file
            training_file = sorted(training_files)[-1]
            print(f"   📂 Loading: {training_file}")
            
            try:
                training_data = pd.read_csv(training_file)
                print(f"   📊 Loaded {len(training_data)} samples")
                
                # Train the model
                atmospheric_model = AtmosphericAnalogModel(city_key)
                success = atmospheric_model.train(training_data, verbose=True)
                
                if success:
                    # Save the model
                    model_path = Path(ATMOSPHERIC_MODEL_PATH) / f"{city_key}_atmospheric.pkl"
                    atmospheric_model.save(str(model_path))
                    print(f"   ✅ Model saved to {model_path}")
                else:
                    print(f"   ❌ Training failed for {city_key}")
                    
            except Exception as e:
                print(f"   ❌ Error training {city_key}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        return


    # Determine target date based on US Eastern time
    if args.date:
        # Use the specific date provided
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        eastern_today = get_eastern_now().date()
        is_today = (target_date == eastern_today)
    elif args.today:
        target_date, is_today = get_target_date(force_today=True)
    elif args.tomorrow:
        # Force tomorrow regardless of time
        eastern_now = get_eastern_now()
        target_date = eastern_now.date() + timedelta(days=1)
        is_today = False
    else:
        # Auto-detect based on Eastern time
        target_date, is_today = get_target_date()

    print_header(target_date, is_today)

    bankroll = STARTING_BANKROLL
    all_bets = []
    city_summaries = []

    # Analyze each city
    for city_key in selected_cities:
        city_bets, city_summary = analyze_city(city_key, bankroll, target_date)
        all_bets.extend(city_bets)
        city_summaries.append(city_summary)

    # Smart bet selection across all cities
    selected_bets = smart_select_bets(all_bets)

    # Print recommendations
    print_recommendations(selected_bets, all_bets, city_summaries, bankroll, target_date)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE (v11 - Atmospheric Analog Enhanced)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()