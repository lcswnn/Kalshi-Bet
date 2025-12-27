"""
KALSHI WEATHER BETTING MODEL v10
================================
ENHANCED V9 + SELECTIVE V12 FEATURES (per analysis recommendations)

Changes from v9:
1. DYNAMIC UNCERTAINTY: Scales with model spread (1.8-4.5°F) instead of fixed 2.8°F
2. CURRENT OBSERVATION CONSTRAINT: Prevents impossible afternoon forecasts
3. LEAD-TIME AWARE UNCERTAINTY: Adjusts for forecast horizon (6hr vs 24hr)
4. SIMPLE CALIBRATION TRACKER: Lightweight tracking for review (no auto-adjustment)

What we KEPT from v9:
- Clean ensemble weighting: 45% HRRR, 30% NWS, 25% Open-Meteo
- Kelly Criterion with fees accounted for
- Smart bet selection (different cities vs same city)
- Price filtering (15¢-90¢)
- Agreement threshold

What we AVOIDED from v12:
- Complex atmospheric analog matching
- Cloud/precipitation timing analysis
- Aggressive calibration boosts
- Weather regime detection
- 850mb temperature analysis
"""

import pandas as pd
import numpy as np
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

# Edge requirements by price bucket (INCREASED TO ACCOUNT FOR ~4% FEE DRAG)
BASE_EDGE_REQUIREMENTS = {
    "sweet_spot": 0.16,    # 16% edge for 15-50¢ contracts
    "high_price": 0.16,    # 16% edge for 50-90¢ contracts
}


# ============ NEW IN V10: DYNAMIC UNCERTAINTY ============

def calculate_dynamic_uncertainty(hrrr_temp, nws_temp, meteo_temp, hours_until_target=18):
    """
    Calculate uncertainty dynamically based on model spread and lead time.
    
    This is MUST ADD #1 from the analysis:
    - When models agree → low uncertainty
    - When models disagree → high uncertainty
    - Adjusts for forecast lead time
    
    Returns:
        uncertainty_std (float): Standard deviation in °F
    """
    # Collect available temperatures
    temps = [t for t in [hrrr_temp, nws_temp, meteo_temp] if t is not None]
    
    if len(temps) < 2:
        # Single model - use higher baseline uncertainty
        model_spread = 0
        base_std = 3.0
    else:
        model_spread = max(temps) - min(temps)
        
        # Base uncertainty from model spread
        if model_spread < 2.0:
            base_std = 1.8   # High agreement - very confident
        elif model_spread < 4.0:
            base_std = 2.5   # Medium agreement
        elif model_spread < 6.0:
            base_std = 3.5   # Low agreement
        else:
            base_std = 4.5   # Major disagreement (frontal system)
    
    # Lead-time adjustment (SHOULD ADD #3 from analysis)
    if hours_until_target <= 6:
        lead_time_mult = 0.7   # HRRR is very accurate at short range
    elif hours_until_target <= 12:
        lead_time_mult = 0.85
    elif hours_until_target <= 18:
        lead_time_mult = 1.0
    else:
        lead_time_mult = 1.2   # Beyond HRRR's sweet spot
    
    final_std = base_std * lead_time_mult
    
    return final_std, model_spread


# ============ NEW IN V10: OBSERVATION CONSTRAINT ============

def fetch_current_observations(lat, lon):
    """
    Fetch current temperature from Open-Meteo's current weather API.
    Free, no key needed.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current_temp = data["current"]["temperature_2m"]
        current_time_str = data["current"]["time"]
        current_time = datetime.fromisoformat(current_time_str.replace('Z', '+00:00'))
        
        return {
            "current_temp": current_temp,
            "current_time": current_time,
        }
        
    except Exception as e:
        return None


def apply_observation_constraint(forecast_temp, current_temp, current_hour):
    """
    Constrain forecast based on current observations.
    
    This is MUST ADD #2 from the analysis:
    - If it's already afternoon, the forecast can't be much higher than current temp
    - Prevents physically impossible forecasts
    
    Returns:
        constrained_temp (float): Adjusted forecast
        constraint_applied (bool): Whether constraint was applied
        message (str): Explanation of constraint
    """
    if current_temp is None or current_hour is None:
        return forecast_temp, False, None
    
    # Morning - don't constrain (high could still occur)
    if current_hour < 12:
        return forecast_temp, False, None
    
    # Late afternoon (3 PM+) - current temp is probably near the high
    if current_hour >= 15:
        if forecast_temp > current_temp + 2.0:
            constrained = current_temp + 2.0
            return constrained, True, f"Late afternoon constraint: {forecast_temp:.1f}→{constrained:.1f}°F (current: {current_temp:.1f}°F)"
    
    # Early afternoon (1-3 PM) - some warming still possible
    elif current_hour >= 13:
        if forecast_temp > current_temp + 4.0:
            constrained = current_temp + 4.0
            return constrained, True, f"Early afternoon constraint: {forecast_temp:.1f}→{constrained:.1f}°F (current: {current_temp:.1f}°F)"
    
    # Noon-1 PM - more warming possible
    else:
        if forecast_temp > current_temp + 6.0:
            constrained = current_temp + 6.0
            return constrained, True, f"Noon constraint: {forecast_temp:.1f}→{constrained:.1f}°F (current: {current_temp:.1f}°F)"
    
    return forecast_temp, False, None


# ============ NEW IN V10: SIMPLE CALIBRATION TRACKER ============

class SimpleCalibrationTracker:
    """
    Lightweight calibration tracking.
    Simpler than v12's version - tracks but doesn't auto-adjust.
    Use this to monitor your predictions vs actual outcomes.
    """
    def __init__(self, db_path="calibration_v10.json"):
        self.db_path = db_path
        self.predictions = self._load()
    
    def _load(self):
        try:
            if Path(self.db_path).exists():
                with open(self.db_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def _save(self):
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.predictions, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save calibration: {e}")
    
    def record(self, date, city, our_prob, market_price, subtitle, side):
        """Record a prediction for later analysis."""
        self.predictions.append({
            'date': str(date),
            'city': city,
            'our_prob': float(our_prob),
            'market_price': float(market_price),
            'subtitle': subtitle,
            'side': side,
            'outcome': None,
            'recorded_at': datetime.now().isoformat()
        })
        self._save()
    
    def update_outcome(self, date, city, won):
        """Update a prediction with actual outcome (True if won, False if lost)."""
        updated = 0
        for pred in self.predictions:
            if pred['date'] == str(date) and pred['city'].lower() == city.lower() and pred['outcome'] is None:
                pred['outcome'] = 'win' if won else 'loss'
                updated += 1
        
        if updated > 0:
            self._save()
            print(f"✅ Updated {updated} prediction(s) for {city} on {date}")
        else:
            print(f"⚠️ No matching predictions found for {city} on {date}")
    
    def get_calibration_report(self):
        """Generate calibration report showing predicted vs actual win rates."""
        completed = [p for p in self.predictions if p['outcome'] is not None]
        
        if not completed:
            return "\n📊 No completed predictions yet. Update outcomes to see calibration.\n"
        
        # Bin predictions
        bins = {}
        for pred in completed:
            prob_bin = round(pred['our_prob'], 1)
            if prob_bin not in bins:
                bins[prob_bin] = {'wins': 0, 'total': 0}
            bins[prob_bin]['total'] += 1
            if pred['outcome'] == 'win':
                bins[prob_bin]['wins'] += 1
        
        report = "\n" + "=" * 50 + "\n"
        report += "📊 CALIBRATION REPORT (v10)\n"
        report += "=" * 50 + "\n"
        report += f"{'Predicted':<12} {'Actual':<12} {'Diff':<10} {'Count':<10}\n"
        report += "-" * 50 + "\n"
        
        for prob_bin in sorted(bins.keys()):
            data = bins[prob_bin]
            actual = data['wins'] / data['total'] if data['total'] > 0 else 0
            diff = actual - prob_bin
            
            report += f"{prob_bin*100:>6.0f}%      {actual*100:>6.0f}%      {diff*100:>+6.0f}%    {data['total']:>4}\n"
        
        total_wins = sum(b['wins'] for b in bins.values())
        total_bets = sum(b['total'] for b in bins.values())
        overall_rate = total_wins / total_bets if total_bets > 0 else 0
        
        report += "-" * 50 + "\n"
        report += f"Overall: {total_wins}/{total_bets} wins ({overall_rate*100:.1f}%)\n"
        report += "\nInterpretation:\n"
        report += "  Positive diff = Too conservative (leaving money on table)\n"
        report += "  Negative diff = Overconfident (betting too aggressively)\n"
        report += "  Goal: All diffs near 0%\n"
        report += "=" * 50 + "\n"
        
        return report


# Global calibration tracker
calibration_tracker = SimpleCalibrationTracker()


def get_min_edge(price, agreement_level='high'):
    """Dynamic edge requirement based on price and model agreement."""
    base_edge = BASE_EDGE_REQUIREMENTS.get(
        "sweet_spot" if price <= SWEET_SPOT_HIGH else "high_price", 
        0.12
    )
    
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
    Calculate optimal bet size using Kelly Criterion, accounting for Kalshi fees.
    """
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

selected_cities = ["chicago", "nyc", "miami", "austin", "la"]


# ============ HEADER ============
def print_header():
    print("=" * 70)
    print("KALSHI WEATHER BETTING MODEL v10")
    print("(v9 Foundation + Dynamic Uncertainty + Observation Constraints)")
    print("=" * 70)
    print(f"\nAnalyzing: Chicago, New York City, Miami, Austin, Los Angeles")
    print(f"Agreement Threshold: ±{ENSEMBLE_AGREEMENT_THRESHOLD}°F")
    print(f"Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")
    print(f"Sweet Spot: {SWEET_SPOT_LOW*100:.0f}¢ - {SWEET_SPOT_HIGH*100:.0f}¢")
    print(f"⚠️ FEE ADJUSTMENT: {KALSHI_FEE_RATE*100:.0f}% entry fee built into odds")
    print(f"\n🎯 SMART BET SELECTION:")
    print(f"   • Different cities: Bet freely (uncorrelated)")
    print(f"   • Same city: Only stack if edge ratio ≥ {SAME_CITY_MULTI_BET_THRESHOLD}x")
    print(f"   • Max {MAX_BETS_PER_CITY} bets/city, {MAX_TOTAL_BETS_PER_DAY} total/day")
    print(f"\n💰 BANKROLL: ${STARTING_BANKROLL:.2f} | {KELLY_FRACTION:.0%} Kelly")
    print("Data Sources: HRRR (45%) + NWS (30%) + Open-Meteo (25%)")
    print(f"\n🆕 V10 ENHANCEMENTS:")
    print(f"   • Dynamic uncertainty (1.8-4.5°F based on model spread)")
    print(f"   • Afternoon observation constraints")
    print(f"   • Lead-time aware uncertainty")


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
    
    # Calculate hours until target for lead-time adjustment
    hours_until_target = hours_to_start + 3  # Approximate middle of forecast window
    
    weather_data = {
        'forecast_high': forecast_high,
        'all_temps': temperatures,
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


# ============ PROBABILITY MODELS ============

def calibrated_probability(contract_low, contract_high, forecast, uncertainty_std, agreement_level='high'):
    """Calibrated probability model using Gaussian distribution with dynamic uncertainty."""
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
    V10 ENHANCEMENT: Uses dynamic uncertainty + observation constraints.
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

    hours_until_target = hrrr_data.get('hours_until_target', 18) if hrrr_data else 18

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

    # ============ V10: CALCULATE DYNAMIC UNCERTAINTY ============
    dynamic_std, model_spread = calculate_dynamic_uncertainty(
        hrrr_forecast, nws_forecast, open_meteo_forecast, hours_until_target
    )
    print(f"\n  🆕 Dynamic Uncertainty: {dynamic_std:.2f}°F (spread: {model_spread:.1f}°F)")

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

    # Calculate agreement level
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

    # ============ V10: APPLY OBSERVATION CONSTRAINT (for same-day bets) ============
    constraint_applied = False
    if target_date == today and current_hour >= 12:
        print(f"\n  🆕 Checking observation constraints...")
        obs = fetch_current_observations(city["lat"], city["lon"])
        if obs:
            current_temp = obs["current_temp"]
            print(f"     Current temp: {current_temp:.1f}°F at {current_hour}:00")
            
            constrained_forecast, constraint_applied, message = apply_observation_constraint(
                ensemble_forecast, current_temp, current_hour
            )
            
            if constraint_applied:
                print(f"     ⚠️ {message}")
                ensemble_forecast = constrained_forecast
                print(f"     🎯 CONSTRAINED FORECAST: {ensemble_forecast:.1f}°F")

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
        "dynamic_uncertainty": dynamic_std,
        "constraint_applied": constraint_applied,
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

    # ============ ANALYZE CONTRACTS (using dynamic uncertainty) ============
    print(f"\n  {'='*60}")
    print(f"  CONTRACT ANALYSIS (Filtered: {MIN_CONTRACT_PRICE*100:.0f}¢-{MAX_CONTRACT_PRICE*100:.0f}¢)")
    print(f"  Using dynamic uncertainty: {dynamic_std:.2f}°F")
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
                                                dynamic_std, agreement_level)
            is_forecast_bin = (low == forecast_bin[0])
            contract_type = "range"

        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = calibrated_below_probability(threshold, ensemble_forecast,
                                                       dynamic_std, agreement_level)
            is_forecast_bin = ensemble_forecast <= threshold
            contract_type = "below"

        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = calibrated_above_probability(threshold, ensemble_forecast,
                                                       dynamic_std, agreement_level)
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
                    "dynamic_uncertainty": dynamic_std,
                })

        # Evaluate NO bet (don't bet NO on forecast bin)
        if not is_forecast_bin:
            no_prob = 1 - model_prob
            no_market = 1 - kalshi_prob
            no_edge = no_prob - no_market
            no_min_edge = get_min_edge(no_market, agreement_level)
            
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
                        "dynamic_uncertainty": dynamic_std,
                    })

    print(f"\n  Found {len(qualifying_bets)} qualifying bets")
    print(f"  Filtered out: {skipped_cheap} cheap, {skipped_expensive} expensive")

    return qualifying_bets, city_summary


# ============ SMART BET SELECTION ============

def smart_select_bets(all_bets):
    """
    Select the top 6 bets by edge ratio.
    """
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
    print("🎯 TODAY'S BETTING RECOMMENDATIONS (v10)")
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
        print(f"   │  Your probability: {bet['our_prob_win']*100:.1f}%")
        print(f"   │  Edge: {bet['edge']*100:+.1f}% ({bet['edge_ratio']:.1f}x minimum)")
        print(f"   │  Kelly: {bet['kelly_info'].get('fractional_kelly_pct', 0):.1f}% of bankroll")
        print(f"   │  Uncertainty: {bet.get('dynamic_uncertainty', 2.8):.2f}°F")  # V10: Show uncertainty
        print(f"   │")
        print(f"   │  💰 BET: ${bet['bet_size']:.2f}")
        print(f"   │  📈 Potential profit: ${potential_profit:.2f}")
        print(f"   └{'─'*50}")
        print()

        total_wager += bet["bet_size"]
        total_potential += potential_profit
        
        # Record to calibration tracker
        calibration_tracker.record(
            date=(datetime.now() + timedelta(days=1 if datetime.now().hour >= 6 else 0)).date(),
            city=bet["city"],
            our_prob=bet["our_prob_win"],
            market_price=bet["bet_price"],
            subtitle=bet["subtitle"],
            side=bet["side"]
        )

    print(f"   {'='*55}")
    print(f"   TOTAL WAGER: ${total_wager:.2f} ({100*total_wager/bankroll:.1f}% of bankroll)")
    print(f"   POTENTIAL PROFIT: ${total_potential:.2f}")
    print(f"   {'='*55}")

    # City forecasts summary
    print(f"\n📊 FORECAST SUMMARY:")
    for summary in city_summaries:
        if summary:
            agreement_emoji = "✅" if summary["agreement_level"] == "high" else "⚠️" if summary["agreement_level"] == "medium" else "❌"
            constraint_marker = "🔒" if summary.get("constraint_applied") else ""
            print(f"   {summary['city']}: {summary['ensemble_forecast']:.1f}°F → Bin {summary['forecast_bin'][0]}-{summary['forecast_bin'][1]}° {agreement_emoji} {summary['agreement_level'].upper()} (σ={summary.get('dynamic_uncertainty', 2.8):.1f}°F) {constraint_marker}")

    not_selected = [b for b in all_bets if b not in selected_bets]
    if not_selected:
        print(f"\n📋 Also considered but not selected: {len(not_selected)} other opportunities")
        print("   (Beyond top 6 by edge ratio)")


# ============ ARGUMENT PARSING ============

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Kalshi Weather Betting Model v10")

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
    
    # V10: Calibration commands
    parser.add_argument("--calibration", action="store_true",
                        help="Show calibration report")
    parser.add_argument("--update-win", nargs=2, metavar=('DATE', 'CITY'),
                        help="Mark a prediction as WON (DATE CITY)")
    parser.add_argument("--update-loss", nargs=2, metavar=('DATE', 'CITY'),
                        help="Mark a prediction as LOST (DATE CITY)")

    return parser.parse_args()


# ============ MAIN ============

def main():
    global KELLY_FRACTION, STARTING_BANKROLL, selected_cities

    args = parse_arguments()

    # Handle calibration commands
    if args.calibration:
        print(calibration_tracker.get_calibration_report())
        return
    
    if args.update_win:
        date_str, city = args.update_win
        calibration_tracker.update_outcome(date_str, city, won=True)
        print(calibration_tracker.get_calibration_report())
        return
    
    if args.update_loss:
        date_str, city = args.update_loss
        calibration_tracker.update_outcome(date_str, city, won=False)
        print(calibration_tracker.get_calibration_report())
        return

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
    print("ANALYSIS COMPLETE (v10 - Dynamic Uncertainty + Observation Constraints)")
    print(f"{'='*70}")
    print("\n💡 TIP: Track your results with:")
    print("   --update-win DATE CITY    (e.g., --update-win 2024-01-15 Chicago)")
    print("   --update-loss DATE CITY")
    print("   --calibration             (show calibration report)")


if __name__ == "__main__":
    main()