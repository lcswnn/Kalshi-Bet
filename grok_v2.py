"""
Kalshi Weather Model Backtester v2.0
=====================================
CHANGELOG from v1.0:
- FIX: Dynamic bin detection from Kalshi market subtitles (no more hardcoded even/odd)
- FIX: Use np.round() for NWS integer settlement (not np.floor())
- FIX: Reduced default σ from 5.0 to 3.0 based on calibration analysis
- FIX: Better outcome determination using actual Kalshi market results
- NEW: Handles "X or below" and "X or above" edge bins
- NEW: More detailed logging and statistics

Usage:
python grok_v2.py --start_date 2025-01-01 --end_date 2025-01-31 --cities miami,austin
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import argparse
from scipy.stats import skewnorm
import re

# ============ MODEL CONFIGURATION ============
ENSEMBLE_WEIGHTS = {
    "NWS": 0.35,      # Approximated with NBM
    "HRRR": 0.25,     # High-res local
    "Open-Meteo": 0.20,  # Blended global
    "GFS": 0.20       # Global baseline
}

STATION_BIAS = {
    "miami": -1.3,    # Model over-forecasts by ~1.8°F, correct down
    "austin": 0.0,    # Neutral for now
    "chicago": 1.5,
    "nyc": -0.5,
}

# Reduced from 5.0-6.0 based on calibration analysis
CITY_UNCERTAINTY = {
    "miami": 3.0,
    "austin": 3.5,
    "chicago": 4.0,
    "nyc": 4.5,
}

CITY_SKEW_PARAMS = {
    "miami": -2.0,    # Coastal ceiling effect
    "austin": 0.0,    # Inland, symmetric
    "chicago": -1.0,  # Lake effect
    "nyc": -1.0,
}

# Calibration factors (may not be needed with correct σ)
CALIBRATION_FACTORS = {
    "miami": 0.82,
    "austin": 0.72,
    "chicago": 0.72,
    "nyc": 0.68,
}
CALIBRATION_ENABLED = False  # Try without first, σ fix should be enough

MAX_PROBABILITY = 0.85

# City configs
cities = {
    "miami": {
        "lat": 25.7959,
        "lon": -80.2870,
        "kalshi_series": "KXHIGHMIA",
    },
    "austin": {
        "lat": 30.2,
        "lon": -97.68,
        "kalshi_series": "KXHIGHAUS",
    },
    "chicago": {
        "lat": 41.8781,
        "lon": -87.6298,
        "kalshi_series": "KXHIGHCHI",
    },
    "nyc": {
        "lat": 40.7128,
        "lon": -74.0060,
        "kalshi_series": "KXHIGHNYC",
    },
}

# Open-Meteo APIs
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HIST_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical_forecast(lat, lon, date_str, model):
    """Fetch historical forecast from Open-Meteo"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_max",
        "models": model,
        "temperature_unit": "fahrenheit",
    }
    try:
        response = requests.get(HIST_FORECAST_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("daily", {}).get("temperature_2m_max", [None])[0]
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Failed to fetch {model} forecast: {e}")
    return None


def fetch_historical_actual(lat, lon, date_str):
    """Fetch actual temperature from Open-Meteo archive"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
    }
    try:
        response = requests.get(HIST_WEATHER_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("daily", {}).get("temperature_2m_max", [None])[0]
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Failed to fetch actual temp: {e}")
    return None


def fetch_kalshi_settled_markets(series_ticker, min_date, max_date):
    """Fetch settled markets from Kalshi API with pagination"""
    kalshi_base = "https://api.elections.kalshi.com/trade-api/v2"
    markets = []
    cursor = None
    
    print(f"Fetching Kalshi markets for {series_ticker}...")
    
    while True:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor
            
        try:
            response = requests.get(f"{kalshi_base}/markets", params=params, timeout=15)
            if response.status_code != 200:
                print(f"  Error fetching Kalshi markets: {response.status_code}")
                break
            data = response.json()
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
            break
    
    # Filter by date range
    filtered = []
    for m in markets:
        ticker = m.get("ticker", "")
        date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
        if date_match:
            yy, mmm, dd = date_match.groups()
            mmm = mmm.title()
            date_str = yy + mmm + dd
            try:
                market_date = datetime.strptime(date_str, "%y%b%d").date()
                if min_date <= market_date <= max_date:
                    filtered.append(m)
            except ValueError:
                pass
    
    print(f"  Found {len(filtered)} settled markets in date range")
    return filtered


def find_bin_for_temp(temp, date_markets):
    """
    Find the Kalshi bin that contains the temperature.
    Uses NWS rounding rules and parses actual market subtitles.
    
    Returns: (bin_tuple, market) or (None, None) if not found
    """
    # NWS rounds to nearest integer for settlement
    temp_rounded = int(np.round(temp))
    
    for market in date_markets:
        subtitle = market.get("subtitle", "")
        
        # Parse "X to Y" format (e.g., "80 to 81", "77° to 78°")
        match = re.search(r'(\d+)°?\s*to\s*(\d+)', subtitle)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if low <= temp_rounded <= high:
                return (low, high), market
        
        # Parse "X or below" format
        elif "or below" in subtitle.lower():
            match = re.search(r'(\d+)', subtitle)
            if match:
                threshold = int(match.group(1))
                if temp_rounded <= threshold:
                    return (None, threshold), market
        
        # Parse "X or above" format
        elif "or above" in subtitle.lower():
            match = re.search(r'(\d+)', subtitle)
            if match:
                threshold = int(match.group(1))
                if temp_rounded >= threshold:
                    return (threshold, None), market
    
    return None, None


def get_all_bins_from_markets(date_markets):
    """Extract all bin structures from a day's markets"""
    bins = []
    for market in date_markets:
        subtitle = market.get("subtitle", "")
        
        match = re.search(r'(\d+)°?\s*to\s*(\d+)', subtitle)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            bins.append({"type": "range", "low": low, "high": high, "market": market})
        elif "or below" in subtitle.lower():
            match = re.search(r'(\d+)', subtitle)
            if match:
                bins.append({"type": "below", "threshold": int(match.group(1)), "market": market})
        elif "or above" in subtitle.lower():
            match = re.search(r'(\d+)', subtitle)
            if match:
                bins.append({"type": "above", "threshold": int(match.group(1)), "market": market})
    
    return sorted(bins, key=lambda x: x.get("low", x.get("threshold", 0)))


def calibrated_probability(low, high, forecast, std, city_key):
    """Calculate probability using skewed normal distribution"""
    skew = CITY_SKEW_PARAMS.get(city_key, 0.0)
    
    # Handle edge bins
    if low is None:  # "X or below"
        prob = skewnorm.cdf(high + 0.5, skew, loc=forecast, scale=std)
    elif high is None:  # "X or above"
        prob = 1 - skewnorm.cdf(low - 0.5, skew, loc=forecast, scale=std)
    else:  # Normal "X to Y" bin
        prob = skewnorm.cdf(high + 0.5, skew, loc=forecast, scale=std) - \
               skewnorm.cdf(low - 0.5, skew, loc=forecast, scale=std)
    
    return max(min(prob, 0.99), 0.01)


def apply_calibration(prob, city_key, side='YES'):
    """Apply probability caps and optional calibration"""
    if side == 'YES':
        return min(prob, MAX_PROBABILITY)
    else:  # NO
        raw_no = 1 - min(prob, MAX_PROBABILITY)
        if CALIBRATION_ENABLED:
            factor = CALIBRATION_FACTORS.get(city_key, 0.78)
            return raw_no * factor
        return raw_no


def backtest_city(city_key, start_date, end_date):
    """Run backtest for a single city"""
    city = cities[city_key]
    results = []
    current_date = start_date
    
    # Fetch all settled markets upfront
    settled_markets = fetch_kalshi_settled_markets(city["kalshi_series"], start_date, end_date)
    
    days_processed = 0
    days_skipped_no_forecast = 0
    days_skipped_no_markets = 0
    days_skipped_no_actual = 0
    days_with_kalshi_match = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Fetch historical forecasts from multiple models
        forecasts = {}
        forecasts["GFS"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "gfs_global")
        forecasts["HRRR"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "hrrr")
        forecasts["NWS"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "nbm")
        forecasts["Open-Meteo"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "gfs_seamless")
        
        available = {k: v for k, v in forecasts.items() if v is not None}
        if len(available) < 2:
            days_skipped_no_forecast += 1
            current_date += timedelta(days=1)
            continue
        
        # Calculate weighted ensemble forecast
        weighted_sum = sum(v * ENSEMBLE_WEIGHTS.get(k, 0.25) for k, v in available.items())
        total_weight = sum(ENSEMBLE_WEIGHTS.get(k, 0.25) for k in available)
        raw_ensemble = weighted_sum / total_weight
        bias = STATION_BIAS.get(city_key, 0.0)
        final_forecast = raw_ensemble + bias
        
        # City-specific uncertainty
        std = CITY_UNCERTAINTY.get(city_key, 3.0)
        
        # Find settled markets for this date
        date_ticker = current_date.strftime("%y%b%d").upper()
        date_markets = [m for m in settled_markets if date_ticker in m.get("ticker", "")]
        
        if not date_markets:
            days_skipped_no_markets += 1
            current_date += timedelta(days=1)
            continue
        
        # Fetch actual temperature
        actual_temp = fetch_historical_actual(city["lat"], city["lon"], date_str)
        
        if actual_temp is None:
            days_skipped_no_actual += 1
            current_date += timedelta(days=1)
            continue
        
        # Find the bin that contains the actual temperature (using Kalshi's actual bins)
        actual_bin, bin_market = find_bin_for_temp(actual_temp, date_markets)
        
        if actual_bin is None:
            # Fallback: create synthetic bin based on rounded temp
            temp_rounded = int(np.round(actual_temp))
            # Determine bin parity from available markets
            all_bins = get_all_bins_from_markets(date_markets)
            if all_bins:
                # Check if bins start on even or odd
                first_bin = all_bins[0]
                if first_bin["type"] == "range":
                    start_parity = first_bin["low"] % 2
                    if temp_rounded % 2 == start_parity:
                        actual_bin = (temp_rounded, temp_rounded + 1)
                    else:
                        actual_bin = (temp_rounded - 1, temp_rounded)
                else:
                    # Default to 2°F bin centered on temp
                    actual_bin = (temp_rounded, temp_rounded + 1)
            else:
                actual_bin = (temp_rounded, temp_rounded + 1)
            bin_market = None
        else:
            days_with_kalshi_match += 1
        
        # Calculate model probability for the actual bin
        model_prob = calibrated_probability(actual_bin[0], actual_bin[1], final_forecast, std, city_key)
        calibrated_prob = apply_calibration(model_prob, city_key, 'YES')
        
        # Determine outcome
        if bin_market:
            # Use actual Kalshi settlement
            outcome = 1 if bin_market.get("result") == "yes" else 0
        else:
            # Fallback: check if rounded temp is in bin
            temp_rounded = int(np.round(actual_temp))
            if actual_bin[0] is None:  # "or below"
                outcome = 1 if temp_rounded <= actual_bin[1] else 0
            elif actual_bin[1] is None:  # "or above"
                outcome = 1 if temp_rounded >= actual_bin[0] else 0
            else:
                outcome = 1 if actual_bin[0] <= temp_rounded <= actual_bin[1] else 0
        
        results.append({
            "date": date_str,
            "forecast": round(final_forecast, 2),
            "actual": actual_temp,
            "actual_rounded": int(np.round(actual_temp)),
            "bin_low": actual_bin[0],
            "bin_high": actual_bin[1],
            "model_prob": round(calibrated_prob, 4),
            "outcome": outcome,
            "kalshi_match": 1 if bin_market else 0
        })
        
        days_processed += 1
        current_date += timedelta(days=1)
    
    print(f"\n{city_key.upper()} BACKTEST SUMMARY:")
    print(f"  Days processed: {days_processed}")
    print(f"  Days with Kalshi bin match: {days_with_kalshi_match}")
    print(f"  Skipped (no forecast): {days_skipped_no_forecast}")
    print(f"  Skipped (no markets): {days_skipped_no_markets}")
    print(f"  Skipped (no actual): {days_skipped_no_actual}")
    
    return pd.DataFrame(results)


def analyze_results(df, city_key):
    """Analyze backtest results and print statistics"""
    if df.empty:
        print(f"No results for {city_key}")
        return
    
    print(f"\n{'='*60}")
    print(f"BACKTEST ANALYSIS: {city_key.upper()}")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total days: {len(df)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Forecast accuracy
    forecast_error = df['forecast'] - df['actual']
    print(f"\n📈 FORECAST ACCURACY")
    print(f"   Mean error: {forecast_error.mean():+.2f}°F")
    print(f"   Std dev: {forecast_error.std():.2f}°F")
    print(f"   MAE: {forecast_error.abs().mean():.2f}°F")
    
    # Calibration
    print(f"\n🎯 PROBABILITY CALIBRATION")
    print(f"   Mean model prob: {df['model_prob'].mean()*100:.1f}%")
    print(f"   Actual outcome rate: {df['outcome'].mean()*100:.1f}%")
    gap = (df['model_prob'].mean() - df['outcome'].mean()) * 100
    print(f"   Calibration gap: {gap:+.1f}%")
    
    # Brier score
    brier = np.mean((df['model_prob'] - df['outcome'])**2)
    print(f"\n📉 BRIER SCORE: {brier:.4f}")
    print(f"   (Lower is better. Perfect=0, Random=0.25)")
    
    # Kalshi match rate
    if 'kalshi_match' in df.columns:
        match_rate = df['kalshi_match'].mean() * 100
        print(f"\n🔗 KALSHI BIN MATCH RATE: {match_rate:.1f}%")
    
    # Calibration by probability bucket
    print(f"\n📊 CALIBRATION BY PROBABILITY BUCKET")
    df['prob_bucket'] = pd.cut(df['model_prob'], 
                               bins=[0, 0.15, 0.25, 0.35, 0.50, 1.0],
                               labels=['<15%', '15-25%', '25-35%', '35-50%', '>50%'])
    
    for bucket in ['<15%', '15-25%', '25-35%', '35-50%', '>50%']:
        bucket_df = df[df['prob_bucket'] == bucket]
        if len(bucket_df) >= 3:
            pred = bucket_df['model_prob'].mean() * 100
            actual = bucket_df['outcome'].mean() * 100
            print(f"   {bucket:>8}: Predicted {pred:5.1f}% → Actual {actual:5.1f}% (n={len(bucket_df)})")


def main():
    parser = argparse.ArgumentParser(description="Kalshi Weather Model Backtester v2.0")
    parser.add_argument("--start_date", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--cities", default="miami,austin", help="Comma-separated city list")
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    selected_cities = [c.strip().lower() for c in args.cities.split(",")]
    
    print(f"Kalshi Weather Model Backtester v2.0")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Cities: {', '.join(selected_cities)}")
    print(f"Calibration enabled: {CALIBRATION_ENABLED}")
    print("="*60)
    
    for city in selected_cities:
        if city not in cities:
            print(f"Warning: Unknown city '{city}', skipping")
            continue
            
        df = backtest_city(city, start_date, end_date)
        
        if not df.empty:
            # Save results
            filename = f"backtest_{city}_v2.csv"
            df.to_csv(filename, index=False)
            print(f"\n✅ Results saved to {filename}")
            
            # Analyze
            analyze_results(df, city)
        else:
            print(f"\n❌ No results for {city}")


if __name__ == "__main__":
    main()