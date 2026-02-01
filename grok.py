"""
Kalshi Weather Model Backtester v1.0
=====================================
This script backtests the Kalshi weather betting model using historical data.
It uses Open-Meteo's Historical Forecast API for past forecasts (GFS, HRRR, and NBM as proxy for NWS).
Actual temperatures are fetched from Open-Meteo's Historical Weather API for verification (optional).
Kalshi settled markets are fetched via API to determine outcomes.

Limitations:
- NWS is approximated with NBM (available since Oct 2024), so backtest period is limited.
- Assumes forecasts are from the latest run before the date.
- Paginate Kalshi API if needed for long periods.
- Requires Kalshi API access (public for markets).
- Adjust cities, dates, etc.

Usage:
python backtester.py --start_date 2025-01-01 --end_date 2026-01-31 --cities miami,austin
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import argparse
from scipy.stats import skewnorm
import json
import os
import re  # For date parsing

# Copy relevant configs from the model
ENSEMBLE_WEIGHTS = {
    "NWS": 0.35,  # Approximated with NBM
    "HRRR": 0.25,
    "Open-Meteo": 0.20,  # Blended, use GFS seamless or similar
    "GFS": 0.20
}

STATION_BIAS = {
    "miami": 0.5,
    "austin": 0.5,
    # Add others
}

CITY_SKEW_PARAMS = {
    "miami": -2.0,
    "austin": 0.0,
}

CALIBRATION_FACTORS = {
    "miami": 0.81,
    "austin": 0.76,
}

CALIBRATION_ENABLED = True

MIN_EDGE_REQUIREMENT = 0.20
MAX_PROBABILITY = 0.85

# City configs (same as model)
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
    # Add more
}

# Open-Meteo APIs
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HIST_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"  # For actuals

def fetch_historical_forecast(lat, lon, date_str, model):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_max",
        "models": model,  # e.g., "gfs_global", "hrrr", "nbm"
        "temperature_unit": "fahrenheit",
    }
    response = requests.get(HIST_FORECAST_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("daily", {}).get("temperature_2m_max", [None])[0]
    return None

def fetch_historical_actual(lat, lon, date_str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
    }
    response = requests.get(HIST_WEATHER_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("daily", {}).get("temperature_2m_max", [None])[0]
    return None

def fetch_kalshi_settled_markets(series_ticker, min_date, max_date):
    kalshi_base = "https://api.elections.kalshi.com/trade-api/v2"
    markets = []
    cursor = None
    while True:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor
        response = requests.get(f"{kalshi_base}/markets", params=params)
        if response.status_code != 200:
            print(f"Error fetching Kalshi markets: {response.status_code}")
            break
        data = response.json()
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    # Filter by date
    filtered = []
    for m in markets:
        ticker = m.get("ticker", "")
        # Extract date from ticker, e.g., HIGHMIA-26JAN31-B82
        date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
        if date_match:
            yy, mmm, dd = date_match.groups()
            mmm = mmm.title()  # "JAN" -> "Jan"
            date_str = yy + mmm + dd
            try:
                market_date = datetime.strptime(date_str, "%y%b%d").date()
                if min_date <= market_date <= max_date:
                    filtered.append(m)
            except ValueError:
                pass
    return filtered

def get_skew_parameter(city_key):
    return CITY_SKEW_PARAMS.get(city_key, 0.0)

def calibrated_probability(low, high, forecast, std, city_key):
    skew = get_skew_parameter(city_key)
    prob = skewnorm.cdf(high + 0.5, skew, loc=forecast, scale=std) - \
           skewnorm.cdf(low - 0.5, skew, loc=forecast, scale=std)
    return max(min(prob, 0.99), 0.01)

def apply_calibration(prob, city_key, side):
    if side == 'YES':
        return min(prob, MAX_PROBABILITY)
    else:  # NO
        raw_no = 1 - min(prob, MAX_PROBABILITY)
        if CALIBRATION_ENABLED:
            factor = CALIBRATION_FACTORS.get(city_key, 0.78)
            return raw_no * factor
        return raw_no

def backtest_city(city_key, start_date, end_date):
    city = cities[city_key]
    results = []
    current_date = start_date
    settled_markets = fetch_kalshi_settled_markets(city["kalshi_series"], start_date, end_date)
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Backtesting {city_key} for {date_str}")
        
        # Fetch historical forecasts
        forecasts = {}
        forecasts["GFS"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "gfs_global")
        forecasts["HRRR"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "hrrr")
        forecasts["NWS"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "nbm")  # Proxy
        forecasts["Open-Meteo"] = fetch_historical_forecast(city["lat"], city["lon"], date_str, "gfs_seamless")  # Blended proxy
        
        available = {k: v for k, v in forecasts.items() if v is not None}
        if len(available) < 2:
            print("Insufficient forecasts")
            current_date += timedelta(days=1)
            continue
        
        # Ensemble
        weighted_sum = sum(v * ENSEMBLE_WEIGHTS.get(k, 0.25) for k, v in available.items())
        total_weight = sum(ENSEMBLE_WEIGHTS.get(k, 0.25) for k in available)
        raw_ensemble = weighted_sum / total_weight
        bias = STATION_BIAS.get(city_key, 0.0)
        final_forecast = raw_ensemble + bias
        
        # Uncertainty - simplified, no spread calc for now
        std = 5.0  # Placeholder, tune based on historical
        
        # Find relevant settled markets for this date
        date_ticker = current_date.strftime("%y%b%d").upper()
        date_markets = [m for m in settled_markets if date_ticker in m.get("ticker", "")]
        
        if not date_markets:
            print("No settled markets")
            current_date += timedelta(days=1)
            continue
        
        # Simulate model logic: calculate probs for each market
        for market in date_markets:
            subtitle = market.get("subtitle", "")
            kalshi_prob = (market.get("yes_bid", 0) + market.get("yes_ask", 100)) / 200 / 100  # Historical? Wait, settled has no bids
            # Problem: Settled markets don't have historical prices, only outcome.
            # To backtest edges, need historical market prices before settlement.
            # This is a limitation - Kalshi API doesn't provide historical prices, only current.
            # For true backtest, need historical price data, perhaps from scraper or third-party.
            # Placeholder: Assume we have historical prob, but for now, skip edge calc, just prob accuracy.
        
        # Instead, focus on probability calibration: model's P(YES) for the bin vs. actual outcome
        actual_temp = fetch_historical_actual(city["lat"], city["lon"], date_str)  # Reanalysis approx
        
        if actual_temp is None:
            current_date += timedelta(days=1)
            continue
        
        # Find the actual bin
        actual_bin_low = int(np.floor(actual_temp)) if int(np.floor(actual_temp)) % 2 == 0 else int(np.floor(actual_temp)) - 1
        actual_bin = (actual_bin_low, actual_bin_low + 1)
        
        # Model's forecast bin
        forecast_bin = (int(np.floor(final_forecast)), int(np.floor(final_forecast)) + 1) if int(np.floor(final_forecast)) % 2 == 0 else (int(np.floor(final_forecast)) - 1, int(np.floor(final_forecast)))
        
        # Calc model prob for actual bin
        model_prob = calibrated_probability(actual_bin[0], actual_bin[1], final_forecast, std, city_key)
        calibrated_prob = apply_calibration(model_prob, city_key, 'YES')
        
        # Outcome for that bin: Find the market for the bin
        bin_market = next((m for m in date_markets if f"{actual_bin[0]} to {actual_bin[1]}" in m["subtitle"]), None)
        if bin_market:
            outcome = 1 if bin_market["result"] == "yes" else 0
        else:
            outcome = 1 if actual_bin[0] <= actual_temp <= actual_bin[1] else 0  # Approx with actual
        
        results.append({
            "date": date_str,
            "forecast": final_forecast,
            "actual": actual_temp,
            "model_prob": calibrated_prob,
            "outcome": outcome
        })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", default="2025-01-01")
    parser.add_argument("--end_date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--cities", default="miami,austin")
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    selected_cities = args.cities.split(",")
    
    for city in selected_cities:
        df = backtest_city(city, start_date, end_date)
        df.to_csv(f"backtest_{city}.csv", index=False)
        
        # Calibration metric: Brier score or log loss
        if not df.empty:
            brier = np.mean((df["model_prob"] - df["outcome"])**2)
            print(f"{city} Brier score: {brier:.4f} (lower better, perfect 0)")

if __name__ == "__main__":
    main()