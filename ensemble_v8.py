"""
KALSHI WEATHER BETTING MODEL v8
================================
IMPROVED VERSION based on backtest analysis.

Key changes from v6:
1. PRICE FILTERS: Never bet on contracts below 15¢ or above 90¢
2. SWEET SPOT FOCUS: Prefer contracts in 15-50¢ range
3. DYNAMIC EDGE REQUIREMENTS: Higher bar for extreme prices
4. REMOVED SPIKE WEIGHT: Use pure Gaussian probability model
5. BETTER CALIBRATION: Match actual forecast error distributions

The backtest showed:
- 1-5¢ contracts: -97% ROI (TRAP - AVOID)
- 15-50¢ contracts: Best risk/reward
- High-priced NO bets: Can be profitable with good forecasts
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import requests
from datetime import datetime, timedelta
import re
from scipy import stats

# ============ HRRR/HERBIE IMPORTS ============
try:
    from herbie import Herbie
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False
    print("⚠️ Herbie not installed. Will use Open-Meteo only.")
    print("   Install with: pip install herbie-data xarray cfgrib")

# ============ NEW: PRICE FILTER CONFIGURATION ============
MIN_CONTRACT_PRICE = 0.15   # Never bet on contracts below 15¢
MAX_CONTRACT_PRICE = 0.90   # Never bet on contracts above 90¢
SWEET_SPOT_LOW = 0.15       # Preferred range lower bound
SWEET_SPOT_HIGH = 0.50      # Preferred range upper bound

# Edge requirements by price bucket
def get_min_edge(price, agreement_level):
    """Dynamic edge requirement based on price and model agreement."""
    # Base requirements by price bucket
    if price <= SWEET_SPOT_HIGH:
        base_edge = 0.15  # 15% for sweet spot (15-50¢) - raised for profitability
    else:
        base_edge = 0.12  # 12% for high-priced (50-90¢)
    
    # Adjust for agreement level
    if agreement_level == 'high':
        return base_edge
    elif agreement_level == 'medium':
        return base_edge * 1.25
    else:  # low
        return base_edge * 1.5

def get_price_bucket(price):
    """Categorize price into buckets."""
    if price <= SWEET_SPOT_HIGH:
        return "sweet_spot"
    else:
        return "high_price"

# ============ CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - only bet if models agree within this range
CONFIDENCE_BOOST_THRESHOLD = 2.0   # °F - boost confidence if within this range

# Calibrated forecast error (based on HRRR typical accuracy)
CALIBRATED_FORECAST_STD = 2.8  # °F - typical day-ahead forecast error

# ============ CITY CONFIGURATION ============
cities = {
    "chicago": {
        "name": "Chicago",
        "csv_file": "weather_data_chicago.csv",
        "lat": 41.8781,
        "lon": -87.6298,
        "kalshi_series": "KXHIGHCHI",
        "timezone": "America/Chicago",
        "utc_offset": -6  # CST
    },
    "nyc": {
        "name": "New York City",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.7128,
        "lon": -74.0060,
        "kalshi_series": "KXHIGHNY",
        "timezone": "America/New_York",
        "utc_offset": -5  # EST
    },
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.7959,
        "lon": -80.2870,
        "kalshi_series": "KXHIGHMIA",
        "timezone": "America/New_York",
        "utc_offset": -5  # EST
    }
}
all_bets = []

# Default to all cities
selected_cities = ["chicago", "nyc", "miami"]

# ============ HEADER ============
print("=" * 70)
print("KALSHI WEATHER BETTING MODEL v8")
print("(Improved: Price Filters + Calibrated Probabilities)")
print("=" * 70)
print(f"\nAnalyzing: Chicago, New York City, Miami")
print(f"Agreement Threshold: ±{ENSEMBLE_AGREEMENT_THRESHOLD}°F")
print(f"Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢ - {MAX_CONTRACT_PRICE*100:.0f}¢")
print(f"Sweet Spot: {SWEET_SPOT_LOW*100:.0f}¢ - {SWEET_SPOT_HIGH*100:.0f}¢")
print(f"Data Sources: NOAA HRRR (3km) + Open-Meteo")

# ============ FUNCTIONS ============

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


def fetch_hrrr_forecast_fixed(lat, lon, target_date, utc_offset):
    """
    Fetch HRRR forecast with FIXED sampling to get actual daily high.
    """
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
    """Get the Kalshi bin that contains this temperature."""
    lower = int(temp) if int(temp) % 2 == 0 else int(temp) - 1
    if temp < 0:
        lower = int(temp) - 1 if int(temp) % 2 == 0 else int(temp)
    
    if lower % 2 == 0:
        lower -= 1
    
    return (lower, lower + 1)


# ============ NEW: CALIBRATED PROBABILITY MODEL ============
# Removed the spike weight - use pure Gaussian based on forecast error distribution

def calibrated_probability(contract_low, contract_high, forecast, uncertainty_std, agreement_level='high'):
    """
    IMPROVED probability model using calibrated Gaussian distribution.
    No artificial spike weight - just the statistical distribution of forecast errors.
    """
    # Adjust uncertainty based on agreement level
    if agreement_level == 'high':
        adj_std = uncertainty_std * 0.9
    elif agreement_level == 'medium':
        adj_std = uncertainty_std * 1.0
    else:  # low
        adj_std = uncertainty_std * 1.3
    
    # Pure Gaussian probability
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


def analyze_city(city_key):
    """Main analysis function for a single city using improved model."""
    city = cities[city_key]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {city['name'].upper()}")
    print(f"{'='*70}")
    
    # Load historical data
    try:
        df = load_and_prepare_data(city)
        print(f"Loaded {len(df)} historical records")
    except FileNotFoundError:
        print(f"❌ Error: {city['csv_file']} not found.")
        return
    
    # Set target date
    today = datetime.now().date()
    target_date = today + timedelta(days=1)
    target_str = target_date.strftime("%Y-%m-%d")
    
    # ============ FETCH BOTH FORECASTS ============
    print(f"\n  📡 Fetching forecasts for {target_str}...")
    
    # 1. Open-Meteo forecast
    print(f"\n  [1] Open-Meteo:")
    meteo_data = fetch_open_meteo(city["lat"], city["lon"], city["timezone"])
    meteo_dates = meteo_data["daily"]["time"]
    meteo_temps = meteo_data["daily"]["temperature_2m_max"]
    meteo_wind = meteo_data["daily"]["wind_speed_10m_max"]
    meteo_wind_dir = meteo_data["daily"]["wind_direction_10m_dominant"]
    
    # Find target date in Open-Meteo data
    open_meteo_forecast = None
    open_meteo_wind = None
    for i, d in enumerate(meteo_dates):
        if d == target_str:
            open_meteo_forecast = meteo_temps[i]
            open_meteo_wind = meteo_wind[i]
            print(f"     ✅ Forecast: {open_meteo_forecast:.1f}°F")
            if open_meteo_wind:
                print(f"     💨 Wind: {open_meteo_wind:.1f} mph")
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
        print(f"     ⚠️ HRRR unavailable, using Open-Meteo only")
    
    # ============ ENSEMBLE FORECAST ============
    if hrrr_forecast and open_meteo_forecast:
        ensemble_forecast = (hrrr_forecast * 0.6) + (open_meteo_forecast * 0.4)
        model_diff = abs(hrrr_forecast - open_meteo_forecast)
        
        if model_diff <= CONFIDENCE_BOOST_THRESHOLD:
            agreement_level = 'high'
        elif model_diff <= ENSEMBLE_AGREEMENT_THRESHOLD:
            agreement_level = 'medium'
        else:
            agreement_level = 'low'
        
        print(f"\n  📊 Model Comparison:")
        print(f"     HRRR: {hrrr_forecast:.1f}°F")
        print(f"     Open-Meteo: {open_meteo_forecast:.1f}°F")
        print(f"     Difference: {model_diff:.1f}°F → Agreement: {agreement_level.upper()}")
    elif open_meteo_forecast:
        ensemble_forecast = open_meteo_forecast
        agreement_level = 'medium'
        print(f"\n  📊 Using Open-Meteo only: {ensemble_forecast:.1f}°F")
    else:
        print(f"\n  ❌ No forecasts available!")
        return
    
    print(f"\n  🎯 ENSEMBLE FORECAST: {ensemble_forecast:.1f}°F")
    
    forecast_bin = get_bin_for_temp(ensemble_forecast)
    print(f"     Primary bin: {forecast_bin[0]}°-{forecast_bin[1]}°F")
    
    # ============ FETCH KALSHI DATA ============
    print(f"\n  💰 Fetching Kalshi market data...")
    
    kalshi_base = "https://api.elections.kalshi.com/trade-api/v2"
    
    # Use the /markets endpoint with series_ticker (correct API approach)
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
            return
        
        all_markets = markets_response.json().get("markets", [])
        
        # Filter to target date
        target_kalshi = target_date.strftime("%y%b%d").upper()
        markets = [m for m in all_markets if target_kalshi in m.get("ticker", "")]
        
        if not markets:
            print(f"     ❌ No markets found for {target_str}")
            print(f"        (Looking for ticker containing: {target_kalshi})")
            # Show what dates ARE available
            available_dates = set()
            for m in all_markets[:20]:
                ticker = m.get("ticker", "")
                # Extract date part from ticker like "KXHIGHCHI-25DEC05-B29"
                parts = ticker.split("-")
                if len(parts) >= 2:
                    available_dates.add(parts[1])
            if available_dates:
                print(f"        Available dates: {', '.join(sorted(available_dates)[:5])}")
            return
        
        print(f"     ✅ Found {len(markets)} contracts for {target_str}")
        
    except Exception as e:
        print(f"     ❌ API error: {e}")
        return
    
    # ============ ANALYZE CONTRACTS (WITH PRICE FILTERS) ============
    print(f"\n  {'='*60}")
    print(f"  CONTRACT ANALYSIS (Filtered: {MIN_CONTRACT_PRICE*100:.0f}¢-{MAX_CONTRACT_PRICE*100:.0f}¢)")
    print(f"  {'='*60}")
    print(f"  {'Contract':<18} {'Our Prob':>9} {'Kalshi':>9} {'Edge':>9} {'Status':<12}")
    print(f"  {'-'*58}")
    
    results = []
    skipped_cheap = 0
    skipped_expensive = 0
    
    kalshi_top_prob = 0
    kalshi_top_contract = ""
    
    for market in markets:
        ticker = market.get("ticker", "")
        subtitle = market.get("subtitle", "")
        
        # /markets endpoint uses yes_bid, yes_ask OR last_price
        yes_bid = market.get("yes_bid", 0) or 0
        yes_ask = market.get("yes_ask", 100) or 100
        last_price = market.get("last_price", 0) or 0
        
        # Use mid-point of bid/ask, or last_price if bid/ask not available
        if yes_bid > 0 and yes_ask < 100:
            kalshi_prob = (yes_bid + yes_ask) / 200
        elif last_price > 0:
            kalshi_prob = last_price / 100
        else:
            continue  # No price data
        
        # Track market's top pick
        if kalshi_prob > kalshi_top_prob:
            kalshi_top_prob = kalshi_prob
            kalshi_top_contract = subtitle
        
        # === PRICE FILTERS ===
        if kalshi_prob < MIN_CONTRACT_PRICE:
            skipped_cheap += 1
            continue
        if kalshi_prob > MAX_CONTRACT_PRICE:
            skipped_expensive += 1
            continue
        
        # Parse contract and calculate probability
        numbers = re.findall(r'-?\d+', subtitle)
        contract_type = None
        
        if "to" in subtitle and len(numbers) >= 2:
            low = int(numbers[0])
            high = int(numbers[1])
            model_prob = calibrated_probability(low, high, ensemble_forecast, 
                                                CALIBRATED_FORECAST_STD, agreement_level)
            is_forecast_bin = forecast_bin[0] <= ensemble_forecast <= forecast_bin[1] and low == forecast_bin[0]
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
        
        edge = model_prob - kalshi_prob
        price_bucket = get_price_bucket(kalshi_prob)
        
        results.append({
            "subtitle": subtitle,
            "model_prob": model_prob,
            "kalshi_prob": kalshi_prob,
            "edge": edge,
            "is_forecast_bin": is_forecast_bin,
            "price_bucket": price_bucket,
            "contract_type": contract_type
        })
        
        # Status indicator
        bucket_emoji = "🎯" if price_bucket == "sweet_spot" else "📊"
        marker = "📍" if is_forecast_bin else "  "
        print(f"  {marker}{subtitle:<16} {model_prob:>9.1%} {kalshi_prob:>9.0%} {edge:>+9.1%} {bucket_emoji} {price_bucket}")
    
    print(f"\n  Filtered out: {skipped_cheap} cheap (<{MIN_CONTRACT_PRICE*100:.0f}¢), {skipped_expensive} expensive (>{MAX_CONTRACT_PRICE*100:.0f}¢)")
    
    # ============ BETTING RECOMMENDATIONS ============
    print(f"\n  {'='*60}")
    print("  BETTING RECOMMENDATIONS (Improved v8)")
    print(f"  {'='*60}\n")
    
    # Bankroll settings
    base_bankroll = 20
    if agreement_level == 'high':
        bankroll = base_bankroll
    elif agreement_level == 'medium':
        bankroll = base_bankroll * 0.75
    else:
        bankroll = base_bankroll * 0.5
        print(f"  ⚠️ LOW AGREEMENT: Reduced bankroll to ${bankroll:.2f}")
    
    good_bets = []
    
    for r in sorted(results, key=lambda x: abs(x["edge"]), reverse=True):
        edge = r["edge"]
        kalshi_prob = r["kalshi_prob"]
        
        # Never bet NO on the forecast bin
        if edge < 0 and r["is_forecast_bin"]:
            continue
        
        # Determine bet side and effective price
        if edge > 0:
            bet_side = "YES"
            market_price = kalshi_prob
        else:
            bet_side = "NO"
            market_price = 1 - kalshi_prob
            edge = abs(edge)  # Make positive for comparison
        
        # Skip NO range bets in sweet spot (they lose money based on backtest)
        no_bucket = get_price_bucket(market_price)
        if bet_side == "NO" and r.get("contract_type") == "range" and no_bucket == "sweet_spot":
            continue
        
        # Get dynamic minimum edge for this price
        min_edge = get_min_edge(market_price, agreement_level)
        
        if edge < min_edge:
            continue
        
        # Calculate bet sizing (half Kelly)
        if market_price > 0:
            our_prob = r["model_prob"] if bet_side == "YES" else (1 - r["model_prob"])
            odds = (1 / market_price) - 1
            
            kelly_fraction = (our_prob * odds - (1 - our_prob)) / odds if odds > 0 else 0
            half_kelly = max(kelly_fraction / 2, 0)
            
            if half_kelly > 0.02:
                bet_amount = bankroll * half_kelly
                potential_profit = bet_amount * odds
                
                good_bets.append({
                    "city": city["name"],
                    "subtitle": r["subtitle"],
                    "bet_side": bet_side,
                    "market_price": market_price,
                    "model_prob": r["model_prob"],
                    "our_prob": our_prob,
                    "odds": odds,
                    "half_kelly": half_kelly,
                    "bet_amount": bet_amount,
                    "potential_profit": potential_profit,
                    "edge": edge,
                    "is_forecast_bin": r["is_forecast_bin"],
                    "agreement_level": agreement_level,
                    "price_bucket": r["price_bucket"],
                    "min_edge_required": min_edge
                })
    
    if len(good_bets) == 0:
        print("  🛑 NO BETS RECOMMENDED")
        if agreement_level == 'low':
            print("     Models disagree significantly - waiting for convergence")
        else:
            print("     No contracts with sufficient edge after price filtering")
    else:
        # Sort by price bucket (sweet spot first) then by edge
        bucket_order = {"sweet_spot": 0, "low_price": 1, "high_price": 2}
        good_bets.sort(key=lambda x: (bucket_order[x["price_bucket"]], -x["edge"]))
        
        for bet in good_bets:
            bucket_emoji = "🎯" if bet["price_bucket"] == "sweet_spot" else "📊"
            forecast_marker = " 📍" if bet["is_forecast_bin"] else ""
            
            print(f"  {bucket_emoji} {bet['subtitle']}:{forecast_marker}")
            print(f"     Bet: {bet['bet_side']} at {bet['market_price']:.0%}")
            print(f"     Our prob: {bet['our_prob']:.1%} | Edge: {bet['edge']:+.1%} (min: {bet['min_edge_required']:.0%})")
            print(f"     Suggested: ${bet['bet_amount']:.2f} → Potential: ${bet['potential_profit']:.2f}")
            print()
    
    # Summary
    print(f"  {'='*60}")
    print(f"  SUMMARY - {city['name'].upper()}")
    print(f"  {'='*60}")
    print(f"  Target Date: {target_str}")
    if open_meteo_forecast and hrrr_forecast:
        print(f"  Open-Meteo: {open_meteo_forecast:.1f}°F | HRRR: {hrrr_forecast:.1f}°F")
    print(f"  Ensemble Forecast: {ensemble_forecast:.1f}°F → Bin: {forecast_bin[0]}-{forecast_bin[1]}°")
    print(f"  Agreement Level: {agreement_level.upper()}")
    print(f"  Kalshi Top Pick: {kalshi_top_contract} ({kalshi_top_prob*100:.0f}%)")
    print(f"  Contracts in range: {len(results)} (filtered {skipped_cheap + skipped_expensive})")
    print(f"  Recommended Bets: {len(good_bets)}")
    
    all_bets.extend(good_bets)


# ============ RUN ANALYSIS ============
if __name__ == "__main__":
    for city_key in selected_cities:
        analyze_city(city_key)
    
    # ============ BETTING SUMMARY ============
    target_date = (datetime.now().date() + timedelta(days=1)).strftime("%A, %B %d, %Y")
    print(f"\n{'='*70}")
    print(f"💰 BETTING SUMMARY - ALL CITIES (v8 Improved)")
    print(f"📅 Target Date: {target_date}")
    print(f"🎯 Price Filter: {MIN_CONTRACT_PRICE*100:.0f}¢-{MAX_CONTRACT_PRICE*100:.0f}¢")
    print(f"{'='*70}\n")

    if len(all_bets) == 0:
        print("No recommended bets today.")
        print("\nThis could mean:")
        print("  • No contracts in the price sweet spot with sufficient edge")
        print("  • Models disagree significantly")
        print("  • Market is efficiently priced")
    else:
        # Group by price bucket
        sweet_spot_bets = [b for b in all_bets if b['price_bucket'] == 'sweet_spot']
        other_bets = [b for b in all_bets if b['price_bucket'] != 'sweet_spot']
        
        total_suggested = 0
        total_potential = 0
        
        print(f"{'City':<12} {'Contract':<15} {'Bet':<6} {'Price':<8} {'Edge':<8} {'Bucket':<14} {'Wager':<10} {'Profit':<10}")
        print("-" * 95)
        
        for bet in all_bets:
            bucket_marker = "🎯" if bet['price_bucket'] == 'sweet_spot' else "📊"
            print(f"{bet['city']:<12} {bet['subtitle']:<15} {bet['bet_side']:<6} {bet['market_price']*100:>5.0f}¢   {bet['edge']:>+5.1%}   {bucket_marker} {bet['price_bucket']:<10} ${bet['bet_amount']:>6.2f}    ${bet['potential_profit']:>6.2f}")
            total_suggested += bet["bet_amount"]
            total_potential += bet["potential_profit"]
        
        print("-" * 95)
        print(f"TOTAL: ${total_suggested:.2f} wagered → ${total_potential:.2f} potential profit")
        
        if sweet_spot_bets:
            print(f"\n🎯 SWEET SPOT BETS ({len(sweet_spot_bets)}): These are the highest-confidence opportunities")
        
        print(f"\n📊 Legend: 🎯 = Sweet spot (15-50¢), 📊 = Outside sweet spot")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE (v8 - Improved Price Filtering)")
    print(f"{'='*70}")