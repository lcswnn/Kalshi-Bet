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
    print("⚠️ Herbie not installed. Install with: pip install herbie-data")
    print("   Also need: pip install xarray cfgrib")

# ============ PHYSICAL CONSTANTS ============
# These are used in the primitive equations
EARTH_ROTATION = 7.2921e-5  # rad/s (Ω)
R_EARTH = 6.371e6  # meters

def coriolis_parameter(lat):
    """Calculate Coriolis parameter f = 2Ω sin(φ)"""
    return 2 * EARTH_ROTATION * np.sin(np.radians(lat))

# ============ CITY CONFIGURATION ============
cities = {
    "chicago": {
        "name": "Chicago",
        "csv_file": "weather_data_chicago.csv",
        "lat": 41.8781,
        "lon": -87.6298,
        "kalshi_series": "KXHIGHCHI"
    },
    "nyc": {
        "name": "New York City",
        "csv_file": "weather_data_nyc.csv",
        "lat": 40.7128,
        "lon": -74.0060,
        "kalshi_series": "KXHIGHNY"
    },
    "miami": {
        "name": "Miami",
        "csv_file": "weather_data_miami.csv",
        "lat": 25.7959,
        "lon": -80.2870,
        "kalshi_series": "KXHIGHMIA"
    }
}
all_bets = []

# Default to all cities
selected_cities = ["chicago", "nyc", "miami"]

# ============ RUN ALL CITIES ============
print("=" * 60)
print("KALSHI WEATHER BETTING MODEL v5")
print("(HRRR + Temperature Advection Physics)")
print("=" * 60)
print(f"\nAnalyzing: Chicago, New York City, Miami")
print(f"Data Source: NOAA HRRR (3km resolution)")
print(f"Physics: Primitive equations for temperature advection")

# ============ FUNCTIONS ============
def load_and_prepare_data(city_config):
    """Load historical temperature data from CSV."""
    df = pd.read_csv(city_config["csv_file"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "value"]].copy()
    df = df.rename(columns={"value": "temp"})
    return df


def calculate_temperature_advection(ds_temp, ds_u, ds_v, lat, lon, grid_spacing=3000):
    """
    Calculate temperature advection using the primitive equation:
    
    ∂T/∂t = -u(∂T/∂x) - v(∂T/∂y) + other terms
    
    Temperature advection = -u(∂T/∂x) - v(∂T/∂y)
    
    Negative advection = cold air moving in (cooling)
    Positive advection = warm air moving in (warming)
    
    Parameters:
    -----------
    ds_temp : xarray Dataset with temperature field
    ds_u : xarray Dataset with u-wind component
    ds_v : xarray Dataset with v-wind component
    lat, lon : target location
    grid_spacing : HRRR grid spacing in meters (3km = 3000m)
    
    Returns:
    --------
    advection_rate : Temperature advection in °F/hour
    advection_info : Dict with detailed advection components
    """
    try:
        # Get the data arrays
        temp_data = ds_temp['t2m'].values  # Temperature in Kelvin
        u_data = ds_u['u10'].values if 'u10' in ds_u else ds_u['u'].values  # m/s
        v_data = ds_v['v10'].values if 'v10' in ds_v else ds_v['v'].values  # m/s
        
        lats = ds_temp['latitude'].values
        lons = ds_temp['longitude'].values
        
        # Convert target longitude to HRRR convention
        target_lon = lon if lon > 0 else lon + 360
        
        # Find nearest grid point
        dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        
        # Make sure we have enough points for gradient calculation
        if i < 2 or i >= temp_data.shape[0] - 2 or j < 2 or j >= temp_data.shape[1] - 2:
            return None, None
        
        # Get temperature at target and surrounding points
        T_center = temp_data[i, j]
        T_east = temp_data[i, j+1]
        T_west = temp_data[i, j-1]
        T_north = temp_data[i-1, j]  # Note: i decreases going north in most grids
        T_south = temp_data[i+1, j]
        
        # Get wind components at target
        u = u_data[i, j]  # East-west wind (positive = from west)
        v = v_data[i, j]  # North-south wind (positive = from south)
        
        # Calculate temperature gradients (centered difference)
        # ∂T/∂x ≈ (T_east - T_west) / (2 * dx)
        dT_dx = (T_east - T_west) / (2 * grid_spacing)  # K/m
        
        # ∂T/∂y ≈ (T_north - T_south) / (2 * dy)
        dT_dy = (T_north - T_south) / (2 * grid_spacing)  # K/m
        
        # Temperature advection: -u(∂T/∂x) - v(∂T/∂y)
        # Units: (m/s) * (K/m) = K/s
        advection_x = -u * dT_dx  # Zonal advection
        advection_y = -v * dT_dy  # Meridional advection
        total_advection = advection_x + advection_y  # K/s
        
        # Convert to °F/hour for interpretability
        # K/s → K/hr: multiply by 3600
        # K → °F: multiply by 9/5 (same scale factor)
        advection_f_per_hour = total_advection * 3600 * (9/5)
        
        # Calculate wind speed and direction
        wind_speed = np.sqrt(u**2 + v**2)
        wind_dir = (np.arctan2(-u, -v) * 180 / np.pi) % 360
        
        # Temperature gradient magnitude and direction
        grad_magnitude = np.sqrt(dT_dx**2 + dT_dy**2) * 1000  # K per km
        grad_dir = (np.arctan2(-dT_dx, -dT_dy) * 180 / np.pi) % 360
        
        advection_info = {
            'total_advection': advection_f_per_hour,
            'zonal_advection': advection_x * 3600 * (9/5),
            'meridional_advection': advection_y * 3600 * (9/5),
            'u_wind': u * 2.237,  # Convert to mph
            'v_wind': v * 2.237,
            'wind_speed': wind_speed * 2.237,
            'wind_direction': wind_dir,
            'temp_gradient_magnitude': grad_magnitude * (9/5),  # °F per km
            'temp_gradient_direction': grad_dir,
            'dT_dx': dT_dx * 1000 * (9/5),  # °F/km
            'dT_dy': dT_dy * 1000 * (9/5),  # °F/km
            'temp_center_f': (T_center - 273.15) * 9/5 + 32,
            'temp_east_f': (T_east - 273.15) * 9/5 + 32,
            'temp_west_f': (T_west - 273.15) * 9/5 + 32,
            'temp_north_f': (T_north - 273.15) * 9/5 + 32,
            'temp_south_f': (T_south - 273.15) * 9/5 + 32,
        }
        
        return advection_f_per_hour, advection_info
        
    except Exception as e:
        print(f"     ⚠️ Error calculating advection: {str(e)[:50]}")
        return None, None


def fetch_hrrr_with_physics(lat, lon, target_date):
    """
    Fetch HRRR forecast data with full physics including temperature advection.
    
    This extracts:
    - 2m Temperature (TMP:2 m)
    - 10m U-wind component (UGRD:10 m)
    - 10m V-wind component (VGRD:10 m)
    
    And calculates temperature advection from the primitive equations.
    """
    if not HERBIE_AVAILABLE:
        return None, None
    
    today = datetime.now().date()
    
    # Determine which model run to use
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
    
    print(f"\n  📡 Fetching HRRR data with physics...")
    print(f"     Model run: {model_run_str} UTC")
    
    # Calculate forecast hours
    model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
    target_noon = datetime.combine(target_date, datetime.min.time().replace(hour=12))
    target_evening = datetime.combine(target_date, datetime.min.time().replace(hour=18))
    
    hours_to_noon = int((target_noon - model_run_datetime).total_seconds() / 3600)
    hours_to_evening = int((target_evening - model_run_datetime).total_seconds() / 3600)
    
    # Sample every 3 hours
    forecast_hours = list(range(max(1, hours_to_noon - 6), min(48, hours_to_evening + 1), 3))
    
    print(f"     Forecast hours: {forecast_hours[0]}h to {forecast_hours[-1]}h")
    
    temperatures = []
    advection_values = []
    wind_data = []
    
    for fxx in forecast_hours:
        try:
            H = Herbie(
                model_run_str,
                model='hrrr',
                product='sfc',
                fxx=fxx
            )
            
            # Get temperature
            ds_temp = H.xarray("TMP:2 m", remove_grib=True)
            
            # Get wind components for advection calculation
            try:
                ds_u = H.xarray("UGRD:10 m", remove_grib=True)
                ds_v = H.xarray("VGRD:10 m", remove_grib=True)
                
                # Calculate temperature advection
                advection, adv_info = calculate_temperature_advection(
                    ds_temp, ds_u, ds_v, lat, lon
                )
                
                if advection is not None:
                    advection_values.append({
                        'hour': fxx,
                        'advection': advection,
                        'info': adv_info
                    })
                    wind_data.append(adv_info)
            except Exception as e:
                pass
            
            # Extract temperature at location
            temp_data = ds_temp['t2m']
            lats = ds_temp['latitude'].values
            lons = ds_temp['longitude'].values
            
            target_lon = lon if lon > 0 else lon + 360
            dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            temp_k = float(temp_data.values[min_idx])
            temp_f = (temp_k - 273.15) * 9/5 + 32
            temperatures.append({'hour': fxx, 'temp': temp_f})
            
        except Exception as e:
            print(f"     ⚠️ Error fetching hour {fxx}: {str(e)[:50]}")
            continue
    
    if not temperatures:
        print("     ❌ No HRRR data retrieved")
        return None, None
    
    # Calculate statistics
    temp_values = [t['temp'] for t in temperatures]
    forecast_high = max(temp_values)
    forecast_low = min(temp_values)
    forecast_mean = np.mean(temp_values)
    
    print(f"     ✅ Retrieved {len(temperatures)} temperature samples")
    print(f"     📊 Temperature range: {forecast_low:.1f}°F to {forecast_high:.1f}°F")
    
    # Calculate advection statistics
    avg_advection = None
    max_advection = None
    min_advection = None
    advection_trend = None
    
    if advection_values:
        adv_vals = [a['advection'] for a in advection_values]
        avg_advection = np.mean(adv_vals)
        max_advection = max(adv_vals)
        min_advection = min(adv_vals)
        
        # Trend: is advection becoming more negative (cooling) or positive (warming)?
        if len(adv_vals) >= 3:
            advection_trend = np.polyfit(range(len(adv_vals)), adv_vals, 1)[0]
        
        print(f"     🌡️ Advection: avg={avg_advection:.2f}°F/hr, range=[{min_advection:.2f}, {max_advection:.2f}]°F/hr")
    
    # Wind statistics
    avg_wind_speed = None
    avg_wind_dir = None
    if wind_data:
        avg_wind_speed = np.mean([w['wind_speed'] for w in wind_data])
        # Circular mean for wind direction
        u_mean = np.mean([np.sin(np.radians(w['wind_direction'])) for w in wind_data])
        v_mean = np.mean([np.cos(np.radians(w['wind_direction'])) for w in wind_data])
        avg_wind_dir = (np.arctan2(u_mean, v_mean) * 180 / np.pi) % 360
    
    weather_data = {
        'forecast_high': forecast_high,
        'forecast_low': forecast_low,
        'forecast_mean': forecast_mean,
        'forecast_temps': temperatures,
        'wind_speed': avg_wind_speed,
        'wind_dir': avg_wind_dir,
        'model_run': model_run_str,
        'advection': {
            'average': avg_advection,
            'max': max_advection,
            'min': min_advection,
            'trend': advection_trend,
            'values': advection_values
        },
        'physics_data': wind_data
    }
    
    return forecast_high, weather_data


def interpret_advection(advection_data, forecast_temp):
    """
    Interpret the temperature advection data to adjust forecast confidence.
    
    Returns:
    --------
    adjustment : dict with confidence and uncertainty adjustments
    """
    if advection_data is None or advection_data.get('average') is None:
        return {
            'cold_front': False,
            'warm_front': False,
            'uncertainty_multiplier': 1.0,
            'spike_adjustment': 0.0,
            'description': 'No advection data available'
        }
    
    avg_adv = advection_data['average']
    min_adv = advection_data['min']
    max_adv = advection_data['max']
    trend = advection_data.get('trend', 0)
    
    adjustment = {
        'cold_front': False,
        'warm_front': False,
        'uncertainty_multiplier': 1.0,
        'spike_adjustment': 0.0,
        'description': ''
    }
    
    # Strong cold advection (< -2°F/hr average)
    if avg_adv < -2:
        adjustment['cold_front'] = True
        adjustment['uncertainty_multiplier'] = 1.4  # Much more uncertain
        adjustment['spike_adjustment'] = -0.10  # Less confident in forecast
        adjustment['description'] = f'🥶 STRONG COLD ADVECTION ({avg_adv:.1f}°F/hr)'
    
    # Moderate cold advection
    elif avg_adv < -1:
        adjustment['cold_front'] = True
        adjustment['uncertainty_multiplier'] = 1.2
        adjustment['spike_adjustment'] = -0.05
        adjustment['description'] = f'❄️ Cold advection ({avg_adv:.1f}°F/hr)'
    
    # Strong warm advection (> 2°F/hr average)
    elif avg_adv > 2:
        adjustment['warm_front'] = True
        adjustment['uncertainty_multiplier'] = 1.3
        adjustment['spike_adjustment'] = -0.05
        adjustment['description'] = f'🔥 STRONG WARM ADVECTION ({avg_adv:.1f}°F/hr)'
    
    # Moderate warm advection
    elif avg_adv > 1:
        adjustment['warm_front'] = True
        adjustment['uncertainty_multiplier'] = 1.1
        adjustment['spike_adjustment'] = 0.0
        adjustment['description'] = f'🌡️ Warm advection ({avg_adv:.1f}°F/hr)'
    
    # Weak/neutral advection - highest confidence
    else:
        adjustment['uncertainty_multiplier'] = 0.95  # Slightly more confident
        adjustment['spike_adjustment'] = 0.03
        adjustment['description'] = f'✅ Stable conditions ({avg_adv:.1f}°F/hr)'
    
    # Check for changing advection pattern (trend)
    if trend is not None:
        if trend < -0.5:  # Advection becoming more negative
            adjustment['description'] += ' | Cooling trend'
            adjustment['uncertainty_multiplier'] *= 1.1
        elif trend > 0.5:  # Advection becoming more positive
            adjustment['description'] += ' | Warming trend'
            adjustment['uncertainty_multiplier'] *= 1.1
    
    return adjustment


def fetch_hrrr_recent_temps(lat, lon, num_days=7):
    """Fetch recent observed temperatures from HRRR analysis files."""
    if not HERBIE_AVAILABLE:
        return []
    
    recent_temps = []
    today = datetime.now().date()
    
    print(f"\n  📡 Fetching recent HRRR observations...")
    
    for days_ago in range(1, num_days + 1):
        date = today - timedelta(days=days_ago)
        date_str = date.strftime('%Y-%m-%d')
        
        daily_temps = []
        
        for hour in [15, 18, 21]:
            try:
                H = Herbie(
                    f"{date_str} {hour:02d}:00",
                    model='hrrr',
                    product='sfc',
                    fxx=0
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
                daily_temps.append(temp_f)
                
            except Exception as e:
                continue
        
        if daily_temps:
            daily_high = max(daily_temps)
            recent_temps.append({
                'date': date,
                'temp': daily_high
            })
            print(f"     {date_str}: {daily_high:.1f}°F")
    
    return recent_temps


def build_features(df):
    """Build ML features from historical data."""
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["temp_lag1"] = df["temp"].shift(1)
    df["temp_lag2"] = df["temp"].shift(2)
    df["temp_lag3"] = df["temp"].shift(3)
    df["temp_rolling3"] = df["temp"].shift(1).rolling(window=3).mean()
    df["temp_rolling7"] = df["temp"].shift(1).rolling(window=7).mean()
    df["temp_change"] = df["temp_lag1"] - df["temp_lag2"]
    
    seasonal_avg = df.groupby("day_of_year")["temp"].mean().to_dict()
    df["seasonal_avg"] = df["day_of_year"].map(seasonal_avg)
    df["simulated_forecast"] = (df["temp_lag1"] * 0.6) + (df["seasonal_avg"] * 0.4)
    
    return df.dropna().reset_index(drop=True)


def train_model(df):
    """Train the gradient boosting model."""
    feature_cols = [
        "day_of_year", "month", "day", 
        "temp_lag1", "temp_lag2", "temp_lag3", 
        "temp_rolling3", "temp_rolling7", "temp_change",
        "simulated_forecast"
    ]
    
    X = df[feature_cols]
    y = df["temp"]
    
    split_idx = int(len(df) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    val_predictions = model.predict(X_val)
    model_error_std = np.std(y_val.values - val_predictions)
    mae = mean_absolute_error(y_val, val_predictions)
    
    model.fit(X, y)
    
    return model, model_error_std, mae, feature_cols


def get_bin_for_temp(temp):
    """Get the Kalshi bin that contains this temperature."""
    lower = int(temp) if int(temp) % 2 == 0 else int(temp) - 1
    if temp < 0:
        lower = int(temp) - 1 if int(temp) % 2 == 0 else int(temp)
    
    if lower % 2 == 0:
        lower -= 1
    
    return (lower, lower + 1)


def mixture_probability(contract_low, contract_high, forecast_temp, uncertainty_std, 
                        spike_weight=0.55, physics_adjustment=None):
    """
    Mixture Model with physics-based adjustments.
    """
    if physics_adjustment:
        spike_weight = spike_weight + physics_adjustment.get('spike_adjustment', 0)
        spike_weight = max(0.3, min(0.7, spike_weight))  # Clamp between 0.3 and 0.7
        uncertainty_std = uncertainty_std * physics_adjustment.get('uncertainty_multiplier', 1.0)
    
    spread_weight = 1 - spike_weight
    forecast_bin = get_bin_for_temp(forecast_temp)
    
    forecast_in_this_bin = contract_low <= forecast_temp <= contract_high
    
    spread_prob = stats.norm.cdf(contract_high, forecast_temp, uncertainty_std) - \
                  stats.norm.cdf(contract_low, forecast_temp, uncertainty_std)
    
    if forecast_in_this_bin:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def mixture_below_probability(threshold, forecast_temp, uncertainty_std, 
                              spike_weight=0.55, physics_adjustment=None):
    """Mixture probability for 'X or below' contracts with physics."""
    if physics_adjustment:
        spike_weight = spike_weight + physics_adjustment.get('spike_adjustment', 0)
        spike_weight = max(0.3, min(0.7, spike_weight))
        uncertainty_std = uncertainty_std * physics_adjustment.get('uncertainty_multiplier', 1.0)
    
    spread_weight = 1 - spike_weight
    spread_prob = stats.norm.cdf(threshold + 0.5, forecast_temp, uncertainty_std)
    
    if forecast_temp <= threshold:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def mixture_above_probability(threshold, forecast_temp, uncertainty_std, 
                              spike_weight=0.55, physics_adjustment=None):
    """Mixture probability for 'X or above' contracts with physics."""
    if physics_adjustment:
        spike_weight = spike_weight + physics_adjustment.get('spike_adjustment', 0)
        spike_weight = max(0.3, min(0.7, spike_weight))
        uncertainty_std = uncertainty_std * physics_adjustment.get('uncertainty_multiplier', 1.0)
    
    spread_weight = 1 - spike_weight
    spread_prob = 1 - stats.norm.cdf(threshold - 0.5, forecast_temp, uncertainty_std)
    
    if forecast_temp >= threshold:
        total_prob = spike_weight + (spread_weight * spread_prob)
    else:
        total_prob = spread_weight * spread_prob
    
    return min(total_prob, 0.99)


def analyze_city(city_key):
    """Main analysis function for a single city with physics."""
    city = cities[city_key]
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {city['name'].upper()}")
    print(f"{'='*60}")
    
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
    
    # Fetch HRRR forecast WITH PHYSICS
    forecast_temp, weather_data = fetch_hrrr_with_physics(
        city["lat"], city["lon"], target_date
    )
    
    if forecast_temp is None:
        print("❌ Could not retrieve HRRR forecast. Skipping city.")
        return
    
    # Display HRRR data
    print(f"\n  HRRR Forecast Data:")
    print(f"  {'='*50}")
    print(f"  Model Run: {weather_data['model_run']} UTC")
    print(f"  Target Date: {target_str}")
    print(f"  Forecast High: {forecast_temp:.1f}°F")
    print(f"  Forecast Range: {weather_data['forecast_low']:.1f}°F - {weather_data['forecast_high']:.1f}°F")
    if weather_data['wind_speed']:
        print(f"  Avg Wind: {weather_data['wind_speed']:.1f} mph @ {weather_data['wind_dir']:.0f}°")
    
    # ============ PHYSICS ANALYSIS ============
    print(f"\n  {'='*50}")
    print("  TEMPERATURE ADVECTION PHYSICS")
    print(f"  {'='*50}")
    print(f"  Equation: ∂T/∂t = -u(∂T/∂x) - v(∂T/∂y)")
    
    advection_data = weather_data.get('advection', {})
    physics_adjustment = interpret_advection(advection_data, forecast_temp)
    
    if advection_data.get('average') is not None:
        print(f"\n  Advection Analysis:")
        print(f"    Average: {advection_data['average']:.2f}°F/hr")
        print(f"    Range: [{advection_data['min']:.2f}, {advection_data['max']:.2f}]°F/hr")
        if advection_data.get('trend'):
            print(f"    Trend: {advection_data['trend']:.3f}°F/hr² ({'cooling' if advection_data['trend'] < 0 else 'warming'})")
        print(f"\n  Interpretation: {physics_adjustment['description']}")
        print(f"  Uncertainty multiplier: {physics_adjustment['uncertainty_multiplier']:.2f}x")
        print(f"  Spike weight adjustment: {physics_adjustment['spike_adjustment']:+.2f}")
        
        # Show sample advection data
        if advection_data.get('values') and len(advection_data['values']) > 0:
            sample = advection_data['values'][0]['info']
            print(f"\n  Sample Grid Data (forecast hour {advection_data['values'][0]['hour']}):")
            print(f"    Center temp: {sample['temp_center_f']:.1f}°F")
            print(f"    N/S/E/W temps: {sample['temp_north_f']:.1f}° / {sample['temp_south_f']:.1f}° / {sample['temp_east_f']:.1f}° / {sample['temp_west_f']:.1f}°")
            print(f"    ∂T/∂x: {sample['dT_dx']:.3f}°F/km | ∂T/∂y: {sample['dT_dy']:.3f}°F/km")
            print(f"    U-wind: {sample['u_wind']:.1f} mph | V-wind: {sample['v_wind']:.1f} mph")
    else:
        print("  ⚠️ No advection data available")
    
    # Fetch recent temps
    recent_temps = fetch_hrrr_recent_temps(city["lat"], city["lon"], num_days=5)
    
    # Add recent data to dataframe
    for rt in recent_temps:
        new_row = pd.DataFrame({
            "date": [pd.to_datetime(rt['date'])], 
            "temp": [rt['temp']]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Build features and train
    df = build_features(df)
    model, model_error_std, mae, feature_cols = train_model(df)
    
    print(f"\n  ML Model trained: MAE = {mae:.2f}°F")
    
    # ============ MIXTURE MODEL PARAMETERS ============
    BASE_SPIKE_WEIGHT = 0.60
    BASE_UNCERTAINTY = 2.2
    
    # Apply physics adjustments
    adjusted_spike = BASE_SPIKE_WEIGHT + physics_adjustment.get('spike_adjustment', 0)
    adjusted_uncertainty = BASE_UNCERTAINTY * physics_adjustment.get('uncertainty_multiplier', 1.0)
    
    forecast_bin = get_bin_for_temp(forecast_temp)
    
    print(f"\n  {'='*50}")
    print("  MIXTURE MODEL (v5 - HRRR + Physics)")
    print(f"  {'='*50}")
    print(f"  HRRR Forecast: {forecast_temp:.1f}°F")
    print(f"  Forecast Bin: {forecast_bin[0]}° to {forecast_bin[1]}°")
    print(f"  Base Spike Weight: {BASE_SPIKE_WEIGHT*100:.0f}%")
    print(f"  Adjusted Spike Weight: {adjusted_spike*100:.0f}%")
    print(f"  Base Uncertainty: ±{BASE_UNCERTAINTY}°F")
    print(f"  Adjusted Uncertainty: ±{adjusted_uncertainty:.1f}°F")
    
    # Get Kalshi markets
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"series_ticker": city["kalshi_series"], "status": "open", "limit": 100}
    response = requests.get(f"{BASE_URL}/markets", params=params)
    markets = response.json().get("markets", [])
    
    target_kalshi = target_date.strftime("%y%b%d").upper()
    target_markets = [m for m in markets if target_kalshi in m["ticker"]]
    
    if not target_markets:
        print(f"\n  ⚠️ No Kalshi markets found for {target_str}")
        return
    
    # Find Kalshi's top prediction
    kalshi_top_contract = None
    kalshi_top_prob = 0
    for m in target_markets:
        if m["last_price"] > kalshi_top_prob:
            kalshi_top_prob = m["last_price"]
            kalshi_top_contract = m["subtitle"]
    
    kalshi_implied_temp = None
    if kalshi_top_contract:
        numbers = re.findall(r'\d+', kalshi_top_contract)
        if "to" in kalshi_top_contract and len(numbers) >= 2:
            kalshi_implied_temp = (int(numbers[0]) + int(numbers[1])) / 2
    
    print(f"\n  {'='*50}")
    print("  MARKET EFFICIENCY CHECK")
    print(f"  {'='*50}")
    print(f"  HRRR Forecast: {forecast_temp:.1f}°F")
    print(f"  Kalshi Top Contract: {kalshi_top_contract} ({kalshi_top_prob}%)")
    
    market_efficient = False
    if kalshi_implied_temp:
        print(f"  Kalshi Implied Temp: ~{kalshi_implied_temp}°F")
        forecast_vs_kalshi_diff = abs(forecast_temp - kalshi_implied_temp)
        print(f"  Forecast vs Kalshi Diff: {forecast_vs_kalshi_diff:.1f}°F")
        
        if forecast_vs_kalshi_diff <= 2:
            print("\n  ✅ MARKET IS EFFICIENTLY PRICED")
            market_efficient = True
        else:
            print(f"\n  🎯 POTENTIAL MISPRICING: {forecast_vs_kalshi_diff:.1f}°F difference!")
    
    # Compare with Kalshi
    print(f"\n  {'='*50}")
    print("  MODEL vs KALSHI COMPARISON")
    print(f"  {'='*50}\n")
    
    print(f"  {'Contract':<20} {'Model':>10} {'Kalshi':>10} {'Edge':>10}")
    print("  " + "-" * 55)
    
    results = []
    
    for m in sorted(target_markets, key=lambda x: x["last_price"], reverse=True):
        subtitle = m["subtitle"]
        kalshi_prob = m["last_price"] / 100
        numbers = re.findall(r'\d+', subtitle)
        
        model_prob = 0
        is_forecast_bin = False
        
        if "to" in subtitle and len(numbers) >= 2:
            low = int(numbers[0])
            high = int(numbers[1])
            model_prob = mixture_probability(low, high + 1, forecast_temp, adjusted_uncertainty, 
                                            adjusted_spike, physics_adjustment)
            is_forecast_bin = low <= forecast_temp <= high + 1
            
        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = mixture_below_probability(threshold, forecast_temp, adjusted_uncertainty,
                                                   adjusted_spike, physics_adjustment)
            is_forecast_bin = forecast_temp <= threshold
            
        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            model_prob = mixture_above_probability(threshold, forecast_temp, adjusted_uncertainty,
                                                   adjusted_spike, physics_adjustment)
            is_forecast_bin = forecast_temp >= threshold
        else:
            continue
        
        edge = model_prob - kalshi_prob
        
        results.append({
            "subtitle": subtitle,
            "model_prob": model_prob,
            "kalshi_prob": kalshi_prob,
            "edge": edge,
            "is_forecast_bin": is_forecast_bin
        })
        
        marker = "📍" if is_forecast_bin else "  "
        print(f"  {marker}{subtitle:<18} {model_prob:>9.1%} {kalshi_prob:>9.0%} {edge:>+9.1%}")
    
    # Betting recommendations
    print(f"\n  {'='*50}")
    print("  BETTING RECOMMENDATIONS")
    print(f"  {'='*50}\n")
    
    bankroll = 20
    
    # Adjust model confidence based on physics
    base_confidence = 0.85
    if physics_adjustment['cold_front'] or physics_adjustment['warm_front']:
        model_confidence = 0.75  # Less confident during frontal passages
    else:
        model_confidence = base_confidence
    
    good_bets = []
    
    for r in sorted(results, key=lambda x: abs(x["edge"]), reverse=True):
        edge = r["edge"]
        
        if edge < 0 and r["is_forecast_bin"]:
            continue
        
        if abs(edge) < 0.08:
            continue
        
        if edge > 0:
            bet_side = "YES"
            our_prob = r["model_prob"] * model_confidence + r["kalshi_prob"] * (1 - model_confidence)
            market_price = r["kalshi_prob"]
        else:
            bet_side = "NO"
            our_prob = (1 - r["model_prob"]) * model_confidence + (1 - r["kalshi_prob"]) * (1 - model_confidence)
            market_price = 1 - r["kalshi_prob"]
        
        if market_price > 0:
            odds = (1 / market_price) - 1
            p = our_prob
            q = 1 - our_prob
            b = odds
            
            kelly_fraction = (p * b - q) / b if b > 0 else 0
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
                    "is_forecast_bin": r["is_forecast_bin"]
                })
    
    if len(good_bets) == 0:
        if market_efficient:
            print("  📊 NO CLEAR EDGE TODAY")
            print(f"     Market matches forecast (both ~{forecast_temp:.0f}°F)")
        else:
            print("  ⚠️ NO RECOMMENDED BETS")
            print("     Edges too small or contradict forecast")
    else:
        for bet in good_bets:
            forecast_marker = " 📍 (FORECAST BIN)" if bet["is_forecast_bin"] else ""
            print(f"  {bet['subtitle']}:{forecast_marker}")
            print(f"    Bet: {bet['bet_side']} at {bet['market_price']:.0%}")
            print(f"    Model prob: {bet['model_prob']:.1%}")
            print(f"    Edge: {bet['edge']:+.1%}")
            print(f"    Half Kelly: {bet['half_kelly']:.1%}")
            print(f"    Suggested bet: ${bet['bet_amount']:.2f}")
            print(f"    Potential profit: ${bet['potential_profit']:.2f}")
            print()
    
    # Summary
    print(f"  {'='*50}")
    print(f"  SUMMARY - {city['name'].upper()}")
    print(f"  {'='*50}")
    print(f"  Target Date: {target_str}")
    print(f"  HRRR Forecast: {forecast_temp:.1f}°F → Bin: {forecast_bin[0]}-{forecast_bin[1]}°")
    print(f"  Physics: {physics_adjustment['description']}")
    print(f"  Kalshi Top Pick: {kalshi_top_contract} ({kalshi_top_prob}%)")
    print(f"  Market Status: {'✅ Efficient' if market_efficient else '🎯 Potential Edge'}")
    print(f"  Recommended Bets: {len(good_bets)}")
    
    all_bets.extend(good_bets)


# ============ RUN ANALYSIS ============
if __name__ == "__main__":
    if not HERBIE_AVAILABLE:
        print("\n❌ Cannot run without Herbie. Please install:")
        print("   pip install herbie-data xarray cfgrib eccodes")
        exit(1)
    
    for city_key in selected_cities:
        analyze_city(city_key)
    
    # ============ BETTING SUMMARY ============
    target_date = (datetime.now().date() + timedelta(days=1)).strftime("%A, %B %d, %Y")
    print(f"\n{'='*60}")
    print(f"💰 BETTING SUMMARY - ALL CITIES")
    print(f"📅 Target Date: {target_date}")
    print(f"📡 Data Source: NOAA HRRR (3km) + Temperature Advection")
    print(f"{'='*60}\n")

    if len(all_bets) == 0:
        print("No recommended bets today. Consider skipping.")
    else:
        all_bets_sorted = sorted(all_bets, key=lambda x: abs(x["edge"]), reverse=True)
        
        total_suggested = 0
        total_potential_profit = 0
        
        print(f"{'City':<12} {'Contract':<15} {'Bet':<6} {'Price':<8} {'Edge':<8} {'Wager':<10} {'Profit':<10}")
        print("-" * 80)
        
        for bet in all_bets_sorted:
            forecast_marker = "📍" if bet.get("is_forecast_bin") else "  "
            print(f"{bet['city']:<12} {forecast_marker}{bet['subtitle']:<13} {bet['bet_side']:<6} {bet['market_price']*100:>5.0f}¢    {bet['edge']:>+5.1%}   ${bet['bet_amount']:>6.2f}    ${bet['potential_profit']:>7.2f}")
            total_suggested += bet["bet_amount"]
            total_potential_profit += bet["potential_profit"]
        
        print("-" * 80)
        print(f"{'TOTAL':<47} ${total_suggested:>6.2f}    ${total_potential_profit:>7.2f}")
        
        print(f"\n📍 = Forecast bin (highest confidence)")
        
        positive_edge_bets = [bet for bet in all_bets_sorted if bet["edge"] > 0]
        
        if len(positive_edge_bets) > 0:
            print(f"\n{'='*60}")
            print("⭐ TOP PICKS (Positive Edge)")
            print(f"{'='*60}")
            for bet in positive_edge_bets:
                forecast_marker = " 📍" if bet.get("is_forecast_bin") else ""
                print(f"\n   {bet['city']}: {bet['subtitle']} {bet['bet_side']}{forecast_marker}")
                print(f"   Price: {bet['market_price']*100:.0f}¢ | Edge: {bet['edge']:+.1%}")
                print(f"   Suggested: ${bet['bet_amount']:.2f} → Potential: ${bet['potential_profit']:.2f}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")