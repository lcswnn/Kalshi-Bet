"""
MEDIUM PRIORITY ENHANCEMENTS MODULE
====================================
Adds physical meteorology features to improve forecast accuracy:

1. CLOUD TIMING ANALYSIS
   - Morning clouds (6-10 AM) = minimal impact
   - Midday clouds (10 AM-2 PM) = -5 to -10°F suppression
   - Afternoon clouds (2-6 PM) = -2 to -5°F impact
   
2. PRECIPITATION TIMING
   - Rain during peak heating (11 AM-3 PM) = -8 to -15°F
   - Morning/evening rain = minimal impact on max temp
   - Heavy vs light precipitation scaling
   
3. WEATHER REGIME DETECTION
   - High pressure (stable) = 0.75x uncertainty multiplier
   - Frontal systems (unstable) = 1.5x uncertainty multiplier
   - Transitional patterns = 1.0x baseline

Integrates with Open-Meteo hourly forecast data (free API).
"""

import numpy as np
import requests
from datetime import datetime, timedelta

# ============ CLOUD TIMING ANALYSIS ============

def fetch_hourly_cloud_data(lat, lon, target_date):
    """
    Fetch hourly cloud cover forecast for target date.
    
    Returns:
        dict with hourly cloud cover % (0-100) for key hours
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "cloud_cover",
            "timezone": "auto",
            "forecast_days": 3
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse to find target date hours
        times = data["hourly"]["time"]
        cloud_cover = data["hourly"]["cloud_cover"]
        
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        hourly_clouds = {}
        for i, time_str in enumerate(times):
            if time_str.startswith(target_date_str):
                hour = datetime.fromisoformat(time_str.replace('Z', '+00:00')).hour
                hourly_clouds[hour] = cloud_cover[i]
        
        return hourly_clouds
        
    except Exception as e:
        print(f"     ⚠️ Could not fetch cloud data: {e}")
        return None


def analyze_cloud_timing(hourly_clouds):
    """
    Analyze cloud cover timing to predict temperature suppression.
    
    Key periods:
    - Peak heating: 11 AM - 3 PM (most critical)
    - Midday: 10 AM - 2 PM
    - Morning: 6 AM - 10 AM
    - Afternoon: 2 PM - 6 PM
    
    Returns:
        adjustment (float): Temperature adjustment in °F (negative = cooler)
        analysis (dict): Breakdown of cloud impacts
    """
    if not hourly_clouds:
        return 0.0, None
    
    analysis = {
        "morning_clouds": 0,
        "midday_clouds": 0,
        "peak_heating_clouds": 0,
        "afternoon_clouds": 0,
        "adjustment": 0
    }
    
    # Define critical periods (use local hour approximations)
    morning_hours = [6, 7, 8, 9]
    midday_hours = [10, 11, 12, 13]
    peak_heating_hours = [11, 12, 13, 14]  # 11 AM - 2 PM local
    afternoon_hours = [14, 15, 16, 17]
    
    # Calculate average cloud cover for each period
    def avg_clouds(hours):
        clouds = [hourly_clouds.get(h, 0) for h in hours if h in hourly_clouds]
        return np.mean(clouds) if clouds else 0
    
    analysis["morning_clouds"] = avg_clouds(morning_hours)
    analysis["midday_clouds"] = avg_clouds(midday_hours)
    analysis["peak_heating_clouds"] = avg_clouds(peak_heating_hours)
    analysis["afternoon_clouds"] = avg_clouds(afternoon_hours)
    
    # Temperature adjustment based on cloud timing
    adjustment = 0.0
    
    # Peak heating clouds (11 AM-2 PM) - MOST CRITICAL
    peak_clouds = analysis["peak_heating_clouds"]
    if peak_clouds > 75:
        # Heavy clouds during peak heating
        adjustment -= 8.0  # -8°F suppression
        analysis["impact"] = "Heavy clouds during peak heating"
    elif peak_clouds > 50:
        # Moderate clouds during peak heating
        adjustment -= 5.0  # -5°F suppression
        analysis["impact"] = "Moderate clouds during peak heating"
    elif peak_clouds > 25:
        # Light clouds during peak heating
        adjustment -= 2.5  # -2.5°F suppression
        analysis["impact"] = "Light clouds during peak heating"
    
    # Morning clouds (6-10 AM) - Minimal impact
    # (Max temp usually occurs later, so morning clouds matter less)
    
    # Afternoon clouds (2-6 PM) - Moderate impact if max occurs late
    afternoon_clouds = analysis["afternoon_clouds"]
    if afternoon_clouds > 75 and peak_clouds < 50:
        # Afternoon clouds but clear midday
        adjustment -= 1.5  # Small suppression
        if not analysis.get("impact"):
            analysis["impact"] = "Afternoon clouds (minor impact)"
    
    analysis["adjustment"] = adjustment
    
    return adjustment, analysis


# ============ PRECIPITATION TIMING ANALYSIS ============

def fetch_hourly_precipitation(lat, lon, target_date):
    """
    Fetch hourly precipitation forecast for target date.
    
    Returns:
        dict with hourly precipitation (mm) for key hours
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,precipitation_probability",
            "timezone": "auto",
            "forecast_days": 3
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        times = data["hourly"]["time"]
        precip = data["hourly"]["precipitation"]
        precip_prob = data["hourly"]["precipitation_probability"]
        
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        hourly_precip = {}
        for i, time_str in enumerate(times):
            if time_str.startswith(target_date_str):
                hour = datetime.fromisoformat(time_str.replace('Z', '+00:00')).hour
                hourly_precip[hour] = {
                    "amount": precip[i],
                    "probability": precip_prob[i]
                }
        
        return hourly_precip
        
    except Exception as e:
        print(f"     ⚠️ Could not fetch precipitation data: {e}")
        return None


def analyze_precipitation_timing(hourly_precip):
    """
    Analyze precipitation timing to predict temperature suppression.
    
    Rain during peak heating hours dramatically suppresses max temp because:
    - Evaporative cooling
    - Cloud cover (already counted separately)
    - Reduced solar radiation
    - Increased humidity
    
    Returns:
        adjustment (float): Temperature adjustment in °F (negative = cooler)
        analysis (dict): Breakdown of precipitation impacts
    """
    if not hourly_precip:
        return 0.0, None
    
    analysis = {
        "morning_precip": 0,
        "peak_heating_precip": 0,
        "afternoon_precip": 0,
        "total_precip": 0,
        "adjustment": 0
    }
    
    # Define critical periods
    morning_hours = [6, 7, 8, 9, 10]
    peak_heating_hours = [11, 12, 13, 14, 15]  # 11 AM - 3 PM
    afternoon_hours = [15, 16, 17, 18]
    
    # Calculate total precipitation and probability for each period
    def total_precip_period(hours):
        total = 0
        max_prob = 0
        for h in hours:
            if h in hourly_precip:
                total += hourly_precip[h]["amount"]
                max_prob = max(max_prob, hourly_precip[h]["probability"])
        return total, max_prob
    
    morning_amt, morning_prob = total_precip_period(morning_hours)
    peak_amt, peak_prob = total_precip_period(peak_heating_hours)
    afternoon_amt, afternoon_prob = total_precip_period(afternoon_hours)
    
    analysis["morning_precip"] = morning_amt
    analysis["peak_heating_precip"] = peak_amt
    analysis["afternoon_precip"] = afternoon_amt
    analysis["total_precip"] = morning_amt + peak_amt + afternoon_amt
    
    # Temperature adjustment based on precipitation timing
    adjustment = 0.0
    
    # Peak heating precipitation - MOST CRITICAL
    if peak_amt > 5.0 and peak_prob > 70:
        # Heavy rain during peak heating (>5mm, high confidence)
        adjustment -= 12.0  # -12°F suppression
        analysis["impact"] = "Heavy rain during peak heating hours"
    elif peak_amt > 2.5 and peak_prob > 60:
        # Moderate rain during peak heating
        adjustment -= 8.0  # -8°F suppression
        analysis["impact"] = "Moderate rain during peak heating hours"
    elif peak_amt > 0.5 and peak_prob > 50:
        # Light rain during peak heating
        adjustment -= 4.0  # -4°F suppression
        analysis["impact"] = "Light rain during peak heating hours"
    elif peak_prob > 60 and peak_amt > 0:
        # High probability of any rain during peak
        adjustment -= 3.0  # -3°F suppression
        analysis["impact"] = "Likely rain during peak heating hours"
    
    # Morning rain - Minimal direct impact on max temp
    # (But increases humidity which can suppress slightly)
    if morning_amt > 5.0 and peak_amt < 0.5:
        adjustment -= 1.0  # Small suppression from residual moisture
        if not analysis.get("impact"):
            analysis["impact"] = "Morning rain (residual moisture effect)"
    
    # Afternoon rain - Moderate impact if max occurs late
    if afternoon_amt > 3.0 and afternoon_prob > 70 and peak_amt < 0.5:
        adjustment -= 2.0  # Moderate suppression
        if not analysis.get("impact"):
            analysis["impact"] = "Afternoon rain (can suppress late-day max)"
    
    analysis["adjustment"] = adjustment
    
    return adjustment, analysis


# ============ WEATHER REGIME DETECTION ============

def fetch_pressure_and_wind_data(lat, lon, target_date):
    """
    Fetch pressure and wind data to classify weather regime.
    
    Returns:
        dict with pressure, wind, and derived metrics
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "surface_pressure,wind_speed_10m,wind_direction_10m",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "past_days": 2,
            "forecast_days": 3
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Get daily data
        target_date_str = target_date.strftime("%Y-%m-%d")
        target_idx = None
        for i, date_str in enumerate(data["daily"]["time"]):
            if date_str == target_date_str:
                target_idx = i
                break
        
        if target_idx is None:
            return None
        
        # Get hourly data for target date
        times = data["hourly"]["time"]
        pressures = data["hourly"]["surface_pressure"]
        wind_speeds = data["hourly"]["wind_speed_10m"]
        wind_dirs = data["hourly"]["wind_direction_10m"]
        
        target_pressure = []
        target_wind = []
        target_wind_dir = []
        
        for i, time_str in enumerate(times):
            if time_str.startswith(target_date_str):
                target_pressure.append(pressures[i])
                target_wind.append(wind_speeds[i])
                target_wind_dir.append(wind_dirs[i])
        
        # Calculate pressure tendency (change over 24 hours)
        if target_idx > 0:
            prev_day_pressures = []
            prev_date_str = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
            for i, time_str in enumerate(times):
                if time_str.startswith(prev_date_str):
                    prev_day_pressures.append(pressures[i])
            
            if target_pressure and prev_day_pressures:
                pressure_change = np.mean(target_pressure) - np.mean(prev_day_pressures)
            else:
                pressure_change = 0
        else:
            pressure_change = 0
        
        # Get temperature range (diurnal range indicator)
        temp_max = data["daily"]["temperature_2m_max"][target_idx]
        temp_min = data["daily"]["temperature_2m_min"][target_idx]
        diurnal_range = temp_max - temp_min
        
        return {
            "avg_pressure": np.mean(target_pressure) if target_pressure else None,
            "pressure_change_24hr": pressure_change,
            "avg_wind_speed": np.mean(target_wind) if target_wind else None,
            "max_wind_speed": max(target_wind) if target_wind else None,
            "wind_direction_variability": np.std(target_wind_dir) if target_wind_dir else None,
            "diurnal_range": diurnal_range
        }
        
    except Exception as e:
        print(f"     ⚠️ Could not fetch pressure/wind data: {e}")
        return None


def classify_weather_regime(pressure_data, cloud_analysis, precip_analysis):
    """
    Classify weather regime to adjust forecast uncertainty.
    
    Regimes:
    1. HIGH PRESSURE (Stable)
       - High pressure (>1015 mb)
       - Rising or steady pressure
       - Low winds (<10 mph)
       - Minimal clouds/precip
       → 0.75x uncertainty multiplier (more predictable)
    
    2. FRONTAL SYSTEM (Unstable)
       - Rapidly changing pressure (>3 mb/24hr change)
       - High winds (>15 mph)
       - Significant clouds/precip
       - High wind direction variability
       → 1.5x uncertainty multiplier (less predictable)
    
    3. TRANSITIONAL (Baseline)
       - Everything in between
       → 1.0x uncertainty multiplier
    
    Returns:
        regime (str): "high_pressure", "frontal", or "transitional"
        uncertainty_multiplier (float): Adjustment factor for uncertainty
        confidence (dict): Detailed regime indicators
    """
    if not pressure_data:
        return "transitional", 1.0, {"error": "No pressure data"}
    
    # Extract metrics
    pressure = pressure_data.get("avg_pressure", 1013)
    pressure_change = pressure_data.get("pressure_change_24hr", 0)
    wind_speed = pressure_data.get("avg_wind_speed", 0)
    max_wind = pressure_data.get("max_wind_speed", 0)
    wind_variability = pressure_data.get("wind_direction_variability", 0)
    diurnal_range = pressure_data.get("diurnal_range", 0)
    
    # Get cloud/precip impacts
    has_significant_clouds = False
    has_significant_precip = False
    
    if cloud_analysis:
        peak_clouds = cloud_analysis.get("peak_heating_clouds", 0)
        has_significant_clouds = peak_clouds > 50
    
    if precip_analysis:
        peak_precip = precip_analysis.get("peak_heating_precip", 0)
        has_significant_precip = peak_precip > 1.0
    
    # Score indicators
    high_pressure_score = 0
    frontal_score = 0
    
    # Pressure indicators
    if pressure > 1020:
        high_pressure_score += 2
    elif pressure > 1015:
        high_pressure_score += 1
    elif pressure < 1010:
        frontal_score += 1
    
    # Pressure tendency
    if abs(pressure_change) < 1.0:
        high_pressure_score += 2  # Steady pressure = stable
    elif abs(pressure_change) > 3.0:
        frontal_score += 2  # Rapidly changing = frontal
    elif abs(pressure_change) > 2.0:
        frontal_score += 1
    
    # Wind indicators
    if wind_speed < 5:
        high_pressure_score += 1
    elif wind_speed > 15:
        frontal_score += 2
    elif wind_speed > 10:
        frontal_score += 1
    
    if max_wind < 10:
        high_pressure_score += 1
    elif max_wind > 20:
        frontal_score += 2
    
    # Wind direction variability
    if wind_variability < 30:
        high_pressure_score += 1  # Steady winds
    elif wind_variability > 60:
        frontal_score += 2  # Variable winds = frontal
    
    # Cloud/precip indicators
    if not has_significant_clouds and not has_significant_precip:
        high_pressure_score += 2
    if has_significant_clouds or has_significant_precip:
        frontal_score += 1
    
    # Diurnal range (large range = clear skies/stable)
    if diurnal_range > 25:
        high_pressure_score += 1
    elif diurnal_range < 15:
        frontal_score += 1
    
    # Classify regime
    confidence = {
        "high_pressure_score": high_pressure_score,
        "frontal_score": frontal_score,
        "pressure": pressure,
        "pressure_change_24hr": pressure_change,
        "avg_wind": wind_speed,
        "max_wind": max_wind
    }
    
    if high_pressure_score >= 6 and high_pressure_score > frontal_score + 2:
        regime = "high_pressure"
        uncertainty_multiplier = 0.75  # More predictable
        confidence["description"] = "Stable high pressure system"
    elif frontal_score >= 5 and frontal_score > high_pressure_score + 1:
        regime = "frontal"
        uncertainty_multiplier = 1.5  # Less predictable
        confidence["description"] = "Active frontal system or transition"
    else:
        regime = "transitional"
        uncertainty_multiplier = 1.0  # Baseline
        confidence["description"] = "Transitional weather pattern"
    
    return regime, uncertainty_multiplier, confidence


# ============ INTEGRATED ANALYSIS ============

def apply_medium_priority_enhancements(lat, lon, target_date, base_forecast, base_uncertainty):
    """
    Apply all medium priority enhancements to improve forecast.
    
    Args:
        lat, lon: Location coordinates
        target_date: Date to analyze
        base_forecast: Initial ensemble forecast (°F)
        base_uncertainty: Initial uncertainty (°F)
    
    Returns:
        adjusted_forecast: Forecast with cloud/precip adjustments
        adjusted_uncertainty: Uncertainty with regime adjustment
        enhancements_info: Detailed breakdown of all adjustments
    """
    print(f"\n  🌤️  Applying Medium Priority Enhancements...")
    
    enhancements_info = {
        "cloud_adjustment": 0,
        "precip_adjustment": 0,
        "regime": "transitional",
        "uncertainty_multiplier": 1.0,
        "total_forecast_adjustment": 0
    }
    
    # 1. CLOUD TIMING ANALYSIS
    hourly_clouds = fetch_hourly_cloud_data(lat, lon, target_date)
    cloud_adjustment, cloud_analysis = analyze_cloud_timing(hourly_clouds)
    
    if cloud_analysis:
        print(f"     ☁️  Cloud Analysis:")
        print(f"         Peak heating clouds: {cloud_analysis['peak_heating_clouds']:.0f}%")
        if cloud_analysis.get("impact"):
            print(f"         Impact: {cloud_analysis['impact']}")
        if abs(cloud_adjustment) > 0.5:
            print(f"         Temperature adjustment: {cloud_adjustment:+.1f}°F")
        enhancements_info["cloud_adjustment"] = cloud_adjustment
        enhancements_info["cloud_analysis"] = cloud_analysis
    
    # 2. PRECIPITATION TIMING ANALYSIS
    hourly_precip = fetch_hourly_precipitation(lat, lon, target_date)
    precip_adjustment, precip_analysis = analyze_precipitation_timing(hourly_precip)
    
    if precip_analysis:
        print(f"     🌧️  Precipitation Analysis:")
        print(f"         Peak heating precip: {precip_analysis['peak_heating_precip']:.1f}mm")
        if precip_analysis.get("impact"):
            print(f"         Impact: {precip_analysis['impact']}")
        if abs(precip_adjustment) > 0.5:
            print(f"         Temperature adjustment: {precip_adjustment:+.1f}°F")
        enhancements_info["precip_adjustment"] = precip_adjustment
        enhancements_info["precip_analysis"] = precip_analysis
    
    # 3. WEATHER REGIME DETECTION
    pressure_data = fetch_pressure_and_wind_data(lat, lon, target_date)
    regime, uncertainty_mult, regime_confidence = classify_weather_regime(
        pressure_data, cloud_analysis, precip_analysis
    )
    
    print(f"     🌡️  Weather Regime: {regime.upper().replace('_', ' ')}")
    print(f"         {regime_confidence.get('description', 'N/A')}")
    print(f"         Uncertainty multiplier: {uncertainty_mult:.2f}x")
    if pressure_data:
        print(f"         Pressure: {regime_confidence.get('pressure', 0):.1f} mb "
              f"({regime_confidence.get('pressure_change_24hr', 0):+.1f} mb/24hr)")
        print(f"         Wind: {regime_confidence.get('avg_wind', 0):.1f} mph avg, "
              f"{regime_confidence.get('max_wind', 0):.1f} mph max")
    
    enhancements_info["regime"] = regime
    enhancements_info["uncertainty_multiplier"] = uncertainty_mult
    enhancements_info["regime_confidence"] = regime_confidence
    
    # APPLY ADJUSTMENTS
    total_adjustment = cloud_adjustment + precip_adjustment
    adjusted_forecast = base_forecast + total_adjustment
    adjusted_uncertainty = base_uncertainty * uncertainty_mult
    
    enhancements_info["total_forecast_adjustment"] = total_adjustment
    
    if abs(total_adjustment) > 0.5:
        print(f"\n     📊 TOTAL ENHANCEMENT ADJUSTMENT:")
        print(f"         Forecast: {base_forecast:.1f}°F → {adjusted_forecast:.1f}°F ({total_adjustment:+.1f}°F)")
        print(f"         Uncertainty: {base_uncertainty:.1f}°F → {adjusted_uncertainty:.1f}°F "
              f"({uncertainty_mult:.2f}x regime adjustment)")
    
    return adjusted_forecast, adjusted_uncertainty, enhancements_info


# ============ MODULE TEST ============

if __name__ == "__main__":
    # Test the module
    from datetime import date
    
    print("Testing Medium Priority Enhancements Module")
    print("=" * 60)
    
    # Test with Chicago
    lat, lon = 41.8781, -87.6298
    target = date.today() + timedelta(days=1)
    base_forecast = 45.0
    base_uncertainty = 3.0
    
    adjusted_forecast, adjusted_uncertainty, info = apply_medium_priority_enhancements(
        lat, lon, target, base_forecast, base_uncertainty
    )
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Adjusted Forecast: {adjusted_forecast:.1f}°F")
    print(f"  Adjusted Uncertainty: {adjusted_uncertainty:.1f}°F")
    print(f"  Weather Regime: {info['regime']}")
    print("=" * 60)