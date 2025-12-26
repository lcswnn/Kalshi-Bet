"""
KALSHI WEATHER BETTING MODEL v12.6 (PROBABILITY CAP FIXES)
=====================================================================================
CRITICAL FIXES IN V12.6:
- 🚨 FIXED: 100% confidence bug causing 28% win rate
- ✅ Hard caps on BOTH YES and NO bets (max 95%)
- ✅ Uncertainty-based caps (±8°F = max 85% confidence)
- ✅ Major disagreement caps (disagreement = max 85% confidence)
- ✅ Agreement level caps (low = max 88%, medium = max 92%)

CALIBRATED based on your 46 historical bets showing:
- 30% estimates → 70% actual wins (you're too conservative!)
- 40% estimates → 71% actual wins (systematic underestimation)
- 40-60¢ price range: Your sweet spot (71% win rate)
- Overall: 50% win rate but leaving money on the table

OPTIMIZATIONS APPLIED (v12):
1. ✅ LOWER EDGE REQUIREMENTS: 8% sweet spot, 7% mid-range (was 12%)
2. ❌ CALIBRATION BOOST REMOVED: No longer auto-adjusts probabilities (was +15%)
3. ✅ AGGRESSIVE KELLY: 0.80 fraction, 20% max (was 0.75, 15%)
4. ✅ MID-RANGE FOCUS: 40-60¢ contracts prioritized (your strength)
5. ✅ RELAXED THRESHOLDS: Lower barriers for multi-betting same city

NEW IN V12.5 - MEDIUM PRIORITY ENHANCEMENTS:
6. ✅ CLOUD TIMING ANALYSIS: Midday clouds suppress max by 5-10°F
7. ✅ PRECIPITATION TIMING: Rain during peak heating = 8-15°F drop
8. ✅ REGIME DETECTION: High pressure vs frontal systems affects uncertainty

HIGH PRIORITY FIXES (v12):
1. ✅ DYNAMIC FORECAST UNCERTAINTY: Adjusts std based on model spread & conditions (2-6°F)
2. ✅ CURRENT OBSERVATIONS: Uses real-time data to constrain forecasts
3. ✅ CALIBRATION TRACKING: Tracks historical accuracy to adjust probabilities  
4. ✅ SMART MODEL WEIGHTING: Reduces correlation between GFS-based forecasts

Key improvements over v11:
- Uncertainty scales with model disagreement (not fixed 2.8°F)
- Morning observations prevent impossible predictions
- Calibration database learns from your 46 bets
- NWS/Open-Meteo get reduced weight (both use GFS) vs independent HRRR
- Cloud/precip timing analysis prevents major forecast misses
- Weather regime detection adjusts uncertainty appropriately

All v11 features maintained:
- NWS FORECAST integration
- NWS rounding rules (matches Kalshi resolution)
- Valid bins only (prevents betting on non-existent bins)
- Smart bet selection (different cities vs same city)
- Kelly Criterion sizing with bankroll tracking
- Price filtering and edge requirements
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.model_selection import cross_val_score # type: ignore
import requests # type: ignore
from datetime import datetime, timedelta, timezone
import re
from scipy import stats # type: ignore
import glob
import argparse
from pathlib import Path

# Import medium-priority enhancements (v12.5)
try:
    from medium_priority_enhancements import apply_medium_priority_enhancements as enhanced_forecast_adjustment
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

# ============ HRRR/HERBIE IMPORTS ============
try:
    from herbie import Herbie # type: ignore
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False
    print("⚠️ Herbie not installed. Will use Open-Meteo only.")
    print("   Install with: pip install herbie-data xarray cfgrib")

# ============ PRICE FILTER CONFIGURATION ============
# CALIBRATED based on historical performance
MIN_CONTRACT_PRICE = 0.10   # Slightly lower (was 0.15) - some low-price bets worked
MAX_CONTRACT_PRICE = 0.90   # Keep at 90¢
SWEET_SPOT_LOW = 0.15       # Standard lower bound
SWEET_SPOT_HIGH = 0.60      # Extended to 60¢ (was 0.50) - your 40-60¢ bets win 71%!

# ============ SMART BET SELECTION CONFIGURATION ============
# CALIBRATED based on historical performance
SAME_CITY_MULTI_BET_THRESHOLD = 1.5   # Reduced from 2.0 - you're passing on good bets
MAX_BETS_PER_CITY = 2                  # Keep at 2
MAX_TOTAL_BETS_PER_DAY = 6             # Keep at 6
SUPER_CONFIDENT_EDGE_RATIO = 2.0       # Reduced from 2.5 - historical data shows 1.5-2x edge is actually great

# ============ TIMEZONE CONFIGURATION ============
KALSHI_TIMEZONE = ZoneInfo("America/New_York")
EVENING_CUTOFF_HOUR = 18  # 6 PM Eastern

# ============ KELLY CRITERION CONFIGURATION ============
# CALIBRATED based on historical betting patterns
STARTING_BANKROLL = 100        # Your bankroll
KELLY_FRACTION = 0.40          # Increased from 0.75 - your picks are better than you think!
MIN_BET_SIZE = 0.50            # Don't bet less than 50¢
MAX_BET_FRACTION = 0.20        # Increased from 0.15 - bet more on calibrated opportunities

# ============ ENSEMBLE CONFIGURATION ============
ENSEMBLE_AGREEMENT_THRESHOLD = 3.0  # °F - models should agree within this
CONFIDENCE_BOOST_THRESHOLD = 2.0    # °F - boost confidence if within this
# REMOVED: CALIBRATED_FORECAST_STD - now calculated dynamically!
MAJOR_DISAGREEMENT_THRESHOLD = 5.0  # °F - flag as major disagreement (frontal systems)

# ============ ATMOSPHERIC ANALOG MODEL CONFIGURATION ============
ATMOSPHERIC_MODEL_WEIGHT = 0.15     # How much weight to give atmospheric analog prediction
MIN_TRAINING_SAMPLES = 30           # Minimum samples needed to train atmospheric model
ATMOSPHERIC_MODEL_PATH = "atmospheric_models"  # Directory to save/load models

# ============ AIRMASS & WIND ANALYSIS CONFIGURATION ============
WARM_ADVECTION_THRESHOLD = 15.0     # °F - significant warm airmass change
COLD_ADVECTION_THRESHOLD = -15.0    # °F - significant cold airmass change
WIND_SHIFT_THRESHOLD = 90.0         # degrees - significant wind direction change

# ============ CLOUD COVER & PRECIPITATION TIMING CONFIGURATION ============
MORNING_CLOUD_THRESHOLD = 60        # % - significant morning cloud cover (before 10 AM)
AFTERNOON_CLOUD_THRESHOLD = 70      # % - significant afternoon cloud cover
MORNING_CLOUD_IMPACT = -3.0         # °F - typical max temp suppression from morning clouds
HEAVY_PRECIP_THRESHOLD = 0.1        # inches/hour - heavy rain rate
PEAK_HEATING_START = 11             # Hour (local) - start of peak heating period
PEAK_HEATING_END = 15               # Hour (local) - end of peak heating period
PRECIP_PEAK_HEATING_IMPACT = -8.0   # °F - rain during peak heating suppresses max significantly

# ============ REGIME DETECTION CONFIGURATION ============
HIGH_PRESSURE_THRESHOLD = 1020      # mb - strong high pressure
LOW_PRESSURE_THRESHOLD = 1008       # mb - approaching low pressure
FRONTAL_PRESSURE_CHANGE = 3.0       # mb/3hr - rapid pressure change indicates front
STABLE_REGIME_UNCERTAINTY = 1.8     # °F - low uncertainty in stable high pressure
FRONTAL_REGIME_UNCERTAINTY = 4.5    # °F - high uncertainty near fronts

# Edge requirements by price bucket
# TUNED based on historical performance: 30-40% implied prob bets won 70%+
BASE_EDGE_REQUIREMENTS = {
    "sweet_spot": 0.08,    # 8% edge for 15-50¢ contracts (historical strength: 40-50¢ range)
    "mid_range": 0.07,     # 7% edge for 40-60¢ contracts (your best performing range!)
    "high_price": 0.10,    # 10% edge for 60-90¢ contracts (be selective at extremes)
}

def get_min_edge(price, agreement_level='high'):
    """
    Dynamic edge requirement based on price and model agreement.
    
    CALIBRATED based on historical data:
    - 40-60¢ range: Your best performing bets (71% win rate)
    - Lower requirements here to capture more +EV opportunities
    """
    # Determine price bucket
    if 0.40 <= price <= 0.60:
        # Your sweet spot based on historical data
        base_edge = BASE_EDGE_REQUIREMENTS.get("mid_range", 0.07)
    elif price <= SWEET_SPOT_HIGH:
        base_edge = BASE_EDGE_REQUIREMENTS.get("sweet_spot", 0.08)
    else:
        base_edge = BASE_EDGE_REQUIREMENTS.get("high_price", 0.10)
    
    # Adjust for agreement level
    # Historical data shows you're too conservative, so reduce multipliers
    if agreement_level == 'high':
        return base_edge * 0.9  # Reduced from 1.0
    elif agreement_level == 'medium':
        return base_edge * 1.0  # Reduced from 1.25
    else:  # low
        return base_edge * 1.2  # Reduced from 1.5

def get_price_bucket(price):
    """
    Categorize price into buckets.
    
    CALIBRATED buckets based on historical performance:
    - mid_range (40-60¢): Best win rate (71%)
    - sweet_spot (15-50¢): Decent performance
    - high_price (60-90¢): Good for NO bets
    """
    if 0.40 <= price <= 0.60:
        return "mid_range"
    elif price <= SWEET_SPOT_HIGH:
        return "sweet_spot"
    else:
        return "high_price"


# ============ NEW: CALIBRATION TRACKING SYSTEM (V12!) ============

class CalibrationTracker:
    """
    Tracks historical prediction accuracy to calibrate probabilities.
    Answers: "When I said 70% probability, was I right 70% of the time?"
    """
    def __init__(self, db_path="calibration_history.json"):
        self.db_path = db_path
        self.history = self.load_history()
    
    def load_history(self):
        """Load historical predictions and outcomes."""
        try:
            if Path(self.db_path).exists():
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            return {"predictions": [], "bins": {}}
        except:
            return {"predictions": [], "bins": {}}
    
    def save_history(self):
        """Save history to disk."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save calibration history: {e}")
    
    def record_prediction(self, date, city, our_prob, market_price, contract_details):
        """Record a prediction (to be verified later with actual outcome)."""
        prediction = {
            "date": str(date),
            "city": city,
            "our_prob": float(our_prob),
            "market_price": float(market_price),
            "contract": contract_details,
            "outcome": None,  # To be filled in later
            "recorded_at": datetime.now().isoformat()
        }
        self.history["predictions"].append(prediction)
        self.save_history()
    
    def get_calibration_adjustment(self, raw_probability):
        """
        Adjust probability based on historical calibration.
        Returns adjusted probability that better reflects true accuracy.
        
        ENHANCED for your betting history:
        - You systematically underestimate (30% → 70% actual)
        - Apply larger adjustments to capture this pattern
        """
        # Bin the probability (e.g., 0.65-0.75 → "0.70")
        prob_bin = round(raw_probability, 1)
        
        if "bins" not in self.history:
            self.history["bins"] = {}
        
        bin_key = f"{prob_bin:.1f}"
        
        if bin_key not in self.history["bins"]:
            # No calibration data - return raw probability without adjustment
            return raw_probability
        
        bin_data = self.history["bins"][bin_key]
        
        # Use calibration data even with fewer samples (was 10, now 3)
        # Your systematic bias is clear enough to trust smaller samples
        if bin_data.get("count", 0) < 3:
            # Not enough data - return raw probability
            return raw_probability
        
        # Calculate empirical accuracy
        wins = bin_data.get("wins", 0)
        total = bin_data.get("count", 1)
        empirical_accuracy = wins / total
        
        # More aggressive blending: 80% empirical, 20% raw (was 70/30)
        # You have clear systematic bias, so trust the historical data more
        calibrated = 0.80 * empirical_accuracy + 0.20 * raw_probability
        
        # Allow larger adjustments (±15% instead of ±10%)
        # Your historical data shows 30-40% point differences
        max_adjustment = 0.15
        calibrated = max(raw_probability - max_adjustment, 
                         min(raw_probability + max_adjustment, calibrated))
        
        return calibrated
    
    def update_outcome(self, date, city, actual_temp):
        """
        Update predictions with actual outcomes after the fact.
        Call this after you know the actual temperature.
        """
        updated = 0
        
        # Normalize city name for flexible matching
        city_lower = city.lower().strip()
        city_map = {
            "chicago": ["chicago", "chi"],
            "new york": ["new york", "new york city", "nyc"],
            "miami": ["miami", "mia"]
        }
        
        # Find which city group this belongs to
        matching_variants = None
        for key, variants in city_map.items():
            if city_lower in variants or any(v in city_lower for v in variants):
                matching_variants = variants
                break
        
        for pred in self.history["predictions"]:
            pred_city_lower = pred["city"].lower().strip()
            
            # Check if cities match (case-insensitive, with variants)
            cities_match = False
            if matching_variants:
                cities_match = pred_city_lower in matching_variants or any(v in pred_city_lower for v in matching_variants)
            else:
                cities_match = (pred_city_lower == city_lower)
            
            if pred["date"] == str(date) and cities_match and pred["outcome"] is None:
                # Determine if prediction was correct
                contract = pred["contract"]
                
                # Validate contract data before using it
                if "low" in contract and "high" in contract and contract["low"] is not None and contract["high"] is not None:
                    # Range contract
                    won = contract["low"] <= actual_temp <= contract["high"]
                elif "threshold" in contract and "side" in contract and contract["threshold"] is not None:
                    # Above/below contract
                    if contract["side"] == "above":
                        won = actual_temp >= contract["threshold"]
                    else:  # below
                        won = actual_temp <= contract["threshold"]
                else:
                    # Malformed contract data - skip this prediction
                    print(f"   ⚠️  Skipping malformed contract: {contract.get('subtitle', 'Unknown')}")
                    continue
                
                pred["outcome"] = "win" if won else "loss"
                pred["actual_temp"] = actual_temp
                
                # Update bin statistics
                prob_bin = f"{round(pred['our_prob'], 1):.1f}"
                if "bins" not in self.history:
                    self.history["bins"] = {}
                
                if prob_bin not in self.history["bins"]:
                    self.history["bins"][prob_bin] = {"wins": 0, "count": 0}
                
                self.history["bins"][prob_bin]["count"] += 1
                if won:
                    self.history["bins"][prob_bin]["wins"] += 1
                
                updated += 1
        
        if updated > 0:
            self.save_history()
            print(f"✅ Updated {updated} prediction(s) with actual temp {actual_temp}°F")
        else:
            print(f"⚠️  No matching predictions found for {city} on {date}")
            print(f"   (Searched for city variants of '{city}')")
    
    def get_calibration_report(self):
        """Generate a calibration report showing accuracy by probability bin."""
        if "bins" not in self.history or not self.history["bins"]:
            return "No calibration data yet"
        
        report = "\n📊 CALIBRATION REPORT\n" + "="*50 + "\n"
        report += f"{'Predicted':<12} {'Actual':<12} {'Count':<8} {'Calibration'}\n"
        report += "-"*50 + "\n"
        
        for bin_key in sorted(self.history["bins"].keys()):
            bin_data = self.history["bins"][bin_key]
            count = bin_data.get("count", 0)
            wins = bin_data.get("wins", 0)
            
            if count == 0:
                continue
            
            actual_rate = wins / count
            predicted_rate = float(bin_key)
            
            # Calibration quality
            error = abs(actual_rate - predicted_rate)
            if error < 0.05:
                quality = "✅ Excellent"
            elif error < 0.10:
                quality = "📊 Good"
            elif error < 0.15:
                quality = "⚠️  Needs work"
            else:
                quality = "❌ Poor"
            
            report += f"{predicted_rate:>6.0%}      {actual_rate:>6.0%}      {count:<6}   {quality}\n"
        
        return report


# ============ NEW: CURRENT OBSERVATIONS FETCHER (V12!) ============

def fetch_current_observations(lat, lon, city_name):
    """
    Fetch current temperature and recent observations to constrain forecasts.
    Uses Open-Meteo's current weather API (free, no key needed).
    
    Returns:
        dict with:
            - current_temp: Current temperature (°F)
            - current_time: Time of observation
            - temp_trend: Warming/cooling trend if detectable
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
        
        # Parse time
        current_time = datetime.fromisoformat(current_time_str.replace('Z', '+00:00'))
        
        obs = {
            "current_temp": current_temp,
            "current_time": current_time,
            "temp_trend": None  # Could add trend detection here
        }
        
        return obs
        
    except Exception as e:
        print(f"     ⚠️ Could not fetch current observations: {e}")
        return None


def apply_observation_constraints(forecast, current_obs, target_date):
    """
    Constrain forecast based on current observations.
    
    Rules:
    - If morning (6 AM - noon), max temp must be >= current temp
    - If it's already hot early, upward adjust forecast
    - If it's already cold late morning, downward adjust forecast
    
    Returns:
        adjusted_forecast, constraint_applied (bool), constraint_message
    """
    if not current_obs:
        return forecast, False, None
    
    current_temp = current_obs["current_temp"]
    current_time = current_obs["current_time"]
    
    # Check if observations are same-day as forecast
    if current_time.date() != target_date:
        return forecast, False, None
    
    # Get hour in local time
    hour = current_time.hour
    
    constraint_message = None
    adjusted_forecast = forecast
    constraint_applied = False
    
    # Morning constraint (6 AM - noon)
    if 6 <= hour <= 12:
        if current_temp > forecast:
            # Already warmer than forecast max - adjust up
            # Use a gradient: early morning = more aggressive adjustment
            adjustment_factor = 1.0 - ((hour - 6) / 6)  # 1.0 at 6 AM, 0.0 at noon
            
            temp_excess = current_temp - forecast
            adjustment = temp_excess * adjustment_factor
            
            adjusted_forecast = forecast + adjustment
            constraint_applied = True
            constraint_message = f"Morning temp ({current_temp:.1f}°F at {hour}:00) above forecast → adjusted +{adjustment:.1f}°F"
        
        elif current_temp > forecast - 5 and hour <= 8:
            # Early morning already near forecast max - slight upward bias
            adjusted_forecast = forecast + 1.0
            constraint_applied = True
            constraint_message = f"Early morning already {current_temp:.1f}°F (close to forecast) → adjusted +1.0°F"
    
    # Late morning constraint (9 AM - 1 PM)
    elif 9 <= hour <= 13:
        if current_temp < forecast - 15:
            # Noon temp way below forecast - unlikely to reach high forecast
            expected_gain = (forecast - current_temp) * 0.6  # Typical afternoon gain
            
            if expected_gain < (forecast - current_temp):
                adjustment = -(forecast - current_temp - expected_gain) * 0.5
                adjusted_forecast = forecast + adjustment
                constraint_applied = True
                constraint_message = f"Late morning temp ({current_temp:.1f}°F) suggests lower max → adjusted {adjustment:.1f}°F"
    
    return adjusted_forecast, constraint_applied, constraint_message


# ============ NEW: DYNAMIC UNCERTAINTY MODEL (V12!) ============

def calculate_dynamic_uncertainty(forecasts, forecast_sources, agreement_level, airmass_data=None):
    """
    Calculate forecast uncertainty dynamically based on:
    1. Model spread (disagreement)
    2. Atmospheric conditions (frontal systems = higher uncertainty)
    3. Forecast source diversity
    
    Returns:
        uncertainty_std (float): Standard deviation in °F
        uncertainty_factors (dict): Breakdown of what drove uncertainty
    """
    
    # Base uncertainty (best case: models agree, stable conditions)
    BASE_UNCERTAINTY = 2.5  # °F (increased from 2.0)
    
    # Maximum uncertainty (worst case: models disagree, frontal system)
    MAX_UNCERTAINTY = 8.0   # °F (increased from 6.0)
    
    factors = {}
    
    # 1. MODEL SPREAD COMPONENT
    if len(forecasts) >= 2:
        model_spread = max(forecasts) - min(forecasts)
        
        # Uncertainty scales MORE aggressively with spread
        # 0°F spread → 0 added uncertainty
        # 6°F spread → +3.0°F uncertainty (was +1.8°F)
        # 10°F spread → +5°F uncertainty (was +3.0°F)
        spread_uncertainty = min(model_spread * 0.5, 5.0)  # Increased from 0.3
        factors["model_spread"] = spread_uncertainty
    else:
        # Single model = higher baseline uncertainty
        spread_uncertainty = 1.0
        factors["model_spread"] = 1.0
        factors["single_model_penalty"] = True
    
    # 2. ATMOSPHERIC INSTABILITY COMPONENT
    instability_uncertainty = 0.0
    
    if airmass_data and airmass_data.get('temps_850mb'):
        temp_850_change = airmass_data['temps_850mb'].get('temp_850mb_change_24hr')
        
        if temp_850_change is not None:
            # Large airmass changes = frontal systems = higher uncertainty
            # ±15°F change → +1.5°F uncertainty
            # ±25°F change → +2.5°F uncertainty
            instability_uncertainty = min(abs(temp_850_change) * 0.1, 2.5)
            factors["airmass_instability"] = instability_uncertainty
    
    # 3. FORECAST SOURCE DIVERSITY
    # If NWS and Open-Meteo both present, they're correlated (both use GFS)
    # This gives false confidence
    diversity_penalty = 0.0
    
    if "NWS" in forecast_sources and "Open-Meteo" in forecast_sources:
        # Both GFS-based sources present
        if "HRRR" not in forecast_sources:
            # Only GFS-based sources = less diversity
            diversity_penalty = 0.5
            factors["low_diversity"] = diversity_penalty
        else:
            # Have HRRR too, so some diversity
            diversity_penalty = 0.2
            factors["medium_diversity"] = diversity_penalty
    
    # 4. AGREEMENT LEVEL MODIFIER
    agreement_modifier = {
        'high': 0.8,    # Reduce uncertainty by 20%
        'medium': 1.0,  # No change
        'low': 1.3      # Increase uncertainty by 30%
    }
    
    modifier = agreement_modifier.get(agreement_level, 1.0)
    factors["agreement_modifier"] = modifier
    
    # CALCULATE TOTAL UNCERTAINTY
    total_uncertainty = BASE_UNCERTAINTY + spread_uncertainty + instability_uncertainty + diversity_penalty
    total_uncertainty *= modifier
    
    # Clamp to reasonable bounds
    total_uncertainty = max(BASE_UNCERTAINTY, min(total_uncertainty, MAX_UNCERTAINTY))
    
    factors["final_uncertainty"] = total_uncertainty
    
    return total_uncertainty, factors


# ============ NEW: SMART MODEL WEIGHTING (V12!) ============

def calculate_smart_weights(forecast_sources, model_spread, agreement_level):
    """
    Calculate weights for ensemble forecast that account for:
    1. Model independence (HRRR is independent, NWS/Open-Meteo both use GFS)
    2. Model skill in different regimes
    3. Forecast agreement
    
    Returns:
        weights (list): Weights corresponding to forecast_sources order
        weight_explanation (str): Human-readable explanation
    """
    
    n_models = len(forecast_sources)
    
    if n_models == 1:
        return [1.0], "Single forecast source"
    
    # Initialize equal weights
    weights = [1.0 / n_models] * n_models
    
    explanation_parts = []
    
    # DETECT GFS CORRELATION
    has_hrrr = "HRRR" in forecast_sources
    has_nws = "NWS" in forecast_sources
    has_open_meteo = "Open-Meteo" in forecast_sources
    
    gfs_count = sum([has_nws, has_open_meteo])
    
    if has_hrrr and gfs_count >= 1:
        # We have both independent HRRR and GFS-based forecasts
        # Give HRRR more weight since it's truly independent
        
        hrrr_idx = forecast_sources.index("HRRR")
        
        if gfs_count == 2:
            # All three sources: HRRR, NWS, Open-Meteo
            # NWS and Open-Meteo are correlated, so downweight them
            
            if model_spread <= 2.0:
                # High agreement - give HRRR 50%, split rest between GFS sources
                weights[hrrr_idx] = 0.50
                
                for i, source in enumerate(forecast_sources):
                    if source in ["NWS", "Open-Meteo"]:
                        weights[i] = 0.25
                
                explanation_parts.append("High agreement: HRRR 50%, GFS sources 25% each")
            
            elif model_spread <= 4.0:
                # Medium agreement - balanced but favor HRRR
                weights[hrrr_idx] = 0.45
                
                for i, source in enumerate(forecast_sources):
                    if source in ["NWS", "Open-Meteo"]:
                        weights[i] = 0.275
                
                explanation_parts.append("Medium agreement: HRRR 45%, GFS sources 27.5% each")
            
            else:
                # Low agreement (>4°F spread) - EQUAL weights
                # When models disagree this much, no model should dominate
                weights[hrrr_idx] = 0.333
                
                for i, source in enumerate(forecast_sources):
                    if source in ["NWS", "Open-Meteo"]:
                        weights[i] = 0.333
                
                explanation_parts.append("Low agreement (>4°F): Equal weights (no model dominant)")
        
        else:
            # HRRR + one GFS source (either NWS or Open-Meteo)
            # More balanced since only one GFS source
            
            if model_spread <= 2.0:
                weights[hrrr_idx] = 0.55
                weights[1 - hrrr_idx] = 0.45
                explanation_parts.append("High agreement: HRRR 55%, GFS 45%")
            else:
                weights[hrrr_idx] = 0.50
                weights[1 - hrrr_idx] = 0.50
                explanation_parts.append("Low agreement: Equal weights")
    
    elif gfs_count == 2 and not has_hrrr:
        # Only GFS-based sources (NWS + Open-Meteo)
        # They're correlated, so treat cautiously
        
        explanation_parts.append("Only GFS sources (correlated) - equal weights")
        # Keep equal weights but note the correlation issue
    
    explanation = "; ".join(explanation_parts) if explanation_parts else "Equal weights"
    
    return weights, explanation


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
    },
    "austin": {
        "name": "Austin",
        "csv_file": "weather_data_austin.csv",
        "lat": 30.2672,
        "lon": -97.7431,
        "kalshi_series": "KXHIGHAUS",
        "timezone": "America/Chicago",
        "utc_offset": -6
    },
    "la": {
        "name": "Los Angeles",
        "csv_file": "weather_data_la.csv",
        "lat": 34.0522,
        "lon": -118.2437,
        "kalshi_series": "KXHIGHLAX",
        "timezone": "America/Los_Angeles",
        "utc_offset": -8
    }
}

# Will collect all qualifying bets across all cities
all_qualifying_bets = []

# Default to all cities
selected_cities = ["chicago", "nyc", "miami", "austin", "la"]


# ============ TIMEZONE HELPERS ============
def get_eastern_now():
    """Get the current datetime in US Eastern time."""
    return datetime.now(KALSHI_TIMEZONE)

def get_target_date(force_today=False):
    """
    Get the target date for market lookup.

    By default:
    - Before 6 PM Eastern → look at today's markets
    - After 6 PM Eastern → look at tomorrow's markets

    Args:
        force_today: If True, always return today regardless of time
    """
    eastern_now = get_eastern_now()
    
    if force_today:
        return eastern_now.date(), True
    
    # Check hour in Eastern time
    hour = eastern_now.hour
    
    # Before evening cutoff → today's markets
    # After evening cutoff → tomorrow's markets
    if hour < EVENING_CUTOFF_HOUR:
        return eastern_now.date(), True
    else:
        return (eastern_now + timedelta(days=1)).date(), False

def print_header(target_date, is_today):
    """Print analysis header."""
    print("\n" + "="*70)
    print(" KALSHI WEATHER BETTING MODEL v12.6 (PROBABILITY CAP FIXES)")
    print("="*70)
    print(f" Target Date: {target_date.strftime('%A, %B %d, %Y')}")
    print(f" Analysis Time: {get_eastern_now().strftime('%I:%M %p ET')}")
    print(f" Analyzing: {'TODAY' if is_today else 'TOMORROW'}'s markets")
    print("="*70)
    
    print("\n🚨 V12.6 CRITICAL FIXES:")
    print("   • FIXED: 100% confidence bug (was winning only 28%)")
    print("   • Hard cap at 95% for BOTH YES and NO bets")
    print("   • Uncertainty caps: ±8°F = max 85%, ±6°F = max 88%, ±5°F = max 90%")
    print("   • Major disagreement cap: max 85% confidence")
    print("   • Agreement level caps: low=88%, medium=92%, high=95%")
    
    print("\n🎯 OPTIMIZED FOR YOUR BETTING HISTORY:")
    print("   Based on your 46 historical bets:")
    print("   • 30% estimates → 70% actual wins (you're too conservative!)")
    print("   • 40-60¢ contracts: Your sweet spot (71% win rate)")
    print("   • Edge requirements lowered: 7-10% (was 12%)")
    print("   • Kelly increased to 0.80 (was 0.75)")
    print("   • Calibration boost REMOVED (was causing overconfidence)")
    
    print("\n🌤️  V12.5 ENHANCEMENTS (BUILT-IN):")
    print("   ✅ Cloud timing analysis (morning/peak heating clouds)")
    print("   ✅ Precipitation timing analysis (rain during peak heating)")
    print("   ✅ Weather regime detection (pressure-based uncertainty)")
    print("   ✅ Smart model weighting (reduces GFS correlation)")
    print("   ✅ Dynamic uncertainty (scales with conditions)")
    
    print("\n✨ V12 CORE ENHANCEMENTS:")
    print("   • Dynamic forecast uncertainty (scales with model spread)")
    print("   • Current observation constraints (uses real-time temps)")
    print("   • Calibration tracking (learns from your past accuracy)")
    print("   • Smart model weighting (accounts for GFS correlation)")
    print()

# ============ END OF PART 1/4 ============

# ============ PART 2/4 OF ENSEMBLE_V9.PY ============
# Data fetching, wind/airmass analysis, atmospheric features

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
    """Get the most recent available HRRR model run."""
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
    
    # Convert to local time for display
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
    cloud_covers = []  # NEW: Track cloud cover
    precip_rates = []   # NEW: Track precipitation rates
    pressures = []      # NEW: Track surface pressure for regime detection
    forecast_times = [] # NEW: Track actual forecast times
    
    for fxx in forecast_hours:
        try:
            H = Herbie(
                model_run_str,
                model='hrrr',
                product='sfc',
                fxx=fxx
            )
            
            # Calculate actual forecast time
            forecast_time = model_run_datetime + timedelta(hours=fxx)
            forecast_times.append(forecast_time)
            
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
            
            # Fetch wind data
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
            
            # NEW: Fetch cloud cover data (total cloud cover)
            try:
                ds_cloud = H.xarray("TCDC:entire atmosphere", remove_grib=True)
                cloud_var = ds_cloud['tcc'] if 'tcc' in ds_cloud else ds_cloud['tcdceatm']
                cloud_cover = float(cloud_var.values[min_idx])
                cloud_covers.append(cloud_cover)
            except:
                pass
            
            # NEW: Fetch precipitation rate
            try:
                ds_precip = H.xarray("PRATE:surface", remove_grib=True)
                precip_var = ds_precip['prate'] if 'prate' in ds_precip else ds_precip['pratessfc']
                precip_rate_kg = float(precip_var.values[min_idx])
                # Convert kg/m²/s to inches/hour
                precip_rate_in = precip_rate_kg * 0.1417  # Conversion factor
                precip_rates.append(precip_rate_in)
            except:
                precip_rates.append(0.0)
            
            # NEW: Fetch surface pressure for regime detection
            try:
                ds_pres = H.xarray("PRES:surface", remove_grib=True)
                pres_var = ds_pres['sp'] if 'sp' in ds_pres else ds_pres['pressfc']
                pressure_pa = float(pres_var.values[min_idx])
                pressure_mb = pressure_pa / 100.0  # Convert Pa to mb
                pressures.append(pressure_mb)
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
        'source': 'HRRR',
        # NEW: Cloud cover and precipitation timing data
        'cloud_covers': cloud_covers,
        'precip_rates': precip_rates,
        'forecast_times': forecast_times,
        # NEW: Pressure data for regime detection
        'pressures': pressures,
    }
    
    return forecast_high, weather_data


def fetch_hrrr_850mb_temp(lat, lon, target_date, utc_offset):
    """
    Fetch 850mb temperature from HRRR for airmass analysis.
    850mb is ~5000 feet elevation - represents the "true" airmass.
    """
    if not HERBIE_AVAILABLE:
        return None
    
    try:
        model_run_date, model_run_hour = get_most_recent_hrrr_run()
        if not model_run_date:
            return None
        
        model_run_str = f"{model_run_date.strftime('%Y-%m-%d')} {model_run_hour:02d}:00"
        model_run_datetime = datetime.combine(model_run_date, datetime.min.time().replace(hour=model_run_hour))
        
        # Get 850mb temp at noon local time for target date
        target_noon_local_utc = 12 - utc_offset
        target_noon = datetime.combine(target_date, datetime.min.time().replace(hour=target_noon_local_utc))
        if (12 - utc_offset) >= 24:
            target_noon += timedelta(days=1)
        elif (12 - utc_offset) < 0:
            target_noon -= timedelta(days=1)
        
        hours_to_noon = int((target_noon - model_run_datetime).total_seconds() / 3600)
        
        if hours_to_noon < 1 or hours_to_noon > 48:
            return None
        
        # Fetch 850mb temp
        H = Herbie(
            model_run_str,
            model='hrrr',
            product='prs',
            fxx=hours_to_noon
        )
        
        ds = H.xarray("TMP:850 mb", remove_grib=True)
        temp_data = ds['t']
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        
        target_lon = lon if lon > 0 else lon + 360
        dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        
        temp_k = float(temp_data.values[min_idx])
        temp_f = (temp_k - 273.15) * 9/5 + 32
        
        # Try to get 24hr change
        temp_change_24hr = None
        try:
            if hours_to_noon >= 24:
                H_24hr_ago = Herbie(
                    model_run_str,
                    model='hrrr',
                    product='prs',
                    fxx=hours_to_noon - 24
                )
                ds_24hr = H_24hr_ago.xarray("TMP:850 mb", remove_grib=True)
                temp_24hr_k = float(ds_24hr['t'].values[min_idx])
                temp_24hr_f = (temp_24hr_k - 273.15) * 9/5 + 32
                temp_change_24hr = temp_f - temp_24hr_f
        except:
            pass
        
        return {
            'temp_850mb_f': temp_f,
            'temp_850mb_change_24hr': temp_change_24hr
        }
        
    except Exception as e:
        return None


def nws_round(temp):
    """
    Apply NWS/NOAA rounding rules used by Kalshi for market resolution.
    
    NWS rounds UP when temperature is at 0.5 or higher:
    - Positive: 38.5 → 39, 38.4 → 38
    - Negative: -2.5 → -3, -2.4 → -2
    """
    if temp >= 0:
        return int(np.floor(temp + 0.5))
    else:
        return int(np.ceil(temp - 0.5))


def get_bin_for_temp(temp, valid_bins=None):
    """
    Get the Kalshi bin that contains this temperature.
    
    IMPORTANT: This applies NWS rounding first, since Kalshi uses NWS-rounded
    temperatures for market resolution.
    """
    # First apply NWS rounding
    rounded_temp = nws_round(temp)
    
    # If valid bins provided, find which one contains this temperature
    if valid_bins:
        for low, high in sorted(valid_bins):
            if low <= rounded_temp <= high:
                return (low, high)
        print(f"     ⚠️ Warning: NWS-rounded temp {rounded_temp}°F not in any Kalshi bin")
    
    # Calculate theoretical bin (2-degree bins with odd lower bounds)
    if rounded_temp % 2 == 1:  # odd
        lower = rounded_temp
    else:  # even
        lower = rounded_temp - 1
    
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

    # Determine cardinal direction and advection
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
    else:  # 292.5 to 337.5
        cardinal = "NW"
        advection_type = "cold"
    
    return {
        "cardinal": cardinal,
        "advection_type": advection_type,
        "degrees": wind_dir
    }


def extract_atmospheric_features(hrrr_data):
    """
    Extract atmospheric features from HRRR data for ML model.
    
    Returns:
        dict of features or None if insufficient data
    """
    if not hrrr_data:
        return None
    
    features = {}
    
    # Wind features
    wind_speed = hrrr_data.get('wind_speed')
    wind_dir = hrrr_data.get('wind_dir')
    
    if wind_speed is not None:
        features['wind_speed'] = wind_speed
    
    if wind_dir is not None:
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
    
    # 850mb features (airmass)
    temps_850mb = hrrr_data.get('temps_850mb')
    if temps_850mb:
        temp_850 = temps_850mb.get('temp_850mb_f')
        temp_change = temps_850mb.get('temp_850mb_change_24hr')
        
        if temp_850 is not None:
            features['temp_850mb'] = temp_850
        
        if temp_change is not None:
            features['temp_850mb_change_24hr'] = temp_change
            # Binary flags for significant changes
            features['warm_advection_strong'] = 1 if temp_change >= WARM_ADVECTION_THRESHOLD else 0
            features['cold_advection_strong'] = 1 if temp_change <= COLD_ADVECTION_THRESHOLD else 0
    
    return features or None


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


# ============ CLOUD COVER TIMING ANALYSIS ============

def analyze_cloud_timing(hrrr_data, utc_offset):
    """
    Analyze cloud cover timing to predict impact on max temperature.
    
    Morning clouds (before 10 AM) have MUCH bigger impact than afternoon clouds.
    Physics: Morning clouds block sunrise heating, preventing temperature rise.
             Afternoon clouds arrive after heating is done, minimal impact.
    
    Args:
        hrrr_data: Dictionary with 'cloud_covers' and 'forecast_times'
        utc_offset: UTC offset for the location
    
    Returns:
        dict with cloud timing analysis and forecast adjustment
    """
    if not hrrr_data or 'cloud_covers' not in hrrr_data or 'forecast_times' not in hrrr_data:
        return None
    
    cloud_covers = hrrr_data.get('cloud_covers', [])
    forecast_times = hrrr_data.get('forecast_times', [])
    
    if not cloud_covers or not forecast_times or len(cloud_covers) != len(forecast_times):
        return None
    
    # Separate into morning (6 AM - 10 AM) and afternoon (10 AM - 4 PM) periods
    morning_clouds = []
    afternoon_clouds = []
    peak_heating_clouds = []
    
    for cloud, time in zip(cloud_covers, forecast_times):
        # Convert UTC time to local hour
        local_hour = (time.hour + utc_offset) % 24
        
        if 6 <= local_hour < 10:
            morning_clouds.append(cloud)
        elif 10 <= local_hour < 16:
            afternoon_clouds.append(cloud)
        
        if PEAK_HEATING_START <= local_hour < PEAK_HEATING_END:
            peak_heating_clouds.append(cloud)
    
    analysis = {
        "morning_cloud_avg": np.mean(morning_clouds) if morning_clouds else 0,
        "afternoon_cloud_avg": np.mean(afternoon_clouds) if afternoon_clouds else 0,
        "peak_heating_cloud_avg": np.mean(peak_heating_clouds) if peak_heating_clouds else 0,
        "forecast_adjustment": 0.0,
        "message": None
    }
    
    # Assess impact
    morning_avg = analysis["morning_cloud_avg"]
    
    if morning_avg >= MORNING_CLOUD_THRESHOLD:
        # Significant morning clouds - suppress max temp
        # More clouds = more suppression (non-linear)
        cloud_factor = (morning_avg - MORNING_CLOUD_THRESHOLD) / (100 - MORNING_CLOUD_THRESHOLD)
        adjustment = MORNING_CLOUD_IMPACT * (0.5 + 0.5 * cloud_factor)  # -1.5°F to -3.0°F
        
        analysis["forecast_adjustment"] = adjustment
        analysis["message"] = f"☁️ Morning clouds ({morning_avg:.0f}% avg) will suppress max by ~{abs(adjustment):.1f}°F"
        
    elif analysis["peak_heating_cloud_avg"] >= AFTERNOON_CLOUD_THRESHOLD:
        # Afternoon clouds during peak heating - minor impact
        adjustment = -1.0
        analysis["forecast_adjustment"] = adjustment
        analysis["message"] = f"⛅ Afternoon clouds ({analysis['peak_heating_cloud_avg']:.0f}% during peak heating) - minor impact (~1°F)"
    
    else:
        analysis["message"] = "☀️ Mostly clear skies - no cloud suppression"
    
    return analysis


# ============ PRECIPITATION TIMING ANALYSIS ============

def analyze_precip_timing(hrrr_data, utc_offset):
    """
    Analyze precipitation timing to predict impact on max temperature.
    
    Rain during peak heating hours (11 AM - 3 PM) can DROP max temp by 5-15°F!
    Physics: Rain cools through evaporation + clouds block sun + wet ground doesn't heat.
    
    Args:
        hrrr_data: Dictionary with 'precip_rates' and 'forecast_times'
        utc_offset: UTC offset for the location
    
    Returns:
        dict with precipitation timing analysis and forecast adjustment
    """
    if not hrrr_data or 'precip_rates' not in hrrr_data or 'forecast_times' not in hrrr_data:
        return None
    
    precip_rates = hrrr_data.get('precip_rates', [])
    forecast_times = hrrr_data.get('forecast_times', [])
    
    if not precip_rates or not forecast_times or len(precip_rates) != len(forecast_times):
        return None
    
    # Focus on peak heating period
    peak_heating_precip = []
    total_precip_amount = 0
    
    for precip, time in zip(precip_rates, forecast_times):
        local_hour = (time.hour + utc_offset) % 24
        total_precip_amount += precip
        
        if PEAK_HEATING_START <= local_hour < PEAK_HEATING_END:
            peak_heating_precip.append(precip)
    
    analysis = {
        "peak_heating_precip_avg": np.mean(peak_heating_precip) if peak_heating_precip else 0,
        "total_precip": total_precip_amount,
        "has_peak_heating_rain": any(p > 0.01 for p in peak_heating_precip),
        "forecast_adjustment": 0.0,
        "message": None
    }
    
    peak_avg = analysis["peak_heating_precip_avg"]
    
    if peak_avg >= HEAVY_PRECIP_THRESHOLD:
        # Heavy rain during peak heating - major impact
        # The heavier the rain, the bigger the suppression
        rain_factor = min(peak_avg / HEAVY_PRECIP_THRESHOLD, 2.0)  # Cap at 2x
        adjustment = PRECIP_PEAK_HEATING_IMPACT * rain_factor  # Up to -16°F for extreme rain
        
        analysis["forecast_adjustment"] = adjustment
        analysis["message"] = f"🌧️ HEAVY RAIN during peak heating ({peak_avg:.2f} in/hr) - MAJOR temp suppression (~{abs(adjustment):.0f}°F)"
        
    elif analysis["has_peak_heating_rain"]:
        # Light rain during peak heating - moderate impact
        adjustment = PRECIP_PEAK_HEATING_IMPACT * 0.4  # ~-3°F
        
        analysis["forecast_adjustment"] = adjustment
        analysis["message"] = f"🌦️ Light rain during peak heating - moderate suppression (~{abs(adjustment):.0f}°F)"
    
    elif total_precip_amount > 0.05:
        # Rain outside peak heating - minor impact (wet ground effect)
        adjustment = -1.5
        analysis["forecast_adjustment"] = adjustment
        analysis["message"] = "💧 Rain outside peak heating - minor impact from wet ground"
    
    else:
        analysis["message"] = "☀️ No significant precipitation - no rain suppression"
    
    return analysis


# ============ WEATHER REGIME DETECTION ============

def detect_weather_regime(hrrr_data):
    """
    Detect weather regime to adjust forecast uncertainty.
    
    Stable high pressure = low uncertainty (±1.8°F)
    Frontal systems = high uncertainty (±4.5°F)
    
    Physics: High pressure = calm, predictable weather
             Fronts = rapid changes, poor predictability
    
    Args:
        hrrr_data: Dictionary with 'pressures' and 'forecast_times'
    
    Returns:
        dict with regime type and uncertainty adjustment
    """
    if not hrrr_data or 'pressures' not in hrrr_data:
        return {
            "regime": "unknown",
            "uncertainty_adjustment": 1.0,  # No adjustment
            "message": "Regime detection unavailable"
        }
    
    pressures = hrrr_data.get('pressures', [])
    
    if not pressures or len(pressures) < 3:
        return {
            "regime": "unknown",
            "uncertainty_adjustment": 1.0,
            "message": "Insufficient pressure data"
        }
    
    avg_pressure = np.mean(pressures)
    pressure_change = (pressures[-1] - pressures[0]) / max(1, len(pressures) / 3)  # Per 3-hour change
    pressure_variance = np.std(pressures)
    
    # Detect regime
    if avg_pressure >= HIGH_PRESSURE_THRESHOLD and abs(pressure_change) < 1.0:
        regime = "stable_high_pressure"
        uncertainty_mult = 0.7  # Lower uncertainty
        message = f"🌤️ STABLE HIGH PRESSURE ({avg_pressure:.0f} mb) - Predictable conditions"
        
    elif avg_pressure <= LOW_PRESSURE_THRESHOLD or abs(pressure_change) >= FRONTAL_PRESSURE_CHANGE:
        regime = "frontal_system"
        uncertainty_mult = 1.5  # Higher uncertainty
        
        if pressure_change < -FRONTAL_PRESSURE_CHANGE:
            message = f"🌀 APPROACHING LOW/FRONT ({avg_pressure:.0f} mb, falling {abs(pressure_change):.1f} mb/3hr) - High uncertainty"
        else:
            message = f"🌀 ACTIVE FRONT ({avg_pressure:.0f} mb, changing {abs(pressure_change):.1f} mb/3hr) - High uncertainty"
    
    elif pressure_variance > 2.0:
        regime = "variable"
        uncertainty_mult = 1.2  # Moderately higher uncertainty
        message = f"🌥️ Variable conditions ({avg_pressure:.0f} mb) - Moderate uncertainty"
    
    else:
        regime = "normal"
        uncertainty_mult = 1.0  # Normal uncertainty
        message = f"⛅ Normal conditions ({avg_pressure:.0f} mb)"
    
    return {
        "regime": regime,
        "avg_pressure": avg_pressure,
        "pressure_change_3hr": pressure_change,
        "uncertainty_adjustment": uncertainty_mult,
        "message": message
    }


# ============ ATMOSPHERIC ANALOG MODEL ============

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
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate cross-validated MAE
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        if verbose:
            print(f"      ✅ Training complete")
            print(f"      Samples: {self.training_samples}")
            print(f"      Features: {len(feature_cols)}")
            print(f"      CV MAE: {cv_mae:.2f}°F")
        
        return True
    
    def predict_adjustment(self, atmospheric_features):
        """
        Predict forecast adjustment based on current atmospheric conditions.
        
        Args:
            atmospheric_features: Dict of current atmospheric features
        
        Returns:
            adjustment: Predicted °F adjustment
            confidence: Confidence score (0-1)
        """
        if not self.is_trained:
            return 0.0, 0.0
        
        # Build feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in atmospheric_features:
                feature_vector.append(atmospheric_features[feature_name])
            else:
                # Feature missing - use 0 (models trained on this will handle it)
                feature_vector.append(0)
        
        # Predict
        X = np.array([feature_vector])
        adjustment = self.model.predict(X)[0]
        
        # Estimate confidence based on feature similarity to training data
        # Simple approach: confidence decreases if features are far from training distribution
        confidence = 0.7  # Default medium confidence
        
        return adjustment, confidence
    
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


# ============ END OF PART 2/4 ============

# ============ PART 3/4 OF ENSEMBLE_V9.PY ============
# Modified probability functions and enhanced analyze_city with V9 features

# ============ PROBABILITY MODELS (MODIFIED FOR V9!) ============

def calibrated_probability(contract_low, contract_high, forecast, uncertainty_std, 
                          agreement_level='high', calibration_tracker=None):
    """
    Calibrated probability model with DYNAMIC uncertainty (V12!).
    
    V12 CHANGES:
    - No longer adjusts std based on agreement (that's in dynamic_uncertainty)
    - Can apply calibration adjustment if tracker provided
    """
    # Just use the provided std directly (already accounts for agreement)
    prob = stats.norm.cdf(contract_high + 0.5, forecast, uncertainty_std) - \
           stats.norm.cdf(contract_low - 0.5, forecast, uncertainty_std)
    
    prob = max(min(prob, 0.99), 0.01)
    
    # Apply calibration adjustment if available (V12!)
    if calibration_tracker:
        prob = calibration_tracker.get_calibration_adjustment(prob)
    
    return prob


def calibrated_below_probability(threshold, forecast, uncertainty_std, 
                                 agreement_level='high', calibration_tracker=None):
    """Calibrated probability for 'X or below' contracts with dynamic uncertainty (V12!)."""
    prob = stats.norm.cdf(threshold + 0.5, forecast, uncertainty_std)
    prob = max(min(prob, 0.99), 0.01)
    
    if calibration_tracker:
        prob = calibration_tracker.get_calibration_adjustment(prob)
    
    return prob


def calibrated_above_probability(threshold, forecast, uncertainty_std, 
                                 agreement_level='high', calibration_tracker=None):
    """Calibrated probability for 'X or above' contracts with dynamic uncertainty (V12!)."""
    prob = 1 - stats.norm.cdf(threshold - 0.5, forecast, uncertainty_std)
    prob = max(min(prob, 0.99), 0.01)
    
    if calibration_tracker:
        prob = calibration_tracker.get_calibration_adjustment(prob)
    
    return prob


# ============ CITY ANALYSIS (ENHANCED FOR V12!) ============

def analyze_city(city_key, bankroll, target_date, calibration_tracker=None):
    """
    Analyze a single city and return all qualifying bets.
    
    V12 ENHANCEMENTS:
    - Fetches current observations
    - Uses dynamic uncertainty calculation
    - Applies smart model weighting
    - Uses calibration tracker
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

    target_str = target_date.strftime("%Y-%m-%d")

    # ============ FETCH CURRENT OBSERVATIONS (V12 NEW!) ============
    print(f"\n  🌡️  Fetching current observations...")
    current_obs = fetch_current_observations(city["lat"], city["lon"], city["name"])
    
    if current_obs:
        print(f"     ✅ Current: {current_obs['current_temp']:.1f}°F at {current_obs['current_time'].strftime('%I:%M %p')}")
    else:
        print(f"     ⚠️  Current observations unavailable")

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
        
        # Analyze wind and airmass
        if hrrr_data:
            airmass_info = analyze_airmass_and_wind(hrrr_data, city["name"])
            if airmass_info:
                if airmass_info.get("wind_analysis"):
                    print(f"     {airmass_info['wind_analysis']}")
                if airmass_info.get("airmass_analysis"):
                    print(f"     {airmass_info['airmass_analysis']}")
            
            # NEW: Cloud cover timing analysis
            print(f"\n     ☁️ Cloud Cover Timing Analysis:")
            cloud_analysis = analyze_cloud_timing(hrrr_data, city["utc_offset"])
            if cloud_analysis and cloud_analysis.get("message"):
                print(f"     {cloud_analysis['message']}")
                if abs(cloud_analysis.get("forecast_adjustment", 0)) > 0.5:
                    print(f"        Impact: {cloud_analysis['forecast_adjustment']:+.1f}°F")
            else:
                print(f"     ⚠️ Cloud timing data unavailable")
            
            # NEW: Precipitation timing analysis  
            print(f"\n     🌧️ Precipitation Timing Analysis:")
            precip_analysis = analyze_precip_timing(hrrr_data, city["utc_offset"])
            if precip_analysis and precip_analysis.get("message"):
                print(f"     {precip_analysis['message']}")
                if abs(precip_analysis.get("forecast_adjustment", 0)) > 0.5:
                    print(f"        Impact: {precip_analysis['forecast_adjustment']:+.1f}°F")
            else:
                print(f"     ⚠️ Precipitation timing data unavailable")
            
            # NEW: Weather regime detection
            print(f"\n     🌡️ Weather Regime Detection:")
            regime_info = detect_weather_regime(hrrr_data)
            if regime_info and regime_info.get("message"):
                print(f"     {regime_info['message']}")
                if regime_info.get("uncertainty_adjustment", 1.0) != 1.0:
                    print(f"        Uncertainty multiplier: {regime_info['uncertainty_adjustment']:.2f}x")
            else:
                print(f"     ⚠️ Regime detection unavailable")
                
    else:
        print("     ⚠️ HRRR unavailable")
        airmass_info = None
        cloud_analysis = None
        precip_analysis = None
        regime_info = None

    # 3. NWS forecast
    print(f"\n  [3] National Weather Service:")
    nws_forecast = fetch_nws_forecast(city["lat"], city["lon"], target_date)
    
    if nws_forecast:
        print(f"     ✅ Forecast: {nws_forecast:.1f}°F")
    else:
        print("     ⚠️ NWS unavailable")

    # ============ COLLECT FORECASTS ============
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
    
    # ============ CALCULATE MODEL SPREAD & AGREEMENT ============
    if len(forecasts) == 1:
        model_spread = 0
        agreement_level = 'medium'
    else:
        model_spread = max(forecasts) - min(forecasts)
        
        if model_spread <= CONFIDENCE_BOOST_THRESHOLD:
            agreement_level = 'high'
        elif model_spread <= ENSEMBLE_AGREEMENT_THRESHOLD:
            agreement_level = 'medium'
        else:
            agreement_level = 'low'
    
    # ============ SMART MODEL WEIGHTING (V12 NEW!) ============
    weights, weight_explanation = calculate_smart_weights(
        forecast_sources, model_spread, agreement_level
    )
    
    print(f"\n  ⚖️  Smart Weighting (V12!):")
    print(f"     {weight_explanation}")
    for source, weight in zip(forecast_sources, weights):
        print(f"     {source}: {weight:.1%}")
    
    # Calculate weighted ensemble
    ensemble_forecast = sum(f * w for f, w in zip(forecasts, weights))
    # SAVE ORIGINAL for display purposes before adjustments
    original_weighted_ensemble = ensemble_forecast
    
    # ============ DYNAMIC UNCERTAINTY CALCULATION (V12 NEW!) ============
    uncertainty_std, uncertainty_factors = calculate_dynamic_uncertainty(
        forecasts, forecast_sources, agreement_level, hrrr_data
    )
    
    # NEW: Apply regime-based uncertainty adjustment
    if regime_info and regime_info.get("uncertainty_adjustment"):
        original_uncertainty = uncertainty_std
        uncertainty_std *= regime_info["uncertainty_adjustment"]
        print(f"\n  🌡️ Regime Uncertainty Adjustment:")
        print(f"     {regime_info['message']}")
        print(f"     Uncertainty: {original_uncertainty:.1f}°F → {uncertainty_std:.1f}°F")
    
    print(f"\n  📊 Dynamic Uncertainty Analysis (V12!):")
    print(f"     Final uncertainty: ±{uncertainty_std:.1f}°F (68% confidence)")
    for factor, value in uncertainty_factors.items():
        if factor == "final_uncertainty":
            continue
        if isinstance(value, bool):
            if value:
                print(f"       • {factor}")
        elif isinstance(value, float):
            print(f"       • {factor}: {value:.2f}°F")
    
    # ============ APPLY OBSERVATION CONSTRAINTS (V12 NEW!) ============
    original_forecast = ensemble_forecast
    ensemble_forecast, constraint_applied, constraint_message = apply_observation_constraints(
        ensemble_forecast, current_obs, target_date
    )
    
    if constraint_applied:
        print(f"\n  🔒 Observation Constraint Applied (V12!):")
        print(f"     {constraint_message}")
        print(f"     Forecast: {original_forecast:.1f}°F → {ensemble_forecast:.1f}°F")
    
    # ============ CLOUD & PRECIPITATION ADJUSTMENTS (V12 NEW!) ============
    cloud_precip_adjustment = 0.0
    cloud_precip_reasons = []
    
    if cloud_analysis and abs(cloud_analysis.get("forecast_adjustment", 0)) > 0.5:
        cloud_adj = cloud_analysis["forecast_adjustment"]
        cloud_precip_adjustment += cloud_adj
        cloud_precip_reasons.append(f"Cloud timing: {cloud_adj:+.1f}°F")
    
    if precip_analysis and abs(precip_analysis.get("forecast_adjustment", 0)) > 0.5:
        precip_adj = precip_analysis["forecast_adjustment"]
        cloud_precip_adjustment += precip_adj
        cloud_precip_reasons.append(f"Precipitation timing: {precip_adj:+.1f}°F")
    
    if abs(cloud_precip_adjustment) > 0.5:
        original_ensemble = ensemble_forecast
        ensemble_forecast += cloud_precip_adjustment
        print(f"\n  ☁️🌧️ CLOUD/PRECIP TIMING ADJUSTMENT (V12 NEW!):")
        for reason in cloud_precip_reasons:
            print(f"     • {reason}")
        print(f"     Total adjustment: {cloud_precip_adjustment:+.1f}°F")
        print(f"     Ensemble: {original_ensemble:.1f}°F → {ensemble_forecast:.1f}°F")
    
    # ============ ML ADJUSTMENTS (from V11, kept in V12) ============
    # Apply wind/airmass-based adjustments to ensemble forecast
    ml_adjustment = 0.0
    adjustment_reasons = []
    
    if airmass_info:
        forecast_adjustment = airmass_info.get("forecast_adjustment")
        
        # Get airmass data if available
        if hrrr_data and hrrr_data.get('temps_850mb'):
            temp_850_change = hrrr_data['temps_850mb'].get('temp_850mb_change_24hr')
            
            if temp_850_change is not None:
                # Strong airmass signal = adjust forecast
                if temp_850_change >= WARM_ADVECTION_THRESHOLD:
                    if hrrr_forecast and hrrr_forecast < ensemble_forecast:
                        ml_adjustment = model_spread * 0.20
                        adjustment_reasons.append(f"Warm airmass (+{temp_850_change:.1f}°F/24hr)")
                elif temp_850_change <= COLD_ADVECTION_THRESHOLD:
                    if hrrr_forecast and hrrr_forecast > ensemble_forecast:
                        ml_adjustment = model_spread * -0.20
                        adjustment_reasons.append(f"Cold airmass ({temp_850_change:.1f}°F/24hr)")
        
        # Wind-only signal (if no 850mb data)
        elif forecast_adjustment and model_spread > CONFIDENCE_BOOST_THRESHOLD:
            if forecast_adjustment == "warmer":
                if hrrr_forecast and hrrr_forecast < ensemble_forecast:
                    ml_adjustment = model_spread * 0.08
                    adjustment_reasons.append("Warm wind advection")
            elif forecast_adjustment == "colder":
                if hrrr_forecast and hrrr_forecast > ensemble_forecast:
                    ml_adjustment = model_spread * -0.08
                    adjustment_reasons.append("Cold wind advection")
    
    # Apply the ML adjustment (cap at ±1.5°F)
    ml_adjustment = max(-1.5, min(1.5, ml_adjustment))
    
    if abs(ml_adjustment) > 0.2:
        original_ensemble = ensemble_forecast
        ensemble_forecast += ml_adjustment
        print(f"\n  🤖 ML ADJUSTMENT: {original_ensemble:.1f}°F → {ensemble_forecast:.1f}°F ({ml_adjustment:+.1f}°F)")
        for reason in adjustment_reasons:
            print(f"     Reason: {reason}")
    
    # ============ ATMOSPHERIC ANALOG MODEL (from V11, kept in V12) ============
    atmospheric_adjustment = 0.0
    atmospheric_confidence = 0.0
    
    atmospheric_model = AtmosphericAnalogModel(city_key)
    model_path = Path(ATMOSPHERIC_MODEL_PATH) / f"{city_key}_atmospheric.pkl"
    
    if model_path.exists():
        atmospheric_model.load(str(model_path))
        
        if hrrr_data:
            atmos_features = extract_atmospheric_features(hrrr_data)
            
            if atmos_features:
                atmospheric_adjustment, atmospheric_confidence = atmospheric_model.predict_adjustment(atmos_features)
                atmospheric_adjustment = max(-1.0, min(1.0, atmospheric_adjustment))
                
                if abs(atmospheric_adjustment) > 0.3:
                    print(f"\n  🌐 ATMOSPHERIC ANALOG PREDICTION:")
                    print(f"     Based on {atmospheric_model.training_samples} historical patterns")
                    print(f"     Predicted adjustment: {atmospheric_adjustment:+.1f}°F (confidence: {atmospheric_confidence:.0%})")
                    
                    weighted_adjustment = atmospheric_adjustment * ATMOSPHERIC_MODEL_WEIGHT * atmospheric_confidence
                    original_ensemble = ensemble_forecast
                    ensemble_forecast += weighted_adjustment
                    
                    print(f"     Final adjustment: {weighted_adjustment:+.2f}°F")
                    print(f"     Ensemble: {original_ensemble:.1f}°F → {ensemble_forecast:.1f}°F")
    
    # ============ MEDIUM PRIORITY ENHANCEMENTS (V12.5 NEW!) ============
    # DISABLED: This was double-counting cloud/precip adjustments already applied above
    enhancement_adjustments = []
    enhancement_explanations = []
    regime_uncertainty_mult = 1.0
    
    if False and ENHANCEMENTS_AVAILABLE:  # DISABLED to prevent double-counting
        print(f"\n  🌤️  ENHANCED ANALYSIS (V12.5 - Cloud/Precip/Regime):")
        
        try:
            # Apply all medium-priority enhancements
            adjusted_forecast, adjusted_uncertainty, enhancement_report = enhanced_forecast_adjustment(
                city["lat"], city["lon"], target_date,
                ensemble_forecast, uncertainty_std
            )
            
            # Apply forecast adjustments
            forecast_before_enhancements = ensemble_forecast
            total_enhancement_adj = enhancement_report.get("total_forecast_adjustment", 0)

            if abs(total_enhancement_adj) > 0.5:
                ensemble_forecast = adjusted_forecast
                print(f"\n     💡 Total enhancement adjustment: {total_enhancement_adj:+.1f}°F")
                print(f"        Forecast: {forecast_before_enhancements:.1f}°F → {ensemble_forecast:.1f}°F")

                # Store details for city summary
                enhancement_adjustments.append(total_enhancement_adj)
                if enhancement_report.get("cloud_adjustment"):
                    enhancement_explanations.append(f"Cloud timing: {enhancement_report['cloud_adjustment']:+.1f}°F")
                if enhancement_report.get("precip_adjustment"):
                    enhancement_explanations.append(f"Precip timing: {enhancement_report['precip_adjustment']:+.1f}°F")
            
            # Apply uncertainty adjustment from regime detection
            regime_uncertainty_mult = enhancement_report["uncertainty_multiplier"]
            if regime_uncertainty_mult != 1.0:
                uncertainty_before = uncertainty_std
                uncertainty_std = adjusted_uncertainty
                print(f"     📊 Regime uncertainty adjustment: {regime_uncertainty_mult:.2f}x")
                print(f"        Uncertainty: ±{uncertainty_before:.1f}°F → ±{uncertainty_std:.1f}°F")
        
        except Exception as e:
            print(f"     ⚠️ Enhancement analysis failed: {e}")
    
    # Major disagreement detection
    major_disagreement = model_spread > MAJOR_DISAGREEMENT_THRESHOLD
    
    print(f"\n  📊 Model Comparison:")
    for source, forecast in zip(forecast_sources, forecasts):
        print(f"     {source}: {forecast:.1f}°F")
    print(f"     Weighted Ensemble: {original_weighted_ensemble:.1f}°F")
    
    if len(forecasts) > 1:
        print(f"     Spread: {model_spread:.1f}°F → Agreement: {agreement_level.upper()}")
        
        if major_disagreement:
            print(f"     ⚠️  MAJOR DISAGREEMENT DETECTED!")
            print(f"        Possible frontal system - proceed with caution")
            if agreement_level == 'high':
                agreement_level = 'medium'
            elif agreement_level == 'medium':
                agreement_level = 'low'

    # Show adjustment summary if the final forecast differs significantly from weighted ensemble
    total_adjustment = ensemble_forecast - original_weighted_ensemble
    if abs(total_adjustment) > 0.5:
        print(f"\n  📐 ADJUSTMENT SUMMARY:")
        print(f"     Weighted Ensemble: {original_weighted_ensemble:.1f}°F")
        print(f"     Total Adjustments: {total_adjustment:+.1f}°F")
        print(f"     Final Ensemble: {ensemble_forecast:.1f}°F")
    
    print(f"\n  🎯 ENSEMBLE FORECAST: {ensemble_forecast:.1f}°F")
    rounded_forecast = nws_round(ensemble_forecast)
    print(f"     NWS-rounded (for Kalshi resolution): {rounded_forecast}°F")

    # Store city summary
    city_summary = {
        "city": city["name"],
        "ensemble_forecast": ensemble_forecast,
        "uncertainty_std": uncertainty_std,  # V12 NEW!
        "forecast_bin": None,
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
        "current_temp": current_obs["current_temp"] if current_obs else None,  # V12 NEW!
        "constraint_applied": constraint_applied,  # V12 NEW!
    }

# [CONTINUED IN PART 4 - Kalshi market fetching and contract analysis]
# Part 3 focuses on forecast generation, Part 4 handles market analysis and betting decisions.

# ============ CONTINUATION OF analyze_city FROM PART 3 ============
    
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

    # ============ ANALYZE CONTRACTS (V12: uses dynamic uncertainty!) ============
    print(f"\n  {'='*60}")
    print(f"  CONTRACT ANALYSIS")
    print(f"  {'='*60}")

    # Extract all valid bins from Kalshi markets
    valid_bins = set()
    for market in markets:
        subtitle = market.get("subtitle", "")
        numbers = re.findall(r'-?\d+', subtitle)
        
        if "to" in subtitle and len(numbers) >= 2:
            low = int(numbers[0])
            high = int(numbers[1])
            valid_bins.add((low, high))
    
    print(f"     Found {len(valid_bins)} valid temperature bins")
    
    # Calculate the forecast bin using actual Kalshi bins
    forecast_bin = get_bin_for_temp(ensemble_forecast, valid_bins)
    print(f"     📍 Predicted Bin: {forecast_bin[0]}°-{forecast_bin[1]}°F")
    
    # Update city_summary
    city_summary["forecast_bin"] = forecast_bin

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

        # Price filters
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
            
            if (low, high) not in valid_bins:
                continue
            
            # V12: Use dynamic uncertainty_std instead of CALIBRATED_FORECAST_STD!
            model_prob = calibrated_probability(low, high, ensemble_forecast, 
                                                uncertainty_std, agreement_level, calibration_tracker)
            
            rounded_forecast = nws_round(ensemble_forecast)
            is_forecast_bin = (low <= rounded_forecast <= high)
            contract_type = "range"

        elif "or below" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            # V12: Use dynamic uncertainty_std!
            model_prob = calibrated_below_probability(threshold, ensemble_forecast,
                                                       uncertainty_std, agreement_level, calibration_tracker)
            rounded_forecast = nws_round(ensemble_forecast)
            is_forecast_bin = rounded_forecast <= threshold
            contract_type = "below"

        elif "or above" in subtitle and len(numbers) >= 1:
            threshold = int(numbers[0])
            # V12: Use dynamic uncertainty_std!
            model_prob = calibrated_above_probability(threshold, ensemble_forecast,
                                                       uncertainty_std, agreement_level, calibration_tracker)
            rounded_forecast = nws_round(ensemble_forecast)
            is_forecast_bin = rounded_forecast >= threshold
            contract_type = "above"
        else:
            continue

        # Evaluate YES bet
        yes_edge = model_prob - kalshi_prob
        min_edge = get_min_edge(kalshi_prob, agreement_level)
        
        # MARKET SANITY CHECK: Don't fight heavy favorites
        # If market is >70% confident, we need MUCH more edge
        if kalshi_prob > 0.70:
            min_edge *= 2.0  # Double edge requirement
        if kalshi_prob > 0.85:
            min_edge *= 3.0  # Triple edge requirement (rarely worth it)
        
        if yes_edge > min_edge:
            edge_ratio = yes_edge / min_edge
            our_prob_win = model_prob
            
            # ========== CRITICAL PROBABILITY CAPS (V12.6 FIX!) ==========
            # Apply multiple layers of safety caps to prevent overconfidence disasters
            
            # Layer 1: ABSOLUTE HARD CAP - Never exceed 95%
            our_prob_win = min(our_prob_win, 0.95)
            
            # Layer 2: UNCERTAINTY-BASED CAP
            # High uncertainty = lower maximum confidence
            if uncertainty_std > 8.0:
                # Extreme uncertainty (frontal systems, major disagreement)
                our_prob_win = min(our_prob_win, 0.85)
            elif uncertainty_std > 6.0:
                # High uncertainty
                our_prob_win = min(our_prob_win, 0.88)
            elif uncertainty_std > 5.0:
                # Medium-high uncertainty
                our_prob_win = min(our_prob_win, 0.90)
            
            # Layer 3: MAJOR DISAGREEMENT CAP
            # When models disagree significantly, cap confidence
            if major_disagreement:
                our_prob_win = min(our_prob_win, 0.85)
            
            # Layer 4: AGREEMENT LEVEL CAP
            # Low agreement between models = lower max confidence
            if agreement_level == 'low':
                our_prob_win = min(our_prob_win, 0.88)
            elif agreement_level == 'medium':
                our_prob_win = min(our_prob_win, 0.92)
            
            # Recalculate Kelly with capped probability
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
                    "uncertainty_std": uncertainty_std,  # NEW: Track uncertainty
                    "low": low if contract_type == "range" else None,
                    "high": high if contract_type == "range" else None,
                    "threshold": threshold if contract_type != "range" else None,
                })

        # Evaluate NO bet (don't bet NO on forecast bin)
        if not is_forecast_bin:
            no_prob = 1 - model_prob
            no_market = 1 - kalshi_prob
            no_edge = no_prob - no_market
            no_min_edge = get_min_edge(no_market, agreement_level)
            
            # Skip NO range bets in sweet spot
            no_bucket = get_price_bucket(no_market)
            skip_no_range = (contract_type == "range" and no_bucket == "sweet_spot")
            
            # MARKET SANITY CHECK: Don't bet NO when market is very bearish on this outcome
            # If NO market is >70% (YES market <30%), we need more edge
            if no_market > 0.70:
                no_min_edge *= 2.0  # Double edge requirement
            if no_market > 0.85:
                no_min_edge *= 3.0  # Triple edge requirement
            
            if no_edge > no_min_edge and not skip_no_range:
                edge_ratio = no_edge / no_min_edge
                our_prob_win = no_prob
                
                # ========== CRITICAL PROBABILITY CAPS (V12.6 FIX!) ==========
                # Apply multiple layers of safety caps to prevent overconfidence disasters
                # THIS WAS THE BUG: NO bets weren't being capped!
                
                # Layer 1: ABSOLUTE HARD CAP - Never exceed 95%
                our_prob_win = min(our_prob_win, 0.95)
                
                # Layer 2: UNCERTAINTY-BASED CAP
                # High uncertainty = lower maximum confidence
                if uncertainty_std > 8.0:
                    # Extreme uncertainty (frontal systems, major disagreement)
                    our_prob_win = min(our_prob_win, 0.85)
                elif uncertainty_std > 6.0:
                    # High uncertainty
                    our_prob_win = min(our_prob_win, 0.88)
                elif uncertainty_std > 5.0:
                    # Medium-high uncertainty
                    our_prob_win = min(our_prob_win, 0.90)
                
                # Layer 3: MAJOR DISAGREEMENT CAP
                # When models disagree significantly, cap confidence
                if major_disagreement:
                    our_prob_win = min(our_prob_win, 0.85)
                
                # Layer 4: AGREEMENT LEVEL CAP
                # Low agreement between models = lower max confidence
                if agreement_level == 'low':
                    our_prob_win = min(our_prob_win, 0.88)
                elif agreement_level == 'medium':
                    our_prob_win = min(our_prob_win, 0.92)
                
                # Recalculate Kelly with capped probability
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
                        "uncertainty_std": uncertainty_std,  # NEW: Track uncertainty
                        "low": low if contract_type == "range" else None,
                        "high": high if contract_type == "range" else None,
                        "threshold": threshold if contract_type != "range" else None,
                    })

    print(f"\n  Found {len(qualifying_bets)} qualifying bets")
    print(f"  Filtered out: {skipped_cheap} cheap, {skipped_expensive} expensive")

    return qualifying_bets, city_summary


# ============ SMART BET SELECTION ============

def smart_select_bets(all_bets, bankroll=STARTING_BANKROLL):
    """
    Apply smart bet selection logic across all cities.
    
    Rules:
    1. Different cities = uncorrelated → bet freely on best bet from each
    2. Same city = correlated → only stack if SUPER confident (edge_ratio >= 2.0x)
    3. Cap total bets per day
    4. CRITICAL: Ensure total wager never exceeds bankroll
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
    
    # CRITICAL: Ensure total wager doesn't exceed bankroll
    # Target 95% of bankroll to leave buffer for rounding
    max_wager = bankroll * 0.95
    total_wager = sum(bet["bet_size"] for bet in selected_bets)
    
    if total_wager > max_wager:
        # Calculate scaling factor to hit 95% of bankroll
        scale_factor = max_wager / total_wager
        
        # Scale down all bet sizes proportionally
        for bet in selected_bets:
            original_bet_size = bet["bet_size"]
            bet["bet_size"] = original_bet_size * scale_factor
            
            # Update Kelly info to reflect scaling
            bet["kelly_info"]["scaled_down"] = True
            bet["kelly_info"]["scale_factor"] = scale_factor
            bet["kelly_info"]["original_bet_size"] = original_bet_size
            
            # Remove bets that fall below minimum after scaling
            if bet["bet_size"] < MIN_BET_SIZE:
                bet["bet_size"] = 0
        
        # Filter out bets that are now below minimum
        selected_bets = [bet for bet in selected_bets if bet["bet_size"] >= MIN_BET_SIZE]
    
    return selected_bets


def print_recommendations(selected_bets, all_bets, city_summaries, bankroll, target_date):
    """Print the final betting recommendations."""

    target_date_str = target_date.strftime("%A, %B %d, %Y")
    
    print(f"\n")
    print("=" * 80)
    print("📊 FORECAST ANALYSIS BY CITY (V12 ENHANCED)")
    print("=" * 80)
    
    # City forecasts
    for summary in city_summaries:
        if summary:
            print(f"\n🏙️  {summary['city'].upper()}")
            print("-" * 80)
            
            # Show current temp if available (V12!)
            if summary.get('current_temp'):
                print(f"   Current Temperature: {summary['current_temp']:.1f}°F ⏰")
            
            # Model forecasts
            print(f"   Model Forecasts:")
            sources_info = []
            if summary.get("hrrr_forecast"):
                sources_info.append(f"HRRR: {summary['hrrr_forecast']:.1f}°F")
            if summary.get("open_meteo_forecast"):
                sources_info.append(f"Open-Meteo: {summary['open_meteo_forecast']:.1f}°F")
            if summary.get("nws_forecast"):
                sources_info.append(f"NWS: {summary['nws_forecast']:.1f}°F")
            
            for info in sources_info:
                print(f"      • {info}")
            
            # Ensemble and uncertainty (V12!)
            print(f"   Ensemble Forecast: {summary['ensemble_forecast']:.1f}°F")
            print(f"   Forecast Uncertainty: ±{summary.get('uncertainty_std', 2.8):.1f}°F (V12 Dynamic!)")
            
            # Model agreement
            if summary.get("model_spread") is not None and len(sources_info) > 1:
                agreement_emoji = "✅" if summary["agreement_level"] == "high" else "📊" if summary["agreement_level"] == "medium" else "⚠️"
                print(f"   Model Spread: {summary['model_spread']:.1f}°F ({summary['agreement_level'].upper()}) {agreement_emoji}")
            
            # Observation constraint (V12!)
            if summary.get('constraint_applied'):
                print(f"   🔒 Observation constraint applied (V12!)")
            
            # ML adjustments
            if summary.get("ml_adjustment") and abs(summary["ml_adjustment"]) > 0.2:
                print(f"   ML Adjustment: {summary['ml_adjustment']:+.1f}°F")
                if summary.get("adjustment_reasons"):
                    for reason in summary["adjustment_reasons"]:
                        print(f"      └─ {reason}")

    # Recommendations
    print(f"\n")
    print("=" * 80)
    print("💡 BETTING RECOMMENDATIONS")
    print("=" * 80)
    
    if not selected_bets:
        print("\n   🚫 NO BETS RECOMMENDED")
        print("      • No contracts with sufficient edge after filtering")
        return
    
    # Check if any bets were scaled down
    any_scaled = any(bet.get("kelly_info", {}).get("scaled_down", False) for bet in selected_bets)
    if any_scaled:
        scale_factor = selected_bets[0]["kelly_info"].get("scale_factor", 1.0)
        print(f"\n   ⚠️  BANKROLL CONSTRAINT APPLIED")
        print(f"      • Kelly suggested bets would exceed bankroll")
        print(f"      • All bet sizes scaled down by {scale_factor:.1%} to fit ${bankroll:.2f} bankroll")
        print()
    
    print(f"\n   ✅ Found {len(selected_bets)} recommended bet(s)")
    print()
    
    total_wager = 0
    total_potential = 0
    
    for i, bet in enumerate(selected_bets, 1):
        edge_ratio = bet["edge_ratio"]
        our_prob = bet["our_prob_win"]
        
        # Determine label based on edge quality (not outcome certainty!)
        if edge_ratio >= SUPER_CONFIDENT_EDGE_RATIO:
            if our_prob >= 0.65:
                confidence_level = "🔥 EXCEPTIONAL VALUE + HIGH WIN PROBABILITY"
            else:
                confidence_level = "💎 EXCEPTIONAL VALUE (close outcome, great price)"
        elif edge_ratio >= SAME_CITY_MULTI_BET_THRESHOLD:
            if our_prob >= 0.60:
                confidence_level = "✅ STRONG VALUE + GOOD WIN PROBABILITY"
            else:
                confidence_level = "📊 STRONG VALUE (coin flip-ish, but mispriced)"
        else:
            if our_prob >= 0.60:
                confidence_level = "✓ FAIR VALUE + FAVORABLE ODDS"
            else:
                confidence_level = "⚖️ FAIR VALUE (marginal edge)"
        
        bucket_emoji = "🎯" if bet["price_bucket"] == "sweet_spot" else "🔥" if bet["price_bucket"] == "mid_range" else "💎"
        is_forecast_bin = " ⭐ MODEL'S PRIMARY BIN" if bet.get("is_forecast_bin") else ""
        
        odds = bet["kelly_info"].get("odds", 0)
        potential_profit = bet["bet_size"] * odds
        roi_pct = (odds * 100) if odds > 0 else 0
        
        print("   " + "─" * 74)
        print(f"   BET #{i}: {bet['city'].upper()} - {confidence_level}")
        print("   " + "─" * 74)
        print(f"   📋 Contract: {bet['subtitle']}{is_forecast_bin}")
        print(f"   💵 Side: {bet['side'].upper()} at {bet['bet_price']*100:.0f}¢ {bucket_emoji}")
        print()
        print(f"   📊 THE EDGE:")
        print(f"      Your probability: {bet['our_prob_win']*100:.1f}%  |  Market: {bet['bet_price']*100:.1f}%")
        print(f"      Edge: {bet['edge']*100:+.1f}%  ({edge_ratio:.1f}x required minimum)")
        
        # NEW: Show uncertainty and caps applied (V12.6!)
        uncertainty = bet.get('uncertainty_std', 0)
        if uncertainty > 0:
            print(f"      Forecast Uncertainty: ±{uncertainty:.1f}°F")
            
            # Warn about high uncertainty
            if uncertainty > 8.0:
                print(f"      ⚠️  EXTREME UNCERTAINTY - Confidence capped at 85%")
            elif uncertainty > 6.0:
                print(f"      ⚠️  HIGH UNCERTAINTY - Confidence capped at 88%")
            elif uncertainty > 5.0:
                print(f"      ⚠️  MODERATE UNCERTAINTY - Confidence capped at 90%")
        
        # Warn about major disagreement
        if bet.get('major_disagreement'):
            print(f"      ⚠️  MAJOR MODEL DISAGREEMENT - Confidence capped at 85%")
        
        # Clarify what the label means
        if our_prob < 0.55:
            print(f"      ⚠️  Close to 50/50 outcome - value is in the PRICE, not certainty")
        elif our_prob >= 0.70:
            print(f"      ✅ Likely to win AND great price - double advantage")
        
        # Show calibration boost if applied (V12 OPTIMIZED!)
        if bet.get('raw_prob') and abs(bet['our_prob_win'] - bet['raw_prob']) > 0.02:
            boost = (bet['our_prob_win'] - bet['raw_prob']) * 100
            print(f"      📈 Calibration boost: +{boost:.1f}% (based on your 46 bet history)")
        
        print()
        print(f"   💰 POSITION SIZING:")
        print(f"      Kelly %: {bet['kelly_info'].get('fractional_kelly_pct', 0):.1f}% of bankroll")
        
        # Show if bet was scaled down
        if bet['kelly_info'].get('scaled_down'):
            original = bet['kelly_info'].get('original_bet_size', 0)
            print(f"      Kelly suggested: ${original:.2f}")
            print(f"      ➜  BET (scaled to fit bankroll): ${bet['bet_size']:.2f}")
        else:
            print(f"      ➜  BET: ${bet['bet_size']:.2f}")
        
        print(f"      ➜  If you win: ${potential_profit:.2f} profit ({roi_pct:.0f}% ROI)")
        print()
        print(f"   🌡️  MODEL FORECAST: {bet['ensemble_forecast']:.1f}°F")
        print()
        
        total_wager += bet["bet_size"]
        total_potential += potential_profit
    
    # Summary
    print("   " + "=" * 74)
    print("   📊 PORTFOLIO SUMMARY")
    print("   " + "=" * 74)
    print(f"   Total bets: {len(selected_bets)}")
    print(f"   Total wager: ${total_wager:.2f} ({100*total_wager/bankroll:.1f}% of bankroll)")
    print(f"   Total potential profit: ${total_potential:.2f}")
    
    # Sanity check with small epsilon for floating point precision
    epsilon = 0.01  # 1 cent tolerance
    if total_wager > bankroll + epsilon:
        print(f"   ⚠️  WARNING: Total wager exceeds bankroll by ${total_wager - bankroll:.2f}!")
        print(f"   ⚠️  This should not happen - please report this bug")
    elif total_wager > bankroll * 0.90:
        print(f"   ℹ️  Note: Using {100*total_wager/bankroll:.1f}% of available bankroll")
    
    print("   " + "=" * 74)


# ============ MAIN (V12 ENHANCED!) ============

def main():
    global STARTING_BANKROLL, KELLY_FRACTION, selected_cities

    parser = argparse.ArgumentParser(description='Kalshi Weather Betting Model v12 (Enhanced)')
    parser.add_argument('--kelly', type=float, default=KELLY_FRACTION,
                        help=f'Kelly fraction (default: {KELLY_FRACTION})')
    parser.add_argument('--bankroll', type=float, default=STARTING_BANKROLL,
                        help=f'Starting bankroll in dollars (default: {STARTING_BANKROLL})')
    parser.add_argument('--today', action='store_true',
                        help="Look at today's markets")
    parser.add_argument('--tomorrow', action='store_true',
                        help="Force looking at tomorrow's markets")
    parser.add_argument('--date', type=str, default=None,
                        help="Specific date (YYYY-MM-DD)")
    parser.add_argument('--train', action='store_true',
                        help="Train atmospheric models")
    parser.add_argument('--cities', type=str, default=None,
                        help="Comma-separated cities (default: all)")
    
    # V12 NEW ARGUMENTS!
    parser.add_argument('--calibration-report', action='store_true',
                        help="Show calibration report and exit")
    parser.add_argument('--update-outcome', nargs=3, 
                        metavar=('DATE', 'CITY', 'ACTUAL_TEMP'),
                        help="Update calibration with actual outcome")
    
    args = parser.parse_args()

    KELLY_FRACTION = args.kelly
    STARTING_BANKROLL = args.bankroll

    if args.cities:
        cities_to_analyze = [c.strip().lower() for c in args.cities.split(',')]
        cities_to_analyze = [c for c in cities_to_analyze if c in cities]
        if cities_to_analyze:
            selected_cities = cities_to_analyze

    # Initialize calibration tracker (V12!)
    calibration_tracker = CalibrationTracker()

    # Handle calibration commands (V12!)
    if args.calibration_report:
        print(calibration_tracker.get_calibration_report())
        return
    
    if args.update_outcome:
        date_str, city_name, actual_temp = args.update_outcome
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        actual_temp = float(actual_temp)
        calibration_tracker.update_outcome(date, city_name.lower(), actual_temp)
        print(f"\n✅ Updated outcomes for {city_name} on {date_str}")
        print(calibration_tracker.get_calibration_report())
        return

    # Training mode
    if args.train:
        print("=" * 70)
        print("ATMOSPHERIC MODEL TRAINING MODE")
        print("=" * 70)
        
        for city_key in selected_cities:
            city_config = cities[city_key]
            print(f"\n🏙️ Training model for {city_config['name']}...")
            
            training_files = glob.glob(f"training_data_{city_key}_*.csv")
            
            if not training_files:
                print(f"   ❌ No training data found for {city_key}")
                continue
            
            training_file = sorted(training_files)[-1]
            print(f"   📂 Loading: {training_file}")
            
            try:
                training_data = pd.read_csv(training_file)
                print(f"   📊 Loaded {len(training_data)} samples")
                
                atmospheric_model = AtmosphericAnalogModel(city_key)
                success = atmospheric_model.train(training_data, verbose=True)
                
                if success:
                    model_path = Path(ATMOSPHERIC_MODEL_PATH) / f"{city_key}_atmospheric.pkl"
                    atmospheric_model.save(str(model_path))
                    print(f"   ✅ Model saved to {model_path}")
                else:
                    print(f"   ❌ Training failed for {city_key}")
                    
            except Exception as e:
                print(f"   ❌ Error training {city_key}: {e}")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        return

    # Determine target date
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        eastern_today = get_eastern_now().date()
        is_today = (target_date == eastern_today)
    elif args.today:
        target_date, is_today = get_target_date(force_today=True)
    elif args.tomorrow:
        eastern_now = get_eastern_now()
        target_date = eastern_now.date() + timedelta(days=1)
        is_today = False
    else:
        target_date, is_today = get_target_date()

    print_header(target_date, is_today)

    bankroll = STARTING_BANKROLL
    all_bets = []
    city_summaries = []

    # Analyze each city (V12: passes calibration_tracker!)
    for city_key in selected_cities:
        city_bets, city_summary = analyze_city(city_key, bankroll, target_date, calibration_tracker)
        all_bets.extend(city_bets)
        city_summaries.append(city_summary)

    # Smart bet selection (V12: Now ensures total wager ≤ bankroll!)
    selected_bets = smart_select_bets(all_bets, bankroll)

    # Print recommendations
    print_recommendations(selected_bets, all_bets, city_summaries, bankroll, target_date)

    # Record predictions to calibration tracker (V12!)
    if calibration_tracker and selected_bets:
        for bet in selected_bets:
            calibration_tracker.record_prediction(
                date=target_date,
                city=bet["city"],
                our_prob=bet["our_prob_win"],
                market_price=bet["bet_price"],
                contract_details={
                    "subtitle": bet.get("subtitle"),
                    "low": bet.get("low"),
                    "high": bet.get("high"),
                    "threshold": bet.get("threshold"),
                    "side": bet.get("side")
                }
            )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE (v12 ENHANCED)")
    print(f"{'='*70}")
    print()


if __name__ == "__main__":
    main()

# ============ END OF PART 4/4 - V9.PY COMPLETE ============