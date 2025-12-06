"""
KALSHI WEATHER MODEL BACKTESTER - IMPROVED v4 (DOCUMENTED VERSION)
===================================================================

╔══════════════════════════════════════════════════════════════════════════════╗
║                           HOW THIS BACKTESTER WORKS                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERVIEW:
---------
This script tests a weather betting strategy on historical Kalshi temperature 
contracts. It simulates what would have happened if you followed specific rules 
for placing bets on high-temperature contracts.

THE CORE IDEA:
--------------
Kalshi offers contracts like "Will the high temperature in Chicago be 75-76°F?"
Each contract has a market price (e.g., 25¢) which implies the market thinks 
there's a 25% chance of that outcome. If YOUR forecast model says there's actually
a 40% chance, you have a 15% "edge" - that's a potentially profitable bet.

WHAT IS "EDGE"?
---------------
Edge = Your Probability - Market Price

Example:
  - Market price: 25¢ (implies 25% chance)
  - Your model says: 40% chance
  - Your edge: 40% - 25% = 15%

If your model is accurate over many bets, this edge turns into profit.

HOW BETTING WORKS:
------------------
For a YES bet at price P:
  - You pay: $BET_SIZE
  - If you WIN: You get back $BET_SIZE / P  (profit = $BET_SIZE * (1/P - 1))
  - If you LOSE: You lose $BET_SIZE

Example YES bet:
  - Bet $10 on YES at 25¢ (0.25)
  - WIN: Get back $10 / 0.25 = $40, profit = $30
  - LOSE: Lose $10

For a NO bet at price P (market shows P for YES):
  - NO price = 1 - P
  - You pay: $BET_SIZE
  - If you WIN: You get back $BET_SIZE / (1-P)
  - If you LOSE: You lose $BET_SIZE

Example NO bet:
  - Market shows YES at 75¢, so NO costs 25¢
  - Bet $10 on NO at 25¢
  - WIN: Get back $10 / 0.25 = $40, profit = $30
  - LOSE: Lose $10

KEY STRATEGY RULES:
-------------------
1. NEVER bet on contracts below 15¢ (cheap contract trap - they rarely hit)
2. NEVER bet on contracts above 90¢ (expensive, low payout)
3. Focus on "sweet spot" 15-50¢ (best historical ROI)
4. Require minimum 12% edge before betting
5. Only bet on ONE contract per day (the best opportunity)

WHY AVOID CHEAP CONTRACTS?
--------------------------
Historical analysis showed:
  - 1-5¢ contracts: -97% ROI (disaster!)
  - 6-14¢ contracts: Marginal
  - 15-50¢ contracts: +30-55% ROI (sweet spot)

Cheap contracts look appealing (bet $1 to win $20!) but they almost never hit.

THE PROBABILITY MODEL:
----------------------
We use a Gaussian (normal) distribution to model forecast uncertainty.

Given a forecast of F degrees:
  - The actual temperature follows a normal distribution centered on F
  - Standard deviation of ~2.8°F (typical day-ahead forecast error)
  - This lets us calculate P(actual temp falls in any range)

Example:
  - Forecast: 75°F
  - Contract: "Will temp be 73-74°F?"
  - Model calculates: ~18% probability based on the bell curve

╔══════════════════════════════════════════════════════════════════════════════╗
║                              CONFIGURATION                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import os

# ============ CONFIGURATION ============
# Betting mode: "flat" or "kelly"
BETTING_MODE = "kelly"

# Flat betting settings
FLAT_BET_SIZE = 10  # $10 per bet (used if BETTING_MODE = "flat")

# Kelly Criterion settings
STARTING_BANKROLL = 40    # Starting bankroll for Kelly sizing ($20-$40 range)
KELLY_FRACTION = 0.75     # Use 1/2 Kelly (balanced risk/reward)
MIN_BET_SIZE = 0.50       # Minimum bet size (don't bet less than 50¢)
MAX_BET_FRACTION = 0.15   # Never bet more than 15% of bankroll on one bet on one bet

# Price filters - which contracts to even consider
MIN_CONTRACT_PRICE = 0.15  # Never bet on contracts below 15¢
MAX_CONTRACT_PRICE = 0.90  # Never bet on contracts above 90¢

# The "sweet spot" where we've seen the best returns historically
SWEET_SPOT_LOW = 0.15      # 15¢
SWEET_SPOT_HIGH = 0.50     # 50¢

# Minimum edge required before placing a bet (by price bucket)
# Higher edge requirements = fewer but more confident bets
EDGE_REQUIREMENTS = {
    "sweet_spot": 0.12,    # Need 12% edge for 15-50¢ contracts
    "high_price": 0.12,    # Need 12% edge for 50-90¢ contracts
}

# How uncertain weather forecasts typically are
# Higher = more uncertainty = wider probability spreads
FORECAST_ERROR_STD = 2.8  # Degrees Fahrenheit

# ============ BETTING TIMING ASSUMPTION ============
# This is CRITICAL for understanding what this backtest simulates.
#
# The backtest assumes you place bets the EVENING BEFORE the target date.
# 
# Timeline example for a Jan 15th high temperature contract:
#   - Jan 14th, ~6-8 PM: You check the forecast and Kalshi prices
#   - Jan 14th, ~6-8 PM: You place your bet based on current prices
#   - Jan 15th: The actual high temperature is recorded
#   - Jan 15th, evening: Contract settles, you win or lose
#
# WHY EVENING BEFORE?
#   - Day-ahead forecasts (like HRRR) are most accurate 12-24 hours out
#   - Prices are usually stable by evening (less volatility)
#   - Gives you time to analyze without rushing
#   - Markets have good liquidity
#
# IMPORTANT CAVEATS:
#   1. The "last_price_cents" in the data may not exactly match what you'd
#      get at 6 PM the day before - prices fluctuate
#   2. Real trading has bid/ask spreads (you pay slightly more than mid-price)
#   3. This backtest doesn't account for slippage or spread costs
#
# For more aggressive timing (same-day morning bets), forecasts are more
# accurate but prices may have already moved to reflect that.

ASSUMED_BET_TIMING = "evening_before"  # When we assume bets are placed
BET_TIMING_DESCRIPTION = "Evening before (6-8 PM day prior)"

# ============ SINGLE VS MULTIPLE BETS ============
# Smart bet selection strategy:
#
# DIFFERENT CITIES = UNCORRELATED
#   Bets in different cities are independent - Chicago weather doesn't
#   affect Miami weather. So we can safely bet on multiple cities.
#
# SAME CITY = CORRELATED (be careful!)
#   Multiple bets in the same city are highly correlated. If your forecast
#   is wrong, you lose ALL of them. Only stack same-city bets if you're
#   VERY confident (high edge ratio).
#
# THE STRATEGY:
#   1. Always take the best bet in each city (if it meets edge threshold)
#   2. For additional same-city bets, require MUCH higher confidence
#   3. Cap same-city bets to avoid catastrophic losses

# Confidence thresholds (edge_ratio = edge / min_required_edge)
SAME_CITY_MULTI_BET_THRESHOLD = 2.0  # Need 2x the minimum edge to add more same-city bets
MAX_BETS_PER_CITY = 2                 # Never more than 2 bets in same city per day
MAX_TOTAL_BETS_PER_DAY = 5            # Cap total daily bets across all cities

# What counts as "super confident" for same-city stacking
SUPER_CONFIDENT_EDGE_RATIO = 2.5     # 2.5x minimum edge = very confident


# ============ HELPER FUNCTIONS ============

def get_bin(temp):
    """
    Get the Kalshi temperature bin for a given temperature.
    
    Kalshi uses 2-degree bins with ODD lower bounds:
    ..., 27-28, 29-30, 31-32, 33-34, ...
    
    Examples:
        27.5°F → bin (27, 28)
        29.0°F → bin (29, 30)
        29.9°F → bin (29, 30)
        75.3°F → bin (75, 76)
    """
    temp_floor = int(np.floor(temp))
    
    # If floor is odd, that's our lower bound
    # If floor is even, the bin started at floor - 1
    if temp_floor % 2 == 1:  # odd
        lower = temp_floor
    else:  # even
        lower = temp_floor - 1
    
    return (lower, lower + 1)


def get_price_bucket(price):
    """
    Categorize a contract price into buckets for different edge requirements.
    
    Sweet spot (15-50¢): Where we see the best historical returns
    High price (50-90¢): Still tradeable but less attractive
    """
    if price <= SWEET_SPOT_HIGH:
        return "sweet_spot"
    else:
        return "high_price"


def get_min_edge(price):
    """Get the minimum edge required to place a bet at this price."""
    bucket = get_price_bucket(price)
    return EDGE_REQUIREMENTS[bucket]


# ============ KELLY CRITERION ============

def calculate_kelly_bet(bankroll, our_prob, bet_price, kelly_fraction=KELLY_FRACTION):
    """
    Calculate optimal bet size using Kelly Criterion.
    
    The Kelly Criterion formula for binary bets:
        f* = (p * b - q) / b
    
    Where:
        f* = fraction of bankroll to bet
        p = probability of winning (our estimate)
        q = probability of losing (1 - p)
        b = odds received on the bet (payout per $1 wagered, minus the $1)
    
    For Kalshi:
        If you bet $1 at price P and win, you get $1/P back
        So b = (1/P) - 1 = (1-P)/P
    
    Args:
        bankroll: Current bankroll
        our_prob: Our estimated probability of winning
        bet_price: Price we pay for the contract (0 to 1)
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
    
    Returns:
        bet_size: Dollar amount to bet
        kelly_info: Dict with calculation details
    """
    # Odds received: if we bet $1 at price P and win, we get 1/P back
    # So net profit per $1 wagered = (1/P) - 1
    b = (1 / bet_price) - 1  # Net odds (profit per dollar if win)
    
    p = our_prob  # Probability of winning
    q = 1 - p     # Probability of losing
    
    # Full Kelly formula: f* = (p*b - q) / b = p - q/b
    full_kelly = (p * b - q) / b
    
    # Apply fractional Kelly for reduced variance
    fractional_kelly = full_kelly * kelly_fraction
    
    # Clamp to reasonable bounds
    fractional_kelly = max(0, fractional_kelly)  # Never negative
    fractional_kelly = min(fractional_kelly, MAX_BET_FRACTION)  # Cap at max fraction
    
    # Calculate actual bet size
    bet_size = bankroll * fractional_kelly
    
    # Apply minimum bet size
    if bet_size < MIN_BET_SIZE:
        bet_size = 0  # Don't bet if below minimum
    
    kelly_info = {
        "full_kelly_fraction": full_kelly,
        "fractional_kelly": fractional_kelly,
        "bankroll": bankroll,
        "our_prob": our_prob,
        "bet_price": bet_price,
        "odds": b,
    }
    
    return bet_size, kelly_info


def explain_kelly_criterion():
    """Print an explanation of Kelly Criterion."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           KELLY CRITERION EXPLAINED                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IS KELLY CRITERION?
------------------------
Kelly Criterion is a formula that tells you the optimal fraction of your 
bankroll to bet, given your edge and the odds. It maximizes long-term growth
while avoiding ruin.

THE FORMULA:
------------
For a bet with probability p of winning and odds b (profit per $1 if you win):

    Kelly % = (p × b - q) / b

Where q = 1 - p (probability of losing)

EXAMPLE:
--------
  Your probability (p): 40%
  Bet price: 25¢ (odds b = 3, since you profit $3 per $1 bet if you win)
  
  Kelly % = (0.40 × 3 - 0.60) / 3 = (1.20 - 0.60) / 3 = 0.20 = 20%
  
  With $1000 bankroll: Bet $200 (full Kelly)

WHY FRACTIONAL KELLY?
---------------------
Full Kelly is mathematically optimal BUT:
  • Assumes your probability estimates are perfect (they're not)
  • Creates huge volatility (50%+ drawdowns are common)
  • One bad estimate can devastate your bankroll

Using 1/2 Kelly (50% of the full Kelly bet):
  • Reduces variance significantly
  • Sacrifices ~25% of expected growth rate
  • More forgiving of estimation errors
  • Good balance of growth vs risk for smaller bankrolls

THIS BACKTEST USES:
-------------------
  • Starting bankroll: ${:,.0f}
  • Kelly fraction: {:.0%} (half Kelly)
  • Min bet: ${:.2f}
  • Max bet: {:.0%} of bankroll
""".format(STARTING_BANKROLL, KELLY_FRACTION, MIN_BET_SIZE, MAX_BET_FRACTION))


def generate_daily_recommendation(bets_by_city, bankroll):
    """
    Generate a human-readable recommendation message for a day's bets.
    
    This is the format you'd see in a live trading scenario:
    - Multiple cities? Bet on all (uncorrelated)
    - Single city with super confident bets? Stack them
    - Single city, normal confidence? Just take the best one
    
    Args:
        bets_by_city: Dict of {city: [list of qualifying bets]}
        bankroll: Current bankroll for Kelly sizing
    
    Returns:
        recommendation_text: Human-readable recommendation
    """
    all_bets = []
    
    for city, city_bets in bets_by_city.items():
        if not city_bets:
            continue
            
        # Sort by edge ratio
        city_bets = sorted(city_bets, key=lambda x: -x["edge_ratio"])
        
        # Always include best bet from each city
        best_bet = city_bets[0]
        best_bet["city"] = city
        best_bet["recommendation_type"] = "best_in_city"
        all_bets.append(best_bet)
        
        # Add super confident stacks from same city
        for bet in city_bets[1:MAX_BETS_PER_CITY]:
            if bet["edge_ratio"] >= SAME_CITY_MULTI_BET_THRESHOLD:
                bet["city"] = city
                bet["recommendation_type"] = "super_confident_stack"
                all_bets.append(bet)
    
    if not all_bets:
        return "📊 No good bets found today. Sitting this one out.\n"
    
    # Sort all bets by edge ratio for display
    all_bets = sorted(all_bets, key=lambda x: -x["edge_ratio"])
    
    # Count cities
    cities_with_bets = set(b["city"] for b in all_bets)
    num_cities = len(cities_with_bets)
    
    # Build recommendation message
    lines = []
    lines.append("=" * 60)
    lines.append("🎯 TODAY'S BETTING RECOMMENDATIONS")
    lines.append("=" * 60)
    
    if num_cities > 1:
        lines.append(f"\n✅ Found good bets in {num_cities} DIFFERENT CITIES (uncorrelated):")
        lines.append("   → Safe to bet on all of these!\n")
    elif len(all_bets) > 1:
        lines.append(f"\n🔥 Found {len(all_bets)} bets in {list(cities_with_bets)[0]} - SUPER CONFIDENT:")
        lines.append("   → Stacking same-city bets because edge is very high!\n")
    else:
        lines.append(f"\n⭐ Found 1 great bet in {list(cities_with_bets)[0]}:")
        lines.append("   → This is your best opportunity today.\n")
    
    for i, bet in enumerate(all_bets, 1):
        # Calculate Kelly bet size
        our_prob_win = bet["our_prob"] if bet["side"] == "YES" else (1 - bet["our_prob"])
        bet_size, _ = calculate_kelly_bet(bankroll, our_prob_win, bet["bet_price"], KELLY_FRACTION)
        
        confidence = ""
        if bet["edge_ratio"] >= SUPER_CONFIDENT_EDGE_RATIO:
            confidence = "🔥 SUPER CONFIDENT"
        elif bet["edge_ratio"] >= SAME_CITY_MULTI_BET_THRESHOLD:
            confidence = "✓ High confidence"
        else:
            confidence = "• Good edge"
        
        lines.append(f"   BET #{i}: {bet['city']}")
        lines.append(f"   {confidence}")
        lines.append(f"   Contract: {bet.get('contract_name', 'Temperature range')}")
        lines.append(f"   Side: {bet['side']} at {bet['bet_price']*100:.0f}¢")
        lines.append(f"   Your probability: {our_prob_win*100:.1f}%")
        lines.append(f"   Edge: {bet['edge']*100:+.1f}% ({bet['edge_ratio']:.1f}x minimum)")
        lines.append(f"   💰 RECOMMENDED BET: ${bet_size:.2f}")
        lines.append("")
    
    # Summary
    total_bet = sum(
        calculate_kelly_bet(
            bankroll,
            b["our_prob"] if b["side"] == "YES" else (1 - b["our_prob"]),
            b["bet_price"],
            KELLY_FRACTION
        )[0] for b in all_bets
    )
    
    lines.append("-" * 60)
    lines.append(f"   Total to wager: ${total_bet:.2f} ({100*total_bet/bankroll:.1f}% of bankroll)")
    lines.append(f"   Current bankroll: ${bankroll:.2f}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def calculate_bet_outcome(bet_size, price, won):
    """
    Calculate the profit/loss from a bet.
    
    Args:
        bet_size: Amount wagered (e.g., $10)
        price: Price paid per contract (e.g., 0.25 for 25¢)
        won: True if the bet won, False if it lost
    
    Returns:
        profit: Dollar profit (positive) or loss (negative)
        payout: Total amount returned if won
    """
    if won:
        payout = bet_size / price  # Total returned
        profit = payout - bet_size  # Profit after subtracting stake
    else:
        payout = 0
        profit = -bet_size
    
    return profit, payout


def format_bet_explanation(bet_size, price, side, won, actual_temp, forecast):
    """
    Generate a human-readable explanation of a bet.
    """
    if won:
        profit, payout = calculate_bet_outcome(bet_size, price, True)
        return f"""
    💰 BET DETAILS:
       You bet ${bet_size:.2f} on {side} at {price*100:.0f}¢
       Your forecast: {forecast:.1f}°F | Actual: {actual_temp:.1f}°F
       
       ✅ WON! 
       Payout: ${payout:.2f} (${bet_size:.2f} stake + ${profit:.2f} profit)
       Profit: +${profit:.2f}
"""
    else:
        return f"""
    💰 BET DETAILS:
       You bet ${bet_size:.2f} on {side} at {price*100:.0f}¢
       Your forecast: {forecast:.1f}°F | Actual: {actual_temp:.1f}°F
       
       ❌ LOST
       Loss: -${bet_size:.2f}
"""


# ============ PROBABILITY MODEL ============

def model_probability(forecast, contract_type, bound1, bound2, error_std):
    """
    Calculate the probability that a contract will resolve YES.
    
    Uses a Gaussian distribution centered on the forecast with standard
    deviation representing typical forecast error.
    
    Args:
        forecast: Our predicted temperature
        contract_type: "range" (e.g., 73-74°F), "below" (<73°F), or "above" (>76°F)
        bound1: Lower bound (or threshold for below/above)
        bound2: Upper bound (for range contracts)
        error_std: Standard deviation of forecast errors
    
    Returns:
        Probability between 0.01 and 0.99
    """
    if contract_type == "range":
        # P(actual in [bound1, bound2])
        # Uses cumulative distribution function (CDF)
        prob = stats.norm.cdf(bound2, forecast, error_std) - \
               stats.norm.cdf(bound1, forecast, error_std)
    elif contract_type == "below":
        # P(actual <= bound1)
        prob = stats.norm.cdf(bound1 + 0.5, forecast, error_std)
    elif contract_type == "above":
        # P(actual >= bound1)
        prob = 1 - stats.norm.cdf(bound1 - 0.5, forecast, error_std)
    else:
        return 0.5
    
    # Clamp to reasonable bounds (never say 0% or 100%)
    return max(min(prob, 0.99), 0.01)


# ============ BACKTEST ENGINE ============

def simulate_forecast(actual_temp, error_std):
    """
    Simulate what a realistic forecast would have been.
    
    In backtesting, we don't have historical forecasts, so we work backwards:
    take the actual temp and add typical forecast error to simulate
    what we would have predicted.
    
    In LIVE trading, replace this with actual HRRR/Open-Meteo forecasts.
    """
    # Add random noise to simulate forecast error
    error = np.random.normal(0, error_std * 0.7)
    error = np.clip(error, -2.5 * error_std, 2.5 * error_std)
    return actual_temp + error


def run_backtest(kalshi_df, city, use_simulated_forecasts=True):
    """
    Run the backtest for a single city.
    
    For each day:
    1. Get all available contracts
    2. Generate a forecast (simulated or actual)
    3. Calculate our probability for each contract
    4. Find the best betting opportunity (highest edge)
    5. Calculate bet size (Kelly or flat)
    6. Place the bet and record the outcome
    """
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {city.upper()}")
    print(f"{'='*70}")
    
    city_data = kalshi_df[kalshi_df["city"] == city].copy()
    if len(city_data) == 0:
        return None
    
    dates = sorted(city_data["date"].unique())
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Total days: {len(dates)}")
    
    # Initialize bankroll tracking for Kelly
    current_bankroll = STARTING_BANKROLL
    
    # Initialize results tracking
    results = {
        "city": city,
        "total_days": 0,
        "bets_placed": 0,
        "bets_won": 0,
        "total_wagered": 0,
        "net_profit": 0,
        "starting_bankroll": STARTING_BANKROLL,
        "ending_bankroll": STARTING_BANKROLL,
        "peak_bankroll": STARTING_BANKROLL,
        "max_drawdown": 0,
        "bets_by_bucket": {
            "sweet_spot": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
            "high_price": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
        },
        "bets_by_type": {
            "yes_range": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
            "no_range": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
            "yes_above": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
            "no_above": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
            "yes_below": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
            "no_below": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
        },
        "skipped_cheap": 0,
        "skipped_expensive": 0,
        "skipped_low_edge": 0,
        "skipped_kelly_zero": 0,
        "bet_details": [],
        "profits": [],
        "bankroll_history": [STARTING_BANKROLL],
    }
    
    for target_date in dates:
        day_contracts = city_data[city_data["date"] == target_date].copy()
        if len(day_contracts) == 0:
            continue
        
        # Get actual temperature from the winning contract
        winning = day_contracts[day_contracts["result"] == "yes"]
        if len(winning) == 0:
            continue
        
        actual_temp = winning.iloc[0]["actual_temp_approx"]
        if pd.isna(actual_temp):
            continue
        
        results["total_days"] += 1
        
        # Generate our forecast
        if use_simulated_forecasts:
            forecast = simulate_forecast(actual_temp, FORECAST_ERROR_STD)
        else:
            forecast = actual_temp  # Perfect forecast (ceiling test)
        
        our_bin = get_bin(forecast)
        
        # === FIND ALL QUALIFYING BETS FOR THIS DAY ===
        # We'll collect all bets that meet edge requirements, then decide which to place
        qualifying_bets = []
        
        for _, contract in day_contracts.iterrows():
            kalshi_price = contract["last_price_cents"] / 100
            
            # === FILTER 1: Skip cheap contracts (the trap!) ===
            if kalshi_price < MIN_CONTRACT_PRICE:
                results["skipped_cheap"] += 1
                continue
            
            # === FILTER 2: Skip expensive contracts ===
            if kalshi_price > MAX_CONTRACT_PRICE:
                results["skipped_expensive"] += 1
                continue
            
            # Parse what type of contract this is
            mtype = contract["market_type"]
            if mtype == "range" and pd.notna(contract.get("temp_low")):
                ctype = "range"
                b1, b2 = contract["temp_low"], contract["temp_high"] + 1
            elif mtype == "below" and pd.notna(contract.get("temp_high")):
                ctype = "below"
                b1, b2 = contract["temp_high"], None
            elif mtype == "above" and pd.notna(contract.get("temp_low")):
                ctype = "above"
                b1, b2 = contract["temp_low"], None
            else:
                continue
            
            # Calculate OUR probability for this contract
            our_prob = model_probability(forecast, ctype, b1, b2, FORECAST_ERROR_STD)
            
            # Get minimum required edge for this price level
            min_edge = get_min_edge(kalshi_price)
            
            # === EVALUATE YES BET ===
            yes_edge = our_prob - kalshi_price
            if yes_edge > min_edge:
                edge_ratio = yes_edge / min_edge
                qualifying_bets.append({
                    "contract": contract,
                    "side": "YES",
                    "our_prob": our_prob,
                    "kalshi_price": kalshi_price,
                    "edge": yes_edge,
                    "edge_ratio": edge_ratio,
                    "ctype": ctype,
                    "bucket": get_price_bucket(kalshi_price),
                    "bet_price": kalshi_price,
                })
            
            # === EVALUATE NO BET ===
            is_our_bin = (ctype == "range" and contract.get("temp_low") == our_bin[0])
            
            no_prob = 1 - our_prob
            no_market = 1 - kalshi_price
            no_edge = no_prob - no_market
            no_min_edge = get_min_edge(no_market)
            
            no_bucket = get_price_bucket(no_market)
            skip_no_range = (ctype == "range" and no_bucket == "sweet_spot")
            
            if no_edge > no_min_edge and not is_our_bin and not skip_no_range:
                edge_ratio = no_edge / no_min_edge
                qualifying_bets.append({
                    "contract": contract,
                    "side": "NO",
                    "our_prob": our_prob,
                    "kalshi_price": kalshi_price,
                    "edge": no_edge,
                    "edge_ratio": edge_ratio,
                    "ctype": ctype,
                    "bucket": get_price_bucket(no_market),
                    "bet_price": no_market,
                })
        
        # === SELECT WHICH BETS TO PLACE (SMART SELECTION) ===
        # Since this is a single-city backtest, we apply same-city rules:
        # 1. Always take the best bet
        # 2. Only add more bets if edge_ratio >= SAME_CITY_MULTI_BET_THRESHOLD
        # 3. Cap at MAX_BETS_PER_CITY
        
        selected_bets = []
        if qualifying_bets:
            # Sort by edge ratio (best first)
            qualifying_bets = sorted(qualifying_bets, key=lambda x: -x["edge_ratio"])
            
            # Always take the best bet
            selected_bets.append(qualifying_bets[0])
            
            # Add additional bets only if super confident
            for bet in qualifying_bets[1:]:
                if len(selected_bets) >= MAX_BETS_PER_CITY:
                    break
                if bet["edge_ratio"] >= SAME_CITY_MULTI_BET_THRESHOLD:
                    selected_bets.append(bet)
                    # Mark this as a "super confident" additional bet
                    bet["is_super_confident_stack"] = True
        
        # === CALCULATE BET SIZES AND EXECUTE ===
        day_profit = 0
        bets_placed_today = 0
        
        for bet in selected_bets:
            bet_price = bet["bet_price"]
            our_prob_win = bet["our_prob"] if bet["side"] == "YES" else (1 - bet["our_prob"])
            
            if BETTING_MODE == "kelly":
                bet_size, kelly_info = calculate_kelly_bet(
                    current_bankroll, 
                    our_prob_win, 
                    bet_price,
                    KELLY_FRACTION
                )
                bet["kelly_info"] = kelly_info
            else:
                bet_size = FLAT_BET_SIZE
                bet["kelly_info"] = None
            
            # Skip if Kelly says don't bet
            if bet_size < MIN_BET_SIZE:
                results["skipped_kelly_zero"] += 1
                continue
            
            # === EXECUTE THIS BET ===
            results["bets_placed"] += 1
            results["total_wagered"] += bet_size
            bets_placed_today += 1
            
            contract = bet["contract"]
            side = bet["side"]
            
            # Determine if we won
            if side == "YES":
                won = contract["result"] == "yes"
            else:
                won = contract["result"] == "no"
            
            # Calculate profit/loss
            profit, payout = calculate_bet_outcome(bet_size, bet_price, won)
            
            if won:
                results["bets_won"] += 1
            
            results["net_profit"] += profit
            day_profit += profit
            
            # Update bankroll
            current_bankroll += profit
            results["bankroll_history"].append(current_bankroll)
            
            # Track peak and drawdown
            if current_bankroll > results["peak_bankroll"]:
                results["peak_bankroll"] = current_bankroll
            drawdown = (results["peak_bankroll"] - current_bankroll) / results["peak_bankroll"]
            if drawdown > results["max_drawdown"]:
                results["max_drawdown"] = drawdown
            
            # Track by price bucket
            bucket = bet["bucket"]
            results["bets_by_bucket"][bucket]["count"] += 1
            results["bets_by_bucket"][bucket]["won"] += 1 if won else 0
            results["bets_by_bucket"][bucket]["profit"] += profit
            results["bets_by_bucket"][bucket]["wagered"] += bet_size
            
            # Track by bet type
            type_key = f"{side.lower()}_{bet['ctype']}"
            if type_key in results["bets_by_type"]:
                results["bets_by_type"][type_key]["count"] += 1
                results["bets_by_type"][type_key]["won"] += 1 if won else 0
                results["bets_by_type"][type_key]["profit"] += profit
                results["bets_by_type"][type_key]["wagered"] += bet_size
            
            # Store detailed bet info
            bet_detail = {
                "date": str(target_date.date()),
                "forecast": forecast,
                "actual": actual_temp,
                "contract": contract["contract_subtitle"],
                "side": side,
                "bet_price": bet_price,
                "yes_price": bet["kalshi_price"],
                "our_prob": bet["our_prob"],
                "our_prob_win": our_prob_win,
                "edge": bet["edge"],
                "edge_ratio": bet["edge_ratio"],
                "bucket": bucket,
                "won": won,
                "amount_bet": bet_size,
                "payout": payout if won else 0,
                "profit": profit,
                "bankroll_before": current_bankroll - profit,
                "bankroll_after": current_bankroll,
                "bets_this_day": bets_placed_today,
                "is_super_confident_stack": bet.get("is_super_confident_stack", False),
                "city": city,
            }
            
            # Add Kelly details if using Kelly
            if bet["kelly_info"]:
                bet_detail["kelly_full_pct"] = bet["kelly_info"]["full_kelly_fraction"] * 100
                bet_detail["kelly_used_pct"] = bet["kelly_info"]["fractional_kelly"] * 100
                bet_detail["kelly_odds"] = bet["kelly_info"]["odds"]
            
            results["bet_details"].append(bet_detail)
        
        # Track days with no qualifying bets
        if bets_placed_today == 0 and len(qualifying_bets) == 0:
            results["skipped_low_edge"] += 1
        
        results["profits"].append(day_profit)
    
    # Record final bankroll
    results["ending_bankroll"] = current_bankroll
    
    # Calculate final metrics
    if results["bets_placed"] > 0:
        results["win_rate"] = 100 * results["bets_won"] / results["bets_placed"]
        results["roi"] = 100 * results["net_profit"] / results["total_wagered"]
    else:
        results["win_rate"] = 0
        results["roi"] = 0
    
    # Calculate bankroll growth
    results["bankroll_growth"] = 100 * (results["ending_bankroll"] - results["starting_bankroll"]) / results["starting_bankroll"]
    
    return results


def print_results(results):
    """Print formatted results with detailed bet breakdowns."""
    city = results["city"]
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {city.upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 OVERVIEW")
    print(f"   Days analyzed: {results['total_days']}")
    print(f"   Bets placed: {results['bets_placed']}")
    print(f"   Bets won: {results['bets_won']}")
    print(f"   Win rate: {results['win_rate']:.1f}%")
    print(f"   Total wagered: ${results['total_wagered']:.2f}")
    print(f"   Net profit: ${results['net_profit']:+.2f}")
    print(f"   ROI: {results['roi']:+.1f}%")
    
    if BETTING_MODE == "kelly":
        print(f"\n💰 BANKROLL PERFORMANCE")
        print(f"   Starting bankroll: ${results['starting_bankroll']:,.2f}")
        print(f"   Ending bankroll: ${results['ending_bankroll']:,.2f}")
        print(f"   Bankroll growth: {results['bankroll_growth']:+.1f}%")
        print(f"   Peak bankroll: ${results['peak_bankroll']:,.2f}")
        print(f"   Max drawdown: {results['max_drawdown']*100:.1f}%")
    
    print(f"\n🚫 CONTRACTS FILTERED OUT (not bet on)")
    print(f"   Too cheap (<{MIN_CONTRACT_PRICE*100:.0f}¢): {results['skipped_cheap']} contracts")
    print(f"   Too expensive (>{MAX_CONTRACT_PRICE*100:.0f}¢): {results['skipped_expensive']} contracts")
    print(f"   Insufficient edge: {results['skipped_low_edge']} days with no good bet")
    if BETTING_MODE == "kelly":
        print(f"   Kelly said don't bet: {results.get('skipped_kelly_zero', 0)} opportunities")
    
    print(f"\n💰 PERFORMANCE BY PRICE BUCKET")
    for bucket, stats in results["bets_by_bucket"].items():
        if stats["count"] > 0:
            wr = 100 * stats["won"] / stats["count"]
            roi = 100 * stats["profit"] / stats["wagered"] if stats["wagered"] > 0 else 0
            print(f"   {bucket:<12}: {stats['count']:>4} bets | {wr:>5.1f}% win | ${stats['wagered']:>8.2f} wagered | ${stats['profit']:>+8.2f} profit ({roi:>+6.1f}% ROI)")
    
    print(f"\n📈 PERFORMANCE BY BET TYPE")
    for btype, stats in sorted(results["bets_by_type"].items(), key=lambda x: -x[1]["profit"]):
        if stats["count"] > 0:
            wr = 100 * stats["won"] / stats["count"]
            roi = 100 * stats["profit"] / stats["wagered"] if stats["wagered"] > 0 else 0
            print(f"   {btype:<12}: {stats['count']:>4} bets | {wr:>5.1f}% win | ${stats['wagered']:>8.2f} wagered | ${stats['profit']:>+8.2f} profit ({roi:>+6.1f}% ROI)")
    
    # Detailed sample bets
    if results["bet_details"]:
        print(f"\n" + "="*70)
        print(f"📝 DETAILED BET LOG (last 10 bets)")
        print("="*70)
        
        for bet in results["bet_details"][-10:]:
            status = "✅ WON" if bet["won"] else "❌ LOST"
            
            # Show bet confidence level
            confidence_tag = ""
            if bet.get("is_super_confident_stack"):
                confidence_tag = " 🔥 SUPER CONFIDENT STACK"
            elif bet.get("bets_this_day", 1) == 1:
                confidence_tag = " ⭐ BEST BET"
            
            city_name = bet.get("city", results["city"])
            
            print(f"\n   {bet['date']} | {city_name} | {bet['contract']}{confidence_tag}")
            print(f"   ─────────────────────────────────────────────────────")
            print(f"   ⏰ Bet placed: Evening before {bet['date']}")
            print(f"   Forecast: {bet['forecast']:.1f}°F → Actual: {bet['actual']:.1f}°F")
            
            # Show confidence level explanation
            edge_ratio = bet.get('edge_ratio', 0)
            if edge_ratio >= SUPER_CONFIDENT_EDGE_RATIO:
                print(f"   Edge ratio: {edge_ratio:.2f}x minimum required 🔥 (SUPER CONFIDENT)")
            elif edge_ratio >= SAME_CITY_MULTI_BET_THRESHOLD:
                print(f"   Edge ratio: {edge_ratio:.2f}x minimum required ✓ (high confidence)")
            else:
                print(f"   Edge ratio: {edge_ratio:.2f}x minimum required")
            
            print(f"   ")
            print(f"   BET: {bet['side']} at {bet['bet_price']*100:.0f}¢")
            
            # Show Kelly calculation details
            if BETTING_MODE == "kelly" and "kelly_full_pct" in bet:
                print(f"   ")
                print(f"   KELLY CALCULATION:")
                print(f"      Your win probability: {bet['our_prob_win']*100:.1f}%")
                print(f"      Odds (profit per $1): {bet['kelly_odds']:.2f}x")
                print(f"      Full Kelly: {bet['kelly_full_pct']:.1f}% of bankroll")
                print(f"      {KELLY_FRACTION:.0%} Kelly used: {bet['kelly_used_pct']:.2f}% of bankroll")
                print(f"      Bankroll before: ${bet['bankroll_before']:,.2f}")
                print(f"      → Bet size: ${bet['amount_bet']:.2f}")
            else:
                print(f"   Amount wagered: ${bet['amount_bet']:.2f}")
            
            print(f"   ")
            print(f"   Your probability: {bet['our_prob_win']*100:.1f}%")
            print(f"   Your edge: {bet['edge']*100:+.1f}%")
            print(f"   ")
            if bet['won']:
                print(f"   {status}")
                print(f"   Payout: ${bet['payout']:.2f}")
                print(f"   PROFIT: ${bet['profit']:+.2f}")
                if BETTING_MODE == "kelly":
                    print(f"   Bankroll after: ${bet['bankroll_after']:,.2f}")
            else:
                print(f"   {status}")
                print(f"   LOSS: ${bet['profit']:.2f}")
                if BETTING_MODE == "kelly":
                    print(f"   Bankroll after: ${bet['bankroll_after']:,.2f}")


def print_how_to_read_bets():
    """Print an explanation of how to interpret bet results."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        HOW TO READ THE BET DETAILS                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Each bet shows:

┌─ DATE & CONTRACT ─────────────────────────────────────────────────────────────
│  The date and which temperature range the contract covers
│  Example: "2024-01-15 | Between 73°F and 74°F (inclusive)"

┌─ FORECAST vs ACTUAL ──────────────────────────────────────────────────────────
│  Forecast: What the model predicted
│  Actual: What the temperature actually was

┌─ BET DETAILS ─────────────────────────────────────────────────────────────────
│  BET: YES or NO, and at what price
│  Amount wagered: How much you risked (always $10 in flat betting)
│  Our probability: What our model thinks the true probability is
│  Our edge: Our probability minus the market price

┌─ OUTCOME ─────────────────────────────────────────────────────────────────────
│  ✅ WON: You get paid out
│     Payout = Amount wagered / Bet price
│     Profit = Payout - Amount wagered
│  
│  ❌ LOST: You lose your wager
│     Loss = Amount wagered

EXAMPLE:
  BET: YES at 25¢
  Amount wagered: $10.00
  
  If WON:
    Payout = $10 / 0.25 = $40
    Profit = $40 - $10 = +$30
  
  If LOST:
    Loss = -$10

╔══════════════════════════════════════════════════════════════════════════════╗
║                     SMART BET SELECTION EXPLAINED                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

The model uses SMART selection to decide how many bets to recommend:

SCENARIO 1: Multiple cities have good bets
────────────────────────────────────────────
  "Hey, here are multiple good bets in DIFFERENT cities!"
  → Chicago: YES on 75-76°F at 30¢ (edge ratio 1.8x)
  → Miami: YES on 85-86°F at 25¢ (edge ratio 1.5x)
  
  ✅ Safe to bet on BOTH - different cities are uncorrelated!

SCENARIO 2: One city, SUPER confident
────────────────────────────────────────────
  "I'm SUPER confident in Chicago today - stacking bets!"
  → Chicago: YES on 75-76°F at 30¢ (edge ratio 2.8x) 🔥
  → Chicago: YES on 73-74°F at 20¢ (edge ratio 2.2x) 🔥
  
  ✅ Both edge ratios are ≥ 2.0x, so we stack same-city bets

SCENARIO 3: One city, normal confidence  
────────────────────────────────────────────
  "Here's ONE great bet - I'm confident but not stacking"
  → Chicago: YES on 75-76°F at 30¢ (edge ratio 1.5x) ⭐
  
  ✅ Just the best bet - other same-city bets aren't confident enough

The key insight: DIFFERENT CITIES = safe to multi-bet (weather is independent)
                SAME CITY = only stack if SUPER confident (correlated risk)
""")


def print_recommendation_examples():
    """Print example recommendation outputs."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXAMPLE DAILY RECOMMENDATIONS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Here's what a daily recommendation might look like:

============================================================
🎯 TODAY'S BETTING RECOMMENDATIONS  
============================================================

✅ Found good bets in 2 DIFFERENT CITIES (uncorrelated):
   → Safe to bet on all of these!

   BET #1: Chicago
   🔥 SUPER CONFIDENT
   Contract: High temp 75-76°F
   Side: YES at 28¢
   Your probability: 42.0%
   Edge: +14.0% (2.3x minimum)
   💰 RECOMMENDED BET: $2.85

   BET #2: Miami  
   • Good edge
   Contract: High temp 85-86°F
   Side: YES at 35¢
   Your probability: 48.0%
   Edge: +13.0% (1.6x minimum)
   💰 RECOMMENDED BET: $1.92

------------------------------------------------------------
   Total to wager: $4.77 (15.9% of bankroll)
   Current bankroll: $30.00
============================================================
""")


def main():
    print("="*70)
    print("KALSHI WEATHER MODEL BACKTEST - IMPROVED v4 (DOCUMENTED)")
    print("="*70)
    
    print_how_to_read_bets()
    print_recommendation_examples()
    
    if BETTING_MODE == "kelly":
        explain_kelly_criterion()
    
    print(f"\n{'='*70}")
    print("STRATEGY CONFIGURATION")
    print(f"{'='*70}")
    print(f"\n  Betting mode: {BETTING_MODE.upper()}")
    if BETTING_MODE == "kelly":
        print(f"  Starting bankroll: ${STARTING_BANKROLL:,.0f}")
        print(f"  Kelly fraction: {KELLY_FRACTION:.0%} (half Kelly)")
        print(f"  Min bet size: ${MIN_BET_SIZE:.2f}")
        print(f"  Max bet: {MAX_BET_FRACTION:.0%} of bankroll")
    else:
        print(f"  Bet size: ${FLAT_BET_SIZE:.2f} per bet")
    print(f"  Min contract price: {MIN_CONTRACT_PRICE*100:.0f}¢ (skip cheap contracts)")
    print(f"  Max contract price: {MAX_CONTRACT_PRICE*100:.0f}¢ (skip expensive contracts)")
    print(f"  Sweet spot: {SWEET_SPOT_LOW*100:.0f}¢-{SWEET_SPOT_HIGH*100:.0f}¢ (best historical returns)")
    print(f"  Min edge required: {EDGE_REQUIREMENTS['sweet_spot']*100:.0f}%")
    print(f"  Forecast error model: σ = {FORECAST_ERROR_STD}°F")
    print(f"\n  ⏰ BET TIMING ASSUMPTION: {BET_TIMING_DESCRIPTION}")
    print(f"     (Using day-ahead forecast and previous day's closing prices)")
    
    print(f"\n  🎯 SMART BET SELECTION:")
    print(f"     • Different cities: Bet freely (uncorrelated)")
    print(f"     • Same city: Only stack if edge ratio ≥ {SAME_CITY_MULTI_BET_THRESHOLD}x")
    print(f"     • Max {MAX_BETS_PER_CITY} bets per city, {MAX_TOTAL_BETS_PER_DAY} total per day")
    print(f"     • 'Super confident' threshold: {SUPER_CONFIDENT_EDGE_RATIO}x edge ratio")

    # Load data
    kalshi_file = "kalshi_backtest_data.csv"
    if not os.path.exists(kalshi_file):
        print(f"\n❌ Error: {kalshi_file} not found!")
        print("   Please ensure the Kalshi historical data file is in the same directory.")
        return

    kalshi_df = pd.read_csv(kalshi_file)
    kalshi_df["date"] = pd.to_datetime(kalshi_df["date"])

    print(f"\n  Loaded {len(kalshi_df)} contract records")

    # Set seed for reproducibility
    np.random.seed(42)

    all_results = []

    for city in kalshi_df["city"].unique():
        results = run_backtest(kalshi_df, city, use_simulated_forecasts=True)
        if results:
            all_results.append(results)
            print_results(results)

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY - ALL CITIES COMBINED")
    print(f"{'='*70}\n")

    total_wagered = sum(r["total_wagered"] for r in all_results)
    total_profit = sum(r["net_profit"] for r in all_results)
    total_bets = sum(r["bets_placed"] for r in all_results)
    total_wins = sum(r["bets_won"] for r in all_results)

    overall_roi = 100 * total_profit / total_wagered if total_wagered > 0 else 0
    overall_wr = 100 * total_wins / total_bets if total_bets > 0 else 0

    print(f"{'City':<25} {'Bets':<8} {'Won':<8} {'Win%':<10} {'Wagered':<12} {'Profit':<12} {'ROI':<10}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['city']:<25} {r['bets_placed']:<8} {r['bets_won']:<8} {r['win_rate']:>5.1f}%    ${r['total_wagered']:<9.2f} ${r['net_profit']:>+9.2f}   {r['roi']:>+6.1f}%")
    print("-" * 85)
    print(f"{'TOTAL':<25} {total_bets:<8} {total_wins:<8} {overall_wr:>5.1f}%    ${total_wagered:<9.2f} ${total_profit:>+9.2f}   {overall_roi:>+6.1f}%")

    # Bankroll summary for Kelly
    if BETTING_MODE == "kelly":
        print(f"\n💰 BANKROLL SUMMARY (Kelly Criterion)")
        print("-" * 70)
        total_ending = sum(r["ending_bankroll"] for r in all_results)
        total_starting = STARTING_BANKROLL * len(all_results)
        overall_growth = 100 * (total_ending - total_starting) / total_starting
        
        print(f"{'City':<25} {'Start':<12} {'End':<12} {'Growth':<12} {'Max DD':<10}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['city']:<25} ${r['starting_bankroll']:<10,.0f} ${r['ending_bankroll']:<10,.2f} {r['bankroll_growth']:>+7.1f}%    {r['max_drawdown']*100:>5.1f}%")
        print("-" * 70)
        avg_ending = total_ending / len(all_results) if all_results else 0
        max_dd = max(r["max_drawdown"] for r in all_results) if all_results else 0
        print(f"{'AVERAGE':<25} ${STARTING_BANKROLL:<10,.0f} ${avg_ending:<10,.2f} {overall_growth/len(all_results):>+7.1f}%    {max_dd*100:>5.1f}%")

    # Aggregate by bucket across all cities
    print(f"\n📊 AGGREGATE BY PRICE BUCKET (All Cities)")
    print("-" * 70)

    agg_buckets = {
        "sweet_spot": {"count": 0, "won": 0, "profit": 0, "wagered": 0},
        "high_price": {"count": 0, "won": 0, "profit": 0, "wagered": 0}
    }

    for r in all_results:
        for bucket, stats in r["bets_by_bucket"].items():
            agg_buckets[bucket]["count"] += stats["count"]
            agg_buckets[bucket]["won"] += stats["won"]
            agg_buckets[bucket]["profit"] += stats["profit"]
            agg_buckets[bucket]["wagered"] += stats["wagered"]

    for bucket, stats in agg_buckets.items():
        if stats["count"] > 0:
            wr = 100 * stats["won"] / stats["count"]
            roi = 100 * stats["profit"] / stats["wagered"]
            print(f"  {bucket:<12}: {stats['count']:>4} bets | {wr:>5.1f}% win | ${stats['wagered']:.0f} wagered | ${stats['profit']:>+8.2f} profit ({roi:>+6.1f}% ROI)")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION & NEXT STEPS")
    print(f"{'='*70}")

    if overall_roi > 0:
        print(f"\n✅ PROFITABLE: {overall_roi:+.1f}% ROI on ${total_wagered:.0f} wagered")
        print(f"   Net profit: ${total_profit:+.2f}")
        print("   The strategy filters appear to be working!")
    else:
        print(f"\n⚠️  NOT PROFITABLE: {overall_roi:.1f}% ROI")
        print("   Consider adjusting edge requirements or price filters.")

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              IMPORTANT NOTES                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. BET TIMING ASSUMPTION
   This backtest assumes you place bets the EVENING BEFORE (6-8 PM).
   
   Example timeline for Jan 15th contract:
     • Jan 14th, 6 PM: Check forecast + Kalshi prices
     • Jan 14th, 6 PM: Place your bet
     • Jan 15th: Actual high temp recorded
     • Jan 15th, evening: Contract settles
   
   This gives you a ~18-24 hour forecast window, which matches the
   FORECAST_ERROR_STD of 2.8°F used in the probability model.

2. SIMULATED FORECASTS
   This backtest uses SIMULATED forecasts (actual temp + random noise).
   Real performance depends on actual forecast quality from HRRR/Open-Meteo.

3. PRICE ASSUMPTIONS
   The backtest uses "last_price_cents" from historical data.
   Real trading considerations:
     • Bid/ask spread (you pay slightly more than mid-price)
     • Price movement between decision and execution
     • Liquidity (large bets may move the market)

4. FOR LIVE TRADING
   • Check forecasts around 5-7 PM the day before
   • Use real HRRR/Open-Meteo ensemble forecasts
   • Place bets when you see good edge (>12%)
   • Apply the same price filters (15-90¢ range)
   • Focus on the sweet spot (15-50¢)

5. RISK MANAGEMENT
   • Never bet more than you can afford to lose
   • Expect losing streaks (even +20% ROI has bad weeks)
   • Track your results and adjust as needed
""")


if __name__ == "__main__":
    main()