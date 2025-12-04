"""
KALSHI WEATHER MODEL BACKTESTER - REAL DATA
============================================
Uses ACTUAL Kalshi historical market data to test our model.

This is the REAL test:
- Real market prices (what Kalshi was pricing contracts at)
- Real outcomes (which bin actually won)
- Real edge calculation (our model vs actual market)

Run fetch_kalshi_history.py first to download the data!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import json
import os

# ============ CONFIGURATION ============
FLAT_BET_SIZE = 10
MIN_EDGE_THRESHOLD = 0.10  # 10% minimum edge to bet

# Model parameters
SPIKE_WEIGHT = 0.55
UNCERTAINTY_STD = 2.5

# ============ LOAD DATA ============

def load_kalshi_data(csv_file="kalshi_backtest_data.csv"):
    """Load the Kalshi historical data."""
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        print("   Run fetch_kalshi_history.py first to download data.")
        return None
    
    df = pd.read_csv(csv_file)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_temperature_history(city):
    """Load historical temperature data for forecast building."""
    files = {
        "Chicago (Midway)": "weather_data_chicago.csv",
        "NYC (Central Park)": "weather_data_nyc.csv",
        "Miami (MIA)": "weather_data_miami.csv"
    }
    
    filename = files.get(city)
    if not filename or not os.path.exists(filename):
        return None
    
    df = pd.read_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "value"]].rename(columns={"value": "temp"})


# ============ MODEL FUNCTIONS ============

def get_bin(temp):
    """Get Kalshi bin for a temperature."""
    lower = int(np.floor(temp))
    if lower % 2 == 0:
        lower -= 1
    return (lower, lower + 1)


def build_forecast(temp_history, target_date):
    """
    Build our model's forecast for a target date using historical data.
    """
    # Get data before target date
    hist = temp_history[temp_history["date"] < target_date].copy()
    
    if len(hist) < 14:
        return None
    
    # Get the last 7 days
    recent = hist.tail(7)
    
    # Seasonal average
    doy = target_date.dayofyear
    seasonal = hist[hist["date"].dt.dayofyear == doy]["temp"]
    if len(seasonal) >= 2:
        seasonal_avg = seasonal.mean()
    else:
        # Use nearby days
        nearby = hist[(hist["date"].dt.dayofyear >= doy - 7) & 
                      (hist["date"].dt.dayofyear <= doy + 7)]["temp"]
        seasonal_avg = nearby.mean() if len(nearby) > 0 else recent["temp"].mean()
    
    # Recent trend
    yesterday = recent["temp"].iloc[-1]
    three_day_avg = recent["temp"].iloc[-3:].mean()
    seven_day_avg = recent["temp"].mean()
    
    # Build forecast
    forecast = (
        seasonal_avg * 0.35 +
        yesterday * 0.35 +
        three_day_avg * 0.20 +
        seven_day_avg * 0.10
    )
    
    return forecast


def model_probability(contract_low, contract_high, forecast, uncertainty_std, spike_weight):
    """Calculate our model's probability for a contract."""
    spread_weight = 1 - spike_weight
    forecast_bin = get_bin(forecast)
    forecast_in_bin = contract_low <= forecast <= contract_high
    
    spread_prob = stats.norm.cdf(contract_high, forecast, uncertainty_std) - \
                  stats.norm.cdf(contract_low, forecast, uncertainty_std)
    
    if forecast_in_bin:
        return min(spike_weight + (spread_weight * spread_prob), 0.99)
    else:
        return spread_weight * spread_prob


def model_prob_below(threshold, forecast, uncertainty_std, spike_weight):
    """Probability for 'X or below' contracts."""
    spread_weight = 1 - spike_weight
    spread_prob = stats.norm.cdf(threshold + 0.5, forecast, uncertainty_std)
    
    if forecast <= threshold:
        return min(spike_weight + (spread_weight * spread_prob), 0.99)
    else:
        return spread_weight * spread_prob


def model_prob_above(threshold, forecast, uncertainty_std, spike_weight):
    """Probability for 'X or above' contracts."""
    spread_weight = 1 - spike_weight
    spread_prob = 1 - stats.norm.cdf(threshold - 0.5, forecast, uncertainty_std)
    
    if forecast >= threshold:
        return min(spike_weight + (spread_weight * spread_prob), 0.99)
    else:
        return spread_weight * spread_prob


# ============ BACKTEST ============

def run_backtest(kalshi_df, city):
    """
    Run backtest for a city using REAL Kalshi data.
    """
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {city.upper()}")
    print(f"{'='*70}")
    
    # Filter to this city
    city_data = kalshi_df[kalshi_df["city"] == city].copy()
    
    if len(city_data) == 0:
        print(f"  No data for {city}")
        return None
    
    # Load temperature history
    temp_history = load_temperature_history(city)
    if temp_history is None:
        print(f"  ❌ No temperature history for {city}")
        return None
    
    # Get unique dates
    dates = sorted(city_data["date"].unique())
    print(f"  Dates with Kalshi data: {len(dates)}")
    print(f"  Range: {dates[0].date()} to {dates[-1].date()}")
    
    # Results tracking
    results = {
        "total_days": 0,
        "forecast_correct_bin": 0,
        "market_correct_bin": 0,
        "bets_placed": 0,
        "bets_won": 0,
        "total_wagered": 0,
        "net_profit": 0,
        "bet_details": [],
        "daily_results": [],
    }
    
    for target_date in dates:
        # Get all contracts for this date
        day_contracts = city_data[city_data["date"] == target_date].copy()
        
        if len(day_contracts) == 0:
            continue
        
        # Build our forecast
        forecast = build_forecast(temp_history, target_date)
        if forecast is None:
            continue
        
        results["total_days"] += 1
        
        # Find the actual winning bin
        winning_contract = day_contracts[day_contracts["result"] == "yes"]
        if len(winning_contract) == 0:
            continue
        
        actual_temp = winning_contract.iloc[0]["actual_temp_approx"]
        winning_subtitle = winning_contract.iloc[0]["contract_subtitle"]
        
        # Our forecast bin
        our_bin = get_bin(forecast)
        our_bin_str = f"{our_bin[0]}° to {our_bin[1]}°"
        
        # Did our forecast bin match the winning bin?
        # Check if our bin is in the winning subtitle
        forecast_correct = (f"{our_bin[0]}°" in winning_subtitle and f"{our_bin[1]}°" in winning_subtitle)
        if forecast_correct:
            results["forecast_correct_bin"] += 1
        
        # Find market's top pick (highest priced contract)
        market_top = day_contracts.loc[day_contracts["last_price_cents"].idxmax()]
        market_correct = market_top["result"] == "yes"
        if market_correct:
            results["market_correct_bin"] += 1
        
        # === BETTING LOGIC ===
        # Look for contracts where we have edge
        
        best_bet = None
        best_edge = MIN_EDGE_THRESHOLD
        
        for _, contract in day_contracts.iterrows():
            kalshi_price = contract["last_price_cents"] / 100  # Convert to probability
            
            if kalshi_price <= 0.01 or kalshi_price >= 0.99:
                continue  # Skip extreme prices
            
            # Calculate our probability
            if contract["market_type"] == "range":
                our_prob = model_probability(
                    contract["temp_low"], 
                    contract["temp_high"] + 1,
                    forecast, UNCERTAINTY_STD, SPIKE_WEIGHT
                )
            elif contract["market_type"] == "below":
                our_prob = model_prob_below(
                    contract["temp_high"],
                    forecast, UNCERTAINTY_STD, SPIKE_WEIGHT
                )
            elif contract["market_type"] == "above":
                our_prob = model_prob_above(
                    contract["temp_low"],
                    forecast, UNCERTAINTY_STD, SPIKE_WEIGHT
                )
            else:
                continue
            
            # Calculate edge for YES bet
            yes_edge = our_prob - kalshi_price
            if yes_edge > best_edge:
                best_edge = yes_edge
                best_bet = {
                    "contract": contract,
                    "side": "YES",
                    "our_prob": our_prob,
                    "kalshi_price": kalshi_price,
                    "edge": yes_edge
                }
            
            # Calculate edge for NO bet
            no_edge = (1 - kalshi_price) - (1 - our_prob)
            # Don't bet NO on our forecast bin
            is_our_bin = (contract["market_type"] == "range" and 
                         contract["temp_low"] == our_bin[0])
            
            if no_edge > best_edge and not is_our_bin:
                best_edge = no_edge
                best_bet = {
                    "contract": contract,
                    "side": "NO", 
                    "our_prob": our_prob,
                    "kalshi_price": kalshi_price,
                    "edge": no_edge
                }
        
        # Place bet if we found edge
        day_result = {
            "date": target_date,
            "forecast": forecast,
            "actual_temp": actual_temp,
            "forecast_correct": forecast_correct,
            "market_correct": market_correct,
            "bet_placed": False,
            "profit": 0
        }
        
        if best_bet:
            results["bets_placed"] += 1
            results["total_wagered"] += FLAT_BET_SIZE
            
            contract = best_bet["contract"]
            
            # Did we win?
            if best_bet["side"] == "YES":
                won = contract["result"] == "yes"
                price = best_bet["kalshi_price"]
            else:
                won = contract["result"] == "no"
                price = 1 - best_bet["kalshi_price"]
            
            if won:
                results["bets_won"] += 1
                payout = FLAT_BET_SIZE / price
                profit = payout - FLAT_BET_SIZE
            else:
                profit = -FLAT_BET_SIZE
            
            results["net_profit"] += profit
            
            day_result["bet_placed"] = True
            day_result["profit"] = profit
            
            results["bet_details"].append({
                "date": str(target_date.date()),
                "contract": contract["contract_subtitle"],
                "side": best_bet["side"],
                "kalshi_price": best_bet["kalshi_price"],
                "our_prob": best_bet["our_prob"],
                "edge": best_bet["edge"],
                "won": won,
                "profit": profit
            })
        
        results["daily_results"].append(day_result)
    
    # Calculate final metrics
    n = results["total_days"]
    if n > 0:
        results["forecast_accuracy"] = results["forecast_correct_bin"] / n * 100
        results["market_accuracy"] = results["market_correct_bin"] / n * 100
    
    if results["bets_placed"] > 0:
        results["win_rate"] = results["bets_won"] / results["bets_placed"] * 100
        results["roi"] = results["net_profit"] / results["total_wagered"] * 100
    else:
        results["win_rate"] = 0
        results["roi"] = 0
    
    return results


def print_results(results, city):
    """Print formatted results."""
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {city.upper()}")
    print(f"{'='*70}")
    
    n = results["total_days"]
    
    print(f"\n📊 FORECAST ACCURACY (vs REAL Kalshi outcomes)")
    print(f"   Days analyzed: {n}")
    print(f"   Our model correct bin: {results['forecast_correct_bin']}/{n} ({results['forecast_accuracy']:.1f}%)")
    print(f"   Market top pick correct: {results['market_correct_bin']}/{n} ({results['market_accuracy']:.1f}%)")
    print(f"   Our edge vs market: {results['forecast_accuracy'] - results['market_accuracy']:+.1f}%")
    
    print(f"\n💰 BETTING PERFORMANCE (vs REAL Kalshi prices)")
    print(f"   Bets placed: {results['bets_placed']}")
    print(f"   Bets won: {results['bets_won']} ({results['win_rate']:.1f}%)")
    print(f"   Total wagered: ${results['total_wagered']:.2f}")
    print(f"   Net profit: ${results['net_profit']:.2f}")
    print(f"   ROI: {results['roi']:+.1f}%")
    
    # Show some example bets
    if results["bet_details"]:
        print(f"\n📝 SAMPLE BETS (last 10):")
        for bet in results["bet_details"][-10:]:
            won_str = "✅ WON" if bet["won"] else "❌ LOST"
            print(f"   {bet['date']}: {bet['side']} {bet['contract']}")
            print(f"      Kalshi: {bet['kalshi_price']*100:.0f}¢ | Our prob: {bet['our_prob']*100:.0f}% | Edge: {bet['edge']*100:+.0f}% | {won_str} ${bet['profit']:+.2f}")
    
    # Cumulative P&L
    if results["daily_results"]:
        profits = [d["profit"] for d in results["daily_results"]]
        cumsum = np.cumsum(profits)
        print(f"\n📈 P&L SUMMARY")
        print(f"   Max drawdown: ${min(cumsum):.2f}")
        print(f"   Peak profit: ${max(cumsum):.2f}")
        print(f"   Final P&L: ${cumsum[-1]:.2f}")


def run_full_backtest():
    """Run backtest for all cities."""
    
    print("="*70)
    print("KALSHI WEATHER MODEL BACKTEST - REAL DATA")
    print("="*70)
    print("\nThis uses ACTUAL Kalshi market prices and outcomes!")
    
    # Load Kalshi data
    kalshi_df = load_kalshi_data()
    if kalshi_df is None:
        return
    
    print(f"\nLoaded {len(kalshi_df)} contract records")
    print(f"Cities: {kalshi_df['city'].unique()}")
    
    all_results = {}
    
    for city in kalshi_df["city"].unique():
        results = run_backtest(kalshi_df, city)
        if results:
            all_results[city] = results
            print_results(results, city)
    
    # Overall summary
    if all_results:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY - REAL KALSHI DATA")
        print(f"{'='*70}\n")
        
        print(f"{'City':<20} {'Days':<8} {'Forecast':<12} {'Market':<12} {'ROI':<10} {'Profit':<10}")
        print("-" * 75)
        
        total_bets = 0
        total_wins = 0
        total_wagered = 0
        total_profit = 0
        
        for city, r in all_results.items():
            print(f"{city:<20} {r['total_days']:<8} {r['forecast_accuracy']:>6.1f}%     {r['market_accuracy']:>6.1f}%     {r['roi']:>+6.1f}%    ${r['net_profit']:>+8.2f}")
            total_bets += r['bets_placed']
            total_wins += r['bets_won']
            total_wagered += r['total_wagered']
            total_profit += r['net_profit']
        
        print("-" * 75)
        overall_roi = total_profit / total_wagered * 100 if total_wagered > 0 else 0
        overall_wr = total_wins / total_bets * 100 if total_bets > 0 else 0
        print(f"{'TOTAL':<20} {'':<8} {'':<12} {'':<12} {overall_roi:>+6.1f}%    ${total_profit:>+8.2f}")
        
        print(f"\n💡 INTERPRETATION:")
        if overall_roi > 0:
            print(f"   ✅ Model is PROFITABLE against real Kalshi prices!")
            print(f"   💰 ROI: {overall_roi:+.1f}% on ${total_wagered:.0f} wagered")
        else:
            print(f"   ❌ Model is NOT profitable against real Kalshi prices")
            print(f"   📉 ROI: {overall_roi:.1f}%")
        
        print(f"\n   This is the REAL test - actual market prices, actual outcomes.")


if __name__ == "__main__":
    run_full_backtest()