#!/usr/bin/env python3
"""
COMPLETE BACKTESTING EXAMPLE
============================

This script demonstrates a complete workflow:
1. Generate sample data
2. Run backtest
3. Analyze results
4. Optimize parameters

Run this to see the backtesting framework in action!
"""

import subprocess
import os
import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and print output."""
    print(f"🔄 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running command")
        return False
    return True

def main():
    print_header("SPORTS BETTING MODEL - COMPLETE BACKTESTING DEMO")
    
    print("""
This demo will:
1. Generate 3 seasons of sample NBA data (~7,500 games)
2. Run a backtest with default parameters
3. Run a backtest with conservative parameters
4. Run a backtest with aggressive parameters
5. Run Monte Carlo simulation
6. Show you all the output files

Press Enter to continue (or Ctrl+C to cancel)...
""")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled")
        return
    
    # ========================================================================
    # STEP 1: Generate Sample Data
    # ========================================================================
    
    print_header("STEP 1: Generate Sample NBA Data")
    
    if Path("sample_nba_games.csv").exists():
        print("✅ sample_nba_games.csv already exists, skipping generation")
    else:
        if not run_command(
            [sys.executable, "generate_sample_data.py"],
            "Generating 3 seasons of NBA data"
        ):
            return
    
    input("\n✅ Data generated! Press Enter to continue...")
    
    # ========================================================================
    # STEP 2: Basic Backtest
    # ========================================================================
    
    print_header("STEP 2: Run Basic Backtest (Default Parameters)")
    print("""
Parameters:
- Bankroll: $1,000
- Kelly fraction: 0.5 (half Kelly)
- Min edge: 8%
- Train/test split: 70/30
""")
    
    if not run_command(
        [sys.executable, "backtest_betting_model.py", 
         "sample_nba_games.csv",
         "--bankroll", "1000",
         "--kelly", "0.5",
         "--min-edge", "0.08"],
        "Running backtest with default parameters"
    ):
        return
    
    print("""
✅ Backtest complete! Files generated:
   - backtest_results.png (visualizations)
   - backtest_bets.csv (detailed bet log)
   - backtest_summary.json (results summary)
""")
    
    input("Press Enter to continue...")
    
    # ========================================================================
    # STEP 3: Conservative Backtest
    # ========================================================================
    
    print_header("STEP 3: Conservative Strategy")
    print("""
Parameters:
- Bankroll: $1,000
- Kelly fraction: 0.25 (quarter Kelly - very conservative)
- Min edge: 10% (only bet on high-edge opportunities)
""")
    
    if not run_command(
        [sys.executable, "backtest_betting_model.py",
         "sample_nba_games.csv",
         "--bankroll", "1000",
         "--kelly", "0.25",
         "--min-edge", "0.10"],
        "Running conservative backtest"
    ):
        return
    
    print("\n✅ Conservative backtest complete!")
    print("   Compare ROI and max drawdown to default strategy")
    
    input("Press Enter to continue...")
    
    # ========================================================================
    # STEP 4: Aggressive Backtest
    # ========================================================================
    
    print_header("STEP 4: Aggressive Strategy")
    print("""
Parameters:
- Bankroll: $1,000
- Kelly fraction: 0.75 (3/4 Kelly - aggressive)
- Min edge: 5% (bet on more opportunities)
""")
    
    if not run_command(
        [sys.executable, "backtest_betting_model.py",
         "sample_nba_games.csv",
         "--bankroll", "1000",
         "--kelly", "0.75",
         "--min-edge", "0.05"],
        "Running aggressive backtest"
    ):
        return
    
    print("\n✅ Aggressive backtest complete!")
    print("   Higher ROI but also higher risk (max drawdown)")
    
    input("Press Enter to continue...")
    
    # ========================================================================
    # STEP 5: Monte Carlo Simulation
    # ========================================================================
    
    print_header("STEP 5: Monte Carlo Simulation (1,000 runs)")
    print("""
This simulates 1,000 different outcomes to show you:
- Range of possible results
- Probability of profit
- Confidence intervals
""")
    
    if not run_command(
        [sys.executable, "backtest_betting_model.py",
         "sample_nba_games.csv",
         "--bankroll", "1000",
         "--kelly", "0.5",
         "--min-edge", "0.08",
         "--monte-carlo", "1000"],
        "Running Monte Carlo simulation"
    ):
        return
    
    print("\n✅ Monte Carlo complete!")
    print("   Check the percentiles to understand risk/reward")
    
    input("Press Enter to see file summary...")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_header("DEMO COMPLETE - FILES GENERATED")
    
    files_to_check = [
        ("sample_nba_games.csv", "Historical game data"),
        ("backtest_results.png", "Visualization charts"),
        ("backtest_bets.csv", "Detailed bet log"),
        ("backtest_summary.json", "Results summary"),
    ]
    
    print("Files in current directory:\n")
    for filename, description in files_to_check:
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            print(f"✅ {filename:30} - {description}")
            print(f"   Size: {size:,} bytes")
        else:
            print(f"❌ {filename:30} - Not found")
        print()
    
    print_header("NEXT STEPS")
    print("""
1. Open backtest_results.png to see visualizations
   
2. Open backtest_bets.csv in Excel/Google Sheets to analyze individual bets
   
3. Review backtest_summary.json for full results

4. Try parameter optimization:
   python backtest_betting_model.py sample_nba_games.csv --optimize-params

5. Use your own historical data:
   - Get NBA game data from Basketball Reference or Kaggle
   - Format as CSV: date,home_team,away_team,home_won
   - Run: python backtest_betting_model.py YOUR_DATA.csv --bankroll 1000

6. Update the main model with real Elo ratings:
   python sports_betting_model.py --update-elo sample_nba_games.csv

7. Start finding real betting opportunities:
   python sports_betting_model.py --bankroll 100 --min-edge 0.08
""")
    
    print("\n" + "="*70)
    print("🎉 DEMO COMPLETE! Happy backtesting!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()