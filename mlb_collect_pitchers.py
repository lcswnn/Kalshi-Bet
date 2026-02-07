"""
MLB Pitcher Stats Collector
=============================
Collects season-level pitching stats from FanGraphs via pybaseball.
These get merged into the prediction model as pitcher quality features.

Usage:
    python mlb_collect_pitchers.py

Output:
    - mlb_pitcher_stats.csv

Takes ~2-3 minutes (one request per season).
"""

import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))  # One extra year back for prior-season lookups


def collect_pitcher_stats():
    """Pull season-level pitching stats from FanGraphs."""
    from pybaseball import pitching_stats

    all_data = []

    for year in SEASONS:
        print(f"Fetching {year} pitching stats...")
        try:
            # qual=30 means minimum 30 IP to be included (filters out position players)
            df = pitching_stats(year, year, qual=30)
            df["Season"] = year
            all_data.append(df)
            print(f"  ✓ {len(df)} pitchers")
            time.sleep(2)
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            time.sleep(5)

    combined = pd.concat(all_data, ignore_index=True)

    # Keep the columns we care about
    keep_cols = [
        "Season", "Name", "Team", "Age", "W", "L", "ERA", "G", "GS",
        "IP", "WAR", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9",
        "K%", "BB%", "BABIP", "LOB%", "HR/FB",
    ]
    keep_cols = [c for c in keep_cols if c in combined.columns]
    combined = combined[keep_cols]

    combined.to_csv("mlb_pitcher_stats.csv", index=False)
    print(f"\n✓ Saved mlb_pitcher_stats.csv: {len(combined)} pitcher-seasons")
    print(f"  Seasons: {sorted(combined['Season'].unique())}")
    print(f"  Sample pitchers: {combined.head(3)['Name'].tolist()}")

    return combined


if __name__ == "__main__":
    print("=" * 50)
    print("MLB Pitcher Stats Collector")
    print("=" * 50)
    collect_pitcher_stats()
    print("\nDone! Now run: python mlb_model_v2.py")