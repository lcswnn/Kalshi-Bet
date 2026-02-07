"""
MLB Odds from SportsBookReview (via ArnavSaraogi/mlb-odds-scraper)
===================================================================
Downloads the pre-scraped 76MB dataset from GitHub (2021-2025) and
converts it to a clean CSV with closing moneylines from all sportsbooks.

If you want to scrape fresh/updated data yourself:
    git clone https://github.com/ArnavSaraogi/mlb-odds-scraper
    cd mlb-odds-scraper
    pip install -r requirements.txt
    python scraper.py 2021-03-20 2025-12-31 -t moneyline -c 5 -o mlb_odds.json

This script:
    1. Downloads the pre-built dataset (or reads local mlb_odds.json)
    2. Extracts closing moneylines per game per sportsbook
    3. Computes consensus closing line (average across books)
    4. Computes no-vig implied probability
    5. Saves mlb_closing_odds.csv for use by mlb_model_v2.py

Usage:
    python mlb_convert_odds.py

Output:
    - mlb_closing_odds.csv
"""

import os
import json
import numpy as np
import pandas as pd


# Team name normalization (SBR uses full names / short codes)
TEAM_NORMALIZE = {
    "ARI": "ARI", "Arizona Diamondbacks": "ARI", "Arizona": "ARI",
    "ATL": "ATL", "Atlanta Braves": "ATL", "Atlanta": "ATL",
    "BAL": "BAL", "Baltimore Orioles": "BAL", "Baltimore": "BAL",
    "BOS": "BOS", "Boston Red Sox": "BOS", "Boston": "BOS",
    "CHC": "CHC", "Chicago Cubs": "CHC",
    "CWS": "CHW", "CHW": "CHW", "Chicago White Sox": "CHW",
    "CIN": "CIN", "Cincinnati Reds": "CIN", "Cincinnati": "CIN",
    "CLE": "CLE", "Cleveland Guardians": "CLE", "Cleveland Indians": "CLE", "Cleveland": "CLE",
    "COL": "COL", "Colorado Rockies": "COL", "Colorado": "COL",
    "DET": "DET", "Detroit Tigers": "DET", "Detroit": "DET",
    "HOU": "HOU", "Houston Astros": "HOU", "Houston": "HOU",
    "KC": "KCR", "KCR": "KCR", "Kansas City Royals": "KCR",
    "LAA": "LAA", "Los Angeles Angels": "LAA", "LA Angels": "LAA",
    "LAD": "LAD", "Los Angeles Dodgers": "LAD", "LA Dodgers": "LAD",
    "MIA": "MIA", "Miami Marlins": "MIA", "Miami": "MIA",
    "MIL": "MIL", "Milwaukee Brewers": "MIL", "Milwaukee": "MIL",
    "MIN": "MIN", "Minnesota Twins": "MIN", "Minnesota": "MIN",
    "NYM": "NYM", "New York Mets": "NYM",
    "NYY": "NYY", "New York Yankees": "NYY",
    "OAK": "OAK", "Oakland Athletics": "OAK", "Athletics": "OAK", "ATH": "OAK",
    "PHI": "PHI", "Philadelphia Phillies": "PHI", "Philadelphia": "PHI",
    "PIT": "PIT", "Pittsburgh Pirates": "PIT", "Pittsburgh": "PIT",
    "SD": "SDP", "SDP": "SDP", "San Diego Padres": "SDP",
    "SF": "SFG", "SFG": "SFG", "San Francisco Giants": "SFG",
    "SEA": "SEA", "Seattle Mariners": "SEA", "Seattle": "SEA",
    "STL": "STL", "St. Louis Cardinals": "STL",
    "TB": "TBR", "TBR": "TBR", "Tampa Bay Rays": "TBR",
    "TEX": "TEX", "Texas Rangers": "TEX", "Texas": "TEX",
    "TOR": "TOR", "Toronto Blue Jays": "TOR", "Toronto": "TOR",
    "WSH": "WSN", "WSN": "WSN", "Washington Nationals": "WSN",
}


def normalize_team(name):
    name = str(name).strip()
    return TEAM_NORMALIZE.get(name, name)


def american_to_implied_prob(ml):
    ml = float(ml)
    if ml > 0:
        return 100 / (ml + 100)
    elif ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return np.nan


def no_vig_prob(home_ml, away_ml):
    hp = american_to_implied_prob(home_ml)
    ap = american_to_implied_prob(away_ml)
    total = hp + ap
    if total > 0:
        return hp / total
    return np.nan


def load_json(path="mlb_odds.json"):
    """Load the SBR scraper JSON output."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print(f"\nTo get the data:")
        print(f"  Option A: Download the pre-scraped dataset from:")
        print(f"    https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/tag/dataset")
        print(f"    Unzip and place mlb_odds.json in this directory.")
        print(f"")
        print(f"  Option B: Scrape it yourself:")
        print(f"    git clone https://github.com/ArnavSaraogi/mlb-odds-scraper")
        print(f"    cd mlb-odds-scraper")
        print(f"    pip install -r requirements.txt")
        print(f"    python scraper.py 2021-03-20 2025-12-31 -t moneyline -c 5")
        print(f"    cp mlb_odds.json /path/to/your/model/dir/")
        return None

    print(f"Loading {path}...")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} dates")
    return data


def convert_to_csv(data):
    """Convert the nested JSON to a flat CSV with one row per game."""
    rows = []

    for date_str, games in data.items():
        for game in games:
            gv = game.get("gameView", {})
            odds_data = game.get("odds", {})
            ml_odds = odds_data.get("moneyline", [])

            # Parse game info
            home_team_raw = gv.get("homeTeam", {})
            away_team_raw = gv.get("awayTeam", {})

            home_team = normalize_team(
                home_team_raw.get("shortName", "") or home_team_raw.get("fullName", "")
            )
            away_team = normalize_team(
                away_team_raw.get("shortName", "") or away_team_raw.get("fullName", "")
            )

            home_score = gv.get("homeTeamScore")
            away_score = gv.get("awayTeamScore")
            status = gv.get("gameStatusText", "")
            game_type = gv.get("gameType", "")
            venue = gv.get("venueName", "")
            start_time = gv.get("startDate", "")

            # Skip non-regular season games
            if game_type and game_type not in ("R", ""):
                continue

            # Skip games that weren't completed
            if "Final" not in str(status) and "final" not in str(status).lower():
                continue

            # Determine winner
            if home_score is not None and away_score is not None:
                home_win = 1 if home_score > away_score else 0
            else:
                home_win = np.nan

            # Extract closing lines from each sportsbook
            book_lines = {}
            for book_entry in ml_odds:
                book_name = book_entry.get("sportsbook", "unknown")
                # "currentLine" is the closing line (last line before game start)
                current = book_entry.get("currentLine", {})
                opening = book_entry.get("openingLine", {})

                home_ml = current.get("homeOdds")
                away_ml = current.get("awayOdds")

                if home_ml is not None and away_ml is not None:
                    book_lines[book_name] = {
                        "home_ml": home_ml,
                        "away_ml": away_ml,
                    }

                    # Also store opening line
                    open_home = opening.get("homeOdds")
                    open_away = opening.get("awayOdds")
                    if open_home is not None and open_away is not None:
                        book_lines[f"{book_name}_open"] = {
                            "home_ml": open_home,
                            "away_ml": open_away,
                        }

            if not book_lines:
                continue

            # Compute consensus closing line (average no-vig prob across books)
            closing_probs = []
            for book, lines in book_lines.items():
                if "_open" in book:
                    continue  # Skip opening lines for consensus
                prob = no_vig_prob(lines["home_ml"], lines["away_ml"])
                if not np.isnan(prob):
                    closing_probs.append(prob)

            consensus_prob = np.mean(closing_probs) if closing_probs else np.nan

            # Build the row
            row = {
                "date": date_str,
                "start_time": start_time,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "venue": venue,
                "game_type": game_type,
                "market_prob_home": consensus_prob,
                "num_books": len([b for b in book_lines if "_open" not in b]),
            }

            # Add individual sportsbook lines
            for book, lines in book_lines.items():
                if "_open" not in book:
                    row[f"{book}_home_ml"] = lines["home_ml"]
                    row[f"{book}_away_ml"] = lines["away_ml"]
                    row[f"{book}_prob_home"] = no_vig_prob(lines["home_ml"], lines["away_ml"])

            rows.append(row)

    df = pd.DataFrame(rows)

    # Add season
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year
    # Games in March are sometimes the prior year's spring training
    # but SBR dataset should only have regular season

    # Sort
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("MLB Odds Converter (SBR Dataset -> CSV)")
    print("=" * 60)

    data = load_json()
    if data is None:
        exit(1)

    df = convert_to_csv(data)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total games: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Avg consensus probability: {df['market_prob_home'].mean():.3f}")
    print(f"Avg number of books per game: {df['num_books'].mean():.1f}")

    # Show which sportsbooks are available
    book_cols = [c for c in df.columns if c.endswith("_home_ml") and "_open" not in c]
    books = [c.replace("_home_ml", "") for c in book_cols]
    print(f"\nSportsbooks found: {', '.join(books)}")

    # Coverage by season
    print(f"\nGames per season:")
    for season, count in df.groupby("season").size().items():
        print(f"  {season}: {count} games")

    # Vig analysis
    for book in books[:5]:  # Top 5
        hml = df[f"{book}_home_ml"].dropna()
        aml = df[f"{book}_away_ml"].dropna()
        if len(hml) > 100:
            vigs = []
            for h, a in zip(hml, aml):
                hp = american_to_implied_prob(h)
                ap = american_to_implied_prob(a)
                vigs.append(hp + ap - 1)
            avg_vig = np.mean(vigs)
            print(f"  {book:15s}: {len(hml):5d} games, avg vig: {avg_vig*100:.1f}%")

    # Save
    df.to_csv("mlb_closing_odds.csv", index=False)
    print(f"\nSaved: mlb_closing_odds.csv ({len(df)} rows)")
    print(f"\nNext: python mlb_model_v2.py")