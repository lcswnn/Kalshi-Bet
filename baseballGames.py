"""
MLB Historical Game Data Collector for ML Prediction Models
============================================================
Collects game-by-game results for all 30 MLB teams across multiple seasons,
then engineers features useful for predicting game outcomes.

Pulls data from Baseball Reference via pybaseball.

Usage:
    python collect_mlb_data.py

Output:
    - mlb_raw_games.csv          (raw game logs, every game every team)
    - mlb_matchups_features.csv  (feature-engineered dataset ready for ML)
"""

import time
import warnings
import numpy as np
import pandas as pd
from pybaseball import schedule_and_record

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────
SEASONS = list(range(2019, 2026))  # 2019-2025
TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SFG",
    "SEA", "STL", "TBR", "TEX", "TOR", "WSN"
]

# If you already have raw data and just want to re-run feature engineering,
# set this to True and place mlb_raw_games.csv in the same directory.
SKIP_COLLECTION = False


# ── Step 1: Collect Raw Game Logs ──────────────────────────────────────────
def collect_all_games():
    """Pull schedule_and_record for every team/season combo."""
    all_games = []
    total = len(TEAMS) * len(SEASONS)
    count = 0

    for year in SEASONS:
        for team in TEAMS:
            count += 1
            print(f"[{count}/{total}] Fetching {team} {year}...")
            try:
                df = schedule_and_record(year, team)
                df["Team"] = team
                df["Season"] = year
                all_games.append(df)
                time.sleep(1.5)  # be polite to Baseball Reference
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
                time.sleep(3)

    raw = pd.concat(all_games, ignore_index=True)
    print(f"\nCollected {len(raw)} total game entries across {len(SEASONS)} seasons.")
    return raw


# ── Step 2: Clean & Normalize ──────────────────────────────────────────────
def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and types from the Baseball Reference format."""
    df = raw.copy()

    # Actual columns from Baseball Reference via pybaseball:
    # Date, Tm, Home_Away, Opp, W/L, R, RA, Inn, W-L, Rank, GB,
    # Win, Loss, Save, Time, D/N, Attendance, cLI, Streak, Orig. Scheduled, Team, Season

    rename_map = {
        "Date": "date_str",
        "Tm": "team_name",
        "Opp": "opponent_name",
        "R": "runs_scored",
        "RA": "runs_allowed",
        "W/L": "result",
        "Win": "winning_pitcher",
        "Loss": "losing_pitcher",
        "W-L": "record",
        "Rank": "rank",
        "GB": "games_back",
        "Inn": "innings",
        "Streak": "streak",
        "D/N": "day_night",
        "cLI": "leverage_index",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Home/Away
    if "Home_Away" in df.columns:
        df["is_home"] = df["Home_Away"].apply(lambda x: 0 if str(x).strip() == "@" else 1)
    else:
        df["is_home"] = np.nan

    # Parse W/L into binary
    if "result" in df.columns:
        df["win"] = df["result"].apply(
            lambda x: 1 if str(x).startswith("W") else (0 if str(x).startswith("L") else np.nan)
        )

    # Numeric columns
    for col in ["runs_scored", "runs_allowed", "innings", "leverage_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse record into wins/losses
    if "record" in df.columns:
        split = df["record"].str.split("-", expand=True)
        if split.shape[1] >= 2:
            df["season_wins"] = pd.to_numeric(split[0], errors="coerce")
            df["season_losses"] = pd.to_numeric(split[1], errors="coerce")

    # Parse streak into numeric — NOTE: BB-Ref streak includes current game,
    # so we shift it by 1 to avoid leakage (pre-game streak only)
    if "streak" in df.columns:
        def parse_streak(s):
            s = str(s).strip()
            try:
                return int(s)
            except ValueError:
                return 0
        df["streak_num_raw"] = df["streak"].apply(parse_streak)

    # Day/Night binary
    if "day_night" in df.columns:
        df["is_night"] = (df["day_night"] == "N").astype(int)

    # Extra innings flag
    if "innings" in df.columns:
        df["extra_innings"] = (df["innings"] > 9).astype(int)

    # Parse date_str into proper YYYY-MM-DD game_date
    # Format: "Thursday, Mar 28" or "Saturday, Apr 20 (1)" (doubleheader)
    import re
    def parse_game_date(date_str, season):
        """Parse Baseball Reference date string into YYYY-MM-DD."""
        if pd.isna(date_str):
            return None
        s = str(date_str).strip()
        # Remove doubleheader suffix like " (1)" or " (2)"
        s = re.sub(r'\s*\(\d+\)\s*$', '', s)
        # Remove day-of-week prefix: "Thursday, Mar 28" -> "Mar 28"
        if ',' in s:
            s = s.split(',', 1)[1].strip()
        try:
            parsed = pd.to_datetime(f"{s} {int(season)}", format="%b %d %Y")
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return None

    if "date_str" in df.columns and "Season" in df.columns:
        df["game_date"] = df.apply(lambda r: parse_game_date(r["date_str"], r["Season"]), axis=1)

    # Drop postponed / non-game rows
    df = df.dropna(subset=["win"])

    # Game number within season
    df["game_num"] = df.groupby(["Team", "Season"]).cumcount() + 1

    return df


# ── Step 3: Feature Engineering ────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling/contextual features using ONLY past data (no leakage).
    """
    df = df.sort_values(["Team", "Season", "game_num"]).reset_index(drop=True)

    windows = [5, 10, 20, 40]
    grp = df.groupby(["Team", "Season"])

    for w in windows:
        df[f"win_pct_L{w}"] = grp["win"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
        df[f"avg_rs_L{w}"] = grp["runs_scored"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
        df[f"avg_ra_L{w}"] = grp["runs_allowed"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
        df[f"run_diff_L{w}"] = df[f"avg_rs_L{w}"] - df[f"avg_ra_L{w}"]

    # Pythagorean win expectation (Bill James) over last 40
    rs40 = df["avg_rs_L40"] * 40
    ra40 = df["avg_ra_L40"] * 40
    df["pythag_win_exp"] = (rs40 ** 1.83) / (rs40 ** 1.83 + ra40 ** 1.83)

    # Season-to-date win %
    df["season_win_pct"] = grp["win"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Season-to-date run differential per game
    df["_rd"] = df["runs_scored"] - df["runs_allowed"]
    df["season_run_diff"] = grp["_rd"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df.drop(columns=["_rd"], inplace=True)

    # Season progress
    df["season_progress"] = df["game_num"] / 162.0

    # Streak: shift by 1 so we only see pre-game streak (no leakage)
    if "streak_num_raw" in df.columns:
        df["streak_num"] = grp["streak_num_raw"].transform(lambda s: s.shift(1))
        df["streak_num"] = df["streak_num"].fillna(0)

    # Rolling extra innings rate (fatigue proxy)
    if "extra_innings" in df.columns:
        df["extra_inn_rate_L10"] = grp["extra_innings"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )

    # Rolling leverage index
    if "leverage_index" in df.columns:
        df["avg_cli_L10"] = grp["leverage_index"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )

    return df


# ── Step 4: Build Matchup Rows ────────────────────────────────────────────

NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Athletics": "OAK", "ATH": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
    # Abbreviation passthrough
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SFG": "SFG",
    "SEA": "SEA", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
}


def build_matchup_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game from the HOME team's perspective,
    with away team's rolling features merged in.
    """
    # Feature columns (rolling stats, NOT identifiers like game_num)
    feature_cols = [c for c in df.columns if any(
        c.startswith(p) for p in [
            "win_pct_L", "avg_rs_L", "avg_ra_L", "run_diff_L",
            "pythag_win_exp", "season_win_pct", "season_run_diff",
            "streak_num", "season_progress", "extra_inn_rate",
            "avg_cli_L",
        ]
    )]

    # Build fast lookup: (Team, Season, game_num) -> features dict
    lookup = {}
    for _, row in df.iterrows():
        key = (row["Team"], int(row["Season"]), int(row["game_num"]))
        lookup[key] = {fc: row[fc] for fc in feature_cols}

    # Build name -> abbreviation from data + known mappings
    name_to_abbr = dict(NAME_TO_ABBR)
    for _, row in df[["team_name", "Team"]].drop_duplicates().iterrows():
        if pd.notna(row["team_name"]):
            name_to_abbr[str(row["team_name"]).strip()] = row["Team"]

    # Only home games
    home_games = df[df["is_home"] == 1].copy()
    print(f"\nBuilding matchup rows for {len(home_games)} home games...")

    rows = []
    missed = set()

    for _, row in home_games.iterrows():
        opp_raw = str(row["opponent_name"]).strip()
        opp_abbr = name_to_abbr.get(opp_raw)

        if not opp_abbr:
            missed.add(opp_raw)
            continue

        season = int(row["Season"])
        gn = int(row["game_num"])

        # Base columns
        r = {
            "season": season,
            "home_team": row["Team"],
            "away_team": opp_abbr,
            "home_win": row["win"],
            "home_runs": row["runs_scored"],
            "away_runs": row["runs_allowed"],
            "game_num": gn,
            "date_str": row.get("date_str", ""),
            "game_date": row.get("game_date", ""),
            "is_night": row.get("is_night", np.nan),
        }

        # Home features
        for fc in feature_cols:
            r[f"home_{fc}"] = row[fc]

        # Away features: find opponent's most recent game in same season
        opp_found = False
        for og in range(gn, 0, -1):
            candidate = (opp_abbr, season, og)
            if candidate in lookup:
                for fc in feature_cols:
                    r[f"away_{fc}"] = lookup[candidate].get(fc, np.nan)
                opp_found = True
                break

        if not opp_found:
            for fc in feature_cols:
                r[f"away_{fc}"] = np.nan

        rows.append(r)

    if missed:
        print(f"  ⚠ Could not map {len(missed)} opponent names: {missed}")

    matchups = pd.DataFrame(rows)
    print(f"Built {len(matchups)} matchup rows.")
    return matchups


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MLB Historical Data Collector")
    print(f"Seasons: {SEASONS[0]}-{SEASONS[-1]} | Teams: {len(TEAMS)}")
    print("=" * 60)

    if SKIP_COLLECTION:
        print("Loading existing mlb_raw_games.csv...")
        raw = pd.read_csv("mlb_raw_games.csv")
    else:
        raw = collect_all_games()
        raw.to_csv("mlb_raw_games.csv", index=False)
        print("✓ Saved mlb_raw_games.csv")

    # Clean
    clean = clean_raw(raw)
    print(f"✓ Cleaned: {len(clean)} games")

    # Features
    featured = engineer_features(clean)
    print(f"✓ Engineered features")

    # Matchups
    matchups = build_matchup_dataset(featured)
    matchups.to_csv("mlb_matchups_features.csv", index=False)
    print("✓ Saved mlb_matchups_features.csv")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total matchup rows:  {len(matchups)}")
    feat_count = len([c for c in matchups.columns
                      if c.startswith(("home_", "away_"))
                      and c not in ["home_team", "away_team", "home_win", "home_runs", "away_runs"]])
    print(f"Feature columns:     {feat_count}")
    print(f"Seasons covered:     {sorted(matchups['season'].unique())}")
    print(f"Home win rate:       {matchups['home_win'].mean():.3f}")
    print(f"\nTarget variable:     'home_win' (1 = home team won)")
    print(f"\nSample features (home side):")
    sample = [c for c in matchups.columns if c.startswith("home_") and "team" not in c][:10]
    for c in sample:
        print(f"  {c}")
    print(f"\nNull % (top 5):")
    nulls = matchups.isnull().mean().sort_values(ascending=False).head(5)
    for col, pct in nulls.items():
        print(f"  {col}: {pct:.1%}")
    print("\nReady for ML training! 🚀")