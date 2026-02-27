"""
Kalshi MLB Live Predictor
=========================
Fetches current MLB games from Kalshi API and generates betting recommendations
using the pre-trained MLB model (mlb_model.py).

Usage:
    python kalshi_mlb_predictor.py --kelly 0.5 --bankroll 100
"""

import argparse
import json
import os
import sys
import pickle
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

EASTERN = ZoneInfo("America/New_York")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "mlb_historical_data")

# Kalshi API endpoints
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_MARKETS_ENDPOINT = f"{KALSHI_API_BASE}/markets"


# ═══════════════════════════════════════════════════════════════════════════
# ELO SYSTEM (from mlb_model.py)
# ═══════════════════════════════════════════════════════════════════════════

class EloSystem:
    """MLB Elo with staggered K-factor."""

    # Spring training: reduced K and home advantage (rotated lineups, neutral-ish sites)
    SPRING_TRAINING_K = 8
    SPRING_TRAINING_HOME_ADV = 12  # Half of regular season

    def __init__(self, home_advantage=24, mean_rating=1500, season_reversion=0.33):
        self.home_adv = home_advantage
        self.mean = mean_rating
        self.reversion = season_reversion
        self.ratings = {}
        self.last_season = None

    def _k_factor(self, rating):
        if rating < 1450:
            return 32
        elif rating <= 1550:
            return 24
        else:
            return 16

    def get_rating(self, team):
        return self.ratings.get(team, self.mean)

    def expected_score(self, home_elo, away_elo):
        diff = home_elo + self.home_adv - away_elo
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update_rating(self, home_team, away_team, home_won, spring_training=False):
        """Update Elo ratings after a game result.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_won: True if home team won
            spring_training: Use reduced K-factor for spring training
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        if spring_training:
            k_home = self.SPRING_TRAINING_K
            k_away = self.SPRING_TRAINING_K
            home_adv = self.SPRING_TRAINING_HOME_ADV
        else:
            k_home = self._k_factor(home_elo)
            k_away = self._k_factor(away_elo)
            home_adv = self.home_adv

        expected_home = 1.0 / (1.0 + 10 ** (-(home_elo + home_adv - away_elo) / 400.0))
        actual_home = 1.0 if home_won else 0.0

        self.ratings[home_team] = home_elo + k_home * (actual_home - expected_home)
        self.ratings[away_team] = away_elo + k_away * (expected_home - actual_home)


# ═══════════════════════════════════════════════════════════════════════════
# SPRING TRAINING RESULTS (MLB Stats API)
# ═══════════════════════════════════════════════════════════════════════════

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"

# MLB Stats API team ID -> standard abbreviation
MLB_TEAM_ID_MAP = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}


def fetch_spring_training_results(quiet=False):
    """Fetch completed spring training game results from MLB Stats API.

    Returns list of dicts with: date, home_abbr, away_abbr, home_score, away_score, split_squad
    """
    # Spring training window: Feb 20 through today
    eastern_now = datetime.now(EASTERN)
    start_date = f"{eastern_now.year}-02-20"
    end_date = eastern_now.strftime("%Y-%m-%d")

    if not quiet:
        print(f"\nFetching spring training results ({start_date} to {end_date})...")

    try:
        params = {
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "gameType": "S"
        }
        response = requests.get(f"{MLB_STATS_API}/schedule", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        games = []
        for date_obj in data.get("dates", []):
            for game in date_obj.get("games", []):
                # Only completed games
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue

                away = game.get("teams", {}).get("away", {})
                home = game.get("teams", {}).get("home", {})

                away_id = away.get("team", {}).get("id")
                home_id = home.get("team", {}).get("id")

                away_abbr = MLB_TEAM_ID_MAP.get(away_id)
                home_abbr = MLB_TEAM_ID_MAP.get(home_id)

                if not away_abbr or not home_abbr:
                    continue

                home_score = home.get("score", 0) or 0
                away_score = away.get("score", 0) or 0

                split_squad = away.get("splitSquad", False) or home.get("splitSquad", False)

                games.append({
                    "date": game.get("officialDate", ""),
                    "home_abbr": home_abbr,
                    "away_abbr": away_abbr,
                    "home_score": home_score,
                    "away_score": away_score,
                    "split_squad": split_squad,
                })

        # Sort by date
        games.sort(key=lambda g: g["date"])

        if not quiet:
            print(f"  Found {len(games)} completed spring training games")
            split_count = sum(1 for g in games if g["split_squad"])
            if split_count:
                print(f"  ({split_count} split-squad games)")

        return games

    except requests.exceptions.RequestException as e:
        if not quiet:
            print(f"  Failed to fetch spring training results: {e}")
        return []


def update_elo_with_spring_training(elo_system, quiet=False):
    """Fetch spring training results and update Elo ratings."""
    games = fetch_spring_training_results(quiet=quiet)

    if not games:
        return 0

    updated = 0
    for game in games:
        home_won = game["home_score"] > game["away_score"]
        # Ties in spring training: skip (no meaningful update)
        if game["home_score"] == game["away_score"]:
            continue

        elo_system.update_rating(
            game["home_abbr"],
            game["away_abbr"],
            home_won,
            spring_training=True
        )
        updated += 1

    if not quiet:
        print(f"  Updated Elo with {updated} spring training results")

    return updated


# ═══════════════════════════════════════════════════════════════════════════
# LOAD TRAINED MODEL AND ELO RATINGS
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_elo(quiet=False):
    """Load the trained ensemble model and current Elo ratings."""
    
    # Try multiple sources for Elo ratings
    if not quiet:
        print("Loading Elo ratings...")
    
    # First try: elo_features.csv (generated by mlb_model.py)
    elo_path = os.path.join(DATA_DIR, "elo_features.csv")
    
    if os.path.exists(elo_path):
        if not quiet:
            print(f"  Loading from {elo_path}")
        elo_df = pd.read_csv(elo_path)
        
        # Get the most recent Elo ratings for each team
        latest_elo = {}
        for team in elo_df['home_team'].unique():
            home_games = elo_df[elo_df['home_team'] == team]
            away_games = elo_df[elo_df['away_team'] == team]
            
            if not home_games.empty:
                latest_home = home_games.iloc[-1]['home_elo']
                latest_elo[team] = latest_home
            elif not away_games.empty:
                latest_away = away_games.iloc[-1]['away_elo']
                latest_elo[team] = latest_away
    
    else:
        # Fallback: Try mlb_matchups_features_advanced.csv
        advanced_path = os.path.join(DATA_DIR, "mlb_matchups_features_advanced.csv")
        
        if os.path.exists(advanced_path):
            if not quiet:
                print(f"  elo_features.csv not found, loading from {advanced_path}")
            df = pd.read_csv(advanced_path)
            
            # Convert game_date to datetime and sort chronologically
            df['game_date'] = pd.to_datetime(df['game_date'])
            df = df.sort_values('game_date')
            
            # Get the most recent Elo for each team
            latest_elo = {}
            for team in df['home_team'].unique():
                home_games = df[df['home_team'] == team]
                away_games = df[df['away_team'] == team]
                
                if not home_games.empty and pd.notna(home_games.iloc[-1]['home_elo']):
                    latest_elo[team] = home_games.iloc[-1]['home_elo']
                elif not away_games.empty and pd.notna(away_games.iloc[-1]['away_elo']):
                    latest_elo[team] = away_games.iloc[-1]['away_elo']
        else:
            print(f"ERROR: Neither {elo_path} nor {advanced_path} found.")
            print("You need to either:")
            print("  1. Run mlb_model.py to generate elo_features.csv, OR")
            print("  2. Have mlb_matchups_features_advanced.csv with home_elo/away_elo columns")
            sys.exit(1)
    
    # Initialize Elo system with current ratings
    elo_system = EloSystem()
    elo_system.ratings = latest_elo

    if not quiet:
        print(f"  Loaded base Elo ratings for {len(latest_elo)} teams")

    # Update with spring training results
    st_count = update_elo_with_spring_training(elo_system, quiet=quiet)

    if not quiet:
        sorted_teams = sorted(elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
        label = "Top 3 teams (with ST updates)" if st_count else "Top 3 teams"
        print(f"  {label}: {', '.join([f'{t[0]}={t[1]:.0f}' for t in sorted_teams[:3]])}")
    
    # Try to load saved model if it exists
    model_path = os.path.join(BASE_DIR, "mlb_ensemble_model.pkl")
    
    if os.path.exists(model_path):
        if not quiet:
            print(f"Loading saved model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return elo_system, model_data
    else:
        if not quiet:
            print("No saved model found. Using Elo-only predictions.")
        return elo_system, None


# ═══════════════════════════════════════════════════════════════════════════
# FETCH KALSHI MARKETS
# ═══════════════════════════════════════════════════════════════════════════

def get_market_price(market):
    """Get best available price for a market: last_price if traded, else midpoint."""
    volume = market.get("volume", 0)
    last_price = market.get("last_price", 0)
    yes_bid = market.get("yes_bid", 0)
    yes_ask = market.get("yes_ask", 100)

    if volume and volume > 0 and last_price and last_price > 0:
        return last_price
    if yes_bid and yes_ask and yes_bid > 0:
        return (yes_bid + yes_ask) // 2
    return yes_ask if yes_ask else 50


def fetch_kalshi_mlb_markets(quiet=False, target_date=None):
    """Fetch current MLB game markets from Kalshi API.

    Args:
        quiet: Suppress print output
        target_date: Optional date object to filter games (by expected_expiration_time)
    """

    if not quiet:
        print("\nFetching Kalshi MLB markets...")
        if target_date:
            print(f"  Filtering for date: {target_date}")

    # Fetch from both regular season and spring training series
    all_markets = []
    series_tickers = ["KXMLB", "KXMLBSTGAME"]

    for series in series_tickers:
        try:
            params = {
                "series_ticker": series,
                "status": "open",
                "limit": 200
            }
            response = requests.get(KALSHI_MARKETS_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json().get("markets", [])
            if not quiet:
                print(f"  {series}: {len(markets)} open markets")
            all_markets.extend(markets)
        except requests.exceptions.RequestException as e:
            if not quiet:
                print(f"  {series}: fetch failed ({e})")

    if not all_markets:
        if not quiet:
            print("  No markets found from any series.")
        return []

    # Group all markets by event_ticker
    events = defaultdict(list)
    for market in all_markets:
        events[market.get("event_ticker", "")].append(market)

    games = []
    skipped_futures = 0

    for event_ticker, market_group in events.items():
        # Skip championship/futures markets
        sample_title = market_group[0].get("title", "")
        if "Championship" in sample_title or "win the 20" in sample_title:
            skipped_futures += len(market_group)
            continue

        # ── Spring training paired markets ("Team A vs Team B Winner?") ──
        if "KXMLBSTGAME" in event_ticker and len(market_group) == 2:
            # Extract team abbreviations from ticker suffixes
            team_markets = {}
            for m in market_group:
                ticker = m.get("ticker", "")
                team_suffix = ticker.rsplit("-", 1)[-1]
                std_abbr = normalize_ticker_abbr(team_suffix)
                team_markets[std_abbr] = m

            if len(team_markets) != 2:
                continue

            teams = list(team_markets.keys())

            # Determine home/away from event_ticker
            # Event ticker ends with concatenated team abbrs (e.g., ...ATLMIN)
            # First team = away, second team = home (baseball convention)
            # Find which ordering matches the event_ticker suffix
            team_a, team_b = teams[0], teams[1]
            # Check original ticker suffixes (before normalization) for matching
            raw_suffixes = []
            for m in market_group:
                raw_suffixes.append(m.get("ticker", "").rsplit("-", 1)[-1])

            # Try both orderings against event_ticker
            suffix_concat_ab = raw_suffixes[0] + raw_suffixes[1]
            suffix_concat_ba = raw_suffixes[1] + raw_suffixes[0]
            event_end = event_ticker.split("-")[1] if "-" in event_ticker else ""

            if suffix_concat_ab in event_end:
                away_abbr, home_abbr = team_a, team_b
            elif suffix_concat_ba in event_end:
                away_abbr, home_abbr = team_b, team_a
            else:
                # Can't determine order; just pick one
                home_abbr, away_abbr = team_a, team_b

            home_market = team_markets[home_abbr]
            away_market = team_markets[away_abbr]

            home_name = ABBR_TO_FULL_NAME.get(home_abbr, home_abbr)
            away_name = ABBR_TO_FULL_NAME.get(away_abbr, away_abbr)

            # Date filtering using expected_expiration_time
            exp_time = home_market.get("expected_expiration_time", "")
            if target_date and exp_time:
                try:
                    game_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00"))
                    game_date = game_dt.astimezone(EASTERN).date()
                    if game_date != target_date:
                        continue
                except (ValueError, TypeError):
                    pass  # Can't parse date; include the game

            # Use the home team's market ticker for the bet
            # yes_price = cost to buy "home wins", no_price = cost to buy "away wins"
            home_price = get_market_price(home_market)
            away_price = get_market_price(away_market)

            # Volume = sum of both sides
            home_vol = home_market.get("volume", 0) or 0
            away_vol = away_market.get("volume", 0) or 0
            total_volume = home_vol + away_vol

            games.append({
                "ticker": home_market.get("ticker", ""),
                "ticker_away": away_market.get("ticker", ""),
                "home_team": home_name,
                "away_team": away_name,
                "yes_price": home_price,
                "no_price": away_price,
                "close_time": home_market.get("close_time", ""),
                "expected_expiration_time": exp_time,
                "title": sample_title,
                "volume": total_volume,
                "source": "spring_training"
            })

        # ── Regular season single markets ("Will X beat Y?") ──
        else:
            for market in market_group:
                title = market.get("title", "")
                subtitle = market.get("subtitle", "")

                home_team = None
                away_team = None

                # Pattern: "Will (the) [Team] beat (the) [Opponent]?"
                if " beat " in title.lower():
                    cleaned = title.replace("Will the ", "Will ").replace(" beat the ", " beat ")
                    if "Will " in cleaned and " beat " in cleaned:
                        try:
                            parts = cleaned.replace("Will ", "").split(" beat ")
                            if len(parts) == 2:
                                home_team = parts[0].strip()
                                away_team = parts[1].strip().rstrip("?")
                        except:
                            pass

                # Fallback: subtitle parsing
                if not home_team and subtitle:
                    for sep in [" @ ", " vs "]:
                        if sep in subtitle:
                            parts = subtitle.split(sep)
                            if len(parts) == 2:
                                if sep == " @ ":
                                    away_team, home_team = parts[0].strip(), parts[1].strip()
                                else:
                                    home_team, away_team = parts[0].strip(), parts[1].strip()
                            break

                if home_team and away_team:
                    # Date filtering
                    exp_time = market.get("expected_expiration_time", "")
                    if target_date and exp_time:
                        try:
                            game_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00"))
                            game_date = game_dt.astimezone(EASTERN).date()
                            if game_date != target_date:
                                continue
                        except (ValueError, TypeError):
                            pass

                    games.append({
                        "ticker": market.get("ticker", ""),
                        "home_team": home_team,
                        "away_team": away_team,
                        "yes_price": market.get("yes_ask", 50),
                        "no_price": market.get("no_ask", 50),
                        "close_time": market.get("close_time", ""),
                        "expected_expiration_time": exp_time,
                        "title": title,
                        "volume": market.get("volume", 0) or 0,
                        "source": "regular_season"
                    })

    if not quiet:
        print(f"  Skipped {skipped_futures} championship/futures markets")
        print(f"  Parsed {len(games)} MLB games")
        if games:
            sources = defaultdict(int)
            for g in games:
                sources[g.get("source", "unknown")] += 1
            for src, count in sources.items():
                print(f"    - {src}: {count} games")

    return games


# ═══════════════════════════════════════════════════════════════════════════
# TEAM NAME MAPPING
# ═══════════════════════════════════════════════════════════════════════════

# Map Kalshi team names (nicknames) to standard abbreviations
TEAM_NAME_MAP = {
    "Athletics": "OAK", "A's": "OAK",
    "Angels": "LAA",
    "Astros": "HOU",
    "Blue Jays": "TOR",
    "Braves": "ATL",
    "Brewers": "MIL",
    "Cardinals": "STL",
    "Cubs": "CHC",
    "Diamondbacks": "ARI", "D-backs": "ARI",
    "Dodgers": "LAD",
    "Giants": "SF",
    "Guardians": "CLE",
    "Mariners": "SEA",
    "Marlins": "MIA",
    "Mets": "NYM",
    "Nationals": "WSH",
    "Orioles": "BAL",
    "Padres": "SD",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Rangers": "TEX",
    "Rays": "TB",
    "Red Sox": "BOS",
    "Reds": "CIN",
    "Rockies": "COL",
    "Royals": "KC",
    "Tigers": "DET",
    "Twins": "MIN",
    "White Sox": "CHW",
    "Yankees": "NYY"
}

# Map ticker abbreviations (used in KXMLBSTGAME tickers) to standard abbreviations
TICKER_ABBR_MAP = {
    "ATH": "OAK",
    "ARI": "ARI", "AZ": "ARI",
    "CHC": "CHC", "CHW": "CHW", "CWS": "CHW",
    "KC": "KC", "KCR": "KC",
    "LAA": "LAA", "LAD": "LAD",
    "SD": "SD", "SDP": "SD",
    "SF": "SF", "SFG": "SF",
    "TB": "TB", "TBR": "TB",
    "WSH": "WSH", "WAS": "WSH",
}

# Standard abbreviation to full team name
ABBR_TO_FULL_NAME = {
    "OAK": "Athletics", "LAA": "Angels", "HOU": "Astros", "TOR": "Blue Jays",
    "ATL": "Braves", "MIL": "Brewers", "STL": "Cardinals", "CHC": "Cubs",
    "ARI": "Diamondbacks", "LAD": "Dodgers", "SF": "Giants", "CLE": "Guardians",
    "SEA": "Mariners", "MIA": "Marlins", "NYM": "Mets", "WSH": "Nationals",
    "BAL": "Orioles", "SD": "Padres", "PHI": "Phillies", "PIT": "Pirates",
    "TEX": "Rangers", "TB": "Rays", "BOS": "Red Sox", "CIN": "Reds",
    "COL": "Rockies", "KC": "Royals", "DET": "Tigers", "MIN": "Twins",
    "CHW": "White Sox", "NYY": "Yankees",
}

def normalize_ticker_abbr(ticker_abbr):
    """Convert ticker abbreviation to standard MLB abbreviation."""
    return TICKER_ABBR_MAP.get(ticker_abbr, ticker_abbr)

def map_team_name(kalshi_name):
    """Map Kalshi team name to abbreviation."""
    for key, abbr in TEAM_NAME_MAP.items():
        if key.lower() in kalshi_name.lower():
            return abbr
    return kalshi_name  # Return as-is if no mapping found


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_predictions(games, elo_system, model_data=None, quiet=False):
    """Generate predictions for each game using Elo and/or trained model."""
    
    if not quiet:
        print("\nGenerating predictions...")
    
    predictions = []
    
    for game in games:
        home_abbr = map_team_name(game["home_team"])
        away_abbr = map_team_name(game["away_team"])
        
        # Get Elo ratings
        home_elo = elo_system.get_rating(home_abbr)
        away_elo = elo_system.get_rating(away_abbr)
        
        # Calculate Elo-based probability
        elo_prob = elo_system.expected_score(home_elo, away_elo)
        
        # For now, use Elo probability as the model probability
        # In future versions, this would use the full ensemble model
        model_prob = elo_prob
        
        # Calculate implied probabilities from market prices
        yes_implied = game["yes_price"] / 100.0
        no_implied = game["no_price"] / 100.0
        
        # Calculate edges
        home_edge = model_prob - yes_implied
        away_edge = (1 - model_prob) - no_implied
        
        predictions.append({
            "ticker": game["ticker"],
            "ticker_away": game.get("ticker_away", game["ticker"]),
            "title": game["title"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "model_prob_home": round(model_prob * 100, 2),
            "yes_price": game["yes_price"],
            "no_price": game["no_price"],
            "yes_implied": round(yes_implied * 100, 2),
            "no_implied": round(no_implied * 100, 2),
            "home_edge": round(home_edge * 100, 2),
            "away_edge": round(away_edge * 100, 2),
            "close_time": game["close_time"],
            "volume": game.get("volume", 0)
        })
    
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# BETTING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_betting_recommendations(predictions, bankroll, kelly_fraction, min_edge=5.0, quiet=False):
    """
    Generate betting recommendations based on predictions.
    
    Args:
        predictions: List of prediction dictionaries
        bankroll: Starting bankroll amount
        kelly_fraction: Fraction of Kelly criterion to use (0-1)
        min_edge: Minimum edge percentage to trigger a bet
        quiet: If True, suppress print statements
    
    Returns:
        List of betting recommendations sorted by edge
    """
    
    if not quiet:
        print(f"\nGenerating betting recommendations (min edge: {min_edge}%)...")
    
    bets = []
    
    for pred in predictions:
        home_edge = pred["home_edge"]
        away_edge = pred["away_edge"]
        
        # Only bet on the side with the larger edge, and only if > min_edge
        if home_edge > away_edge and home_edge > min_edge:
            # Bet on home team
            win_prob = pred["model_prob_home"] / 100.0
            market_price = pred["yes_price"] / 100.0
            
            # Kelly criterion: f = (p * (b + 1) - 1) / b
            # where b = (1 - market_price) / market_price (decimal odds - 1)
            b = (1 - market_price) / market_price
            kelly_fraction_optimal = (win_prob * (b + 1) - 1) / b
            
            # Apply Kelly fraction multiplier and ensure non-negative
            bet_fraction = max(0, kelly_fraction_optimal * kelly_fraction)
            bet_amount = min(bet_fraction * bankroll, bankroll * 0.20)  # Cap at 20% of bankroll
            
            if bet_amount > 1.0:  # Only recommend bets > $1
                potential_profit = bet_amount * b
                expected_value = bet_amount * home_edge / 100.0
                
                bets.append({
                    "bet_on": pred["home_team"],
                    "bet_abbr": pred["home_abbr"],
                    "opponent": pred["away_team"],
                    "opponent_abbr": pred["away_abbr"],
                    "home_away": "Home",
                    "elo_team": pred["home_elo"],
                    "elo_opponent": pred["away_elo"],
                    "model_prob": pred["model_prob_home"],
                    "market_price": pred["yes_price"],
                    "market_implied": pred["yes_implied"],
                    "edge": home_edge,
                    "bet_amount": round(bet_amount, 2),
                    "potential_profit": round(potential_profit, 2),
                    "expected_value": round(expected_value, 2),
                    "ticker": pred["ticker"],
                    "title": pred["title"],
                    "close_time": pred["close_time"],
                    "volume": pred.get("volume", 0)
                })

        elif away_edge > home_edge and away_edge > min_edge:
            # Bet on away team
            win_prob = (100 - pred["model_prob_home"]) / 100.0
            market_price = pred["no_price"] / 100.0
            
            b = (1 - market_price) / market_price
            kelly_fraction_optimal = (win_prob * (b + 1) - 1) / b
            
            bet_fraction = max(0, kelly_fraction_optimal * kelly_fraction)
            bet_amount = min(bet_fraction * bankroll, bankroll * 0.20)
            
            if bet_amount > 1.0:
                potential_profit = bet_amount * b
                expected_value = bet_amount * away_edge / 100.0
                
                bets.append({
                    "bet_on": pred["away_team"],
                    "bet_abbr": pred["away_abbr"],
                    "opponent": pred["home_team"],
                    "opponent_abbr": pred["home_abbr"],
                    "home_away": "Away",
                    "elo_team": pred["away_elo"],
                    "elo_opponent": pred["home_elo"],
                    "model_prob": round(100 - pred["model_prob_home"], 2),
                    "market_price": pred["no_price"],
                    "market_implied": pred["no_implied"],
                    "edge": away_edge,
                    "bet_amount": round(bet_amount, 2),
                    "potential_profit": round(potential_profit, 2),
                    "expected_value": round(expected_value, 2),
                    "ticker": pred.get("ticker_away", pred["ticker"]),
                    "title": pred["title"],
                    "close_time": pred["close_time"],
                    "volume": pred.get("volume", 0)
                })
    
    # Sort by edge (highest first)
    bets.sort(key=lambda x: x["edge"], reverse=True)
    
    if not quiet:
        print(f"  Found {len(bets)} betting opportunities")
    
    return bets


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def format_close_time(iso_time):
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        return dt.strftime("%I:%M %p ET")
    except:
        return "Unknown"


def print_recommendations(bets, bankroll, elo_system):
    """Print betting recommendations in formatted output."""
    
    print("\n" + "=" * 70)
    print("KALSHI MLB BETTING RECOMMENDATIONS")
    print("=" * 70)
    
    # Print Elo rankings
    print("\nTOP 10 TEAMS BY ELO RATING:")
    print("-" * 70)
    
    sorted_teams = sorted(elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (team, rating) in enumerate(sorted_teams[:10], 1):
        print(f"{i:2d}. {team:5s} | Rating: {rating:7.2f}")
    
    print("\n" + "=" * 70)
    
    if not bets:
        print("NO BETTING OPPORTUNITIES FOUND")
        print("\nThe model found no edges meeting minimum requirements.")
        print("This could mean:")
        print("  • No games available on Kalshi today")
        print("  • Market prices are too efficient")
        print("  • No significant edge detected")
        print("=" * 70)
        return
    
    print(f"FOUND {len(bets)} BETTING OPPORTUNITIES")
    print("=" * 70)
    
    total_wager = sum(bet["bet_amount"] for bet in bets)
    total_profit = sum(bet["potential_profit"] for bet in bets)
    total_ev = sum(bet["expected_value"] for bet in bets)
    
    for i, bet in enumerate(bets, 1):
        print(f"\nBET #{i}: {bet['bet_on']} vs {bet['opponent']}")
        print("-" * 70)
        
        home_away_icon = "🏠" if bet["home_away"] == "Home" else "✈️"
        print(f"BET ON: {bet['bet_on']} ({bet['bet_abbr']}) {home_away_icon}")
        print(f"OPPONENT: {bet['opponent']} ({bet['opponent_abbr']})")
        print(f"  Elo: {bet['elo_team']:.1f} vs {bet['elo_opponent']:.1f}")
        print(f"  Model probability: {bet['model_prob']:.2f}%")
        print(f"  Market price: {bet['market_price']:.0f}¢ (implies {bet['market_implied']:.2f}%)")
        print(f"  Edge: {bet['edge']:+.2f}%")
        print(f"  BET: ${bet['bet_amount']:.2f}")
        print(f"  Potential profit: ${bet['potential_profit']:.2f}")
        print(f"  Expected value: ${bet['expected_value']:.2f}")
        print(f"  Ticker: {bet['ticker']}")
        print(f"  Game time: {format_close_time(bet['close_time'])}")
    
    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"TOTAL WAGER: ${total_wager:.2f} ({total_wager/bankroll*100:.1f}% of bankroll)")
    print(f"POTENTIAL PROFIT: ${total_profit:.2f}")
    print(f"EXPECTED VALUE: ${total_ev:.2f}")
    print(f"BANKROLL REMAINING: ${bankroll - total_wager:.2f}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# JSON OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def get_json_output(predictions, bets, bankroll, elo_system):
    """Return results as a JSON-serializable dict."""
    sorted_teams = sorted(elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
    rankings = [
        {"rank": i + 1, "team": team, "rating": round(rating, 2)}
        for i, (team, rating) in enumerate(sorted_teams[:10])
    ]
    
    total_wager = sum(bet["bet_amount"] for bet in bets)
    total_profit = sum(bet["potential_profit"] for bet in bets)
    total_ev = sum(bet["expected_value"] for bet in bets)
    
    return {
        "rankings": rankings,
        "bets": bets,
        "predictions": [
            {
                "home_team": p["home_team"],
                "away_team": p["away_team"],
                "home_abbr": p["home_abbr"],
                "away_abbr": p["away_abbr"],
                "home_elo": p["home_elo"],
                "away_elo": p["away_elo"],
                "model_prob_home": p["model_prob_home"],
                "yes_price": p["yes_price"],
                "no_price": p["no_price"],
                "yes_implied": p["yes_implied"],
                "no_implied": p["no_implied"],
                "home_edge": p["home_edge"],
                "away_edge": p["away_edge"],
                "ticker": p["ticker"],
                "ticker_away": p.get("ticker_away", p["ticker"]),
                "close_time": p["close_time"],
                "title": p["title"],
                "volume": p.get("volume", 0)
            }
            for p in predictions
        ],
        "summary": {
            "totalBets": len(bets),
            "totalWager": round(total_wager, 2),
            "wagerPercent": round(total_wager / bankroll * 100, 1) if bankroll > 0 else 0,
            "totalProfit": round(total_profit, 2),
            "expectedValue": round(total_ev, 2),
            "bankrollRemaining": round(bankroll - total_wager, 2)
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Kalshi MLB Live Predictor")
    parser.add_argument("--kelly", type=float, default=0.5,
                       help="Kelly fraction (default: 0.5)")
    parser.add_argument("--bankroll", type=float, default=100,
                       help="Starting bankroll in dollars (default: 100)")
    parser.add_argument("--min-edge", type=float, default=5.0,
                       help="Minimum edge percentage to bet (default: 5.0)")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON instead of formatted text")
    parser.add_argument("--today", action="store_true",
                       help="Only show games for today")
    parser.add_argument("--tomorrow", action="store_true",
                       help="Only show games for tomorrow")
    parser.add_argument("--date", type=str, default=None,
                       help="Only show games for a specific date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Compute target date for filtering
    target_date = None
    eastern_now = datetime.now(EASTERN)
    if args.today:
        target_date = eastern_now.date()
    elif args.tomorrow:
        target_date = (eastern_now + timedelta(days=1)).date()
    elif args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    if not args.json:
        print("=" * 70)
        print("KALSHI MLB LIVE PREDICTOR")
        print("=" * 70)
        print(f"Settings:")
        print(f"  Kelly Fraction: {args.kelly}")
        print(f"  Starting Bankroll: ${args.bankroll:.2f}")
        print(f"  Minimum Edge: {args.min_edge:.1f}%")
        if target_date:
            print(f"  Date Filter: {target_date.strftime('%A, %B %d, %Y')}")
        else:
            print(f"  Date Filter: All available games")
        print("=" * 70)

    # Load model and Elo ratings
    elo_system, model_data = load_model_and_elo(quiet=args.json)

    # Fetch Kalshi markets
    games = fetch_kalshi_mlb_markets(quiet=args.json, target_date=target_date)
    
    if not games:
        if not args.json:
            print("\n" + "=" * 70)
            print("NO MLB GAME MARKETS FOUND")
            print("=" * 70)
            print("\nNo daily game markets found on Kalshi.")
            print("Checked both regular season (KXMLB) and spring training (KXMLBSTGAME).")
            print("\nThis could mean:")
            print("  • No games scheduled today")
            print("  • It's the off-season")
            print("  • Markets haven't been posted yet")
            print("=" * 70)
        else:
            # JSON output for no games
            print(json.dumps({
                "rankings": [],
                "bets": [],
                "predictions": [],
                "summary": {
                    "totalBets": 0,
                    "totalWager": 0,
                    "wagerPercent": 0,
                    "totalProfit": 0,
                    "expectedValue": 0,
                    "bankrollRemaining": args.bankroll
                },
                "message": "No MLB game markets found. Season has not started yet."
            }, indent=2))
        return
    
    # Generate predictions
    predictions = generate_predictions(games, elo_system, model_data, quiet=args.json)
    
    # Generate betting recommendations
    bets = generate_betting_recommendations(
        predictions, 
        args.bankroll, 
        args.kelly,
        args.min_edge,
        quiet=args.json
    )
    
    # Output results
    if args.json:
        # JSON output for programmatic use (e.g., Flask app)
        result = get_json_output(predictions, bets, args.bankroll, elo_system)
        print(json.dumps(result, indent=2))
    else:
        # Human-readable formatted output
        print_recommendations(bets, args.bankroll, elo_system)


if __name__ == "__main__":
    main()