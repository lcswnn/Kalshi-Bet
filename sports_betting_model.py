"""
KALSHI SPORTS BETTING MODEL v1
================================
Professional-grade sports betting model using Elo ratings and market analysis.

Key Features:
1. ELO RATING SYSTEM: Dynamic team strength ratings updated after each game
2. SITUATIONAL ADJUSTMENTS: Home advantage, rest days, back-to-backs, streaks
3. MARKET ANALYSIS: Compares model probability vs Kalshi market prices
4. KELLY CRITERION: Optimal bet sizing with bankroll management
5. EDGE DETECTION: Only bets when model finds significant edge vs market

Methodology (used by sharp bettors):
- Elo ratings capture team strength dynamically
- Logistic conversion to win probabilities
- Adjustments for home court, rest, momentum
- Market inefficiency identification
- Kelly Criterion position sizing

Usage:
    python sports_betting_model.py --bankroll 100 --kelly 0.75
    python sports_betting_model.py --league NBA --min-edge 0.10
    python sports_betting_model.py --update-elo historical_games.csv
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import json
import argparse
from pathlib import Path
import pickle
import time
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ============ CONFIGURATION ============

# Kalshi API Configuration
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_API_BASE = "https://demo-api.elections.kalshi.com/trade-api/v2"
USE_DEMO = False  # Set to False for production

# Price Filter Configuration
MIN_CONTRACT_PRICE = 0.15   # Never bet below 15¢
MAX_CONTRACT_PRICE = 0.85   # Never bet above 85¢
SWEET_SPOT_LOW = 0.20       # Preferred range lower bound
SWEET_SPOT_HIGH = 0.60      # Preferred range upper bound

# Edge Requirements
BASE_MIN_EDGE = 0.08        # 8% minimum edge required
SWEET_SPOT_MIN_EDGE = 0.06  # 6% edge for sweet spot prices
HIGH_CONFIDENCE_EDGE = 0.12 # 12% edge for aggressive betting

# Kelly Criterion Configuration
STARTING_BANKROLL = 100.0
KELLY_FRACTION = 0.5        # Half Kelly (conservative)
MIN_BET_SIZE = 1.0          # Minimum $1 bet
MAX_BET_FRACTION = 0.10     # Never bet more than 10% of bankroll
MAX_TOTAL_BETS_PER_DAY = 5  # Cap daily bet count

# Elo Rating Configuration (FiveThirtyEight methodology)
INITIAL_ELO = 1500          # Starting Elo rating
K_FACTOR = 20               # FiveThirtyEight uses K=20 for NBA
HOME_ADVANTAGE = 100        # FiveThirtyEight: 100 Elo = ~3.5 points
ELO_SCALE = 400             # Logistic scaling factor
POINT_SPREAD_DIVISOR = 28   # FiveThirtyEight: (Elo_diff + 100) / 28 = spread

# League-Specific Settings (can be customized per sport)
LEAGUE_SETTINGS = {
    "NBA": {
        "home_advantage": 100,         # FiveThirtyEight uses 100 Elo points
        "k_factor": 20,                # FiveThirtyEight: K=20 for NBA
        "rest_advantage_per_day": 0,   # FiveThirtyEight doesn't use this
        "back_to_back_penalty": 0,     # FiveThirtyEight doesn't use this
        "win_streak_boost": 0,         # FiveThirtyEight doesn't use this
    },
    "NFL": {
        "home_advantage": 65,
        "k_factor": 40,
        "rest_advantage_per_day": 5,
        "back_to_back_penalty": 0,     # NFL doesn't have back-to-backs
        "win_streak_boost": 8,
    },
    "NHL": {
        "home_advantage": 75,
        "k_factor": 28,
        "rest_advantage_per_day": 12,
        "back_to_back_penalty": 50,
        "win_streak_boost": 4,
    },
    "MLB": {
        "home_advantage": 30,
        "k_factor": 24,
        "rest_advantage_per_day": 5,
        "back_to_back_penalty": 0,
        "win_streak_boost": 3,
    },
    "NCAAB": {  # College Basketball
        "home_advantage": 120,
        "k_factor": 36,
        "rest_advantage_per_day": 8,
        "back_to_back_penalty": 35,
        "win_streak_boost": 6,
    }
}

# Default league
DEFAULT_LEAGUE = "NBA"
CURRENT_LEAGUE = DEFAULT_LEAGUE

# Model persistence
MODEL_SAVE_PATH = "sports_model_elo_ratings.pkl"

# ============ TIMEZONE HELPERS ============

KALSHI_TIMEZONE = ZoneInfo("America/New_York")

def get_eastern_now():
    """Get current time in US Eastern timezone."""
    return datetime.now(KALSHI_TIMEZONE)


# ============ ELO RATING SYSTEM ============

class EloRatingSystem:
    """
    Elo rating system for sports teams.
    
    Elo ratings provide a dynamic measure of team strength that updates
    after each game based on actual results vs expected results.
    """
    
    def __init__(self, league="NBA", k_factor=None, home_advantage=None):
        self.league = league
        self.settings = LEAGUE_SETTINGS.get(league, LEAGUE_SETTINGS["NBA"])
        
        self.k_factor = k_factor if k_factor else self.settings["k_factor"]
        self.home_advantage = home_advantage if home_advantage else self.settings["home_advantage"]
        
        self.ratings = {}  # {team_name: rating}
        self.games_played = {}  # {team_name: count}
        self.win_streaks = {}  # {team_name: current_streak}
        self.last_game_date = {}  # {team_name: datetime}
        
    def get_rating(self, team):
        """Get current Elo rating for a team."""
        if team not in self.ratings:
            self.ratings[team] = INITIAL_ELO
            self.games_played[team] = 0
            self.win_streaks[team] = 0
        return self.ratings[team]
    
    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected score (win probability) for team A.
        
        Uses logistic function: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / ELO_SCALE))
    
    def calculate_win_probability(self, team_a, team_b, team_a_home=True, 
                                  team_a_rest_days=1, team_b_rest_days=1):
        """
        Calculate win probability for team_a vs team_b with situational adjustments.
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            team_a_home: Whether team A is home
            team_a_rest_days: Days of rest for team A
            team_b_rest_days: Days of rest for team B
            
        Returns:
            float: Win probability for team A (0 to 1)
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Adjust for home advantage
        if team_a_home:
            rating_a += self.home_advantage
        else:
            rating_b += self.home_advantage
        
        # Adjust for rest advantage/disadvantage
        rest_diff = team_a_rest_days - team_b_rest_days
        rest_adjustment = rest_diff * self.settings["rest_advantage_per_day"]
        rating_a += rest_adjustment
        
        # Back-to-back penalty
        if team_a_rest_days == 0:
            rating_a -= self.settings["back_to_back_penalty"]
        if team_b_rest_days == 0:
            rating_b += self.settings["back_to_back_penalty"]
        
        # Win streak momentum (capped at 3 games)
        streak_a = min(abs(self.win_streaks.get(team_a, 0)), 3)
        streak_b = min(abs(self.win_streaks.get(team_b, 0)), 3)
        if self.win_streaks.get(team_a, 0) > 0:
            rating_a += streak_a * self.settings["win_streak_boost"]
        if self.win_streaks.get(team_b, 0) > 0:
            rating_b += streak_b * self.settings["win_streak_boost"]
        
        # Calculate probability
        win_prob = self.expected_score(rating_a, rating_b)
        
        return win_prob
    
    def update_ratings(self, team_a, team_b, team_a_won, team_a_home=True, 
                      game_date=None):
        """
        Update Elo ratings after a game.
        
        Args:
            team_a: Name of team A
            team_b: Name of team B  
            team_a_won: Whether team A won (True/False)
            team_a_home: Whether team A was home
            game_date: Date of the game (for tracking rest days)
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Home advantage for prediction
        if team_a_home:
            expected_a = self.expected_score(rating_a + self.home_advantage, rating_b)
        else:
            expected_a = self.expected_score(rating_a, rating_b + self.home_advantage)
        
        # Actual result
        actual_a = 1.0 if team_a_won else 0.0
        
        # Update ratings
        self.ratings[team_a] = rating_a + self.k_factor * (actual_a - expected_a)
        self.ratings[team_b] = rating_b + self.k_factor * ((1 - actual_a) - (1 - expected_a))
        
        # Update games played
        self.games_played[team_a] = self.games_played.get(team_a, 0) + 1
        self.games_played[team_b] = self.games_played.get(team_b, 0) + 1
        
        # Update win streaks
        if team_a_won:
            current_streak = self.win_streaks.get(team_a, 0)
            self.win_streaks[team_a] = current_streak + 1 if current_streak >= 0 else 1
            
            current_streak = self.win_streaks.get(team_b, 0)
            self.win_streaks[team_b] = current_streak - 1 if current_streak <= 0 else -1
        else:
            current_streak = self.win_streaks.get(team_b, 0)
            self.win_streaks[team_b] = current_streak + 1 if current_streak >= 0 else 1
            
            current_streak = self.win_streaks.get(team_a, 0)
            self.win_streaks[team_a] = current_streak - 1 if current_streak <= 0 else -1
        
        # Update last game dates
        if game_date:
            self.last_game_date[team_a] = game_date
            self.last_game_date[team_b] = game_date
    
    def save(self, filepath=MODEL_SAVE_PATH):
        """Save Elo ratings to disk."""
        data = {
            "league": self.league,
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "ratings": self.ratings,
            "games_played": self.games_played,
            "win_streaks": self.win_streaks,
            "last_game_date": self.last_game_date,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Elo ratings saved to {filepath}")
    
    @classmethod
    def load(cls, filepath=MODEL_SAVE_PATH):
        """Load Elo ratings from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            elo_system = cls(
                league=data["league"],
                k_factor=data["k_factor"],
                home_advantage=data["home_advantage"]
            )
            elo_system.ratings = data["ratings"]
            elo_system.games_played = data["games_played"]
            elo_system.win_streaks = data["win_streaks"]
            elo_system.last_game_date = data["last_game_date"]
            
            print(f"✅ Loaded Elo ratings for {len(elo_system.ratings)} teams")
            return elo_system
        except FileNotFoundError:
            print(f"⚠️  No saved model found at {filepath}, starting fresh")
            return cls()
    
    def print_rankings(self, top_n=10):
        """Print top teams by Elo rating."""
        sorted_teams = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        print(f"\n📊 TOP {top_n} TEAMS BY ELO RATING ({self.league}):")
        print("=" * 60)
        for i, (team, rating) in enumerate(sorted_teams[:top_n], 1):
            games = self.games_played.get(team, 0)
            streak = self.win_streaks.get(team, 0)
            streak_str = f"🔥{streak}W" if streak > 0 else f"❄️{abs(streak)}L" if streak < 0 else ""
            print(f"{i:2}. {team:25} | Rating: {rating:7.1f} | Games: {games:3} {streak_str}")


# ============ KALSHI API CLIENT ============

class KalshiClient:
    """Client for interacting with Kalshi API."""
    
    def __init__(self, use_demo=USE_DEMO):
        self.base_url = KALSHI_DEMO_API_BASE if use_demo else KALSHI_API_BASE
        self.session = requests.Session()
    
    def get_sports_markets(self, league=None, status="open", debug=False):
        """
        Fetch sports markets from Kalshi.
        
        Args:
            league: Filter by sport/league (NBA, NFL, etc.)
            status: Market status ('open', 'closed', 'settled')
            debug: Show debug information
            
        Returns:
            list: Market data dictionaries
        """
        try:
            url = f"{self.base_url}/markets"
            
            # Kalshi NBA markets use the KXNBAGAME series
            params = {
                "status": status,
                "limit": 200,
                "series_ticker": "KXNBAGAME",  # This is the NBA series!
            }
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
            if debug:
                print(f"   Searching for series: KXNBAGAME")
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            markets = data.get("markets", [])
            
            if debug:
                print(f"   Found {len(markets)} markets in KXNBAGAME series")
                if markets:
                    print(f"   Sample markets:")
                    for m in markets[:3]:
                        print(f"      {m.get('ticker', 'N/A')}: {m.get('title', 'N/A')}")
            
            return markets
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"⚠️  Rate limited by Kalshi API. Waiting 5 seconds...")
                time.sleep(5)
                # Try one more time
                try:
                    response = self.session.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    markets = data.get("markets", [])
                    print(f"   Retry successful. Got {len(markets)} markets")
                    return markets
                except:
                    print(f"❌ Still rate limited. Please wait a minute and try again.")
                    return []
            else:
                print(f"❌ HTTP Error: {e}")
                return []
        except Exception as e:
            print(f"❌ Error fetching Kalshi markets: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _is_sports_market(self, market):
        """Check if market is a sports betting market."""
        title = market.get("title", "").lower()
        subtitle = market.get("subtitle", "").lower()
        
        # Look for team vs team patterns
        sports_keywords = ["win", "beat", "defeat", " vs ", " v ", "@", "game"]
        team_indicators = ["points", "score", "spread", "over", "under"]
        
        return any(keyword in title or keyword in subtitle 
                  for keyword in sports_keywords + team_indicators)
    
    def get_market_details(self, ticker):
        """Get detailed information for a specific market."""
        try:
            url = f"{self.base_url}/markets/{ticker}"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching market {ticker}: {e}")
            return None


# ============ MARKET PARSER ============

def parse_game_from_market(market):
    """
    Extract game information from Kalshi market data.
    
    Kalshi NBA format:
    - Ticker: KXNBAGAME-25DEC18NYKIND-NYK
    - Format: YYMMMDD (25DEC18 = 2025, December, 18th)
    - Title: "New York vs Indiana Winner?"
    
    Returns:
        dict with keys: team_a, team_b, team_a_home, game_date, market_ticker, 
                        bet_price, subtitle, market_type
        or None if unable to parse
    """
    title = market.get("title", "")
    subtitle = market.get("subtitle", "")
    ticker = market.get("ticker", "")
    
    # Parse teams from TITLE: "New York vs Indiana Winner?"
    # IMPORTANT: In Kalshi format, first team is AWAY, second team is HOME
    teams = None
    original_team_a_home = False  # First team in title is AWAY
    original_team_b_home = True   # Second team in title is HOME
    
    if " vs " in title:
        # Remove " Winner?" from end and split
        title_clean = title.replace(" Winner?", "").strip()
        team_parts = title_clean.split(" vs ")
        
        if len(team_parts) == 2:
            team_name_a = team_parts[0].strip()
            team_name_b = team_parts[1].strip()
            
            # Map short names to full names
            team_map = {
                "Atlanta": "Atlanta Hawks",
                "Boston": "Boston Celtics",
                "Brooklyn": "Brooklyn Nets",
                "Charlotte": "Charlotte Hornets",
                "Chicago": "Chicago Bulls",
                "Cleveland": "Cleveland Cavaliers",
                "Dallas": "Dallas Mavericks",
                "Denver": "Denver Nuggets",
                "Detroit": "Detroit Pistons",
                "Golden State": "Golden State Warriors",
                "Houston": "Houston Rockets",
                "Indiana": "Indiana Pacers",
                "Los Angeles C": "Los Angeles Clippers",
                "Los Angeles L": "Los Angeles Lakers",
                "Memphis": "Memphis Grizzlies",
                "Miami": "Miami Heat",
                "Milwaukee": "Milwaukee Bucks",
                "Minnesota": "Minnesota Timberwolves",
                "New Orleans": "New Orleans Pelicans",
                "New York": "New York Knicks",
                "Oklahoma City": "Oklahoma City Thunder",
                "Orlando": "Orlando Magic",
                "Philadelphia": "Philadelphia 76ers",
                "Phoenix": "Phoenix Suns",
                "Portland": "Portland Trail Blazers",
                "Sacramento": "Sacramento Kings",
                "San Antonio": "San Antonio Spurs",
                "Toronto": "Toronto Raptors",
                "Utah": "Utah Jazz",
                "Washington": "Washington Wizards",
            }
            
            team_a = team_map.get(team_name_a, team_name_a)
            team_b = team_map.get(team_name_b, team_name_b)
            teams = (team_a, team_b)
    
    if not teams:
        return None
    
    # Parse game date from TICKER
    # Format: KXNBAGAME-25DEC18NYKIND-NYK
    # Date part: 25DEC18 means Year 2025, December, Day 18 (YYMMMDD)
    game_date = None
    parts = None
    try:
        parts = ticker.split("-")
        if len(parts) >= 2:
            date_and_teams = parts[1]  # "25DEC18NYKIND"
            # Extract date: first 7 chars "25DEC18"
            date_str = date_and_teams[:7]  # "25DEC18"
            
            # Parse: YYMMMDD
            year_short = date_str[:2]  # "25"
            month_abbr = date_str[2:5]  # "DEC"
            day = date_str[5:7]          # "18"
            
            # Convert month abbreviation
            month_map = {
                "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
                "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
                "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
            }
            month = month_map.get(month_abbr, "01")
            
            # Year: 25 -> 2025, 26 -> 2026, etc.
            year = f"20{year_short}"
            
            # Create date string and parse
            from datetime import datetime, timezone
            date_str_full = f"{year}-{month}-{day}"
            game_date = datetime.strptime(date_str_full, "%Y-%m-%d")
            
            # Use close_time for the TIME component
            close_time = market.get("close_time")
            if close_time:
                try:
                    close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    # Combine ticker date with close_time's time
                    game_date = game_date.replace(
                        hour=close_dt.hour,
                        minute=close_dt.minute,
                        tzinfo=close_dt.tzinfo
                    )
                except:
                    game_date = game_date.replace(tzinfo=timezone.utc)
    except Exception as e:
        # If date parsing fails, use close_time as fallback
        close_time = market.get("close_time")
        if close_time:
            try:
                from datetime import datetime
                game_date = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            except:
                pass
    
    # Get market price (yes_bid is what we'd pay to bet "yes")
    yes_bid = market.get("yes_bid")
    no_bid = market.get("no_bid")
    
    # CRITICAL: Determine which team this market is for
    # The last part of ticker tells us: KXNBAGAME-25DEC18NYKIND-NYK
    # This means we're betting on NYK (New York Knicks)
    betting_on_team = None
    if parts and len(parts) >= 3:
        betting_on_code = parts[-1]  # Last part is the team code (e.g., "NYK")
        
        # Extract team codes from middle part for comparison
        date_and_teams = parts[1]
        team_codes = date_and_teams[7:] if len(date_and_teams) > 7 else ""
        
        # Team codes are typically 3 letters each
        if len(team_codes) == 6:
            team_a_code = team_codes[:3]  # First 3 letters
            team_b_code = team_codes[3:]  # Last 3 letters
            
            # Determine which team we're betting on
            if betting_on_code == team_a_code:
                betting_on_team = "team_a"
            elif betting_on_code == team_b_code:
                betting_on_team = "team_b"
    
    # If we're betting on team_b, swap teams so team_a is always the betting target
    # REMEMBER: Original team_a = away, team_b = home
    team_a_home = original_team_a_home  # False (away)
    
    if betting_on_team == "team_b":
        # We're betting on the home team (team_b)
        # Swap so our betting target becomes team_a
        teams = (teams[1], teams[0])
        # Now team_a is the home team
        team_a_home = original_team_b_home  # True (home)
    # else: we're betting on team_a (the away team), so team_a_home stays False
    
    bet_price = yes_bid / 100 if yes_bid else None
    
    return {
        "team_a": teams[0],
        "team_b": teams[1],
        "team_a_home": team_a_home,
        "game_date": game_date,
        "market_ticker": ticker,
        "bet_price": bet_price,
        "no_price": no_bid / 100 if no_bid else None,
        "subtitle": subtitle,
        "title": title,
        "market_type": "moneyline",
    }


# ============ KELLY CRITERION ============

def calculate_kelly_bet(bankroll, our_prob, bet_price, kelly_fraction=KELLY_FRACTION):
    """
    Calculate optimal bet size using Kelly Criterion.
    
    Args:
        bankroll: Current bankroll
        our_prob: Our calculated win probability
        bet_price: Market price (what we pay)
        kelly_fraction: Fraction of Kelly to use (e.g., 0.5 for half Kelly)
    
    Returns:
        bet_size: Dollar amount to bet
        kelly_info: Dict with calculation details
    """
    if bet_price <= 0 or bet_price >= 1:
        return 0, {}
    
    # Odds: profit per $1 wagered if win
    b = (1 / bet_price) - 1
    
    p = our_prob  # Our win probability
    q = 1 - p     # Loss probability
    
    # Full Kelly formula: f = (bp - q) / b
    full_kelly = (b * p - q) / b if b > 0 else 0
    
    # Apply fractional Kelly
    fractional_kelly = full_kelly * kelly_fraction
    
    # Clamp to reasonable bounds
    fractional_kelly = max(0, fractional_kelly)
    fractional_kelly = min(fractional_kelly, MAX_BET_FRACTION)
    
    # Calculate bet size
    bet_size = bankroll * fractional_kelly
    
    # Apply minimum bet
    if bet_size < MIN_BET_SIZE:
        bet_size = 0
    
    kelly_info = {
        "full_kelly_pct": full_kelly * 100,
        "fractional_kelly_pct": fractional_kelly * 100,
        "odds": b,
        "expected_value": (our_prob * b - q) * bet_size,
    }
    
    return bet_size, kelly_info


# ============ BET ANALYSIS ============

def analyze_betting_opportunity(elo_system, game_info, bankroll):
    """
    Analyze a game and determine if there's a betting opportunity.
    
    Returns:
        dict with bet recommendation or None if no edge
    """
    team_a = game_info["team_a"]
    team_b = game_info["team_b"]
    team_a_home = game_info.get("team_a_home", True)
    bet_price = game_info.get("bet_price")
    
    if bet_price is None or bet_price <= 0:
        return None
    
    # Apply price filters
    if bet_price < MIN_CONTRACT_PRICE or bet_price > MAX_CONTRACT_PRICE:
        return None
    
    # Calculate rest days (simplified - would need actual schedule data)
    team_a_rest = 1
    team_b_rest = 1
    
    # Get base Elo ratings
    elo_a = elo_system.get_rating(team_a)
    elo_b = elo_system.get_rating(team_b)
    
    # Calculate our win probability using Elo
    our_prob_a_wins = elo_system.calculate_win_probability(
        team_a, team_b, 
        team_a_home=team_a_home,
        team_a_rest_days=team_a_rest,
        team_b_rest_days=team_b_rest
    )
    
    # DEBUG: Print calculation details
    # print(f"DEBUG {team_a} vs {team_b}:")
    # print(f"  Base Elo: {elo_a:.0f} vs {elo_b:.0f}")
    # print(f"  Home: {team_a_home}")
    # print(f"  Calculated prob: {our_prob_a_wins*100:.1f}%")
    # print(f"  Market price: {bet_price*100:.0f}¢")
    
    # Market implied probability
    market_prob = bet_price
    
    # Calculate edge
    edge = our_prob_a_wins - market_prob
    
    # Determine minimum edge requirement
    price_bucket = "sweet_spot" if SWEET_SPOT_LOW <= bet_price <= SWEET_SPOT_HIGH else "other"
    min_edge = SWEET_SPOT_MIN_EDGE if price_bucket == "sweet_spot" else BASE_MIN_EDGE
    
    # Check if we have sufficient edge
    if edge < min_edge:
        return None
    
    # Calculate Kelly bet size
    bet_size, kelly_info = calculate_kelly_bet(bankroll, our_prob_a_wins, bet_price)
    
    if bet_size == 0:
        return None
    
    # Calculate potential profit
    potential_profit = bet_size * kelly_info["odds"]
    
    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_home": team_a_home,
        "our_prob": our_prob_a_wins,
        "market_prob": market_prob,
        "edge": edge,
        "edge_pct": edge * 100,
        "bet_price": bet_price,
        "bet_size": bet_size,
        "potential_profit": potential_profit,
        "kelly_info": kelly_info,
        "market_ticker": game_info["market_ticker"],
        "subtitle": game_info["subtitle"],
        "title": game_info["title"],
        "game_date": game_info.get("game_date"),
        "price_bucket": price_bucket,
        "elo_a": elo_a,
        "elo_b": elo_b,
    }


# ============ HISTORICAL DATA PROCESSING ============

def update_elo_from_historical_data(elo_system, csv_file, seasons_back=3):
    """
    Update Elo ratings from historical game results.
    
    Supports two CSV formats:
    
    Format 1 (Simple):
        - date: Game date (YYYY-MM-DD)
        - team_a: Name of team A
        - team_b: Name of team B
        - team_a_score: Score for team A
        - team_b_score: Score for team B
        - team_a_home: 1 if team A is home, 0 otherwise
    
    Format 2 (NBA Games.csv):
        - gameDateTimeEst: Game datetime
        - hometeamCity: Home team city
        - hometeamName: Home team name
        - awayteamCity: Away team city
        - awayteamName: Away team name
        - homeScore: Home team score
        - awayScore: Away team score
    """
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"📂 Loading historical data from {csv_file}")
        print(f"   Found {len(df):,} total games")
        
        # Detect format and convert to standard format
        if "gameDateTimeEst" in df.columns:
            # Format 2: NBA Games.csv format
            print("   Detected NBA Games.csv format")
            
            # Parse date (handles various formats including timezone info)
            df["date"] = pd.to_datetime(df["gameDateTimeEst"], format='mixed', utc=True)
            
            # Filter to recent seasons only (configurable)
            cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=365 * seasons_back), tz='UTC')
            df = df[df["date"] >= cutoff_date].copy()
            print(f"   Filtered to last {seasons_back} seasons: {len(df):,} games")
            
            # Create team names
            df["team_a"] = df["hometeamCity"] + " " + df["hometeamName"]
            df["team_b"] = df["awayteamCity"] + " " + df["awayteamName"]
            df["team_a_score"] = df["homeScore"]
            df["team_b_score"] = df["awayScore"]
            df["team_a_home"] = 1  # Home team is always team_a
            
        elif "date" in df.columns and "team_a" in df.columns:
            # Format 1: Simple format
            print("   Detected simple format")
            df["date"] = pd.to_datetime(df["date"])
            
            required_cols = ["date", "team_a", "team_b", "team_a_score", "team_b_score"]
            if not all(col in df.columns for col in required_cols):
                print(f"❌ CSV must have columns: {required_cols}")
                return False
        else:
            print("❌ Unrecognized CSV format")
            print("   Expected either:")
            print("   - Simple format: date, team_a, team_b, team_a_score, team_b_score")
            print("   - NBA format: gameDateTimeEst, hometeamCity, hometeamName, etc.")
            return False
        
        # Sort chronologically (important for Elo)
        df = df.sort_values("date")
        
        print(f"   Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Count unique teams
        all_teams = set(df["team_a"].unique()) | set(df["team_b"].unique())
        print(f"   Found {len(all_teams)} unique teams")
        
        # Process games
        print("   Processing games...")
        for i, row in df.iterrows():
            team_a = row["team_a"]
            team_b = row["team_b"]
            team_a_won = row["team_a_score"] > row["team_b_score"]
            team_a_home = row.get("team_a_home", 1) == 1
            game_date = row["date"]
            
            elo_system.update_ratings(
                team_a, team_b, 
                team_a_won, 
                team_a_home,
                game_date
            )
            
            # Progress indicator for large datasets
            if (i + 1) % 1000 == 0:
                print(f"   ... processed {i + 1:,} games")
        
        print(f"✅ Updated Elo ratings from {len(df):,} historical games")
        
        # Show sample of ratings
        print(f"\n📊 Sample Elo Ratings:")
        sorted_teams = sorted(elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
        for i, (team, rating) in enumerate(sorted_teams[:5], 1):
            games = elo_system.games_played.get(team, 0)
            print(f"   {i}. {team}: {rating:.1f} ({games} games)")
        
        elo_system.save()
        return True
        
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============ OUTPUT FORMATTING ============

def print_header(league):
    """Print analysis header."""
    eastern_now = get_eastern_now()
    print("=" * 70)
    print(f"🏀 KALSHI SPORTS BETTING MODEL - {league}")
    print(f"📅 Analysis Time: {eastern_now.strftime('%Y-%m-%d %I:%M %p %Z')}")
    print("=" * 70)


def print_recommendations(bets, bankroll, elo_system):
    """Print betting recommendations."""
    
    if not bets:
        print("\n❌ NO BETTING OPPORTUNITIES FOUND")
        print("   Model found no edges meeting minimum requirements.")
        return
    
    print(f"\n✅ FOUND {len(bets)} BETTING OPPORTUNITIES")
    print("=" * 70)
    
    total_wager = 0
    total_potential = 0
    
    for i, bet in enumerate(bets, 1):
        home_marker = "🏠" if bet["team_a_home"] else "✈️"
        bucket_emoji = "🎯" if bet["price_bucket"] == "sweet_spot" else "📊"
        
        # CLEAR BET INDICATION
        bet_team = bet["team_a"]  # We're always betting on team_a after the fix
        opponent = bet["team_b"]
        
        print(f"\n   ┌─ BET #{i}: {bet['title']}")
        print(f"   │")
        print(f"   │  🎰 BET ON: {bet_team} {home_marker}")
        print(f"   │  🆚 OPPONENT: {opponent}")
        print(f"   │  📊 Elo: {bet['elo_a']:.0f} (betting on) vs {bet['elo_b']:.0f} (opponent)")
        print(f"   │")
        print(f"   │  🎲 Model probability: {bet['our_prob']*100:.1f}%")
        print(f"   │  💰 Market price: {bet['bet_price']*100:.0f}¢ (implies {bet['market_prob']*100:.1f}%)")
        print(f"   │  📈 Edge: {bet['edge_pct']:+.1f}% {bucket_emoji}")
        print(f"   │")
        print(f"   │  Kelly sizing: {bet['kelly_info']['fractional_kelly_pct']:.1f}% of bankroll")
        print(f"   │  💵 BET: ${bet['bet_size']:.2f}")
        print(f"   │  🎁 Potential profit: ${bet['potential_profit']:.2f}")
        print(f"   │  📊 Expected value: ${bet['kelly_info']['expected_value']:.2f}")
        print(f"   │")
        print(f"   │  Ticker: {bet['market_ticker']}")
        if bet.get("game_date"):
            print(f"   │  Game time: {bet['game_date'].strftime('%Y-%m-%d %I:%M %p')}")
        print(f"   └{'─'*64}")
        
        total_wager += bet["bet_size"]
        total_potential += bet["potential_profit"]
    
    print(f"\n   {'='*66}")
    print(f"   TOTAL WAGER: ${total_wager:.2f} ({100*total_wager/bankroll:.1f}% of bankroll)")
    print(f"   POTENTIAL PROFIT: ${total_potential:.2f}")
    print(f"   EXPECTED VALUE: ${sum(b['kelly_info']['expected_value'] for b in bets):.2f}")
    print(f"   {'='*66}")


# ============ MAIN ============

def main():
    # Declare globals at the start
    global STARTING_BANKROLL, KELLY_FRACTION, BASE_MIN_EDGE, CURRENT_LEAGUE, USE_DEMO
    
    parser = argparse.ArgumentParser(
        description='Kalshi Sports Betting Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Use module-level defaults for argparse defaults
    parser.add_argument('--bankroll', type=float, default=100.0,
                       help=f'Starting bankroll (default: $100)')
    parser.add_argument('--kelly', type=float, default=0.5,
                       help=f'Kelly fraction (default: 0.5)')
    parser.add_argument('--league', type=str, default='NBA',
                       choices=list(LEAGUE_SETTINGS.keys()),
                       help=f'Sport/league to analyze (default: NBA)')
    parser.add_argument('--min-edge', type=float, default=0.08,
                       help=f'Minimum edge required (default: 0.08)')
    parser.add_argument('--update-elo', type=str, metavar='CSV_FILE',
                       help='Update Elo ratings from historical game CSV')
    parser.add_argument('--seasons', type=int, default=3,
                       help='Number of seasons to use for Elo training (default: 3)')
    parser.add_argument('--show-rankings', action='store_true',
                       help='Show current Elo rankings and exit')
    parser.add_argument('--demo', action='store_true',
                       help='Use Kalshi demo API instead of production')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information about API responses')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all parsed games, even without sufficient edge')
    
    args = parser.parse_args()
    
    # Update global settings
    STARTING_BANKROLL = args.bankroll
    KELLY_FRACTION = args.kelly
    BASE_MIN_EDGE = args.min_edge
    CURRENT_LEAGUE = args.league
    USE_DEMO = args.demo
    
    # Load or create Elo system
    if Path(MODEL_SAVE_PATH).exists():
        elo_system = EloRatingSystem.load(MODEL_SAVE_PATH)
        # Update league settings if different
        if elo_system.league != args.league:
            print(f"⚠️  Model was trained on {elo_system.league}, switching to {args.league}")
            elo_system = EloRatingSystem(league=args.league)
    else:
        elo_system = EloRatingSystem(league=args.league)
    
    # Update Elo from historical data if requested
    if args.update_elo:
        success = update_elo_from_historical_data(elo_system, args.update_elo, seasons_back=args.seasons)
        if not success:
            return
    
    # Show rankings if requested
    if args.show_rankings:
        elo_system.print_rankings(top_n=20)
        return
    
    print_header(args.league)
    
    # Initialize Kalshi client
    client = KalshiClient(use_demo=args.demo)
    
    # Fetch markets from Kalshi
    print(f"\n🔍 Fetching {args.league} markets from Kalshi...")
    if args.debug:
        print(f"   Using API: {client.base_url}")
    
    markets = client.get_sports_markets(league=args.league, debug=args.debug)
    
    if not markets:
        print("❌ No sports markets found on Kalshi")
        print("   Try a different league or check if markets are available")
        return
    
    print(f"   Found {len(markets)} potential markets")
    
    # Get today's date in Eastern time
    eastern_now = get_eastern_now()
    today = eastern_now.date()
    
    print(f"   Filtering to games on {today.strftime('%B %d, %Y')} only...")
    
    # Parse and analyze each market
    all_bets = []
    parsed_count = 0
    skipped_future = 0
    
    for market in markets:
        if args.debug:
            print(f"\n   ═══ RAW MARKET DATA ═══")
            print(f"   Ticker: {market.get('ticker')}")
            print(f"   Title: {market.get('title')}")
            print(f"   Subtitle: {market.get('subtitle')}")
            print(f"   Yes bid: {market.get('yes_bid')}¢")
            print(f"   No bid: {market.get('no_bid')}¢")
            print(f"   Close time: {market.get('close_time')}")
        
        game_info = parse_game_from_market(market)
        
        if not game_info:
            if args.debug:
                print(f"   ⚠️  Could not parse")
            continue
        
        # FILTER: Only show TODAY's games
        game_date = game_info.get('game_date')
        if game_date:
            game_day = game_date.date()
            if game_day != today:
                skipped_future += 1
                if args.debug:
                    print(f"   ⏭️  Skipping (game on {game_day}, not today)")
                continue
        
        if args.debug:
            print(f"   ─── PARSED ───")
            print(f"   Team A: {game_info['team_a']}")
            print(f"   Team B: {game_info['team_b']}")
            print(f"   Price: {game_info.get('bet_price', 'N/A')}")
            print(f"   Date: {game_info.get('game_date')}")
        
        parsed_count += 1
        
        # Analyze betting opportunity
        bet = analyze_betting_opportunity(elo_system, game_info, STARTING_BANKROLL)
        
        if bet:
            all_bets.append(bet)
        elif args.show_all and game_info.get('bet_price'):
            # Show why it didn't qualify
            team_a = game_info["team_a"]
            team_b = game_info["team_b"]
            our_prob = elo_system.calculate_win_probability(
                team_a, team_b, 
                team_a_home=game_info.get("team_a_home", True),
                team_a_rest_days=1,
                team_b_rest_days=1
            )
            edge = our_prob - game_info.get('bet_price', 0)
            edge_pct = edge * 100
            min_edge_needed = BASE_MIN_EDGE * 100
            
            print(f"\n   ⚠️  {team_a} vs {team_b}")
            print(f"      Model: {our_prob*100:.1f}% | Market: {game_info['bet_price']*100:.0f}¢ | Edge: {edge_pct:+.1f}%")
            print(f"      Need {min_edge_needed:.0f}%+ edge (insufficient)")
            print(f"      Date: {game_info.get('game_date', 'N/A')}")
    
    if args.show_all:
        print(f"\n   Successfully parsed {parsed_count} games for today")
        if skipped_future > 0:
            print(f"   Skipped {skipped_future} games on future dates")
    
    # Sort by edge (highest first)
    all_bets.sort(key=lambda x: x["edge"], reverse=True)
    
    # Limit to max bets per day
    selected_bets = all_bets[:MAX_TOTAL_BETS_PER_DAY]
    
    # Print recommendations
    if not selected_bets:
        print(f"\n❌ NO BETTING OPPORTUNITIES TODAY ({today.strftime('%B %d, %Y')})")
        if parsed_count == 0:
            print(f"   No games found for today.")
        else:
            print(f"   Found {parsed_count} games today, but none meet edge requirements.")
            print(f"   All games are priced too efficiently (need {BASE_MIN_EDGE*100:.0f}%+ edge)")
        return
    
    # Print recommendations for today's games
    print_recommendations(selected_bets, STARTING_BANKROLL, elo_system)
    
    # Show current ratings
    if elo_system.ratings:
        elo_system.print_rankings(top_n=10)
    
    # ============ QUICK BET SUMMARY ============
    if selected_bets:
        print(f"\n{'='*70}")
        print("🎯 QUICK BET SUMMARY - TOP 3 PICKS")
        print(f"{'='*70}")
        
        for i, bet in enumerate(selected_bets[:3], 1):
            home_away = "🏠 HOME" if bet["team_a_home"] else "✈️  AWAY"
            
            # Confidence based on edge
            if bet["edge_pct"] > 15:
                confidence = "🔥 HIGH CONFIDENCE"
            elif bet["edge_pct"] > 10:
                confidence = "✅ GOOD"
            else:
                confidence = "⚠️  MODERATE"
            
            print(f"\n#{i}. BET ${bet['bet_size']:.0f} → {bet['team_a']} {home_away} {confidence}")
            print(f"    vs {bet['team_b']}")
            print(f"    Win probability: {bet['our_prob']*100:.0f}% | Edge: {bet['edge_pct']:+.0f}%")
            print(f"    Market price: {bet['bet_price']*100:.0f}¢ | Potential profit: ${bet['potential_profit']:.2f}")
            print(f"    Kalshi ticker: {bet['market_ticker']}")
        
        print(f"\n{'─'*70}")
        top_3_wager = sum(b['bet_size'] for b in selected_bets[:3])
        top_3_profit = sum(b['potential_profit'] for b in selected_bets[:3])
        print(f"💰 TOTAL FOR TOP 3: Bet ${top_3_wager:.2f} → Potential ${top_3_profit:.2f} profit")
        print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\n💡 TIP: Update Elo ratings regularly with:")
    print(f"   python3 {Path(__file__).name} --update-elo Games.csv")


if __name__ == "__main__":
    main()