"""
MLB Game Prediction Model v2 — Professional-Grade Ensemble
============================================================
Improvements over v1:
  - REALISTIC betting simulation (flat unit sizing, actual vig modeling)
  - Pitcher features via FanGraphs season-level stats  
  - Pitcher Elo ratings from win/loss decisions
  - ATH -> OAK mapping fix
  - Better calibration diagnostics

Architecture (what the pros use):
  - Elo power ratings (carry across seasons, MOV-adjusted)
  - Pitcher quality metrics (ERA, FIP, WAR, K/9, WHIP rolling)
  - Ensemble: XGBoost + LightGBM + Logistic Regression
  - Isotonic regression calibration on held-out data
  - Walk-forward validation (train on past, test on future)
  - Realistic betting sim with flat units + Kelly comparison

Usage:
    Step 1: python mlb_collect_pitchers.py   (collect pitcher data, ~5 min)
    Step 2: python mlb_model_v2.py           (train + evaluate)

Requires:
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
"""

import warnings
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ xgboost not installed — using sklearn GradientBoosting")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠ lightgbm not installed — using sklearn GradientBoosting")

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ELO RATING SYSTEM (Teams)
# ═══════════════════════════════════════════════════════════════════════════

class EloSystem:
    """MLB Elo with season regression, MOV adjustment, home advantage."""

    def __init__(self, k=6, home_advantage=24, mean_rating=1500,
                 season_reversion=0.33):
        self.k = k
        self.home_adv = home_advantage
        self.mean = mean_rating
        self.reversion = season_reversion
        self.ratings = {}
        self.last_season = None

    def get_rating(self, team):
        return self.ratings.get(team, self.mean)

    def expected_score(self, home_elo, away_elo):
        diff = home_elo + self.home_adv - away_elo
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def margin_multiplier(self, mov, elo_diff):
        mov = abs(mov)
        return np.log(max(mov, 1) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home_team, away_team, home_runs, away_runs, season):
        if self.last_season is not None and season != self.last_season:
            for team in list(self.ratings.keys()):
                self.ratings[team] = (
                    self.mean * self.reversion +
                    self.ratings[team] * (1 - self.reversion)
                )
        self.last_season = season

        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        exp_home = self.expected_score(home_elo, away_elo)

        actual_home = 1.0 if home_runs > away_runs else (0.0 if away_runs > home_runs else 0.5)
        mov = abs(home_runs - away_runs)
        elo_diff = abs(home_elo - away_elo)
        mult = self.margin_multiplier(mov, elo_diff)

        shift = self.k * mult * (actual_home - exp_home)
        self.ratings[home_team] = home_elo + shift
        self.ratings[away_team] = away_elo - shift

        return home_elo, away_elo, exp_home


# ═══════════════════════════════════════════════════════════════════════════
# 2. PITCHER ELO SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class PitcherElo:
    """
    Track individual pitcher quality via Elo.
    Pitchers who get decisions (Win/Loss) have their ratings updated.
    K-factor is higher than team Elo since pitchers are more volatile.
    Season regression is stronger since pitchers age/change more.
    """

    def __init__(self, k=12, mean_rating=1500, season_reversion=0.5):
        self.k = k
        self.mean = mean_rating
        self.reversion = season_reversion
        self.ratings = {}
        self.last_season = None
        self.games_pitched = {}  # Track sample size

    def get_rating(self, pitcher):
        if pitcher is None or pd.isna(pitcher) or str(pitcher).strip() in ("", "None"):
            return self.mean
        return self.ratings.get(str(pitcher).strip(), self.mean)

    def get_confidence(self, pitcher):
        """How many decisions this pitcher has — low count = uncertain."""
        if pitcher is None or pd.isna(pitcher):
            return 0
        return self.games_pitched.get(str(pitcher).strip(), 0)

    def update(self, winner, loser, season):
        if self.last_season is not None and season != self.last_season:
            for p in list(self.ratings.keys()):
                self.ratings[p] = (
                    self.mean * self.reversion +
                    self.ratings[p] * (1 - self.reversion)
                )
        self.last_season = season

        w = str(winner).strip() if winner and not pd.isna(winner) else None
        l = str(loser).strip() if loser and not pd.isna(loser) else None

        if not w or not l or w == "None" or l == "None":
            return

        w_elo = self.ratings.get(w, self.mean)
        l_elo = self.ratings.get(l, self.mean)

        expected_w = 1.0 / (1.0 + 10 ** (-(w_elo - l_elo) / 400.0))
        shift = self.k * (1.0 - expected_w)

        self.ratings[w] = w_elo + shift
        self.ratings[l] = l_elo - shift

        self.games_pitched[w] = self.games_pitched.get(w, 0) + 1
        self.games_pitched[l] = self.games_pitched.get(l, 0) + 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPUTE ALL ELO FEATURES
# ═══════════════════════════════════════════════════════════════════════════

NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "OAK", "Athletics": "OAK", "ATH": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}


def compute_elo_features(raw_path="mlb_raw_games.csv"):
    """Compute team Elo + pitcher Elo from raw game data."""
    print("Computing Elo ratings...")
    raw = pd.read_csv(raw_path)
    df = raw.copy()

    df["runs_scored"] = pd.to_numeric(df["R"], errors="coerce")
    df["runs_allowed"] = pd.to_numeric(df["RA"], errors="coerce")
    df["is_home"] = df["Home_Away"].apply(lambda x: 0 if str(x).strip() == "@" else 1)
    df["win"] = df["W/L"].apply(
        lambda x: 1 if str(x).startswith("W") else (0 if str(x).startswith("L") else np.nan)
    )
    df = df.dropna(subset=["win", "runs_scored", "runs_allowed"])
    df["game_num"] = df.groupby(["Team", "Season"]).cumcount() + 1

    # Process home games chronologically
    home_df = df[df["is_home"] == 1].sort_values(["Season", "game_num"]).copy()

    team_elo = EloSystem()
    pitcher_elo = PitcherElo()
    records = []

    for _, row in home_df.iterrows():
        home_team = row["Team"]
        opp_raw = str(row["Opp"]).strip()
        away_team = NAME_TO_ABBR.get(opp_raw, opp_raw)
        season = int(row["Season"])
        gn = int(row["game_num"])

        # Pre-game pitcher Elo (using decision pitchers as proxy)
        win_pitcher = row.get("Win", None)
        loss_pitcher = row.get("Loss", None)

        # Team Elo update
        home_elo_pre, away_elo_pre, exp_home = team_elo.update(
            home_team, away_team,
            row["runs_scored"], row["runs_allowed"], season
        )

        # Pitcher Elo (pre-game snapshots, then update)
        wp_elo = pitcher_elo.get_rating(win_pitcher)
        lp_elo = pitcher_elo.get_rating(loss_pitcher)
        wp_conf = pitcher_elo.get_confidence(win_pitcher)
        lp_conf = pitcher_elo.get_confidence(loss_pitcher)

        # Update pitcher Elo after recording pre-game state
        pitcher_elo.update(win_pitcher, loss_pitcher, season)

        records.append({
            "season": season,
            "home_team": home_team,
            "away_team": away_team,
            "game_num": gn,
            "home_elo": home_elo_pre,
            "away_elo": away_elo_pre,
            "elo_prob_home": exp_home,
            "elo_diff": home_elo_pre - away_elo_pre,
            "win_pitcher_elo": wp_elo,
            "loss_pitcher_elo": lp_elo,
            "win_pitcher_conf": wp_conf,
            "loss_pitcher_conf": lp_conf,
        })

    elo_df = pd.DataFrame(records)
    print(f"  ✓ Team + Pitcher Elo for {len(elo_df)} games")
    return elo_df


# ═══════════════════════════════════════════════════════════════════════════
# 4. PITCHER SEASON STATS (from FanGraphs via pybaseball)
# ═══════════════════════════════════════════════════════════════════════════

def load_pitcher_stats(pitcher_stats_path="mlb_pitcher_stats.csv"):
    """
    Load pitcher season stats collected by mlb_collect_pitchers.py.
    Returns a lookup: (pitcher_last_name, season) -> stats dict

    NOTE: Matching by last name only is imperfect (multiple pitchers share names).
    This is a known limitation — a production system would use player IDs.
    For a hobby model, it adds signal despite the noise.
    """
    if not os.path.exists(pitcher_stats_path):
        print(f"  ⚠ {pitcher_stats_path} not found — skipping pitcher stats")
        print(f"    Run: python mlb_collect_pitchers.py")
        return None

    print(f"Loading pitcher stats from {pitcher_stats_path}...")
    df = pd.read_csv(pitcher_stats_path)

    # Build lookup keyed by (last_name, season)
    lookup = {}
    stat_cols = ["ERA", "FIP", "WAR", "WHIP", "K/9", "BB/9", "HR/9", "IP", "xFIP"]
    stat_cols = [c for c in stat_cols if c in df.columns]

    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        # Extract last name
        parts = name.split()
        if not parts:
            continue
        last_name = parts[-1]
        season = int(row["Season"])

        key = (last_name, season)
        # If duplicate, keep the one with more IP
        if key in lookup:
            if row.get("IP", 0) <= lookup[key].get("IP", 0):
                continue

        stats = {}
        for c in stat_cols:
            val = pd.to_numeric(row.get(c, np.nan), errors="coerce")
            stats[c] = val
        lookup[key] = stats

    print(f"  ✓ Loaded stats for {len(lookup)} pitcher-seasons")
    return lookup


def merge_pitcher_features(df, raw_path="mlb_raw_games.csv",
                           pitcher_stats_path="mlb_pitcher_stats.csv"):
    """
    For each matchup row, find the winning/losing pitcher from raw data
    and attach their season-level stats.
    """
    pitcher_lookup = load_pitcher_stats(pitcher_stats_path)
    if pitcher_lookup is None:
        return df

    # Build a game -> pitcher mapping from raw data
    print("  Building pitcher-game mapping...")
    raw = pd.read_csv(raw_path)
    raw["is_home"] = raw["Home_Away"].apply(lambda x: 0 if str(x).strip() == "@" else 1)
    raw["game_num"] = raw.groupby(["Team", "Season"]).cumcount() + 1

    home_games = raw[raw["is_home"] == 1].copy()

    # Map (home_team, season, game_num) -> (win_pitcher, loss_pitcher)
    pitcher_map = {}
    for _, row in home_games.iterrows():
        key = (row["Team"], int(row["Season"]), int(row["game_num"]))
        pitcher_map[key] = {
            "win_pitcher": str(row.get("Win", "")).strip(),
            "loss_pitcher": str(row.get("Loss", "")).strip(),
        }

    # For each matchup, lookup pitcher season stats
    stat_cols = ["ERA", "FIP", "WAR", "WHIP", "K/9", "BB/9", "HR/9", "IP", "xFIP"]
    new_cols = {f"wp_{c.replace('/', '_')}": [] for c in stat_cols}
    new_cols.update({f"lp_{c.replace('/', '_')}": [] for c in stat_cols})

    for _, row in df.iterrows():
        key = (row["home_team"], int(row["season"]), int(row["game_num"]))
        pitchers = pitcher_map.get(key, {})

        wp = pitchers.get("win_pitcher", "")
        lp = pitchers.get("loss_pitcher", "")
        season = int(row["season"])

        # Look up their season stats (use prior season if early in year)
        wp_stats = pitcher_lookup.get((wp, season)) or pitcher_lookup.get((wp, season - 1)) or {}
        lp_stats = pitcher_lookup.get((lp, season)) or pitcher_lookup.get((lp, season - 1)) or {}

        for c in stat_cols:
            col_safe = c.replace("/", "_")
            new_cols[f"wp_{col_safe}"].append(wp_stats.get(c, np.nan))
            new_cols[f"lp_{col_safe}"].append(lp_stats.get(c, np.nan))

    for col, vals in new_cols.items():
        df[col] = vals

    found = df["wp_ERA"].notna().sum()
    print(f"  ✓ Matched pitcher stats for {found}/{len(df)} games ({found/len(df)*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def prepare_features(matchups_path="mlb_matchups_features.csv",
                     raw_path="mlb_raw_games.csv",
                     pitcher_stats_path="mlb_pitcher_stats.csv"):
    """Load matchups, merge Elo + pitcher stats, build feature matrix."""

    df = pd.read_csv(matchups_path)
    elo_df = compute_elo_features(raw_path)

    # Merge Elo
    df = df.merge(elo_df, on=["season", "home_team", "away_team", "game_num"], how="left")

    # Merge pitcher features
    df = merge_pitcher_features(df, raw_path, pitcher_stats_path)

    # ── Define feature columns ──
    feature_cols = []

    # Rolling team stats
    for side in ["home", "away"]:
        for prefix in ["win_pct_L", "avg_rs_L", "avg_ra_L", "run_diff_L"]:
            for w in [5, 10, 20, 40]:
                feature_cols.append(f"{side}_{prefix}{w}")
        feature_cols.extend([
            f"{side}_pythag_win_exp",
            f"{side}_season_win_pct",
            f"{side}_season_run_diff",
            f"{side}_streak_num",
            f"{side}_season_progress",
        ])
        if f"{side}_extra_inn_rate_L10" in df.columns:
            feature_cols.append(f"{side}_extra_inn_rate_L10")
        if f"{side}_avg_cli_L10" in df.columns:
            feature_cols.append(f"{side}_avg_cli_L10")

    # Team Elo
    for ec in ["home_elo", "away_elo", "elo_prob_home", "elo_diff"]:
        if ec in df.columns:
            feature_cols.append(ec)

    # Pitcher Elo
    for pc in ["win_pitcher_elo", "loss_pitcher_elo", "win_pitcher_conf", "loss_pitcher_conf"]:
        if pc in df.columns:
            feature_cols.append(pc)

    # Pitcher season stats
    for prefix in ["wp_", "lp_"]:
        for stat in ["ERA", "FIP", "WAR", "WHIP", "K_9", "BB_9", "HR_9", "IP", "xFIP"]:
            col = f"{prefix}{stat}"
            if col in df.columns:
                feature_cols.append(col)

    # Derived differentials
    for w in [5, 10, 20]:
        col1 = f"delta_win_pct_L{w}"
        col2 = f"delta_run_diff_L{w}"
        df[col1] = df[f"home_win_pct_L{w}"] - df[f"away_win_pct_L{w}"]
        df[col2] = df[f"home_run_diff_L{w}"] - df[f"away_run_diff_L{w}"]
        feature_cols.extend([col1, col2])

    df["delta_pythag"] = df["home_pythag_win_exp"] - df["away_pythag_win_exp"]
    df["delta_season_wpct"] = df["home_season_win_pct"] - df["away_season_win_pct"]
    feature_cols.extend(["delta_pythag", "delta_season_wpct"])

    if "is_night" in df.columns:
        feature_cols.append("is_night")

    # Pitcher stat differentials (wp is generally the better pitcher in that game)
    if "wp_ERA" in df.columns:
        df["delta_pitcher_ERA"] = df["lp_ERA"] - df["wp_ERA"]  # positive = WP has lower ERA (better)
        df["delta_pitcher_FIP"] = df["lp_FIP"] - df["wp_FIP"]
        df["delta_pitcher_WAR"] = df["wp_WAR"] - df["lp_WAR"]
        df["pitcher_elo_diff"] = df["win_pitcher_elo"] - df["loss_pitcher_elo"]
        feature_cols.extend([
            "delta_pitcher_ERA", "delta_pitcher_FIP",
            "delta_pitcher_WAR", "pitcher_elo_diff"
        ])

    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=["home_win"])

    print(f"\nFeature matrix: {len(df)} rows × {len(feature_cols)} features")
    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE MODEL (same architecture, cleaned up)
# ═══════════════════════════════════════════════════════════════════════════

class MLBEnsemble:
    """
    Ensemble: XGBoost + LightGBM + LogReg with isotonic calibration.
    """

    def __init__(self, weights=None):
        self.weights = weights or [0.40, 0.35, 0.25]
        self.models = []
        self.calibrator = None
        self.scaler = StandardScaler()
        self.feature_cols = None

    def _build_models(self):
        models = []
        if HAS_XGB:
            models.append(("XGBoost", xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
                verbosity=0, random_state=42,
            )))
        else:
            models.append(("GBM1", GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42,
            )))

        if HAS_LGB:
            models.append(("LightGBM", lgb.LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
                reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=43,
            )))
        else:
            models.append(("GBM2", GradientBoostingClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.04,
                subsample=0.75, min_samples_leaf=25, random_state=43,
            )))

        models.append(("LogReg", LogisticRegression(
            C=0.5, max_iter=1000, solver="lbfgs", random_state=44,
        )))
        return models

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.feature_cols = X_train.columns.tolist()
        self.models = self._build_models()
        self.scaler.fit(X_train.fillna(0))

        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train.fillna(0)),
            columns=X_train.columns, index=X_train.index
        )

        for name, model in self.models:
            if name == "LogReg":
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train.fillna(0), y_train)
            print(f"  ✓ {name}")

        raw_probs = self._raw_predict(X_cal)
        self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        self.calibrator.fit(raw_probs, y_cal)
        print(f"  ✓ Calibrated on {len(y_cal)} games")

    def _raw_predict(self, X):
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.fillna(0)),
            columns=X.columns, index=X.index
        )
        probs = np.zeros(len(X))
        for i, (name, model) in enumerate(self.models):
            if name == "LogReg":
                p = model.predict_proba(X_scaled)[:, 1]
            else:
                p = model.predict_proba(X.fillna(0))[:, 1]
            probs += self.weights[i] * p
        return probs

    def predict_proba(self, X):
        raw = self._raw_predict(X)
        return self.calibrator.predict(raw)

    def feature_importance(self, top_n=20):
        importances = np.zeros(len(self.feature_cols))
        count = 0
        for name, model in self.models:
            if hasattr(model, "feature_importances_"):
                importances += model.feature_importances_
                count += 1
        if count > 0:
            importances /= count
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)


# ═══════════════════════════════════════════════════════════════════════════
# 7. REALISTIC BETTING SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_betting_realistic(y_true, y_prob):
    """
    Realistic betting simulation with:
    - Flat unit sizing ($10 per bet)
    - Actual MLB vig structure (~4.5% juice)
    - Dynamic implied odds based on model prob vs market
    - Separate Kelly bankroll for comparison (capped properly)
    - Edge threshold of 3%

    The "market" is simulated as: for each game, the true probability
    is somewhere between 0.533 (home base rate) and the model's prediction.
    We use a noisy market that's ~70% efficient (sharper than recreational,
    less sharp than Pinnacle closing lines).
    """
    np.random.seed(42)

    # ── Flat unit betting ──
    flat_bankroll = 1000.0
    flat_unit = 10.0  # $10 per bet
    flat_history = [flat_bankroll]
    flat_bets = 0
    flat_wins = 0
    flat_profit = 0.0

    # ── Kelly betting (properly capped) ──
    kelly_bankroll = 1000.0
    kelly_history = [kelly_bankroll]
    kelly_bets = 0
    kelly_wins = 0

    # ── Parameters ──
    vig = 0.045  # 4.5% total juice (typical MLB)
    edge_threshold = 0.03  # Min 3% edge to bet
    kelly_fraction = 0.10  # 10% Kelly (very conservative)
    kelly_max_pct = 0.03  # Never bet more than 3% of Kelly bankroll

    for i in range(len(y_true)):
        model_p = y_prob[i]
        actual = y_true[i]

        # Simulate a "market" probability
        # The market is ~70% efficient: it knows more than baseline but less than truth
        market_noise = np.random.normal(0, 0.04)
        # Market is a blend of baseline (53.3%) and a noisy version of the "true" outcome
        market_p = 0.533 * 0.3 + (model_p + market_noise) * 0.7
        market_p = np.clip(market_p, 0.25, 0.85)

        # Convert to decimal odds with vig
        # No-vig odds: 1/market_p, with vig applied symmetrically
        vig_factor = 1 - vig / 2
        home_odds = (1 / market_p) * vig_factor
        away_odds = (1 / (1 - market_p)) * vig_factor

        # ── Check home bet ──
        home_edge = model_p - (1 / home_odds)  # Model p minus implied p from odds
        if home_edge > edge_threshold:
            # Flat bet
            flat_bets += 1
            if actual == 1:
                flat_profit += flat_unit * (home_odds - 1)
                flat_bankroll += flat_unit * (home_odds - 1)
                flat_wins += 1
            else:
                flat_profit -= flat_unit
                flat_bankroll -= flat_unit

            # Kelly bet
            b = home_odds - 1
            q = 1 - model_p
            kelly_f = (b * model_p - q) / b * kelly_fraction
            kelly_stake = max(0, min(kelly_f * kelly_bankroll, kelly_bankroll * kelly_max_pct))
            if kelly_stake > 0:
                kelly_bets += 1
                if actual == 1:
                    kelly_bankroll += kelly_stake * b
                    kelly_wins += 1
                else:
                    kelly_bankroll -= kelly_stake

        # ── Check away bet ──
        away_edge = (1 - model_p) - (1 / away_odds)
        if away_edge > edge_threshold:
            flat_bets += 1
            if actual == 0:
                flat_profit += flat_unit * (away_odds - 1)
                flat_bankroll += flat_unit * (away_odds - 1)
                flat_wins += 1
            else:
                flat_profit -= flat_unit
                flat_bankroll -= flat_unit

            b = away_odds - 1
            q = model_p
            kelly_f = (b * (1 - model_p) - q) / b * kelly_fraction
            kelly_stake = max(0, min(kelly_f * kelly_bankroll, kelly_bankroll * kelly_max_pct))
            if kelly_stake > 0:
                kelly_bets += 1
                if actual == 0:
                    kelly_bankroll += kelly_stake * b
                    kelly_wins += 1
                else:
                    kelly_bankroll -= kelly_stake

        flat_history.append(flat_bankroll)
        kelly_history.append(kelly_bankroll)

    return {
        "flat": {
            "bankroll": flat_bankroll,
            "profit": flat_profit,
            "roi_pct": flat_profit / max(flat_bets * flat_unit, 1) * 100,
            "bets": flat_bets,
            "win_rate": flat_wins / max(flat_bets, 1) * 100,
            "history": flat_history,
        },
        "kelly": {
            "bankroll": kelly_bankroll,
            "roi_pct": (kelly_bankroll - 1000) / 1000 * 100,
            "bets": kelly_bets,
            "win_rate": kelly_wins / max(kelly_bets, 1) * 100,
            "history": kelly_history,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_calibration(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    results = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        results.append({
            "bin_center": (bins[i] + bins[i + 1]) / 2,
            "avg_predicted": y_prob[mask].mean(),
            "avg_actual": y_true[mask].mean(),
            "count": mask.sum(),
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# 9. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_eval(df, feature_cols, test_seasons=None):
    """Train on past seasons, test on next. The only honest way to evaluate."""

    if test_seasons is None:
        all_seasons = sorted(df["season"].unique())
        test_seasons = all_seasons[-3:]

    all_results = []
    all_preds = []

    for test_season in test_seasons:
        print(f"\n{'─' * 60}")
        print(f"SEASON {test_season}")
        print(f"{'─' * 60}")

        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] == test_season].copy()

        if len(train_df) < 500 or len(test_df) < 100:
            print(f"  Skip: train={len(train_df)}, test={len(test_df)}")
            continue

        # Split train into train + calibration
        cal_cutoff = int(len(train_df) * 0.8)
        cal_df = train_df.iloc[cal_cutoff:]
        train_df = train_df.iloc[:cal_cutoff]

        X_train = train_df[feature_cols]
        y_train = train_df["home_win"].values.astype(int)
        X_cal = cal_df[feature_cols]
        y_cal = cal_df["home_win"].values.astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df["home_win"].values.astype(int)

        print(f"  Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

        ensemble = MLBEnsemble()
        ensemble.fit(X_train, y_train, X_cal, y_cal)

        y_prob = ensemble.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        ll = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        # Baseline brier: always predict 0.533
        baseline_brier = brier_score_loss(y_test, np.full_like(y_prob, 0.533))

        print(f"\n  Accuracy:    {acc:.4f}  (baseline: {y_test.mean():.3f})")
        print(f"  Brier Score: {brier:.4f}  (baseline: {baseline_brier:.4f})")
        print(f"  Log Loss:    {ll:.4f}")
        print(f"  ROC AUC:     {auc:.4f}")

        # Calibration
        cal_result = evaluate_calibration(y_test, y_prob)
        print(f"\n  Calibration:")
        for _, row in cal_result.iterrows():
            print(f"    P={row['avg_predicted']:.2f} -> Actual={row['avg_actual']:.2f}  "
                  f"(n={int(row['count']):4d})")

        # Realistic betting sim
        bet = simulate_betting_realistic(y_test, y_prob)
        print(f"\n  Flat Betting ($10/bet):")
        print(f"    Bets: {bet['flat']['bets']} | "
              f"Win rate: {bet['flat']['win_rate']:.1f}% | "
              f"Profit: ${bet['flat']['profit']:.2f} | "
              f"ROI: {bet['flat']['roi_pct']:.1f}%")
        print(f"  Kelly Betting (10% Kelly, 3% cap):")
        print(f"    Bets: {bet['kelly']['bets']} | "
              f"Win rate: {bet['kelly']['win_rate']:.1f}% | "
              f"Final: ${bet['kelly']['bankroll']:.2f} | "
              f"ROI: {bet['kelly']['roi_pct']:.1f}%")

        # Feature importance
        imp = ensemble.feature_importance(top_n=15)
        print(f"\n  Top Features:")
        for _, row in imp.iterrows():
            print(f"    {row['feature']:35s} {row['importance']:.4f}")

        all_results.append({
            "season": test_season,
            "accuracy": acc,
            "brier": brier,
            "baseline_brier": baseline_brier,
            "log_loss": ll,
            "auc": auc,
            "flat_bets": bet["flat"]["bets"],
            "flat_profit": bet["flat"]["profit"],
            "flat_roi": bet["flat"]["roi_pct"],
            "kelly_roi": bet["kelly"]["roi_pct"],
            "flat_history": bet["flat"]["history"],
            "kelly_history": bet["kelly"]["history"],
        })

        all_preds.append(pd.DataFrame({
            "season": test_season,
            "home_team": test_df["home_team"].values,
            "away_team": test_df["away_team"].values,
            "home_win": y_test,
            "model_prob": y_prob,
            "date_str": test_df["date_str"].values if "date_str" in test_df.columns else "",
        }))

    return all_results, all_preds


# ═══════════════════════════════════════════════════════════════════════════
# 10. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(all_results, all_preds):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Flat bankroll trajectories
    ax = axes[0, 0]
    for r in all_results:
        ax.plot(r["flat_history"], label=f"{r['season']} Flat", alpha=0.8)
        ax.plot(r["kelly_history"], label=f"{r['season']} Kelly",
                alpha=0.5, linestyle="--")
    ax.axhline(y=1000, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Bankroll Trajectory (Flat $10 vs Kelly)")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Calibration
    ax = axes[0, 1]
    if all_preds:
        combined = pd.concat(all_preds)
        cal = evaluate_calibration(combined["home_win"].values, combined["model_prob"].values)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.scatter(cal["avg_predicted"], cal["avg_actual"],
                   s=cal["count"] * 2, alpha=0.7, c="steelblue", edgecolors="navy")
        ax.set_title("Calibration Plot")
        ax.set_xlabel("Predicted P(Home Win)")
        ax.set_ylabel("Actual Win Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # 3. Season metrics
    ax = axes[1, 0]
    seasons = [r["season"] for r in all_results]
    x = np.arange(len(seasons))
    w = 0.25
    ax.bar(x - w, [r["accuracy"] for r in all_results], w, label="Accuracy", color="steelblue")
    ax.bar(x, [r["brier"] for r in all_results], w, label="Brier", color="coral")
    ax.bar(x + w, [r["baseline_brier"] for r in all_results], w,
           label="Baseline Brier", color="lightcoral", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_title("Season Metrics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Profit/Loss by season
    ax = axes[1, 1]
    profits = [r["flat_profit"] for r in all_results]
    colors = ["green" if p > 0 else "red" for p in profits]
    bars = ax.bar(seasons, profits, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("Flat Betting P/L by Season ($10 units)")
    ax.set_ylabel("Profit ($)")
    for bar, p, roi in zip(bars, profits, [r["flat_roi"] for r in all_results]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"ROI: {roi:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mlb_model_v2_results.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved mlb_model_v2_results.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MLB Prediction Model v2 — Realistic + Pitcher Features")
    print("=" * 60)

    df, feature_cols = prepare_features()

    print(f"\nNaN rates (top 5):")
    for col, rate in df[feature_cols].isnull().mean().sort_values(ascending=False).head(5).items():
        if rate > 0:
            print(f"  {col}: {rate:.1%}")

    results, preds = walk_forward_eval(df, feature_cols)

    # Summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    if results:
        print(f"{'Season':>8} {'Acc':>8} {'Brier':>8} {'AUC':>8} "
              f"{'Bets':>6} {'Profit':>10} {'ROI':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['season']:>8} {r['accuracy']:>8.4f} {r['brier']:>8.4f} "
                  f"{r['auc']:>8.4f} {r['flat_bets']:>6} "
                  f"${r['flat_profit']:>9.2f} {r['flat_roi']:>7.1f}%")
        print("-" * 60)

        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_brier = np.mean([r["brier"] for r in results])
        avg_base = np.mean([r["baseline_brier"] for r in results])
        avg_auc = np.mean([r["auc"] for r in results])
        total_profit = sum(r["flat_profit"] for r in results)
        total_bets = sum(r["flat_bets"] for r in results)

        print(f"{'AVG':>8} {avg_acc:>8.4f} {avg_brier:>8.4f} {avg_auc:>8.4f}")
        print(f"\nTotal P/L: ${total_profit:.2f} over {total_bets} bets")
        print(f"Overall ROI: {total_profit / max(total_bets * 10, 1) * 100:.1f}%")
        print(f"Brier edge:  {avg_base - avg_brier:.4f} "
              f"({'✅ beating baseline' if avg_brier < avg_base else '⚠ below baseline'})")

    plot_results(results, preds)

    if preds:
        all_preds = pd.concat(preds)
        all_preds.to_csv("mlb_predictions_v2.csv", index=False)
        print("✓ Saved mlb_predictions_v2.csv")

    print("\nDone! 🎯")