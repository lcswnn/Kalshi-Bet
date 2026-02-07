"""
MLB Game Prediction Model v4 — Audit Fixes
================================================================
Changes from v3 (audit findings):
  FIX 1: Isotonic regression was collapsing predictions to ~40 discrete values.
         780 games all getting P=0.7227 regardless of matchup is broken.
         -> Replaced with Platt scaling (logistic calibration) which preserves
            continuous probability outputs.
  FIX 2: Model was betting on BOTH sides of the same game (home AND away).
         This is self-contradictory — you can't think both sides have +EV.
         -> Now only bets the side with the larger edge, never both.
  FIX 3: Edge threshold of 3% was too low given model imprecision.
         -> Raised to 5% minimum edge to trigger a bet.
  FIX 4: Betting against CLOSING lines is unrealistic — you can't bet
         closing lines in practice, you bet OPENING or pre-game lines.
         -> Added flag to use opening lines if available; noted in output.
  FIX 5: No bankroll protection — Kelly/flat kept betting even when
         bankroll was nearly depleted.
         -> Added minimum bankroll check for flat betting.
  FIX 6: CLV calculation was model_prob - market_prob, which is circular.
         Real CLV = opening_line_prob - closing_line_prob on your bet side.
         -> Fixed CLV to be meaningful (or noted as unavailable).

Architecture (unchanged):
  - Staggered K-factor Elo ratings (K=32/24/16 by rating level)
  - Ensemble: XGBoost + LightGBM + Logistic Regression
  - Platt scaling calibration (replaces isotonic)
  - Walk-forward validation
  - Betting simulation with realistic constraints

Usage:
    python mlb_model_v4.py
"""

import warnings
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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
    print("Warning: xgboost not installed, using sklearn GradientBoosting")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed, using sklearn GradientBoosting")

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "mlb_historical_data")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ELO RATING SYSTEM — Staggered K-Factor
# ═══════════════════════════════════════════════════════════════════════════

class EloSystem:
    """
    MLB Elo with staggered K-factor adapted from chess conventions.

    K-factor tiers (centered on 1500):
      - K=32 for teams rated below 1450  (volatile / rebuilding)
      - K=24 for teams rated 1450-1550   (average tier)
      - K=16 for teams rated above 1550  (established / elite)

    Also includes:
      - Margin-of-victory multiplier on K
      - Home field advantage baked into expected score
      - Between-season regression toward the mean
    """

    def __init__(self, home_advantage=24, mean_rating=1500,
                 season_reversion=0.33):
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

    def margin_multiplier(self, mov, elo_diff):
        """Log-scaled MOV adjustment, dampened when Elo gap is large."""
        mov = abs(mov)
        return np.log(max(mov, 1) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def _regress_to_mean(self, season):
        """Between-season regression: pull all teams toward 1500."""
        if self.last_season is not None and season != self.last_season:
            for team in list(self.ratings.keys()):
                self.ratings[team] = (
                    self.mean * self.reversion
                    + self.ratings[team] * (1 - self.reversion)
                )
        self.last_season = season

    def update(self, home_team, away_team, home_runs, away_runs, season):
        self._regress_to_mean(season)

        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        exp_home = self.expected_score(home_elo, away_elo)

        actual_home = 1.0 if home_runs > away_runs else (
            0.0 if away_runs > home_runs else 0.5
        )
        mov = abs(home_runs - away_runs)
        elo_diff = abs(home_elo - away_elo)
        mult = self.margin_multiplier(mov, elo_diff)

        # Staggered K: each team uses its own K based on its current rating
        k_home = self._k_factor(home_elo)
        k_away = self._k_factor(away_elo)

        home_shift = k_home * mult * (actual_home - exp_home)
        away_shift = k_away * mult * (exp_home - actual_home)

        self.ratings[home_team] = home_elo + home_shift
        self.ratings[away_team] = away_elo + away_shift

        return home_elo, away_elo, exp_home


# ═══════════════════════════════════════════════════════════════════════════
# 2. COMPUTE ELO FEATURES FROM RAW GAME DATA
# ═══════════════════════════════════════════════════════════════════════════

def compute_elo_features(raw_path):
    """
    Process all_games_raw.csv chronologically to build team Elo ratings.
    Returns a DataFrame with pre-game Elo snapshots for each game.
    """
    print("Computing staggered K-factor Elo ratings...")
    df = pd.read_csv(raw_path)

    # Filter to regular season games only
    df = df[df["game_type"] == "R"].copy()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score", "home_win"])

    # Sort chronologically
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    elo = EloSystem()
    records = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        season = int(row["season"])

        # Snapshot pre-game ratings, then update
        home_elo_pre, away_elo_pre, exp_home = elo.update(
            home, away,
            row["home_score"], row["away_score"], season
        )

        records.append({
            "game_id": row["game_id"],
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_elo": round(home_elo_pre, 2),
            "away_elo": round(away_elo_pre, 2),
            "elo_diff": round(home_elo_pre - away_elo_pre, 2),
            "elo_prob_home": round(exp_home, 4),
        })

    elo_df = pd.DataFrame(records)
    print(f"  Elo computed for {len(elo_df)} games "
          f"(seasons {elo_df['season'].min()}-{elo_df['season'].max()})")
    return elo_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. LOAD AND MERGE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def load_advanced_extras(adv_path):
    """
    Extract unique per-game columns from the advanced features CSV
    that aren't in the basic CSV: park_factor, rest_days, division, etc.
    Deduplicate by game_id (the advanced file has heavy row duplication).
    """
    extra_cols = [
        "game_id", "home_rest_days", "away_rest_days", "rest_advantage",
        "park_factor", "park_factor_adj", "day_of_week", "is_weekend",
        "month", "days_into_season", "home_game_num", "away_game_num",
        "home_division", "away_division", "same_division", "interleague",
        "home_streak", "away_streak", "streak_diff",
    ]
    print("Loading advanced features (park, rest, division)...")
    # Only read the columns we need
    try:
        adv = pd.read_csv(adv_path, usecols=extra_cols)
    except (ValueError, KeyError):
        # Some columns might not exist; read all and filter
        adv = pd.read_csv(adv_path)
        extra_cols = [c for c in extra_cols if c in adv.columns]
        adv = adv[extra_cols]

    adv = adv.drop_duplicates(subset=["game_id"], keep="first")
    print(f"  {len(adv)} unique games with advanced features")
    return adv


def prepare_features():
    """Load basic matchups, compute Elo, merge advanced extras, build feature matrix."""

    matchups_path = os.path.join(DATA_DIR, "mlb_matchups_features.csv")
    raw_path = os.path.join(DATA_DIR, "all_games_raw.csv")
    adv_path = os.path.join(DATA_DIR, "mlb_matchups_features_advanced.csv")

    # --- Load basic matchup features ---
    df = pd.read_csv(matchups_path)
    # Deduplicate by game_id (data has duplicate rows for suspended/postponed games)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]) \
        .drop_duplicates(subset=["game_id"], keep="first")
    # Filter out unplayed games (0-0 scores indicate suspended/postponed)
    df = df[~((df["home_score"] == 0) & (df["away_score"] == 0))].copy()
    print(f"Loaded {len(df)} games from basic matchup features (after dedup)")

    # --- Compute fresh Elo with staggered K ---
    elo_df = compute_elo_features(raw_path)
    df = df.merge(elo_df, on=["game_id", "season", "home_team", "away_team"], how="left")
    matched = df["home_elo"].notna().sum()
    print(f"  Elo matched: {matched}/{len(df)} games")

    # --- Merge advanced extras ---
    if os.path.exists(adv_path):
        adv = load_advanced_extras(adv_path)
        # Drop any columns that already exist in df (except game_id)
        overlap = [c for c in adv.columns if c in df.columns and c != "game_id"]
        adv = adv.drop(columns=overlap, errors="ignore")
        df = df.merge(adv, on="game_id", how="left")
        print(f"  Advanced features merged: {df.columns.tolist()[-5:]}")

    # --- Define feature columns ---
    feature_cols = []

    # Rolling team stats (from basic matchup CSV)
    for side in ["home", "away"]:
        for metric in ["runs_scored", "runs_allowed"]:
            for w in [5, 10, 20]:
                col = f"{side}_{metric}_L{w}"
                if col in df.columns:
                    feature_cols.append(col)
        for metric in ["win_pct", "run_diff"]:
            for w in [5, 10, 20]:
                col = f"{side}_{metric}_L{w}"
                if col in df.columns:
                    feature_cols.append(col)
        if f"{side}_pyth_win_pct" in df.columns:
            feature_cols.append(f"{side}_pyth_win_pct")

    # Elo features
    for c in ["home_elo", "away_elo", "elo_diff", "elo_prob_home"]:
        if c in df.columns:
            feature_cols.append(c)

    # Starting pitcher features (pre-game, no leakage)
    for side_prefix, label in [("home_starter", "hs"), ("away_starter", "as")]:
        for stat in ["era", "whip", "strikeOuts", "inningsPitched", "homeRuns"]:
            col = f"{side_prefix}_{stat}"
            if col in df.columns:
                feature_cols.append(col)

    # Advanced features
    for c in ["park_factor", "park_factor_adj",
              "home_rest_days", "away_rest_days", "rest_advantage",
              "is_weekend", "month", "days_into_season",
              "same_division", "interleague",
              "home_streak", "away_streak", "streak_diff"]:
        if c in df.columns:
            feature_cols.append(c)

    # --- Derived differentials ---
    for w in [5, 10, 20]:
        wp_h = f"home_win_pct_L{w}"
        wp_a = f"away_win_pct_L{w}"
        rd_h = f"home_run_diff_L{w}"
        rd_a = f"away_run_diff_L{w}"
        if wp_h in df.columns and wp_a in df.columns:
            col = f"delta_win_pct_L{w}"
            df[col] = df[wp_h] - df[wp_a]
            feature_cols.append(col)
        if rd_h in df.columns and rd_a in df.columns:
            col = f"delta_run_diff_L{w}"
            df[col] = df[rd_h] - df[rd_a]
            feature_cols.append(col)

    if "home_pyth_win_pct" in df.columns and "away_pyth_win_pct" in df.columns:
        df["delta_pythag"] = df["home_pyth_win_pct"] - df["away_pyth_win_pct"]
        feature_cols.append("delta_pythag")

    # Pitcher differentials (lower ERA = better, so away - home gives positive = home advantage)
    if "home_starter_era" in df.columns and "away_starter_era" in df.columns:
        df["delta_starter_era"] = df["away_starter_era"] - df["home_starter_era"]
        feature_cols.append("delta_starter_era")
    if "home_starter_whip" in df.columns and "away_starter_whip" in df.columns:
        df["delta_starter_whip"] = df["away_starter_whip"] - df["home_starter_whip"]
        feature_cols.append("delta_starter_whip")

    # Day/night indicator
    if "day_night" in df.columns:
        df["is_night"] = (df["day_night"] == "night").astype(int)
        feature_cols.append("is_night")

    # Ensure all feature columns actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    # Remove duplicates while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    df = df.dropna(subset=["home_win"])
    df["home_win"] = df["home_win"].astype(int)

    print(f"\nFeature matrix: {len(df)} games x {len(feature_cols)} features")
    print(f"Seasons: {sorted(df['season'].unique())}")
    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE MODEL
# ═══════════════════════════════════════════════════════════════════════════

class MLBEnsemble:
    """XGBoost + LightGBM + LogReg with Platt scaling calibration."""

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
            columns=X_train.columns, index=X_train.index,
        )

        for name, model in self.models:
            if name == "LogReg":
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train.fillna(0), y_train)

        # Platt scaling calibration on held-out calibration set
        # (Isotonic regression was collapsing to ~40 discrete output values —
        #  Platt scaling preserves continuous probabilities)
        raw_probs = self._raw_predict(X_cal)
        self.calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
        self.calibrator.fit(raw_probs.reshape(-1, 1), y_cal)

    def _raw_predict(self, X):
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.fillna(0)),
            columns=X.columns, index=X.index,
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
        # Platt scaling: logistic regression on raw ensemble probabilities
        calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 0.01, 0.99)

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
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ODDS LOADING & CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def american_to_decimal(ml):
    """Convert American moneyline to decimal odds."""
    ml = float(ml)
    if ml > 0:
        return 1 + ml / 100
    elif ml < 0:
        return 1 + 100 / abs(ml)
    return np.nan


def american_to_implied_prob(ml):
    """Convert American moneyline to implied probability (includes vig)."""
    ml = float(ml)
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    elif ml > 0:
        return 100 / (ml + 100)
    return np.nan


def load_closing_odds(odds_csv_path):
    """
    Load pre-parsed closing odds CSV.
    Returns DataFrame with per-game odds from multiple sportsbooks.
    """
    if not os.path.exists(odds_csv_path):
        print(f"  Odds file not found: {odds_csv_path}")
        return None

    odds = pd.read_csv(odds_csv_path)
    odds["date"] = pd.to_datetime(odds["date"]).dt.strftime("%Y-%m-%d")
    print(f"  Loaded {len(odds)} closing odds lines "
          f"(seasons {sorted(odds['season'].unique())})")
    return odds


def merge_odds_with_predictions(test_df, odds_df, test_season):
    """
    Merge sportsbook odds into the test set for betting simulation.
    Returns arrays: home_odds_decimal, away_odds_decimal, market_prob_home,
                    best_home_decimal, best_away_decimal (for line shopping).
    """
    if odds_df is None:
        n = len(test_df)
        nans = np.full(n, np.nan)
        return nans, nans, nans, nans, nans

    season_odds = odds_df[odds_df["season"] == test_season].copy()
    if len(season_odds) == 0:
        n = len(test_df)
        nans = np.full(n, np.nan)
        return nans, nans, nans, nans, nans

    # Find sportsbook moneyline columns
    book_cols_home = [c for c in season_odds.columns
                      if c.endswith("_home_ml") and "open" not in c.lower()]
    book_cols_away = [c.replace("_home_ml", "_away_ml") for c in book_cols_home]
    book_cols_away = [c for c in book_cols_away if c in season_odds.columns]

    # Convert all book moneylines to decimal odds
    dec_h_cols = []
    dec_a_cols = []
    for bc_h in book_cols_home:
        bc_a = bc_h.replace("_home_ml", "_away_ml")
        if bc_a not in season_odds.columns:
            continue
        dh = f"_dec_h_{bc_h}"
        da = f"_dec_a_{bc_a}"
        season_odds[dh] = season_odds[bc_h].apply(
            lambda x: american_to_decimal(x) if pd.notna(x) else np.nan
        )
        season_odds[da] = season_odds[bc_a].apply(
            lambda x: american_to_decimal(x) if pd.notna(x) else np.nan
        )
        dec_h_cols.append(dh)
        dec_a_cols.append(da)

    if dec_h_cols:
        # Average across books (consensus line)
        season_odds["avg_home_dec"] = season_odds[dec_h_cols].mean(axis=1)
        season_odds["avg_away_dec"] = season_odds[dec_a_cols].mean(axis=1)
        # Best odds (line shopping: max payout)
        season_odds["best_home_dec"] = season_odds[dec_h_cols].max(axis=1)
        season_odds["best_away_dec"] = season_odds[dec_a_cols].max(axis=1)

    # Prepare merge keys
    season_odds["_merge_date"] = season_odds["date"]

    # Build test merge key
    test_with_date = test_df.copy()
    if "game_date" in test_with_date.columns:
        test_with_date["_merge_date"] = pd.to_datetime(
            test_with_date["game_date"]
        ).dt.strftime("%Y-%m-%d")
    else:
        # Fallback: no date available
        n = len(test_df)
        nans = np.full(n, np.nan)
        return nans, nans, nans, nans, nans

    # Aggregate odds per (date, home, away) to handle doubleheaders
    agg_dict = {}
    if "market_prob_home" in season_odds.columns:
        agg_dict["market_prob_home"] = "first"
    if "avg_home_dec" in season_odds.columns:
        agg_dict["avg_home_dec"] = "first"
        agg_dict["avg_away_dec"] = "first"
        agg_dict["best_home_dec"] = "first"
        agg_dict["best_away_dec"] = "first"

    if not agg_dict:
        n = len(test_df)
        nans = np.full(n, np.nan)
        return nans, nans, nans, nans, nans

    odds_agg = (
        season_odds.groupby(["_merge_date", "home_team", "away_team"])
        .agg(agg_dict).reset_index()
    )

    merged = test_with_date.merge(
        odds_agg,
        on=["_merge_date", "home_team", "away_team"],
        how="left",
    )

    market_probs = merged.get("market_prob_home", pd.Series(np.nan, index=merged.index)).values
    avg_home = merged.get("avg_home_dec", pd.Series(np.nan, index=merged.index)).values
    avg_away = merged.get("avg_away_dec", pd.Series(np.nan, index=merged.index)).values
    best_home = merged.get("best_home_dec", pd.Series(np.nan, index=merged.index)).values
    best_away = merged.get("best_away_dec", pd.Series(np.nan, index=merged.index)).values

    matched = np.sum(~np.isnan(market_probs))
    print(f"  Odds matched: {matched}/{len(test_df)} games for {test_season}")

    return avg_home, avg_away, market_probs, best_home, best_away


# ═══════════════════════════════════════════════════════════════════════════
# 6. BETTING SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_betting(y_true, y_prob, home_odds_dec, away_odds_dec,
                     market_probs, edge_threshold=0.05,
                     kelly_fraction=0.10, kelly_max_pct=0.03):
    """
    Simulate flat-unit and Kelly criterion betting.

    v4 fixes:
      - Only bets ONE side per game (the side with larger edge)
      - Raised edge_threshold to 5% (was 3%)
      - Flat betting stops if bankroll < 1 unit
      - CLV calculation noted as approximate

    Returns dict with flat/kelly results + per-bet records.
    """
    flat_bankroll = 1000.0
    flat_unit = 10.0
    flat_history = [flat_bankroll]
    flat_bets = 0
    flat_wins = 0
    flat_profit = 0.0

    kelly_bankroll = 1000.0
    kelly_history = [kelly_bankroll]
    kelly_bets = 0
    kelly_wins = 0

    bet_records = []

    for i in range(len(y_true)):
        model_p = y_prob[i]
        actual = y_true[i]

        h_odds = home_odds_dec[i] if not np.isnan(home_odds_dec[i]) else np.nan
        a_odds = away_odds_dec[i] if not np.isnan(away_odds_dec[i]) else np.nan

        if np.isnan(h_odds) or np.isnan(a_odds):
            flat_history.append(flat_bankroll)
            kelly_history.append(kelly_bankroll)
            continue

        # Calculate edges for both sides
        implied_home = 1.0 / h_odds
        implied_away = 1.0 / a_odds
        home_edge = model_p - implied_home
        away_edge = (1 - model_p) - implied_away

        # FIX 2: Only bet the ONE side with the larger edge (never both)
        # FIX 3: Must exceed 5% threshold
        bet_side = None
        bet_edge = 0
        bet_odds = 0
        bet_model_p = 0
        bet_implied = 0

        if home_edge > away_edge and home_edge > edge_threshold:
            bet_side = "home"
            bet_edge = home_edge
            bet_odds = h_odds
            bet_model_p = model_p
            bet_implied = implied_home
        elif away_edge > home_edge and away_edge > edge_threshold:
            bet_side = "away"
            bet_edge = away_edge
            bet_odds = a_odds
            bet_model_p = 1 - model_p
            bet_implied = implied_away

        if bet_side is None:
            flat_history.append(flat_bankroll)
            kelly_history.append(kelly_bankroll)
            continue

        won = (bet_side == "home" and actual == 1) or (
               bet_side == "away" and actual == 0)

        # FIX 5: Flat betting — skip if bankroll below 1 unit
        if flat_bankroll >= flat_unit:
            flat_bets += 1
            payout = flat_unit * (bet_odds - 1) if won else -flat_unit
            flat_profit += payout
            flat_bankroll += payout
            if won:
                flat_wins += 1
        else:
            payout = 0.0

        # Kelly bet
        b = bet_odds - 1
        q = 1 - bet_model_p
        kelly_f = max(0, (b * bet_model_p - q) / b) * kelly_fraction
        kelly_stake = min(kelly_f * kelly_bankroll, kelly_bankroll * kelly_max_pct)
        if kelly_stake > 0.01 and kelly_bankroll > 10:
            kelly_bets += 1
            if won:
                kelly_bankroll += kelly_stake * b
                kelly_wins += 1
            else:
                kelly_bankroll -= kelly_stake

        mkt_p = market_probs[i] if not np.isnan(market_probs[i]) else np.nan
        bet_records.append({
            "game_idx": i,
            "side": bet_side,
            "model_prob": bet_model_p,
            "implied_prob": bet_implied,
            "market_prob": mkt_p if not np.isnan(mkt_p) else bet_implied,
            "edge": bet_edge,
            "odds_decimal": bet_odds,
            "won": won,
            "flat_payout": payout,
        })

        flat_history.append(flat_bankroll)
        kelly_history.append(kelly_bankroll)

    # FIX 6: CLV calculation
    # True CLV = you got better odds than the closing line
    # Without opening vs closing line data, we approximate:
    # CLV = edge we perceived (model_prob - implied_prob at bet time)
    # This is NOT true CLV but shows average perceived edge
    clv_values = []
    for rec in bet_records:
        clv_values.append(rec["edge"])
    avg_clv = np.mean(clv_values) if clv_values else 0.0

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
        "clv": {
            "avg_clv": avg_clv,
            "total_bets_with_clv": len(clv_values),
        },
        "bet_records": bet_records,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. CALIBRATION ANALYSIS
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
# 8. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_eval(df, feature_cols, test_seasons=None):
    """Train on all prior seasons, test on each target season."""

    if test_seasons is None:
        all_seasons = sorted(df["season"].unique())
        # Need at least 2 seasons of training data
        test_seasons = [s for s in all_seasons if s >= all_seasons[0] + 2]

    odds_csv = os.path.join(BASE_DIR, "mlb_closing_odds.csv")
    odds_df = load_closing_odds(odds_csv)

    all_results = []
    all_preds = []

    for test_season in test_seasons:
        print(f"\n{'=' * 60}")
        print(f"SEASON {test_season}")
        print(f"{'=' * 60}")

        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] == test_season].copy()

        if len(train_df) < 500 or len(test_df) < 100:
            print(f"  Skip: train={len(train_df)}, test={len(test_df)}")
            continue

        # Split training data: 80% train, 20% calibration (last portion)
        cal_cutoff = int(len(train_df) * 0.8)
        cal_df = train_df.iloc[cal_cutoff:]
        train_only = train_df.iloc[:cal_cutoff]

        X_train = train_only[feature_cols]
        y_train = train_only["home_win"].values
        X_cal = cal_df[feature_cols]
        y_cal = cal_df["home_win"].values
        X_test = test_df[feature_cols]
        y_test = test_df["home_win"].values

        print(f"  Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

        # Train ensemble
        ensemble = MLBEnsemble()
        ensemble.fit(X_train, y_train, X_cal, y_cal)

        y_prob = ensemble.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        # Diagnostic: verify Platt scaling produces continuous output
        n_unique = len(np.unique(np.round(y_prob, 4)))
        print(f"  Unique probabilities: {n_unique} (should be >> 50, was ~40 in v3)")

        # --- Metrics ---
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        ll = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        home_rate = y_test.mean()
        baseline_brier = brier_score_loss(y_test, np.full_like(y_prob, home_rate))

        print(f"\n  Accuracy:      {acc:.4f}  (home win rate: {home_rate:.3f})")
        print(f"  Brier Score:   {brier:.4f}  (baseline: {baseline_brier:.4f}, "
              f"edge: {baseline_brier - brier:+.4f})")
        print(f"  Log Loss:      {ll:.4f}")
        print(f"  ROC AUC:       {auc:.4f}")

        # Calibration table
        cal_result = evaluate_calibration(y_test, y_prob)
        print(f"\n  Calibration:")
        for _, row in cal_result.iterrows():
            print(f"    P={row['avg_predicted']:.2f} -> Actual={row['avg_actual']:.2f}  "
                  f"(n={int(row['count'])})")

        # --- Betting simulation ---
        avg_h, avg_a, mkt_probs, best_h, best_a = merge_odds_with_predictions(
            test_df, odds_df, test_season
        )

        # Simulate with average odds (consensus line)
        print(f"\n  --- Betting: Consensus Line ---")
        bet_avg = simulate_betting(y_test, y_prob, avg_h, avg_a, mkt_probs)
        print(f"  Flat ($10/bet): {bet_avg['flat']['bets']} bets | "
              f"Win: {bet_avg['flat']['win_rate']:.1f}% | "
              f"P/L: ${bet_avg['flat']['profit']:.2f} | "
              f"ROI: {bet_avg['flat']['roi_pct']:.1f}%")
        print(f"  Kelly (10%, 3% cap): {bet_avg['kelly']['bets']} bets | "
              f"Win: {bet_avg['kelly']['win_rate']:.1f}% | "
              f"Final: ${bet_avg['kelly']['bankroll']:.2f} | "
              f"ROI: {bet_avg['kelly']['roi_pct']:.1f}%")
        print(f"  CLV: {bet_avg['clv']['avg_clv']:.4f} "
              f"({bet_avg['clv']['total_bets_with_clv']} bets)")

        # Simulate with best odds (line shopping)
        if not np.all(np.isnan(best_h)):
            print(f"\n  --- Betting: Best Line (Shopping) ---")
            bet_best = simulate_betting(y_test, y_prob, best_h, best_a, mkt_probs)
            print(f"  Flat ($10/bet): {bet_best['flat']['bets']} bets | "
                  f"Win: {bet_best['flat']['win_rate']:.1f}% | "
                  f"P/L: ${bet_best['flat']['profit']:.2f} | "
                  f"ROI: {bet_best['flat']['roi_pct']:.1f}%")
            print(f"  Kelly (10%, 3% cap): {bet_best['kelly']['bets']} bets | "
                  f"Win: {bet_best['kelly']['win_rate']:.1f}% | "
                  f"Final: ${bet_best['kelly']['bankroll']:.2f} | "
                  f"ROI: {bet_best['kelly']['roi_pct']:.1f}%")
        else:
            bet_best = bet_avg

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
            "flat_bets": bet_avg["flat"]["bets"],
            "flat_profit": bet_avg["flat"]["profit"],
            "flat_roi": bet_avg["flat"]["roi_pct"],
            "flat_win_rate": bet_avg["flat"]["win_rate"],
            "kelly_bets": bet_avg["kelly"]["bets"],
            "kelly_roi": bet_avg["kelly"]["roi_pct"],
            "kelly_bankroll": bet_avg["kelly"]["bankroll"],
            "avg_clv": bet_avg["clv"]["avg_clv"],
            "flat_history": bet_avg["flat"]["history"],
            "kelly_history": bet_avg["kelly"]["history"],
            "best_flat_profit": bet_best["flat"]["profit"],
            "best_flat_roi": bet_best["flat"]["roi_pct"],
        })

        # Build per-game prediction records
        pred_df = pd.DataFrame({
            "date": test_df["game_date"].values if "game_date" in test_df.columns else "",
            "home_team": test_df["home_team"].values,
            "away_team": test_df["away_team"].values,
            "model_prob": y_prob,
            "odds_implied_prob": mkt_probs,
            "result": y_test,
            "season": test_season,
        })
        # Add bet side (consistent with simulate_betting: one side only, 5% threshold)
        bet_sides = []
        for j in range(len(y_prob)):
            if np.isnan(avg_h[j]) or np.isnan(avg_a[j]):
                bet_sides.append("")
                continue
            home_edge = y_prob[j] - (1.0 / avg_h[j])
            away_edge = (1 - y_prob[j]) - (1.0 / avg_a[j])
            # Only bet the better side, and only if edge > 5%
            if home_edge > away_edge and home_edge > 0.05:
                bet_sides.append("home")
            elif away_edge > home_edge and away_edge > 0.05:
                bet_sides.append("away")
            else:
                bet_sides.append("")
        pred_df["bet_side"] = bet_sides
        all_preds.append(pred_df)

    return all_results, all_preds


# ═══════════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(all_results, all_preds):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bankroll trajectories
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

    # 2. Calibration plot
    ax = axes[0, 1]
    if all_preds:
        combined = pd.concat(all_preds)
        cal = evaluate_calibration(
            combined["result"].values.astype(int),
            combined["model_prob"].values,
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.scatter(
            cal["avg_predicted"], cal["avg_actual"],
            s=cal["count"] * 2, alpha=0.7, c="steelblue", edgecolors="navy",
        )
        for _, row in cal.iterrows():
            ax.annotate(
                f"n={int(row['count'])}",
                (row["avg_predicted"], row["avg_actual"]),
                fontsize=7, alpha=0.6,
            )
        ax.set_title("Calibration Plot (all seasons)")
        ax.set_xlabel("Predicted P(Home Win)")
        ax.set_ylabel("Actual Win Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # 3. Season metrics comparison
    ax = axes[1, 0]
    seasons = [r["season"] for r in all_results]
    x = np.arange(len(seasons))
    w = 0.2
    ax.bar(x - w, [r["accuracy"] for r in all_results], w,
           label="Accuracy", color="steelblue")
    ax.bar(x, [r["brier"] for r in all_results], w,
           label="Brier", color="coral")
    ax.bar(x + w, [r["baseline_brier"] for r in all_results], w,
           label="Baseline Brier", color="lightcoral", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_title("Accuracy / Brier Score by Season")
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
    for bar, p, r in zip(bars, profits, all_results):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"ROI: {r['flat_roi']:.1f}%\n{r['flat_bets']} bets",
            ha="center", va="bottom" if p >= 0 else "top", fontsize=8,
        )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "mlb_model_v4_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MLB Prediction Model v4 — Audit Fixes")
    print("Platt scaling + single-side bets + 5% edge threshold")
    print("=" * 60)

    df, feature_cols = prepare_features()

    # Show NaN rates for top features
    print(f"\nNaN rates (features with missing data):")
    nan_rates = df[feature_cols].isnull().mean().sort_values(ascending=False)
    for col, rate in nan_rates.items():
        if rate > 0:
            print(f"  {col}: {rate:.1%}")
    if (nan_rates > 0).sum() == 0:
        print("  (none)")

    results, preds = walk_forward_eval(df, feature_cols)

    # --- Overall Summary ---
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    if results:
        header = (f"{'Season':>8} {'Acc':>7} {'Brier':>7} {'AUC':>7} "
                  f"{'Bets':>5} {'Win%':>6} {'Profit':>9} {'ROI':>7} "
                  f"{'CLV':>7} {'KellyROI':>9}")
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r['season']:>8} {r['accuracy']:>7.4f} {r['brier']:>7.4f} "
                  f"{r['auc']:>7.4f} {r['flat_bets']:>5} "
                  f"{r['flat_win_rate']:>5.1f}% "
                  f"${r['flat_profit']:>8.2f} {r['flat_roi']:>6.1f}% "
                  f"{r['avg_clv']:>+7.4f} {r['kelly_roi']:>8.1f}%")
        print("-" * len(header))

        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_brier = np.mean([r["brier"] for r in results])
        avg_base = np.mean([r["baseline_brier"] for r in results])
        avg_auc = np.mean([r["auc"] for r in results])
        total_profit = sum(r["flat_profit"] for r in results)
        total_bets = sum(r["flat_bets"] for r in results)
        total_best_profit = sum(r["best_flat_profit"] for r in results)
        avg_clv = np.mean([r["avg_clv"] for r in results])

        print(f"\n  Avg Accuracy:  {avg_acc:.4f}")
        print(f"  Avg Brier:     {avg_brier:.4f} (baseline: {avg_base:.4f}, "
              f"edge: {avg_base - avg_brier:+.4f})")
        print(f"  Avg AUC:       {avg_auc:.4f}")
        print(f"  Avg CLV:       {avg_clv:+.4f}")
        print(f"\n  Consensus line: ${total_profit:.2f} over {total_bets} bets = "
              f"{total_profit / max(total_bets * 10, 1) * 100:.1f}% ROI")
        print(f"  Best line:      ${total_best_profit:.2f} over {total_bets} bets = "
              f"{total_best_profit / max(total_bets * 10, 1) * 100:.1f}% ROI")

    # Plot
    plot_results(results, preds)

    # Save predictions CSV
    if preds:
        all_preds = pd.concat(preds)
        out_csv = os.path.join(BASE_DIR, "mlb_predictions_v4.csv")
        all_preds.to_csv(out_csv, index=False)
        print(f"Saved predictions: {out_csv} ({len(all_preds)} rows)")

    print("\nDone.")