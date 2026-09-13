"""
NFL Prediction System Backtest & Calibration Pipeline

Replays historical games in strict chronological (walk-forward) order,
computing each game's probability using only data available before that
game, then checks predictions against what actually happened.

Reuses functions from predict_nfl.py (bayesian_shrinkage,
prop_floor_probs_yards, compute_league_avg_hit_rate) rather than
reimplementing that math a second time.

Runs independently of predict_nfl.py - separate entry point, own output
(docs/calibration.html), does not touch docs/index.html or docs/report.json.

Loading all 5 seasons of pbp takes real time and bandwidth - watch the
terminal log for the season-by-season game counts to confirm nothing
silently failed partway through.

The opponent-adjustment comparison below tests none / yards-allowed /
EPA at three scales, and the live system's default follows whichever
wins - see EPA_ENABLED and EPA_Z_SCALE in predict_nfl.py.
"""

import json
import math
import os
from datetime import datetime

import pandas as pd
import numpy as np
import nfl_data_py as nfl

import matplotlib
matplotlib.use("Agg")  # headless - no display available in CI
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss, log_loss

# Reuse the SAME tested math from the main script rather than duplicate it.
from predict_nfl import (
    bayesian_shrinkage,
    compute_league_avg_hit_rate,
    prop_floor_probs_yards,
    SHRINKAGE_K,
    EPA_Z_SCALE,
    EPA_ENABLED,
)

# LIVE_EPA_SCALE is DERIVED from predict_nfl.py's own constants, not a
# separately-maintained number. Previously this file defined its own
# LIVE_EPA_SCALE = 0.0 literal, which only matched predict_nfl.py's
# configuration by coincidence (both happened to mean "EPA off") - if
# EPA_ENABLED or EPA_Z_SCALE changed in predict_nfl.py without someone
# remembering to also edit this file, the calibration report would
# silently go back to describing a model that isn't actually running,
# exactly the bug this whole fix chain was catching. Importing directly
# makes that drift structurally impossible instead of just unlikely.
LIVE_EPA_SCALE = EPA_Z_SCALE if EPA_ENABLED else 0.0

BACKTEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_GAMES_BEFORE_PREDICTING = 3  # need at least this many prior games to form a prediction
QB_GAMES_SAMPLE = 10  # match the main script's sample window


def load_all_seasons_pbp(seasons):
    """
    Loads and concatenates pbp for every season in `seasons`. Prints a
    per-season row count as it goes, so a partial/failed season is
    visible in the log immediately rather than silently shrinking the
    final dataset.
    """
    frames = []
    for season in seasons:
        try:
            df = nfl.import_pbp_data([season], downcast=True, cache=False)
        except Exception as e:
            print(f"WARNING: failed to load pbp for season={season}: {e} - skipping this season.")
            continue
        if df is None or df.empty:
            print(f"WARNING: season={season} pbp came back empty - skipping.")
            continue
        print(f"Loaded season={season}: {len(df)} rows.")
        frames.append(df)
    if not frames:
        raise RuntimeError("No historical pbp data could be loaded for any season - cannot backtest.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined pbp across {len(frames)} seasons: {len(combined)} total rows.")
    return combined


def build_walkforward_qb_gamelog(pbp_df):
    """
    Builds a single long table of (passer_id, season, week, game_id,
    posteam, passing_yards, pass_attempts, defteam-of-opponent), sorted
    in strict chronological order (season then week). This is the base
    table the walk-forward loop steps through - at each row, "history"
    is every row before it in this sort order for the same passer_id.
    """
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["passer_id"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0.0)

    grouped = pass_plays.groupby(["passer_id", "season", "week", "game_id", "posteam", "defteam"]).agg(
        passing_yards=("passing_yards", "sum"),
        pass_attempts=("pass_attempt", "sum"),
    ).reset_index()

    grouped = grouped.sort_values(["season", "week"]).reset_index(drop=True)
    return grouped


def build_walkforward_defense_epa(pbp_df):
    """
    Same idea as build_defense_pass_epa in the main script, but keyed by
    (season, week) cumulative-so-far rather than a single full-season
    snapshot - needed so the opponent-adjustment step in the replay loop
    can look up "defense strength as of this point in the season" rather
    than leaking the whole season's final defensive numbers into a
    prediction made mid-season (that would be lookahead bias).

    Returns a DataFrame with one row per (defteam, season, week,
    cumulative pass_epa_allowed_pg through that week).
    """
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["defteam", "epa"]).copy()

    per_game = pass_plays.groupby(["defteam", "season", "week", "game_id"]).agg(
        epa_allowed=("epa", "sum"),
    ).reset_index()
    per_game = per_game.sort_values(["defteam", "season", "week"])

    # cumulative average per team per season, computed BEFORE each week
    # (shift(1) so week N's value only reflects weeks < N, avoiding
    # lookahead - this is the same discipline as the QB gamelog filter).
    per_game["cum_avg_epa_allowed"] = (
        per_game.groupby(["defteam", "season"])["epa_allowed"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    return per_game


def build_walkforward_defense_yards_allowed(pbp_df):
    """
    Mirrors build_walkforward_defense_epa exactly, but over passing_yards
    instead of epa - this is what lets the backtest actually test the
    LIVE system's real opponent-adjustment path (yards-allowed, since
    EPA is disabled - see EPA_ENABLED in predict_nfl.py), which the
    backtest previously never modeled at all (it only ever tested "no
    adjustment" vs. "EPA at various scales" - the yards-allowed path the
    live system actually claims to use was never in the comparison).

    Returns a DataFrame with one row per (defteam, season, week,
    cumulative pass_yds_allowed_pg through that week - computed with the
    same shift(1).expanding().mean() lookahead-safe pattern).
    """
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["defteam"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0.0)

    per_game = pass_plays.groupby(["defteam", "season", "week", "game_id"]).agg(
        yds_allowed=("passing_yards", "sum"),
    ).reset_index()
    per_game = per_game.sort_values(["defteam", "season", "week"])

    per_game["cum_avg_yds_allowed"] = (
        per_game.groupby(["defteam", "season"])["yds_allowed"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    return per_game


def build_walkforward_team_scoring(pbp_df):
    """Builds walk-forward-safe cumulative points-for/against per (team, season, week)."""
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    scores = df.drop_duplicates(subset=["game_id"])[
        ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]
    ].dropna()

    rows = []
    for _, s in scores.iterrows():
        rows.append({"team": s["home_team"], "opponent": s["away_team"], "season": s["season"],
                      "week": s["week"], "game_id": s["game_id"], "is_home": True,
                      "pts_for": s["home_score"], "pts_against": s["away_score"]})
        rows.append({"team": s["away_team"], "opponent": s["home_team"], "season": s["season"],
                      "week": s["week"], "game_id": s["game_id"], "is_home": False,
                      "pts_for": s["away_score"], "pts_against": s["home_score"]})
    team_games = pd.DataFrame(rows).sort_values(["team", "season", "week"])

    team_games["cum_pts_for"] = (
        team_games.groupby(["team", "season"])["pts_for"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    team_games["cum_pts_against"] = (
        team_games.groupby(["team", "season"])["pts_against"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    team_games["games_played_before"] = team_games.groupby(["team", "season"]).cumcount()
    print(f"Built walk-forward team scoring history: {len(team_games)} team-game rows.")
    return team_games


def build_walkforward_qb_out_signal(pbp_df, injuries_df):
    """
    Builds a (team, season, week) -> bool "confirmed starter QB was out"
    signal, using the same walk-forward-safe starter identity (prior-
    weeks attempts leader) as fit_injury_multipliers in predict_nfl.py.
    """
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["passer_id"]).copy()
    weekly_attempts = pass_plays.groupby(["posteam", "season", "week", "passer"])["pass_attempt"].sum().reset_index()

    def _prior_starter(team, season, week):
        prior = weekly_attempts[(weekly_attempts["posteam"] == team) & (weekly_attempts["season"] == season)
                                 & (weekly_attempts["week"] < week)]
        if prior.empty:
            return None
        totals = prior.groupby("passer")["pass_attempt"].sum()
        return totals.idxmax() if not totals.empty else None

    out_rows = injuries_df[(injuries_df["position"] == "QB") & (injuries_df["report_status"] == "Out")]
    qb_out_signal = {}
    for _, row in out_rows.iterrows():
        team, season, week = row["team"], row["season"], row["week"]
        starter_name = _prior_starter(team, season, week)
        if starter_name is None:
            continue
        last_name = row["full_name"].split()[-1].lower()
        if last_name in starter_name.lower():
            qb_out_signal[(team, season, week)] = True
    return qb_out_signal


def replay_team_totals(team_games, min_games_before_predicting=3, use_injury_adjustment=True,
                        injury_mults=None, qb_out_signal=None):
    """
    Walk-forward replay of team-total over/under predictions, using only
    cumulative scoring data available before each game.

    The threshold for each prediction is that team's OWN cumulative
    scoring rate going into the game (cum_pts_for), not a global league-
    average constant - the live system compares a projected score
    against a game-specific line, and a constant threshold would test a
    different, easier question ("is this team above league average")
    instead of the one that matters.
    """
    predictions = []
    league_avg_by_week = (
        team_games.dropna(subset=["cum_pts_for"])
        .groupby(["season", "week"])["cum_pts_for"]
        .mean()
        .to_dict()
    )
    rows = team_games.to_dict("records")

    for row in rows:
        if row["games_played_before"] < min_games_before_predicting:
            continue
        if pd.isna(row["cum_pts_for"]) or pd.isna(row["cum_pts_against"]):
            continue

        league_avg = league_avg_by_week.get((row["season"], row["week"]))
        if not league_avg:
            continue

        own_rate = bayesian_shrinkage(row["cum_pts_for"] * row["games_played_before"], row["games_played_before"], league_avg, k=SHRINKAGE_K)
        pct_diff = (row["cum_pts_against"] - league_avg) / league_avg if league_avg else 0
        adj_factor = max(-0.15, min(0.15, 0.4 * pct_diff))
        projected = own_rate * (1 + adj_factor)

        if use_injury_adjustment and injury_mults and qb_out_signal:
            if qb_out_signal.get((row["team"], row["season"], row["week"])):
                projected *= injury_mults.get("qb", 0.90)

        actual = row["pts_for"]
        # Team's own historical baseline, not the league average - a
        # team that's scored 28/game shouldn't be scored against a 22.5
        # league-wide line; their own baseline is the meaningful
        # comparison point, matching how a real posted team-total line
        # tracks the team's actual scoring level, not the league's.
        threshold = round(row["cum_pts_for"], 1)
        predictions.append({
            "game_id": row["game_id"],
            "team": row["team"],
            "season": row["season"],
            "week": row["week"],
            "threshold": threshold,
            "predicted_points": round(projected, 1),
            "actual_points": actual,
            "predicted_prob": None,  # filled in below once std_dev is known
        })

    return predictions


def brier_with_se(y_true, y_prob):
    """Brier score with its standard error: SE = std((p-y)^2) / sqrt(n)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    terms = (y_prob - y_true) ** 2
    brier = terms.mean()
    se = terms.std(ddof=1) / np.sqrt(len(terms)) if len(terms) > 1 else float("nan")
    return brier, se, len(terms)


def score_against_threshold(predicted_points, actual_points, threshold, std_dev):
    """Converts a points projection into an over/under probability against a given threshold."""
    z = (predicted_points - threshold) / std_dev
    prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    hit = 1 if actual_points > threshold else 0
    return prob, hit


def run_topdown_vs_bottomup_matrix(team_games, bu_predictions, td_predictions, team_total_std):
    """
    Scores top-down and bottom-up predictions against BOTH a league-
    average threshold and a per-team threshold, on the same sample, so
    the Brier comparison reflects the models - not which test each one
    happened to be scored against. Also scores a simple-average ensemble
    of the two. Prints Brier + SE for all combinations; does not declare
    a winner - if the gap is smaller than 1 SE, says so explicitly.

    td_predictions must be the output of replay_team_totals() - the real
    top-down projection (shrinkage + opponent adjustment applied), not a
    raw cumulative average. Using the raw average here would make the
    projection nearly identical to the per-team threshold by
    construction, artificially collapsing the variance.
    """
    league_avg_score = team_games["pts_for"].mean()

    td_lookup = {(p["game_id"], p["team"]): p for p in td_predictions}
    bu_lookup = {(p["game_id"], p["team"]): p for p in bu_predictions if p.get("actual_points") is not None}

    common_keys = set(td_lookup.keys()) & set(bu_lookup.keys())
    print(f"Common sample for top-down/bottom-up comparison: {len(common_keys)} team-games.")

    rows = {"topdown": {"league": {"p": [], "y": []}, "team": {"p": [], "y": []}},
            "bottomup": {"league": {"p": [], "y": []}, "team": {"p": [], "y": []}},
            "ensemble": {"league": {"p": [], "y": []}, "team": {"p": [], "y": []}}}

    for key in common_keys:
        td = td_lookup[key]
        bu = bu_lookup[key]
        actual = td["actual_points"]

        td_proj = td["predicted_points"]
        team_threshold = td["threshold"]  # the per-team baseline replay_team_totals already computed
        bu_proj = bu["predicted_points"]
        ensemble_proj = round((td_proj + bu_proj) / 2, 1)

        for model_name, proj in (("topdown", td_proj), ("bottomup", bu_proj), ("ensemble", ensemble_proj)):
            p_league, y_league = score_against_threshold(proj, actual, round(league_avg_score, 1), team_total_std)
            p_team, y_team = score_against_threshold(proj, actual, team_threshold, team_total_std)
            rows[model_name]["league"]["p"].append(p_league)
            rows[model_name]["league"]["y"].append(y_league)
            rows[model_name]["team"]["p"].append(p_team)
            rows[model_name]["team"]["y"].append(y_team)

    print()
    print("=" * 78)
    print("TOP-DOWN vs BOTTOM-UP vs ENSEMBLE: 2x2 THRESHOLD MATRIX")
    print("=" * 78)
    print(f"{'Model':<12}{'Threshold':<18}{'Brier':>10}{'SE':>10}{'n':>8}{'NaiveBase':>12}")
    print("-" * 78)
    results = {}
    naive_results = {}
    for model_name in ("topdown", "bottomup", "ensemble"):
        for threshold_name in ("league", "team"):
            p = rows[model_name][threshold_name]["p"]
            y = rows[model_name][threshold_name]["y"]
            brier, se, n = brier_with_se(y, p)
            base_rate = float(np.mean(y)) if len(y) > 0 else float("nan")
            naive_brier = base_rate * (1 - base_rate)
            results[(model_name, threshold_name)] = (brier, se, n)
            naive_results[(model_name, threshold_name)] = naive_brier
            label = "league-average" if threshold_name == "league" else "per-team"
            print(f"{model_name:<12}{label:<18}{round(brier,4):>10}{round(se,4):>10}{n:>8}{round(naive_brier,4):>12}")
    print("=" * 78)

    td_team = results[("topdown", "team")]
    bu_team = results[("bottomup", "team")]
    td_team_naive = naive_results[("topdown", "team")]
    print(f"Top-down (per-team) vs its own naive baseline: model={round(td_team[0],4)}, "
          f"naive={round(td_team_naive,4)}, skill={round(td_team_naive - td_team[0],4)}")
    gap = abs(td_team[0] - bu_team[0])
    combined_se = math.sqrt(td_team[1]**2 + bu_team[1]**2)
    print(f"Top-down vs bottom-up gap (per-team threshold): {round(gap,4)}, combined SE: {round(combined_se,4)}")
    t_stat = gap / combined_se if combined_se > 0 else 0.0
    print(f"t-stat (gap / combined SE): {round(t_stat, 2)}")
    if t_stat < 2:
        print(f"Gap {round(gap,4)} vs combined SE {round(combined_se,4)} - does not clear "
              f"2 SE (t={round(t_stat,2)}, ~95% confidence). Suggestive, not statistically significant.")
    else:
        print(f"Gap {round(gap,4)} vs combined SE {round(combined_se,4)} - clears 2 SE "
              f"(t={round(t_stat,2)}). Statistically significant on this sample.")
    print("=" * 78)
    print()
    return results


def score_team_total_predictions(predictions, std_dev):
    """Converts projected points into over/under probabilities against each prediction's own threshold."""
    for p in predictions:
        z = (p["predicted_points"] - p["threshold"]) / std_dev
        p["predicted_prob"] = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 4)
        p["hit"] = 1 if p["actual_points"] > p["threshold"] else 0
    return predictions


def fit_yards_to_points_rate(pbp_df):
    """Fits a real points-per-total-offensive-yard rate from history, for the bottom-up comparison."""
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()

    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["posteam"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0)
    pass_yards = pass_plays.groupby(["game_id", "posteam"])["passing_yards"].sum().reset_index()

    rush_plays = df[df["rush_attempt"] == 1].dropna(subset=["posteam"]).copy()
    rush_plays["rushing_yards"] = rush_plays["rushing_yards"].fillna(0)
    rush_yards = rush_plays.groupby(["game_id", "posteam"])["rushing_yards"].sum().reset_index()

    total_yards = pass_yards.merge(rush_yards, on=["game_id", "posteam"], how="outer").fillna(0)
    total_yards["total_yards"] = total_yards["passing_yards"] + total_yards["rushing_yards"]

    scores = df.drop_duplicates(subset=["game_id"])[["game_id", "home_team", "away_team", "home_score", "away_score"]].dropna()
    rows = []
    for _, s in scores.iterrows():
        rows.append({"game_id": s["game_id"], "team": s["home_team"], "actual_pts": s["home_score"]})
        rows.append({"game_id": s["game_id"], "team": s["away_team"], "actual_pts": s["away_score"]})
    actuals = pd.DataFrame(rows)

    merged = total_yards.merge(actuals, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="inner")
    if merged.empty or merged["total_yards"].sum() == 0:
        print("WARNING: could not fit yards-to-points rate - using fallback 0.05 (1 point per 20 yards).")
        return 0.05, merged

    rate = merged["actual_pts"].sum() / merged["total_yards"].sum()
    corr = merged["total_yards"].corr(merged["actual_pts"])
    print(f"Fitted yards-to-points rate from {len(merged)} team-games: {round(rate,4)} "
          f"(~1 point per {round(1/rate,1)} total offensive yards), correlation={round(corr,3)}.")
    return rate, merged


def build_walkforward_bottom_up_yards(pbp_df):
    """
    Walk-forward-safe cumulative total offensive yards (passing +
    rushing) per team per game, same shift(1).expanding().mean() pattern
    as build_walkforward_team_scoring - needed so the bottom-up
    comparison predicts each game from PRIOR games' yardage, not that
    game's own final yardage (which would just be a correlation check,
    not a prediction, and would make bottom-up look artificially strong).
    """
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()

    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["posteam"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0)
    pass_yards = pass_plays.groupby(["game_id", "season", "week", "posteam"])["passing_yards"].sum().reset_index()

    rush_plays = df[df["rush_attempt"] == 1].dropna(subset=["posteam"]).copy()
    rush_plays["rushing_yards"] = rush_plays["rushing_yards"].fillna(0)
    rush_yards = rush_plays.groupby(["game_id", "season", "week", "posteam"])["rushing_yards"].sum().reset_index()

    total_yards = pass_yards.merge(rush_yards, on=["game_id", "season", "week", "posteam"], how="outer").fillna(0)
    total_yards["total_yards"] = total_yards["passing_yards"] + total_yards["rushing_yards"]
    total_yards = total_yards.sort_values(["posteam", "season", "week"])

    total_yards["cum_yards"] = (
        total_yards.groupby(["posteam", "season"])["total_yards"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    total_yards["games_played_before"] = total_yards.groupby(["posteam", "season"]).cumcount()
    return total_yards


def replay_bottom_up_team_totals(walkforward_yards, points_per_yard, min_games_before_predicting=3):
    """
    Walk-forward bottom-up prediction: projects each game's points from
    that team's PRIOR games' average total yards times the fitted
    points_per_yard rate. Mirrors replay_team_totals' discipline exactly,
    so the two are actually comparable.
    """
    predictions = []
    for row in walkforward_yards.to_dict("records"):
        if row["games_played_before"] < min_games_before_predicting or pd.isna(row["cum_yards"]):
            continue
        predictions.append({
            "game_id": row["game_id"],
            "team": row["posteam"],
            "season": row["season"],
            "week": row["week"],
            "predicted_points": round(row["cum_yards"] * points_per_yard, 1),
        })
    return predictions


def build_bottom_up_team_totals(pbp_df, points_per_yard=None):
    """
    Full-sample (non-walk-forward) bottom-up total, using each game's
    own final yardage. This is a correlation/fit check, not a genuine
    prediction - use replay_bottom_up_team_totals for a fair, walk-
    forward comparison against the top-down model.
    """
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()

    if points_per_yard is None:
        points_per_yard, merged = fit_yards_to_points_rate(pbp_df)
    else:
        pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["posteam"]).copy()
        pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0)
        pass_yards = pass_plays.groupby(["game_id", "posteam"])["passing_yards"].sum().reset_index()
        rush_plays = df[df["rush_attempt"] == 1].dropna(subset=["posteam"]).copy()
        rush_plays["rushing_yards"] = rush_plays["rushing_yards"].fillna(0)
        rush_yards = rush_plays.groupby(["game_id", "posteam"])["rushing_yards"].sum().reset_index()
        total_yards = pass_yards.merge(rush_yards, on=["game_id", "posteam"], how="outer").fillna(0)
        total_yards["total_yards"] = total_yards["passing_yards"] + total_yards["rushing_yards"]
        merged = total_yards

    results = []
    for row in merged.to_dict("records"):
        results.append({
            "game_id": row["game_id"],
            "team": row["posteam"],
            "bottom_up_points": round(row["total_yards"] * points_per_yard, 1),
        })
    return pd.DataFrame(results)


def replay_all_games(qb_gamelog, defense_epa_history, defense_yards_history=None,
                      season_filter=None, adjustment_mode="none", epa_scale=0.10):
    """
    The core walk-forward loop. For every QB-game in qb_gamelog (in
    chronological order), if that QB already has >= MIN_GAMES_BEFORE_PREDICTING
    prior games, computes the SAME probability the live system would have
    computed at that point in time - bayesian_shrinkage, _build_qb_thresholds,
    prop_floor_probs_yards, plus an opponent adjustment - then records the
    predicted probability at each threshold alongside what actually happened.

    adjustment_mode controls which opponent adjustment is applied:
      "none" - no opponent adjustment at all (pure shrinkage baseline)
      "yards" - the LIVE SYSTEM'S ACTUAL DEFAULT PATH (see predict_nfl.py's
                opponent_adjustment(), yards-allowed branch). Uses the SAME
                formula: pct_diff = (opp_value - league_avg) / league_avg,
                adj_factor = clamp(0.4 * pct_diff, -0.15, 0.15). Requires
                defense_yards_history (build_walkforward_defense_yards_allowed).
      "epa"  - the EPA z-score path (disabled in the live system by
                default - see EPA_ENABLED in predict_nfl.py), scaled by
                epa_scale.

    CRITICAL: previously this backtest only ever tested "none" vs. "epa at
    various scales" - the "yards" mode (the live system's ACTUAL default
    adjustment) was never in the comparison at all, because a separate,
    now-fixed bug meant the live yards-allowed adjustment was silently a
    no-op (see finalize_qb_floors' docstring in predict_nfl.py). Now that
    the live adjustment genuinely does something, the backtest has to be
    able to model it too, or the comparison is testing configurations
    that don't match what's actually deployed.

    Returns a list of prediction records: {game_id, passer_id, season,
    week, threshold, predicted_prob, actual_yards, hit, games_of_history}.
    """
    if adjustment_mode not in ("none", "yards", "epa"):
        raise ValueError(f"adjustment_mode must be 'none', 'yards', or 'epa', got {adjustment_mode!r}")
    if adjustment_mode == "yards" and defense_yards_history is None:
        raise ValueError("adjustment_mode='yards' requires defense_yards_history "
                          "(build_walkforward_defense_yards_allowed(pbp)).")

    predictions = []
    history_by_passer = {}  # passer_id -> list of prior game dicts, built up as we go

    defense_epa_indexed = defense_epa_history.set_index(["defteam", "season", "week"])
    league_epa_stats = (
        defense_epa_history.dropna(subset=["cum_avg_epa_allowed"])
        .groupby(["season", "week"])["cum_avg_epa_allowed"]
        .agg(["mean", "std"])
        .to_dict("index")
    )

    defense_yards_indexed = None
    league_yards_stats = {}
    if defense_yards_history is not None:
        defense_yards_indexed = defense_yards_history.set_index(["defteam", "season", "week"])
        # league_avg only (no std needed - the live "yards" formula uses
        # pct-of-mean, not a z-score, matching opponent_adjustment()'s
        # yards-allowed branch exactly).
        league_yards_stats = (
            defense_yards_history.dropna(subset=["cum_avg_yds_allowed"])
            .groupby(["season", "week"])["cum_avg_yds_allowed"]
            .mean()
            .to_dict()
        )

    rows = qb_gamelog.to_dict("records")
    total_rows = len(rows)
    league_avg_cache = {}  # (season, week) -> league_avg_hit_rate

    for i, row in enumerate(rows):
        pid = row["passer_id"]
        season, week = row["season"], row["week"]

        if season_filter and season not in season_filter:
            history_by_passer.setdefault(pid, []).append(row)
            continue

        prior_games = history_by_passer.get(pid, [])
        if len(prior_games) < MIN_GAMES_BEFORE_PREDICTING:
            history_by_passer.setdefault(pid, []).append(row)
            continue

        recent = prior_games[-QB_GAMES_SAMPLE:]
        raw_floors = prop_floor_probs_yards(recent)
        if not raw_floors:
            history_by_passer.setdefault(pid, []).append(row)
            continue

        cache_key = (season, week)
        if cache_key not in league_avg_cache:
            snapshot_candidates = []
            for other_pid, other_games in history_by_passer.items():
                if len(other_games) < MIN_GAMES_BEFORE_PREDICTING:
                    continue
                other_recent = other_games[-QB_GAMES_SAMPLE:]
                other_floors = prop_floor_probs_yards(other_recent)
                if other_floors:
                    snapshot_candidates.append({
                        "floors": {"passing_yards": other_floors},
                        "games_sampled": len(other_recent),
                    })
            league_avg_cache[cache_key] = compute_league_avg_hit_rate(snapshot_candidates, stat_key="passing_yards")
        league_avg = league_avg_cache[cache_key]

        defteam = row["defteam"]

        # EPA snapshot (used only if adjustment_mode == "epa")
        opp_cum_epa = None
        league_epa_snapshot = None
        if adjustment_mode == "epa":
            try:
                opp_cum_epa = defense_epa_indexed.loc[(defteam, season, week), "cum_avg_epa_allowed"]
                if pd.isna(opp_cum_epa):
                    opp_cum_epa = None
            except KeyError:
                opp_cum_epa = None
            league_epa_snapshot = league_epa_stats.get((season, week))

        # Yards-allowed snapshot (used only if adjustment_mode == "yards" -
        # this is the LIVE SYSTEM'S ACTUAL DEFAULT PATH)
        opp_cum_yards = None
        league_yards_avg = None
        if adjustment_mode == "yards":
            try:
                opp_cum_yards = defense_yards_indexed.loc[(defteam, season, week), "cum_avg_yds_allowed"]
                if pd.isna(opp_cum_yards):
                    opp_cum_yards = None
            except KeyError:
                opp_cum_yards = None
            league_yards_avg = league_yards_stats.get((season, week))

        for threshold, raw_hr in raw_floors.items():
            hits = round(raw_hr * len(recent))
            shrunk = bayesian_shrinkage(hits, len(recent), league_avg, k=SHRINKAGE_K)
            predicted_prob = shrunk

            if adjustment_mode == "epa" and opp_cum_epa is not None and league_epa_snapshot and league_epa_snapshot.get("std"):
                std = league_epa_snapshot["std"]
                if std and std > 0:
                    z = (opp_cum_epa - league_epa_snapshot["mean"]) / std
                    adj_factor = max(-0.15, min(0.15, epa_scale * z))
                    predicted_prob = max(0.0, min(1.0, shrunk * (1 + adj_factor)))

            elif adjustment_mode == "yards" and opp_cum_yards is not None and league_yards_avg:
                # EXACT same formula as opponent_adjustment()'s yards-allowed
                # branch in predict_nfl.py: pct_diff = (value - avg) / avg,
                # scaled by 0.4, capped at +/-15%.
                pct_diff = (opp_cum_yards - league_yards_avg) / league_yards_avg
                adj_factor = max(-0.15, min(0.15, 0.4 * pct_diff))
                predicted_prob = max(0.0, min(1.0, shrunk * (1 + adj_factor)))

            predictions.append({
                "game_id": row["game_id"],
                "passer_id": pid,
                "season": season,
                "week": week,
                "threshold": threshold,
                "predicted_prob": round(predicted_prob, 4),
                "actual_yards": row["passing_yards"],
                "hit": 1 if row["passing_yards"] >= threshold else 0,
                "games_of_history": len(prior_games),  # used for the early-season bucket
            })

        history_by_passer.setdefault(pid, []).append(row)

        if (i + 1) % 5000 == 0:
            print(f"  replayed {i+1}/{total_rows} QB-games...")

    print(f"Replay complete ({adjustment_mode}): {len(predictions)} threshold-level predictions generated "
          f"across {len(set(p['game_id'] for p in predictions))} games.")
    return predictions


EARLY_SEASON_MAX_PRIOR_GAMES = 5  # early-season-equivalent threshold


def compute_calibration_buckets(predictions, n_buckets=10):
    """
    Bucket table: predicted-probability bucket, N predictions, actual
    hit rate, delta (actual - predicted midpoint).

    Each bucket also computes the same stats restricted to predictions
    where games_of_history <= EARLY_SEASON_MAX_PRIOR_GAMES - closer to
    what the live system actually has early in a real season, since
    history_by_passer here accumulates across seasons while the live
    system only sees the current season.
    """
    if not predictions:
        return []
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    buckets = []
    for i in range(n_buckets):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        in_bucket = [p for p in predictions if lo <= p["predicted_prob"] < hi or (i == n_buckets - 1 and p["predicted_prob"] == hi)]
        n = len(in_bucket)
        if n == 0:
            continue
        actual_rate = sum(p["hit"] for p in in_bucket) / n
        midpoint = (lo + hi) / 2

        early_season = [p for p in in_bucket if p.get("games_of_history", 99) <= EARLY_SEASON_MAX_PRIOR_GAMES]
        early_n = len(early_season)
        early_actual_rate = (sum(p["hit"] for p in early_season) / early_n) if early_n > 0 else None
        early_delta = round(early_actual_rate - midpoint, 3) if early_actual_rate is not None else None

        buckets.append({
            "bucket_label": f"{round(lo*100)}-{round(hi*100)}%",
            "n": n,
            "actual_hit_rate": round(actual_rate, 3),
            "predicted_midpoint": round(midpoint, 3),
            "delta": round(actual_rate - midpoint, 3),
            "early_n": early_n,
            "early_actual_hit_rate": round(early_actual_rate, 3) if early_actual_rate is not None else None,
            "early_delta": early_delta,
        })
    return buckets


def build_joint_outcome_table(pbp_df):
    """
    Builds the historical joint outcome table used to answer conditional
    questions like "in games where the QB threw 275+, what fraction had
    the combined game total over 24?"

    Carries both team_total (this team's own score) and game_total
    (home_score + away_score) on every row - callers must condition on
    whichever one actually matches what they're comparing against, since
    they're on very different scales.

    Returns a DataFrame: game_id, team, opponent, qb_yards, team_total,
    game_total, opponent_qb_yards.
    """
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()

    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["passer_id"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0)
    qb_by_game = pass_plays.groupby(["game_id", "posteam"])["passing_yards"].sum().reset_index()
    qb_lookup = qb_by_game.set_index(["game_id", "posteam"])["passing_yards"].to_dict()

    scores = df.drop_duplicates(subset=["game_id"])[
        ["game_id", "home_team", "away_team", "home_score", "away_score"]
    ].dropna()

    rows = []
    for _, s in scores.iterrows():
        gid = s["game_id"]
        game_total = s["home_score"] + s["away_score"]  # combined, matches ESPN's total_line scale
        for team, opp_team, team_score in (
            (s["home_team"], s["away_team"], s["home_score"]),
            (s["away_team"], s["home_team"], s["away_score"]),
        ):
            qb_yards = qb_lookup.get((gid, team))
            if qb_yards is None:
                continue
            rows.append({
                "game_id": gid,
                "team": team,
                "opponent": opp_team,
                "qb_yards": qb_yards,
                "team_total": team_score,
                "game_total": game_total,
                "opponent_qb_yards": qb_lookup.get((gid, opp_team)),
            })

    joint_df = pd.DataFrame(rows)
    print(f"Built joint outcome table: {len(joint_df)} team-game rows.")
    return joint_df


def query_joint_probability(joint_df, qb_threshold, total_threshold, min_samples=20, condition_field="team_total"):
    """
    Answers: "in historical games where this team's QB threw >= qb_threshold
    yards, what fraction also had condition_field > total_threshold?"
    condition_field must be "team_total" (this team's own score) or
    "game_total" (combined score). Passing the wrong one silently
    answers a different, usually much rarer question.

    Returns {"joint_prob": float, "n": int, "unconditional_prob": float}
    or None if fewer than min_samples qualifying historical games exist.
    """
    if joint_df is None or joint_df.empty:
        return None
    if condition_field not in ("team_total", "game_total"):
        raise ValueError(f"condition_field must be 'team_total' or 'game_total', got {condition_field!r}")
    if condition_field not in joint_df.columns:
        return None  # older cached CSV without game_total - fail closed, not silently wrong

    subset = joint_df[joint_df["qb_yards"] >= qb_threshold]
    n = len(subset)
    if n < min_samples:
        return None
    joint_prob = (subset[condition_field] > total_threshold).mean()
    unconditional_prob = (joint_df[condition_field] > total_threshold).mean()
    return {
        "joint_prob": round(float(joint_prob), 3),
        "n": n,
        "unconditional_prob": round(float(unconditional_prob), 3),
    }


def render_calibration_html(predictions, buckets, brier, logloss, png_path):
    worst_bucket = max(buckets, key=lambda b: abs(b["delta"])) if buckets else None
    early_buckets = [b for b in buckets if b["early_n"] > 0]
    worst_early = max(early_buckets, key=lambda b: abs(b["early_delta"])) if early_buckets else None

    rows_html = "".join(
        f"<tr><td>{b['bucket_label']}</td><td>{b['n']}</td>"
        f"<td>{round(b['actual_hit_rate']*100,1)}%</td>"
        f"<td>{round(b['predicted_midpoint']*100,1)}%</td>"
        f"<td>{'+' if b['delta']>=0 else ''}{round(b['delta']*100,1)}%</td>"
        f"<td>{b['early_n'] if b['early_n'] > 0 else '\u2014'}</td>"
        f"<td>{(str(round(b['early_actual_hit_rate']*100,1))+'%') if b['early_actual_hit_rate'] is not None else '\u2014'}</td>"
        f"<td>{(('+' if b['early_delta']>=0 else '')+str(round(b['early_delta']*100,1))+'%') if b['early_delta'] is not None else '\u2014'}</td>"
        f"</tr>"
        for b in buckets
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFL Model Calibration</title>
<style>
body {{ background:#0b0f19; color:#e8ecf4; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; margin:0; padding:20px; }}
h1 {{ font-size:1.3em; }}
.summary {{ background:#131a2b; border-radius:12px; padding:16px; margin:16px 0; }}
.summary p {{ margin:4px 0; font-size:0.9em; }}
.caveat {{ background:#2b1d0e; border:1px solid #a56a1f; border-radius:12px; padding:16px; margin:16px 0; }}
.caveat p {{ margin:4px 0; font-size:0.85em; color:#f0c987; }}
.caveat strong {{ color:#ffdca0; }}
table {{ width:100%; border-collapse:collapse; margin-top:16px; font-size:0.82em; }}
th, td {{ padding:7px; text-align:left; border-bottom:1px solid rgba(255,255,255,0.1); }}
th {{ color:#8892a6; text-transform:uppercase; font-size:0.68em; letter-spacing:0.03em; }}
th.early, td.early {{ border-left:1px solid rgba(255,255,255,0.15); }}
img {{ max-width:100%; border-radius:12px; margin-top:16px; }}
</style>
</head>
<body>
<h1>NFL Model Calibration Report</h1>
<p style="color:#8892a6; font-size:0.8em;">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</p>

<div class="caveat">
  <p><strong>Sample-size note:</strong> Backtest predictions at week N use games from
  multiple seasons. The live system only uses the current season. Early-season
  calibration will be worse than the chart below shows. Treat weeks 1-6 with caution.</p>
  <p>The "Early-Season" columns restrict to predictions made with
  &le;{EARLY_SEASON_MAX_PRIOR_GAMES} prior games - closer to what the live
  system has access to early in a real season.</p>
</div>

<div class="summary">
  <p><strong>Brier score:</strong> {round(brier, 4)} (lower is better; 0 = perfect, 0.25 = no skill on a 50/50 task)</p>
  <p><strong>Log loss:</strong> {round(logloss, 4)} (lower is better)</p>
  <p><strong>Total predictions:</strong> {len(predictions)}</p>
  <p><strong>Worst-calibrated bucket (full sample):</strong> {worst_bucket['bucket_label'] if worst_bucket else 'N/A'}
     ({'+' if worst_bucket and worst_bucket['delta']>=0 else ''}{round(worst_bucket['delta']*100,1) if worst_bucket else 0}% off, n={worst_bucket['n'] if worst_bucket else 0})</p>
  <p><strong>Worst-calibrated bucket (early-season only):</strong> {worst_early['bucket_label'] if worst_early else 'N/A (insufficient early-season samples)'}
     {f"({'+' if worst_early['early_delta']>=0 else ''}{round(worst_early['early_delta']*100,1)}% off, n={worst_early['early_n']})" if worst_early else ''}</p>
</div>
<img src="calibration_curve.png" alt="Calibration reliability diagram">
<table>
  <thead><tr>
    <th>Predicted Bucket</th><th>N</th><th>Actual Hit Rate</th><th>Predicted Midpoint</th><th>Delta</th>
    <th class="early">Early-Season N</th><th class="early">Early-Season Actual</th><th class="early">Early-Season Delta</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""


if __name__ == "__main__":
    print(f"Starting backtest across seasons: {BACKTEST_SEASONS}")
    pbp = load_all_seasons_pbp(BACKTEST_SEASONS)

    qb_gamelog = build_walkforward_qb_gamelog(pbp)
    print(f"Built walk-forward QB gamelog: {len(qb_gamelog)} QB-game rows.")

    defense_epa_history = build_walkforward_defense_epa(pbp)
    print(f"Built walk-forward defense EPA history: {len(defense_epa_history)} team-week rows.")

    defense_yards_history = build_walkforward_defense_yards_allowed(pbp)
    print(f"Built walk-forward defense yards-allowed history: {len(defense_yards_history)} team-week rows.")

    # Five-row comparison (was four-row EPA-only) - "yards" is the LIVE
    # SYSTEM'S ACTUAL DEFAULT opponent-adjustment path (now that its
    # no-op bug is fixed - see finalize_qb_floors' docstring in
    # predict_nfl.py), and it was NEVER in the comparison before this
    # fix. Run EVERY time the backtest runs (not just once during
    # development) so the decision of which adjustment to use stays
    # visible and re-checked as more seasons/weeks accumulate.
    print()
    print("=" * 60)
    print("OPPONENT ADJUSTMENT COMPARISON (none / yards / epa x3)")
    print("=" * 60)
    adjustment_results = []
    configs = [
        ("none", None, "none", 0.0),
        ("yards", defense_yards_history, "yards", 0.0),
        ("epa_0.05", None, "epa", 0.05),
        ("epa_0.10", None, "epa", 0.10),
        ("epa_0.15", None, "epa", 0.15),
    ]
    for label, yards_hist, mode, scale in configs:
        cfg_preds = replay_all_games(qb_gamelog, defense_epa_history, defense_yards_history=yards_hist,
                                      adjustment_mode=mode, epa_scale=scale)
        cfg_df = pd.DataFrame(cfg_preds)
        cfg_brier = brier_score_loss(cfg_df["hit"].values, cfg_df["predicted_prob"].values)
        cfg_logloss = log_loss(cfg_df["hit"].values, np.clip(cfg_df["predicted_prob"].values, 1e-6, 1 - 1e-6))
        adjustment_results.append((label, cfg_brier, cfg_logloss))
        print(f"  {label:<12} Brier={round(cfg_brier,4):<8} LogLoss={round(cfg_logloss,4)}")
    best_label, best_brier, _ = min(adjustment_results, key=lambda r: r[1])
    print("-" * 60)
    print(f"WINNER: {best_label} (Brier={round(best_brier,4)})")
    if best_label == "yards":
        print("This matches the live system's current default (yards-allowed opponent "
              "adjustment, EPA disabled) - no code change needed.")
    elif best_label == "none":
        print("Pure shrinkage (no opponent adjustment at all) wins - consider disabling "
              "the yards-allowed adjustment too if this holds at larger sample sizes.")
    else:
        print(f"An EPA scaling beats the live system's current default - consider setting "
              f"EPA_ENABLED=True and EPA_Z_SCALE accordingly in predict_nfl.py if this is "
              f"consistent across runs, NOT based on a single comparison.")
    print("=" * 60)
    print()

    # LIVE_ADJUSTMENT_MODE tracks what predict_nfl.py's opponent_adjustment()
    # actually uses in production: EPA_ENABLED=False means the live system
    # uses the yards-allowed path by default (now that its no-op bug is
    # fixed - see finalize_qb_floors' docstring in predict_nfl.py), NOT
    # "no adjustment at all". An earlier version of this comment/constant
    # (LIVE_EPA_SCALE=0.0) was written when the live yards-allowed path
    # was still silently a no-op, so "EPA disabled" and "no adjustment"
    # were indistinguishable in practice. They are NOT the same thing now
    # that yards-allowed actually works - this constant has to say WHICH
    # non-EPA adjustment is live, not just that EPA isn't.
    LIVE_ADJUSTMENT_MODE = "epa" if EPA_ENABLED else "yards"
    LIVE_DEFENSE_YARDS_HISTORY = defense_yards_history if LIVE_ADJUSTMENT_MODE == "yards" else None
    predictions = replay_all_games(
        qb_gamelog, defense_epa_history,
        defense_yards_history=LIVE_DEFENSE_YARDS_HISTORY,
        adjustment_mode=LIVE_ADJUSTMENT_MODE,
        epa_scale=LIVE_EPA_SCALE,
    )

    if not predictions:
        print("WARNING: 0 predictions generated - check MIN_GAMES_BEFORE_PREDICTING and input data.")
    else:
        os.makedirs("backtest_output", exist_ok=True)
        pred_df = pd.DataFrame(predictions)
        csv_path = "backtest_output/predictions.csv"
        pred_df.to_csv(csv_path, index=False)
        print(f"Saved {len(pred_df)} predictions to {csv_path}")

        y_true = pred_df["hit"].values
        y_prob = pred_df["predicted_prob"].values

        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))
        base_rate = float(np.mean(y_true))
        naive_brier = base_rate * (1 - base_rate)
        skill = naive_brier - brier
        print(f"Brier score: {round(brier, 4)}")
        print(f"Log loss: {round(logloss, 4)}")
        print(f"Model Brier: {round(brier,4)} | Naive baseline (base rate = {round(base_rate,3)}): "
              f"{round(naive_brier,4)} | Skill: {'+' if skill>=0 else ''}{round(skill,4)}")

        buckets = compute_calibration_buckets(predictions)
        print()
        print("=" * 100)
        print("CALIBRATION BUCKET TABLE (predicted probability vs. actual hit rate)")
        print("=" * 100)
        print("SAMPLE-SIZE CAVEAT: backtest predictions accumulate QB history across")
        print("seasons. The live system only sees the current season's data. A week-5")
        print("backtest prediction may use up to 10 prior games, while the live system")
        print(f"has only about 4. The Early-Season columns below restrict to predictions")
        print(f"with <= {EARLY_SEASON_MAX_PRIOR_GAMES} prior games - closer to what the live system sees early")
        print("in a real season. If those deltas are worse than the full-sample deltas,")
        print("treat early-season live predictions with extra caution.")
        print("-" * 100)
        print(f"{'Bucket':<10}{'N':>7}{'Actual%':>9}{'Pred%':>8}{'Delta':>8}   "
              f"{'EarlyN':>7}{'EarlyAct%':>11}{'EarlyDelta':>12}")
        print("-" * 100)
        for bk in buckets:
            delta_str = f"{'+' if bk['delta']>=0 else ''}{round(bk['delta']*100,1)}%"
            if bk["early_n"] > 0:
                early_act_str = f"{round(bk['early_actual_hit_rate']*100,1)}%"
                early_delta_str = f"{'+' if bk['early_delta']>=0 else ''}{round(bk['early_delta']*100,1)}%"
            else:
                early_act_str = "n/a"
                early_delta_str = "n/a"
            print(f"{bk['bucket_label']:<10}{bk['n']:>7}{round(bk['actual_hit_rate']*100,1):>8}%"
                  f"{round(bk['predicted_midpoint']*100,1):>7}%{delta_str:>8}   "
                  f"{bk['early_n']:>7}{early_act_str:>11}{early_delta_str:>12}")
        print("=" * 100)
        worst = max(buckets, key=lambda b: abs(b["delta"])) if buckets else None
        if worst:
            print(f"WORST-CALIBRATED BUCKET (full sample): {worst['bucket_label']} "
                  f"(delta={'+' if worst['delta']>=0 else ''}{round(worst['delta']*100,1)}%, n={worst['n']})")
            if worst["n"] < 30:
                print(f"  NOTE: n={worst['n']} is a small sample - this bucket's delta may not be reliable "
                      f"yet, check if it persists across future weekly runs before trusting it.")
        early_season_buckets = [b for b in buckets if b["early_n"] > 0]
        worst_early = max(early_season_buckets, key=lambda b: abs(b["early_delta"])) if early_season_buckets else None
        if worst_early:
            print(f"WORST-CALIBRATED BUCKET (early-season only): {worst_early['bucket_label']} "
                  f"(delta={'+' if worst_early['early_delta']>=0 else ''}{round(worst_early['early_delta']*100,1)}%, "
                  f"n={worst_early['early_n']})")
        print("=" * 100)
        print()

        os.makedirs("docs", exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax)
        ax.set_title("NFL QB Passing Yards OVER - Calibration (secondary model; team totals are primary)")
        fig.tight_layout()
        fig.savefig("docs/calibration_curve.png", dpi=120)
        print("Saved calibration curve PNG to docs/calibration_curve.png")

        html = render_calibration_html(predictions, buckets, brier, logloss, "docs/calibration_curve.png")
        with open("docs/calibration.html", "w") as f:
            f.write(html)
        print("Saved docs/calibration.html")

        # Build and save the joint outcome table so the live dashboard
        # can query real conditional probabilities without re-running
        # the whole backtest. Saved as CSV, not parquet, so the main
        # script doesn't need an extra file-format dependency.
        joint_df = build_joint_outcome_table(pbp)
        joint_csv_path = "docs/joint_outcomes.csv"
        joint_df.to_csv(joint_csv_path, index=False)
        print(f"Saved joint outcome table ({len(joint_df)} rows) to {joint_csv_path}")

    print()
    print("=" * 70)
    print("TEAM TOTAL BACKTEST")
    print("=" * 70)
    team_games = build_walkforward_team_scoring(pbp)
    print(f"[diag] team_games total rows: {len(team_games)}")
    print(f"[diag] cum_pts_for non-null: {team_games['cum_pts_for'].notna().sum()}")
    print(f"[diag] games_played_before >= 3: {(team_games['games_played_before'] >= 3).sum()}")
    team_total_std = 9.86
    try:
        from predict_nfl import fit_std_dev_constants, fit_injury_multipliers
        _, _, team_total_std = fit_std_dev_constants(pbp)
        all_injuries = []
        for season in BACKTEST_SEASONS:
            try:
                season_injuries = nfl.import_injuries([season])
                all_injuries.append(season_injuries)
            except Exception as e:
                print(f"WARNING: could not load injuries for season={season}: {e}")
        injuries_all = pd.concat(all_injuries, ignore_index=True) if all_injuries else pd.DataFrame()
        injury_mults = fit_injury_multipliers(pbp, injuries_all) if not injuries_all.empty else None
        qb_out_signal = build_walkforward_qb_out_signal(pbp, injuries_all) if not injuries_all.empty else None
    except Exception as e:
        print(f"WARNING: could not build injury adjustment inputs: {e}")
        injury_mults, qb_out_signal = None, None

    team_predictions_no_injury = replay_team_totals(team_games, use_injury_adjustment=False)
    print(f"[diag] team_predictions_no_injury count: {len(team_predictions_no_injury)}")
    team_predictions_no_injury = score_team_total_predictions(team_predictions_no_injury, std_dev=team_total_std)
    tp_no_inj_df = pd.DataFrame(team_predictions_no_injury)
    brier_no_inj, se_no_inj, n_no_inj = brier_with_se(tp_no_inj_df["hit"].values, tp_no_inj_df["predicted_prob"].values)
    base_rate_no_inj = float(tp_no_inj_df["hit"].mean())
    naive_brier_no_inj = base_rate_no_inj * (1 - base_rate_no_inj)

    if injury_mults and qb_out_signal:
        team_predictions_with_injury = replay_team_totals(team_games, use_injury_adjustment=True,
                                                            injury_mults=injury_mults, qb_out_signal=qb_out_signal)
        team_predictions_with_injury = score_team_total_predictions(team_predictions_with_injury, std_dev=team_total_std)
        tp_with_inj_df = pd.DataFrame(team_predictions_with_injury)
        brier_with_inj, se_with_inj, n_with_inj = brier_with_se(tp_with_inj_df["hit"].values, tp_with_inj_df["predicted_prob"].values)
        base_rate_with_inj = float(tp_with_inj_df["hit"].mean())
        naive_brier_with_inj = base_rate_with_inj * (1 - base_rate_with_inj)
        team_predictions = team_predictions_with_injury
        team_brier = brier_with_inj
    else:
        team_predictions = team_predictions_no_injury
        team_brier = brier_no_inj
        brier_with_inj = None
        naive_brier_with_inj = None

    team_logloss = log_loss(pd.DataFrame(team_predictions)["hit"].values,
                             np.clip(pd.DataFrame(team_predictions)["predicted_prob"].values, 1e-6, 1 - 1e-6))
    print()
    print("=" * 78)
    print("INJURY ADJUSTMENT COMPARISON")
    print("=" * 78)
    print(f"{'Condition':<24}{'Brier':>10}{'SE':>10}{'n':>8}{'NaiveBase':>12}")
    print("-" * 78)
    print(f"{'without injury adj':<24}{round(brier_no_inj,4):>10}{round(se_no_inj,4):>10}{n_no_inj:>8}{round(naive_brier_no_inj,4):>12}")
    if brier_with_inj is not None:
        print(f"{'with injury adj':<24}{round(brier_with_inj,4):>10}{round(se_with_inj,4):>10}{n_with_inj:>8}{round(naive_brier_with_inj,4):>12}")
        gap = abs(brier_with_inj - brier_no_inj)
        combined_se = math.sqrt(se_with_inj**2 + se_no_inj**2)
        print(f"Gap: {round(gap,4)}, combined SE: {round(combined_se,4)}")
        t_stat = gap / combined_se if combined_se > 0 else 0.0
        print(f"t-stat (gap / combined SE): {round(t_stat, 2)}")
        if t_stat < 2:
            print(f"Gap {round(gap,4)} vs combined SE {round(combined_se,4)} - does not clear "
                  f"2 SE (t={round(t_stat,2)}, ~95% confidence). Suggestive, not statistically significant.")
        else:
            print(f"Gap {round(gap,4)} vs combined SE {round(combined_se,4)} - clears 2 SE "
                  f"(t={round(t_stat,2)}). Statistically significant on this sample.")
    else:
        print("Injury adjustment comparison skipped - could not build the QB-out signal (see warnings above).")
    print("=" * 78)
    print()
    points_per_yard, _ = fit_yards_to_points_rate(pbp)
    wf_yards = build_walkforward_bottom_up_yards(pbp)
    bu_predictions = replay_bottom_up_team_totals(wf_yards, points_per_yard)
    actual_lookup = team_games.set_index(["game_id", "team"])["pts_for"].to_dict()
    for pr in bu_predictions:
        pr["actual_points"] = actual_lookup.get((pr["game_id"], pr["team"]))
    bu_predictions = [pr for pr in bu_predictions if pr["actual_points"] is not None]
    print(f"[diag] bu_predictions with actuals: {len(bu_predictions)}")

    if bu_predictions and team_predictions_no_injury:
        run_topdown_vs_bottomup_matrix(team_games, bu_predictions, team_predictions_no_injury, team_total_std)
    else:
        print("WARNING: could not build the top-down/bottom-up comparison - skipping.")

    print("Backtest complete.")
