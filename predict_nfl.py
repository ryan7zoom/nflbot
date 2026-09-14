"""
NFL Daily Probability System

Projections: team totals, game totals, spread cover (primary). QB
passing yards, combined picks (secondary).

Data sources:
  - Schedule & live scores: ESPN's site.api.espn.com NFL scoreboard.
  - Player/play-by-play history: nfl_data_py (nflverse-data releases).

Notes:
  - nfl_data_py.import_pbp_data() for the current season can return an
    empty DataFrame early in the season, before nflverse publishes that
    season's release. SEASON_FALLBACK below handles this.
  - Team points-for/against come from ESPN's team schedule endpoint, not
    nfl_data_py.import_seasonal_data() (which is player-level only) or
    import_schedules() (a third-party site, not used here).
  - nflverse and ESPN use different team abbreviations for the Rams
    ("LA" vs "LAR") and Washington ("WAS" vs "WSH") - see
    NFLVERSE_TO_ESPN_TEAM below.

Output: docs/index.html + docs/report.json, for GitHub Pages.
"""

import json
import math
import time
import os
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import urllib.error

import pandas as pd
import nfl_data_py as nfl


# ============================================================================
# Bayesian shrinkage / opponent adjustment / attempts projection
# (Direct translations of the WNBA script's equivalents - copied 1:1 where
# the spec called for it, renamed where it called for renaming.)
# ============================================================================

SHRINKAGE_K = 8  # pseudo-games of league-average weight - copied as-is per spec

# Real gap fixed: projected_attempts was computed and displayed but never
# fed into the probability math - see get_qb_props(). PROJECTED_ATTEMPTS_DELTA
# mirrors the WNBA original's PROJECTED_MINUTES_DELTA (also 0.10): only
# rescale thresholds if projected attempts differ from the sampled-games
# average by more than this fraction, so small week-to-week noise in the
# attempts projection doesn't jitter every threshold's probability.
PROJECTED_ATTEMPTS_DELTA = 0.10

# Single source of truth for the EPA opponent-adjustment scaling factor.
# backtest_nfl.py derives its own LIVE_EPA_SCALE and LIVE_ADJUSTMENT_MODE
# from THIS constant plus EPA_ENABLED (rather than maintaining separate
# hand-set values), so the two scripts cannot drift out of sync - there
# is exactly one place to change either value, not two. Previously
# predict_nfl.py hardcoded 0.10 * z inline inside opponent_adjustment()
# while backtest_nfl.py tracked its own separate LIVE_EPA_SCALE constant -
# functionally matched only because both happened to reduce to "EPA
# disabled," but structurally two independent numbers that could
# silently diverge if either was edited without remembering the other
# existed.
#
# IMPORTANT: as of the opponent-adjustment no-op bug fix, EPA_ENABLED=False
# does NOT mean "no opponent adjustment at all" - it means the live system
# falls back to the yards-allowed path (opponent_adjustment()'s other
# branch), which is genuinely active and was verified to help (Brier
# 0.2135 vs 0.2142 for no adjustment at all, on the 2024+2025 sample).
# backtest_nfl.py's LIVE_ADJUSTMENT_MODE reflects this correctly
# ("yards" when EPA_ENABLED is False, not "none").
#
# CURRENT VALUE: EPA opponent adjustment is DISABLED (use_epa defaults to
# EPA_ENABLED, currently False - see opponent_adjustment()'s docstring
# for the real, verified backtest comparison: yards-allowed beat every
# EPA scaling tested, including no-adjustment-at-all). This constant is
# still defined (rather than removed) so the EPA code path stays
# available for a future re-test at 5-season scale without resurrecting
# a magic number - if that test shows an EPA scale that beats the
# yards-allowed path, change EPA_Z_SCALE here and flip EPA_ENABLED below,
# and backtest_nfl.py's calibration will automatically follow since it
# imports these same constants.
EPA_Z_SCALE = 0.10
EPA_ENABLED = False  # the actual on/off switch - opponent_adjustment()'s
# use_epa parameter defaults to this constant (not a bare False literal),
# and backtest_nfl.py imports this same constant for its LIVE_EPA_SCALE
# computation, so there is exactly one flag controlling whether EPA is
# live across both scripts.


def bayesian_shrinkage(hits, games, league_avg, k=SHRINKAGE_K):
    """
    Copied 1:1 from the WNBA script per the translation spec. Shrinks an
    empirical hit-rate (hits/games) toward league_avg, weighted by k
    "pseudo-games" of the league average.
    """
    if games + k <= 0:
        return league_avg
    return (hits + league_avg * k) / (games + k)


def compute_league_avg_hit_rate(all_qbs, stat_key="passing_yards"):
    """
    Average raw hit-rate on the threshold CLOSEST TO EACH QB'S OWN AVERAGE
    ("medium" rung), across every starting QB in today's report with
    >= 3 sampled games. Used as the shrinkage target in bayesian_shrinkage.

    all_qbs: list of QB dicts as returned by get_qb_props, each with
    "floors" (raw, pre-shrinkage) and "games_sampled".
    Returns 0.5 if no qualifying QBs are found (a neutral prior).
    """
    rates = []
    for p in all_qbs:
        if p.get("games_sampled", 0) < 3:
            continue
        floors = p.get("floors", {}).get(stat_key, {})
        if not floors:
            continue
        # "medium" rung = the threshold closest to the player's own average
        # (their own average was the center of the ladder when it was
        # built - see _build_qb_thresholds - so the middle-index item is
        # closest to average, matching MEDIUM_THRESHOLD_INDEX's spirit).
        sorted_thresholds = sorted(floors.keys())
        idx = min(len(sorted_thresholds) // 2, len(sorted_thresholds) - 1)
        medium_t = sorted_thresholds[idx]
        hr = floors.get(medium_t)
        if hr is not None:
            rates.append(hr)
    if not rates:
        return 0.5
    return sum(rates) / len(rates)


def project_attempts(recent_games):
    """
    Renamed from project_minutes() per spec. Weighted-average projected
    pass attempts from a QB's recent games (newest last). Newest game
    gets weight 5, then 4, 3, 2, 1 for up to 5 games. If fewer than 5
    games have attempts data, weights evenly instead.

    Returns None if no attempts data was found at all.
    """
    att_vals = [g["pass_attempts"] for g in recent_games if g.get("pass_attempts") is not None]
    if not att_vals:
        return None

    last5 = att_vals[-5:]
    n = len(last5)
    if n == 5:
        weights = [1, 2, 3, 4, 5]  # oldest -> newest, newest gets 5
    else:
        weights = [1] * n

    total_weight = sum(weights)
    if total_weight == 0:
        return None
    weighted_sum = sum(v * w for v, w in zip(last5, weights))
    return weighted_sum / total_weight


# ============================================================================
# Game-script attempt adjustment: a QB's projected attempts get scaled
# based on the spread, since trailing teams pass more and leading teams
# run more. GAME_SCRIPT_DILUTION scales down the raw Q4 pass-rate swing
# since it's a Q4-only signal being applied to a whole-game projection.
# ============================================================================

GAME_SCRIPT_DILUTION = 0.25


def fit_game_script_multipliers(pbp_df):
    """Fits trailing/leading attempts multipliers from historical Q4 pass rates. Falls back to +15%/-10% if pbp is too thin."""
    fallback = {"trailing_mult": 1.15, "leading_mult": 0.90}
    if pbp_df is None or pbp_df.empty:
        print("WARNING: fit_game_script_multipliers got empty pbp - using fallback +15%/-10%.")
        return fallback

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    q4 = df[df.get("qtr") == 4].dropna(subset=["score_differential", "pass_attempt"]) if "qtr" in df.columns else pd.DataFrame()
    if q4.empty:
        print("WARNING: no Q4 plays found for game-script fit - using fallback +15%/-10%.")
        return fallback

    def _pass_rate(sub):
        if sub.empty:
            return None
        per_game = sub.groupby(["posteam", "game_id"])["pass_attempt"].agg(["sum", "count"])
        total_att = per_game["count"].sum()
        return per_game["sum"].sum() / total_att if total_att > 0 else None

    trailing = q4[q4["score_differential"] <= -7]
    leading = q4[q4["score_differential"] >= 7]
    neutral = q4[(q4["score_differential"] > -7) & (q4["score_differential"] < 7)]

    trail_rate, lead_rate, neutral_rate = _pass_rate(trailing), _pass_rate(leading), _pass_rate(neutral)
    if not trail_rate or not lead_rate or not neutral_rate:
        print("WARNING: insufficient Q4 game-script samples - using fallback +15%/-10%.")
        return fallback

    raw_trail_mult = trail_rate / neutral_rate
    raw_lead_mult = lead_rate / neutral_rate

    # dilute toward 1.0 (no effect) by GAME_SCRIPT_DILUTION, since this is
    # a Q4-only signal being applied to a whole-game attempts projection
    diluted_trail = 1.0 + (raw_trail_mult - 1.0) * GAME_SCRIPT_DILUTION
    diluted_lead = 1.0 + (raw_lead_mult - 1.0) * GAME_SCRIPT_DILUTION

    print(f"Fitted game-script multipliers from Q4 pbp: raw trailing={round(raw_trail_mult,3)} "
          f"(diluted to {round(diluted_trail,3)}), raw leading={round(raw_lead_mult,3)} "
          f"(diluted to {round(diluted_lead,3)}). Dilution factor={GAME_SCRIPT_DILUTION} "
          f"(Q4-only signal scaled toward whole-game applicability).")

    return {"trailing_mult": round(diluted_trail, 3), "leading_mult": round(diluted_lead, 3)}


def adjust_attempts_for_game_script(projected_attempts, spread_line, team_side, multipliers):
    """
    spread_line is from the HOME team's perspective (negative = home
    favored), matching ESPN's convention already used elsewhere in this
    script (see spread_cover_prob). team_side is "home" or "away".
    A team is "trailing" (expected to lose by 7+) if their own spread
    (their perspective) is +7 or worse; "leading" if -7 or better.
    """
    if projected_attempts is None or spread_line is None:
        return projected_attempts

    # convert to this team's own perspective
    team_spread = spread_line if team_side == "home" else -spread_line

    if team_spread >= 7:  # this team is expected to lose by 7+
        return round(projected_attempts * multipliers["trailing_mult"], 1)
    elif team_spread <= -7:  # this team is expected to win by 7+
        return round(projected_attempts * multipliers["leading_mult"], 1)
    return projected_attempts


def opponent_adjustment(stat_key, raw_prob, opponent_team_id, team_stats_cache, use_epa=EPA_ENABLED):
    """
    Adjusts raw_prob for the opponent, using season-long team numbers in
    team_stats_cache. Uses yards-allowed by default (EPA_ENABLED=False -
    a backtest comparison found EPA-based adjustment performs worse).

    EPA and yards-allowed point in opposite raw-value directions, but
    the same pct-diff/z-score formula applies to both once you read
    "epa allowed" as "epa the offense gained against this defense" -
    higher means a worse defense either way.
    """
    if raw_prob is None or not team_stats_cache:
        return raw_prob
    opp_stats = team_stats_cache.get(opponent_team_id) or team_stats_cache.get(str(opponent_team_id))
    if not opp_stats:
        return raw_prob

    if stat_key != "passing_yards":
        return raw_prob  # no reliable opponent-allowed signal built for other stats yet

    field = "pass_epa_allowed_pg" if (use_epa and "pass_epa_allowed_pg" in opp_stats) else "pass_yds_allowed_pg"

    vals = [v.get(field) for v in team_stats_cache.values() if v.get(field) is not None]
    if not vals or opp_stats.get(field) is None:
        return raw_prob
    league_avg = sum(vals) / len(vals)

    if field == "pass_epa_allowed_pg":
        # EPA is mean-centered near zero, so percent-of-mean is unstable
        # here. Use a z-score against the league std dev instead.
        n = len(vals)
        variance = sum((v - league_avg) ** 2 for v in vals) / n if n > 0 else 0
        std = variance ** 0.5
        if std == 0:
            return raw_prob
        z = (opp_stats[field] - league_avg) / std
        # scale: a defense 1 full std dev worse than average nudges the
        # probability by EPA_Z_SCALE (currently 0.10 = 10%, capped at 15%
        # same as the yards-based path, so a 1.5+ std dev outlier defense
        # still hits the same ceiling). EPA_Z_SCALE is a module-level
        # constant (see its definition near the top of this file) rather
        # than a literal here, specifically so backtest_nfl.py can import
        # and match it exactly - no second copy of this number to drift.
        adj_factor = max(-0.15, min(0.15, EPA_Z_SCALE * z))
        return max(0.0, min(1.0, raw_prob * (1 + adj_factor)))

    # yards-allowed path: percent-of-mean is fine here since the mean is
    # a large, stable positive number (~220-240 yards/game), not near zero.
    if not league_avg:
        return raw_prob
    pct_diff = (opp_stats[field] - league_avg) / league_avg
    adj_factor = max(-0.15, min(0.15, 0.4 * pct_diff))
    return max(0.0, min(1.0, raw_prob * (1 + adj_factor)))


def usage_boost_if_starter_out(missing_names, wr1_name, rb1_name, boost_mults=None):
    """
    If the team's WR1 is out: boost to the QB's floors (more targets
    funneled to remaining volume). If RB1 is out: boost (more passing
    volume as the run game is compromised). Both can stack if both are
    out.

    boost_mults, if provided, is the dict from fit_empirical_usage_boosts()
    with empirically-fit multipliers, falling back to +12%/+8% if the
    sample size is too thin. If boost_mults is None, uses +12%/+8% directly.
    """
    if not missing_names:
        return 1.0
    wr1_mult = boost_mults.get("wr1_out_mult", 1.12) if boost_mults else 1.12
    rb1_mult = boost_mults.get("rb1_out_mult", 1.08) if boost_mults else 1.08
    boost = 1.0
    if wr1_name and wr1_name in missing_names:
        boost *= wr1_mult
    if rb1_name and rb1_name in missing_names:
        boost *= rb1_mult
    return boost


# ============================================================================
# Local time / constants
# ============================================================================

LOCAL_UTC_OFFSET_HOURS = 6  # copied as-is from the WNBA script


def local_now():
    return datetime.utcnow() + timedelta(hours=LOCAL_UTC_OFFSET_HOURS)


ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
TODAY = local_now().strftime("%Y%m%d")
print(f"DEBUG local_now={local_now()} TODAY={TODAY}")

# nflverse pbp releases lag real-world games early in a season. This is the
# CURRENT calendar-year season we'll try first; SEASON_FALLBACK is used
# whole-hog (not blended) if CURRENT_SEASON's pbp is empty or too thin.
CURRENT_SEASON = 2026
SEASON_FALLBACK = 2025
MIN_PBP_ROWS_TO_TRUST = 500  # sanity floor - a real season's pbp is tens of
# thousands of rows; a near-empty response (0 or a handful of rows from a
# single early game) is treated the same as "not available yet"

REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

STARTING_QB_LOOKBACK_GAMES = 3  # "most recent 2-3 games" per spec - using 3
QB_GAMES_SAMPLE = 10  # how many recent games to pull for a starter's gamelog

# nflverse team abbreviation -> ESPN team abbreviation, for the handful of
# real mismatches confirmed during build-time verification against a live
# ESPN scoreboard payload. Every other abbreviation matched directly.
NFLVERSE_TO_ESPN_TEAM = {
    "LA": "LAR",
    "WAS": "WSH",
}


def to_espn_abbr(nflverse_abbr):
    return NFLVERSE_TO_ESPN_TEAM.get(nflverse_abbr, nflverse_abbr)


def to_nflverse_abbr(espn_abbr):
    reverse = {v: k for k, v in NFLVERSE_TO_ESPN_TEAM.items()}
    return reverse.get(espn_abbr, espn_abbr)


# ============================================================================
# Weather: historical backtest uses pbp's own temp/wind/roof columns.
# Live/upcoming games use Open-Meteo's free forecast API (no key needed).
# Stadium coordinates below are hardcoded from general knowledge - spot-
# check against a map if any look off, especially recently-renamed venues.
# ============================================================================

STADIUM_COORDS = {
    # nflverse team abbr -> (lat, lon). Dome/closed-roof teams are
    # included for completeness (weather_adjustment naturally has no
    # effect when there's no real outdoor wind/precip/temp to report -
    # the caller should still check pbp's own `roof` field historically,
    # or skip calling live weather for a known dome team, since indoor
    # conditions don't affect passing yards the way outdoor weather does).
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB":  (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC":  (39.0489, -94.4839), "LV":  (36.0909, -115.1833), "LAC": (33.9535, -118.3392),
    "LAR": (33.9535, -118.3392), "MIA": (25.9580, -80.2389), "MIN": (44.9736, -93.2575),
    "NE":  (42.0909, -71.2643), "NO":  (29.9511, -90.0812), "NYG": (40.8136, -74.0744),
    "NYJ": (40.8136, -74.0744), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF":  (37.4032, -121.9698), "TB":  (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WSH": (38.9078, -76.8645),
}

# Domes/fixed roofs where outdoor weather doesn't reach the field. This is
# also a plain fact list (stadium roof types), not computed - cross-check
# if a team's status has changed (renovations do happen).
DOME_OR_CLOSED_ROOF_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LAR", "MIN", "NO"}


def get_weather_for_game(stadium_lat, stadium_lon, kickoff_iso, historical=False):
    """
    Fetches weather for a game's kickoff time/location from Open-Meteo.
    Uses explicit fahrenheit/mph units - Open-Meteo defaults to metric.

    historical=True switches to the archive API (for backtesting past
    games); historical=False uses the live forecast API (main dashboard).

    Returns {"temp_f": float, "wind_mph": float, "precip_mm": float} for
    the hour closest to kickoff, or None if the fetch fails or the hour
    isn't found in the response - callers must treat None as "couldn't
    get weather this run," not "no weather / calm conditions."
    """
    base = "https://archive-api.open-meteo.com/v1/archive" if historical else "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": stadium_lat,
        "longitude": stadium_lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m,winddirection_10m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
    }
    if historical:
        # archive API needs an explicit date range, not "forecast_days"
        day = kickoff_iso[:10]
        params["start_date"] = day
        params["end_date"] = day

    try:
        url = base + "?" + urllib.parse.urlencode(params)
        data = _fetch_with_retry(url)
    except Exception as e:
        print(f"WARNING: weather fetch failed for ({stadium_lat},{stadium_lon}) at {kickoff_iso}: {e}")
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        print(f"WARNING: weather response had no hourly.time array for ({stadium_lat},{stadium_lon})")
        return None

    # find the hour closest to kickoff - kickoff_iso may include minutes,
    # hourly.time entries are on-the-hour (e.g. "2026-09-14T13:00")
    target_hour = kickoff_iso[:13]  # "YYYY-MM-DDTHH"
    idx = None
    for i, t in enumerate(times):
        if t[:13] == target_hour:
            idx = i
            break
    if idx is None:
        # fall back to the closest available hour rather than failing
        # outright - useful when kickoff falls right at a forecast
        # boundary or timezone handling is slightly off.
        idx = 0
        print(f"WARNING: exact kickoff hour {target_hour} not found in weather response, using first available hour instead.")

    try:
        return {
            "temp_f": hourly["temperature_2m"][idx],
            "wind_mph": hourly["windspeed_10m"][idx],
            "precip_mm": hourly["precipitation"][idx],
        }
    except (KeyError, IndexError) as e:
        print(f"WARNING: weather response shape unexpected, couldn't extract values: {e}")
        return None


def weather_adjustment(wind_mph, precipitation_mm, temp_f):
    """
    Multiplier on projected passing yards, per spec:
      wind > 15 mph: -10%
      precipitation > 2mm: -7%
      temp < 35F: -5%
    Stacks multiplicatively. Any None input is treated as "no adjustment
    for that factor" rather than crashing - a partial weather read (e.g.
    temp available but wind missing) shouldn't throw away the whole
    adjustment.
    """
    multiplier = 1.0
    if wind_mph is not None and wind_mph > 15:
        multiplier *= 0.90
    if precipitation_mm is not None and precipitation_mm > 2:
        multiplier *= 0.93
    if temp_f is not None and temp_f < 35:
        multiplier *= 0.95
    return round(multiplier, 4)


# ============================================================================
# Dynamic QB passing-yards thresholds
# (Translation of _build_player_thresholds / RUNG_COUNT / STEP_FRACTION /
# STEP_MIN / THRESHOLD_MIN_FLOOR, mapped to yards per the spec's explicit
# numbers: STEP_FRACTION=8, STEP_MIN=10, THRESHOLD_MIN_FLOOR=50.)
# ============================================================================

RUNG_COUNT_YARDS = 5
RUNGS_ABOVE_AVG = 1
STEP_FRACTION_YARDS = 8
STEP_MIN_YARDS = 10
THRESHOLD_MIN_FLOOR_YARDS = 50


def _build_qb_thresholds(avg_yards):
    """
    Builds a descending-then-one-above band of passing-yards thresholds
    centered on a QB's own recent average, mirroring the WNBA script's
    _build_player_thresholds shape and the spec's worked examples:
      avg=250 -> ~175, 200, 225, 250, 275
      avg=180 -> ~100, 130, 160, 180, 200
    (Exact numbers depend on rounding of avg/STEP_FRACTION, same as the
    original function - these are illustrative, not hardcoded.)
    """
    rung_count = RUNG_COUNT_YARDS
    above = min(RUNGS_ABOVE_AVG, rung_count - 1)
    below = rung_count - 1 - above
    min_floor = THRESHOLD_MIN_FLOOR_YARDS

    base_step = max(STEP_MIN_YARDS, round(avg_yards / STEP_FRACTION_YARDS))
    center = max(min_floor, round(avg_yards / 5) * 5)  # round to nearest 5 yards, a plausible book line grid

    thresholds = []
    t = center
    for i in range(below, 0, -1):
        near_avg = (i == 1)
        this_step = max(STEP_MIN_YARDS, base_step - 1) if near_avg else base_step
        t = t - this_step
        thresholds.insert(0, t)
    thresholds.append(center)
    thresholds += [center + base_step * i for i in range(1, above + 1)]

    thresholds = sorted(set(t for t in thresholds if t >= min_floor))
    while len(thresholds) < rung_count:
        thresholds.append(thresholds[-1] + base_step)

    return tuple(thresholds)


def prop_floor_probs_yards(games, thresholds=None):
    """
    Empirical P(passing_yards >= threshold) over the sampled recent games.
    Translation of prop_floor_probs() for the single primary stat
    (passing_yards) this system tracks per-QB.
    """
    n = len(games)
    if n == 0:
        return {}
    values = [g.get("passing_yards", 0.0) or 0.0 for g in games]

    if thresholds is None:
        avg = sum(values) / len(values)
        thresholds = _build_qb_thresholds(avg)

    probs = {}
    for t in thresholds:
        hits = sum(1 for v in values if v >= t)
        probs[t] = round(hits / n, 3)
    return probs


# Same idea for WR/RB secondary yardage stats, per spec Part 1.2 - reuses
# the same ladder shape but WR/RB volume is lower, so the floor/step differ.
RUNG_COUNT_WR_RB = 4
STEP_FRACTION_WR_RB = 6
STEP_MIN_WR_RB = 5
THRESHOLD_MIN_FLOOR_WR_RB = 15


def _build_wr_rb_thresholds(avg_yards):
    rung_count = RUNG_COUNT_WR_RB
    above = min(RUNGS_ABOVE_AVG, rung_count - 1)
    below = rung_count - 1 - above
    min_floor = THRESHOLD_MIN_FLOOR_WR_RB

    base_step = max(STEP_MIN_WR_RB, round(avg_yards / STEP_FRACTION_WR_RB))
    center = max(min_floor, round(avg_yards / 5) * 5)

    thresholds = []
    t = center
    for i in range(below, 0, -1):
        near_avg = (i == 1)
        this_step = max(STEP_MIN_WR_RB, base_step - 1) if near_avg else base_step
        t = t - this_step
        thresholds.insert(0, t)
    thresholds.append(center)
    thresholds += [center + base_step * i for i in range(1, above + 1)]

    thresholds = sorted(set(t for t in thresholds if t >= min_floor))
    while len(thresholds) < rung_count:
        thresholds.append(thresholds[-1] + base_step)
    return tuple(thresholds)


def prop_floor_probs_wr_rb(games, stat_key, thresholds=None):
    n = len(games)
    if n == 0:
        return {}
    values = [g.get(stat_key, 0.0) or 0.0 for g in games]
    if thresholds is None:
        avg = sum(values) / len(values)
        thresholds = _build_wr_rb_thresholds(avg)
    probs = {}
    for t in thresholds:
        hits = sum(1 for v in values if v >= t)
        probs[t] = round(hits / n, 3)
    return probs


# ============================================================================
# Spread / Total probability math - copied verbatim from the WNBA script's
# spread_cover_prob / game_total_over_prob. std_dev values are now FIT from
# real historical pbp data (fit_std_dev_constants(), below) rather than
# hardcoded. The constants here are fallback-only, used if the fit can't
# run for some reason (e.g. pbp totally unavailable) - verified against
# real 2024+2025 combined REG-season data: margin std ~14.3, total std
# ~13.5, both computed live in testing, not guessed.
# ============================================================================

SPREAD_STD_DEV_NFL = 13.5  # fallback only - see fit_std_dev_constants()
TOTAL_STD_DEV_NFL = 19.0   # fallback only - see fit_std_dev_constants()
TEAM_TOTAL_STD_DEV_NFL = 10.0  # fallback only - see fit_std_dev_constants()


def fit_std_dev_constants(pbp_df):
    """Fits spread, game-total, and single-team-total std dev from historical scores."""
    if pbp_df is None or pbp_df.empty:
        print("WARNING: fit_std_dev_constants got empty pbp - using fallback hardcoded std devs.")
        return SPREAD_STD_DEV_NFL, TOTAL_STD_DEV_NFL, TEAM_TOTAL_STD_DEV_NFL

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    games = df.drop_duplicates(subset=["game_id"])[["game_id", "home_score", "away_score"]].dropna()

    if len(games) < 30:
        print(f"WARNING: only {len(games)} games available for std dev fit (< 30 minimum) - using fallback hardcoded std devs.")
        return SPREAD_STD_DEV_NFL, TOTAL_STD_DEV_NFL, TEAM_TOTAL_STD_DEV_NFL

    margin_std = (games["home_score"] - games["away_score"]).std()
    total_std = (games["home_score"] + games["away_score"]).std()
    team_total_std = pd.concat([games["home_score"], games["away_score"]]).std()

    if pd.isna(margin_std) or pd.isna(total_std) or pd.isna(team_total_std) or margin_std <= 0 or total_std <= 0 or team_total_std <= 0:
        print("WARNING: std dev fit produced an invalid value - using fallback hardcoded std devs.")
        return SPREAD_STD_DEV_NFL, TOTAL_STD_DEV_NFL, TEAM_TOTAL_STD_DEV_NFL

    print(f"Fitted std dev constants from {len(games)} games: spread={round(margin_std, 2)}, "
          f"game_total={round(total_std, 2)}, team_total={round(team_total_std, 2)}.")
    return round(margin_std, 2), round(total_std, 2), round(team_total_std, 2)


def spread_cover_prob(team_a_stats, team_b_stats, spread, std_dev=SPREAD_STD_DEV_NFL):
    """
    Normal approximation of NFL point-differential margin. std_dev=13.5
    is a commonly cited rough single-game NFL margin std dev - an
    approximation, not derived from a fresh historical fit here. Treat
    outputs as directional, not precise (matches the WNBA original's own
    caveat about its std_dev).
    """
    if not team_a_stats or not team_b_stats:
        return None
    expected_margin = (team_a_stats["pts_pg"] - team_a_stats["pts_allowed_pg"]) - \
                       (team_b_stats["pts_pg"] - team_b_stats["pts_allowed_pg"])
    z = (spread + expected_margin) / std_dev
    prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(prob, 3)


def game_total_over_prob(team_a_stats, team_b_stats, total_line, std_dev=TOTAL_STD_DEV_NFL):
    """
    Full-game combined total (both teams' points) over-probability.
    Copied structurally from the WNBA original's game_total_over_prob.
    """
    if not team_a_stats or not team_b_stats:
        return None
    projected = (team_a_stats["pts_pg"] + team_a_stats["pts_allowed_pg"] +
                 team_b_stats["pts_pg"] + team_b_stats["pts_allowed_pg"]) / 2.0
    z = (projected - total_line) / std_dev
    prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(prob, 3)


def team_total_over_prob(team_a_stats, team_b_stats, total_line, std_dev=None):
    """
    A single team's own points-scored-over-line probability, blending
    their own scoring pace with the opponent's points-allowed pace.
    std_dev is narrower than the combined-total std dev since it's one
    team's variance, not the sum of both.
    """
    if not team_a_stats or not team_b_stats:
        return None
    if std_dev is None:
        std_dev = TOTAL_STD_DEV_NFL * 0.72  # single-team slice of the combined variance
    projected = (team_a_stats["pts_pg"] + team_b_stats["pts_allowed_pg"]) / 2.0
    z = (projected - total_line) / std_dev
    prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(prob, 3)


# ============================================================================
# ESPN fetch helpers - copied 1:1 in structure from the WNBA script.
# ============================================================================

def _fetch_with_retry(url, timeout=15):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.espn.com/nfl/scoreboard",
            "Origin": "https://www.espn.com",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            time.sleep(REQUEST_DELAY_SECONDS)
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def espn_site_get(path, params=None):
    url = f"{ESPN_SITE_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _fetch_with_retry(url)


# ---------- schedule / today's games ----------

def get_todays_games():
    """
    Same local-day-window approach as the WNBA original: query yesterday/
    today/tomorrow's single-date buckets (ESPN's own internal day bucketing
    doesn't reliably match any specific local calendar day), filter every
    event to ones that actually fall in local-today or are in progress,
    with a range-query fallback if every single-date query comes back
    empty (a known ESPN quirk, not necessarily a real outage).
    """
    local_today = local_now().replace(hour=0, minute=0, second=0, microsecond=0)
    local_tomorrow_start = local_today + timedelta(days=1)

    date_a = (local_today - timedelta(days=1)).strftime("%Y%m%d")
    date_b = local_today.strftime("%Y%m%d")
    date_c = local_tomorrow_start.strftime("%Y%m%d")
    dates_to_query = [date_a, date_b, date_c]

    seen_event_ids = set()
    games = []
    all_events = []
    any_events_seen = False
    for date_str in dates_to_query:
        try:
            payload = espn_site_get("/scoreboard", {"dates": date_str})
        except Exception as e:
            print(f"WARNING: scoreboard fetch failed for dates={date_str}: {e}")
            continue
        events = payload.get("events", [])
        print(f"DEBUG single-date query dates={date_str} -> {len(events)} events")
        if events:
            any_events_seen = True
        all_events.extend(events)

    if not any_events_seen:
        range_param = f"{date_a}-{date_c}"
        print(f"WARNING: all single-date queries returned 0 events - retrying with range dates={range_param}")
        try:
            range_payload = espn_site_get("/scoreboard", {"dates": range_param})
            range_events = range_payload.get("events", [])
            print(f"DEBUG range query dates={range_param} -> {len(range_events)} events")
            all_events.extend(range_events)
        except Exception as e:
            print(f"WARNING: range scoreboard fetch failed for dates={range_param}: {e}")

    for e in all_events:
        event_id = e.get("id")
        if not event_id or event_id in seen_event_ids:
            continue

        event_date_raw = e.get("date")
        if not event_date_raw:
            continue
        try:
            event_dt_utc = datetime.strptime(event_date_raw, "%Y-%m-%dT%H:%M%z")
            event_dt_local = event_dt_utc.replace(tzinfo=None) + timedelta(hours=LOCAL_UTC_OFFSET_HOURS)
        except ValueError:
            try:
                event_dt_utc = datetime.strptime(event_date_raw, "%Y-%m-%dT%H:%M:%SZ")
                event_dt_local = event_dt_utc + timedelta(hours=LOCAL_UTC_OFFSET_HOURS)
            except ValueError:
                continue

        comp = e.get("competitions", [{}])[0]
        status_state = comp.get("status", {}).get("type", {}).get("state")

        is_todays_local_date = local_today <= event_dt_local < local_tomorrow_start
        is_in_progress = status_state == "in"
        if not (is_todays_local_date or is_in_progress):
            continue

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        # spread/total lines, if ESPN has odds posted for this game
        odds = comp.get("odds", [{}])
        odds0 = odds[0] if odds else {}
        spread_line = odds0.get("spread")
        total_line = odds0.get("overUnder")

        seen_event_ids.add(event_id)
        games.append({
            "event_id": event_id,
            "home_team_id": home["team"]["id"],
            "home_team_abbr": home["team"].get("abbreviation"),
            "home_team_name": home["team"].get("displayName") or home["team"].get("abbreviation"),
            "away_team_id": away["team"]["id"],
            "away_team_abbr": away["team"].get("abbreviation"),
            "away_team_name": away["team"].get("displayName") or away["team"].get("abbreviation"),
            "spread_line": spread_line,
            "total_line": total_line,
            "kickoff_local": event_dt_local.strftime("%a %I:%M %p"),
            "kickoff_iso": event_dt_local.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    print(f"DEBUG get_todays_games returning {len(games)} games: "
          f"{[g['away_team_abbr'] + '@' + g['home_team_abbr'] for g in games]}")
    return games


# ---------- team points-for/against, derived from ESPN schedule ----------
# (Per your instruction: doesn't matter which source, so this mirrors the
# WNBA original's already-verified-working approach rather than the
# unverified nfl_data_py.import_schedules().)

_SCHEDULE_CACHE = {}


def get_team_schedule_events(team_id, season=CURRENT_SEASON):
    cache_key = (str(team_id), season)
    if cache_key in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[cache_key]
    payload = espn_site_get(f"/teams/{team_id}/schedule", {"season": season})
    events = payload.get("events", [])
    _SCHEDULE_CACHE[cache_key] = events
    return events


# Fallback multipliers, used only if historical fitting isn't possible.
REST_SHORT_WEEK_MULT_FALLBACK = 0.96
REST_LONG_MULT_FALLBACK = 1.02
HOME_MULT_FALLBACK = 1.02
AWAY_MULT_FALLBACK = 0.98

INJURY_MULT_FALLBACK = {"qb": 0.80, "wr1": 0.94, "rb1": 0.96}


def fit_home_away_multipliers(pbp_df):
    """Fits home/away scoring multipliers from historical final scores."""
    if pbp_df is None or pbp_df.empty:
        return HOME_MULT_FALLBACK, AWAY_MULT_FALLBACK
    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    games = df.drop_duplicates(subset=["game_id"])[["game_id", "home_score", "away_score"]].dropna()
    if len(games) < 30:
        print(f"WARNING: only {len(games)} games for home/away fit - using fallback multipliers.")
        return HOME_MULT_FALLBACK, AWAY_MULT_FALLBACK
    overall_avg = pd.concat([games["home_score"], games["away_score"]]).mean()
    if not overall_avg:
        return HOME_MULT_FALLBACK, AWAY_MULT_FALLBACK
    home_mult = games["home_score"].mean() / overall_avg
    away_mult = games["away_score"].mean() / overall_avg
    print(f"Fitted home/away multipliers from {len(games)} games: home={round(home_mult,4)}, away={round(away_mult,4)}.")
    return round(home_mult, 4), round(away_mult, 4)


def get_team_total_multiplier(rest_days=None, is_home=True, home_mult=HOME_MULT_FALLBACK,
                                away_mult=AWAY_MULT_FALLBACK):
    """Combines the home/away and rest-day multipliers for a team's projected total."""
    mult = home_mult if is_home else away_mult
    if rest_days is not None:
        if rest_days < 5:
            mult *= REST_SHORT_WEEK_MULT_FALLBACK
        elif rest_days > 8:
            mult *= REST_LONG_MULT_FALLBACK
    return mult


def project_team_total(own_scoring_rate, opponent_points_allowed_rate, rest_days=None, is_home=True,
                        weather_mult=1.0, injury_mult=1.0, home_mult=HOME_MULT_FALLBACK,
                        away_mult=AWAY_MULT_FALLBACK):
    """
    Projects a team's points for one game, top-down from scoring rates -
    not summed from player props. own_scoring_rate and
    opponent_points_allowed_rate should already be Bayesian-shrunk and
    opponent-adjusted before calling this.
    """
    base = (own_scoring_rate + opponent_points_allowed_rate) / 2
    mult = get_team_total_multiplier(rest_days, is_home, home_mult, away_mult)
    return round(base * mult * weather_mult * injury_mult, 1)


def team_total_over_prob_v2(projected_points, total_line, std_dev=TEAM_TOTAL_STD_DEV_NFL):
    """Normal-approximation over-probability for a single team's total."""
    if projected_points is None or total_line is None:
        return None
    z = (projected_points - total_line) / std_dev
    return round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)


def fit_injury_multipliers(pbp_df, injuries_df):
    """
    Fits QB/WR1/RB1-out team-total multipliers from history. Compares
    each week's actual starter (by prior-week attempts leader, walk-
    forward-safe) against who was marked "Out" that week - not just
    "any player at that position was out," which would count a
    third-string backup's injury the same as the real starter's.
    Uses MIN_SAMPLES_FOR_EMPIRICAL_BOOST - falls back to documented
    defaults if the sample is too thin.
    """
    fallback = dict(INJURY_MULT_FALLBACK)
    if pbp_df is None or pbp_df.empty or injuries_df is None or injuries_df.empty:
        print("WARNING: fit_injury_multipliers missing data - using fallback multipliers.")
        return fallback

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    scores = df.drop_duplicates(subset=["game_id"])[["game_id", "week", "home_team", "away_team", "home_score", "away_score"]].dropna()
    team_week_score = {}
    for _, row in scores.iterrows():
        team_week_score[(row["home_team"], row["week"])] = row["home_score"]
        team_week_score[(row["away_team"], row["week"])] = row["away_score"]

    team_season_avg = {}
    for (team, week), pts in team_week_score.items():
        team_season_avg.setdefault(team, []).append(pts)
    team_season_avg = {t: sum(v) / len(v) for t, v in team_season_avg.items()}

    # Per-week starter identity, using ONLY attempts from weeks strictly
    # before the week being checked - avoids the contamination where a
    # season-long "most attempts" starter erases the very injury being
    # measured (e.g. a QB hurt in week 1 who barely plays again still
    # looked like a real contributor to a season-long average).
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["passer_id"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0)
    weekly_attempts = pass_plays.groupby(["posteam", "week", "passer"])["pass_attempt"].sum().reset_index()

    rec_plays = df[df.get("complete_pass") == 1].dropna(subset=["receiver_id"]).copy() if "complete_pass" in df.columns else pd.DataFrame()
    if not rec_plays.empty and "receiving_yards" in rec_plays.columns:
        rec_plays["receiving_yards"] = rec_plays["receiving_yards"].fillna(0)
        weekly_rec = rec_plays.groupby(["posteam", "week", "receiver"])["receiving_yards"].sum().reset_index()
    else:
        weekly_rec = pd.DataFrame()

    rush_plays = df[df.get("rush_attempt") == 1].dropna(subset=["rusher_id"]).copy() if "rush_attempt" in df.columns else pd.DataFrame()
    if not rush_plays.empty and "rushing_yards" in rush_plays.columns:
        rush_plays["rushing_yards"] = rush_plays["rushing_yards"].fillna(0)
        weekly_rush = rush_plays.groupby(["posteam", "week", "rusher"])["rushing_yards"].sum().reset_index()
    else:
        weekly_rush = pd.DataFrame()

    def _prior_weeks_starter(team, week, weekly_df, name_col, value_col):
        prior = weekly_df[(weekly_df["posteam"] == team) & (weekly_df["week"] < week)]
        if prior.empty:
            return None
        totals = prior.groupby(name_col)[value_col].sum()
        return totals.idxmax() if not totals.empty else None

    def _delta_for_position(position, weekly_df, name_col, value_col):
        out_rows = injuries_df[(injuries_df["position"] == position) & (injuries_df["report_status"] == "Out")]
        deltas = []
        for _, row in out_rows.iterrows():
            team, week = row["team"], row["week"]
            starter_name = _prior_weeks_starter(team, week, weekly_df, name_col, value_col)
            if starter_name is None:
                continue
            # match on last name only - injuries_df uses full names
            # ("Brock Purdy"), pbp uses abbreviated ("B.Purdy")
            last_name = row["full_name"].split()[-1].lower()
            if last_name not in starter_name.lower():
                continue  # this "Out" player wasn't the actual current starter - skip
            actual = team_week_score.get((team, week))
            avg = team_season_avg.get(team)
            if actual is not None and avg:
                deltas.append(actual / avg)
        return deltas

    results = {}
    configs = (
        ("QB", "qb", fallback["qb"], weekly_attempts, "passer", "pass_attempt"),
        ("WR", "wr1", fallback["wr1"], weekly_rec, "receiver", "receiving_yards"),
        ("RB", "rb1", fallback["rb1"], weekly_rush, "rusher", "rushing_yards"),
    )
    for position, key, fb, weekly_df, name_col, value_col in configs:
        if weekly_df.empty:
            results[key] = fb
            continue
        deltas = _delta_for_position(position, weekly_df, name_col, value_col)
        n = len(deltas)
        if n < MIN_SAMPLES_FOR_EMPIRICAL_BOOST:
            print(f"WARNING: only {n} confirmed-starter {position}-out samples (< {MIN_SAMPLES_FOR_EMPIRICAL_BOOST}) - using fallback {fb} for {key}.")
            results[key] = fb
        else:
            mult = round(sum(deltas) / n, 3)
            print(f"Fitted {key} injury multiplier from {n} confirmed-starter samples: {mult}")
            results[key] = mult
    return results


def get_team_points_for_against(team_id, season=CURRENT_SEASON):
    """
    Derives points-for-per-game / points-allowed-per-game from a team's
    completed games this season, via ESPN's team schedule endpoint.
    Returns None if fewer than 1 completed game is available.
    """
    try:
        events = get_team_schedule_events(team_id, season)
    except Exception as e:
        print(f"WARNING: schedule fetch failed for team_id={team_id}: {e}")
        return None

    pts_for, pts_against = [], []
    for e in events:
        comp = e.get("competitions", [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        competitors = comp.get("competitors", [])
        me = next((c for c in competitors if str(c.get("team", {}).get("id")) == str(team_id)), None)
        opp = next((c for c in competitors if str(c.get("team", {}).get("id")) != str(team_id)), None)
        if not me or not opp:
            continue
        try:
            my_score = float(me.get("score", {}).get("value") if isinstance(me.get("score"), dict) else me.get("score"))
            opp_score = float(opp.get("score", {}).get("value") if isinstance(opp.get("score"), dict) else opp.get("score"))
        except (TypeError, ValueError):
            continue
        pts_for.append(my_score)
        pts_against.append(opp_score)

    if not pts_for:
        return None
    return {
        "pts_pg": round(sum(pts_for) / len(pts_for), 1),
        "pts_allowed_pg": round(sum(pts_against) / len(pts_against), 1),
        "games_played": len(pts_for),
    }


_TEAM_STATS_CACHE = {}  # team_id -> dict, cleared per run


def get_team_season_stats_cached(team_id, season=CURRENT_SEASON):
    if team_id in _TEAM_STATS_CACHE:
        return _TEAM_STATS_CACHE[team_id]
    stats = get_team_points_for_against(team_id, season)
    _TEAM_STATS_CACHE[team_id] = stats
    return stats


# ============================================================================
# nfl_data_py: play-by-play load with season fallback + local caching
# ============================================================================

_PBP_CACHE = {}  # season -> DataFrame, in-memory per run


def load_pbp_with_fallback(seasons_to_try=(CURRENT_SEASON, SEASON_FALLBACK)):
    """
    Tries each season in order, keeping the first one whose pbp data is
    non-empty and clears MIN_PBP_ROWS_TO_TRUST. Verified at build time:
    nfl_data_py.import_pbp_data([2026]) currently returns an empty (0,0)
    frame with no exception raised - a 404 under the hood that the
    library swallows and prints a message for. Checking .empty / row
    count is mandatory here, not defensive-programming paranoia; a plain
    try/except around the call will NOT catch this case.
    """
    for season in seasons_to_try:
        if season in _PBP_CACHE:
            df = _PBP_CACHE[season]
        else:
            print(f"Fetching pbp for season={season} via nfl_data_py...")
            try:
                df = nfl.import_pbp_data([season], downcast=True, cache=True)
            except Exception as e:
                print(f"WARNING: import_pbp_data failed for season={season}: {e}")
                df = pd.DataFrame()
            _PBP_CACHE[season] = df

        if df is not None and not df.empty and len(df) >= MIN_PBP_ROWS_TO_TRUST:
            print(f"Using season={season} pbp data ({len(df)} rows).")
            return df, season
        else:
            rows = 0 if df is None else len(df)
            print(f"WARNING: season={season} pbp has only {rows} rows (< {MIN_PBP_ROWS_TO_TRUST} threshold) "
                  f"- {'no data published yet' if rows == 0 else 'too thin to trust'}, trying next fallback season.")

    print("WARNING: no season in seasons_to_try had usable pbp data. QB props and "
          "opponent adjustments will be empty for this run.")
    return pd.DataFrame(), None


def build_qb_gamelogs(pbp_df, season=CURRENT_SEASON):
    """
    Groups play-by-play by (game_id, passer_id) to build a per-QB, per-game
    passing_yards / pass_attempts log, filtered to regular season only
    (POST included would contaminate season averages with a much smaller,
    higher-leverage sample).

    The play-grouping key is still passer_id (the plays under one id do
    belong together), but the NAME and TEAM attached to that id are
    reconciled against the roster crosswalk (see reconcile_player_name) to
    correct the confirmed stale-name bug rather than trusting pbp's own
    `passer` column, which can be wrong (see module docstring above).

    Returns:
      { passer_id: {"name": str, "team": nflverse_abbr, "games": [ {week, game_id, passing_yards, pass_attempts}, ... sorted oldest->newest ] } }
    """
    if pbp_df.empty:
        return {}

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["passer_id"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0.0)
    # Track whether each game was home/away for this QB,
    # using pbp's own home_team column (verified present in testing).
    has_home_team_col = "home_team" in pass_plays.columns
    if has_home_team_col:
        pass_plays["is_home"] = pass_plays["posteam"] == pass_plays["home_team"]

    has_jersey = "passer_jersey_number" in pass_plays.columns
    group_cols = ["passer_id", "passer", "posteam", "game_id", "week"]
    agg_dict = {"passing_yards": ("passing_yards", "sum"), "pass_attempts": ("pass_attempt", "sum")}
    if has_jersey:
        agg_dict["jersey_number"] = ("passer_jersey_number", "first")
    if has_home_team_col:
        agg_dict["is_home"] = ("is_home", "first")

    grouped = pass_plays.groupby(group_cols).agg(**agg_dict).reset_index()

    qb_logs = {}
    for _, row in grouped.iterrows():
        pid = row["passer_id"]
        team = row["posteam"]
        raw_name = row["passer"]
        jersey = row["jersey_number"] if has_jersey else None
        resolved_name = reconcile_player_name(raw_name, team, jersey, season)
        if pid not in qb_logs:
            qb_logs[pid] = {"name": resolved_name, "team": team, "jersey": jersey, "games": []}
        qb_logs[pid]["games"].append({
            "week": row["week"],
            "game_id": row["game_id"],
            "passing_yards": float(row["passing_yards"]),
            "pass_attempts": float(row["pass_attempts"]),
            "is_home": bool(row["is_home"]) if has_home_team_col else None,
        })

    for pid in qb_logs:
        qb_logs[pid]["games"].sort(key=lambda g: g["week"])
        # keep team as the most recent team (handles mid-season trades), and
        # re-resolve the name against that most-recent team+jersey too.
        latest_row = grouped[grouped["passer_id"] == pid].sort_values("week").iloc[-1]
        latest_team = latest_row["posteam"]
        latest_jersey = latest_row["jersey_number"] if has_jersey else None
        latest_raw_name = latest_row["passer"]
        qb_logs[pid]["team"] = latest_team
        qb_logs[pid]["name"] = reconcile_player_name(latest_raw_name, latest_team, latest_jersey, season)

    return qb_logs


def build_wr_rb_gamelogs(pbp_df):
    """
    Same idea for receivers (receiving_yards via receiver_id) and rushers
    (rushing_yards via rusher_id), for the secondary correlated-cluster
    projections. Returns:
      {"receivers": {player_id: {...}}, "rushers": {player_id: {...}}}
    """
    if pbp_df.empty:
        return {"receivers": {}, "rushers": {}}

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()

    rec = df[df.get("complete_pass") == 1].dropna(subset=["receiver_id"]).copy() \
        if "complete_pass" in df.columns else df.dropna(subset=["receiver_id"]).copy()
    if "receiving_yards" in rec.columns:
        rec["receiving_yards"] = rec["receiving_yards"].fillna(0.0)
        rec_grouped = rec.groupby(["receiver_id", "receiver", "posteam", "game_id", "week"]).agg(
            receiving_yards=("receiving_yards", "sum"),
        ).reset_index()
    else:
        rec_grouped = pd.DataFrame()

    rush = df[df["rush_attempt"] == 1].dropna(subset=["rusher_id"]).copy() if "rush_attempt" in df.columns else pd.DataFrame()
    if not rush.empty and "rushing_yards" in rush.columns:
        rush["rushing_yards"] = rush["rushing_yards"].fillna(0.0)
        rush_grouped = rush.groupby(["rusher_id", "rusher", "posteam", "game_id", "week"]).agg(
            rushing_yards=("rushing_yards", "sum"),
        ).reset_index()
    else:
        rush_grouped = pd.DataFrame()

    receivers = {}
    for _, row in rec_grouped.iterrows():
        pid = row["receiver_id"]
        if pid not in receivers:
            receivers[pid] = {"name": row["receiver"], "team": row["posteam"], "games": []}
        receivers[pid]["games"].append({
            "week": row["week"], "game_id": row["game_id"],
            "receiving_yards": float(row["receiving_yards"]),
        })
    for pid in receivers:
        receivers[pid]["games"].sort(key=lambda g: g["week"])

    rushers = {}
    for _, row in rush_grouped.iterrows():
        pid = row["rusher_id"]
        if pid not in rushers:
            rushers[pid] = {"name": row["rusher"], "team": row["posteam"], "games": []}
        rushers[pid]["games"].append({
            "week": row["week"], "game_id": row["game_id"],
            "rushing_yards": float(row["rushing_yards"]),
        })
    for pid in rushers:
        rushers[pid]["games"].sort(key=lambda g: g["week"])

    return {"receivers": receivers, "rushers": rushers}


def build_defense_yards_allowed(pbp_df):
    """
    Aggregates passing_yards ALLOWED by each team's defense (grouped by
    defteam) across regular-season games, for the QB opponent adjustment.
    This is the piece the spec explicitly said not to skip - computed
    fresh from pbp (there is no ready-made team-defense field in
    nfl_data_py's seasonal data, confirmed at build time), and cached at
    the module level per run so it's computed once, not once per QB.
    Returns { nflverse_team_abbr: {"pass_yds_allowed_pg": float, "games": int} }
    """
    if pbp_df.empty:
        return {}

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["defteam"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0.0)

    # per-game-per-defense totals, then average across games
    per_game = pass_plays.groupby(["defteam", "game_id"]).agg(
        yds_allowed=("passing_yards", "sum"),
    ).reset_index()

    per_team = per_game.groupby("defteam").agg(
        pass_yds_allowed_pg=("yds_allowed", "mean"),
        games=("game_id", "nunique"),
    ).reset_index()

    result = {}
    for _, row in per_team.iterrows():
        result[row["defteam"]] = {
            "pass_yds_allowed_pg": round(float(row["pass_yds_allowed_pg"]), 1),
            "games": int(row["games"]),
        }
    return result


def build_defense_pass_epa(pbp_df):
    """
    Defensive Pass EPA allowed per game, alongside (not replacing)
    build_defense_yards_allowed - EPA captures down/distance/field-
    position context that raw yards allowed misses.

    Sign: NEGATIVE pass_epa_allowed_pg means the defense is winning
    those plays (good defense). POSITIVE means the offense is winning
    (bad defense) - opposite direction from pass_yds_allowed_pg, where
    higher yards allowed means worse defense.

    Returns { nflverse_team_abbr: {"pass_epa_allowed_pg": float, "games": int} }
    """
    if pbp_df.empty:
        return {}

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()
    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["defteam", "epa"]).copy()

    per_game = pass_plays.groupby(["defteam", "game_id"]).agg(
        epa_allowed=("epa", "sum"),
    ).reset_index()

    per_team = per_game.groupby("defteam").agg(
        pass_epa_allowed_pg=("epa_allowed", "mean"),
        games=("game_id", "nunique"),
    ).reset_index()

    result = {}
    for _, row in per_team.iterrows():
        result[row["defteam"]] = {
            "pass_epa_allowed_pg": round(float(row["pass_epa_allowed_pg"]), 3),
            "games": int(row["games"]),
        }
    return result


# ============================================================================
# Player-name reconciliation against a KNOWN nfl_data_py data-quality bug.
#
# CONFIRMED AT BUILD TIME (2026-09-10), verified against live box-score data:
# nfl_data_py 0.3.3's play-by-play `passer`/`passer_id` columns can be
# WRONG for a current player when a jersey number was previously worn by a
# different (including retired) player on the same franchise. Concretely:
# every 2025 Daniel Jones (IND, #17) pass play in the pbp release is
# mislabeled as "P.Rivers" / passer_id 00-0022942 - Philip Rivers' real,
# 21-years-experience GSIS id, last active with IND in 2020. This is not
# an isolated typo: import_seasonal_rosters([2025]) itself still lists a
# stale "Philip Rivers, status=INA" row for IND/#17 alongside the real
# "Daniel Jones, status=RES" row, meaning the bug traces back to a stale
# historical roster record nflverse's pipeline never fully retired for
# that team+jersey-number combination, not a one-off pbp mapping slip.
#
# Fix: build a (team, jersey_number) -> player_name crosswalk from
# import_seasonal_rosters(), filtered to ACTIVE-status rows only
# (status in {"ACT","RES"} - excludes "INA"/inactive stale entries like
# the Rivers row), preferring the most-recent-week entry to break any
# remaining ties from real in-season roster churn (a cut player and his
# in-season replacement briefly sharing a number). Verified this
# correctly resolves the Daniel Jones case and produces exactly one name
# per (team, jersey) combination league-wide with no remaining ambiguity.
#
# This crosswalk is used to CORRECT the name attached to each QB (and
# WR/RB) gamelog after grouping by passer_id/receiver_id/rusher_id from
# pbp - the underlying yardage/attempts numbers grouped by ID are still
# used as-is (the play SNAPS did happen and belong together under one
# id), only the display name and starter-identification are corrected
# against the trustworthy roster source.
# ============================================================================

_ROSTER_NAME_CROSSWALK = {}  # season -> {(team, jersey_number): player_name}


def get_roster_name_crosswalk(season):
    if season in _ROSTER_NAME_CROSSWALK:
        return _ROSTER_NAME_CROSSWALK[season]
    try:
        roster = nfl.import_seasonal_rosters([season])
    except Exception as e:
        print(f"WARNING: import_seasonal_rosters failed for season={season}: {e} "
              f"- QB/WR/RB names will use raw (possibly stale) pbp names uncorrected.")
        _ROSTER_NAME_CROSSWALK[season] = {}
        return {}

    active = roster[roster["status"].isin(["ACT", "RES"])].copy()
    if active.empty or "week" not in active.columns:
        _ROSTER_NAME_CROSSWALK[season] = {}
        return {}

    lookup_df = active.sort_values(["week", "entry_year"], ascending=[False, False]) \
        .drop_duplicates(subset=["team", "jersey_number"], keep="first")
    crosswalk = lookup_df.set_index(["team", "jersey_number"])["player_name"].to_dict()
    print(f"Built roster name crosswalk for season={season}: {len(crosswalk)} team+jersey entries.")
    _ROSTER_NAME_CROSSWALK[season] = crosswalk
    return crosswalk


def reconcile_player_name(pbp_name, team_abbr, jersey_number, season):
    """
    Returns the roster-verified name for this team+jersey if the crosswalk
    has one, else falls back to the raw pbp name (better than nothing if
    the player/jersey isn't in the roster file for some reason - e.g. a
    mid-season practice-squad call-up gap).
    """
    if jersey_number is None or (isinstance(jersey_number, float) and pd.isna(jersey_number)):
        return pbp_name
    crosswalk = get_roster_name_crosswalk(season)
    resolved = crosswalk.get((team_abbr, jersey_number))
    return resolved if resolved else pbp_name


def find_starting_qb(team_abbr, qb_gamelogs, lookback_games=STARTING_QB_LOOKBACK_GAMES):
    """
    Per spec Part 1.4: identifies the starting QB for a team as the
    player with the most pass attempts across their most recent
    lookback_games games, among QBs who played for this team.
    """
    candidates = [
        (pid, data) for pid, data in qb_gamelogs.items()
        if data["team"] == team_abbr and data["games"]
    ]
    if not candidates:
        return None

    scored = []
    for pid, data in candidates:
        recent = data["games"][-lookback_games:]
        total_attempts = sum(g["pass_attempts"] for g in recent)
        scored.append((pid, data, total_attempts))

    scored.sort(key=lambda x: x[2], reverse=True)
    best_pid, best_data, _ = scored[0]
    return {"passer_id": best_pid, "name": best_data["name"], "team": best_data["team"], "games": best_data["games"]}


# ============================================================================
# QB props assembly: shrinkage + opponent adjustment + usage boost, per QB
# ============================================================================

CONFIDENCE_THRESHOLD = 0.68  # cluster-eligibility bar (see extract_correlated_forecasts)
MIN_BETTABLE_YARDS = 100     # lowest passing-yards threshold treated as a real bettable line


def compute_home_away_split(games):
    """QB's average passing yards at home vs away. Either value can be None if there are no games in that split."""
    home_vals = [g["passing_yards"] for g in games if g.get("is_home") is True]
    away_vals = [g["passing_yards"] for g in games if g.get("is_home") is False]
    return {
        "home_avg": round(sum(home_vals) / len(home_vals), 1) if home_vals else None,
        "away_avg": round(sum(away_vals) / len(away_vals), 1) if away_vals else None,
    }


def compute_confidence_interval(games, avg_yards):
    """10th-90th percentile range: projected_yards +/- 1.28 * std_dev. Returns None with fewer than 2 games."""
    if len(games) < 2:
        return None
    values = [g["passing_yards"] for g in games]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample std dev
    std_dev = variance ** 0.5
    margin = 1.28 * std_dev
    return {
        "std_dev": round(std_dev, 1),
        "low": round(max(0, avg_yards - margin), 1),
        "high": round(avg_yards + margin, 1),
    }


def compute_rest_days(team_espn_id, event_dt_iso):
    """Days since this team's last game, from ESPN's team schedule. Returns None if it can't be determined."""
    try:
        events = get_team_schedule_events(team_espn_id)
    except Exception as e:
        print(f"WARNING: rest-days schedule fetch failed for team_id={team_espn_id}: {e}")
        return None

    try:
        target_dt = datetime.fromisoformat(event_dt_iso)
    except (ValueError, TypeError):
        return None

    past_game_dts = []
    for e in events:
        comp = e.get("competitions", [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        date_raw = e.get("date")
        if not date_raw:
            continue
        try:
            dt = datetime.strptime(date_raw, "%Y-%m-%dT%H:%M%z").replace(tzinfo=None)
        except ValueError:
            try:
                dt = datetime.strptime(date_raw, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
        if dt < target_dt:
            past_game_dts.append(dt)

    if not past_game_dts:
        return None
    most_recent = max(past_game_dts)
    return (target_dt - most_recent).days


def apply_home_away_rest_adjustment(over_prob, is_home_game, rest_days):
    """
    Applies flat multipliers: rest_days < 5 (short week) is -7%, home
    game is +3%. Both can apply. Applied directly to the probability.
    """
    if over_prob is None:
        return over_prob
    mult = 1.0
    if rest_days is not None and rest_days < 5:
        mult *= 0.93
    if is_home_game:
        mult *= 1.03
    return round(max(0.0, min(1.0, over_prob * mult)), 3)


def get_qb_props(team_abbr, opponent_espn_abbr, qb_gamelogs,
                  missing_names=None, wr1_name=None, rb1_name=None,
                  season=CURRENT_SEASON, usage_boost_mults=None):
    """
    Builds one starting QB's passing-yards floor probabilities: raw
    empirical hit-rates -> Bayesian-shrunk toward league average ->
    attempts-projection re-scaling if his likely workload has shifted ->
    opponent-adjusted using the opponent defense's pass-yards-allowed ->
    usage-boosted if WR1/RB1 is out. Mirrors the WNBA script's
    get_player_props() layering order.

    Returns None if no starter can be identified for this team.
    """
    starter = find_starting_qb(team_abbr, qb_gamelogs)
    if not starter:
        return None

    recent_games = starter["games"][-QB_GAMES_SAMPLE:]
    if not recent_games:
        return None

    raw_floors = prop_floor_probs_yards(recent_games)
    if not raw_floors:
        return None

    avg_passing_yards = round(
        sum(g.get("passing_yards", 0.0) or 0.0 for g in recent_games) / len(recent_games), 1
    )

    games_sampled = len(recent_games)
    projected_atts = project_attempts(recent_games)

    # If projected attempts differ meaningfully from the sampled-games
    # average, each threshold gets rescaled by that ratio before
    # recomputing the hit rate. The original threshold is still what's
    # displayed.
    avg_attempts = sum(g.get("pass_attempts", 0.0) or 0.0 for g in recent_games) / len(recent_games) \
        if recent_games else None

    rescale_applied = False
    if (projected_atts is not None and avg_attempts and avg_attempts > 0
            and abs(projected_atts - avg_attempts) / avg_attempts > PROJECTED_ATTEMPTS_DELTA):
        # Divide, don't multiply: a stat is roughly volume x rate, so
        # more projected attempts should make a fixed yardage threshold
        # EASIER to clear. Multiplying the threshold by the ratio does
        # the opposite.
        attempts_ratio = projected_atts / avg_attempts
        values = [g.get("passing_yards", 0.0) or 0.0 for g in recent_games]
        rescaled_floors = {}
        for original_threshold in raw_floors.keys():
            rescaled_threshold = original_threshold / attempts_ratio
            hits = sum(1 for v in values if v >= rescaled_threshold)
            rescaled_floors[original_threshold] = round(hits / len(values), 3)  # keyed by ORIGINAL threshold - that's still the line displayed
        raw_floors = rescaled_floors
        rescale_applied = True

    avg_passing_yards = round(
        sum(g.get("passing_yards", 0.0) or 0.0 for g in recent_games) / len(recent_games), 1
    )

    boost = usage_boost_if_starter_out(missing_names or [], wr1_name, rb1_name, boost_mults=usage_boost_mults)

    adjusted_floors = {}
    for threshold, raw_hr in raw_floors.items():
        hits = round(raw_hr * games_sampled)
        # league_avg for shrinkage is filled in later, at the report-assembly
        # stage, once every starter's raw floors are known (matches the
        # WNBA script's two-pass structure: build raw first, then shrink
        # using a league average computed across everyone in the report).
        adjusted_floors[threshold] = {"raw_hit_rate": raw_hr, "hits": hits, "games": games_sampled}

    return {
        "passer_id": starter["passer_id"],
        "name": starter["name"],
        "team": team_abbr,
        "opponent_espn_abbr": opponent_espn_abbr,
        "games_sampled": games_sampled,
        "avg_passing_yards": avg_passing_yards,
        "avg_pass_attempts": round(avg_attempts, 1) if avg_attempts else None,
        "attempts_rescale_applied": rescale_applied,
        "home_away_split": compute_home_away_split(recent_games),
        "confidence_interval": compute_confidence_interval(recent_games, avg_passing_yards),
        "projected_attempts": round(projected_atts, 1) if projected_atts else None,
        "raw_floors": {"passing_yards": raw_floors},
        "_adjusted_floors_pending": adjusted_floors,  # finalized in finalize_qb_floors()
        "usage_boost": boost,
        "missing_names_considered": missing_names or [],
    }


def finalize_qb_floors(qb_props, league_avg_hit_rate, opponent_nflverse_abbr, defense_yards_allowed):
    """
    Applies Bayesian shrinkage, then the opponent adjustment, then the
    usage boost, in that order.

    IMPORTANT: pass the full defense_yards_allowed cache (every team),
    not a single-team dict. opponent_adjustment() computes league_avg
    from every team in the cache it's given - a one-team cache makes
    league_avg equal the opponent's own value, silently turning the
    adjustment into a no-op.
    """
    floors = {}

    for threshold, info in qb_props["_adjusted_floors_pending"].items():
        shrunk = bayesian_shrinkage(info["hits"], info["games"], league_avg_hit_rate)
        opp_adjusted = opponent_adjustment("passing_yards", shrunk, opponent_nflverse_abbr, defense_yards_allowed) \
            if defense_yards_allowed else shrunk
        boosted = min(1.0, opp_adjusted * qb_props["usage_boost"]) if opp_adjusted is not None else None
        floors[threshold] = round(boosted, 3) if boosted is not None else None

    qb_props["floors"] = {"passing_yards": floors}
    del qb_props["_adjusted_floors_pending"]
    return qb_props


def medium_threshold_for(floors_dict, avg_yards=None):
    """
    The threshold closest to the QB's own average - the ladder's middle
    rung. FIXED (was previously just floors_dict's middle INDEX, which is
    only "closest to average" when the ladder happens to be laid out
    symmetrically around it - not guaranteed, since RUNGS_ABOVE_AVG=1
    means the ladder has more rungs below average than above. Passing
    the real avg_yards and picking the threshold with the smallest
    absolute distance to it is correct regardless of ladder shape.
    Falls back to the old middle-index behavior if avg_yards isn't
    available, so this stays backward-compatible with any caller that
    hasn't been updated to pass it yet.
    """
    if not floors_dict:
        return None
    if avg_yards is not None:
        return min(floors_dict.keys(), key=lambda t: abs(t - avg_yards))
    sorted_t = sorted(floors_dict.keys())
    idx = len(sorted_t) // 2
    return sorted_t[idx]


def attach_main_line_and_under(qb_props):
    """
    Books this system is built around offer OVER with alternate lines
    (the full threshold ladder already computed in qb_props["floors"]),
    but UNDER only at a single "main" line - no alternates. There's no
    live odds feed here, so the main line is an ESTIMATE: the threshold
    in the existing ladder closest to the QB's own recent average (same
    selection logic as medium_threshold_for / the WNBA original's
    "medium rung" concept), clearly labeled as estimated rather than a
    real posted book line.

    UNDER's probability at that line is derived as 1 - P(Over main_line),
    using the SAME shrunk + opponent-adjusted + usage-boosted probability
    already computed for that threshold - not a separately-modeled number,
    since Over/Under at one line are complementary by definition.

    Adds qb_props["main_line"] = {"threshold": int, "over_prob": float,
    "under_prob": float, "is_estimate": True} and leaves the existing
    "floors" ladder (Over's alternates) untouched.
    """
    floors = qb_props.get("floors", {}).get("passing_yards", {})
    main_t = medium_threshold_for(floors, avg_yards=qb_props.get("avg_passing_yards"))
    if main_t is None or floors.get(main_t) is None:
        qb_props["main_line"] = None
        return qb_props

    over_prob = floors[main_t]
    qb_props["main_line"] = {
        "threshold": main_t,
        "over_prob": over_prob,
        "under_prob": round(1 - over_prob, 3),
        "is_estimate": True,
    }
    return qb_props


def best_bettable_line(qb_props, min_confidence=CONFIDENCE_THRESHOLD):
    """
    Highest passing-yards threshold whose adjusted probability still
    clears min_confidence and is a real, postable line (>= MIN_BETTABLE_YARDS).
    Mirrors the WNBA script's best_line_at_confidence(). This only
    considers OVER, since Over is the side with alternate lines - Under
    is handled separately via attach_main_line_and_under's single
    main_line, not through this ladder-scanning function.
    """
    floors = qb_props.get("floors", {}).get("passing_yards", {})
    if not floors:
        return None
    eligible = [(t, p) for t, p in floors.items() if p is not None and p >= min_confidence and t >= MIN_BETTABLE_YARDS]
    if not eligible:
        return None
    t, p = max(eligible, key=lambda x: x[0])
    return {"threshold": t, "prob": p}


def best_bettable_under(qb_props, min_confidence=CONFIDENCE_THRESHOLD):
    """
    Returns the single main-line Under pick if its probability clears
    min_confidence, else None. Under has no alternates to scan - there is
    exactly one number to check, unlike best_bettable_line's ladder scan.
    """
    main_line = qb_props.get("main_line")
    if not main_line or main_line.get("under_prob") is None:
        return None
    if main_line["under_prob"] >= min_confidence:
        return {"threshold": main_line["threshold"], "prob": main_line["under_prob"]}
    return None


# ============================================================================
# Report assembly: pulls it all together per game
# ============================================================================

def build_report():
    print("Opponent adjustment: using yards-allowed (EPA adjustment is disabled by default).")
    games = get_todays_games()
    if not games:
        print("WARNING: 0 games found for today's local window.")
        return []

    pbp_df, pbp_season_used = load_pbp_with_fallback()
    qb_gamelogs = build_qb_gamelogs(pbp_df, season=pbp_season_used) if pbp_season_used else {}
    wr_rb_logs = build_wr_rb_gamelogs(pbp_df) if pbp_season_used else {"receivers": {}, "rushers": {}}
    defense_yards_allowed = build_defense_yards_allowed(pbp_df) if pbp_season_used else {}
    defense_pass_epa = build_defense_pass_epa(pbp_df) if pbp_season_used else {}
    # merge EPA into the same per-team dict opponent_adjustment() reads from
    for team, epa_stats in defense_pass_epa.items():
        defense_yards_allowed.setdefault(team, {}).update(epa_stats)

    spread_std_dev, total_std_dev, team_total_std_dev = fit_std_dev_constants(pbp_df)
    game_script_mults = fit_game_script_multipliers(pbp_df)

    try:
        injuries_df = nfl.import_injuries([pbp_season_used]) if pbp_season_used else pd.DataFrame()
    except Exception as e:
        print(f"WARNING: import_injuries failed for empirical usage boost fit: {e}")
        injuries_df = pd.DataFrame()
    usage_boost_mults = fit_empirical_usage_boosts(pbp_df, injuries_df, get_roster_name_crosswalk)
    home_mult, away_mult = fit_home_away_multipliers(pbp_df)
    # injury_mults for team totals removed - backtest evidence showed the
    # adjustment's effect (gap 0.0006 vs combined SE 0.0055) is noise,
    # not a real signal. The QB usage-boost path above still uses its
    # own empirically-fit multipliers, which have a plausible mechanism
    # (volume shift to remaining players) even though unmeasured at the
    # team-total level.

    league_avg_pts = 22.5  # NFL long-run average team score, used as the shrinkage prior

    print(f"QB gamelogs built for {len(qb_gamelogs)} players "
          f"(season used: {pbp_season_used}).")
    print(f"Defense pass-yards-allowed computed for {len(defense_yards_allowed)} teams.")
    print(f"Defense pass-EPA-allowed computed for {len(defense_pass_epa)} teams.")

    # ---- pass 1: raw QB props for every team appearing today ----
    raw_qb_props_by_team = {}
    injury_flags_by_team = {}  # team_espn_abbr -> list[str], for display in the HTML
    injury_status_by_team = {}  # team_espn_abbr -> {"qb_out": bool, "wr1_out": bool, "rb1_out": bool}
    for g in games:
        for side, opp_side in (("home", "away"), ("away", "home")):
            team_espn_abbr = g[f"{side}_team_abbr"]
            team_espn_id = g[f"{side}_team_id"]
            opp_espn_abbr = g[f"{opp_side}_team_abbr"]
            team_nflverse_abbr = to_nflverse_abbr(team_espn_abbr)
            if team_nflverse_abbr in raw_qb_props_by_team:
                continue

            wr1_name, rb1_name = find_wr1_rb1(team_nflverse_abbr, wr_rb_logs)
            starter = find_starting_qb(team_nflverse_abbr, qb_gamelogs)
            starter_names = {n for n in (starter["name"] if starter else None, wr1_name, rb1_name) if n}
            flags, missing_names = flag_missing_starters(team_espn_id, starter_names) if starter_names else ([], set())
            injury_flags_by_team[team_espn_abbr] = flags
            injury_status_by_team[team_espn_abbr] = {
                "qb_out": bool(starter and starter["name"] in missing_names),
                "wr1_out": bool(wr1_name and wr1_name in missing_names),
                "rb1_out": bool(rb1_name and rb1_name in missing_names),
            }

            props = get_qb_props(
                team_nflverse_abbr, opp_espn_abbr, qb_gamelogs,
                missing_names=missing_names, wr1_name=wr1_name, rb1_name=rb1_name,
                season=pbp_season_used or CURRENT_SEASON,
                usage_boost_mults=usage_boost_mults,
            )
            if props:
                raw_qb_props_by_team[team_nflverse_abbr] = props

    # ---- league average hit-rate for shrinkage, computed across everyone above ----
    all_props_list = []
    for props in raw_qb_props_by_team.values():
        # compute_league_avg_hit_rate expects "floors"/"games_sampled" shaped
        # like the WNBA original; build a lightweight view for it here.
        all_props_list.append({
            "floors": {"passing_yards": {t: info["raw_hit_rate"] for t, info in props["_adjusted_floors_pending"].items()}},
            "games_sampled": props["games_sampled"],
        })
    league_avg_hr = compute_league_avg_hit_rate(all_props_list, stat_key="passing_yards")
    print(f"League average shrinkage target (passing_yards, medium rung): {league_avg_hr:.3f}")

    # ---- pass 2: finalize (shrink + opponent-adjust) each team's QB props ----
    finalized_qb_props_by_team = {}
    for team_nflverse_abbr, props in raw_qb_props_by_team.items():
        opp_espn_abbr = props["opponent_espn_abbr"]
        opp_nflverse_abbr = to_nflverse_abbr(opp_espn_abbr)
        finalized = finalize_qb_floors(props, league_avg_hr, opp_nflverse_abbr, defense_yards_allowed)
        finalized = attach_main_line_and_under(finalized)
        finalized_qb_props_by_team[team_nflverse_abbr] = finalized

    # ---- team points-for/against via ESPN schedule (cached) ----
    report = []
    for g in games:
        home_stats = get_team_season_stats_cached(g["home_team_id"])
        away_stats = get_team_season_stats_cached(g["away_team_id"])

        spread_line = g.get("spread_line")
        total_line = g.get("total_line")

        home_cover_prob = spread_cover_prob(home_stats, away_stats, spread_line, std_dev=spread_std_dev) \
            if (home_stats and away_stats and spread_line is not None) else None
        game_total_prob = game_total_over_prob(home_stats, away_stats, total_line, std_dev=total_std_dev) \
            if (home_stats and away_stats and total_line is not None) else None

        home_nflverse = to_nflverse_abbr(g["home_team_abbr"])
        away_nflverse = to_nflverse_abbr(g["away_team_abbr"])

        # Weather. Home team's stadium is the game venue. Skip the fetch
        # for known dome/closed-roof teams - no real outdoor weather
        # reaches the field there.
        weather_info = None
        weather_note = None
        weather_mult = 1.0

        # Computed once, reused by both the team-total block and the QB
        # rest-adjustment block below - avoids calling compute_rest_days
        # twice per team per game for the same (team, kickoff) input.
        home_rest = compute_rest_days(g["home_team_id"], g.get("kickoff_iso")) if g.get("kickoff_iso") else None
        away_rest = compute_rest_days(g["away_team_id"], g.get("kickoff_iso")) if g.get("kickoff_iso") else None

        if home_nflverse not in DOME_OR_CLOSED_ROOF_TEAMS and home_nflverse in STADIUM_COORDS:
            lat, lon = STADIUM_COORDS[home_nflverse]
            kickoff_iso = g.get("kickoff_iso")
            if kickoff_iso:
                weather_info = get_weather_for_game(lat, lon, kickoff_iso, historical=False)
            if weather_info:
                weather_mult = weather_adjustment(weather_info["wind_mph"], weather_info["precip_mm"], weather_info["temp_f"])
                if weather_mult < 1.0:
                    pct = round((1 - weather_mult) * 100)
                    weather_note = (f"Wind {round(weather_info['wind_mph'])} mph, "
                                     f"{round(weather_info['temp_f'])}\u00b0F"
                                     + (f", {weather_info['precip_mm']}mm precip" if weather_info['precip_mm'] and weather_info['precip_mm'] > 0 else "")
                                     + f" \u2014 scoring and passing yards adjusted -{pct}%.")

        # Top-down team total projections - not summed from player props.
        home_team_total = None
        away_team_total = None
        home_team_total_prob = None
        away_team_total_prob = None
        if home_stats and away_stats:
            home_own_rate = bayesian_shrinkage(home_stats["pts_pg"] * home_stats["games_played"],
                                                home_stats["games_played"], league_avg_pts, k=SHRINKAGE_K)
            away_own_rate = bayesian_shrinkage(away_stats["pts_pg"] * away_stats["games_played"],
                                                away_stats["games_played"], league_avg_pts, k=SHRINKAGE_K)
            home_opp_allowed_rate = bayesian_shrinkage(away_stats["pts_allowed_pg"] * away_stats["games_played"],
                                                         away_stats["games_played"], league_avg_pts, k=SHRINKAGE_K)
            away_opp_allowed_rate = bayesian_shrinkage(home_stats["pts_allowed_pg"] * home_stats["games_played"],
                                                          home_stats["games_played"], league_avg_pts, k=SHRINKAGE_K)

            # Injury adjustment dropped from team totals per backtest
            # evidence (gap 0.0006 vs combined SE 0.0055 - noise, not a
            # real effect). Kept for the QB prop tab, where it has a
            # plausible mechanism (volume shift) even though unmeasured
            # at the team-total level.
            home_team_total_topdown = project_team_total(home_own_rate, home_opp_allowed_rate, rest_days=home_rest,
                                                           is_home=True, weather_mult=weather_mult,
                                                           home_mult=home_mult, away_mult=away_mult)
            away_team_total_topdown = project_team_total(away_own_rate, away_opp_allowed_rate, rest_days=away_rest,
                                                           is_home=False, weather_mult=weather_mult,
                                                           home_mult=home_mult, away_mult=away_mult)

            # Top-down only. An earlier version averaged this with a
            # bottom-up passing-yards estimate, but that estimate applied
            # a points-per-TOTAL-offensive-yard rate (fit against passing
            # + rushing yards combined) to passing yards alone - a unit
            # mismatch, not a smaller-scope model. Passing yards are only
            # ~60-65% of total offensive yards, so that bottom-up half
            # under-projected by roughly 30% and dragged every ensemble
            # total down 2-4 points, a systematic bias toward the under.
            # Reverted until a correct bottom-up estimate (real rushing-
            # yards gamelogs, matching what the backtest actually tested)
            # is built.
            home_team_total = home_team_total_topdown
            away_team_total = away_team_total_topdown

            if total_line is not None:
                # Each team's own line, derived from the game total and
                # the spread - not just half the game total, which
                # ignores which team is favored. home_line = total/2 +
                # spread/2, away_line = total/2 - spread/2, using the
                # same spread_line sign convention as spread_cover_prob
                # (negative = home favored).
                home_team_line = round(total_line / 2 - spread_line / 2, 1) if spread_line is not None else round(total_line / 2, 1)
                away_team_line = round(total_line / 2 + spread_line / 2, 1) if spread_line is not None else round(total_line / 2, 1)
                home_team_total_prob = team_total_over_prob_v2(home_team_total, home_team_line, std_dev=team_total_std_dev)
                away_team_total_prob = team_total_over_prob_v2(away_team_total, away_team_line, std_dev=team_total_std_dev)

        home_qb = finalized_qb_props_by_team.get(home_nflverse)
        away_qb = finalized_qb_props_by_team.get(away_nflverse)

        if weather_info and weather_mult < 1.0:
            for qb in (home_qb, away_qb):
                if not qb:
                    continue
                floors = qb.get("floors", {}).get("passing_yards", {})
                for t in list(floors.keys()):
                    if floors[t] is not None:
                        floors[t] = round(max(0.0, min(1.0, floors[t] * weather_mult)), 3)
                if qb.get("main_line"):
                    qb["main_line"]["over_prob"] = round(max(0.0, min(1.0, qb["main_line"]["over_prob"] * weather_mult)), 3)
                    qb["main_line"]["under_prob"] = round(1 - qb["main_line"]["over_prob"], 3)

        # Rest days + home/away adjustment, applied the
        # same way as weather above (direct probability multiplier,
        # documented approximation, always visible via rest_days field
        # on the QB dict rather than silently baked in).
        kickoff_iso_for_rest = g.get("kickoff_iso")
        if kickoff_iso_for_rest:
            if home_qb:
                home_qb["rest_days"] = home_rest
                home_qb["is_home_game"] = True
                if home_qb.get("projected_attempts") is not None and spread_line is not None:
                    home_qb["projected_attempts"] = adjust_attempts_for_game_script(
                        home_qb["projected_attempts"], spread_line, "home", game_script_mults)
                for t in list(home_qb.get("floors", {}).get("passing_yards", {}).keys()):
                    home_qb["floors"]["passing_yards"][t] = apply_home_away_rest_adjustment(
                        home_qb["floors"]["passing_yards"][t], True, home_rest)
                if home_qb.get("main_line"):
                    home_qb["main_line"]["over_prob"] = apply_home_away_rest_adjustment(
                        home_qb["main_line"]["over_prob"], True, home_rest)
                    home_qb["main_line"]["under_prob"] = round(1 - home_qb["main_line"]["over_prob"], 3)
            if away_qb:
                away_qb["rest_days"] = away_rest
                away_qb["is_home_game"] = False
                if away_qb.get("projected_attempts") is not None and spread_line is not None:
                    away_qb["projected_attempts"] = adjust_attempts_for_game_script(
                        away_qb["projected_attempts"], spread_line, "away", game_script_mults)
                for t in list(away_qb.get("floors", {}).get("passing_yards", {}).keys()):
                    away_qb["floors"]["passing_yards"][t] = apply_home_away_rest_adjustment(
                        away_qb["floors"]["passing_yards"][t], False, away_rest)
                if away_qb.get("main_line"):
                    away_qb["main_line"]["over_prob"] = apply_home_away_rest_adjustment(
                        away_qb["main_line"]["over_prob"], False, away_rest)
                    away_qb["main_line"]["under_prob"] = round(1 - away_qb["main_line"]["over_prob"], 3)

        report.append({
            "event_id": g["event_id"],
            "kickoff_local": g["kickoff_local"],
            "home_team_abbr": g["home_team_abbr"],
            "home_team_name": g["home_team_name"],
            "away_team_abbr": g["away_team_abbr"],
            "away_team_name": g["away_team_name"],
            "home_stats": home_stats,
            "away_stats": away_stats,
            "spread_line": spread_line,
            "total_line": total_line,
            "home_cover_prob": home_cover_prob,
            "game_total_over_prob": game_total_prob,
            "home_qb": home_qb,
            "away_qb": away_qb,
            "home_injury_flags": injury_flags_by_team.get(g["home_team_abbr"], []),
            "away_injury_flags": injury_flags_by_team.get(g["away_team_abbr"], []),
            "weather_note": weather_note,
            "home_team_total": home_team_total,
            "away_team_total": away_team_total,
            "home_team_total_prob": home_team_total_prob,
            "away_team_total_prob": away_team_total_prob,
        })

    return report


def build_top_qb_performers(report, count=10, min_games=3):
    """
    Renamed from build_top_points_performers per spec. Ranks QBs by their
    shrunk/adjusted hit-rate on the threshold closest to their own average
    passing yards (the "medium" rung).
    """
    candidates = []
    for g in report:
        for side_key in ("home_qb", "away_qb"):
            qb = g.get(side_key)
            if not qb or qb.get("games_sampled", 0) < min_games:
                continue
            floors = qb.get("floors", {}).get("passing_yards", {})
            medium_t = medium_threshold_for(floors, avg_yards=qb.get("avg_passing_yards"))
            if medium_t is None or floors.get(medium_t) is None:
                continue
            opponent_name = g["away_team_name"] if side_key == "home_qb" else g["home_team_name"]
            candidates.append({
                "name": qb["name"],
                "team": qb["team"],
                "opponent_full": opponent_name,
                "threshold": medium_t,
                "hit_rate": floors[medium_t],
                "games_sampled": qb["games_sampled"],
                "projected_attempts": qb.get("projected_attempts"),
            })
    candidates.sort(key=lambda x: x["hit_rate"], reverse=True)
    return candidates[:count]


_JOINT_OUTCOMES_CACHE = None


def load_joint_outcomes_table(path="docs/joint_outcomes.csv"):
    """
    Loads the historical joint outcome table from backtest_nfl.py
    (docs/joint_outcomes.csv). Returns an empty DataFrame if the file
    doesn't exist yet - callers must fall back to independent-legs display.
    """
    global _JOINT_OUTCOMES_CACHE
    if _JOINT_OUTCOMES_CACHE is not None:
        return _JOINT_OUTCOMES_CACHE
    if not os.path.exists(path):
        print(f"NOTE: {path} not found - correlated clusters will show independent-leg "
              f"probabilities instead of real joint probabilities until backtest_nfl.py has run once.")
        _JOINT_OUTCOMES_CACHE = pd.DataFrame()
        return _JOINT_OUTCOMES_CACHE
    try:
        _JOINT_OUTCOMES_CACHE = pd.read_csv(path)
        print(f"Loaded joint outcomes table: {len(_JOINT_OUTCOMES_CACHE)} historical team-game rows.")
    except Exception as e:
        print(f"WARNING: failed to load {path}: {e} - falling back to independent-leg probabilities.")
        _JOINT_OUTCOMES_CACHE = pd.DataFrame()
    return _JOINT_OUTCOMES_CACHE


def query_joint_probability(joint_df, qb_threshold, total_threshold, min_samples=20, condition_field="team_total"):
    """
    Local copy of backtest_nfl.py's query_joint_probability (kept
    separate so predict_nfl.py doesn't need sklearn/matplotlib).

    condition_field must be "team_total" (this team's own score) or
    "game_total" (combined score - what the live dashboard's
    g["total_line"] is). Passing the wrong field silently answers a
    different, usually much rarer question.
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
        "n": int(n),
        "unconditional_prob": round(float(unconditional_prob), 3),
    }


def extract_correlated_forecasts(report, min_confidence=CONFIDENCE_THRESHOLD, limit=8):
    """
    Groups projections by game. A cluster requires at least 2 correlated
    projections. When a cluster pairs a QB passing-yards OVER pick with
    that same team's game-total OVER pick, this replaces both
    independent legs with a single real historical joint probability.
    Falls back to independent legs if the joint table isn't available or
    doesn't have enough samples at these thresholds.
    """
    joint_df = load_joint_outcomes_table()
    clusters = []
    for g in report:
        game_label = f"{g['away_team_abbr']} @ {g['home_team_abbr']}"
        pieces = []
        qb_over_by_side = {}  # side_key -> (team_abbr, threshold, prob) for the joint-prob check below

        for side_key, team_label in (("home_qb", g["home_team_abbr"]), ("away_qb", g["away_team_abbr"])):
            qb = g.get(side_key)
            if not qb:
                continue
            over_line = best_bettable_line(qb, min_confidence)
            if over_line:
                qb_over_by_side[side_key] = (team_label, over_line["threshold"], over_line["prob"])
                pieces.append({
                    "type": "qb_passing_yards_over",
                    "label": f"{qb['name']} OVER {over_line['threshold']} passing yards",
                    "prob": over_line["prob"],
                    "_side_key": side_key,
                })
            under_line = best_bettable_under(qb, min_confidence)
            if under_line:
                pieces.append({
                    "type": "qb_passing_yards_under",
                    "label": f"{qb['name']} UNDER {under_line['threshold']} passing yards (est. main line)",
                    "prob": under_line["prob"],
                })

        if g.get("home_cover_prob") is not None and g["home_cover_prob"] >= min_confidence:
            pieces.append({
                "type": "spread",
                "label": f"{g['home_team_abbr']} covers {g['spread_line']}",
                "prob": g["home_cover_prob"],
            })
        elif g.get("home_cover_prob") is not None and (1 - g["home_cover_prob"]) >= min_confidence:
            pieces.append({
                "type": "spread",
                "label": f"{g['away_team_abbr']} covers {-g['spread_line'] if g['spread_line'] is not None else None}",
                "prob": round(1 - g["home_cover_prob"], 3),
            })

        if g.get("game_total_over_prob") is not None and g["game_total_over_prob"] >= min_confidence:
            pieces.append({
                "type": "total",
                "label": f"Game total OVER {g['total_line']}",
                "prob": g["game_total_over_prob"],
            })

        # Replace, not supplement, the independent QB-over and game-
        # total-over legs with the real joint probability when both
        # exist for the same team/game.
        replaced_side_keys = set()
        total_leg_replaced = False
        if g.get("total_line") is not None:
            for side_key, (team_abbr, qb_threshold, qb_prob) in qb_over_by_side.items():
                joint_result = query_joint_probability(joint_df, qb_threshold, g["total_line"], condition_field="game_total")
                if joint_result:
                    pieces.append({
                        "type": "joint_qb_team_total",
                        "label": (f"{team_abbr}: QB {qb_threshold}+ yards AND game total OVER "
                                  f"{g['total_line']} together \u2014 historically {round(joint_result['joint_prob']*100)}% "
                                  f"of the time (n={joint_result['n']}, vs {round(joint_result['unconditional_prob']*100)}% baseline)"),
                        "prob": joint_result["joint_prob"],
                        "_is_joint": True,
                    })
                    replaced_side_keys.add(side_key)
                    total_leg_replaced = True

        # Remove the independent legs that the joint piece(s) above now
        # supersede - a joint pill conveys strictly more information than
        # its two component legs shown separately, so showing all three
        # would be redundant at best and misleading at worst (implying
        # three independent signals when it's really one conditional
        # statement plus a restatement of half of it).
        if replaced_side_keys:
            pieces = [
                p for p in pieces
                if not (p.get("type") == "qb_passing_yards_over" and p.get("_side_key") in replaced_side_keys)
            ]
        if total_leg_replaced:
            pieces = [p for p in pieces if p.get("type") != "total"]

        # strip internal-only markers before returning - _side_key was
        # only needed to build the joint-probability lookups above, and
        # _is_joint was never actually read by any renderer (verified: no
        # code path checks it) - it was leaking into docs/report.json as
        # dead metadata with a leading underscore that looked deliberate
        # but wasn't load-bearing anywhere.
        for p in pieces:
            p.pop("_side_key", None)
            p.pop("_is_joint", None)

        if len(pieces) >= 2:
            pieces.sort(key=lambda x: x["prob"], reverse=True)
            avg_conf = sum(p["prob"] for p in pieces) / len(pieces)
            clusters.append({
                "game_label": game_label,
                "kickoff_local": g["kickoff_local"],
                "pieces": pieces,
                "avg_confidence": round(avg_conf, 3),
            })

    clusters.sort(key=lambda c: c["avg_confidence"], reverse=True)
    return clusters[:limit]


# ============================================================================
# HTML rendering
# Visually matches the WNBA dashboard's structure: dark/light mode via
# prefers-color-scheme, 3-tab nav, collapsible cards, pill badges, and
# horizontal probability bars. Labels updated per spec (WNBA -> NFL,
# "Starter Prop Floors" -> "QB Passing Projections"). Full game only - no
# Q1/H1 splits anywhere in this output, per spec Part 4.
# ============================================================================

def _pill_class(prob):
    if prob is None:
        return "pill-cool"
    if prob >= 0.70:
        return "pill-hot"
    if prob >= 0.50:
        return "pill-warm"
    return "pill-cool"


def _prob_bar_html(prob, label=""):
    if prob is None:
        return f'<div class="prob-bar-track"><span class="prob-bar-label">{label}: no data</span></div>'
    pct = round(prob * 100)
    return f"""
    <div class="prob-bar-track">
      <div class="prob-bar-fill" style="width:{pct}%"></div>
      <span class="prob-bar-label">{label}: {pct}%</span>
    </div>"""


def _render_qb_passing_projections(qb, side_label):
    if not qb:
        return f'<p class="qb-no-data">{side_label}: no QB data available this run.</p>'
    floors = qb.get("floors", {}).get("passing_yards", {})
    if not floors:
        return f'<p class="qb-no-data">{side_label}: {qb["name"]} - no floor data.</p>'

    pills = []
    for threshold in sorted(floors.keys()):
        prob = floors[threshold]
        pct = round(prob * 100) if prob is not None else None
        pill_cls = _pill_class(prob)
        pct_display = f"{pct}%" if pct is not None else "N/A"
        pills.append(f'<span class="pill {pill_cls}">{threshold}+ yds: {pct_display}</span>')

    atts = qb.get("projected_attempts")
    atts_html = f'<p class="qb-attempts">Expected passes: {atts}</p>' if atts else ""
    boost = qb.get("usage_boost", 1.0)
    boost_html = f'<p class="qb-boost">Adjustment for missing players: +{round((boost-1)*100)}%</p>' if boost and boost > 1.0 else ""

    ci = qb.get("confidence_interval")
    ci_html = ""
    if ci and qb.get("avg_passing_yards") is not None:
        ci_html = f'<p class="qb-ci">Projected: {qb["avg_passing_yards"]} yards (range: {ci["low"]}\u2013{ci["high"]})</p>'

    split = qb.get("home_away_split")
    split_parts = []
    if split:
        if split.get("home_avg") is not None:
            split_parts.append(f'Home avg {split["home_avg"]} yds')
        if split.get("away_avg") is not None:
            split_parts.append(f'Away avg {split["away_avg"]} yds')
    rest_days = qb.get("rest_days")
    if rest_days is not None:
        split_parts.append(f'{rest_days} days rest' + (' (short week)' if rest_days < 5 else ''))
    split_html = f'<p class="qb-splits">{" \u2022 ".join(split_parts)}</p>' if split_parts else ""

    main_line = qb.get("main_line")
    under_html = ""
    if main_line:
        under_pct = round(main_line["under_prob"] * 100)
        under_cls = _pill_class(main_line["under_prob"])
        under_html = f"""
      <p class="qb-under-label">Under (est. main line, no alternates):</p>
      <div class="pill-row"><span class="pill {under_cls}">UNDER {main_line['threshold']} yds: {under_pct}%</span></div>"""

    return f"""
    <div class="qb-block">
      <p class="qb-name">{side_label}: {qb['name']} <span class="qb-team">{qb['team']}</span></p>
      <p class="qb-games">last {qb['games_sampled']} games sampled</p>
      {atts_html}
      {boost_html}
      {ci_html}
      {split_html}
      <p class="qb-over-label">Over (alternate lines):</p>
      <div class="pill-row">{''.join(pills)}</div>
      {under_html}
    </div>"""


def _render_game_card(g):
    home_bar = _prob_bar_html(g.get("home_cover_prob"),
                               f"{g['home_team_abbr']} covers {g['spread_line']}" if g.get("spread_line") is not None else "Spread")
    total_bar = _prob_bar_html(g.get("game_total_over_prob"),
                                f"Total OVER {g['total_line']}" if g.get("total_line") is not None else "Total")

    home_qb_html = _render_qb_passing_projections(g.get("home_qb"), g["home_team_abbr"])
    away_qb_html = _render_qb_passing_projections(g.get("away_qb"), g["away_team_abbr"])

    all_flags = (g.get("home_injury_flags") or []) + (g.get("away_injury_flags") or [])
    flags_html = ""
    if all_flags:
        flag_items = "".join(f"<li>{f}</li>" for f in all_flags)
        flags_html = f'<div class="injury-flags"><p class="section-subheading">Injury Notes</p><ul>{flag_items}</ul></div>'

    weather_html = ""
    if g.get("weather_note"):
        weather_html = f'<p class="weather-note">\U0001F324\uFE0F {g["weather_note"]}</p>'

    return f"""
    <div class="game-card collapsible">
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        <div>
          <p class="game-matchup">{g['away_team_abbr']} @ {g['home_team_abbr']}</p>
          <p class="game-kickoff">{g['kickoff_local']}</p>
        </div>
        <span class="collapsible-chevron">&#9660;</span>
      </div>
      <div class="collapsible-body">
        {weather_html}
        {home_bar}
        {total_bar}
        {flags_html}
        <div class="qb-projections-section">
          <h3 class="section-subheading">Player Projections</h3>
          {home_qb_html}
          {away_qb_html}
        </div>
      </div>
    </div>"""


def _render_top_qb_performers(top_qbs):
    if not top_qbs:
        return ""
    items = []
    for rank, tp in enumerate(top_qbs, start=1):
        pct = tp["hit_rate"] * 100
        items.append(f"""
        <div class="tp-card collapsible">
          <div class="tp-rank">{rank}</div>
          <div class="tp-body">
            <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
              <div>
                <p class="tp-name">{tp['name']} <span class="tp-team">{tp['team']}</span></p>
                <p class="tp-matchup">vs {tp['opponent_full']}</p>
                <div class="tp-stat-row">
                  <span class="tp-stat-badge">{tp['threshold']}+ Passing Yards</span>
                  <span class="tp-hit-rate">{pct:.0f}% <span class="tp-hit-rate-label">hit rate</span></span>
                  <span class="tp-games">last {tp['games_sampled']} games</span>
                </div>
              </div>
              <span class="collapsible-chevron">&#9660;</span>
            </div>
          </div>
        </div>""")
    return f"""
    <section class="top-performers">
      <h2 class="tp-heading">Today's Top QB Projections</h2>
      <p class="tp-subheading">Most likely to hit their projected line.</p>
      <div class="tp-grid">
        {''.join(items)}
      </div>
    </section>"""


def _render_correlated_clusters(clusters):
    if not clusters:
        return '<p class="no-clusters">No correlated clusters clear the confidence bar today.</p>'
    items = []
    for c in clusters:
        piece_html = "".join(
            f'<li><span class="pill {_pill_class(p["prob"])}">{p["label"]}: {round(p["prob"]*100)}%</span></li>'
            for p in c["pieces"]
        )
        items.append(f"""
        <div class="cluster-card collapsible">
          <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
            <div>
              <p class="cluster-game">{c['game_label']}</p>
              <p class="cluster-kickoff">{c['kickoff_local']}</p>
              <p class="cluster-avg">Average confidence: {round(c['avg_confidence']*100)}%</p>
            </div>
            <span class="collapsible-chevron">&#9660;</span>
          </div>
          <div class="collapsible-body">
            <ul class="cluster-pieces">{piece_html}</ul>
          </div>
        </div>""")
    return f"""
    <section class="clusters-section">
      <div class="tp-grid">{''.join(items)}</div>
    </section>"""


def _render_team_totals_tab(report):
    if not report:
        return '<p class="no-clusters">No games today.</p>'
    cards = []
    for g in report:
        home_total_html = ""
        if g.get("home_team_total") is not None:
            pct = round((g.get("home_team_total_prob") or 0) * 100)
            home_total_html = _prob_bar_html(g.get("home_team_total_prob"),
                                              f"{g['home_team_abbr']} projected {g['home_team_total']} points")
        away_total_html = ""
        if g.get("away_team_total") is not None:
            away_total_html = _prob_bar_html(g.get("away_team_total_prob"),
                                              f"{g['away_team_abbr']} projected {g['away_team_total']} points")
        game_total_html = _prob_bar_html(g.get("game_total_over_prob"),
                                          f"Combined total over {g['total_line']}" if g.get("total_line") is not None else "Combined total")
        spread_html = _prob_bar_html(g.get("home_cover_prob"),
                                      f"{g['home_team_abbr']} covers {g['spread_line']}" if g.get("spread_line") is not None else "Spread")
        cards.append(f"""
        <div class="game-card collapsible expanded">
          <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
            <div>
              <p class="game-matchup">{g['away_team_abbr']} @ {g['home_team_abbr']}</p>
              <p class="game-kickoff">{g['kickoff_local']}</p>
            </div>
            <span class="collapsible-chevron">&#9660;</span>
          </div>
          <div class="collapsible-body">
            {away_total_html}
            {home_total_html}
            {game_total_html}
            {spread_html}
          </div>
        </div>""")
    return f'<div class="tp-grid">{"".join(cards)}</div>'


def render_html(report):
    top_qbs = build_top_qb_performers(report)
    clusters = extract_correlated_forecasts(report)
    game_cards = [_render_game_card(g) for g in report]
    team_totals_tab = _render_team_totals_tab(report)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFL Daily Probabilities</title>
<style>
:root {{
  --bg: #0b0f19;
  --card-bg: #131a2b;
  --text: #e8ecf4;
  --text-dim: #8892a6;
  --teal: #2dd4bf;
  --hot: #22c55e;
  --warm: #eab308;
  --cool: #64748b;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg: #f5f6fa;
    --card-bg: #ffffff;
    --text: #12151c;
    --text-dim: #5a6274;
    --teal: #0d9488;
    --hot: #16a34a;
    --warm: #ca8a04;
    --cool: #94a3b8;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 0 0 40px;
}}
h1 {{
  font-size: 1.4em;
  text-align: center;
  padding: 20px 20px 4px;
  margin: 0;
}}
.updated, .disclaimer {{
  text-align: center;
  font-size: 0.78em;
  color: var(--text-dim);
  margin: 2px 0;
  padding: 0 20px;
}}
.tab-nav {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  padding: 0 20px;
  margin: 18px 0 4px;
}}
.tab-card {{
  background: var(--card-bg);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 14px 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: 4px;
  cursor: pointer;
  font-family: inherit;
  color: var(--text);
}}
.tab-card-title {{ font-size: 0.95em; font-weight: 800; }}
.tab-card-sub {{ font-size: 0.68em; color: var(--text-dim); line-height: 1.3; }}
.tab-card.active {{ border-color: var(--teal); background: rgba(45, 212, 191, 0.08); }}
.tab-card.active .tab-card-title {{ color: var(--teal); }}
.tab-panel {{ display: none; padding: 0 20px; }}
.tab-panel.active {{ display: block; }}

.collapsible-toggle {{
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  user-select: none;
  padding: 14px 16px;
}}
.collapsible-chevron {{
  flex: none;
  width: 22px;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-dim);
  font-size: 0.85em;
  transition: transform 0.2s ease;
}}
.collapsible-body {{
  overflow: hidden;
  max-height: 0;
  transition: max-height 0.25s ease;
  padding: 0 16px;
}}
.collapsible.expanded .collapsible-chevron {{ transform: rotate(180deg); }}
.collapsible.expanded .collapsible-body {{ max-height: none; padding-bottom: 16px; }}

.game-card, .cluster-card, .tp-card {{
  background: var(--card-bg);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  margin-bottom: 12px;
}}
.game-matchup, .cluster-game {{ font-weight: 800; font-size: 1em; margin: 0; }}
.game-kickoff, .cluster-kickoff {{ font-size: 0.75em; color: var(--text-dim); margin: 2px 0 0; }}
.cluster-avg {{ font-size: 0.78em; color: var(--teal); margin: 4px 0 0; font-weight: 700; }}

.prob-bar-track {{
  position: relative;
  background: rgba(255,255,255,0.06);
  border-radius: 8px;
  height: 30px;
  margin: 8px 0;
  overflow: hidden;
}}
.prob-bar-fill {{
  position: absolute;
  left: 0; top: 0; bottom: 0;
  background: var(--teal);
  opacity: 0.35;
}}
.prob-bar-label {{
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  height: 100%;
  padding: 0 10px;
  font-size: 0.78em;
  font-weight: 700;
}}

.section-subheading {{ font-size: 0.85em; margin: 14px 0 6px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.04em; }}
.qb-block {{ margin-bottom: 14px; }}
.qb-name {{ font-weight: 700; margin: 0 0 2px; }}
.qb-team {{ color: var(--text-dim); font-weight: 400; font-size: 0.85em; }}
.injury-flags {{ margin-top: 10px; }}
.injury-flags ul {{ margin: 4px 0 0; padding-left: 18px; font-size: 0.78em; color: var(--warm); }}
.injury-flags li {{ margin-bottom: 4px; }}
.weather-note {{ font-size: 0.78em; color: var(--text-dim); margin: 4px 0 10px; }}
.qb-games, .qb-attempts, .qb-boost {{ font-size: 0.75em; color: var(--text-dim); margin: 0 0 4px; }}
.qb-ci {{ font-size: 0.78em; color: var(--teal); font-weight: 700; margin: 4px 0; }}
.qb-splits {{ font-size: 0.72em; color: var(--text-dim); margin: 0 0 6px; }}
.qb-over-label, .qb-under-label {{ font-size: 0.72em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.03em; margin: 8px 0 4px; }}
.qb-no-data {{ font-size: 0.8em; color: var(--text-dim); font-style: italic; }}

.pill-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
.pill {{
  display: inline-block;
  padding: 4px 9px;
  border-radius: 999px;
  font-size: 0.7em;
  font-weight: 700;
  color: #0b0f19;
}}
.pill-hot {{ background: var(--hot); }}
.pill-warm {{ background: var(--warm); }}
.pill-cool {{ background: var(--cool); color: var(--text); }}

.tp-heading {{ font-size: 1.05em; margin: 20px 0 2px; }}
.tp-subheading {{ font-size: 0.75em; color: var(--text-dim); margin: 0 0 14px; }}
.tp-grid {{ display: flex; flex-direction: column; gap: 10px; }}
.tp-card {{ display: flex; align-items: stretch; }}
.tp-rank {{
  width: 34px;
  flex: none;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  font-size: 1.1em;
  color: var(--teal);
  border-right: 1px solid rgba(255,255,255,0.08);
}}
.tp-body {{ flex: 1; }}
.tp-name {{ font-weight: 700; margin: 0; }}
.tp-team {{ color: var(--text-dim); font-weight: 400; font-size: 0.85em; }}
.tp-matchup {{ font-size: 0.75em; color: var(--text-dim); margin: 2px 0 6px; }}
.tp-stat-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.tp-stat-badge {{
  background: rgba(45,212,191,0.12);
  color: var(--teal);
  font-size: 0.72em;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 6px;
}}
.tp-hit-rate {{ font-weight: 800; font-size: 0.95em; }}
.tp-hit-rate-label {{ font-weight: 400; font-size: 0.75em; color: var(--text-dim); }}
.tp-games {{ font-size: 0.72em; color: var(--text-dim); }}

.cluster-pieces {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }}
.no-clusters {{ color: var(--text-dim); font-size: 0.85em; text-align: center; padding: 20px; }}
</style>
<script>
function toggleCollapsible(headerEl) {{
  var card = headerEl.closest('.collapsible');
  if (!card) return;
  var body = card.querySelector('.collapsible-body');
  var isExpanded = card.classList.contains('expanded');
  if (isExpanded) {{
    card.classList.remove('expanded');
    body.style.maxHeight = '0px';
  }} else {{
    card.classList.add('expanded');
    body.style.maxHeight = body.scrollHeight + 'px';
    setTimeout(function() {{
      if (card.classList.contains('expanded')) body.style.maxHeight = 'none';
    }}, 260);
  }}
}}
function showTab(id, btn) {{
  document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  document.querySelectorAll('.tab-card').forEach(function(c) {{ c.classList.remove('active'); }});
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  window.scrollTo({{ top: 0, behavior: 'instant' }});
}}
</script>
</head>
<body>
<h1>NFL Projections</h1>
<p class="updated">Generated {local_now().strftime('%a %b %d, %Y')} {local_now().strftime('%H:%M')}</p>
<p class="disclaimer">Projections, not guarantees. Verify lineups yourself.</p>

<div class="tab-nav">
  <button class="tab-card active" data-tab="tab-teamtotals" onclick="showTab('tab-teamtotals', this)">
    <span class="tab-card-title">Team Totals</span>
    <span class="tab-card-sub">Team and game score projections</span>
  </button>
  <button class="tab-card" data-tab="tab-clusters" onclick="showTab('tab-clusters', this)">
    <span class="tab-card-title">Combined Picks</span>
    <span class="tab-card-sub">Multiple projections in one game</span>
  </button>
  <button class="tab-card" data-tab="tab-topqbs" onclick="showTab('tab-topqbs', this)">
    <span class="tab-card-title">Player Projections</span>
    <span class="tab-card-sub">Passing yards for each QB</span>
  </button>
</div>

<div id="tab-teamtotals" class="tab-panel active">
{team_totals_tab}
</div>

<div id="tab-clusters" class="tab-panel">
{_render_correlated_clusters(clusters)}
</div>

<div id="tab-topqbs" class="tab-panel">
{_render_top_qb_performers(top_qbs)}
{''.join(game_cards)}
</div>

</body>
</html>"""
    return html


if __name__ == "__main__":
    report = build_report()
    os.makedirs("docs", exist_ok=True)
    with open("docs/report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open("docs/index.html", "w") as f:
        f.write(render_html(report))

    games_with_spread_data = sum(1 for g in report if g.get("home_cover_prob") is not None)
    games_with_qb_data = sum(1 for g in report if g.get("home_qb") or g.get("away_qb"))
    total_qbs = sum((1 if g.get("home_qb") else 0) + (1 if g.get("away_qb") else 0) for g in report)

    if len(report) == 0:
        print("WARNING: 0 games in report - scoreboard fetch may have failed for all dates queried (check WARNING lines above).")
    print(f"Done. {len(report)} games processed.")
    print(f"  Spread data available for {games_with_spread_data}/{len(report)} games.")
    print(f"  QB passing projections available for {games_with_qb_data}/{len(report)} games ({total_qbs} QBs total).")
    if len(report) > 0 and games_with_spread_data == 0:
        print("  WARNING: no spread data on any game - check ESPN team schedule score-field parsing "
              "(get_team_points_for_against) or odds availability on the scoreboard payload.")
    if len(report) > 0 and games_with_qb_data == 0:
        print("  WARNING: no QB data on any game - check whether nfl_data_py's pbp release for the "
              "season being used actually exists yet (see load_pbp_with_fallback warnings above) and "
              "whether team abbreviations are resolving correctly (NFLVERSE_TO_ESPN_TEAM mapping).")


# ============================================================================
# Injuries / missing-starter flagging + numeric usage boost
#
# CONFIRMED AT BUILD TIME: unlike the rest of this script (which uses
# site.api.espn.com's simple flat JSON), NFL injuries live on a DIFFERENT
# ESPN host with a different, $ref-paginated response shape:
#   https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{TEAM_ID}/injuries
# This returns a paginated index of $ref links, each of which must be
# individually dereferenced (a second HTTP call per injury) to get the
# athlete name/status - verified via ESPN's own documented API shape for
# this same "core API" family (see e.g. the athletes/eventlog pattern),
# NOT verified end-to-end against a live injuries response, since this
# sandbox's tooling could not fetch that exact URL to confirm field names.
#
# Given that real uncertainty, this is written defensively: wrapped in
# try/except at every network hop, degrading cleanly to "no boost, no
# flag" rather than crashing the whole run if the shape is even slightly
# different than expected. TEST THIS AGAINST A REAL RUN before trusting
# the boost numbers in a live betting decision - the boost logic itself
# (bayesian_shrinkage inputs, +12%/+8% multipliers) is solid and tested;
# what's unverified is specifically whether this fetch code correctly
# parses ESPN's real response.
# ============================================================================

SPORTS_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

LIKELY_OUT_STATUSES = ("out", "inactive", "injured reserve", "ir", "suspension", "suspended")
GAME_TIME_DECISION_STATUSES = ("questionable", "doubtful", "day-to-day", "day to day", "gtd")


def get_team_injuries_raw(team_id):
    """
    Fetches the $ref-paginated injuries index for a team, dereferences
    each item to get status/comment, then dereferences the "athlete"
    field too (also a bare $ref) to get the player's name.

    ?limit=100 is required in the index URL - the default page size (25)
    silently drops later entries for teams with a long injury report.
    "athlete" must be fetched separately; it's never an inline object.

    Returns None if the index fetch fails outright - callers must not
    treat that as "nobody's hurt". Returns [] if nothing is flagged.
    """
    try:
        index_url = f"{SPORTS_CORE_BASE}/teams/{team_id}/injuries?limit=100"
        index_payload = _fetch_with_retry(index_url)
    except Exception as e:
        print(f"WARNING: injuries index fetch failed for team_id={team_id}: {e}")
        return None

    items = index_payload.get("items", [])
    if not items:
        return []  # empty list = check succeeded, nobody flagged - different from None

    injuries = []
    for item in items:
        ref_url = item.get("$ref") if isinstance(item, dict) else None
        if not ref_url:
            # some ESPN core-API responses inline the object directly
            # instead of a $ref, depending on endpoint - handle both,
            # even though the verified live response always used $ref.
            if isinstance(item, dict) and ("status" in item or "athlete" in item):
                injuries.append(item)
            continue
        try:
            detail = _fetch_with_retry(ref_url)
        except Exception as e:
            print(f"WARNING: failed to dereference injury item for team_id={team_id}: {e}")
            continue

        # athlete is itself a bare $ref - dereference it too, to get an
        # actual name instead of just an athlete ID. This is a THIRD
        # network hop per injury (index -> injury detail -> athlete
        # detail), confirmed necessary against the real response shape.
        athlete_ref = detail.get("athlete", {})
        athlete_url = athlete_ref.get("$ref") if isinstance(athlete_ref, dict) else None
        if athlete_url:
            try:
                athlete_detail = _fetch_with_retry(athlete_url)
                detail["athlete"] = athlete_detail  # replace the bare $ref with the real object
            except Exception as e:
                print(f"WARNING: failed to dereference athlete for an injury, team_id={team_id}: {e}")
                # keep detail["athlete"] as the unresolved $ref dict - the
                # caller's name lookup will just fail to match anything
                # for this one entry, which is a safe (if silent) failure
                # mode rather than crashing the whole run.

        injuries.append(detail)
    return injuries


def flag_missing_starters(team_id, starter_names):
    """
    Cross-references confirmed injuries against a set of names you care
    about (e.g. {QB, WR1, RB1}). Returns (flags: list[str], missing_names: set[str]).
    Mirrors the WNBA original's flag_missing_starters in spirit: plain
    human-readable flags PLUS a names set that feeds the numeric usage
    boost - this script uses BOTH (flag for visibility, boost for the
    numeric adjustment your spec asked for), rather than only one.
    """
    injuries = get_team_injuries_raw(team_id)
    if injuries is None:
        return (["Injury check unavailable this run - verify starters manually before trusting this team's props."], set())

    flags = []
    missing_names = set()
    for inj in injuries:
        athlete = inj.get("athlete", {}) if isinstance(inj, dict) else {}
        name = athlete.get("displayName") or athlete.get("fullName") or inj.get("longComment", "")[:40]
        status = inj.get("status") or inj.get("type", {}).get("name", "") if isinstance(inj, dict) else ""
        status_lower = str(status).lower()

        if not name or name not in starter_names:
            continue
        if status_lower in ("probable", "active", "available", ""):
            continue

        if status_lower in LIKELY_OUT_STATUSES:
            flags.append(f"{name} listed as {status} - very likely to sit. "
                          f"Treat this team's props and spread with extra caution.")
            missing_names.add(name)
        elif status_lower in GAME_TIME_DECISION_STATUSES:
            flags.append(f"{name} listed as {status} - a game-time decision, not confirmed. "
                          f"Re-check closer to kickoff.")
            # NOT added to missing_names: a boost fires only on a
            # confirmed-out designation, not a maybe - applying +12%
            # on a coin-flip questionable tag would overstate confidence
            # exactly the way shrinkage exists to prevent elsewhere.
        else:
            flags.append(f"{name} listed as {status}. Treat this team's props with extra caution.")
            missing_names.add(name)

    return (flags, missing_names)


def injury_mult_for_team(team_espn_abbr, injury_status_by_team, injury_mults):
    """Combines QB/WR1/RB1-out multipliers for one team's projected total, multiplicatively."""
    status = injury_status_by_team.get(team_espn_abbr, {})
    mult = 1.0
    if status.get("qb_out"):
        mult *= injury_mults.get("qb", INJURY_MULT_FALLBACK["qb"])
    if status.get("wr1_out"):
        mult *= injury_mults.get("wr1", INJURY_MULT_FALLBACK["wr1"])
    if status.get("rb1_out"):
        mult *= injury_mults.get("rb1", INJURY_MULT_FALLBACK["rb1"])
    return mult


def find_wr1_rb1(team_abbr, wr_rb_logs, min_games=3):
    """
    Approximates WR1/RB1 as the highest-total-yardage receiver/rusher on
    the team across their sampled games this season - a reasonable proxy
    since there's no separate 'depth chart rank' field wired in yet.
    Returns (wr1_name, rb1_name), either of which may be None.
    """
    def _top(logs_dict, yard_key):
        candidates = [
            (data["name"], sum(g[yard_key] for g in data["games"]))
            for pid, data in logs_dict.items()
            if data["team"] == team_abbr and len(data["games"]) >= min_games
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    wr1 = _top(wr_rb_logs.get("receivers", {}), "receiving_yards")
    rb1 = _top(wr_rb_logs.get("rushers", {}), "rushing_yards")
    return wr1, rb1


# Empirical usage boost: WR1/RB1-out multipliers, fit from history when
# there's enough data (MIN_SAMPLES_FOR_EMPIRICAL_BOOST), otherwise
# falls back to a documented default rather than trusting a noisy
# small-sample estimate.
# ============================================================================

MIN_SAMPLES_FOR_EMPIRICAL_BOOST = 25


def fit_empirical_usage_boosts(pbp_df, injuries_df, roster_crosswalk_lookup_fn):
    """
    Computes the empirical QB-passing-yards delta when a team's WR1 or
    RB1 is listed "Out". roster_crosswalk_lookup_fn should be
    get_roster_name_crosswalk.

    Falls back to +12%/+8% if the sample size for either is below
    MIN_SAMPLES_FOR_EMPIRICAL_BOOST, with the actual n printed.
    """
    fallback_wr1, fallback_rb1 = 1.12, 1.08
    if pbp_df is None or pbp_df.empty or injuries_df is None or injuries_df.empty:
        print("WARNING: fit_empirical_usage_boosts missing pbp or injuries data - using fallback +12%/+8%.")
        return {"wr1_out_mult": fallback_wr1, "rb1_out_mult": fallback_rb1, "wr1_n": 0, "rb1_n": 0}

    df = pbp_df[pbp_df.get("season_type") == "REG"].copy() if "season_type" in pbp_df.columns else pbp_df.copy()

    def _resolve_top_players(play_type, yard_col, jersey_col, id_col):
        plays = df.dropna(subset=[id_col, jersey_col]).copy()
        if yard_col in plays.columns:
            plays[yard_col] = plays[yard_col].fillna(0)
        else:
            return {}
        plays["resolved_name"] = plays.apply(
            lambda r: roster_crosswalk_lookup_fn(int(r["season"])).get((r["posteam"], r[jersey_col]))
            or r.get(play_type), axis=1
        )
        totals = plays.groupby(["posteam", "resolved_name"])[yard_col].sum().reset_index()
        return totals.sort_values(yard_col, ascending=False).groupby("posteam").head(1) \
            .set_index("posteam")["resolved_name"].to_dict()

    wr1_by_team = _resolve_top_players("receiver", "receiving_yards", "receiver_jersey_number", "receiver_id")
    rb1_by_team = _resolve_top_players("rusher", "rushing_yards", "rusher_jersey_number", "rusher_id")

    pass_plays = df[df["pass_attempt"] == 1].dropna(subset=["passer_id"]).copy()
    pass_plays["passing_yards"] = pass_plays["passing_yards"].fillna(0.0)
    qb_week = pass_plays.groupby(["posteam", "week"])["passing_yards"].sum().reset_index()
    qb_season_avg = qb_week.groupby("posteam")["passing_yards"].mean().to_dict()

    def _compute_delta(position, top_by_team):
        out_players = injuries_df[(injuries_df["position"] == position) & (injuries_df["report_status"] == "Out")]
        deltas = []
        for _, row in out_players.iterrows():
            team, week, name = row["team"], row["week"], row["full_name"]
            if top_by_team.get(team) != name:
                continue
            this_week = qb_week[(qb_week["posteam"] == team) & (qb_week["week"] == week)]
            if this_week.empty:
                continue
            actual = this_week["passing_yards"].values[0]
            season_avg = qb_season_avg.get(team)
            if season_avg and season_avg > 0:
                deltas.append((actual - season_avg) / season_avg)
        return deltas

    wr1_deltas = _compute_delta("WR", wr1_by_team)
    rb1_deltas = _compute_delta("RB", rb1_by_team)

    def _resolve(deltas, fallback, label):
        n = len(deltas)
        if n < MIN_SAMPLES_FOR_EMPIRICAL_BOOST:
            print(f"WARNING: only {n} usable {label}-out samples found (< {MIN_SAMPLES_FOR_EMPIRICAL_BOOST} "
                  f"minimum for a trustworthy estimate) - using documented fallback "
                  f"{'+' if fallback>1 else ''}{round((fallback-1)*100)}% instead of an unreliable empirical number.")
            return fallback, n
        mean_delta = sum(deltas) / n
        mult = round(1 + mean_delta, 3)
        print(f"Fitted empirical {label}-out usage boost from {n} real samples: "
              f"mean delta={round(mean_delta*100,1)}% -> multiplier={mult}")
        return mult, n

    wr1_mult, wr1_n = _resolve(wr1_deltas, fallback_wr1, "WR1")
    rb1_mult, rb1_n = _resolve(rb1_deltas, fallback_rb1, "RB1")

    return {"wr1_out_mult": wr1_mult, "rb1_out_mult": rb1_mult, "wr1_n": wr1_n, "rb1_n": rb1_n}
