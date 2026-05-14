"""
session_profiler.py — Session-Regime Duration Intelligence + Adaptive Thresholds
=================================================================================

Answers two questions the rest of the system cannot:

  Q1: "How long does this regime typically last in THIS session?"
      e.g. BULL_TREND during London = avg 72 min, 80% chance still alive at 30min

  Q2: "Given regime + volatility + session, what meta-probability threshold
       should we require before taking this trade?"
      e.g. REVERSAL + high vol = require 0.72 not 0.55

─────────────────────────────────────────────────────────────────────
WHY THIS MATTERS

XGBoost is stateless — it sees features at a single bar and decides.
It cannot reason: "BULL_TREND usually lasts 72 min in London and
we're 65 min in — this entry is late-cycle."

The GMM-HMM's transmat encodes transition probabilities but never
surfaces them as human-readable timing estimates.

This module bridges that gap:

  After HMM training → scan full label history →
  compute per (regime, session) duration statistics →
  save to session_profiles.json →
  enrich every predict() output with timing intelligence →
  Claude reads: "BULL_TREND London: 65/72 min elapsed (AGEING)"

─────────────────────────────────────────────────────────────────────
ADAPTIVE THRESHOLDS

Fixed threshold (e.g. 0.55) treats all market conditions identically.
That's wrong:

  BULL_TREND aligned trade     → confidence justified → 0.52 is fine
  REVERSAL + counter-signal   → high uncertainty    → need 0.72+
  COMPRESSION pre-breakout    → direction unknown   → need 0.75+
  Late-cycle AGEING regime    → exhaustion risk     → raise threshold
  High normalised volatility  → widen threshold     → more cautious

Adaptive thresholds are computed from:
  1. Base threshold per regime (from walk-forward win-rate data)
  2. Volatility adjustment (norm_vol vs historical)
  3. Regime age adjustment (FRESH=lower, OVERDUE=higher)
  4. Session adjustment (London open=lower, dead session=higher)
  5. Signal alignment adjustment (aligned=lower, counter=higher)

─────────────────────────────────────────────────────────────────────
PUBLIC API

build_session_profiles(labels, features_df)
    → called once in trainer.py after HMM decode
    → saves session_profiles.json

enrich_regime_result(regime_result, features_row, signal, session)
    → called in regime_detector.predict() / main_bot
    → returns regime_result with session_profile + adaptive_threshold added

get_adaptive_threshold(regime, session, norm_vol, age_status, signal_aligned)
    → called in meta_labeller.predict_meta()
    → returns float threshold

format_for_prompt(regime_result)
    → returns string injected into Claude prompt
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

current_dir   = os.path.dirname(os.path.abspath(__file__))
import sys as _sys_sp
_root_sp = os.path.normpath(os.path.join(current_dir, '..', '..'))
if _root_sp not in _sys_sp.path: _sys_sp.path.insert(0, _root_sp)
from paths import SESSION_PROFILES_PATH as PROFILES_PATH, create_all_dirs as _cad_sp
_cad_sp()

# ── Candles to minutes (M5 data) ─────────────────────────────────
M5_MINUTES = 5

# ── Sessions (NY time hours) ─────────────────────────────────────
SESSIONS = {
    "Asian":  lambda h: h >= 19 or h < 2,
    "London": lambda h: 2 <= h < 7,
    "NY":     lambda h: 7 <= h < 17,
    "Dead":   lambda h: 17 <= h < 19,
}

# ── Regime age classification ─────────────────────────────────────
# Expressed as fraction of typical duration (elapsed / mean_duration)
AGE_FRESH   = (0.0,  0.35)   # 0–35% through typical duration
AGE_MATURE  = (0.35, 0.70)   # 35–70% — prime trading window
AGE_AGEING  = (0.70, 1.10)   # 70–110% — late cycle, caution
AGE_OVERDUE = (1.10, 9999)   # >110% of typical — statistically overdue to end

# ── Regime base thresholds ────────────────────────────────────────
# Derived from expected win rates per regime type.
# These are starting points — walk-forward calibration will refine them.
REGIME_BASE_THRESHOLDS = {
    "BULL_TREND":    0.52,   # directional, high probability aligned trades
    "BEAR_TREND":    0.52,   # same
    "LOW_VOL_RANGE": 0.58,   # mean reversion valid but less directional
    "COMPRESSION":   0.72,   # pre-breakout, direction unknown → be strict
    "REVERSAL":      0.68,   # spike/news — high volatility → need confidence
    "UNCERTAIN":     0.80,   # model near-random → only take very high prob
}

# ── Volatility adjustment buckets ────────────────────────────────
# norm_vol = current_rolling_vol / historical_rolling_vol_mean
# If gold is running at 2× its normal daily volatility → raise threshold
VOL_ADJUSTMENTS = [
    (2.0,  9999, +0.08),   # extreme vol → much stricter
    (1.5,  2.0,  +0.05),   # high vol
    (1.2,  1.5,  +0.02),   # slightly elevated
    (0.8,  1.2,   0.00),   # normal
    (0.5,  0.8,  -0.03),   # low vol → slightly looser (mean-reversion works)
    (0.0,  0.5,  -0.05),   # very low vol
]

# ── Age adjustments ───────────────────────────────────────────────
AGE_ADJUSTMENTS = {
    "FRESH":   -0.03,   # regime just started → best entries → allow more
    "MATURE":   0.00,   # prime window → no adjustment
    "AGEING":  +0.04,   # late cycle → cautious
    "OVERDUE": +0.08,   # statistically overdue to flip → strict
}

# ── Session adjustments ───────────────────────────────────────────
SESSION_ADJUSTMENTS = {
    "London": -0.02,   # London open: institutional flow, most reliable
    "NY":      0.00,   # NY: good but more news risk
    "Asian":  +0.03,   # Asian: low vol, setups less reliable
    "Dead":   +0.06,   # dead session: avoid entirely
}

# ── Alignment adjustments ─────────────────────────────────────────
ALIGNMENT_ADJUSTMENTS = {
    True:  -0.03,   # signal aligned with regime direction → lower threshold
    False: +0.06,   # counter-regime trade → require much more confidence
}

# ── Survival probability model ────────────────────────────────────
# P(regime still active at +N candles | current age)
# Modelled as exponential decay: P = exp(-lambda * n_candles)
# Lambda calibrated from duration mean: lambda = 1 / mean_candles
# This gives a simple but calibrated survival probability.

def _survival_prob(elapsed_candles: int, mean_candles: float,
                   horizon_candles: int = 3) -> float:
    """
    Probability that the current regime survives the next `horizon_candles` bars.
    Uses exponential survival model: P(T > t) = exp(-t / mean)

    Args:
        elapsed_candles:  how many M5 bars this regime has been active
        mean_candles:     historical mean regime duration in M5 bars
        horizon_candles:  how many bars ahead to estimate (default 3 = 15min)
    """
    if mean_candles <= 0:
        return 0.5
    lam     = 1.0 / max(mean_candles, 1.0)
    p_now   = float(np.exp(-lam * elapsed_candles))
    p_later = float(np.exp(-lam * (elapsed_candles + horizon_candles)))
    # Conditional: P(survive another horizon | survived until now)
    p_cond  = p_later / max(p_now, 1e-9)
    return float(np.clip(p_cond, 0.01, 0.99))


def _classify_age(elapsed_candles: int, mean_candles: float) -> str:
    """Returns FRESH / MATURE / AGEING / OVERDUE based on elapsed/mean ratio."""
    if mean_candles <= 0:
        return "MATURE"
    ratio = elapsed_candles / mean_candles
    if   ratio < AGE_FRESH[1]:   return "FRESH"
    elif ratio < AGE_MATURE[1]:  return "MATURE"
    elif ratio < AGE_AGEING[1]:  return "AGEING"
    else:                         return "OVERDUE"


def _get_session(hour: int) -> str:
    for name, fn in SESSIONS.items():
        if fn(hour):
            return name
    return "Dead"


# ================================================================
# BUILD SESSION PROFILES (called once in trainer.py)
# ================================================================

def build_session_profiles(
    labels:       pd.Series,
    features_df:  pd.DataFrame,
) -> dict:
    """
    Scans full 10-year label history and computes per (regime, session)
    duration statistics.

    Called once at the end of trainer.py after HMM Viterbi decode.
    Output saved to session_profiles.json — loaded at bot startup.

    Args:
        labels:      pd.Series of regime strings (M5-indexed)
        features_df: feature matrix — must contain 'is_asian', 'is_london',
                     'is_ny' flags OR index must be timezone-aware

    Returns:
        dict — full profile data (also saved to disk)
    """
    import pytz
    NY_TZ = pytz.timezone('America/New_York')

    print("\n[SessionProfiler] Building session-regime duration profiles...")
    print(f"[SessionProfiler] Dataset: {len(labels):,} bars "
          f"({labels.index[0].date()} → {labels.index[-1].date()})")

    # ── Determine session for each bar ───────────────────────────
    def _bar_session(ts):
        try:
            if ts.tzinfo is None:
                ts_ny = ts.tz_localize('UTC').tz_convert(NY_TZ)
            else:
                ts_ny = ts.tz_convert(NY_TZ)
            return _get_session(ts_ny.hour)
        except Exception:
            return "Dead"

    print("[SessionProfiler] Assigning sessions to bars...")
    sessions = pd.Series(
        [_bar_session(ts) for ts in labels.index],
        index=labels.index, name='session'
    )

    # ── Scan for regime runs ──────────────────────────────────────
    # A run = consecutive sequence of same regime
    print("[SessionProfiler] Scanning regime runs...")

    runs = []   # list of {regime, session, duration_candles, duration_min}
    current_regime  = None
    current_session = None
    current_start   = 0

    label_vals   = labels.values
    session_vals = sessions.values
    n = len(label_vals)

    for i in range(n):
        regime  = label_vals[i]
        session = session_vals[i]

        if regime != current_regime:
            # Regime changed → record completed run
            if current_regime is not None:
                dur_candles = i - current_start
                # Dominant session during this run
                run_sessions = session_vals[current_start:i]
                dominant_session = (
                    pd.Series(run_sessions).value_counts().index[0]
                    if len(run_sessions) > 0 else "Dead"
                )
                runs.append({
                    "regime":          current_regime,
                    "session":         dominant_session,
                    "duration_candles": dur_candles,
                    "duration_min":    dur_candles * M5_MINUTES,
                })
            current_regime  = regime
            current_session = session
            current_start   = i

    # Final run
    if current_regime is not None:
        dur_candles = n - current_start
        runs.append({
            "regime":           current_regime,
            "session":          current_session,
            "duration_candles": dur_candles,
            "duration_min":     dur_candles * M5_MINUTES,
        })

    runs_df = pd.DataFrame(runs)
    print(f"[SessionProfiler] Found {len(runs_df):,} regime runs")

    # ── Compute statistics per (regime, session) ──────────────────
    profiles = {}
    all_sessions = ["Asian", "London", "NY", "Dead", "ALL"]

    for regime in ["REVERSAL", "BULL_TREND", "BEAR_TREND",
                   "COMPRESSION", "LOW_VOL_RANGE"]:
        profiles[regime] = {}
        regime_runs = runs_df[runs_df["regime"] == regime]

        for session in all_sessions:
            if session == "ALL":
                subset = regime_runs
            else:
                subset = regime_runs[regime_runs["session"] == session]

            if len(subset) < 5:
                # Not enough data for this specific session combo.
                # FIX: use the session's own data if any exists (even <5),
                # otherwise mark as no-data rather than collapsing to ALL-regime
                # stats. The old behaviour made Asian and London show identical
                # numbers when one of them had <5 runs (both fell back to ALL).
                if len(subset) == 0:
                    profiles[regime][session] = {
                        "mean_candles":    0,
                        "median_candles":  0,
                        "p25_candles":     0,
                        "p75_candles":     0,
                        "p90_candles":     0,
                        "mean_min":        0,
                        "median_min":      0,
                        "sample_size":     0,
                        "fallback_used":   True,
                    }
                    continue
                fallback = True
                # Use own sparse data — at least the numbers are session-specific
            else:
                fallback = False

            durs = subset["duration_candles"].values

            if len(durs) == 0:
                profiles[regime][session] = {
                    "mean_candles":    12,
                    "median_candles":  10,
                    "p25_candles":     5,
                    "p75_candles":     20,
                    "p90_candles":     30,
                    "mean_min":        60,
                    "median_min":      50,
                    "sample_size":     0,
                    "fallback_used":   True,
                }
                continue

            profiles[regime][session] = {
                "mean_candles":   round(float(np.mean(durs)),   1),
                "median_candles": round(float(np.median(durs)), 1),
                "p25_candles":    round(float(np.percentile(durs, 25)), 1),
                "p75_candles":    round(float(np.percentile(durs, 75)), 1),
                "p90_candles":    round(float(np.percentile(durs, 90)), 1),
                "mean_min":       round(float(np.mean(durs)) * M5_MINUTES, 0),
                "median_min":     round(float(np.median(durs)) * M5_MINUTES, 0),
                "sample_size":    int(len(subset)),
                "fallback_used":  fallback,
            }

    # ── Print summary ─────────────────────────────────────────────
    print("\n[SessionProfiler] ─── Session-Regime Duration Summary ───")
    print(f"{'Regime':<16} {'Session':<8} {'Mean':>8} {'Median':>8} "
          f"{'P75':>8} {'N':>6}")
    print("─" * 58)
    for regime in profiles:
        for session in ["Asian", "London", "NY"]:
            p = profiles[regime][session]
            print(f"{regime:<16} {session:<8} "
                  f"{p['mean_min']:>6.0f}m  "
                  f"{p['median_min']:>6.0f}m  "
                  f"{p['p75_candles']*M5_MINUTES:>6.0f}m  "
                  f"{p['sample_size']:>6,}")

    # ── Save ──────────────────────────────────────────────────────
    output = {
        "built_date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_runs":         len(runs_df),
        "m5_minutes":     M5_MINUTES,
        "profiles":       profiles,
    }
    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    with open(PROFILES_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[SessionProfiler] Saved → {PROFILES_PATH}")

    return output


# ================================================================
# LOAD PROFILES (called once at bot startup)
# ================================================================

_profiles_cache = None

def _load_profiles() -> dict:
    global _profiles_cache
    if _profiles_cache is not None:
        return _profiles_cache

    if not os.path.exists(PROFILES_PATH):
        print("[SessionProfiler] No profile file found. "
              "Run trainer.py to build profiles.")
        return {}

    try:
        with open(PROFILES_PATH, 'r') as f:
            data = json.load(f)
        _profiles_cache = data.get("profiles", {})
        print(f"[SessionProfiler] Profiles loaded "
              f"(built {data.get('built_date','?')}, "
              f"{data.get('n_runs','?')} historical runs)")
        return _profiles_cache
    except Exception as e:
        print(f"[SessionProfiler] Load error: {e}")
        return {}


# ================================================================
# ADAPTIVE THRESHOLD (core of the upgrade)
# ================================================================

def get_adaptive_threshold(
    regime:          str,
    session:         str,
    norm_vol:        float,
    age_status:      str,
    signal_aligned:  bool,
) -> dict:
    """
    Computes the adaptive meta-probability threshold for this specific
    combination of regime + volatility + age + session + signal direction.

    Returns a dict with the threshold AND breakdown so Claude can see why.

    Args:
        regime:         current regime string ("BULL_TREND" etc.)
        session:        "Asian" | "London" | "NY" | "Dead"
        norm_vol:       normalised volatility = current_vol / hist_vol_mean
                        Values: 0.5=very calm, 1.0=normal, 2.0=very hot
        age_status:     "FRESH" | "MATURE" | "AGEING" | "OVERDUE"
        signal_aligned: True if signal direction matches regime direction

    Returns:
        {
            "threshold":       float,   # final threshold to apply
            "base":            float,
            "vol_adj":         float,
            "age_adj":         float,
            "session_adj":     float,
            "alignment_adj":   float,
            "breakdown":       str,     # human-readable
        }
    """
    base         = REGIME_BASE_THRESHOLDS.get(regime, 0.60)
    vol_adj      = 0.0
    age_adj      = AGE_ADJUSTMENTS.get(age_status, 0.0)
    session_adj  = SESSION_ADJUSTMENTS.get(session, 0.0)
    align_adj    = ALIGNMENT_ADJUSTMENTS.get(signal_aligned, 0.0)

    # Volatility adjustment
    for lo, hi, adj in VOL_ADJUSTMENTS:
        if lo <= norm_vol < hi:
            vol_adj = adj
            break

    threshold = float(np.clip(
        base + vol_adj + age_adj + session_adj + align_adj,
        0.40, 0.90
    ))

    breakdown = (
        f"base={base:.2f} "
        f"+ vol_adj={vol_adj:+.2f} ({norm_vol:.1f}x vol) "
        f"+ age_adj={age_adj:+.2f} ({age_status}) "
        f"+ session_adj={session_adj:+.2f} ({session}) "
        f"+ align_adj={align_adj:+.2f} "
        f"({'aligned' if signal_aligned else 'counter'}) "
        f"= {threshold:.2f}"
    )

    return {
        "threshold":     round(threshold,   3),
        "base":          round(base,         3),
        "vol_adj":       round(vol_adj,      3),
        "age_adj":       round(age_adj,      3),
        "session_adj":   round(session_adj,  3),
        "alignment_adj": round(align_adj,    3),
        "breakdown":     breakdown,
    }


# ================================================================
# ENRICH REGIME RESULT (called in regime_detector / main_bot)
# ================================================================

def enrich_regime_result(
    regime_result: dict,
    session:       str,
    signal:        str = None,
    norm_vol:      float = 1.0,
) -> dict:
    """
    Adds session-timing intelligence to a regime_result dict.

    Called after regime_detector.predict() — enriches the output before
    it flows into regime_router, meta_labeller, and the Claude prompt.

    Args:
        regime_result: output of regime_detector.predict()
        session:       current session string
        signal:        "BUY" or "SELL" (for alignment check) — optional
        norm_vol:      normalised volatility

    Returns:
        Enriched regime_result with new "session_profile" key.
    """
    profiles = _load_profiles()
    regime   = regime_result.get("regime", "LOW_VOL_RANGE")
    persist  = regime_result.get("persistence", {})

    elapsed_candles = int(persist.get("candles_since_regime_start", 1))
    dur_mean_candles = float(persist.get("regime_duration_mean", 20.0))

    # Get session-specific profile
    regime_profiles = profiles.get(regime, {})
    session_prof    = regime_profiles.get(session) or regime_profiles.get("ALL", {})

    # Use profile mean if available, else persistence mean
    profile_mean = float(session_prof.get("mean_candles", dur_mean_candles))
    use_mean     = profile_mean if profile_mean > 0 else dur_mean_candles

    # Compute derived timing values
    age_status    = _classify_age(elapsed_candles, use_mean)
    survival_prob = _survival_prob(elapsed_candles, use_mean, horizon_candles=3)
    elapsed_min   = elapsed_candles * M5_MINUTES
    typical_min   = session_prof.get("mean_min", use_mean * M5_MINUTES)
    remaining_pct = max(0.0, 1.0 - (elapsed_candles / max(use_mean, 1.0)))

    # Alignment check
    signal_aligned = True   # default assume aligned
    if signal:
        sig = signal.upper()
        if   regime == "BULL_TREND" and sig == "SELL": signal_aligned = False
        elif regime == "BEAR_TREND" and sig == "BUY":  signal_aligned = False

    # Adaptive threshold
    threshold_data = get_adaptive_threshold(
        regime, session, norm_vol, age_status, signal_aligned)

    # Build enriched session profile block
    session_profile = {
        "session":            session,
        "typical_dur_min":    int(typical_min),
        "typical_dur_candles": int(use_mean),
        "elapsed_candles":    elapsed_candles,
        "elapsed_min":        elapsed_min,
        "remaining_pct":      round(remaining_pct, 2),
        "survival_prob_15m":  round(survival_prob, 2),
        "age_status":         age_status,
        "norm_vol":           round(norm_vol, 2),
        "signal_aligned":     signal_aligned,
        "adaptive_threshold": threshold_data,
        "sample_size":        int(session_prof.get("sample_size", 0)),
        # Percentile bands for context
        "p25_min":   int(session_prof.get("p25_candles", 5) * M5_MINUTES),
        "p75_min":   int(session_prof.get("p75_candles", 30) * M5_MINUTES),
        "p90_min":   int(session_prof.get("p90_candles", 50) * M5_MINUTES),
    }

    enriched = dict(regime_result)
    enriched["session_profile"] = session_profile
    return enriched


# ================================================================
# COMPUTE NORMALISED VOLATILITY (live helper)
# ================================================================

def compute_norm_vol(features_row: pd.DataFrame = None,
                     m5_df: pd.DataFrame = None) -> float:
    """
    Computes normalised volatility for adaptive threshold.
    norm_vol = current rolling vol / historical rolling vol mean

    Uses m5_rolling_vol_20 / m5_rolling_vol_100 ratio if available
    (already computed in feature_engineer.py as vol_ratio_20_100).

    Returns float — 1.0 = normal, 2.0 = twice normal, 0.5 = very calm.
    """
    if features_row is not None:
        # Best case: use pre-computed vol ratio from feature matrix
        if "m5_vol_ratio_20_100" in features_row.columns:
            v = float(features_row["m5_vol_ratio_20_100"].iloc[-1])
            if not np.isnan(v) and v > 0:
                return float(np.clip(v, 0.1, 5.0))

        # Fallback: use ATR percentile as proxy
        if "m5_atr_percentile" in features_row.columns:
            pct = float(features_row["m5_atr_percentile"].iloc[-1])
            if not np.isnan(pct):
                # Convert percentile to multiplier: p50=1.0, p90=2.0, p10=0.5
                return float(np.clip(0.5 + pct * 1.5, 0.3, 3.0))

    # Fallback: no features available
    return 1.0


# ================================================================
# FORMAT FOR GEMINI PROMPT
# ================================================================

def format_session_profile_for_prompt(regime_result: dict) -> str:
    """
    Formats session timing + adaptive threshold into a concise string
    for injection into the Claude prompt.

    Claude sees exactly how old this regime is relative to its typical
    lifespan in this session — and what threshold the meta-model is using.
    """
    sp = regime_result.get("session_profile")
    if not sp:
        return ""

    regime   = regime_result.get("regime", "?")
    conf     = regime_result.get("confidence")
    conf_str = f"{conf:.0%}" if conf else "N/A"
    at       = sp.get("adaptive_threshold", {})

    age      = sp["age_status"]
    age_icon = {"FRESH": "🟢", "MATURE": "🟡",
                "AGEING": "🟠", "OVERDUE": "🔴"}.get(age, "⚪")

    align_str = ("✓ ALIGNED" if sp.get("signal_aligned", True)
                 else "⚠ COUNTER-REGIME")

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  SESSION-REGIME TIMING INTELLIGENCE                     ║",
        "╚══════════════════════════════════════════════════════════╝",
        f"Regime         : {regime} (conf {conf_str})",
        f"Session        : {sp['session']}",
        f"Regime Age     : {age_icon} {age} "
        f"— {sp['elapsed_min']}min elapsed of typical {sp['typical_dur_min']}min "
        f"({int((1-sp['remaining_pct'])*100)}% complete)",
        f"Survival (15m) : {sp['survival_prob_15m']:.0%} probability regime "
        f"still active in next 15 minutes",
        f"Duration Bands : P25={sp['p25_min']}min | "
        f"P75={sp['p75_min']}min | P90={sp['p90_min']}min",
        f"Signal Align   : {align_str}",
        f"Volatility     : {sp['norm_vol']:.1f}× normal",
        f"Meta Threshold : {at.get('threshold', 0.55):.2f} "
        f"(adaptive — {at.get('breakdown', '')})",
        "══════════════════════════════════════════════════════════",
    ]

    # Regime-specific timing warnings
    if age == "OVERDUE":
        lines.insert(-1,
            f"⚠ WARNING: {regime} in {sp['session']} is {sp['elapsed_min']}min old "
            f"vs typical {sp['typical_dur_min']}min. "
            f"Statistically overdue to end. New entries carry high reversal risk. "
            f"If taking a trade, use minimum size and tight TP.")
    elif age == "AGEING":
        lines.insert(-1,
            f"⚠ CAUTION: Regime is in late cycle "
            f"({sp['elapsed_min']}/{sp['typical_dur_min']}min). "
            f"Tighten TP targets. Do not chase extensions.")
    elif age == "FRESH":
        lines.insert(-1,
            f"✓ FRESH REGIME: Best entry window. "
            f"Typical duration for {regime}/{sp['session']}: "
            f"{sp['typical_dur_min']}min. "
            f"~{sp['remaining_pct']:.0%} of typical duration remaining.")

    return "\n".join(lines)


# ================================================================
# WALK-FORWARD THRESHOLD CALIBRATION (post-training utility)
# ================================================================

def calibrate_thresholds_from_history(
    trade_history: list,
    n_thresholds:  int = 20,
) -> dict:
    """
    Walk-forward threshold calibration from completed trade history.

    For each (regime, session) combination, sweeps threshold values
    and finds the one that maximises risk-adjusted return
    (Sharpe proxy = win_rate * rr - (1-win_rate)).

    Args:
        trade_history: list of dicts with:
            regime, session, meta_prob, actual_outcome ("WIN"/"LOSS"),
            signal_aligned (bool)

    Returns:
        dict of optimal thresholds per (regime, session) combo
    """
    if len(trade_history) < 50:
        print("[SessionProfiler] Insufficient trade history for calibration. "
              "Need ≥50 completed trades.")
        return {}

    df = pd.DataFrame(trade_history)
    df["win"] = (df["actual_outcome"] == "WIN").astype(int)

    thresholds = np.linspace(0.40, 0.85, n_thresholds)
    results    = {}

    for regime in df["regime"].unique():
        for session in df["session"].unique():
            subset = df[
                (df["regime"] == regime) & (df["session"] == session)
            ]
            if len(subset) < 10:
                continue

            best_score = -np.inf
            best_thresh = 0.55

            for t in thresholds:
                taken   = subset[subset["meta_prob"] >= t]
                if len(taken) < 5:
                    continue
                win_rate = taken["win"].mean()
                # Sharpe proxy assuming RR=1.5
                score = win_rate * 1.5 - (1 - win_rate)
                if score > best_score:
                    best_score  = score
                    best_thresh = t

            key = f"{regime}_{session}"
            results[key] = {
                "optimal_threshold": round(float(best_thresh), 3),
                "best_score":        round(float(best_score), 3),
                "sample_size":       len(subset),
                "win_rate_at_thresh": round(float(
                    subset[subset["meta_prob"] >= best_thresh]["win"].mean()
                    if len(subset[subset["meta_prob"] >= best_thresh]) > 0 else 0
                ), 3),
            }

    if results:
        print("\n[SessionProfiler] ─── Walk-Forward Threshold Calibration ───")
        print(f"{'Combo':<30} {'Threshold':>10} {'Win Rate':>10} {'N':>6}")
        print("─" * 60)
        for combo, r in sorted(results.items()):
            print(f"{combo:<30} {r['optimal_threshold']:>10.3f} "
                  f"{r['win_rate_at_thresh']:>10.1%} {r['sample_size']:>6}")

    return results
