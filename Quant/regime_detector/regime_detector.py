"""
regime_detector.py
===================
Live/backtest regime prediction — XGBoost (trained on GMM-HMM labels).

Public API unchanged — main_bot.py, backtest_engine.py call predict() identically.

5-State Regime Map (v4 — DIRECTIONAL):
    REVERSAL       — ATR spike, news/event driven. 50% size max.
    BULL_TREND     — DI+ > DI-, H4 EMA bullish, HH/HL structure. Full size on BUY.
    BEAR_TREND     — DI- > DI+, H4 EMA bearish, LH/LL structure. Full size on SELL.
    COMPRESSION    — extreme ATR + BB squeeze, pre-breakout. Hard block.
    LOW_VOL_RANGE  — quiet, directionless, mean-reversion valid.

Two modes:
    MODEL    — XGBoost (after trainer.py has run)
    FALLBACK — raw threshold rules (labeler.py) before model exists

Output dict:
    {
        "regime":        "BULL_TREND",
        "confidence":    0.91,
        "probabilities": {"BULL_TREND": 0.91, "LOW_VOL_RANGE": 0.05, ...},
        "mode":          "model",
        "guidance":      "...",
        "persistence":   {"candles_since_regime_start": 12, "regime_duration_mean": 34.5}
    }
"""

import os, sys, json
from collections import Counter          # BUG-01 FIX: Counter must be at module level.
                                          # It was previously only imported inside _smooth_regime()
                                          # (a local-scope import that does NOT leak to callers).
                                          # _predict_xgb() uses Counter directly for vote_tally —
                                          # that raised NameError on every XGBoost prediction,
                                          # silently falling back to threshold rules every cycle.
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from feature_engineer import (
    build_multi_tf_features, build_features, build_time_features,
    FEATURE_COLS, ALL_FEATURE_COLS, PERSISTENCE_FEATURE_COLS,
)
from labeler import (
    label_candles,
    REGIME_REVERSAL, REGIME_BULL_TREND, REGIME_BEAR_TREND,
    REGIME_COMPRESSION, REGIME_LOW_VOL_RANGE,
    # backward compat aliases
    REGIME_HIGH_VOLATILITY, REGIME_TRENDING_STRONG, REGIME_TRENDING_WEAK,
    REGIME_LOW_VOLATILITY,  REGIME_RANGING,
    ALL_REGIMES,
)

import sys as _sys_rd
_root_rd = os.path.normpath(os.path.join(current_dir, '..', '..'))
if _root_rd not in _sys_rd.path: _sys_rd.path.insert(0, _root_rd)
from paths import (HMM_PATH, REGIME_SCALER_PATH as SCALER_PATH,
                   LABEL_ENCODER_PATH as LE_PATH, REGIME_XGB_PATH as XGB_PATH,
                   MODEL_META_PATH as META_PATH,
                   DRIFT_LOG_PATH, RELOAD_FLAG_PATH,
                   REVERSAL_DETECTOR_PATH,
                   REGIME_MODEL as MODEL_DIR, create_all_dirs as _cad_rd)
_cad_rd()

DRIFT_LOG_MAX_READINGS = 2000   # rolling window — ~7 days of active session candles

# MEM-03 FIX: In-memory buffer for drift_log writes.
# OLD DESIGN: log_prediction() read the full drift_log.json (~200KB at 2000 entries),
# appended one record, and wrote the whole file back — on EVERY XGBoost prediction.
# With 2 regime calls per cycle at 5-min intervals: ~24 full R/W cycles per hour.
# The file is only read by auto_retrainer every 4 hours, so this was pure waste.
#
# NEW DESIGN: readings are buffered in-memory (_drift_log_buffer).
# Every DRIFT_LOG_FLUSH_EVERY readings, the buffer is merged with the on-disk log
# and flushed.  This cuts disk I/O from 24×/hr to 4×/hr (~6 predictions × 24/day).
_drift_log_buffer: list = []
_DRIFT_LOG_FLUSH_EVERY: int = 6   # flush to disk every 6 predictions (~30 minutes)

# ── Regime guidance injected into AI prompt ───────────────────────
REGIME_GUIDANCE = {
    REGIME_REVERSAL: (
        "MARKET REGIME: REVERSAL / HIGH VOLATILITY — "
        "ATR spike detected. News event, panic move, or liquidity grab in progress. "
        "EXTREME CAUTION. Widen SL expectations significantly. Reduce size to 50% max. "
        "Wait for at least 2 stabilisation candles before scanning for entries. "
        "False breakouts are very common in this regime."
    ),
    REGIME_BULL_TREND: (
        "MARKET REGIME: BULL TREND — "
        "Confirmed upward directional move across multiple timeframes. "
        "DI+ > DI-, H4 EMA stack bullish, clear higher-highs / higher-lows structure. "
        "FAVOUR BUY setups aggressively. Trend-continuation and momentum entries have "
        "maximum probability. Counter-trend SELL signals have low probability — "
        "require very strong structural justification. Trail stops to lock gains."
    ),
    REGIME_BEAR_TREND: (
        "MARKET REGIME: BEAR TREND — "
        "Confirmed downward directional move across multiple timeframes. "
        "DI- > DI+, H4 EMA stack bearish, clear lower-highs / lower-lows structure. "
        "FAVOUR SELL setups aggressively. Short-continuation entries have maximum "
        "probability. Counter-trend BUY signals have low probability — "
        "require very strong structural justification. Trail stops to lock gains."
    ),
    REGIME_COMPRESSION: (
        "MARKET REGIME: COMPRESSION — "
        "Extreme ATR squeeze detected. Market is coiling — pre-breakout state. "
        "DO NOT enter until an expansion candle confirms breakout direction. "
        "BB width is in the bottom quartile of recent history. "
        "The next move will be explosive but direction is unknown. Wait."
    ),
    REGIME_LOW_VOL_RANGE: (
        "MARKET REGIME: LOW VOLATILITY RANGE — "
        "Quiet, directionless market. No clear trend. "
        "Mean-reversion setups have higher probability than breakouts. "
        "Buy at range lows (liquidity sweeps of SSL), sell at range highs (BSL sweeps). "
        "Use tighter targets. Do not chase momentum. "
        "Watch for COMPRESSION developing if ATR keeps contracting."
    ),
}

# ── Singleton — loaded once per process ──────────────────────────
_model        = None
_le           = None
_mode         = None
_trained_cols = None
_state_map    = None   # Bug 8 fix: loaded from model_meta.json for transition_prob feature

# ── REVERSAL binary pre-filter (Stage 1) ─────────────────────────
# Loaded alongside the main 5-class model.  Runs first every cycle.
# If P(REVERSAL) >= REVERSAL_FIRE_THRESHOLD → return REVERSAL immediately.
_reversal_filter_loaded = False

# ── Directional confidence margin ────────────────────────────────
# When 5-class XGBoost predicts BULL or BEAR but the probability gap
# between them is narrow, it means the model is guessing direction.
# In that case, downgrade to LOW_VOL_RANGE (directionless = no bias).
#
# DIRECTIONAL_MIN_MARGIN: minimum gap P(BULL) - P(BEAR) required to
# commit to a directional label.
# Example: P(BULL)=0.48, P(BEAR)=0.38 → gap=0.10 < 0.15 → LOW_VOL_RANGE
# Example: P(BULL)=0.65, P(BEAR)=0.20 → gap=0.45 ≥ 0.15 → BULL_TREND (kept)
#
# This is the inference-time complement to the labeler's DI_MARGIN_MIN fix.
# Labeler fixes training labels. This fixes live prediction on borderline candles.
DIRECTIONAL_MIN_MARGIN = 0.15

# ── Regime persistence tracker — updated every predict() call ────
_persistence_last_regime     = None
_persistence_counter         = 0
_persistence_completed_durs  = []
_persistence_prev_regime     = None     # for previous_regime_encoded feature
_persistence_transmat        = None     # loaded from HMM at startup

# ── Smoothing buffer — last 5 raw predictions ─────────────────────
# Rolling majority vote over 5 candles prevents regime flipping every tick.
# UNCERTAIN is never added to the buffer — only real regime predictions.
_smooth_buffer   = []           # list of last 5 regime strings
SMOOTH_WINDOW    = 5

# ── Confidence threshold ───────────────────────────────────────────
UNCERTAIN_THRESHOLD = 0.45      # below this → return UNCERTAIN regime


def _load_model():
    global _model, _le, _mode, _trained_cols, _persistence_transmat
    global _reversal_filter_loaded

    if os.path.exists(XGB_PATH):
        try:
            import xgboost as xgb

            _model = xgb.XGBClassifier()
            _model.load_model(XGB_PATH)

            if os.path.exists(LE_PATH):
                _le = joblib.load(LE_PATH)

            # Load HMM transmat for live transition probability feature
            if os.path.exists(HMM_PATH):
                hmm = joblib.load(HMM_PATH)
                if hasattr(hmm, 'transmat_'):
                    _persistence_transmat = hmm.transmat_

            _mode = "model"

            if os.path.exists(META_PATH):
                with open(META_PATH, 'r') as f:
                    meta = json.load(f)
                _trained_cols = meta.get("feature_cols", ALL_FEATURE_COLS)
                arch     = meta.get("architecture", "Unknown")
                n_states = meta.get("hmm_n_states", "?")
                split    = meta.get("split_strategy", "?")
                # Bug 8 fix: load state_map so transition_prob feature works at live inference
                global _state_map
                _state_map = meta.get("state_map", None)
                print(f"[RegimeDetector] Loaded: {arch}")
                print(f"[RegimeDetector] States: {n_states} | "
                      f"Split: {split} | "
                      f"Features: {len(_trained_cols)} | "
                      f"Accuracy: {meta.get('test_accuracy', '?')} | "
                      f"Trained: {meta.get('trained_date', '?')}")
            else:
                _trained_cols = ALL_FEATURE_COLS

        except Exception as e:
            print(f"[RegimeDetector] Model load failed: {e}. Using fallback rules.")
            _mode = "fallback"
    else:
        print("[RegimeDetector] No trained model found — using threshold fallback.")
        print("[RegimeDetector] Train: python Quant/regime_detector/trainer.py --n-states 5")
        _mode = "fallback"

    # ── Load REVERSAL binary pre-filter (Stage 1) ─────────────────
    # Non-fatal — if not trained yet, predict() skips Stage 1 silently.
    if os.path.exists(REVERSAL_DETECTOR_PATH):
        try:
            import reversal_detector as _rd
            loaded = _rd.load()
            _reversal_filter_loaded = loaded
            if loaded:
                print(f"[RegimeDetector] Stage 1 REVERSAL pre-filter: ACTIVE")
            else:
                print(f"[RegimeDetector] Stage 1 REVERSAL pre-filter: load failed")
        except Exception as e:
            print(f"[RegimeDetector] Stage 1 pre-filter unavailable: {e}")
            _reversal_filter_loaded = False
    else:
        print(f"[RegimeDetector] Stage 1 REVERSAL pre-filter: not trained yet "
              f"(run trainer.py to build it)")


def predict(h1_df, m5_df=None, h4_df=None, d1_df=None):
    """
    Main entry point — called every cycle from main_bot.py and regime_router.py.

    Returns:
        dict: {
            "regime":        str,
            "confidence":    float | None,
            "probabilities": dict | None,
            "mode":          str,
            "guidance":      str
        }
    """
    global _model, _mode, _trained_cols

    if _mode is None:
        _load_model()

    try:
        if (m5_df is not None and h4_df is not None and
                d1_df is not None and not m5_df.empty):
            features = build_multi_tf_features(m5_df, h1_df, h4_df, d1_df)
        else:
            h1_feats   = build_features(h1_df, prefix='h1')
            time_feats = build_time_features(h1_df.index)
            features   = pd.concat([h1_feats, time_feats], axis=1).dropna()

        if features.empty:
            print("[RegimeDetector] Insufficient candles for prediction.")
            return _safe_default()

        # ── Apply rolling z-score to TF cols on the entire features DataFrame first ──
        # This prevents the single-row z-scoring from zeroing out standard deviation.
        from feature_engineer import apply_rolling_zscore, TF_FEATURE_COLS
        
        features_raw = features.copy()
        
        # O(N^2) FIX: Only calculate the rolling metric on the trailing window, not the whole history
        window = 500
        sliced_features = features.iloc[-window:] if len(features) >= window else features
        
        tf_cols = [c for c in TF_FEATURE_COLS if c in features.columns]
        features_z = apply_rolling_zscore(sliced_features, cols=tf_cols, window=window)
        features_z = features_z.fillna(0.0)

        latest_raw = features_raw.iloc[[-1]]
        latest_z = features_z.iloc[[-1]]

        if _mode == "model":
            return _predict_xgb(latest_z, latest_raw, state_map=_state_map)
        else:
            return _predict_fallback(latest_raw)

    except Exception as e:
        print(f"[RegimeDetector] Error: {e}")
        return _safe_default()


def _update_persistence(regime):
    """
    Updates live persistence tracker. Called after every confirmed (smoothed) prediction.
    Returns (candles_since, dur_mean, prev_regime_encoded, transition_prob).
    """
    global _persistence_last_regime, _persistence_counter, \
           _persistence_completed_durs, _persistence_prev_regime
    from labeler import ALL_REGIMES
    regime_to_int = {r: i for i, r in enumerate(ALL_REGIMES)}

    if regime != _persistence_last_regime:
        if _persistence_last_regime is not None:
            _persistence_completed_durs.append(_persistence_counter)
            if len(_persistence_completed_durs) > 50:
                _persistence_completed_durs = _persistence_completed_durs[-50:]
        _persistence_prev_regime  = _persistence_last_regime
        _persistence_last_regime  = regime
        _persistence_counter      = 1
    else:
        _persistence_counter += 1

    dur_mean      = (float(np.mean(_persistence_completed_durs[-20:]))
                     if _persistence_completed_durs else float(_persistence_counter))
    prev_enc      = float(regime_to_int.get(_persistence_prev_regime, 4)
                          if _persistence_prev_regime else 4)
    return float(_persistence_counter), dur_mean, prev_enc


def _get_transition_prob(prev_regime, curr_regime, state_map):
    """Looks up P(curr_state | prev_state) from loaded HMM transmat."""
    if _persistence_transmat is None or state_map is None:
        return 0.5
    regime_to_state = {v: k for k, v in state_map.items()}
    prev_s = regime_to_state.get(prev_regime)
    curr_s = regime_to_state.get(curr_regime)
    if prev_s is None or curr_s is None:
        return 0.5
    try:
        return float(_persistence_transmat[int(prev_s), int(curr_s)])
    except (ValueError, TypeError, IndexError):
        return 0.5


def _smooth_regime(raw_regime):
    """
    Rolling majority vote over last SMOOTH_WINDOW raw predictions.
    Prevents regime flipping every candle on noisy market data.
    Returns smoothed regime string.
    """
    global _smooth_buffer
    _smooth_buffer.append(raw_regime)
    if len(_smooth_buffer) > SMOOTH_WINDOW:
        _smooth_buffer = _smooth_buffer[-SMOOTH_WINDOW:]
    # Majority vote — Counter now imported at module level (BUG-01 FIX)
    return Counter(_smooth_buffer).most_common(1)[0][0]


def _predict_xgb(latest_z, latest_raw, state_map=None):
    """
    Two-stage regime prediction pipeline.

    Stage 1 — REVERSAL binary pre-filter (reversal_detector.py):
        Runs first.  If P(REVERSAL) >= REVERSAL_FIRE_THRESHOLD → return
        REVERSAL immediately.  Skips Stage 2 entirely for this candle.
        WHY: A dedicated binary model catches more REVERSAL candles than
        a 5-class model because it has full capacity for one distinction.

    Stage 2 — 5-class XGBoost (main model):
        0. Build feature row (z-score + persistence injection)
        1. Predict — get raw regime + full probability vector
        2. If confidence < UNCERTAIN_THRESHOLD → return UNCERTAIN
        3. If predicted BULL or BEAR AND P(BULL)−P(BEAR) gap < DIRECTIONAL_MIN_MARGIN
           → downgrade to LOW_VOL_RANGE (model is unsure about direction)
        4. Smooth via rolling majority vote over SMOOTH_WINDOW candles
        5. Update persistence tracker with smoothed regime
    """
    try:
        # ── STAGE 1 — REVERSAL binary pre-filter ─────────────────
        # Runs before any feature engineering or persistence injection.
        # Uses the z-scored latest row (same features it was trained on).
        # If it fires, we return REVERSAL immediately — no 5-class call needed.
        if _reversal_filter_loaded:
            try:
                import reversal_detector as _rd
                fired, rev_prob = _rd.is_reversal(latest_z)
                if fired:
                    # Update persistence with REVERSAL and return early
                    candles_since, dur_mean, prev_enc = _update_persistence(REGIME_REVERSAL)
                    _smooth_buffer.clear()   # hard reset — spike overrides smoothing history
                    log_prediction(REGIME_REVERSAL, rev_prob, latest_features=latest_raw)
                    print(f"[RegimeDetector] REVERSAL ← Stage1 pre-filter "
                          f"P={rev_prob:.2f} ≥ {_rd.REVERSAL_FIRE_THRESHOLD} "
                          f"[binary/XGB]")
                    tf = compute_transition_forecast(REGIME_REVERSAL)
                    return {
                        "regime":        REGIME_REVERSAL,
                        "confidence":    round(rev_prob, 3),
                        "probabilities": {"REVERSAL": round(rev_prob, 3),
                                          "_source": "reversal_prefilter"},
                        "mode":          "model+prefilter",
                        "guidance":      REGIME_GUIDANCE.get(REGIME_REVERSAL, ""),
                        "transition_forecast": tf if tf else {},
                        "persistence": {
                            "candles_since_regime_start": int(candles_since),
                            "regime_duration_mean":       round(dur_mean, 1),
                            "regime_transition_prob":     0.5,
                        },
                    }
            except Exception as _stage1_err:
                pass   # Stage 1 failure is non-fatal — fall through to Stage 2

        cols       = _trained_cols or ALL_FEATURE_COLS
        base_cols  = [c for c in cols if c not in PERSISTENCE_FEATURE_COLS]
        pers_cols  = [c for c in cols if c in PERSISTENCE_FEATURE_COLS]

        # ── Fill missing base cols ────────────────────────────────
        missing = [c for c in base_cols if c not in latest_z.columns]
        if missing:
            latest_z = latest_z.copy()
            for c in missing:
                latest_z[c] = 0.0
            if len(missing) > 5:
                print(f"[RegimeDetector] WARNING: {len(missing)} cols missing — retrain.")

        # ── Build persistence vector (live values from tracker) ──
        # Bug 4 fix: read current tracker state WITHOUT calling _update_persistence()
        # (which would increment the counter prematurely). Update is done once,
        # after the smoothed regime is known, further below.
        from labeler import ALL_REGIMES as _ALL_REGIMES
        regime_to_int = {r: i for i, r in enumerate(_ALL_REGIMES)}
        candles_since = float(_persistence_counter)
        dur_mean      = (float(np.mean(_persistence_completed_durs[-20:]))
                         if _persistence_completed_durs else 1.0)
        prev_enc      = float(regime_to_int.get(_persistence_prev_regime, 4)
                              if _persistence_prev_regime else 4)
        trans_prob = _get_transition_prob(
            _persistence_prev_regime,
            _persistence_last_regime or "LOW_VOL_RANGE",
            state_map,
        )

        pers_vals = {
            "candles_since_regime_start": candles_since,
            "regime_duration_mean":       dur_mean,
            "previous_regime_encoded":    prev_enc,
            "regime_transition_prob":     trans_prob,
        }

        # ── Assemble full feature row ─────────────────────────────
        row = latest_z[base_cols].copy()
        for pc in pers_cols:
            row[pc] = pers_vals.get(pc, 0.0)

        X = row[cols].values.reshape(1, -1)

        # ── Predict ───────────────────────────────────────────────
        pred_enc = _model.predict(X)[0]
        proba    = _model.predict_proba(X)[0]
        conf     = float(np.max(proba))

        regime_map = {0: 'REVERSAL', 1: 'BULL_TREND',
                      2: 'BEAR_TREND', 3: 'COMPRESSION', 4: 'LOW_VOL_RANGE'}
        if _le is not None:
            raw_regime    = _le.inverse_transform([pred_enc])[0]
            probabilities = {cls: round(float(proba[i]), 3)
                             for i, cls in enumerate(_le.classes_)}
        else:
            raw_regime    = regime_map.get(int(pred_enc), 'LOW_VOL_RANGE')
            probabilities = None

        # ── Step 9: UNCERTAIN threshold ───────────────────────────
        if conf < UNCERTAIN_THRESHOLD:
            print(f"[RegimeDetector] UNCERTAIN (raw={raw_regime}, conf={conf:.2f} "
                  f"< {UNCERTAIN_THRESHOLD}) — skipping trade")
            return {
                "regime":        "UNCERTAIN",
                "confidence":    round(conf, 3),
                "probabilities": probabilities,
                "mode":          "model",
                "guidance":      "Confidence too low. No trade.",
                "persistence":   {"candles_since_regime_start": int(candles_since),
                                  "regime_duration_mean": round(dur_mean, 1),
                                  "regime_transition_prob": round(trans_prob, 3)},
            }

        # ── Directional margin gate ───────────────────────────────
        # If the model predicted BULL or BEAR but the probability gap between
        # them is below DIRECTIONAL_MIN_MARGIN, the model doesn't have a clear
        # directional opinion — it's guessing.  Downgrade to LOW_VOL_RANGE.
        #
        # This is the inference-time complement to labeler.py's DI_MARGIN_MIN fix.
        # Together they eliminate both bad training labels AND bad live predictions
        # on borderline directional candles.
        dir_gap         = None   # filled below if prediction is directional
        downgrade_fired = False  # True if margin gate converted BULL/BEAR → LOW_VOL
        if raw_regime in (REGIME_BULL_TREND, REGIME_BEAR_TREND) and probabilities:
            p_bull  = probabilities.get(REGIME_BULL_TREND, 0.0)
            p_bear  = probabilities.get(REGIME_BEAR_TREND, 0.0)
            dir_gap = round(abs(p_bull - p_bear), 3)
            if dir_gap < DIRECTIONAL_MIN_MARGIN:
                print(f"[RegimeDetector] Directional margin too narrow "
                      f"(P_bull={p_bull:.2f} P_bear={p_bear:.2f} gap={dir_gap:.2f} "
                      f"< {DIRECTIONAL_MIN_MARGIN}) → LOW_VOL_RANGE")
                raw_regime      = REGIME_LOW_VOL_RANGE
                downgrade_fired = True

        # ── Step 8: smooth via majority vote ──────────────────────
        # Snapshot buffer BEFORE updating it — used for signal quality reporting
        pre_smooth_buffer = list(_smooth_buffer)   # copy before _smooth_regime mutates it
        smoothed_regime   = _smooth_regime(raw_regime)
        vote_tally        = Counter(pre_smooth_buffer + [raw_regime])
        regime_votes      = vote_tally.get(smoothed_regime, 1)
        vote_unanimity    = regime_votes / max(len(pre_smooth_buffer) + 1, 1)

        # Update tracker with smoothed regime
        candles_since, dur_mean, prev_enc = _update_persistence(smoothed_regime)

        print(f"[RegimeDetector] {smoothed_regime} "
              f"(raw={raw_regime}, conf={conf:.2f}) "
              f"vote={regime_votes}/{len(pre_smooth_buffer)+1} "
              f"| age={int(candles_since)}c "
              f"| trans_p={trans_prob:.2f} "
              f"[XGB/GMM-HMM v5]")

        # Log to drift_log.json — auto_retrainer reads this every 4h
        log_prediction(smoothed_regime, conf, latest_features=latest_raw)

        result = {
            "regime":        smoothed_regime,
            "confidence":    round(conf, 3),
            "probabilities": probabilities,
            "mode":          "model",
            "guidance":      REGIME_GUIDANCE.get(smoothed_regime, ""),
            "persistence": {
                "candles_since_regime_start": int(candles_since),
                "regime_duration_mean":       round(dur_mean, 1),
                "regime_transition_prob":     round(trans_prob, 3),
            },
            # ── Signal quality metadata ────────────────────────────
            # These fields expose XGBoost's internal decision process
            # to Claude so it can calibrate how much to trust the call.
            "signal_quality": {
                "raw_regime":       raw_regime,          # pre-smoothing prediction
                "smoothed_regime":  smoothed_regime,     # post-majority-vote
                "regime_changed":   raw_regime != smoothed_regime,   # is model flip-flopping?
                "vote_tally":       dict(vote_tally),    # {regime: count} over last 5 candles
                "vote_unanimity":   round(vote_unanimity, 2),  # 1.0=all same, 0.4=minority win
                "smooth_window":    SMOOTH_WINDOW,
                "directional_gap":  dir_gap,             # |P(BULL)-P(BEAR)|, None if not directional
                "downgrade_fired":  downgrade_fired,     # True if margin gate converted BULL/BEAR→LOW_VOL
                "uncertain_threshold": UNCERTAIN_THRESHOLD,
                "directional_margin":  DIRECTIONAL_MIN_MARGIN,
            },
        }

        # ── HMM transition forecast ───────────────────────────────
        # Compute BEFORE session enrichment so it's available to all
        # downstream consumers (router, prompt formatter, logger).
        # Non-fatal — if transmat not loaded, returns {} silently.
        tf = compute_transition_forecast(smoothed_regime)
        if tf:
            result["transition_forecast"] = tf

        # ── Enrich with session-timing intelligence ───────────────
        # Adds: typical_dur_min, elapsed_min, age_status, survival_prob,
        #       adaptive_threshold — all per (regime, session) from profiles.
        # Non-fatal: if profiler not built yet, result is returned as-is.
        try:
            from session_profiler import enrich_regime_result, compute_norm_vol
            import datetime as _dt
            _hour   = _dt.datetime.utcnow().hour
            _sess_h = {range(19,24): "Asian", range(0,2): "Asian",
                       range(2,7): "London", range(7,17): "NY"}
            _session = next((s for r, s in _sess_h.items() if _hour in r), "Dead")
            norm_vol = compute_norm_vol(features_row=latest_raw)
            result   = enrich_regime_result(result, _session,
                                            signal=None, norm_vol=norm_vol)
        except Exception:
            pass   # profiler optional — never crash predict()

        return result

    except Exception as e:
        print(f"[RegimeDetector] XGBoost predict failed: {e}. Using fallback.")
        return _predict_fallback(latest_raw)


def log_prediction(regime, confidence, latest_features=None):
    """
    Logs each XGBoost prediction to drift_log.json.
    Called from _predict_xgb() after every successful prediction.

    Stores:
        timestamp       — when prediction was made
        regime          — smoothed XGBoost regime
        confidence      — max class probability
        labeler_agrees  — whether labeler.py threshold rules agree
                          (used by auto_retrainer divergence trigger)

    MEM-03 FIX: Buffered batch writes.
    Readings are accumulated in _drift_log_buffer and flushed to disk
    every DRIFT_LOG_FLUSH_EVERY calls (~30 min at 5-min cycles).
    This cuts disk I/O from ~24 full read+write cycles per hour to ~4.
    auto_retrainer.py reads this file every 4 hours, so the batch lag
    has zero effect on retraining triggers.
    """
    global _drift_log_buffer
    try:
        # Compare against labeler fallback to detect divergence
        labeler_agrees = None
        if latest_features is not None:
            try:
                labeler_regime = label_candles(latest_features).iloc[0]
                labeler_agrees = (labeler_regime == regime)
            except Exception:
                pass

        reading = {
            "ts":             datetime.now().strftime('%Y-%m-%d %H:%M'),
            "regime":         regime,
            "confidence":     round(float(confidence), 3) if confidence else None,
            "labeler_agrees": labeler_agrees,
        }

        _drift_log_buffer.append(reading)

        # Only flush to disk every DRIFT_LOG_FLUSH_EVERY readings (MEM-03 FIX)
        if len(_drift_log_buffer) < _DRIFT_LOG_FLUSH_EVERY:
            return   # buffer not full yet — skip disk I/O this call

        # ── Flush: merge buffer with on-disk log ──────────────────────
        log = {"readings": []}
        if os.path.exists(DRIFT_LOG_PATH):
            try:
                with open(DRIFT_LOG_PATH, 'r') as f:
                    log = json.load(f)
            except Exception:
                pass

        log.setdefault("readings", []).extend(_drift_log_buffer)
        # Rolling: keep last DRIFT_LOG_MAX_READINGS
        log["readings"] = log["readings"][-DRIFT_LOG_MAX_READINGS:]

        with open(DRIFT_LOG_PATH, 'w') as f:
            json.dump(log, f)   # compact — no indent (performance)

        _drift_log_buffer = []   # clear buffer after successful flush

    except Exception:
        pass   # Never let logging crash the prediction path


def reload_if_needed():
    """
    Called at the top of every main_bot trading cycle.
    Checks reload_flag.json (written by auto_retrainer after retraining).
    If flag is present, reloads all model files in-process and clears the flag.

    Cost: one os.path.exists() check per cycle (~0ms). Flag file is only
    written when a retrain actually completed, so this is almost always a no-op.
    """
    global _model, _le, _mode, _trained_cols, _persistence_transmat

    if not os.path.exists(RELOAD_FLAG_PATH):
        return   # no-op — most cycles take this path

    try:
        with open(RELOAD_FLAG_PATH, 'r') as f:
            flag = json.load(f)

        if not flag.get('reload'):
            return

        retrained_at = flag.get('retrained_at', 'unknown')
        new_accuracy = flag.get('new_accuracy')
        acc_str      = f"{new_accuracy:.1%}" if new_accuracy else "N/A"

        print(f"\n[RegimeDetector] RELOAD TRIGGERED by auto_retrainer.")
        print(f"[RegimeDetector] Retrained at: {retrained_at} | New accuracy: {acc_str}")

        # Reset singleton so _load_model() runs fresh
        _model        = None
        _le           = None
        _mode         = None
        _trained_cols = None
        _persistence_transmat = None

        # Clear smoothing buffer — stale smoothing from old model is wrong
        global _smooth_buffer
        _smooth_buffer = []

        _load_model()

        # Clear the flag so we don't reload every cycle
        os.remove(RELOAD_FLAG_PATH)
        print(f"[RegimeDetector] Model reloaded successfully. Reload flag cleared.\n")

    except Exception as e:
        print(f"[RegimeDetector] reload_if_needed error: {e}")
        # Try to remove a potentially corrupt flag file
        try:
            os.remove(RELOAD_FLAG_PATH)
        except Exception:
            pass


def _predict_fallback(latest):
    """Threshold-based fallback — ADX/ATR rules (labeler.py logic)."""
    labels = label_candles(latest)
    regime = labels.iloc[0]
    print(f"[RegimeDetector] {regime} [fallback thresholds]")
    return {
        "regime":        regime,
        "confidence":    None,
        "probabilities": None,
        "mode":          "fallback",
        "guidance":      REGIME_GUIDANCE.get(regime, ""),
    }


def _safe_default():
    return {
        "regime":        REGIME_LOW_VOL_RANGE,
        "confidence":    None,
        "probabilities": None,
        "mode":          "error_fallback",
        "guidance":      REGIME_GUIDANCE[REGIME_LOW_VOL_RANGE],
    }


def format_for_prompt(regime_result):
    """Formats regime result for injection into the AI prompt."""
    regime     = regime_result.get("regime",     REGIME_LOW_VOL_RANGE)
    confidence = regime_result.get("confidence")
    mode       = regime_result.get("mode",       "fallback")
    guidance   = regime_result.get("guidance",   "")
    probs      = regime_result.get("probabilities")

    conf_str   = f" | Confidence: {confidence:.0%}" if confidence else ""

    # Probability bar across all regimes
    prob_str = ""
    if probs:
        prob_bar = "  ".join(
            f"{r[:8]}: {probs.get(r, 0):.0%}"
            for r in [REGIME_BULL_TREND, REGIME_BEAR_TREND,
                      REGIME_LOW_VOL_RANGE, REGIME_COMPRESSION, REGIME_REVERSAL]
        )
        prob_str = f"\nProbabilities: {prob_bar}"

    return (
        f"{'='*55}\n"
        f"REGIME DETECTOR{conf_str} | Mode: {mode}\n"
        f"{guidance}"
        f"{prob_str}\n"
        f"{'='*55}\n"
    )


def format_model_signal_quality_for_prompt(regime_result: dict) -> str:
    """
    Exposes XGBoost's internal decision process to Claude.

    Covers three things that were previously computed and silently dropped:

    1. SMOOTH BUFFER VOTE TALLY
       The model runs a 5-candle majority vote before committing to a regime.
       "BULL_TREND 5/5" is very different from "BULL_TREND 3/5".
       A 3/5 win means the model changed its mind twice in the last 5 candles —
       the regime is potentially transitioning, not firmly established.

    2. DIRECTIONAL GAP
       |P(BULL) - P(BEAR)| for directional calls.
       Gap=0.83 → very decisive. Gap=0.17 → barely above threshold.
       Claude should be more cautious sizing a directional trade when the
       model barely resolved the direction.

    3. DOWNGRADE FLAG
       If the directional margin gate fired (BULL/BEAR → LOW_VOL_RANGE),
       Claude sees LOW_VOL_RANGE — but that means "model can't confirm direction",
       NOT a normal ranging market. This is a completely different situation
       and Claude must not treat it as a standard range.

    Returns "" if signal_quality not in result (pre-model fallback mode).
    """
    sq = regime_result.get("signal_quality")
    if not sq:
        return ""

    regime    = regime_result.get("regime", "?")
    conf      = regime_result.get("confidence")
    conf_s    = f"{conf:.0%}" if conf else "N/A"

    raw       = sq.get("raw_regime",      regime)
    smoothed  = sq.get("smoothed_regime", regime)
    changed   = sq.get("regime_changed",  False)
    tally     = sq.get("vote_tally",      {})
    unanimity = sq.get("vote_unanimity",  1.0)
    window    = sq.get("smooth_window",   5)
    dir_gap   = sq.get("directional_gap")
    downgrade = sq.get("downgrade_fired", False)

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  MODEL SIGNAL QUALITY — XGBOOST DECISION TRANSPARENCY  ║",
        "╚══════════════════════════════════════════════════════════╝",
    ]

    # ── Vote tally ─────────────────────────────────────────────
    total_votes = sum(tally.values()) if tally else window
    tally_str   = "  ".join(
        f"{r}:{n}/{total_votes}"
        for r, n in sorted(tally.items(), key=lambda x: x[1], reverse=True)
    ) if tally else "N/A"

    winning_votes = tally.get(smoothed, total_votes)

    # Stability classification
    if   unanimity >= 0.80: stability = "STABLE ✅"
    elif unanimity >= 0.60: stability = "FORMING 🟡"
    else:                   stability = "UNSTABLE ⚠"

    lines.append(f"Vote Tally     : {tally_str}")
    lines.append(f"Regime Support : {winning_votes}/{total_votes} candles voted {smoothed}")
    lines.append(f"Stability      : {stability}  ({unanimity:.0%} unanimity over last {window} candles)")

    # Explain what changed vs raw
    if changed:
        lines.append(f"Smoothing Note : Raw prediction was {raw}, smoothed to {smoothed} by majority vote.")
        lines.append(f"                 ⚠ Model changed its prediction this candle — regime may be transitioning.")
    else:
        lines.append(f"Smoothing Note : Raw = Smoothed = {smoothed}  (consistent this candle)")

    # ── Directional gap ────────────────────────────────────────
    lines.append("")
    if dir_gap is not None:
        probs     = regime_result.get("probabilities", {})
        p_bull    = probs.get(REGIME_BULL_TREND, 0.0)
        p_bear    = probs.get(REGIME_BEAR_TREND, 0.0)
        margin_pct= sq.get("directional_margin", 0.15)

        if   dir_gap >= 0.50: gap_label = "DECISIVE ✅"
        elif dir_gap >= 0.25: gap_label = "MODERATE 🟡"
        else:                 gap_label = "BORDERLINE ⚠"

        lines.append(f"Directional Gap: {gap_label}  gap={dir_gap:.2f}  "
                     f"(P_bull={p_bull:.0%} vs P_bear={p_bear:.0%}, "
                     f"threshold={margin_pct})")
        lines.append(
            f"                 A gap of {dir_gap:.2f} means the model is "
            + ("very sure about direction — trend entries have strong model support."
               if dir_gap >= 0.50 else
               "moderately sure — trend entries carry some directional uncertainty."
               if dir_gap >= 0.25 else
               f"barely above the {margin_pct} threshold — treat as LOW CONVICTION directional call.")
        )
    else:
        lines.append("Directional Gap: N/A  (regime is not directional)")

    # ── Downgrade flag ─────────────────────────────────────────
    lines.append("")
    if downgrade:
        lines.append("⚠ DIRECTIONAL DOWNGRADE FIRED")
        lines.append(f"  The model's raw prediction was {raw} but the directional")
        lines.append(f"  confidence gap was below the required {sq.get('directional_margin',0.15)} threshold.")
        lines.append(f"  This regime label ({regime}) means 'direction unconfirmed',")
        lines.append(f"  NOT a standard ranging market. Do NOT apply mean-reversion logic.")
        lines.append(f"  CORRECT interpretation: conflicting directional signals — WAIT for clarity.")
    else:
        lines.append(f"Downgrade Gate : Not fired  (regime label is authentic)")

    # ── Confidence context ─────────────────────────────────────
    lines.append("")
    uncertain_thr = sq.get("uncertain_threshold", 0.45)
    margin_above  = round(conf - uncertain_thr, 2) if conf else None
    if conf and margin_above is not None:
        if   margin_above >= 0.30: conf_label = "HIGH MARGIN above uncertainty threshold ✅"
        elif margin_above >= 0.10: conf_label = "MODERATE MARGIN above uncertainty threshold 🟡"
        else:                      conf_label = "LOW MARGIN above uncertainty threshold ⚠"
        lines.append(f"Confidence     : {conf_s}  — {conf_label}")
        lines.append(f"                 ({conf:.0%} conf vs {uncertain_thr:.0%} uncertain threshold, "
                     f"margin=+{margin_above:.0%})")

    lines.append("══════════════════════════════════════════════════════════")
    return "\n".join(lines)


def compute_transition_forecast(current_regime: str) -> dict:
    """
    Reads one row of the HMM transition matrix and returns the probability
    distribution over next regimes given the current regime.

    HOW IT WORKS:
        The GMM-HMM learned a 5×5 Markov transition matrix during training.
        Each row i contains [P(→state_0), P(→state_1), ... P(→state_4)]
        given the model is currently in state i.

        This function:
          1. Finds the HMM state index for current_regime via state_map
          2. Looks up that row in _persistence_transmat
          3. Maps each column index → regime name via state_map
          4. Returns {regime_name: probability} sorted descending

    Returns:
        dict: {"LOW_VOL_RANGE": 0.41, "BULL_TREND": 0.31, ...}
        Returns {} if transmat or state_map not loaded yet (pre-training).

    Example outputs for gold (approximate — depends on training data):
        After REVERSAL:  LOW_VOL_RANGE 41%, BULL_TREND 31%, BEAR_TREND 28%
        After BULL_TREND: BULL_TREND 72%, LOW_VOL_RANGE 18%, BEAR_TREND 7%
        After COMPRESSION: BULL_TREND 38%, BEAR_TREND 35%, REVERSAL 16%
    """
    if _persistence_transmat is None or _state_map is None:
        return {}

    # state_map stored as {"0": "COMPRESSION", "2": "BULL_TREND", ...} from JSON
    # normalise to {int: str}
    try:
        sm = {int(k): v for k, v in _state_map.items()}
    except (TypeError, ValueError):
        sm = _state_map

    # Find HMM state index for current_regime
    regime_to_state = {v: k for k, v in sm.items()}
    state_idx = regime_to_state.get(current_regime)
    if state_idx is None:
        return {}

    row = _persistence_transmat[state_idx]   # shape (n_states,)

    # Map column index → regime name → probability
    forecast = {}
    for col_idx, prob in enumerate(row):
        regime_name = sm.get(col_idx)
        if regime_name:
            # Accumulate in case two states map to same regime
            forecast[regime_name] = forecast.get(regime_name, 0.0) + float(prob)

    # Sort by probability descending
    return dict(sorted(forecast.items(), key=lambda x: x[1], reverse=True))


def format_transition_forecast_for_prompt(regime_result: dict) -> str:
    """
    Formats the HMM transition matrix row for a regime into a Claude prompt block.

    Claude gets:
      - The ranked probability distribution over next regimes
      - A visual bar for quick scanning
      - A one-sentence natural-language interpretation tailored to the
        current regime and distribution shape
      - A directional bias score so Claude can weight its analysis

    Called from main_bot.py alongside format_for_prompt() and
    format_session_profile_for_prompt().

    Returns "" if transition forecast not available (model not trained yet).
    """
    tf = regime_result.get("transition_forecast")
    if not tf:
        return ""

    current = regime_result.get("regime", "?")
    conf    = regime_result.get("confidence")
    conf_s  = f" (conf {conf:.0%})" if conf else ""

    # ── Build probability rows ────────────────────────────────────
    regime_labels = {
        REGIME_BULL_TREND:    "BULL_TREND  ",
        REGIME_BEAR_TREND:    "BEAR_TREND  ",
        REGIME_LOW_VOL_RANGE: "LOW_VOL_RANGE",
        REGIME_COMPRESSION:   "COMPRESSION ",
        REGIME_REVERSAL:      "REVERSAL    ",
    }
    MAX_BAR = 28
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  HMM TRANSITION FORECAST — WHAT TYPICALLY COMES NEXT   ║",
        "╚══════════════════════════════════════════════════════════╝",
        f"Current Regime : {current}{conf_s}",
        f"Source         : GMM-HMM transition matrix (learned from full training history)",
        "",
        "Next-Regime Probabilities:",
    ]

    for regime_name, prob in tf.items():
        label   = regime_labels.get(regime_name, f"{regime_name:<13}")
        bar_len = int(prob * MAX_BAR)
        bar     = "█" * bar_len + "░" * (MAX_BAR - bar_len)
        pct     = f"{prob:.0%}"
        lines.append(f"  {label}  {pct:>4}  {bar}")

    # ── Natural-language interpretation ──────────────────────────
    # Derive directional vs non-directional probability
    p_bull  = tf.get(REGIME_BULL_TREND,    0.0)
    p_bear  = tf.get(REGIME_BEAR_TREND,    0.0)
    p_quiet = tf.get(REGIME_LOW_VOL_RANGE, 0.0)
    p_comp  = tf.get(REGIME_COMPRESSION,   0.0)
    p_rev   = tf.get(REGIME_REVERSAL,      0.0)
    p_dir   = p_bull + p_bear          # total directional probability
    p_ndir  = p_quiet + p_comp + p_rev  # total non-directional probability

    # Top next regime and its probability
    top_next, top_prob = next(iter(tf.items()))

    interp = _build_transition_interpretation(
        current, top_next, top_prob, p_bull, p_bear,
        p_quiet, p_comp, p_dir, p_ndir)

    lines.append("")
    lines.append(f"Interpretation : {interp}")

    # ── Directional bias score ─────────────────────────────────
    # Net directional bias: positive = bullish lean, negative = bearish lean
    # Useful for Claude to weight trade direction probability
    bias_score = round(p_bull - p_bear, 2)
    bias_label = ("BULLISH LEAN"   if bias_score >  0.10 else
                  "BEARISH LEAN"   if bias_score < -0.10 else
                  "NEUTRAL")
    lines.append(f"Directional    : {bias_label}  "
                 f"(P_bull={p_bull:.0%} vs P_bear={p_bear:.0%}, "
                 f"net={bias_score:+.2f})")
    lines.append(f"Non-Directional: {p_ndir:.0%} probability of quiet/compression/spike next")
    lines.append("══════════════════════════════════════════════════════════")

    return "\n".join(lines)


def _build_transition_interpretation(
        current, top_next, top_prob,
        p_bull, p_bear, p_quiet, p_comp,
        p_dir, p_ndir) -> str:
    """
    Returns a one-to-two sentence plain-English interpretation of the
    transition matrix row for the current regime.

    Covers all 5 current-regime cases with distribution-shape logic so
    Claude gets a useful heuristic, not just the numbers.
    """
    if current == REGIME_REVERSAL:
        if p_quiet >= 0.35:
            return (
                f"After a spike/news event, gold historically consolidates quietly "
                f"{p_quiet:.0%} of the time — the move exhausts itself. "
                f"Directional continuation is less likely (only {p_dir:.0%} combined). "
                f"Wait for the dust to settle before committing to a direction."
            )
        elif p_dir >= 0.55:
            dom = "bullish" if p_bull > p_bear else "bearish"
            return (
                f"Unusually, this spike historically resolves directionally "
                f"({p_dir:.0%} directional). "
                f"Slight {dom} bias (P_bull={p_bull:.0%} vs P_bear={p_bear:.0%}). "
                f"Monitor for a confirmed break before entering."
            )
        else:
            return (
                f"Uncertain post-spike environment — probabilities spread across "
                f"multiple regimes. No strong statistical edge in either direction. "
                f"Reduce size and widen levels."
            )

    elif current == REGIME_BULL_TREND:
        if top_next == REGIME_BULL_TREND and top_prob >= 0.65:
            return (
                f"Strong trend persistence — {top_prob:.0%} probability the bull trend "
                f"continues next candle. Momentum setups and trend-continuation entries "
                f"carry a statistical edge. Trail stops rather than targeting fixed TP."
            )
        elif p_quiet >= 0.25:
            return (
                f"Bull trend has {top_prob:.0%} continuation probability but "
                f"{p_quiet:.0%} chance of transitioning to a quiet range. "
                f"The move may be maturing. Consider tightening TP on existing positions."
            )
        else:
            return (
                f"Bull trend continuation likely ({top_prob:.0%}) but transition "
                f"probabilities are spread — treat as a MATURING trend. "
                f"Favour existing longs over new entries."
            )

    elif current == REGIME_BEAR_TREND:
        if top_next == REGIME_BEAR_TREND and top_prob >= 0.65:
            return (
                f"Strong trend persistence — {top_prob:.0%} probability the bear trend "
                f"continues next candle. Short-continuation and momentum entries "
                f"carry a statistical edge. Trail stops rather than targeting fixed TP."
            )
        elif p_quiet >= 0.25:
            return (
                f"Bear trend has {top_prob:.0%} continuation probability but "
                f"{p_quiet:.0%} chance of transitioning to a quiet range. "
                f"The move may be maturing. Consider tightening TP on existing shorts."
            )
        else:
            return (
                f"Bear trend continuation likely ({top_prob:.0%}) but transition "
                f"probabilities are spread — treat as a MATURING trend. "
                f"Favour existing shorts over new entries."
            )

    elif current == REGIME_COMPRESSION:
        dom_dir = "bullish" if p_bull > p_bear else "bearish"
        other   = p_bear if p_bull > p_bear else p_bull
        dom_p   = max(p_bull, p_bear)
        if p_dir >= 0.55:
            return (
                f"Compression historically resolves directionally {p_dir:.0%} of the time "
                f"with a {dom_dir} lean (P={dom_p:.0%} vs {other:.0%}). "
                f"Wait for an expansion candle to confirm breakout direction before entering. "
                f"Do NOT pre-position — the HMM bias is useful only AFTER confirmation."
            )
        elif p_rev >= 0.18:
            return (
                f"High reversal/spike probability post-compression ({p_rev:.0%}). "
                f"A breakout here carries elevated news-spike risk. "
                f"Use minimum size on breakout entries and keep SL wide."
            )
        else:
            return (
                f"Compression with no clear directional lean — probabilities are spread. "
                f"Wait for price to declare. This squeeze could resolve in either direction "
                f"with roughly equal probability."
            )

    elif current == REGIME_LOW_VOL_RANGE:
        if top_next == REGIME_LOW_VOL_RANGE and top_prob >= 0.55:
            return (
                f"Quiet range tends to persist ({top_prob:.0%} continuation). "
                f"Mean-reversion setups have the highest statistical edge here. "
                f"Breakout trades carry low probability — the range is not coiling yet."
            )
        elif p_comp >= 0.20:
            return (
                f"Quiet range with {p_comp:.0%} probability of transitioning to "
                f"COMPRESSION (pre-breakout squeeze). Watch ATR and BB width — "
                f"if both contract further, a breakout setup is forming."
            )
        elif p_dir >= 0.40:
            dom_dir = "bullish" if p_bull > p_bear else "bearish"
            return (
                f"Quiet range but with elevated directional transition probability "
                f"({p_dir:.0%}), leaning {dom_dir}. "
                f"A catalyst could initiate a trend from here. "
                f"Stay nimble — mean-reversion is still the primary play but "
                f"watch for a break."
            )
        else:
            return (
                f"Range market tends to stay quiet ({top_prob:.0%} continuation). "
                f"No strong edge for directional trades. "
                f"Favour range-bound strategies and tighter targets."
            )

    # Fallback for any unexpected regime
    return (
        f"Most likely next regime: {top_next} ({top_prob:.0%}). "
        f"Directional probability: {p_dir:.0%}. "
        f"Non-directional: {p_ndir:.0%}."
    )
