"""
meta_labeller.py  — XU-L Meta-Labelling Engine
================================================
Sits on top of the XGBoost regime/direction model (primary model) and
answers a different question:

    PRIMARY MODEL asks: "What regime is the market in?"
    META MODEL asks:    "Given regime + context, will THIS trade actually win?"

Architecture:
    Step 1 — Primary model emits signal + confidence + regime
    Step 2 — Triple Barrier labelling with ATR-dynamic barriers per regime
    Step 3 — Fractional differentiation (preserves price memory)
    Step 4 — Uniqueness-weighted samples (sequential bootstrapping)
    Step 5 — Build advanced meta-features:
                xgb_confidence, prediction_entropy, regime,
                RVOL, time-to-event (bars since regime shift),
                distance to S/R, session, news gate status
    Step 6 — Train meta-XGBoost with Purged + Embargoed CV
    Step 7 — Live predict: output meta_probability + Kelly size
    Step 8 — SHAP explanation → structured Wisdom lesson

Public API (called from main_bot.py / regime_router.py):
    predict_meta(primary_signal, regime_result, market_features)
        → {
            "meta_prob":        float,   # 0.0–1.0 win probability
            "kelly_size":       float,   # position size multiplier (0.0–1.0)
            "should_trade":     bool,    # meta_prob >= META_MIN_THRESHOLD
            "shap_reason":      str,     # human-readable block/allow reason
            "mode":             str,     # "model" or "rule_based"
          }

    train(primary_signals_df, ohlcv_df, regime_labels)
        → saves model to data/model/

    generate_wisdom_lesson(ticket, meta_prob, actual_outcome, shap_values)
        → str   # injected into Wisdom rebuild
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'regime_detector'))

from labeler import (
    REGIME_REVERSAL, REGIME_BULL_TREND, REGIME_BEAR_TREND,
    REGIME_COMPRESSION, REGIME_LOW_VOL_RANGE, ALL_REGIMES,
)
from feature_engineer import compute_atr, compute_rsi, compute_volume_zscore

_root_ml = os.path.normpath(os.path.join(current_dir, '..', '..'))
if _root_ml not in sys.path: sys.path.insert(0, _root_ml)
from paths import (META_MODEL_PATH, META_MODEL_META_PATH as META_META_PATH,
                   META_WISDOM_LOG_PATH as WISDOM_LOG_PATH,
                   META_MODEL as MODEL_DIR, create_all_dirs as _cad_ml)
_cad_ml()

# ── Singleton ────────────────────────────────────────────────────
_meta_model    = None
_meta_mode     = None   # "model" | "rule_based"
_meta_cols     = None
_meta_rr_stats = {"mean_win": 1.5, "mean_loss": 1.0}  # updated from history

# ── Configuration ────────────────────────────────────────────────

# Triple Barrier: multipliers × ATR per regime
# (tp_mult, sl_mult, max_bars)
REGIME_BARRIER_PARAMS = {
    REGIME_BULL_TREND:    {"tp": 2.0, "sl": 1.0, "bars": 24},
    REGIME_BEAR_TREND:    {"tp": 2.0, "sl": 1.0, "bars": 24},
    REGIME_LOW_VOL_RANGE: {"tp": 1.0, "sl": 0.8, "bars": 16},
    REGIME_COMPRESSION:   {"tp": 1.2, "sl": 0.6, "bars": 12},
    REGIME_REVERSAL:      {"tp": 0.8, "sl": 1.5, "bars": 8},
}
DEFAULT_BARRIER = {"tp": 1.5, "sl": 1.0, "bars": 20}

# Meta model thresholds
import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from master_controls import META_MIN_THRESHOLD  # tunable in master_controls.py
META_KELLY_CAP      = 1.0    # never exceed full size
META_KELLY_FRACTION = 0.5    # half-Kelly for safety (standard practice)

# Fractional differentiation
FRAC_DIFF_D     = 0.4    # d=0.4 balances stationarity vs memory
FRAC_DIFF_THRES = 1e-4   # weight threshold — truncate tiny weights


# ================================================================
# STEP 1 — FRACTIONAL DIFFERENTIATION
# Preserves long-term price memory while achieving stationarity.
# López de Prado (2018), Chapter 5.
# ================================================================

def _frac_diff_weights(d: float, size: int, threshold: float = FRAC_DIFF_THRES):
    """
    Compute binomial series weights for fractional differentiation.
    w_k = product_{j=0}^{k-1} (d - j) / (j + 1)
    Truncated when |w_k| < threshold.
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1])   # oldest to newest


def fractional_diff(series: pd.Series, d: float = FRAC_DIFF_D,
                    threshold: float = FRAC_DIFF_THRES) -> pd.Series:
    """
    Apply fractional differentiation to a price series.

    d=0 → original series (fully non-stationary, full memory)
    d=1 → standard returns (fully stationary, zero memory)
    d=0.4 → stationary enough for XGBoost, retains ~60% of long-range memory

    This allows the meta-model to 'see' where price is relative to
    long-term S/R levels — something standard returns destroy entirely.
    """
    weights = _frac_diff_weights(d, len(series), threshold)
    w_len   = len(weights)
    result  = pd.Series(index=series.index, dtype=float)

    for i in range(w_len - 1, len(series)):
        window       = series.iloc[i - w_len + 1: i + 1].values
        result.iloc[i] = float(np.dot(weights, window))

    return result


# ================================================================
# STEP 2 — ATR-DYNAMIC TRIPLE BARRIER LABELLING
# ================================================================

def apply_triple_barrier(
    ohlcv_df:      pd.DataFrame,
    events_df:     pd.DataFrame,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Labels each trade event using the Triple Barrier method with
    ATR-dynamic barriers sized per the active regime.

    Args:
        ohlcv_df:      OHLCV data (must cover all event horizons)
        events_df:     DataFrame with columns:
                           timestamp  — bar index of signal
                           signal     — "long" or "short"
                           entry_price
                           xgb_confidence
                           regime
        regime_labels: pd.Series of regime labels, same index as ohlcv_df

    Returns:
        events_df with added columns:
            meta_label      — 1 (win) 0 (loss) based on which barrier hit first
            pnl_at_exit     — actual P&L in ATR units at barrier touch
            exit_reason     — "tp" | "sl" | "time"
            tp_price
            sl_price
            exit_bar_idx
            barrier_atr     — ATR value used for this trade's barriers
    """
    atr_series = compute_atr(ohlcv_df, period=14)
    close      = ohlcv_df['close']
    high       = ohlcv_df['high']
    low        = ohlcv_df['low']
    all_idx    = ohlcv_df.index.tolist()
    idx_map    = {ts: i for i, ts in enumerate(all_idx)}

    records = []

    for _, event in events_df.iterrows():
        ts       = event['timestamp']
        signal   = str(event.get('signal', 'long')).lower()
        entry    = float(event['entry_price'])
        regime   = str(event.get('regime', REGIME_LOW_VOL_RANGE))

        if ts not in idx_map:
            continue

        bar_i   = idx_map[ts]
        params  = REGIME_BARRIER_PARAMS.get(regime, DEFAULT_BARRIER)

        # ATR at signal bar — basis for dynamic barriers
        atr_val  = float(atr_series.iloc[bar_i]) if bar_i < len(atr_series) else 10.0
        if np.isnan(atr_val) or atr_val == 0:
            atr_val = 10.0

        tp_dist  = atr_val * params["tp"]
        sl_dist  = atr_val * params["sl"]
        max_bars = params["bars"]

        if signal == "long":
            tp_price = entry + tp_dist
            sl_price = entry - sl_dist
        else:
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist

        # Scan forward bar by bar
        meta_label  = 0
        pnl_at_exit = 0.0
        exit_reason = "time"
        exit_bar    = bar_i

        end_bar = min(bar_i + max_bars + 1, len(all_idx))
        for j in range(bar_i + 1, end_bar):
            h = float(high.iloc[j])
            l = float(low.iloc[j])
            c = float(close.iloc[j])

            if signal == "long":
                if h >= tp_price:
                    meta_label  = 1
                    pnl_at_exit = tp_dist / atr_val
                    exit_reason = "tp"
                    exit_bar    = j
                    break
                if l <= sl_price:
                    meta_label  = 0
                    pnl_at_exit = -(sl_dist / atr_val)
                    exit_reason = "sl"
                    exit_bar    = j
                    break
            else:
                if l <= tp_price:
                    meta_label  = 1
                    pnl_at_exit = tp_dist / atr_val
                    exit_reason = "tp"
                    exit_bar    = j
                    break
                if h >= sl_price:
                    meta_label  = 0
                    pnl_at_exit = -(sl_dist / atr_val)
                    exit_reason = "sl"
                    exit_bar    = j
                    break
        else:
            # Time barrier — label based on final P&L direction
            final_price = float(close.iloc[min(end_bar - 1, len(close) - 1)])
            raw_pnl     = (final_price - entry) if signal == "long" else (entry - final_price)
            pnl_at_exit = raw_pnl / atr_val
            meta_label  = 1 if raw_pnl > 0 else 0
            exit_reason = "time"
            exit_bar    = end_bar - 1

        # Continuous label: pnl_r in R-multiples, clipped to [-3, 3]
        # Used for magnitude-weighted training (Labeling Magnitude fix).
        # A 3R win has 3x the training weight of a 0.5R winner.
        pnl_r_continuous = round(max(-3.0, min(3.0, pnl_at_exit)), 4)

        rec = event.to_dict()
        rec.update({
            "meta_label":        meta_label,
            "pnl_at_exit":       round(pnl_at_exit, 4),
            "pnl_r_continuous":  pnl_r_continuous,   # NEW: magnitude label
            "exit_reason":       exit_reason,
            "tp_price":          round(tp_price, 4),
            "sl_price":          round(sl_price, 4),
            "exit_bar_idx":      exit_bar,
            "barrier_atr":       round(atr_val, 4),
        })
        records.append(rec)

    result = pd.DataFrame(records)
    if len(result) > 0:
        wins   = (result["meta_label"] == 1).sum()
        total  = len(result)
        print(f"[MetaLabeller] Triple Barrier: {total} events | "
              f"Win: {wins} ({wins/total*100:.1f}%) | "
              f"SL: {(result['exit_reason']=='sl').sum()} | "
              f"TP: {(result['exit_reason']=='tp').sum()} | "
              f"Time: {(result['exit_reason']=='time').sum()}")
    return result


# ================================================================
# STEP 3 — SAMPLE UNIQUENESS + SEQUENTIAL BOOTSTRAPPING
# Prevents meta-model overfitting to correlated event clusters.
# ================================================================

def compute_sample_uniqueness(
    events_df:  pd.DataFrame,
    ohlcv_df:   pd.DataFrame,
) -> pd.Series:
    """
    Computes uniqueness weight for each trade event.

    Each bar in ohlcv_df may be 'used' by multiple overlapping trade windows.
    A bar used by N concurrent trades contributes 1/N to each trade's uniqueness.
    A trade's uniqueness = average 1/N across all bars in its window.

    Trades during unique market moments get weight ~1.0.
    Trades in crowded windows (5 signals at same time) get weight ~0.2.

    This tells XGBoost: "don't overfit to those lucky 5-trade
    winning clusters — they're not that special."

    Returns:
        pd.Series of weights, indexed same as events_df.
    """
    if len(events_df) == 0:
        return pd.Series(dtype=float)

    all_idx   = ohlcv_df.index.tolist()
    idx_map   = {ts: i for i, ts in enumerate(all_idx)}
    n_bars    = len(all_idx)
    n_events  = len(events_df)

    # Build indicator matrix: indicator[i, j] = 1 if event i uses bar j
    indicator = np.zeros((n_events, n_bars), dtype=np.float32)

    for ei, (_, event) in enumerate(events_df.iterrows()):
        ts      = event['timestamp']
        regime  = str(event.get('regime', REGIME_LOW_VOL_RANGE))
        params  = REGIME_BARRIER_PARAMS.get(regime, DEFAULT_BARRIER)
        max_bars = params["bars"]

        if ts not in idx_map:
            continue
        bar_i = idx_map[ts]
        end_i = min(bar_i + max_bars + 1, n_bars)
        indicator[ei, bar_i:end_i] = 1.0

    # Count how many events use each bar: concurrency[j] = sum of column j
    concurrency = indicator.sum(axis=0)   # shape: (n_bars,)
    concurrency = np.where(concurrency == 0, 1.0, concurrency)

    # Each event's uniqueness = mean(1/concurrency) over its active bars
    weights = np.zeros(n_events, dtype=float)
    for ei in range(n_events):
        active_bars = np.where(indicator[ei] > 0)[0]
        if len(active_bars) == 0:
            weights[ei] = 1.0
        else:
            weights[ei] = float(np.mean(1.0 / concurrency[active_bars]))

    # Normalise to [0.05, 1.0] — never give zero weight
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        weights = 0.05 + 0.95 * (weights - w_min) / (w_max - w_min)
    else:
        weights = np.ones(n_events)

    return pd.Series(weights, index=events_df.index)


# ================================================================
# STEP 4 — META FEATURE ENGINEERING
# ================================================================

def build_meta_features(
    labelled_events: pd.DataFrame,
    ohlcv_df:        pd.DataFrame,
    regime_labels:   pd.Series,
    primary_probs:   pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Builds the full meta-feature matrix from labelled trade events.

    Features:
        Signal quality    : xgb_confidence, prediction_entropy, xgb_margin
        Regime            : regime_encoded, regime_confidence,
                            bars_since_regime_shift (time-to-event)
        Technical         : rsi_at_entry, atr_percentile, bb_width_at_entry
        Volume            : rvol (relative volume), volume_zscore_at_entry
        Structure         : distance_to_ema200, close_position, body_ratio
        Frac diff price   : frac_diff_close (memory-preserving price signal)
        Session           : is_asian, is_london, is_ny
        Trade quality     : barrier_atr (regime-adjusted barrier size)
        Temporal          : hour, day_of_week

    Args:
        labelled_events: output of apply_triple_barrier()
        ohlcv_df:        raw OHLCV
        regime_labels:   pd.Series of regime strings, indexed to ohlcv_df
        primary_probs:   optional — DataFrame with per-regime probabilities
                         from XGBoost (columns = regime names)
                         If None, entropy / margin features will be 0.5

    Returns:
        pd.DataFrame — one row per trade event, meta_label column included
    """
    all_idx  = ohlcv_df.index.tolist()
    idx_map  = {ts: i for i, ts in enumerate(all_idx)}
    close    = ohlcv_df['close']
    high     = ohlcv_df['high']
    low      = ohlcv_df['low']

    # Pre-compute series we'll need.
    #
    # ISSUE 2 FIX — Feature Leakage: shift(1) every rolling/EWM series.
    #
    # Why: at signal bar_i, accessing series.iloc[bar_i] WITHOUT shift gives
    # the value computed using bar_i's own close/volume — data the model
    # cannot legally see until the bar has fully closed and the NEXT bar opens.
    # shift(1) means series.iloc[bar_i] returns bar_{i-1}'s completed value:
    # the last fully known candle before entry. This eliminates look-ahead.
    #
    # NOT shifted (intentionally): body_ratio, close_position — these describe
    # the actual entry candle's OHLC structure which IS known at bar-close entry.
    atr_series  = compute_atr(ohlcv_df, period=14).shift(1)
    rsi_series  = compute_rsi(ohlcv_df, period=14).shift(1)
    vol_zscore  = compute_volume_zscore(ohlcv_df, period=20).shift(1)

    # Relative Volume (RVOL): ratio of current bar's volume to 20-bar mean
    if 'tick_volume' in ohlcv_df.columns:
        raw_vol = ohlcv_df['tick_volume'].astype(float)
    elif 'volume' in ohlcv_df.columns:
        raw_vol = ohlcv_df['volume'].astype(float)
    else:
        raw_vol = pd.Series(1.0, index=ohlcv_df.index)
    rvol_series = (raw_vol / raw_vol.rolling(20).mean().replace(0, np.nan)).shift(1)

    # EMA200 — shifted: use last completed bar's EMA, not current
    ema200 = close.ewm(span=200, adjust=False).mean().shift(1)

    # Bollinger Band width — shift components before deriving width
    bb_mid   = close.rolling(20).mean().shift(1)
    bb_std   = close.rolling(20).std().shift(1)
    bb_width = (4 * bb_std) / bb_mid.replace(0, np.nan)  # inherits shift

    # ATR percentile — atr_series already shifted, rolling on shifted series is clean
    atr_pct  = atr_series.rolling(100).apply(
        lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1), raw=True)

    # Fractional differentiation — shift final z-score
    print("[MetaLabeller] Computing fractional differentiation (d=0.4)...")
    frac_close = fractional_diff(close, d=FRAC_DIFF_D)
    frac_mu    = frac_close.rolling(500, min_periods=100).mean()
    frac_std   = frac_close.rolling(500, min_periods=100).std()
    frac_z     = ((frac_close - frac_mu) / frac_std.replace(0, np.nan)).shift(1)

    # Build regime shift tracker (bars_since_regime_shift)
    regime_shift_bars = _compute_regime_shift_bars(regime_labels)

    # Regime encoding
    regime_enc_map = {r: i for i, r in enumerate(ALL_REGIMES)}

    # Session from index
    import pytz
    NY_TZ = pytz.timezone('America/New_York')

    rows = []
    for _, event in labelled_events.iterrows():
        ts     = event['timestamp']
        regime = str(event.get('regime', REGIME_LOW_VOL_RANGE))
        signal = str(event.get('signal', 'long')).lower()

        if ts not in idx_map:
            continue
        bar_i = idx_map[ts]

        # ── Safe index helpers ──────────────────────────────────
        def _safe(series, i, default=0.0):
            if i < 0 or i >= len(series):
                return default
            v = series.iloc[i] if hasattr(series, 'iloc') else series[i]
            return float(v) if not (pd.isna(v)) else default

        # ── Signal quality features ─────────────────────────────
        xgb_conf = float(event.get('xgb_confidence', 0.5))

        # Prediction entropy: measures how "confused" the primary model was
        # High entropy (probs ~0.2 each) → model uncertain → meta should reject
        if primary_probs is not None and ts in primary_probs.index:
            probs_row = primary_probs.loc[ts].values.astype(float)
            probs_row = probs_row / probs_row.sum()   # normalise
            entropy   = float(-np.sum(probs_row * np.log(probs_row + 1e-9)))
            max_prob  = float(probs_row.max())
            sec_prob  = float(np.sort(probs_row)[-2]) if len(probs_row) > 1 else 0.0
            margin    = max_prob - sec_prob
        else:
            # Without full probs, estimate from confidence
            p         = xgb_conf
            q         = (1 - p) / max(len(ALL_REGIMES) - 1, 1)
            entropy   = float(-(p * np.log(p + 1e-9) +
                                (len(ALL_REGIMES) - 1) * q * np.log(q + 1e-9)))
            margin    = p - q

        # ── Regime features ─────────────────────────────────────
        regime_encoded = float(regime_enc_map.get(regime, 4))
        bars_since_shift = _safe(regime_shift_bars, bar_i, default=1.0)

        # ── Technical features at entry bar ─────────────────────
        rsi_val      = _safe(rsi_series,  bar_i, default=50.0)
        atr_val      = _safe(atr_series,  bar_i, default=10.0)
        atr_pct_val  = _safe(atr_pct,     bar_i, default=0.5)
        bb_w_val     = _safe(bb_width,    bar_i, default=0.02)
        vol_z_val    = _safe(vol_zscore,  bar_i, default=0.0)
        rvol_val     = _safe(rvol_series, bar_i, default=1.0)
        frac_z_val   = _safe(frac_z,      bar_i, default=0.0)

        # Distance to EMA200 (normalised by ATR)
        ema200_val   = _safe(ema200, bar_i, default=float(close.iloc[bar_i]))
        entry_price  = float(event.get('entry_price', _safe(close, bar_i)))
        dist_ema200  = (entry_price - ema200_val) / (atr_val if atr_val > 0 else 1.0)

        # Candle structure at entry
        c_range      = max(float(high.iloc[bar_i]) - float(low.iloc[bar_i]), 1e-8)
        body_ratio   = abs(float(close.iloc[bar_i]) - float(ohlcv_df['open'].iloc[bar_i])) / c_range
        close_pos    = (float(close.iloc[bar_i]) - float(low.iloc[bar_i])) / c_range

        # Signal direction encoded: long=1, short=-1
        direction_enc = 1.0 if signal == "long" else -1.0

        # ── Session features ────────────────────────────────────
        try:
            if ts.tzinfo is None:
                ts_ny = ts.tz_localize('UTC').tz_convert(NY_TZ)
            else:
                ts_ny = ts.tz_convert(NY_TZ)
            hour = ts_ny.hour
            dow  = ts_ny.dayofweek
        except Exception:
            hour = ts.hour if hasattr(ts, 'hour') else 12
            dow  = ts.dayofweek if hasattr(ts, 'dayofweek') else 2

        is_asian  = 1.0 if (hour >= 19 or hour < 2)  else 0.0
        is_london = 1.0 if (2 <= hour < 7)            else 0.0
        is_ny     = 1.0 if (7 <= hour < 17)           else 0.0
        hour_sin  = float(np.sin(2 * np.pi * hour / 24))
        hour_cos  = float(np.cos(2 * np.pi * hour / 24))

        row = {
            # Signal quality
            "xgb_confidence":      xgb_conf,
            "prediction_entropy":  entropy,
            "xgb_margin":          margin,
            "direction_encoded":   direction_enc,
            # Regime
            "regime_encoded":      regime_encoded,
            "bars_since_shift":    bars_since_shift,
            # Technical
            "rsi_at_entry":        rsi_val,
            "atr_percentile":      atr_pct_val,
            "bb_width_at_entry":   bb_w_val,
            "dist_to_ema200_atr":  dist_ema200,
            "body_ratio":          body_ratio,
            "close_position":      close_pos,
            # Volume
            "volume_zscore":       vol_z_val,
            "rvol":                rvol_val,
            # Frac diff
            "frac_diff_close_z":   frac_z_val,
            # Session
            "is_asian":            is_asian,
            "is_london":           is_london,
            "is_ny":               is_ny,
            "hour_sin":            hour_sin,
            "hour_cos":            hour_cos,
            "day_of_week":         float(dow),
            # Barrier quality
            "barrier_atr":         float(event.get('barrier_atr', atr_val)),
            # Target
            "meta_label":          int(event.get('meta_label', 0)),
            "pnl_at_exit":         float(event.get('pnl_at_exit', 0.0)),
            "exit_reason":         str(event.get('exit_reason', 'time')),
            "timestamp":           ts,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"[MetaLabeller] Meta features built: {len(df)} events × "
              f"{len([c for c in df.columns if c not in ['meta_label','pnl_at_exit','exit_reason','timestamp']])} features")
    return df


def _compute_regime_shift_bars(regime_labels: pd.Series) -> pd.Series:
    """Returns bars since last regime change for each bar."""
    result  = np.zeros(len(regime_labels), dtype=float)
    counter = 0
    prev    = None
    for i, regime in enumerate(regime_labels.values):
        if regime != prev:
            counter = 0
            prev    = regime
        else:
            counter += 1
        result[i] = float(counter)
    return pd.Series(result, index=regime_labels.index)


# ================================================================
# STEP 5 — PURGED + EMBARGOED CROSS-VALIDATION
# Prevents lookahead via overlapping Triple Barrier windows.
# ================================================================

def purged_kfold_cv(n: int, n_splits: int = 5, embargo_pct: float = 0.01):
    """
    Generator yielding (train_idx, test_idx) pairs for Purged K-Fold CV.

    Purging: removes training samples whose barrier windows overlap
             with the test period (would see future test data).
    Embargo: removes training samples AFTER the test fold
             (contaminated by price continuation).

    Args:
        n:           total number of samples (sequential)
        n_splits:    number of CV folds
        embargo_pct: fraction of fold size to embargo after test fold

    Yields:
        (train_indices, test_indices) — numpy arrays
    """
    indices    = np.arange(n)
    fold_size  = n // n_splits
    embargo    = max(1, int(fold_size * embargo_pct))

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end   = test_start + fold_size if fold < n_splits - 1 else n

        test_idx  = indices[test_start:test_end]

        # Purge: remove training samples that overlap with test window
        # Conservative: remove anything within 1 fold-size before test_end
        purge_start = max(0, test_start - fold_size)
        purge_end   = min(n, test_end + embargo)

        train_idx = np.concatenate([
            indices[:purge_start],
            indices[purge_end:]
        ])

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield train_idx, test_idx


# ================================================================
# STEP 6 — TRAIN META MODEL
# ================================================================

META_FEATURE_COLS = [
    "xgb_confidence", "prediction_entropy", "xgb_margin", "direction_encoded",
    "regime_encoded", "bars_since_shift",
    "rsi_at_entry", "atr_percentile", "bb_width_at_entry",
    "dist_to_ema200_atr", "body_ratio", "close_position",
    "volume_zscore", "rvol", "frac_diff_close_z",
    "is_asian", "is_london", "is_ny", "hour_sin", "hour_cos",
    "day_of_week", "barrier_atr",
]


def train(
    primary_signals_df: pd.DataFrame,
    ohlcv_df:           pd.DataFrame,
    regime_labels:      pd.Series,
    primary_probs:      pd.DataFrame = None,
    n_splits:           int = 5,
):
    """
    Full training pipeline:
        1. Apply Triple Barrier with dynamic ATR barriers per regime
        2. Build meta features (incl. frac diff, RVOL, entropy)
        3. Compute sample uniqueness weights
        4. Train XGBoost with Purged K-Fold CV
        5. Save model + meta
        6. Return CV accuracy report

    Args:
        primary_signals_df: DataFrame with columns:
            timestamp, signal ("long"/"short"), entry_price,
            xgb_confidence, regime
        ohlcv_df:           M5 OHLCV, indexed by timestamp
        regime_labels:      pd.Series of regime labels (same index as ohlcv_df)
        primary_probs:      optional — full probability distribution per bar
        n_splits:           number of Purged K-Fold CV splits

    Returns:
        dict with training summary
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("[MetaLabeller] XGBoost not installed. pip install xgboost")
        return None

    print("\n[MetaLabeller] ══════════════════════════════════════")
    print("[MetaLabeller] Starting XU-L Meta-Model Training")
    print("[MetaLabeller] ══════════════════════════════════════")

    # Step 1: Triple Barrier
    print("\n[MetaLabeller] Step 1/5: Applying Triple Barrier labelling...")
    labelled = apply_triple_barrier(ohlcv_df, primary_signals_df, regime_labels)

    if len(labelled) < 50:
        print(f"[MetaLabeller] Insufficient events ({len(labelled)}). Need ≥50.")
        return None

    # Step 2: Meta features
    print("\n[MetaLabeller] Step 2/5: Building meta features...")
    meta_df = build_meta_features(labelled, ohlcv_df, regime_labels, primary_probs)
    meta_df = meta_df.dropna(subset=META_FEATURE_COLS)

    if len(meta_df) < 30:
        print(f"[MetaLabeller] Too few clean samples ({len(meta_df)}) after dropna.")
        return None

    # Step 3: Sample uniqueness weights
    print("\n[MetaLabeller] Step 3/5: Computing sample uniqueness weights...")
    weights = compute_sample_uniqueness(
        meta_df.rename(columns={"timestamp": "timestamp"}),
        ohlcv_df
    )

    X       = meta_df[META_FEATURE_COLS].values.astype(float)
    y       = meta_df["meta_label"].values.astype(int)
    w_uniq  = weights.values if len(weights) == len(X) else np.ones(len(X))

    # LABELING MAGNITUDE FIX: blend uniqueness weights with outcome magnitude.
    # |pnl_r_continuous| gives the magnitude — large wins/losses get heavier weight.
    # This makes the model distinguish a lucky break-even from a perfect 3R trade.
    # Normalised so the mean weight stays ≈ 1.0 (no change to learning rate).
    if "pnl_r_continuous" in meta_df.columns:
        mag = meta_df["pnl_r_continuous"].abs().values
        mag = np.where(mag < 0.1, 0.1, mag)   # floor at 0.1 to avoid zero weights
        mag_norm = mag / mag.mean()            # normalise to mean ≈ 1.0
        w = w_uniq * mag_norm
        print(f"[MetaLabeller] Magnitude weighting applied: "
              f"max_mag={mag.max():.2f}R, mean_mag={mag.mean():.2f}R — "
              f"large outcomes get up to {mag_norm.max():.1f}x weight")
    else:
        w = w_uniq
        print("[MetaLabeller] No pnl_r_continuous column — using uniqueness weights only.")

    # ── FISF: strict filtering for meta model ─────────────────────
    # Meta model needs strict stability — spurious correlations here
    # mean false confidence in trade quality = real losses.
    print("\n[MetaLabeller] Running FISF (strict mode for meta model)...")
    stable_meta_cols = META_FEATURE_COLS   # fallback
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(current_dir)),
            'regime_detector'))
        from feature_stability import run_full_fisf_pipeline

        meta_X_df = meta_df[META_FEATURE_COLS].copy()
        meta_y    = pd.Series(meta_df["meta_label"].values)

        filtered_cols = run_full_fisf_pipeline(
            meta_X_df, meta_y,
            weights=w,
            mode="meta",
            use_shap=False,
            n_windows=5,
            verbose=True,
        )
        if len(filtered_cols) >= 8:
            stable_meta_cols = filtered_cols
            print(f"[MetaLabeller] FISF: {len(stable_meta_cols)} stable features "
                  f"(was {len(META_FEATURE_COLS)})")
        else:
            print("[MetaLabeller] FISF too aggressive — using full feature set.")
    except Exception as e:
        print(f"[MetaLabeller] FISF skipped ({e})")

    X = meta_df[stable_meta_cols].values.astype(float)

    # Class imbalance ratio
    n_pos   = y.sum()
    n_neg   = len(y) - n_pos
    imb     = n_neg / max(n_pos, 1)

    print(f"[MetaLabeller] Dataset: {len(X)} samples | "
          f"Win: {n_pos} ({n_pos/len(y)*100:.1f}%) | "
          f"Loss: {n_neg} | Imbalance ratio: {imb:.2f}")

    # Step 4: Purged K-Fold CV evaluation
    print(f"\n[MetaLabeller] Step 4/5: Purged K-Fold CV ({n_splits} folds)...")
    cv_scores = []
    cv_precisions = []

    xgb_params = {
        "max_depth":        5,
        "learning_rate":    0.03,
        "n_estimators":     500,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": imb,
        "eval_metric":      "logloss",
        "use_label_encoder": False,
        "verbosity":        0,
        "random_state":     42,
    }

    for fold_i, (train_idx, test_idx) in enumerate(purged_kfold_cv(len(X), n_splits)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        w_tr       = w[train_idx]

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_te, y_te)], verbose=False)

        proba_te   = clf.predict_proba(X_te)[:, 1]
        pred_te    = (proba_te >= 0.5).astype(int)
        acc        = float((pred_te == y_te).mean())
        # Precision at threshold
        tp         = int(((pred_te == 1) & (y_te == 1)).sum())
        fp         = int(((pred_te == 1) & (y_te == 0)).sum())
        prec       = tp / max(tp + fp, 1)
        cv_scores.append(acc)
        cv_precisions.append(prec)
        print(f"  Fold {fold_i+1}: Acc={acc:.3f} | Precision={prec:.3f}")

    mean_acc  = float(np.mean(cv_scores))
    mean_prec = float(np.mean(cv_precisions))
    print(f"\n[MetaLabeller] CV Mean Accuracy : {mean_acc:.3f} ± {np.std(cv_scores):.3f}")
    print(f"[MetaLabeller] CV Mean Precision: {mean_prec:.3f}")

    # Step 5: Final model on ALL data
    print("\n[MetaLabeller] Step 5/5: Training final model on full dataset...")
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X, y, sample_weight=w, verbose=False)

    # Compute R:R stats for Kelly Criterion
    wins        = meta_df[meta_df["meta_label"] == 1]["pnl_at_exit"]
    losses      = meta_df[meta_df["meta_label"] == 0]["pnl_at_exit"].abs()
    mean_win    = float(wins.mean()) if len(wins) > 0 else 1.5
    mean_loss   = float(losses.mean()) if len(losses) > 0 else 1.0

    global _meta_rr_stats
    _meta_rr_stats = {"mean_win": mean_win, "mean_loss": mean_loss}

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model.save_model(META_MODEL_PATH)

    # Magnitude stats for reporting
    if "pnl_r_continuous" in meta_df.columns:
        pnl_r_vals = meta_df["pnl_r_continuous"]
        mean_pnl_r_win  = float(pnl_r_vals[meta_df["meta_label"] == 1].mean()) if n_pos > 0 else 0
        mean_pnl_r_loss = float(pnl_r_vals[meta_df["meta_label"] == 0].mean()) if n_neg > 0 else 0
    else:
        mean_pnl_r_win = mean_win
        mean_pnl_r_loss = -mean_loss

    meta_info = {
        "trained_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_samples":         int(len(X)),
        "win_rate":          round(float(n_pos / len(y)), 4),
        "cv_accuracy":       round(mean_acc, 4),
        "cv_precision":      round(mean_prec, 4),
        "mean_win_atr":      round(mean_win,  4),
        "mean_loss_atr":     round(mean_loss, 4),
        "mean_pnl_r_win":    round(mean_pnl_r_win, 4),   # NEW: magnitude stats
        "mean_pnl_r_loss":   round(mean_pnl_r_loss, 4),  # NEW: magnitude stats
        "magnitude_weighting_used": "pnl_r_continuous" in meta_df.columns,
        "feature_cols":      stable_meta_cols,
        "barrier_params":    {k: v for k, v in REGIME_BARRIER_PARAMS.items()},
        "frac_diff_d":       FRAC_DIFF_D,
        "threshold":         META_MIN_THRESHOLD,
    }
    with open(META_META_PATH, 'w') as f:
        json.dump(meta_info, f, indent=4)

    print(f"\n[MetaLabeller] Model saved → {META_MODEL_PATH}")
    print(f"[MetaLabeller] Win rate: {n_pos/len(y)*100:.1f}% | "
          f"CV Accuracy: {mean_acc:.1%} | "
          f"Mean Win: {mean_win:.2f} ATR | Mean Loss: {mean_loss:.2f} ATR")
    print("[MetaLabeller] ══════════════════════════════════════\n")

    _load_meta_model()
    return meta_info


# ================================================================
# STEP 7 — LIVE PREDICTION: meta_prob → Kelly size
# ================================================================

def _load_meta_model():
    global _meta_model, _meta_mode, _meta_cols, _meta_rr_stats
    try:
        import xgboost as xgb
        if os.path.exists(META_MODEL_PATH):
            _meta_model = xgb.XGBClassifier()
            _meta_model.load_model(META_MODEL_PATH)
            _meta_mode  = "model"
            if os.path.exists(META_META_PATH):
                with open(META_META_PATH, 'r') as f:
                    meta = json.load(f)
                _meta_cols    = meta.get("feature_cols", META_FEATURE_COLS)
                _meta_rr_stats = {
                    "mean_win":  meta.get("mean_win_atr",  1.5),
                    "mean_loss": meta.get("mean_loss_atr", 1.0),
                }
            print(f"[MetaLabeller] Model loaded | Threshold: {META_MIN_THRESHOLD}")
        else:
            _meta_mode = "rule_based"
            print("[MetaLabeller] No model found — using rule-based scoring.")
    except Exception as e:
        _meta_mode = "rule_based"
        print(f"[MetaLabeller] Load failed ({e}) — using rule-based scoring.")


def _rule_based_meta(primary_signal: dict, regime_result: dict) -> dict:
    """
    Rule-based meta scoring used before model is trained.
    Based on the same regime × confidence × session logic
    that will eventually be learned by the meta-XGBoost.
    """
    regime    = regime_result.get("regime", REGIME_LOW_VOL_RANGE)
    conf      = regime_result.get("confidence") or 0.5
    signal    = str(primary_signal.get("signal", "long")).upper()
    session   = str(primary_signal.get("session", "NY"))

    # Base probability from regime + confidence
    base_probs = {
        REGIME_BULL_TREND:    0.62,
        REGIME_BEAR_TREND:    0.62,
        REGIME_LOW_VOL_RANGE: 0.55,
        REGIME_COMPRESSION:   0.35,   # breakout direction unknown → low prob
        REGIME_REVERSAL:      0.45,
    }
    p = base_probs.get(regime, 0.50)

    # Adjust for confidence
    p += (conf - 0.5) * 0.20

    # Penalise counter-regime trades
    if regime == REGIME_BULL_TREND and signal == "SELL":
        p -= 0.10
    elif regime == REGIME_BEAR_TREND and signal == "BUY":
        p -= 0.10

    # Penalise dead session
    if session not in ("London", "NY"):
        p -= 0.05

    p = float(np.clip(p, 0.10, 0.90))
    sizing = _combined_size(p, META_MIN_THRESHOLD)
    return {
        "meta_prob":          round(p, 3),
        "kelly_size":         sizing["kelly_size"],
        "edge_size":          sizing["edge_size"],
        "size":               sizing["size"],
        "size_method":        sizing["method"],
        "should_trade":       p >= META_MIN_THRESHOLD,
        "shap_reason":        f"Rule-based: regime={regime} conf={conf:.0%} → p={p:.2f}",
        "mode":               "rule_based",
        "threshold_used":     META_MIN_THRESHOLD,
        "threshold_breakdown": f"fixed={META_MIN_THRESHOLD}",
        "age_status":         "MATURE",
    }


def _kelly(p: float) -> float:
    """
    Half-Kelly position size from meta win probability.
    f* = (p*b - q) / b  where b = mean_win/mean_loss
    Capped at META_KELLY_CAP.
    """
    b = _meta_rr_stats["mean_win"] / max(_meta_rr_stats["mean_loss"], 0.01)
    q = 1.0 - p
    f = (p * b - q) / max(b, 0.01)
    f_half = f * META_KELLY_FRACTION
    return float(np.clip(f_half, 0.0, META_KELLY_CAP))


def _edge_size(meta_prob: float, threshold: float) -> float:
    """
    Edge-proportional size: scales linearly with how far meta_prob
    sits above threshold.

        edge = meta_prob - threshold
        size = edge / (1 - threshold)

    A barely-passing trade (prob = threshold + 0.01) gets ~2% size.
    A high-conviction trade (prob=0.90, threshold=0.55) gets 78%.
    More intuitive than Kelly but less mathematically rigorous.
    """
    if meta_prob <= threshold:
        return 0.0
    denom = max(1.0 - threshold, 1e-6)
    return float(np.clip((meta_prob - threshold) / denom, 0.0, META_KELLY_CAP))


def _combined_size(meta_prob: float, threshold: float) -> dict:
    """
    Final position size = min(Kelly, edge-proportional).

    Why the minimum:
        Kelly optimises long-run growth given win rate + R:R.
        Edge-proportional scales with model conviction above threshold.
        Both must justify the size — conservatism in the right direction.

    Returns dict with all components for full logging transparency.
    """
    k_size = _kelly(meta_prob)
    e_size = _edge_size(meta_prob, threshold)
    final  = min(k_size, e_size)
    return {
        "size":       round(final,  3),
        "kelly_size": round(k_size, 3),
        "edge_size":  round(e_size, 3),
        "method":     "kelly" if k_size <= e_size else "edge",
    }


def predict_meta(
    primary_signal:   dict,
    regime_result:    dict,
    market_features:  pd.DataFrame = None,
) -> dict:
    """
    Main live prediction entry point.

    Called from main_bot.py after regime gate passes, before execution.

    Args:
        primary_signal: dict with keys:
            signal     — "BUY" or "SELL"
            confidence — XGBoost confidence (float)
            session    — "Asian" | "London" | "NY"
            entry_price
            (optionally: hour, day_of_week)
        regime_result:  output of regime_detector.predict()
        market_features: latest single-row feature DataFrame from
                         build_multi_tf_features() — optional but improves accuracy

    Returns:
        {
            "meta_prob":    float,   # 0–1 win probability
            "kelly_size":   float,   # position size multiplier
            "should_trade": bool,
            "shap_reason":  str,
            "mode":         str,
        }
    """
    global _meta_model, _meta_mode

    if _meta_mode is None:
        _load_meta_model()

    if _meta_mode == "rule_based" or _meta_model is None:
        return _rule_based_meta(primary_signal, regime_result)

    try:
        import xgboost as xgb

        regime   = regime_result.get("regime", REGIME_LOW_VOL_RANGE)
        conf     = regime_result.get("confidence") or 0.5
        signal   = str(primary_signal.get("signal", "BUY")).lower()
        probs    = regime_result.get("probabilities") or {}
        persist  = regime_result.get("persistence", {})

        # Entropy from probability distribution
        if probs:
            p_arr    = np.array(list(probs.values()), dtype=float)
            p_arr    = p_arr / p_arr.sum()
            entropy  = float(-np.sum(p_arr * np.log(p_arr + 1e-9)))
            p_sorted = np.sort(p_arr)[::-1]
            margin   = float(p_sorted[0] - p_sorted[1]) if len(p_sorted) > 1 else conf
        else:
            p        = conf
            q        = (1 - p) / max(len(ALL_REGIMES) - 1, 1)
            entropy  = float(-(p * np.log(p+1e-9) +
                               (len(ALL_REGIMES)-1)*q*np.log(q+1e-9)))
            margin   = conf - q

        regime_enc = float({r: i for i, r in enumerate(ALL_REGIMES)}.get(regime, 4))
        bars_since = float(persist.get("candles_since_regime_start", 1))
        dir_enc    = 1.0 if signal == "buy" else -1.0

        # Pull from market_features if available, else defaults
        def _feat(col, default):
            if market_features is not None and col in market_features.columns:
                v = market_features[col].iloc[-1]
                return float(v) if not pd.isna(v) else default
            return default

        import datetime as _dt
        now_hour = _dt.datetime.now().hour
        now_dow  = _dt.datetime.now().weekday()

        row = {
            "xgb_confidence":      conf,
            "prediction_entropy":  entropy,
            "xgb_margin":          margin,
            "direction_encoded":   dir_enc,
            "regime_encoded":      regime_enc,
            "bars_since_shift":    bars_since,
            "rsi_at_entry":        _feat("m5_rsi", 50.0),
            "atr_percentile":      _feat("m5_atr_percentile", 0.5),
            "bb_width_at_entry":   _feat("m5_bb_width", 0.02),
            "dist_to_ema200_atr":  _feat("m5_ema_pos_200", 0.0),
            "body_ratio":          _feat("m5_body_ratio", 0.5),
            "close_position":      _feat("m5_close_position", 0.5),
            "volume_zscore":       _feat("m5_volume_zscore", 0.0),
            "rvol":                1.0,   # not available live without raw volume
            "frac_diff_close_z":   0.0,   # requires full series — skip live
            "is_asian":            _feat("is_asian", 0.0),
            "is_london":           _feat("is_london", 0.0),
            "is_ny":               _feat("is_ny", 0.0),
            "hour_sin":            float(np.sin(2 * np.pi * now_hour / 24)),
            "hour_cos":            float(np.cos(2 * np.pi * now_hour / 24)),
            "day_of_week":         float(now_dow),
            "barrier_atr":         _feat("m5__atr_raw", 10.0),
        }

        cols = _meta_cols or META_FEATURE_COLS
        X    = np.array([[row.get(c, 0.0) for c in cols]], dtype=float)

        meta_prob  = float(_meta_model.predict_proba(X)[0][1])

        # ── Adaptive threshold — replaces fixed META_MIN_THRESHOLD ──
        # Pulls from session_profile if enriched, else falls back to
        # static get_adaptive_threshold() with available context.
        try:
            from session_profiler import get_adaptive_threshold
            sp = regime_result.get("session_profile", {})
            if sp:
                # Already computed during enrich_regime_result()
                threshold = sp.get("adaptive_threshold", {}).get(
                    "threshold", META_MIN_THRESHOLD)
                thresh_breakdown = sp.get("adaptive_threshold", {}).get(
                    "breakdown", f"fixed={META_MIN_THRESHOLD}")
                age_status = sp.get("age_status", "MATURE")
            else:
                # Compute fresh
                norm_vol   = float(regime_result.get(
                    "session_profile", {}).get("norm_vol", 1.0))
                age_status = "MATURE"
                at = get_adaptive_threshold(
                    regime, session,
                    norm_vol=norm_vol,
                    age_status=age_status,
                    signal_aligned=(
                        not (regime == "BULL_TREND" and signal == "sell") and
                        not (regime == "BEAR_TREND" and signal == "buy")
                    )
                )
                threshold        = at["threshold"]
                thresh_breakdown = at["breakdown"]
        except Exception:
            threshold        = META_MIN_THRESHOLD
            thresh_breakdown = f"fixed={META_MIN_THRESHOLD}"
            age_status       = "MATURE"

        kelly_size = _kelly(meta_prob)
        should     = meta_prob >= threshold

        # Combined sizing: min(Kelly, edge-proportional)
        sizing = _combined_size(meta_prob, threshold)

        # SHAP explanation
        shap_reason = _build_shap_reason(X, cols, meta_prob, should)

        result = {
            "meta_prob":           round(meta_prob,        3),
            "kelly_size":          sizing["kelly_size"],
            "edge_size":           sizing["edge_size"],
            "size":                sizing["size"],
            "size_method":         sizing["method"],
            "should_trade":        should,
            "shap_reason":         shap_reason,
            "mode":                "model",
            "threshold_used":      round(threshold,        3),
            "threshold_breakdown": thresh_breakdown,
            "age_status":          age_status,
        }

        print(f"[MetaLabeller] {regime} | signal={signal.upper()} | "
              f"meta_p={meta_prob:.2f} vs thresh={threshold:.2f} | "
              f"size={sizing['size']:.2f} ({sizing['method']}: "
              f"kelly={sizing['kelly_size']:.2f} edge={sizing['edge_size']:.2f}) | "
              f"age={age_status} | "
              f"{'✓ TRADE' if should else '✗ SKIP'}")

        return result

    except Exception as e:
        print(f"[MetaLabeller] predict_meta error: {e}. Falling back to rules.")
        return _rule_based_meta(primary_signal, regime_result)


# ================================================================
# STEP 8 — SHAP → WISDOM BRIDGE
# Converts meta-model decisions into structured lessons for Wisdom.
# ================================================================

def _build_shap_reason(X: np.ndarray, cols: list,
                        meta_prob: float, should_trade: bool) -> str:
    """
    Generates a human-readable explanation of the meta-model decision
    using SHAP values. Falls back to feature importance if SHAP unavailable.
    """
    try:
        import shap
        explainer   = shap.TreeExplainer(_meta_model)
        shap_values = explainer.shap_values(X)
        sv          = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

        feature_contributions = sorted(
            zip(cols, sv), key=lambda x: abs(x[1]), reverse=True)[:4]

        parts = []
        for feat, shap_val in feature_contributions:
            direction = "↑ supports trade" if shap_val > 0 else "↓ opposes trade"
            parts.append(f"{feat}: {shap_val:+.3f} ({direction})")

        decision = "ALLOWED" if should_trade else "BLOCKED"
        return (f"Meta {decision} (p={meta_prob:.2f}). "
                f"Key factors: {' | '.join(parts)}")

    except Exception:
        # Fallback: top features by absolute value
        decision = "ALLOWED" if should_trade else "BLOCKED"
        return (f"Meta {decision} (p={meta_prob:.2f}). "
                f"SHAP unavailable — install with: pip install shap")


def generate_wisdom_lesson(
    ticket:          str,
    meta_prob:       float,
    actual_outcome:  str,      # "WIN" or "LOSS"
    regime:          str,
    session:         str,
    signal:          str,
    shap_reason:     str,
    # ── Previously missing fields (now required) ─────────────────
    threshold_used:  float = None,   # the adaptive threshold that was applied
    norm_vol:        float = 1.0,    # normalised volatility at trade time
    trade_taken:     bool  = True,   # always True here — see log_blocked_trade()
) -> dict:
    """
    Generates a structured wisdom lesson from a completed meta-labelled trade.
    Called by wisdom_builder.py during the rebuild cycle.

    All fields needed for walk-forward threshold calibration are now logged:
        threshold_used → lets calibrate_thresholds_from_history() know
                         exactly what bar was set for each trade
        norm_vol       → enables volatility-stratified analysis
        trade_taken    → always True for this function (False = log_blocked_trade)
    """
    if threshold_used is None:
        threshold_used = META_MIN_THRESHOLD

    predicted_correctly = (
        (meta_prob >= threshold_used and actual_outcome == "WIN") or
        (meta_prob < threshold_used  and actual_outcome == "LOSS")
    )

    lesson_key = (f"meta_{regime.lower()}_{session.lower()}_"
                  f"{signal.lower()}_{actual_outcome.lower()}")

    lesson_text = (
        f"Meta-model predicted {'WIN' if meta_prob >= threshold_used else 'LOSS'} "
        f"(p={meta_prob:.2f}, thresh={threshold_used:.2f}) "
        f"for {signal} in {regime} during {session} session "
        f"(vol={norm_vol:.1f}x normal). "
        f"Actual outcome: {actual_outcome}. "
        f"Prediction was {'✓ correct' if predicted_correctly else '✗ wrong'}. "
        f"Model reasoning: {shap_reason}"
    )

    entry = {
        "key":                  lesson_key,
        "lesson":               lesson_text,
        "ticket":               str(ticket),
        "meta_prob":            round(meta_prob,       3),
        "threshold_used":       round(threshold_used,  3),  # ← was missing
        "norm_vol":             round(norm_vol,         2),  # ← was missing
        "trade_taken":          True,                        # ← was missing
        "actual_outcome":       actual_outcome,
        "regime":               regime,
        "session":              session,
        "signal":               signal,
        "predicted_correctly":  predicted_correctly,
        "shap_reason":          shap_reason,
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M"),
        "entry_type":           "executed",
    }

    _append_wisdom_log(entry)
    return entry


def log_blocked_trade(
    signal:         str,
    regime:         str,
    session:        str,
    meta_prob:      float,
    threshold_used: float,
    norm_vol:       float = 1.0,
    shap_reason:    str   = "",
) -> dict:
    """
    Logs a trade that was BLOCKED by the meta gate.

    This is the most critical calibration data:
    If thresholds are too strict, blocked trades are predominantly winners.
    If thresholds are correct, blocked trades are predominantly losers.

    Outcome is logged as "UNKNOWN" — the actual result must be filled in
    retrospectively by reviewing the chart (or via backtest replay).
    walk-forward calibration uses this data to detect over-filtering.

    Called from main_bot.py when meta gate blocks a signal.
    """
    entry = {
        "key":            f"blocked_{regime.lower()}_{session.lower()}_{signal.lower()}",
        "lesson":         (
            f"Meta gate BLOCKED {signal} in {regime}/{session} "
            f"(p={meta_prob:.2f} < thresh={threshold_used:.2f}, "
            f"vol={norm_vol:.1f}x). "
            f"Actual outcome unknown — review chart to label. "
            f"Reason: {shap_reason}"
        ),
        "ticket":         "BLOCKED",
        "meta_prob":      round(meta_prob,       3),
        "threshold_used": round(threshold_used,  3),
        "norm_vol":       round(norm_vol,         2),
        "trade_taken":    False,                       # ← key distinction
        "actual_outcome": "UNKNOWN",
        "regime":         regime,
        "session":        session,
        "signal":         signal,
        "shap_reason":    shap_reason,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "entry_type":     "blocked",
    }

    _append_wisdom_log(entry)
    print(f"[MetaLabeller] Blocked trade logged "
          f"(p={meta_prob:.2f} < {threshold_used:.2f}) — "
          f"{regime}/{session}/{signal}")
    return entry


def _append_wisdom_log(entry: dict):
    """Shared log writer for both executed and blocked trades."""
    try:
        os.makedirs(os.path.dirname(WISDOM_LOG_PATH), exist_ok=True)
        log = []
        if os.path.exists(WISDOM_LOG_PATH):
            with open(WISDOM_LOG_PATH, 'r') as f:
                log = json.load(f)
        log.append(entry)
        log = log[-500:]   # rolling 500 entries
        with open(WISDOM_LOG_PATH, 'w') as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"[MetaLabeller] Could not write wisdom log: {e}")


def get_regime_meta_stats() -> dict:
    """
    Returns per-regime meta prediction accuracy + blocked trade stats.
    Used by WisdomBuilder to surface calibration data into Claude prompt.

    Now includes:
        accuracy         — prediction accuracy on executed trades
        blocked_count    — how many trades were blocked in this regime
        block_rate       — blocked / (blocked + executed)
        vol_mean         — mean normalised volatility when trades occurred
        threshold_mean   — mean adaptive threshold used (for calibration)
    """
    if not os.path.exists(WISDOM_LOG_PATH):
        return {}

    try:
        with open(WISDOM_LOG_PATH, 'r') as f:
            log = json.load(f)

        stats = {}
        for entry in log[-200:]:
            r    = entry.get("regime", "UNKNOWN")
            taken = entry.get("trade_taken", True)
            cor  = entry.get("predicted_correctly", False)

            if r not in stats:
                stats[r] = {
                    "correct": 0, "executed": 0, "blocked": 0,
                    "vol_sum": 0.0, "thresh_sum": 0.0, "thresh_count": 0,
                }

            if taken:
                stats[r]["executed"] += 1
                stats[r]["correct"]  += int(cor)
            else:
                stats[r]["blocked"]  += 1

            stats[r]["vol_sum"]    += float(entry.get("norm_vol", 1.0))
            thresh = entry.get("threshold_used")
            if thresh:
                stats[r]["thresh_sum"]   += float(thresh)
                stats[r]["thresh_count"] += 1

        result = {}
        for regime, s in stats.items():
            total    = s["executed"] + s["blocked"]
            acc      = s["correct"] / max(s["executed"], 1)
            blk_rate = s["blocked"] / max(total, 1)
            vol_mean = s["vol_sum"] / max(total, 1)
            thr_mean = (s["thresh_sum"] / s["thresh_count"]
                        if s["thresh_count"] > 0 else META_MIN_THRESHOLD)

            result[regime] = {
                "accuracy":       round(acc,      3),
                "executed":       s["executed"],
                "blocked":        s["blocked"],
                "block_rate":     round(blk_rate, 3),
                "vol_mean":       round(vol_mean, 2),
                "threshold_mean": round(thr_mean, 3),
            }
        return result

    except Exception:
        return {}


# ── Auto-load on import ───────────────────────────────────────────
if os.path.exists(META_MODEL_PATH):
    _load_meta_model()
else:
    _meta_mode = "rule_based"
    print("[MetaLabeller] No trained model found — rule-based scoring active.")
    print("[MetaLabeller] Train with: meta_labeller.train(signals_df, ohlcv_df, regime_labels)")
