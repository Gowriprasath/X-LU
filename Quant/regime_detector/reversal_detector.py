"""
reversal_detector.py — Binary REVERSAL Pre-Filter
===================================================

WHY THIS EXISTS
────────────────
REVERSAL is ~4-6% of all candles (NFP, FOMC, news spikes).
In a 5-class XGBoost with balanced class weights, REVERSAL still competes
against 4 other classes for every split.  The model learns "ATR spike →
probably REVERSAL" — but "probably" is 55-65% confident because LOW_VOL
and COMPRESSION also occasionally have ATR above 1.0.

The consequences of a missed REVERSAL are asymmetric:
  - False negative (missed REVERSAL): trade fires into a news spike → SL
    hunted within seconds, full loss, possibly slippage beyond SL
  - False positive (spurious REVERSAL): trade blocked → missed profit

False negatives are the critical failure. A dedicated binary model fixes this.

HOW IT WORKS
─────────────
Stage 1  — This file (runs first every cycle, ~0.3ms):
    Binary XGBoost: P(REVERSAL) vs P(not-REVERSAL)
    If P(REVERSAL) >= REVERSAL_FIRE_THRESHOLD → return REVERSAL immediately
    Skip the 5-class model entirely for this candle

Stage 2  — regime_detector._predict_xgb() (runs only if Stage 1 did not fire)

WHY A BINARY MODEL CATCHES MORE REVERSALS
───────────────────────────────────────────
In a 5-class model, XGBoost has 5 output nodes competing.
A REVERSAL candle with P=0.52 fires correctly in 5-class.

In binary: only 2 output nodes.  The same candle now has P=0.78.
Binary classification is inherently easier — the model can use its full
capacity to distinguish "spike" from "no spike" without also worrying about
BULL vs BEAR vs COMPRESSION.

FEATURES USED
──────────────
Not all 124+ features — only the 12 that are most directly informative
about ATR spikes and news-driven volatility.  Using fewer, targeted features
prevents the model from memorising patterns that correlate with REVERSAL in
training data but don't generalise (e.g. specific time-of-day patterns
that happened to coincide with NFP in the training period).

PRIMARY features (volatility spike signal):
    h1_atr_ratio       — ATR / rolling mean: the core signal, threshold ~1.5
    h1_atr_slope       — rate of change of ATR: is the spike building or fading?
    h4_atr_ratio       — higher-TF confirmation: spike visible on H4 too?
    h1_atr_percentile  — where current ATR sits in rolling 100-candle distribution
    h1_rolling_vol_20  — returns volatility: independent confirmation of ATR
    h4_rolling_vol_20  — H4 returns volatility
    h1_vol_ratio_20_100— short/long vol ratio: sudden spike vs sustained high vol

SECONDARY features (context — prevents mislabeling sustained trends as REVERSAL):
    h1_adx             — high ADX = trend, not spike
    h1_bb_width        — BB width: spike usually expands BB rapidly
    h1_body_ratio      — large candle body = directional move, not spike
    is_ny              — NY session: most economic releases happen here
    is_london          — London open: second-highest news risk

REVERSAL_FIRE_THRESHOLD = 0.68
    Conservative by design.  We'd rather miss 20% of REVERSAL candles than
    block trades on 5% of trending candles.  The 5-class model has a second
    chance to catch REVERSAL anyway.

PUBLIC API
──────────
train(features, labels)          → saves model, returns accuracy
predict_reversal_prob(row)        → float: P(REVERSAL)
is_reversal(row)                  → bool (prob >= REVERSAL_FIRE_THRESHOLD)
load()                            → loads model into module state
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import (REVERSAL_DETECTOR_PATH, REVERSAL_DETECTOR_META_PATH,
                   create_all_dirs as _cad_rd)
_cad_rd()

# ── Configuration ──────────────────────────────────────────────────
REVERSAL_FIRE_THRESHOLD = 0.63   # P(REVERSAL) above this → fire pre-filter
                                  # Calibration history:
                                  #   0.68 → recall=0.009, broken (never fired)
                                  #   0.40 → blocking 67%, broken (fired constantly)
                                  #   0.55 → blocking 34%, still too aggressive
                                  #   0.63 → targets ~15-20% block rate, recall ~0.45-0.50
                                  # Gold trends more than it reverses — lower recall
                                  # is correct. 5-class model handles the remainder.

# Features used by the binary model — pure volatility / price signals ONLY.
# WHY NO SESSION FLAGS (is_ny / is_london removed):
#   In the previous run, is_ny dominated at 72% importance — the model learned
#   "NY session = reversal" instead of "ATR spike = reversal". Most NY candles
#   are NOT reversals. Session context is handled by the 5-class model.
#   The binary pre-filter must focus on price structure alone.
REVERSAL_FEATURES = [
    # Core volatility spike signal
    'h1_atr_ratio',        # ATR / rolling mean — primary signal
    'h1_atr_slope',        # rate of change of ATR
    'h4_atr_ratio',        # H4 confirmation
    'h1_atr_percentile',   # rank within 100-candle window
    'h1_rolling_vol_20',   # returns-based vol
    'h4_rolling_vol_20',   # H4 returns vol
    'h1_vol_ratio_20_100', # short/long vol ratio — spike vs sustained

    # Context: prevents confusing strong trends with reversal spikes
    'h1_adx',              # high ADX = trend, not spike
    'h1_bb_width',         # BB expansion on spikes
    'h1_body_ratio',       # large body = directional, not spike
    'h1_upper_wick_ratio', # large wick = spike exhaustion
    'h1_lower_wick_ratio', # large wick = spike exhaustion
    # NOTE: is_ny and is_london intentionally excluded — session flags caused
    # is_ny to dominate at 72% importance, learning session timing not price spikes.
]

# Fallback: columns available if some REVERSAL_FEATURES are missing
_FALLBACK_FEATURES = ['h1_atr_ratio', 'h4_atr_ratio', 'h1_adx', 'h1_bb_width']

# Module state — loaded once, reused every predict()
_reversal_model = None
_reversal_cols  = None   # actual feature columns used (subset of REVERSAL_FEATURES)
_loaded         = False


# ================================================================
# MODEL LOAD
# ================================================================

def load() -> bool:
    """
    Loads the reversal detector from disk into module state.
    Called automatically on first predict_reversal_prob() call.
    Safe to call multiple times — only loads if not already loaded.

    Returns True if model loaded successfully, False if not found.
    """
    global _reversal_model, _reversal_cols, _loaded

    if not os.path.exists(REVERSAL_DETECTOR_PATH):
        return False   # not trained yet — train() needed first

    try:
        import xgboost as xgb
        _reversal_model = xgb.XGBClassifier()
        _reversal_model.load_model(REVERSAL_DETECTOR_PATH)

        if os.path.exists(REVERSAL_DETECTOR_META_PATH):
            with open(REVERSAL_DETECTOR_META_PATH) as f:
                meta = json.load(f)
            _reversal_cols = meta.get("feature_cols", REVERSAL_FEATURES)
            print(f"[ReversalDetector] Loaded binary pre-filter "
                  f"(threshold={REVERSAL_FIRE_THRESHOLD} | "
                  f"features={len(_reversal_cols)} | "
                  f"trained={meta.get('trained_date','?')})")
        else:
            _reversal_cols = REVERSAL_FEATURES

        _loaded = True
        return True

    except Exception as e:
        print(f"[ReversalDetector] Load failed: {e}")
        _loaded = False
        return False


# ================================================================
# PREDICT
# ================================================================

def predict_reversal_prob(row: pd.DataFrame) -> float:
    """
    Returns P(REVERSAL) for the given feature row.

    Args:
        row: pd.DataFrame with 1 row — the current candle's feature vector.
             Must contain the columns in REVERSAL_FEATURES (or subset).

    Returns:
        float: probability that this candle is a REVERSAL regime.
        Returns 0.0 if model not loaded or row is missing all features.
    """
    global _reversal_model, _reversal_cols, _loaded

    if not _loaded:
        load()
    if _reversal_model is None:
        return 0.0

    try:
        cols = _reversal_cols or REVERSAL_FEATURES
        avail = [c for c in cols if c in row.columns]
        if len(avail) < len(_FALLBACK_FEATURES):
            return 0.0   # insufficient features — don't fire

        # BUG-RD-01 FIX: Build COMPLETE feature vector (all cols expected by model).
        # Previous code used row[avail] — only columns present in the row —
        # producing a 10-feature array when the model expects 12. This caused:
        #   "Feature shape mismatch, expected: 12, got 10"
        # on every candle during the D1 warmup period (~first 50 trading days of
        # any backtest run), where get_regime_from_slice() falls back to H1-only
        # features because d1_slice has ≤ 50 rows. The two missing features are
        # h4_atr_ratio and h4_rolling_vol_20 (H4 features absent in H1-only mode).
        #
        # Fix: always build a len(cols)-wide array. Missing features → 0.0.
        # Zero-filling h4 features is safe and conservative: 0.0 = "no H4 ATR spike"
        # rather than blocking the filter entirely. Matches how _predict_xgb Stage 2
        # fills missing base_cols with 0 before the 5-class XGB call.
        X = np.array(
            [[float(row[c].iloc[0]) if c in row.columns and not pd.isna(row[c].iloc[0]) else 0.0
              for c in cols]],
            dtype=float
        )

        # Binary model: class 0 = not_reversal, class 1 = REVERSAL
        proba = _reversal_model.predict_proba(X)[0]
        # proba shape: [P(not_reversal), P(reversal)]
        return float(proba[1]) if len(proba) == 2 else 0.0

    except Exception as e:
        print(f"[ReversalDetector] Predict error: {e}")
        return 0.0


def is_reversal(row: pd.DataFrame) -> tuple:
    """
    Returns (fired: bool, probability: float).

    fired = True means P(REVERSAL) >= REVERSAL_FIRE_THRESHOLD.
    When fired, regime_detector should return REVERSAL immediately
    without calling the 5-class model.

    Args:
        row: single-row pd.DataFrame of features (same as predict())
    """
    prob = predict_reversal_prob(row)
    return (prob >= REVERSAL_FIRE_THRESHOLD), prob


# ================================================================
# TRAIN
# ================================================================

def train(features: pd.DataFrame, labels: pd.Series,
          verbose: bool = True) -> dict:
    """
    Trains the binary REVERSAL detector and saves to REVERSAL_DETECTOR_PATH.

    Called automatically at the end of trainer.train() so both models
    are always in sync (same training data, same feature matrix).

    Args:
        features: full feature matrix (same as 5-class trainer input)
        labels:   full label series aligned to features

    Returns:
        dict with training results (accuracy, recall, precision)
        or {} on failure.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import (classification_report, accuracy_score,
                                     recall_score, precision_score)
        from sklearn.utils.class_weight import compute_class_weight
        from feature_engineer import apply_rolling_zscore, TF_FEATURE_COLS
    except ImportError as e:
        print(f"[ReversalDetector] Missing library: {e}")
        return {}

    if verbose:
        print("\n" + "=" * 65)
        print("  REVERSAL BINARY PRE-FILTER — Training")
        print("=" * 65)

    # ── Build binary labels ───────────────────────────────────────
    # 1 = REVERSAL, 0 = everything else
    y_binary = (labels == "REVERSAL").astype(int)
    n_total  = len(y_binary)
    n_rev    = int(y_binary.sum())
    n_not    = n_total - n_rev
    pct_rev  = n_rev / n_total * 100

    if verbose:
        print(f"\n  REVERSAL candles : {n_rev:,}  ({pct_rev:.1f}%)")
        print(f"  Non-REVERSAL     : {n_not:,}  ({100-pct_rev:.1f}%)")

    if n_rev < 100:
        print(f"[ReversalDetector] Too few REVERSAL candles ({n_rev}). "
              f"Need ≥ 100. Skipping training.")
        return {}

    # ── Select and prepare features ───────────────────────────────
    avail_cols = [c for c in REVERSAL_FEATURES if c in features.columns]
    if len(avail_cols) < len(_FALLBACK_FEATURES):
        print(f"[ReversalDetector] Too few features available "
              f"({len(avail_cols)}/{len(REVERSAL_FEATURES)}). "
              f"Need at least: {_FALLBACK_FEATURES}")
        return {}

    if verbose:
        missing = [c for c in REVERSAL_FEATURES if c not in features.columns]
        if missing:
            print(f"  ⚠  {len(missing)} features not in matrix (using {len(avail_cols)}): "
                  f"{missing[:5]}")
        else:
            print(f"  All {len(avail_cols)} REVERSAL features available.")

    X = features[avail_cols].copy()

    # Apply rolling z-score to TF volatility features only
    tf_avail = [c for c in TF_FEATURE_COLS if c in avail_cols]
    X = apply_rolling_zscore(X, cols=tf_avail, window=500)

    y = y_binary.reindex(X.index)
    valid = X.notna().all(axis=1) & y.notna()
    X, y  = X[valid], y[valid]

    if verbose:
        print(f"\n  After z-score + dropna: {len(X):,} rows")

    # ── Chronological 70 / 15 / 15 split ─────────────────────────
    n      = len(X)
    split1 = int(n * 0.70)
    split2 = int(n * 0.85)

    X_train, y_train = X.values[:split1],        y.values[:split1]
    X_val,   y_val   = X.values[split1:split2],  y.values[split1:split2]
    X_test,  y_test  = X.values[split2:],        y.values[split2:]

    if verbose:
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  "
              f"Test: {len(X_test):,}")
        print(f"  Train REVERSAL rate: {y_train.mean():.1%}")
        print(f"  Test  REVERSAL rate: {y_test.mean():.1%}")

    # ── Class weights — REVERSAL is rare, needs upweighting ───────
    # Use a manual scale_pos_weight for binary XGBoost:
    # scale_pos_weight = n_negative / n_positive
    # This is XGBoost's native way to handle binary imbalance — more direct
    # than sample_weight for binary problems.
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw   = n_neg / max(n_pos, 1)   # e.g. 18:1 ratio → scale_pos_weight=18

    if verbose:
        print(f"\n  scale_pos_weight: {spw:.1f}  "
              f"(upweights REVERSAL {spw:.0f}× to counter class imbalance)")

    # ── XGBoost params — binary classification ────────────────────
    # Recall-optimised: lower min_child_weight and gamma than main model
    # so the model splits more aggressively on rare REVERSAL patterns.
    params = dict(
        objective         = 'binary:logistic',
        n_estimators      = 500,
        max_depth         = 5,          # shallower than 5-class — binary is simpler
        learning_rate     = 0.03,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        gamma             = 0.1,        # lower than main model — REVERSAL is rare
        min_child_weight  = 3,          # lower — don't miss small REVERSAL clusters
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        scale_pos_weight  = spw,        # binary-native imbalance handling
        eval_metric       = 'aucpr',    # area under precision-recall — better than AUC for rare events
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
        early_stopping_rounds = 40,
    )

    if verbose:
        print(f"\n  Training binary XGBoost (objective=binary:logistic, "
              f"eval_metric=aucpr)...")

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set   = [(X_val, y_val)],
              verbose    = False)

    n_trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') else 500
    if verbose:
        print(f"  Early stopped at tree {n_trees} / 500")

    # ── Evaluate on blind test set ────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= REVERSAL_FIRE_THRESHOLD).astype(int)

    acc     = float(accuracy_score(y_test, y_pred))
    recall  = float(recall_score(y_test, y_pred, zero_division=0))
    prec    = float(precision_score(y_test, y_pred, zero_division=0))
    # F2 score: recall weighted 2× — we care more about catching REVERSAL than precision
    f2      = (5 * prec * recall) / (4 * prec + recall) if (prec + recall) > 0 else 0.0

    if verbose:
        print(f"\n{'─' * 65}")
        print(f"  TEST SET — Threshold: {REVERSAL_FIRE_THRESHOLD}")
        print(f"{'─' * 65}")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  Recall   : {recall:.3f}  ← primary metric (catching REVERSAL)")
        print(f"  Precision: {prec:.3f}  (of fired signals, how many were REVERSAL)")
        print(f"  F2 score : {f2:.3f}   (recall-weighted combined metric)")
        print(f"\n  At P ≥ {REVERSAL_FIRE_THRESHOLD}: blocking {y_pred.mean():.1%} of candles as REVERSAL")
        print(f"\n  Full classification report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['NOT_REVERSAL', 'REVERSAL'],
                                    digits=3, zero_division=0))

        # Feature importances
        imps = sorted(zip(avail_cols, model.feature_importances_),
                      key=lambda x: x[1], reverse=True)
        print("  Feature importances (binary model):")
        for feat, imp in imps:
            print(f"    {feat:<30}: {imp:.4f}  {'█' * int(imp * 300)}")

    # ── Save model ────────────────────────────────────────────────
    model.save_model(REVERSAL_DETECTOR_PATH)

    meta = {
        "trained_date":          datetime.now().strftime('%Y-%m-%d %H:%M'),
        "model_type":            "binary_reversal_prefilter",
        "fire_threshold":        REVERSAL_FIRE_THRESHOLD,
        "feature_cols":          avail_cols,
        "n_features":            len(avail_cols),
        "n_trees_used":          n_trees,
        "scale_pos_weight":      round(spw, 2),
        "test_accuracy":         round(acc, 4),
        "test_recall":           round(recall, 4),
        "test_precision":        round(prec, 4),
        "test_f2":               round(f2, 4),
        "train_reversal_rate":   round(float(y_train.mean()), 4),
        "test_reversal_rate":    round(float(y_test.mean()), 4),
        "n_train":               int(len(y_train)),
        "n_test":                int(len(y_test)),
    }
    with open(REVERSAL_DETECTOR_META_PATH, 'w') as f:
        json.dump(meta, f, indent=4)

    if verbose:
        print(f"\n  ✅ Saved: {REVERSAL_DETECTOR_PATH}")
        print(f"  ✅ Saved: {REVERSAL_DETECTOR_META_PATH}")

    # Reload into module state immediately
    global _reversal_model, _reversal_cols, _loaded
    _reversal_model = model
    _reversal_cols  = avail_cols
    _loaded         = True

    return {
        "accuracy":  acc,
        "recall":    recall,
        "precision": prec,
        "f2":        f2,
        "n_trees":   n_trees,
    }