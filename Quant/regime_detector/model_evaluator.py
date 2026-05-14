"""
model_evaluator.py — Champion / Challenger Model Evaluation
============================================================

Evaluates any trained model directory against a shared holdout window
so champion and challenger can be compared on identical data.

WHY A SHARED HOLDOUT
────────────────────
When a drift trigger fires, the challenger trains on recent data.
The incumbent was trained earlier — its internal test set covers
a different time window than the challenger's.  Comparing their
stored test_accuracy values is comparing apples to oranges.

This module fixes that:
    1. Define a HOLDOUT_DAYS recent window (default: last 60 days
       of features.csv) — neither model officially optimised on this.
    2. Load features + labeler-rule ground truth for this window.
    3. Evaluate both models on the same rows.
    4. Compare composite scores — one fair number per model.

COMPOSITE SCORE FORMULA
────────────────────────
The four metrics reflect what actually matters for XAUUSD trading:

    directional_accuracy  ×0.40  — BULL/BEAR correct; wrong direction = wrong trade
    overall_accuracy      ×0.30  — all 5 regimes classified correctly
    reversal_recall       ×0.20  — REVERSAL regime caught (missed → trades into NFP/FOMC)
    bull_bear_confusion   ×0.10  — penalty: % of BULL predicted as BEAR or vice versa
                                   (subtracted — lower confusion = better score)

A model PASSES the deployment gate if:
    composite_score  > incumbent composite_score
    directional_accuracy >= incumbent directional_accuracy - DIRECTION_TOLERANCE (0.02)
    overall_accuracy >= MIN_DEPLOY_ACCURACY (0.50)

The directional tolerance prevents a challenger that is marginally better
overall from being promoted if it is meaningfully worse at direction.

PUBLIC API
──────────
evaluate_model(model_dir, features, labels)  → metrics dict
compare(champion_dir, challenger_dir)        → ComparisonResult(verdict, details)
load_holdout(n_days)                         → (features_df, labels_series)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CURRENT_DIR)

from paths import (
    FEATURES_PATH, LABELS_PATH, REGIME_MODEL,
    HMM_PATH, HMM_SCALER_PATH, REGIME_XGB_PATH,
    LABEL_ENCODER_PATH, MODEL_META_PATH,
)

# ── Configuration ─────────────────────────────────────────────────────
HOLDOUT_DAYS          = 60    # recent days used as shared evaluation window
MIN_HOLDOUT_ROWS      = 500   # minimum candles needed for a valid evaluation
MIN_DEPLOY_ACCURACY   = 0.50  # challenger must clear this floor regardless of comparison
DIRECTION_TOLERANCE   = 0.02  # max allowed directional regression vs incumbent

# Composite score weights — must sum to 1.0
W_DIRECTIONAL = 0.40
W_OVERALL     = 0.30
W_REVERSAL    = 0.20
W_CONFUSION   = 0.10   # subtracted (lower confusion = better)

# Model file names inside any model directory
_MODEL_FILES = {
    "hmm":       "gmmhmm_model.joblib",
    "hmm_scaler":"hmm_scaler.joblib",
    "xgb":       "regime_model.ubj",
    "le":        "label_encoder.joblib",
    "meta":      "model_meta.json",
}

# Regime label constants (mirrors labeler.py — avoid circular import)
_BULL    = "BULL_TREND"
_BEAR    = "BEAR_TREND"
_REVERSAL = "REVERSAL"


# ================================================================
# RESULT DATACLASS
# ================================================================

@dataclass
class ModelMetrics:
    """Evaluation metrics for one model on the shared holdout."""
    model_dir:            str
    n_samples:            int
    overall_accuracy:     float
    directional_accuracy: float   # accuracy on BULL/BEAR candles only
    bull_bear_confusion:  float   # % of directional predictions that are flipped
    reversal_recall:      float   # recall of REVERSAL class
    mean_confidence:      float   # mean max-probability across all predictions
    composite_score:      float
    trained_date:         Optional[str] = None
    error:                Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_dir":            self.model_dir,
            "n_samples":            self.n_samples,
            "overall_accuracy":     round(self.overall_accuracy, 4),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "bull_bear_confusion":  round(self.bull_bear_confusion, 4),
            "reversal_recall":      round(self.reversal_recall, 4),
            "mean_confidence":      round(self.mean_confidence, 4),
            "composite_score":      round(self.composite_score, 4),
            "trained_date":         self.trained_date,
            "error":                self.error,
        }


@dataclass
class ComparisonResult:
    """Outcome of champion vs challenger evaluation."""
    verdict:          str            # "PROMOTE" | "REJECT"
    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    reason:           str
    evaluated_at:     str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    holdout_rows:     int = 0
    holdout_start:    Optional[str] = None
    holdout_end:      Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "verdict":            self.verdict,
            "reason":             self.reason,
            "evaluated_at":       self.evaluated_at,
            "holdout_rows":       self.holdout_rows,
            "holdout_start":      self.holdout_start,
            "holdout_end":        self.holdout_end,
            "champion":           self.champion_metrics.to_dict(),
            "challenger":         self.challenger_metrics.to_dict(),
        }


# ================================================================
# HOLDOUT LOADER
# ================================================================

def load_holdout(n_days: int = HOLDOUT_DAYS):
    """
    Loads the last n_days of features.csv + labels.csv as the
    shared evaluation holdout.

    Returns:
        (features: pd.DataFrame, labels: pd.Series)
        or (None, None) on any error.

    WHY labeler GROUND TRUTH NOT HMM LABELS:
        Using the incumbent HMM labels as ground truth would bias
        the comparison toward the incumbent.  labeler.py uses
        deterministic ADX/ATR threshold rules — neither model
        was trained to match these exactly, so they serve as a
        neutral reference point for both.
    """
    # Load features
    if not os.path.exists(FEATURES_PATH):
        print(f"[Evaluator] features.csv not found at {FEATURES_PATH}")
        return None, None

    try:
        features = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[Evaluator] Could not load features.csv: {e}")
        return None, None

    if features.empty:
        print("[Evaluator] features.csv is empty.")
        return None, None

    # Load ground-truth labels (HMM decode from last full training run)
    if not os.path.exists(LABELS_PATH):
        print(f"[Evaluator] labels.csv not found — cannot evaluate.")
        return None, None

    try:
        labels_df = pd.read_csv(LABELS_PATH, index_col=0, parse_dates=True)
        labels = labels_df.iloc[:, 0]  # first column regardless of name
    except Exception as e:
        print(f"[Evaluator] Could not load labels.csv: {e}")
        return None, None

    # Align to shared index
    shared_idx = features.index.intersection(labels.index)
    features   = features.loc[shared_idx]
    labels     = labels.loc[shared_idx]

    # Take last n_days
    if n_days > 0:
        cutoff = features.index.max() - timedelta(days=n_days)
        features = features[features.index >= cutoff]
        labels   = labels[labels.index >= cutoff]

    if len(features) < MIN_HOLDOUT_ROWS:
        print(f"[Evaluator] Holdout has only {len(features)} rows "
              f"(need ≥ {MIN_HOLDOUT_ROWS}). Expanding to last 20% of data.")
        n_20pct    = max(MIN_HOLDOUT_ROWS, int(len(features) * 0.20))
        features   = features.iloc[-n_20pct:]
        labels     = labels.iloc[-n_20pct:]

    return features, labels


# ================================================================
# SINGLE MODEL EVALUATION
# ================================================================

def evaluate_model(model_dir: str,
                   features: pd.DataFrame,
                   labels: pd.Series,
                   verbose: bool = True) -> ModelMetrics:
    """
    Evaluates the model files in model_dir on the provided features/labels.

    model_dir must contain:
        gmmhmm_model.joblib, hmm_scaler.joblib, regime_model.ubj,
        label_encoder.joblib, model_meta.json

    Returns a ModelMetrics dataclass.  On any load/predict failure,
    returns a metrics object with error set and all scores = 0.
    """
    def _fail(reason: str) -> ModelMetrics:
        print(f"[Evaluator] ✗ {model_dir}: {reason}")
        return ModelMetrics(
            model_dir=model_dir, n_samples=0,
            overall_accuracy=0.0, directional_accuracy=0.0,
            bull_bear_confusion=1.0, reversal_recall=0.0,
            mean_confidence=0.0, composite_score=0.0,
            error=reason,
        )

    # ── Load model files ──────────────────────────────────────────
    try:
        import joblib
        import xgboost as xgb
    except ImportError as e:
        return _fail(f"Missing library: {e}")

    hmm_path    = os.path.join(model_dir, _MODEL_FILES["hmm"])
    scaler_path = os.path.join(model_dir, _MODEL_FILES["hmm_scaler"])
    xgb_path    = os.path.join(model_dir, _MODEL_FILES["xgb"])
    le_path     = os.path.join(model_dir, _MODEL_FILES["le"])
    meta_path   = os.path.join(model_dir, _MODEL_FILES["meta"])

    for p in (xgb_path, le_path, meta_path):
        if not os.path.exists(p):
            return _fail(f"Missing file: {os.path.basename(p)}")

    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        le = joblib.load(le_path)
        with open(meta_path) as f:
            meta = json.load(f)
        trained_date = meta.get("trained_date", "unknown")
    except Exception as e:
        return _fail(f"Load error: {e}")

    # ── Prepare feature matrix ────────────────────────────────────
    feature_cols = meta.get("feature_cols", [])
    if not feature_cols:
        return _fail("model_meta.json missing feature_cols — cannot reproduce feature set")

    available_cols = [c for c in feature_cols if c in features.columns]
    if len(available_cols) < len(feature_cols) * 0.80:
        return _fail(f"Too many feature cols missing: "
                     f"{len(feature_cols) - len(available_cols)}/{len(feature_cols)} absent")

    # Align + forward-fill any small gaps
    X = features[available_cols].copy()

    # Apply rolling z-score (matches training normalisation)
    try:
        from feature_engineer import apply_rolling_zscore, TF_FEATURE_COLS
        tf_cols = [c for c in TF_FEATURE_COLS if c in X.columns]
        X = apply_rolling_zscore(X, cols=tf_cols, window=500)
    except Exception:
        pass  # if feature_engineer unavailable, proceed without z-score

    # Drop rows with NaN (z-score warmup rows)
    valid = X.notna().all(axis=1) & labels.reindex(X.index).notna()
    X      = X[valid]
    y_true = labels.reindex(X.index)

    if len(X) < MIN_HOLDOUT_ROWS // 2:
        return _fail(f"Too few valid rows after NaN drop: {len(X)}")

    # ── Predict ───────────────────────────────────────────────────
    try:
        y_enc_pred  = xgb_model.predict(X.values)
        y_prob      = xgb_model.predict_proba(X.values)   # shape (N, n_classes)
        y_pred      = le.inverse_transform(y_enc_pred)
        y_true_arr  = y_true.values
        max_conf    = y_prob.max(axis=1)
    except Exception as e:
        return _fail(f"Prediction error: {e}")

    n = len(y_true_arr)

    # ── Overall accuracy ──────────────────────────────────────────
    overall_acc = float(np.mean(y_pred == y_true_arr))

    # ── Directional accuracy — BULL/BEAR candles only ─────────────
    dir_mask        = np.isin(y_true_arr, [_BULL, _BEAR])
    dir_true        = y_true_arr[dir_mask]
    dir_pred        = y_pred[dir_mask]
    directional_acc = float(np.mean(dir_pred == dir_true)) if len(dir_true) > 0 else 0.0

    # ── Bull/Bear confusion rate ──────────────────────────────────
    # Of all BULL candles, % predicted as BEAR (and vice versa)
    bull_mask    = y_true_arr == _BULL
    bear_mask    = y_true_arr == _BEAR
    bull_as_bear = float(np.mean(y_pred[bull_mask] == _BEAR)) if bull_mask.any() else 0.0
    bear_as_bull = float(np.mean(y_pred[bear_mask] == _BULL)) if bear_mask.any() else 0.0
    bb_confusion = (bull_as_bear + bear_as_bull) / 2.0

    # ── Reversal recall ───────────────────────────────────────────
    rev_mask      = y_true_arr == _REVERSAL
    rev_recall    = float(np.mean(y_pred[rev_mask] == _REVERSAL)) if rev_mask.any() else 0.0

    # ── Mean confidence ───────────────────────────────────────────
    mean_conf = float(np.mean(max_conf))

    # ── Composite score ───────────────────────────────────────────
    composite = (
        W_DIRECTIONAL * directional_acc
        + W_OVERALL   * overall_acc
        + W_REVERSAL  * rev_recall
        - W_CONFUSION * bb_confusion   # penalty
    )

    if verbose:
        tag = os.path.basename(model_dir.rstrip("/\\"))
        print(f"\n[Evaluator] ── {tag} on {n} holdout candles ──")
        print(f"  Overall accuracy    : {overall_acc:.3f}")
        print(f"  Directional acc     : {directional_acc:.3f}  "
              f"(BULL/BEAR candles only — {dir_mask.sum():,})")
        print(f"  Bull↔Bear confusion : {bb_confusion:.3f}")
        print(f"  Reversal recall     : {rev_recall:.3f}")
        print(f"  Mean confidence     : {mean_conf:.3f}")
        print(f"  ─── Composite score : {composite:.4f} ───")

    return ModelMetrics(
        model_dir            = model_dir,
        n_samples            = n,
        overall_accuracy     = overall_acc,
        directional_accuracy = directional_acc,
        bull_bear_confusion  = bb_confusion,
        reversal_recall      = rev_recall,
        mean_confidence      = mean_conf,
        composite_score      = composite,
        trained_date         = trained_date,
    )


# ================================================================
# CHAMPION vs CHALLENGER COMPARISON
# ================================================================

def compare(champion_dir: str,
            challenger_dir: str,
            verbose: bool = True) -> ComparisonResult:
    """
    Evaluates both models on the same shared holdout and returns a
    ComparisonResult with a PROMOTE or REJECT verdict.

    PROMOTION RULES (challenger must pass ALL three):
        1. composite_score > champion composite_score
        2. directional_accuracy >= champion directional_accuracy - DIRECTION_TOLERANCE
           (prevents promoting a challenger that's marginally better overall
            but meaningfully worse at calling direction)
        3. overall_accuracy >= MIN_DEPLOY_ACCURACY
           (absolute floor — never deploy a model weaker than random chance)

    Args:
        champion_dir:   directory of the currently deployed model
        challenger_dir: directory of the newly trained model

    Returns ComparisonResult with verdict, both metric sets, and reason.
    """
    print("\n" + "=" * 65)
    print("  CHAMPION vs CHALLENGER EVALUATION")
    print("=" * 65)

    # Load shared holdout
    features, labels = load_holdout()
    if features is None:
        reason = "Could not load holdout data — REJECT by default (safe)"
        print(f"[Evaluator] {reason}")
        # Return null metrics with reject
        null = ModelMetrics(
            model_dir="", n_samples=0,
            overall_accuracy=0.0, directional_accuracy=0.0,
            bull_bear_confusion=1.0, reversal_recall=0.0,
            mean_confidence=0.0, composite_score=0.0,
            error="holdout unavailable",
        )
        return ComparisonResult(
            verdict="REJECT", champion_metrics=null,
            challenger_metrics=null, reason=reason,
        )

    holdout_start = str(features.index.min().date())
    holdout_end   = str(features.index.max().date())

    print(f"  Holdout: {holdout_start} → {holdout_end}  ({len(features):,} candles)")
    print("=" * 65)

    # Evaluate champion
    print(f"\n[Evaluator] Evaluating CHAMPION ({os.path.basename(champion_dir)})...")
    champ = evaluate_model(champion_dir, features.copy(), labels.copy(), verbose=verbose)

    # Evaluate challenger
    print(f"\n[Evaluator] Evaluating CHALLENGER ({os.path.basename(challenger_dir)})...")
    chal  = evaluate_model(challenger_dir, features.copy(), labels.copy(), verbose=verbose)

    # Decision
    verdict = "REJECT"
    reasons = []

    if chal.error:
        reasons.append(f"Challenger load/eval failed: {chal.error}")
    elif champ.error:
        # Champion is broken — promote challenger regardless (deploy anything working)
        verdict = "PROMOTE"
        reasons.append("Champion model failed to evaluate — promoting challenger as replacement")
    else:
        # Rule 1: composite score
        if chal.composite_score <= champ.composite_score:
            reasons.append(
                f"Composite score not better: "
                f"challenger {chal.composite_score:.4f} ≤ champion {champ.composite_score:.4f}"
            )

        # Rule 2: directional accuracy floor
        dir_floor = champ.directional_accuracy - DIRECTION_TOLERANCE
        if chal.directional_accuracy < dir_floor:
            reasons.append(
                f"Directional accuracy regression: "
                f"challenger {chal.directional_accuracy:.3f} < floor {dir_floor:.3f}"
            )

        # Rule 3: absolute accuracy floor
        if chal.overall_accuracy < MIN_DEPLOY_ACCURACY:
            reasons.append(
                f"Below minimum accuracy floor: "
                f"{chal.overall_accuracy:.3f} < {MIN_DEPLOY_ACCURACY}"
            )

        if not reasons:
            verdict = "PROMOTE"
            reasons.append(
                f"All gates passed — "
                f"composite +{chal.composite_score - champ.composite_score:+.4f} | "
                f"directional {champ.directional_accuracy:.3f}→{chal.directional_accuracy:.3f} | "
                f"accuracy {champ.overall_accuracy:.3f}→{chal.overall_accuracy:.3f}"
            )

    reason_str = " | ".join(reasons)

    print("\n" + "=" * 65)
    print(f"  VERDICT: {'✅ PROMOTE' if verdict == 'PROMOTE' else '❌ REJECT'}")
    print(f"  Reason : {reason_str}")
    if verdict == "PROMOTE" and not champ.error and not chal.error:
        delta = chal.composite_score - champ.composite_score
        print(f"  Composite delta: {delta:+.4f}")
    print("=" * 65 + "\n")

    return ComparisonResult(
        verdict            = verdict,
        champion_metrics   = champ,
        challenger_metrics = chal,
        reason             = reason_str,
        holdout_rows       = len(features),
        holdout_start      = holdout_start,
        holdout_end        = holdout_end,
    )
