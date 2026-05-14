"""
walkforward_trainer.py — Walk-Forward Model Training
======================================================

Trains N expanding-window models, evaluates each on its own out-of-sample
test window, and promotes the best-performing model as the challenger for
the champion/challenger comparison in auto_retrainer.py.

EXPANDING WINDOW DESIGN
────────────────────────
Each window trains on ALL data up to the split point (expanding, not rolling).
This mirrors how the bot will actually use the model in live trading — it
always has access to all historical data, not just a fixed recent window.

Example with 5 windows and 8 years of data (2017–2024):

    Window 1 — Train: 2017–2020  |  Test: 2021
    Window 2 — Train: 2017–2021  |  Test: 2022
    Window 3 — Train: 2017–2022  |  Test: 2023
    Window 4 — Train: 2017–2023  |  Test: 2024
    Window 5 — Train: 2017–2024  |  Test: 2025 (most recent)

SELECTION CRITERIA
───────────────────
Each window produces an in-process evaluation using model_evaluator metrics.
The window whose test-set composite score is highest gets its model files
copied to the output_dir as the final challenger.

Why not average the scores and retrain on all data?
    The final model trained on ALL data cannot be evaluated on any holdout
    (there's no unseen data left).  Picking the best window's model means
    we have a model that was already evaluated out-of-sample and passed.

INTEGRATION
────────────
Called from auto_retrainer._run_retrain() when USE_WALKFORWARD = True:

    from walkforward_trainer import run_walkforward
    success = run_walkforward(
        output_dir = challenger_dir,   # where to write winning model files
        n_windows  = 5,
    )

PUBLIC API
──────────
run_walkforward(output_dir, n_windows, verbose) → bool
"""

import os
import sys
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CURRENT_DIR)

from paths import (FEATURES_PATH, LABELS_PATH, REGIME_DATA,
                   create_all_dirs as _cad_wf)
_cad_wf()

# ── Configuration ──────────────────────────────────────────────────
DEFAULT_N_WINDOWS      = 5      # number of expanding windows
MIN_TRAIN_YEARS        = 3      # minimum years of data for a valid training window
TEST_WINDOW_MONTHS     = 12     # months of data used as test set per window
MIN_TEST_ROWS          = 2000   # minimum candles in test window to count
MIN_COMPOSITE_TO_KEEP  = 0.30   # discard windows whose model scores below this
N_HMM_INIT             = 10     # inits per window — fewer than full train (speed)


# ================================================================
# WALK-FORWARD WINDOW GENERATOR
# ================================================================

def _generate_windows(index: pd.DatetimeIndex, n_windows: int,
                       test_months: int = TEST_WINDOW_MONTHS):
    """
    Generates expanding train/test splits from a DatetimeIndex.

    Each split:
        train_end   = some date T
        test_start  = T
        test_end    = T + test_months

    Windows are spaced evenly across the data so that the last window's
    test set covers the most recent period.

    Yields:
        (train_end_idx: int, test_end_idx: int, window_label: str)
    """
    from dateutil.relativedelta import relativedelta

    n = len(index)
    total_days  = (index[-1] - index[0]).days
    test_days   = test_months * 30

    # Ensure minimum train data exists before first window
    min_train_days = MIN_TRAIN_YEARS * 365
    first_test_start = index[0] + pd.Timedelta(days=min_train_days)

    # Last window test ends at the final row
    last_test_end = index[-1]
    last_test_start = last_test_end - pd.Timedelta(days=test_days)

    if last_test_start <= first_test_start:
        print(f"[WF] Not enough data for {n_windows} windows with "
              f"{MIN_TRAIN_YEARS}yr minimum train. "
              f"Generating as many as possible.")

    # Generate window test-start dates, evenly spaced from first_test_start
    # to last_test_start (inclusive of both endpoints)
    range_days = max(1, (last_test_start - first_test_start).days)
    step_days  = range_days / max(n_windows - 1, 1)

    windows = []
    for i in range(n_windows):
        test_start_dt  = first_test_start + pd.Timedelta(days=int(i * step_days))
        test_end_dt    = test_start_dt + pd.Timedelta(days=test_days)

        # Find integer indices
        train_end_idx  = index.searchsorted(test_start_dt, side='left')
        test_end_idx   = min(index.searchsorted(test_end_dt, side='right'), n)

        if train_end_idx < 100 or (test_end_idx - train_end_idx) < MIN_TEST_ROWS:
            continue

        label = (f"W{i+1}  "
                 f"train: {index[0].date()}→{index[train_end_idx-1].date()} "
                 f"test: {index[train_end_idx].date()}→{index[test_end_idx-1].date()}")
        windows.append((train_end_idx, test_end_idx, label))

    return windows


# ================================================================
# SINGLE WINDOW TRAINING
# ================================================================

def _train_window(features: pd.DataFrame,
                  labels: pd.Series,
                  train_end_idx: int,
                  test_end_idx: int,
                  window_dir: str,
                  window_label: str,
                  verbose: bool = True) -> Optional[dict]:
    """
    Trains a single walk-forward window using the in-process trainer API
    (not subprocess) for speed.

    Args:
        features:      full feature matrix
        labels:        full label series
        train_end_idx: index of last training row (exclusive)
        test_end_idx:  index of last test row (exclusive)
        window_dir:    directory to write this window's model files
        window_label:  human-readable label for logging

    Returns:
        dict with window evaluation metrics, or None on failure.
    """
    os.makedirs(window_dir, exist_ok=True)

    X_train = features.iloc[:train_end_idx]
    y_train = labels.iloc[:train_end_idx]
    X_test  = features.iloc[train_end_idx:test_end_idx]
    y_test  = labels.iloc[train_end_idx:test_end_idx]

    if verbose:
        print(f"\n[WF] {window_label}")
        print(f"     Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    try:
        import joblib
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score
        from sklearn.utils.class_weight import compute_class_weight
        from feature_engineer import apply_rolling_zscore, TF_FEATURE_COLS, FEATURE_COLS
        from trainer import (
            HMM_OBS_COLS, DEFAULT_N_MIX, DEFAULT_N_ITER,
            fit_gmmhmm_best, map_states_to_regimes, decode_regimes
        )
        from labeler import ALL_REGIMES
        from paths import OPTUNA_PARAMS_PATH
    except ImportError as e:
        print(f"[WF] Import error: {e}")
        return None

    # ── HMM: fit on training data only ────────────────────────────
    hmm_cols = [c for c in HMM_OBS_COLS if c in X_train.columns]
    if len(hmm_cols) < 4:
        hmm_cols = [c for c in X_train.columns
                    if any(k in c for k in ['atr_ratio', 'adx', 'momentum_20'])][:9]

    from sklearn.preprocessing import StandardScaler
    hmm_scaler = StandardScaler()
    hmm_obs    = hmm_scaler.fit_transform(X_train[hmm_cols].values)

    hmm_model = fit_gmmhmm_best(hmm_obs, n_states=5, n_mix=DEFAULT_N_MIX,
                                n_iter=DEFAULT_N_ITER, n_init=N_HMM_INIT)
    if hmm_model is None:
        print(f"[WF]   HMM training failed for this window — skipping.")
        return None

    state_map = map_states_to_regimes(hmm_model, hmm_cols)

    # Decode train labels using this window's HMM
    train_labels = pd.Series(
        decode_regimes(hmm_model, hmm_obs, state_map),
        index=X_train.index, name='regime',
    )

    # ── XGBoost: train on window's train set ──────────────────────
    base_avail = [c for c in FEATURE_COLS if c in X_train.columns]
    X_xgb      = X_train[base_avail].copy()
    X_xgb      = apply_rolling_zscore(
        X_xgb, cols=[c for c in TF_FEATURE_COLS if c in X_xgb.columns], window=500)

    y_xgb = train_labels.reindex(X_xgb.index)
    valid  = X_xgb.notna().all(axis=1) & y_xgb.notna()
    X_xgb, y_xgb = X_xgb[valid], y_xgb[valid]

    le = LabelEncoder()
    le.fit(ALL_REGIMES)
    y_enc = le.transform(y_xgb)

    # Class weights
    classes  = np.unique(y_enc)
    weights  = compute_class_weight("balanced", classes=classes, y=y_enc)
    wt_map   = dict(zip(classes, weights))
    sw       = np.array([wt_map[yi] for yi in y_enc])

    # XGBoost params: load Optuna-tuned if available
    xgb_params = dict(
        objective='multi:softprob', num_class=len(le.classes_),
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, gamma=0.2,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='mlogloss', random_state=42, n_jobs=-1,
        verbosity=0, early_stopping_rounds=30,
    )
    try:
        if os.path.exists(OPTUNA_PARAMS_PATH):
            with open(OPTUNA_PARAMS_PATH) as _f:
                _optuna_p = json.load(_f).get("best_params", {})
            if _optuna_p:
                xgb_params.update(_optuna_p)
    except Exception:
        pass

    # Val split (last 15% of train)
    n_tr    = len(X_xgb)
    val_idx = int(n_tr * 0.85)
    X_tr_arr, y_tr = X_xgb.values[:val_idx], y_enc[:val_idx]
    X_val_arr, y_val = X_xgb.values[val_idx:], y_enc[val_idx:]
    sw_tr   = sw[:val_idx]

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr_arr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_val_arr, y_val)], verbose=False)

    # ── Evaluate on this window's test set ─────────────────────────
    X_test_z = X_test[base_avail].copy()
    X_test_z = apply_rolling_zscore(
        X_test_z, cols=[c for c in TF_FEATURE_COLS if c in X_test_z.columns],
        window=500)

    # Decode test labels using incumbent (not window) HMM — neutral reference
    # Use labeler rule-based labels from labels.csv if available, else skip
    y_test_enc = le.transform(y_test.reindex(X_test_z.dropna().index))
    y_pred_enc = xgb_model.predict(X_test_z.dropna().values)
    test_acc   = float(accuracy_score(y_test_enc, y_pred_enc))

    # Directional accuracy
    from model_evaluator import _BULL, _BEAR, _REVERSAL
    y_test_names = le.inverse_transform(y_test_enc)
    y_pred_names = le.inverse_transform(y_pred_enc)
    dir_mask = np.isin(y_test_names, [_BULL, _BEAR])
    dir_acc  = (float(np.mean(y_pred_names[dir_mask] == y_test_names[dir_mask]))
                if dir_mask.any() else 0.0)

    bull_mask    = y_test_names == _BULL
    bear_mask    = y_test_names == _BEAR
    bb_conf      = 0.0
    if bull_mask.any() and bear_mask.any():
        bb_conf = (np.mean(y_pred_names[bull_mask] == _BEAR) +
                   np.mean(y_pred_names[bear_mask] == _BULL)) / 2.0

    rev_mask = y_test_names == _REVERSAL
    rev_rec  = (float(np.mean(y_pred_names[rev_mask] == _REVERSAL))
                if rev_mask.any() else 0.0)

    from model_evaluator import W_DIRECTIONAL, W_OVERALL, W_REVERSAL, W_CONFUSION
    composite = (W_DIRECTIONAL * dir_acc + W_OVERALL * test_acc
                 + W_REVERSAL * rev_rec - W_CONFUSION * bb_conf)

    if verbose:
        print(f"     accuracy={test_acc:.3f}  directional={dir_acc:.3f}  "
              f"bb_confusion={bb_conf:.3f}  reversal_recall={rev_rec:.3f}  "
              f"composite={composite:.4f}")

    # ── Save window model to window_dir ───────────────────────────
    joblib.dump(hmm_model,  os.path.join(window_dir, "gmmhmm_model.joblib"),  compress=3)
    joblib.dump(hmm_scaler, os.path.join(window_dir, "hmm_scaler.joblib"),    compress=3)
    joblib.dump(le,         os.path.join(window_dir, "label_encoder.joblib"), compress=3)
    xgb_model.save_model(   os.path.join(window_dir, "regime_model.ubj"))

    meta = {
        "trained_date":        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        "architecture":        "GMM-HMM (5-state) + XGBoost — walk-forward window",
        "walkforward_label":   window_label,
        "test_accuracy":       round(test_acc, 4),
        "directional_accuracy":round(dir_acc, 4),
        "bull_bear_confusion": round(float(bb_conf), 4),
        "reversal_recall":     round(rev_rec, 4),
        "composite_score":     round(composite, 4),
        "n_train":             int(train_end_idx),
        "n_test":              int(test_end_idx - train_end_idx),
        "feature_cols":        base_avail,
        "n_feature_cols":      len(base_avail),
        "hmm_obs_cols":        hmm_cols,
        "state_map":           {str(k): v for k, v in state_map.items()},
        "regimes":             ALL_REGIMES,
    }
    with open(os.path.join(window_dir, "model_meta.json"), 'w') as f:
        json.dump(meta, f, indent=4)

    return {
        "window_dir":          window_dir,
        "window_label":        window_label,
        "test_accuracy":       test_acc,
        "directional_accuracy":dir_acc,
        "bull_bear_confusion": float(bb_conf),
        "reversal_recall":     rev_rec,
        "composite_score":     composite,
        "n_train":             train_end_idx,
        "n_test":              test_end_idx - train_end_idx,
    }


# ================================================================
# MAIN ENTRY POINT
# ================================================================

def run_walkforward(output_dir: str,
                    n_windows: int = DEFAULT_N_WINDOWS,
                    verbose: bool = True) -> bool:
    """
    Runs N walk-forward windows, evaluates each, and copies the
    best-scoring window's model files to output_dir.

    Args:
        output_dir: where to write the winning model files
                    (this becomes the challenger_dir in auto_retrainer)
        n_windows:  number of expanding windows
        verbose:    print per-window progress

    Returns:
        True if at least one window succeeded and a model was written.
        False if all windows failed or no model passed MIN_COMPOSITE_TO_KEEP.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"  WALK-FORWARD TRAINING  ({n_windows} windows)")
    print("=" * 65)

    # Load features + labels
    if not os.path.exists(FEATURES_PATH):
        print(f"[WF] features.csv not found at {FEATURES_PATH}")
        return False
    try:
        features = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
        labels_df = pd.read_csv(LABELS_PATH, index_col=0, parse_dates=True)
        labels    = labels_df.iloc[:, 0]
    except Exception as e:
        print(f"[WF] Could not load features/labels: {e}")
        return False

    shared_idx = features.index.intersection(labels.index)
    features   = features.loc[shared_idx]
    labels     = labels.loc[shared_idx]

    print(f"\n[WF] Data: {len(features):,} candles | "
          f"{features.index[0].date()} → {features.index[-1].date()}")

    # Generate windows
    windows = _generate_windows(features.index, n_windows)
    if not windows:
        print("[WF] Could not generate any valid windows. "
              "Need more data or fewer windows.")
        return False

    print(f"[WF] Generated {len(windows)} windows:")
    for _, _, lbl in windows:
        print(f"   {lbl}")
    print()

    # Train each window
    window_results = []
    base_tmp = os.path.join(output_dir, "_windows")
    os.makedirs(base_tmp, exist_ok=True)

    for i, (train_end_idx, test_end_idx, label) in enumerate(windows):
        window_dir = os.path.join(base_tmp, f"window_{i+1}")
        result = _train_window(
            features, labels,
            train_end_idx, test_end_idx,
            window_dir, label, verbose=verbose,
        )
        if result is not None:
            window_results.append(result)

    if not window_results:
        print("[WF] All windows failed to train.")
        return False

    # ── Select best window by composite score ─────────────────────
    window_results.sort(key=lambda r: r["composite_score"], reverse=True)
    best = window_results[0]

    print(f"\n[WF] ─── Walk-Forward Results ───")
    for r in window_results:
        flag = " ← BEST" if r is best else ""
        print(f"  composite={r['composite_score']:.4f}  "
              f"directional={r['directional_accuracy']:.3f}  "
              f"accuracy={r['test_accuracy']:.3f}  {r['window_label'][:60]}{flag}")

    if best["composite_score"] < MIN_COMPOSITE_TO_KEEP:
        print(f"\n[WF] Best window composite {best['composite_score']:.4f} "
              f"< threshold {MIN_COMPOSITE_TO_KEEP}. "
              f"No model is good enough to promote.")
        return False

    # Copy best window's model files to output_dir
    best_dir = best["window_dir"]
    MODEL_FILENAMES = [
        "gmmhmm_model.joblib", "hmm_scaler.joblib",
        "regime_model.ubj",    "label_encoder.joblib",
        "model_meta.json",
    ]
    copied = 0
    for fname in MODEL_FILENAMES:
        src = os.path.join(best_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1

    # Write walk-forward summary to challenger dir
    summary = {
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_windows":    len(windows),
        "n_trained":    len(window_results),
        "best_window":  best["window_label"],
        "best_composite": round(best["composite_score"], 4),
        "all_results":  window_results,
    }
    with open(os.path.join(output_dir, "walkforward_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    # Cleanup temp window dirs
    try:
        shutil.rmtree(base_tmp, ignore_errors=True)
    except Exception:
        pass

    print(f"\n[WF] ✅ Walk-forward complete.")
    print(f"[WF] Best window: {best['window_label'][:70]}")
    print(f"[WF] Composite: {best['composite_score']:.4f}  "
          f"Directional: {best['directional_accuracy']:.3f}  "
          f"Accuracy: {best['test_accuracy']:.3f}")
    print(f"[WF] Model files written to: {output_dir}  ({copied} files)")

    # ── Train REVERSAL pre-filter on full dataset ─────────────────
    # Walk-forward picks the best window's model as challenger.
    # The REVERSAL binary model trains on ALL data (not per-window)
    # because REVERSAL events (~4-6% of candles) need the full history.
    print(f"\n[WF] Training REVERSAL binary pre-filter on full dataset...")
    try:
        from reversal_detector import train as train_reversal
        rev_results = train_reversal(features, labels, verbose=False)
        if rev_results:
            rev_dst = os.path.join(output_dir, "reversal_detector.ubj")
            from paths import REVERSAL_DETECTOR_PATH, REVERSAL_DETECTOR_META_PATH
            if os.path.exists(REVERSAL_DETECTOR_PATH):
                shutil.copy2(REVERSAL_DETECTOR_PATH, rev_dst)
            print(f"[WF] REVERSAL pre-filter: recall={rev_results.get('recall',0):.3f} "
                  f"F2={rev_results.get('f2',0):.3f}")
    except Exception as e:
        print(f"[WF] REVERSAL pre-filter skipped: {e}")

    return copied >= 4   # need at least xgb + le + hmm + meta
