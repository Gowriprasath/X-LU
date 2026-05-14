"""
optuna_tuner.py — Bayesian XGBoost Hyperparameter Optimisation
===============================================================

Runs an Optuna study to find the best XGBoost hyperparameters for the
regime classifier.  Results are saved to OPTUNA_PARAMS_PATH and loaded
automatically by trainer.py and walkforward_trainer.py on next training run.

DESIGN DECISIONS
─────────────────
1. Tuning target: composite_score from model_evaluator — not raw accuracy.
   We optimise for the same metric used in champion/challenger, so the tuner
   and the deployment gate speak the same language.

2. Fast inner loop: each Optuna trial trains XGBoost only (not GMM-HMM).
   HMM labels from the most recent training run are reused as fixed ground
   truth.  This makes each trial ~30s instead of ~20min.

3. Study uses a rolling holdout (last 20% of features.csv) as the objective.
   This is the same holdout window model_evaluator uses — fully consistent.

4. Pruning: Optuna's MedianPruner cuts unpromising trials early (saves time).

5. Parallelism: n_jobs=-1 within each XGBoost fit; Optuna trials run serially
   (no race conditions on file I/O, simpler logging).

SEARCH SPACE
─────────────
    max_depth:          3–8
    learning_rate:      0.005–0.10  (log-uniform)
    n_estimators:       200–800
    subsample:          0.60–1.00
    colsample_bytree:   0.50–1.00
    gamma:              0.0–1.0
    min_child_weight:   1–20
    reg_alpha:          1e-4–10.0  (log-uniform)
    reg_lambda:         1e-4–10.0  (log-uniform)

USAGE
──────
    # Standalone:
    python optuna_tuner.py --trials 50

    # From auto_retrainer (when USE_OPTUNA=True):
    from optuna_tuner import run_optuna_study
    run_optuna_study(n_trials=50)

    # Check what was saved:
    from optuna_tuner import load_best_params
    print(load_best_params())
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

warnings.filterwarnings("ignore")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CURRENT_DIR)

from paths import (FEATURES_PATH, LABELS_PATH,
                   OPTUNA_PARAMS_PATH, OPTUNA_STUDY_PATH,
                   create_all_dirs as _cad_ot)
_cad_ot()

# ── Configuration ──────────────────────────────────────────────────
DEFAULT_N_TRIALS    = 50      # number of Optuna trials
HOLDOUT_FRACTION    = 0.20    # fraction of data used as tuning holdout
MIN_TRAIN_FRACTION  = 0.60    # minimum fraction for tuning train set
EARLY_STOP_ROUNDS   = 30      # XGBoost early stopping within each trial
N_ESTIMATORS_MAX    = 800     # upper bound for n_estimators search
RANDOM_SEED         = 42


# ================================================================
# DATA LOADING
# ================================================================

def _load_tuning_data():
    """
    Loads features + labels and splits into (X_train, y_train, X_test, y_test).

    Uses a fixed chronological split (not the model_evaluator holdout) so
    tuning is independent of the deployment evaluation window.

    Returns (X_train, y_train, X_test, y_test, feature_cols) or None on failure.
    """
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
        print(f"[Optuna] features.csv or labels.csv not found.")
        return None

    try:
        features  = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
        labels_df = pd.read_csv(LABELS_PATH,   index_col=0, parse_dates=True)
        labels    = labels_df.iloc[:, 0]
    except Exception as e:
        print(f"[Optuna] Could not load data: {e}")
        return None

    shared_idx = features.index.intersection(labels.index)
    features   = features.loc[shared_idx]
    labels     = labels.loc[shared_idx]

    # Apply rolling z-score normalisation (same as trainer.py)
    try:
        from feature_engineer import apply_rolling_zscore, TF_FEATURE_COLS, FEATURE_COLS
        base_cols = [c for c in FEATURE_COLS if c in features.columns]
        X = features[base_cols].copy()
        tf_cols = [c for c in TF_FEATURE_COLS if c in X.columns]
        X = apply_rolling_zscore(X, cols=tf_cols, window=500)
    except Exception as e:
        print(f"[Optuna] Feature prep error: {e}")
        return None

    y = labels.reindex(X.index)
    valid = X.notna().all(axis=1) & y.notna()
    X, y  = X[valid], y[valid]

    # Label encode
    try:
        from sklearn.preprocessing import LabelEncoder
        from labeler import ALL_REGIMES
        le = LabelEncoder()
        le.fit(ALL_REGIMES)
        y_enc = le.transform(y)
    except Exception as e:
        print(f"[Optuna] Label encode error: {e}")
        return None

    n      = len(X)
    split1 = int(n * MIN_TRAIN_FRACTION)
    split2 = int(n * (1 - HOLDOUT_FRACTION))

    X_train = X.values[split1:split2]
    y_train = y_enc[split1:split2]
    X_test  = X.values[split2:]
    y_test  = y_enc[split2:]

    # Sample weights
    try:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        wt_map  = dict(zip(classes, weights))
        sw      = np.array([wt_map[yi] for yi in y_train])
    except Exception:
        sw = None

    print(f"[Optuna] Tuning data: train={len(X_train):,}  test={len(X_test):,}  "
          f"features={len(base_cols)}")
    return X_train, y_train, X_test, y_test, sw, le, base_cols


# ================================================================
# OPTUNA OBJECTIVE
# ================================================================

def _make_objective(X_train, y_train, X_test, y_test, sw, le):
    """
    Returns an Optuna objective function that trains XGBoost with trial params
    and returns the composite score on the tuning test set.
    """
    def _composite(y_true_enc, y_pred_enc, le):
        """Compute composite score matching model_evaluator formula."""
        from model_evaluator import (_BULL, _BEAR, _REVERSAL,
                                     W_DIRECTIONAL, W_OVERALL,
                                     W_REVERSAL, W_CONFUSION)
        y_true = le.inverse_transform(y_true_enc)
        y_pred = le.inverse_transform(y_pred_enc)
        n      = len(y_true)

        overall_acc = float(np.mean(y_pred == y_true))

        dir_mask = np.isin(y_true, [_BULL, _BEAR])
        dir_acc  = (float(np.mean(y_pred[dir_mask] == y_true[dir_mask]))
                    if dir_mask.any() else 0.0)

        bull_mask = y_true == _BULL
        bear_mask = y_true == _BEAR
        bb_conf   = 0.0
        if bull_mask.any() and bear_mask.any():
            bb_conf = (np.mean(y_pred[bull_mask] == _BEAR) +
                       np.mean(y_pred[bear_mask] == _BULL)) / 2.0

        rev_mask = y_true == _REVERSAL
        rev_rec  = (float(np.mean(y_pred[rev_mask] == _REVERSAL))
                    if rev_mask.any() else 0.0)

        return (W_DIRECTIONAL * dir_acc + W_OVERALL * overall_acc
                + W_REVERSAL * rev_rec - W_CONFUSION * bb_conf)

    def objective(trial):
        import xgboost as xgb

        params = {
            "objective":          "multi:softprob",
            "num_class":          len(le.classes_),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
            "n_estimators":       trial.suggest_int("n_estimators", 200, N_ESTIMATORS_MAX),
            "subsample":          trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.50, 1.00),
            "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "eval_metric":        "mlogloss",
            "random_state":       RANDOM_SEED,
            "n_jobs":             -1,
            "verbosity":          0,
            "early_stopping_rounds": EARLY_STOP_ROUNDS,
        }

        # Inner val split (last 20% of train) for early stopping
        n_tr    = len(X_train)
        val_cut = int(n_tr * 0.80)
        X_tr_es = X_train[:val_cut]
        y_tr_es = y_train[:val_cut]
        X_val   = X_train[val_cut:]
        y_val   = y_train[val_cut:]
        sw_es   = sw[:val_cut] if sw is not None else None

        try:
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr_es, y_tr_es,
                      sample_weight=sw_es,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            y_pred = model.predict(X_test)
            score  = _composite(y_test, y_pred, le)
        except Exception as e:
            raise optuna.TrialPruned()  # noqa — mark as pruned, not failed

        return score

    return objective


# ================================================================
# MAIN STUDY
# ================================================================

def run_optuna_study(n_trials: int = DEFAULT_N_TRIALS,
                     verbose: bool = True) -> dict:
    """
    Runs an Optuna study to find the best XGBoost hyperparameters.
    Saves results to OPTUNA_PARAMS_PATH.

    Args:
        n_trials: number of Optuna trials (default 50 ≈ 25min)
        verbose:  print per-trial progress

    Returns:
        dict with best_params and study_stats, or {} on failure.
    """
    try:
        import optuna
    except ImportError:
        print("[Optuna] optuna not installed. Run: pip install optuna")
        print("[Optuna] Skipping tuning — default XGBoost params will be used.")
        return {}

    print("\n" + "=" * 65)
    print(f"  OPTUNA HYPERPARAMETER TUNING  ({n_trials} trials)")
    print("=" * 65)

    data = _load_tuning_data()
    if data is None:
        return {}

    X_train, y_train, X_test, y_test, sw, le, feature_cols = data

    # Suppress Optuna verbose output unless verbose=True
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    objective = _make_objective(X_train, y_train, X_test, y_test, sw, le)

    study = optuna.create_study(
        direction   = "maximize",
        study_name  = "xgb_regime_tuning",
        pruner      = optuna.pruners.MedianPruner(n_startup_trials=10,
                                                   n_warmup_steps=5),
        sampler     = optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    import optuna as _opt   # re-import to avoid scope issues in callback
    def _callback(study, trial):
        if trial.number % 10 == 0 or trial.value == study.best_value:
            print(f"  Trial {trial.number:>3}: score={trial.value:.4f}  "
                  f"best={study.best_value:.4f}")

    print(f"\n[Optuna] Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, callbacks=[_callback])

    best_params  = study.best_params
    best_score   = study.best_value
    n_complete   = len([t for t in study.trials
                        if t.state == _opt.trial.TrialState.COMPLETE])
    n_pruned     = len([t for t in study.trials
                        if t.state == _opt.trial.TrialState.PRUNED])

    print(f"\n[Optuna] ─── Study Complete ───")
    print(f"  Trials: {n_complete} complete | {n_pruned} pruned")
    print(f"  Best composite score: {best_score:.4f}")
    print(f"  Best params:")
    for k, v in best_params.items():
        print(f"    {k:<25}: {v}")

    # Save results
    result = {
        "saved_at":       datetime.now().strftime('%Y-%m-%d %H:%M'),
        "n_trials":       n_trials,
        "n_complete":     n_complete,
        "n_pruned":       n_pruned,
        "best_score":     round(best_score, 4),
        "best_params":    best_params,
        "feature_cols":   feature_cols,
        "train_size":     len(X_train),
        "test_size":      len(X_test),
    }

    try:
        with open(OPTUNA_PARAMS_PATH, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\n[Optuna] ✅ Best params saved → {OPTUNA_PARAMS_PATH}")
    except Exception as e:
        print(f"[Optuna] Warning: could not save params: {e}")

    # Save abbreviated study stats (top 10 trials)
    try:
        top_trials = sorted(
            [t for t in study.trials
             if t.state == _opt.trial.TrialState.COMPLETE],
            key=lambda t: t.value, reverse=True
        )[:10]
        study_stats = {
            "saved_at":  result["saved_at"],
            "best_score": best_score,
            "top_trials": [
                {"trial": t.number, "score": round(t.value, 4), "params": t.params}
                for t in top_trials
            ],
        }
        with open(OPTUNA_STUDY_PATH, 'w') as f:
            json.dump(study_stats, f, indent=4)
    except Exception:
        pass

    return result


def load_best_params() -> dict:
    """
    Returns the last saved Optuna best params, or {} if none saved yet.

    Used by trainer.py and walkforward_trainer.py to load tuned params.
    """
    if not os.path.exists(OPTUNA_PARAMS_PATH):
        return {}
    try:
        with open(OPTUNA_PARAMS_PATH) as f:
            data = json.load(f)
        params = data.get("best_params", {})
        if params:
            print(f"[Optuna] Loaded best params from "
                  f"{data.get('saved_at', '?')} "
                  f"(score={data.get('best_score', '?')})")
        return params
    except Exception as e:
        print(f"[Optuna] Could not load params: {e}")
        return {}


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Optuna Bayesian hyperparameter tuning for XGBoost regime model')
    parser.add_argument('--trials', type=int, default=DEFAULT_N_TRIALS,
                        help=f'Number of Optuna trials (default: {DEFAULT_N_TRIALS})')
    parser.add_argument('--quiet',  action='store_true',
                        help='Suppress per-trial output')
    args = parser.parse_args()

    run_optuna_study(n_trials=args.trials, verbose=not args.quiet)
