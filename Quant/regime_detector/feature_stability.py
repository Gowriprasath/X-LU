"""
feature_stability.py — Feature Importance Stability Filter (FISF)
==================================================================
Removes spurious or unstable features from XGBoost models before
final training. Applied separately to:

    PRIMARY model  — regime detector XGBoost (loose threshold)
    META model     — trade outcome XGBoost   (strict threshold)

WHY THIS IS CRITICAL FOR YOUR ARCHITECTURE
──────────────────────────────────────────
You have 156 features. 124 are TF-derived price features (31 × 4 TFs).
Many are correlated or redundant:

    m5_rsi, h1_rsi, h4_rsi, d1_rsi        — all measure RSI at different TFs
    m5_momentum_20, m5_roc_10              — near-identical information
    h4_rolling_vol_20, h4_rolling_vol_100  — redundant in trending markets

Without FISF, XGBoost splits on whichever correlates with labels in THAT
specific training window. Different window → different "important" feature.
Live trading: model has learned noise, not structure.

TWO-TIER FILTERING STRATEGY
────────────────────────────
Primary XGBoost (regime detection):
    Loose threshold (CV < 1.5, mean_rank_pct < 0.60)
    Reason: some features ARE legitimately regime-specific.
    RSI might matter in BULL_TREND but not LOW_VOL_RANGE — that's real,
    not noise. Too aggressive filtering destroys regime-specific signal.

Meta XGBoost (trade quality prediction):
    Strict threshold (CV < 0.80, mean_rank_pct < 0.45)
    Reason: spurious correlations here = false confidence = real losses.
    Meta model must generalise to unseen market conditions.

STABILITY METRICS
─────────────────
Three metrics per feature, each capturing something different:

    1. Importance CV (coefficient of variation = std/mean)
       What it detects: features whose importance swings wildly
       Limitation: sensitive to scale — low-importance features
                   have artificially high CV

    2. Mean Rank Percentile (average rank / n_features)
       What it detects: consistently unimportant features
       Advantage: scale-free, robust to importance magnitude
       0.0 = always ranked #1 (most important)
       1.0 = always ranked last (least important)

    3. SHAP Stability Score (optional — requires shap library)
       What it detects: features that contribute inconsistently
       to individual predictions across windows
       Most reliable metric — used when available

DECISION RULE (features must pass ALL applied criteria):
    mean_importance > MIN_MEAN_IMPORTANCE  (not just consistently zero)
    AND importance_cv < cv_threshold       (not wildly variable)
    AND mean_rank_pct < rank_threshold     (consistently ranked in top half)

PUBLIC API
──────────
run_fisf(X, y, weights, n_windows, mode)
    → dict with stable_cols, dropped_cols, report

filter_features(X, stable_cols)
    → X filtered to stable columns only

load_stable_cols(model_type)
    → list of stable column names (from saved JSON)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

current_dir   = os.path.dirname(os.path.abspath(__file__))
import sys as _sys_fs
_root_fs = os.path.normpath(os.path.join(current_dir, '..', '..'))
if _root_fs not in _sys_fs.path: _sys_fs.path.insert(0, _root_fs)
from paths import FISF_PRIMARY_PATH as PRIMARY_FISF, FISF_META_PATH as META_FISF,                   REGIME_MODEL as FISF_DIR, create_all_dirs as _cad_fs
_cad_fs()

# ── Thresholds ────────────────────────────────────────────────────
# PRIMARY: loose — preserve regime-specific signal
PRIMARY_CV_THRESHOLD      = 1.50   # importance std/mean < 1.5
PRIMARY_RANK_THRESHOLD    = 0.60   # average rank in top 60% of features
PRIMARY_MIN_IMPORTANCE    = 0.003  # must have mean importance > 0.3%

# META: strict — prevent false confidence in trade quality model
META_CV_THRESHOLD         = 0.80   # importance std/mean < 0.8
META_RANK_THRESHOLD       = 0.45   # average rank in top 45% of features
META_MIN_IMPORTANCE       = 0.005  # must have mean importance > 0.5%

# NEVER FILTER — these columns are structural, always valid
# Time features: cyclical encodings, session flags — never spurious
# Persistence: regime state — core to the model's temporal reasoning
NEVER_DROP_PATTERNS = [
    'is_asian', 'is_london', 'is_ny', 'is_dead',
    'is_monday', 'is_friday', 'is_month_start', 'is_month_end',
    'hour_sin', 'hour_cos', 'minute_of_day_sin', 'minute_of_day_cos',
    'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
    'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
    'candles_since_regime_start', 'regime_duration_mean',
    'previous_regime_encoded', 'regime_transition_prob',
    # Meta model structural features
    'xgb_confidence', 'prediction_entropy', 'xgb_margin',
    'regime_encoded', 'bars_since_shift', 'direction_encoded',
]


def _is_protected(col: str) -> bool:
    return any(p in col for p in NEVER_DROP_PATTERNS)


# ================================================================
# ROLLING TIME SPLITS (chronological — no lookahead)
# ================================================================

def _make_time_splits(n: int, n_windows: int = 6, min_train_pct: float = 0.40):
    """
    Creates expanding or rolling chronological splits.
    Each window: train on first k%, test on next slice.

    Unlike k-fold, splits NEVER shuffle — financial data is time-ordered.

    Args:
        n:             total number of samples
        n_windows:     number of windows (default 6 — enough for stability signal)
        min_train_pct: minimum fraction for training (default 0.40)

    Yields:
        (train_indices, test_indices)
    """
    # Use expanding window: each split adds more training data
    # This reflects how the model will actually be retrained over time
    step     = int(n * (1 - min_train_pct) / n_windows)
    min_train = int(n * min_train_pct)

    for i in range(n_windows):
        train_end  = min_train + i * step
        test_start = train_end
        test_end   = min(train_end + step, n)

        if test_end <= test_start or train_end < 50:
            continue

        yield np.arange(0, train_end), np.arange(test_start, test_end)


# ================================================================
# CORE FISF ENGINE
# ================================================================

def run_fisf(
    X:           pd.DataFrame,
    y:           pd.Series,
    weights:     np.ndarray = None,
    n_windows:   int = 6,
    mode:        str = "primary",   # "primary" or "meta"
    use_shap:    bool = False,      # SHAP is more accurate but slower
    xgb_params:  dict = None,
    verbose:     bool = True,
) -> dict:
    """
    Run Feature Importance Stability Filter across rolling time splits.

    Args:
        X:          Feature DataFrame (all candidate features)
        y:          Target Series (regime labels or meta labels)
        weights:    Sample weights (optional — uniqueness weights for meta)
        n_windows:  Number of rolling time windows (default 6)
        mode:       "primary" (loose) or "meta" (strict) thresholds
        use_shap:   Use SHAP values instead of feature_importances_
                    More accurate but requires: pip install shap
        xgb_params: Override XGBoost params (else uses sensible defaults)
        verbose:    Print progress

    Returns:
        {
            "stable_cols":      list of stable feature names,
            "dropped_cols":     list of unstable feature names,
            "stability_report": DataFrame with per-feature metrics,
            "n_windows_used":   int,
            "mode":             str,
        }
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("[FISF] xgboost not installed.")
        return {"stable_cols": list(X.columns), "dropped_cols": []}

    # ── Threshold selection ───────────────────────────────────────
    if mode == "meta":
        cv_thresh   = META_CV_THRESHOLD
        rank_thresh = META_RANK_THRESHOLD
        min_imp     = META_MIN_IMPORTANCE
    else:
        cv_thresh   = PRIMARY_CV_THRESHOLD
        rank_thresh = PRIMARY_RANK_THRESHOLD
        min_imp     = PRIMARY_MIN_IMPORTANCE

    # ── XGBoost config ────────────────────────────────────────────
    default_params = {
        "max_depth":        4,     # shallower = faster for FISF passes
        "learning_rate":    0.05,
        "n_estimators":     300,   # fewer trees = faster, still enough for importance
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "verbosity":        0,
        "use_label_encoder": False,
        "eval_metric":      "mlogloss" if mode == "primary" else "logloss",
        "random_state":     42,
    }
    if xgb_params:
        default_params.update(xgb_params)

    feature_cols   = [c for c in X.columns]
    n_features     = len(feature_cols)
    n_samples      = len(X)
    X_vals         = X[feature_cols].values
    y_vals         = y.values if hasattr(y, 'values') else np.array(y)

    # Label encode if string labels (primary model)
    if y_vals.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le     = LabelEncoder()
        y_vals = le.fit_transform(y_vals)

    if verbose:
        print(f"\n[FISF] ═══════════════════════════════════════════════")
        print(f"[FISF] Mode: {mode.upper()} | Features: {n_features} | "
              f"Samples: {n_samples:,} | Windows: {n_windows}")
        print(f"[FISF] Thresholds: CV<{cv_thresh} | "
              f"RankPct<{rank_thresh} | MinImp>{min_imp}")
        print(f"[FISF] ═══════════════════════════════════════════════")

    # ── Collect importance per window ─────────────────────────────
    importance_matrix = []   # shape: (n_windows, n_features)
    shap_matrix       = []   # shape: (n_windows, n_features) — if use_shap

    splits = list(_make_time_splits(n_samples, n_windows))
    actual_windows = 0

    for wi, (train_idx, test_idx) in enumerate(splits):
        X_tr = X_vals[train_idx]
        y_tr = y_vals[train_idx]
        w_tr = weights[train_idx] if weights is not None else None

        if len(X_tr) < 100:
            if verbose:
                print(f"  Window {wi+1}: too small ({len(X_tr)} samples) — skip")
            continue

        model = xgb.XGBClassifier(**default_params)
        try:
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  Window {wi+1}: fit failed ({e}) — skip")
            continue

        imp = model.feature_importances_   # shape: (n_features,)
        importance_matrix.append(imp)
        actual_windows += 1

        if use_shap:
            try:
                import shap
                X_te    = X_vals[test_idx[:min(200, len(test_idx))]]
                explainer = shap.TreeExplainer(model)
                sv        = explainer.shap_values(X_te)
                # Multi-class: sv is list of arrays — take mean abs across classes
                if isinstance(sv, list):
                    sv = np.mean([np.abs(s) for s in sv], axis=0)
                shap_imp = np.abs(sv).mean(axis=0)
                shap_matrix.append(shap_imp)
            except Exception:
                pass   # SHAP optional

        if verbose:
            top3_idx = np.argsort(imp)[::-1][:3]
            top3     = [(feature_cols[i], round(float(imp[i]), 4))
                        for i in top3_idx]
            print(f"  Window {wi+1}/{len(splits)}: "
                  f"train={len(X_tr):,} | "
                  f"top3: {top3}")

    if actual_windows < 2:
        print("[FISF] Not enough windows to measure stability. "
              "Returning all features.")
        return {
            "stable_cols":      feature_cols,
            "dropped_cols":     [],
            "stability_report": None,
            "n_windows_used":   actual_windows,
            "mode":             mode,
        }

    # ── Compute stability metrics ─────────────────────────────────
    imp_matrix   = np.array(importance_matrix)    # (W, F)
    mean_imp     = imp_matrix.mean(axis=0)        # (F,)
    std_imp      = imp_matrix.std(axis=0)         # (F,)

    # CV: std/mean — avoid division by zero
    cv           = np.where(mean_imp > 1e-8,
                            std_imp / mean_imp,
                            999.0)

    # Rank matrix: within each window, rank features by importance
    # Rank 0 = most important, rank F-1 = least important
    rank_matrix  = np.zeros_like(imp_matrix)
    for wi in range(len(imp_matrix)):
        ranks = np.argsort(np.argsort(-imp_matrix[wi]))   # descending rank
        rank_matrix[wi] = ranks

    mean_rank     = rank_matrix.mean(axis=0)              # lower = better
    mean_rank_pct = mean_rank / max(n_features - 1, 1)   # 0=best, 1=worst

    # Rank CV: consistency of rank position
    rank_cv       = rank_matrix.std(axis=0) / (rank_matrix.mean(axis=0) + 1e-8)

    # SHAP stability (if available)
    shap_cv       = np.full(n_features, np.nan)
    if len(shap_matrix) >= 2:
        shap_arr  = np.array(shap_matrix)
        shap_mean = shap_arr.mean(axis=0)
        shap_std  = shap_arr.std(axis=0)
        shap_cv   = np.where(shap_mean > 1e-8, shap_std / shap_mean, 999.0)

    # ── Decision: stable or not ───────────────────────────────────
    stable_mask = (
        (mean_imp     >= min_imp) &
        (cv           <  cv_thresh) &
        (mean_rank_pct < rank_thresh)
    )

    # SHAP override: if SHAP available and feature is unstable by SHAP,
    # drop it even if importance metrics say stable
    if not np.all(np.isnan(shap_cv)):
        shap_unstable = shap_cv > cv_thresh
        stable_mask   = stable_mask & (~shap_unstable)

    # Protected features always survive
    protected_mask = np.array([_is_protected(c) for c in feature_cols])
    stable_mask    = stable_mask | protected_mask

    # ── Build report ──────────────────────────────────────────────
    report = pd.DataFrame({
        "feature":       feature_cols,
        "mean_imp":      np.round(mean_imp, 5),
        "std_imp":       np.round(std_imp, 5),
        "imp_cv":        np.round(cv, 3),
        "mean_rank":     np.round(mean_rank, 1),
        "mean_rank_pct": np.round(mean_rank_pct, 3),
        "rank_cv":       np.round(rank_cv, 3),
        "shap_cv":       np.round(shap_cv, 3),
        "protected":     protected_mask,
        "stable":        stable_mask,
    }).sort_values("mean_imp", ascending=False)

    stable_cols  = [feature_cols[i] for i in range(n_features) if stable_mask[i]]
    dropped_cols = [feature_cols[i] for i in range(n_features) if not stable_mask[i]]

    # ── Print summary ─────────────────────────────────────────────
    if verbose:
        n_stable      = len(stable_cols)
        n_dropped     = len(dropped_cols)
        n_protected   = int(protected_mask.sum())
        n_filtered    = n_dropped

        print(f"\n[FISF] ─── Stability Results ({mode.upper()}) ───")
        print(f"  Total features:    {n_features}")
        print(f"  Protected (never drop): {n_protected}")
        print(f"  Stable (kept):     {n_stable}  "
              f"({n_stable/n_features*100:.0f}%)")
        print(f"  Dropped:           {n_filtered} "
              f"({n_filtered/n_features*100:.0f}%)")
        print(f"  Windows used:      {actual_windows}")

        # Show top stable features
        stable_report = report[report["stable"] & ~report["protected"]]
        print(f"\n  Top 10 STABLE features (non-protected):")
        for _, row in stable_report.head(10).iterrows():
            print(f"    {row['feature']:<35} "
                  f"imp={row['mean_imp']:.4f} "
                  f"CV={row['imp_cv']:.2f} "
                  f"rank_pct={row['mean_rank_pct']:.2f}")

        # Show what was dropped
        if dropped_cols:
            dropped_report = report[~report["stable"]]
            print(f"\n  Top 10 DROPPED features (most important among dropped):")
            for _, row in dropped_report.head(10).iterrows():
                reason = []
                if row['mean_imp'] < min_imp:
                    reason.append(f"low_imp({row['mean_imp']:.4f})")
                if row['imp_cv'] >= cv_thresh:
                    reason.append(f"unstable_cv({row['imp_cv']:.2f})")
                if row['mean_rank_pct'] >= rank_thresh:
                    reason.append(f"poor_rank({row['mean_rank_pct']:.2f})")
                print(f"    {row['feature']:<35} → {', '.join(reason)}")

    return {
        "stable_cols":      stable_cols,
        "dropped_cols":     dropped_cols,
        "stability_report": report,
        "n_windows_used":   actual_windows,
        "mode":             mode,
    }


# ================================================================
# SAVE / LOAD STABLE FEATURE SETS
# ================================================================

def save_stable_cols(stable_cols: list, dropped_cols: list,
                     report: pd.DataFrame, mode: str):
    """Saves FISF results to JSON for use at training and inference time."""
    os.makedirs(FISF_DIR, exist_ok=True)
    path = PRIMARY_FISF if mode == "primary" else META_FISF

    # Convert report to serialisable dict
    report_data = None
    if report is not None:
        report_data = report.to_dict(orient='records')
        # Convert numpy types to native python
        for row in report_data:
            for k, v in row.items():
                if hasattr(v, 'item'):
                    row[k] = v.item()

    data = {
        "saved_date":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode":        mode,
        "n_stable":    len(stable_cols),
        "n_dropped":   len(dropped_cols),
        "stable_cols": stable_cols,
        "dropped_cols": dropped_cols,
        "report":      report_data,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[FISF] Saved {mode} stable cols ({len(stable_cols)} features) → {path}")


def load_stable_cols(mode: str = "primary") -> list:
    """
    Loads stable feature list from saved FISF results.
    Returns empty list if not yet built (caller should use all features).
    """
    path = PRIMARY_FISF if mode == "primary" else META_FISF
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        cols = data.get("stable_cols", [])
        print(f"[FISF] Loaded {mode} stable cols: "
              f"{len(cols)} features "
              f"(saved {data.get('saved_date','?')})")
        return cols
    except Exception as e:
        print(f"[FISF] Load error: {e}")
        return []


def filter_features(X: pd.DataFrame, stable_cols: list) -> pd.DataFrame:
    """
    Filters X to only stable columns.
    Falls back to full X if stable_cols is empty (first run before FISF built).
    """
    if not stable_cols:
        return X
    available = [c for c in stable_cols if c in X.columns]
    missing   = [c for c in stable_cols if c not in X.columns]
    if missing:
        print(f"[FISF] Warning: {len(missing)} stable cols not in X: "
              f"{missing[:5]}{'...' if len(missing)>5 else ''}")
    return X[available]


# ================================================================
# FEATURE IMPORTANCE HALF-LIFE (Point 8 from guide)
# Estimates how quickly each feature's predictive power decays.
# ================================================================

def compute_importance_halflife(
    X:         pd.DataFrame,
    y:         pd.Series,
    weights:   np.ndarray = None,
    n_windows: int = 8,
    mode:      str = "primary",
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Estimates feature importance half-life:
    how many windows until a feature's importance decays to 50% of peak.

    Features with short half-life are predictive only in specific market
    regimes or time periods — not reliable for live trading.

    Args:
        Same as run_fisf()

    Returns:
        DataFrame with columns: feature, halflife_windows, peak_importance,
                                 decay_rate, recommendation
    """
    try:
        import xgboost as xgb
    except ImportError:
        return pd.DataFrame()

    default_params = {
        "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
        "verbosity": 0, "use_label_encoder": False,
        "eval_metric": "mlogloss" if mode == "primary" else "logloss",
        "random_state": 42,
    }

    feature_cols = list(X.columns)
    n_samples    = len(X)
    X_vals       = X.values
    y_vals       = y.values if hasattr(y, 'values') else np.array(y)
    if y_vals.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        y_vals = LabelEncoder().fit_transform(y_vals)

    splits = list(_make_time_splits(n_samples, n_windows))
    importance_over_time = []

    for train_idx, _ in splits:
        X_tr = X_vals[train_idx]
        y_tr = y_vals[train_idx]
        w_tr = weights[train_idx] if weights is not None else None
        if len(X_tr) < 100:
            continue
        model = xgb.XGBClassifier(**default_params)
        try:
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            importance_over_time.append(model.feature_importances_)
        except Exception:
            continue

    if len(importance_over_time) < 3:
        return pd.DataFrame()

    imp_matrix = np.array(importance_over_time)   # (W, F)
    records    = []

    for fi, feat in enumerate(feature_cols):
        imp_series  = imp_matrix[:, fi]
        peak        = float(imp_series.max())
        if peak < 0.001:
            halflife = 0
            decay    = 999.0
        else:
            # Find window where importance drops to 50% of peak
            half_target = peak * 0.5
            halflife_w  = next(
                (w for w, v in enumerate(imp_series) if v <= half_target),
                len(imp_series)
            )
            # Approximate exponential decay rate
            end_val  = float(imp_series[-1])
            n_w      = len(imp_series)
            if end_val > 0 and peak > 0:
                decay = -np.log(end_val / peak) / max(n_w, 1)
            else:
                decay = 999.0
            halflife = halflife_w

        recommendation = (
            "STABLE"    if halflife >= n_windows * 0.75 else
            "SEASONAL"  if halflife >= n_windows * 0.40 else
            "FRAGILE"
        )

        records.append({
            "feature":           feat,
            "halflife_windows":  halflife,
            "peak_importance":   round(peak, 5),
            "final_importance":  round(float(imp_series[-1]), 5),
            "decay_rate":        round(float(decay), 3),
            "recommendation":    recommendation,
        })

    df = pd.DataFrame(records).sort_values(
        "halflife_windows", ascending=False)

    if verbose:
        print(f"\n[FISF] ─── Feature Importance Half-Life ({mode.upper()}) ───")
        print(f"{'Feature':<35} {'Half-Life':>10} {'Peak Imp':>10} "
              f"{'Decay':>8} {'Status':>12}")
        print("─" * 78)
        for _, row in df.head(20).iterrows():
            icon = "✓" if row['recommendation'] == "STABLE" else \
                   "~" if row['recommendation'] == "SEASONAL" else "✗"
            print(f"  {icon} {row['feature']:<33} "
                  f"{row['halflife_windows']:>10} "
                  f"{row['peak_importance']:>10.4f} "
                  f"{row['decay_rate']:>8.3f} "
                  f"{row['recommendation']:>12}")

        by_status = df.groupby("recommendation").size()
        print(f"\n  STABLE: {by_status.get('STABLE',0)} | "
              f"SEASONAL: {by_status.get('SEASONAL',0)} | "
              f"FRAGILE: {by_status.get('FRAGILE',0)}")

    return df


# ================================================================
# WALK-FORWARD THRESHOLD CALIBRATION FOR STABLE FEATURES
# (ties into session_profiler calibration)
# ================================================================

def run_full_fisf_pipeline(
    X:         pd.DataFrame,
    y:         pd.Series,
    weights:   np.ndarray = None,
    mode:      str = "primary",
    use_shap:  bool = False,
    n_windows: int = 6,
    verbose:   bool = True,
) -> list:
    """
    Full pipeline:
        1. Run FISF across rolling windows
        2. Compute importance half-life
        3. Save stable feature set
        4. Return stable column list (ready for final model training)

    This is the single function you call from trainer.py and
    meta_labeller.train() before the final XGBoost fit.

    Args:
        X:         Feature DataFrame
        y:         Labels
        weights:   Sample weights (pass uniqueness weights for meta)
        mode:      "primary" or "meta"
        use_shap:  Use SHAP stability (slower but more accurate)
        n_windows: Rolling windows for stability measurement

    Returns:
        list of stable column names — pass directly as feature set to
        the final XGBoost .fit() call
    """
    print(f"\n[FISF] Starting full pipeline — mode={mode.upper()}, "
          f"features={len(X.columns)}, samples={len(X):,}")

    # Step 1: Core stability filter
    result = run_fisf(
        X, y, weights=weights, n_windows=n_windows,
        mode=mode, use_shap=use_shap, verbose=verbose)

    stable_cols  = result["stable_cols"]
    dropped_cols = result["dropped_cols"]
    report       = result["stability_report"]

    # Step 2: Half-life analysis (informational — doesn't filter further)
    if verbose and len(stable_cols) > 0:
        print(f"\n[FISF] Computing importance half-life for {len(stable_cols)} "
              f"stable features...")
        halflife_df = compute_importance_halflife(
            X[stable_cols], y, weights=weights,
            n_windows=n_windows, mode=mode, verbose=verbose)

        # Flag FRAGILE features that passed stability but have short half-life
        # These are worth watching in paper trading
        if len(halflife_df) > 0:
            fragile = halflife_df[
                halflife_df["recommendation"] == "FRAGILE"]["feature"].tolist()
            if fragile:
                print(f"\n[FISF] ⚠  {len(fragile)} features passed stability "
                      f"but have SHORT half-life — monitor closely in live trading:")
                for f in fragile[:10]:
                    print(f"    {f}")

    # Step 3: Save
    save_stable_cols(stable_cols, dropped_cols, report, mode)

    print(f"\n[FISF] ✓ Pipeline complete. "
          f"{len(stable_cols)}/{len(X.columns)} features retained "
          f"for {mode.upper()} model.")
    print(f"[FISF] Feature reduction: "
          f"{len(dropped_cols)} removed "
          f"({len(dropped_cols)/len(X.columns)*100:.0f}% of total)")

    return stable_cols
