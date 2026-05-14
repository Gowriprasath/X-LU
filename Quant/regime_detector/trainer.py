"""
trainer.py — GMM-HMM (5 states, 15-init) + XGBoost v4

WHAT CHANGED IN v4:
    1. Unified directional regime system — one system everywhere:
           REVERSAL       — ATR spike / news event
           BULL_TREND     — DI+ dominant, H4 EMA bullish, structure up
           BEAR_TREND     — DI- dominant, H4 EMA bearish, structure down
           COMPRESSION    — extreme ATR + BB squeeze, pre-breakout
           LOW_VOL_RANGE  — quiet, directionless, default

    2. HMM obs now include DI+/DI- for directional Bull/Bear split.
       map_states_to_regimes() sorts middle states by momentum → Bull/Bear.

    3. Regime persistence features added to XGBoost training:
           candles_since_regime_start — how long current regime has run
           regime_duration_mean       — rolling mean of past 20 regime durations
       These are computed AFTER Viterbi decode (they depend on HMM labels).
       Added to XGBoost feature matrix alongside the 124 technical features.
       Also tracked live in regime_detector.py via a stateful counter.

    4. Feature column hard-check: training fails loudly if >5 FEATURE_COLS
       are missing from the feature matrix (was silent drop before).

WHAT CHANGED IN v3 (directional map_states_to_regimes):
    Middle states sorted by momentum → always produces BULL_TREND + BEAR_TREND.

HOW MULTIPLE INITIALISATIONS WORK:
    Training (happens ONCE, manually or via run_backtest.py):
        Loop i in 0..14:
            model_i = GMMHMM(random_state=i).fit(observations)
            ll_i    = model_i.score(observations)
        best_model = model with highest ll_i
        joblib.dump(best_model, ...)

    Live trading:
        On startup: _load_model() → joblib.load(HMM_PATH) → model in RAM
        Every cycle: regime_detector.predict() uses already-loaded model
        ZERO re-initialisation. ZERO extra compute. <2ms per candle.

ARCHITECTURE:
    Stage 1 — GMM-HMM (unsupervised, time-conditioned regime discovery)
        • Observes BOTH price behaviour AND temporal context simultaneously
        • Price obs (13): ATR, ATR ratio, ADX, DI+/DI-, momentum, BB width,
                          trend structure, H4 EMA stack
        • Time obs  (9) : hour_sin/cos (cyclical), session flags (Asian/London/NY),
                          dow_sin/cos (cyclical), is_monday, is_friday
        • Result: discovers TIME-CONDITIONED regimes:
              "Asian session LOW_VOL_RANGE" ≠ "NY session LOW_VOL_RANGE"
              "London open BULL_TREND"  — detected by ADX rise + is_london
              "NY REVERSAL"             — detected by ATR spike + is_ny
        • GMM emissions handle gold's fat-tailed return distribution
        • Temporal memory via Markov transitions — learns session→session flows
        • 5 hidden states mapped to 5 directional regime labels

    Stage 2 — XGBoost Classifier (supervised, trained on GMM-HMM labels)
        • Fast single-candle inference (<2ms) for live trading
        • Handles imbalanced regimes via sample_weight
        • Gradient boosting focuses on hard-to-classify candles

Usage:
    python trainer.py                           # BIC auto-select n_states
    python trainer.py --n-states 5              # force 5 states (recommended)
    python trainer.py --from 2018-01-01
    python trainer.py --no-rebuild              # skip feature rebuild
    python trainer.py --n-init 20              # more initialisations (slower, better)
"""

import os, sys, json, argparse, warnings
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
backtest_dir = os.path.join(project_root, 'Backtest')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)
sys.path.insert(0, backtest_dir)

# Market data CSVs now in Data/Market/ — data_downloader saves there
from paths import MARKET_DATA_DIR as _market_dir

from feature_engineer import (
    build_multi_tf_features, save_features, load_features,
    FEATURE_COLS, TF_FEATURE_COLS, ALL_FEATURE_COLS, PERSISTENCE_FEATURE_COLS,
    compute_persistence_features, apply_rolling_zscore,
)
from labeler import ALL_REGIMES
from data_downloader import load_data

# ── Paths ─────────────────────────────────────────────────────────
if project_root not in sys.path: sys.path.insert(0, project_root)
from paths import (FEATURES_PATH, LABELS_PATH, BIC_PATH,
                   HMM_PATH, REGIME_SCALER_PATH as SCALER_PATH,
                   HMM_SCALER_PATH as HMM_SCALER,
                   LABEL_ENCODER_PATH as LE_PATH, REGIME_XGB_PATH as XGB_PATH,
                   MODEL_META_PATH as META_PATH,
                   REGIME_MODEL as MODEL_DIR, REGIME_DATA as DATA_DIR,
                   create_all_dirs as _cad_tr)
_cad_tr()

# ================================================================
# HMM OBSERVATION FEATURES
# ================================================================
# Two categories: price behaviour + temporal context.
#
# WHY TIME FEATURES IN HMM:
#   Without time, the HMM discovers "LOW_VOL" without knowing it always
#   happens at 21:00 (Asian session). With time, it discovers:
#       "Asian session LOW_VOL_RANGE"  — hour_sin/cos + is_asian + low ATR
#       "London open BULL_TREND"       — is_london + rising ADX + DI+ dominant
#       "NY open REVERSAL"             — is_ny + ATR spike + high momentum
#       "Friday COMPRESSION"           — is_friday + BB squeeze
#   The Markov transition matrix also learns session transitions:
#   Asian→London has elevated probability of BULL_TREND emergence.
#
# WHY CYCLICAL ENCODING (sin/cos) NOT RAW HOUR:
#   Raw hour=23 is numerically far from hour=0 (1-hour gap).
#   hour_sin/cos wraps correctly: 23:55 is close to 00:05 in cos/sin space.
#   Same applies to day-of-week (Friday close to Monday, not 4 days away).
#
# WHY NOT month/quarter/week_of_year:
#   Too slow-moving. HMM is learning candle-by-candle transitions.
#   Seasonal effects are better handled by XGBoost (all features, tree splits).

HMM_OBS_COLS = [
    # ── Price behaviour (13 features) ───────────────────────────
    'h1__atr_raw', 'h4__atr_raw',           # raw ATR — volatility magnitude
    'h1_atr_ratio', 'h4_atr_ratio',          # ATR relative to rolling mean
    'h1_adx', 'h4_adx',                      # trend strength
    'h1_plus_di', 'h1_minus_di',             # direction: DI+ > DI- = bull
    'h1_momentum_20', 'h4_momentum_20',      # price momentum
    'h1_bb_width',                           # volatility compression
    'h1_trend_structure', 'h4_ema_stack',    # structure / EMA alignment

    # ── Temporal context (9 features) ───────────────────────────
    # Cyclical time-of-day: wraps correctly (23:55 ≈ 00:05)
    'hour_sin', 'hour_cos',

    # Session flags: strongest temporal signal for XAUUSD
    # Asian  = low vol range 80% of the time
    # London = trend initiation, breakout
    # NY     = news spikes, reversals
    'is_asian', 'is_london', 'is_ny',

    # Day-of-week: cyclical (Friday ≈ Monday in cos/sin space)
    'dow_sin', 'dow_cos',

    # High-impact binary day flags
    # Monday = accumulation, weekly range establishment
    # Friday = distribution, position squaring, compression
    'is_monday', 'is_friday',
]

# ── Default config ────────────────────────────────────────────────
DEFAULT_N_STATES = 5      # 5 states for XAUUSD (see architecture above)
DEFAULT_N_MIX    = 3      # GMM mixtures per state
DEFAULT_N_ITER   = 200    # EM iterations per init
DEFAULT_N_INIT   = 15     # random initialisations — picks best log-likelihood


# ================================================================
# STAGE 1a: BIC STATE SELECTION
# ================================================================
def select_n_states(obs, state_range=(3, 7), n_init=5, n_iter=100):
    """
    BIC-based automatic state count selection.
    Tests state counts in [state_range[0], state_range[1]).
    Returns the n_states with lowest BIC score.

    Called only when --n-states is not specified.
    Each candidate uses n_init initialisations internally.
    """
    try:
        from hmmlearn.hmm import GMMHMM
    except ImportError:
        print("[BIC] hmmlearn not installed. Defaulting to 5 states.")
        return DEFAULT_N_STATES

    print(f"\n[BIC] Testing state counts {state_range[0]}–{state_range[1]-1} "
          f"({n_init} inits each)...")

    # ── Data sanitisation (critical for numerical stability) ──────
    # StandardScaler output can still contain extreme values when raw features
    # have heavy tails (XAUUSD ATR during COVID / Ukraine spikes).
    # Clipping to ±4σ keeps 99.994% of a normal distribution and prevents
    # the GMM covariance matrices from collapsing to zero or exploding.
    obs_safe = np.nan_to_num(obs, nan=0.0, posinf=4.0, neginf=-4.0)
    obs_safe = np.clip(obs_safe, -4.0, 4.0)

    # ── Subsampling for BIC speed (not used for final fit) ────────
    # BIC only needs a representative subset. 80k rows captures all regime
    # transitions while cutting fit time from ~10 min to ~1 min per candidate.
    BIC_MAX_ROWS = 80_000
    if len(obs_safe) > BIC_MAX_ROWS:
        step = len(obs_safe) // BIC_MAX_ROWS
        obs_bic = obs_safe[::step]
        print(f"[BIC] Subsampled {len(obs_safe):,} → {len(obs_bic):,} rows "
              f"(step={step}) for BIC speed")
    else:
        obs_bic = obs_safe

    bic_results = {}
    n_obs, n_feat = obs_bic.shape
    best_bic  = np.inf
    best_n    = DEFAULT_N_STATES

    for n in range(state_range[0], state_range[1]):
        best_ll = -np.inf
        for seed in range(n_init):
            try:
                m = GMMHMM(n_components=n, n_mix=DEFAULT_N_MIX,
                           covariance_type='diag', n_iter=n_iter,
                           tol=1e-2, random_state=seed, verbose=False,
                           covars_prior=1e-2, covars_weight=1.0)
                m.fit(obs_bic, lengths=[len(obs_bic)])
                ll = m.score(obs_bic, [len(obs_bic)])
                if np.isfinite(ll) and ll > best_ll:
                    best_ll = ll
            except Exception as e:
                print(f"    [BIC] n={n} seed={seed} failed: {e}")
                continue

        # BIC = -2 * log-likelihood + k * log(n_obs)
        # k = free parameters (approximate)
        k   = n * (n - 1) + n * DEFAULT_N_MIX * n_feat * 2
        if not np.isfinite(best_ll):
            bic = np.inf
            print(f"  n_states={n}: ALL INITS FAILED (all {n_init} seeds diverged)")
        else:
            bic = -2 * best_ll + k * np.log(n_obs)
        bic_results[n] = {"bic": round(float(bic), 2), "log_likelihood": round(float(best_ll), 2)}
        marker = " ← BEST" if bic < best_bic else ""
        print(f"  n_states={n}: BIC={bic:.1f} | LL={best_ll:.1f}{marker}")

        if bic < best_bic:
            best_bic = bic
            best_n   = n

    with open(BIC_PATH, 'w') as f:
        json.dump({"results": bic_results, "selected": best_n,
                   "run_date": datetime.now().strftime('%Y-%m-%d %H:%M')}, f, indent=4)

    print(f"\n[BIC] Selected n_states={best_n} (BIC={best_bic:.1f})")
    return best_n


# ================================================================
# STAGE 1b: GMM-HMM — MULTI-INIT BEST FIT
# ================================================================
def fit_gmmhmm_best(obs, n_states=DEFAULT_N_STATES,
                    n_mix=DEFAULT_N_MIX, n_iter=DEFAULT_N_ITER,
                    n_init=DEFAULT_N_INIT):
    """
    Fits GMM-HMM n_init times with different random seeds.
    Returns the model with the highest log-likelihood.

    WHY MULTIPLE INITS:
        HMM training (Baum-Welch / EM) is non-convex.
        Different random starting points can get stuck in different
        local optima, producing wildly different state assignments.
        Running n_init=15 times and picking the best ensures we find
        the statistical structure that best explains the data.

    FULLY AUTOMATIC — runs only during training, never during live trading.
    """
    try:
        from hmmlearn.hmm import GMMHMM
    except ImportError:
        print("\n❌ hmmlearn not installed. Run: pip install hmmlearn\n")
        return None

    print(f"\n[GMM-HMM] Fitting {n_states} states × {n_mix} mixtures | "
          f"{obs.shape[0]:,} obs × {obs.shape[1]} features | "
          f"{n_iter} EM iterations × {n_init} random inits...")
    print(f"[GMM-HMM] Running {n_init} initialisations — picking best log-likelihood...")

    # ── Data sanitisation ─────────────────────────────────────────
    # After StandardScaler, heavy-tailed assets (XAUUSD ATR spikes during
    # COVID/Ukraine/2024 all-time-highs) can still produce values >>4σ.
    # These extreme points cause GMM covariance matrices to collapse
    # (det→0, log-likelihood→-inf, Baum-Welch diverges).
    # Clipping to ±4σ retains 99.994% of a Gaussian distribution and
    # forces the EM algorithm into a numerically stable region.
    obs_safe = np.nan_to_num(obs, nan=0.0, posinf=4.0, neginf=-4.0)
    obs_safe = np.clip(obs_safe, -4.0, 4.0)

    # ── Subsampling for large sequences ──────────────────────────
    # GMMHMM Baum-Welch scales O(N × S²) in sequence length N.
    # 688k rows makes training slow and can amplify floating-point
    # rounding across the forward-backward recursion.
    # Keeping every Kth row preserves the regime structure while
    # reducing sequence length to ≤150k. The final model is still
    # used to decode the FULL sequence via Viterbi.
    MAX_HMM_ROWS = 150_000
    if len(obs_safe) > MAX_HMM_ROWS:
        step = len(obs_safe) // MAX_HMM_ROWS
        obs_fit = obs_safe[::step]
        print(f"[GMM-HMM] Subsampled {len(obs_safe):,} → {len(obs_fit):,} rows "
              f"(step={step}) for numerical stability")
    else:
        obs_fit = obs_safe

    best_model = None
    best_ll    = -np.inf

    for seed in range(n_init):
        try:
            model = GMMHMM(
                n_components=n_states, n_mix=n_mix,
                covariance_type='diag',
                n_iter=n_iter,
                tol=1e-2,          # was 1e-4 — looser tolerance prevents
                                   # infinite loop chasing noise
                random_state=seed,
                verbose=False,
                covars_prior=1e-2, # covariance floor: prevents near-zero
                covars_weight=1.0, # variances from causing log(0) in EM
            )
            model.fit(obs_fit, lengths=[len(obs_fit)])
            ll = model.score(obs_fit, [len(obs_fit)])

            if not np.isfinite(ll):
                print(f"  Init {seed+1:>2}/{n_init}: LL={ll} (non-finite, skipped)")
                continue

            marker = " ← NEW BEST" if ll > best_ll else ""
            print(f"  Init {seed+1:>2}/{n_init}: LL={ll:.2f}{marker}")

            if ll > best_ll:
                best_ll    = ll
                best_model = model

        except Exception as e:
            print(f"  Init {seed+1:>2}/{n_init}: FAILED ({type(e).__name__}: {e})")
            continue

    if best_model is None:
        print("[GMM-HMM] All initialisations failed.")
        return None

    print(f"\n[GMM-HMM] Best model: LL={best_ll:.2f} "
          f"(selected from {n_init} initialisations)")
    return best_model


# ================================================================
# STAGE 1c: MAP HIDDEN STATES → REGIME LABELS
# ================================================================
def map_states_to_regimes(hmm_model, hmm_cols):
    """
    Maps each hidden HMM state integer to a regime label — v2 directional.

    For 5 states (recommended for XAUUSD):
        Sort all 5 by ATR mean (ascending):
            atr_rank[0]  → COMPRESSION    (lowest ATR = extreme squeeze)
            atr_rank[1]  → LOW_VOL_RANGE  (second lowest = quiet, no squeeze)
            atr_rank[-1] → REVERSAL       (highest ATR = spike/news)

        Remaining 2 middle states → BULL_TREND or BEAR_TREND
        sorted by momentum_20 mean:
            highest momentum → BULL_TREND
            lowest  momentum → BEAR_TREND
        (If DI+/DI- cols available, uses those instead — more reliable)

    For 4 states (backward compat):
        atr extremes → COMPRESSION / REVERSAL
        mid by momentum → BULL_TREND / BEAR_TREND

    For 3 states:
        LOW_VOL_RANGE / BULL_TREND / REVERSAL
    """
    from labeler import (
        REGIME_REVERSAL, REGIME_BULL_TREND, REGIME_BEAR_TREND,
        REGIME_COMPRESSION, REGIME_LOW_VOL_RANGE,
    )

    n_states    = hmm_model.n_components
    state_means = hmm_model.means_[:, 0, :]   # dominant GMM mixture

    # Column index finders
    atr_col     = next((i for i, c in enumerate(hmm_cols) if '_atr_raw' in c or 'atr_ratio' in c), 0)
    adx_col     = next((i for i, c in enumerate(hmm_cols) if 'h1_adx' in c), 2)
    di_plus_col = next((i for i, c in enumerate(hmm_cols) if 'plus_di' in c), None)
    di_minus_col= next((i for i, c in enumerate(hmm_cols) if 'minus_di' in c), None)
    mom_col     = next((i for i, c in enumerate(hmm_cols) if 'h1_momentum' in c), 6)

    atr_means = state_means[:, atr_col]
    atr_rank  = np.argsort(atr_means)   # ascending: [lowest ATR ... highest ATR]

    state_map = {}

    def _assign_direction(s):
        """Returns BULL_TREND or BEAR_TREND for a given state index."""
        if di_plus_col is not None and di_minus_col is not None:
            return (REGIME_BULL_TREND
                    if state_means[s, di_plus_col] >= state_means[s, di_minus_col]
                    else REGIME_BEAR_TREND)
        # Fallback to momentum if DI not available
        return (REGIME_BULL_TREND
                if state_means[s, mom_col] >= 0
                else REGIME_BEAR_TREND)

    if n_states >= 5:
        state_map[int(atr_rank[0])]  = REGIME_COMPRESSION    # extreme squeeze
        state_map[int(atr_rank[1])]  = REGIME_LOW_VOL_RANGE  # quiet range
        state_map[int(atr_rank[-1])] = REGIME_REVERSAL        # spike

        # Middle states (could be 2 for n=5, 3 for n=6, etc.)
        mid_states = [int(atr_rank[i]) for i in range(2, n_states - 1)]

        # Sort by momentum — ensures we always get one Bull, one Bear
        mid_by_mom = sorted(mid_states, key=lambda s: state_means[s, mom_col])
        state_map[mid_by_mom[0]]  = REGIME_BEAR_TREND    # lowest momentum
        state_map[mid_by_mom[-1]] = REGIME_BULL_TREND    # highest momentum
        # Any extra middle states (n > 5) → direction-based
        for s in mid_by_mom[1:-1]:
            state_map[s] = _assign_direction(s)

    elif n_states == 4:
        state_map[int(atr_rank[0])]  = REGIME_COMPRESSION
        state_map[int(atr_rank[-1])] = REGIME_REVERSAL
        mid_states = [int(atr_rank[1]), int(atr_rank[2])]
        mid_by_mom = sorted(mid_states, key=lambda s: state_means[s, mom_col])
        state_map[mid_by_mom[0]]  = REGIME_BEAR_TREND
        state_map[mid_by_mom[-1]] = REGIME_BULL_TREND

    else:  # 3 states
        state_map[int(atr_rank[0])]  = REGIME_LOW_VOL_RANGE
        state_map[int(atr_rank[-1])] = REGIME_REVERSAL
        state_map[int(atr_rank[1])]  = REGIME_BULL_TREND

    print("\n[GMM-HMM v4] State → Regime mapping (directional + time-conditioned):")
    for s, r in sorted(state_map.items()):
        di_info = ""
        if di_plus_col is not None and di_minus_col is not None:
            di_info = (f" | DI+: {state_means[s, di_plus_col]:.1f}"
                       f" DI-: {state_means[s, di_minus_col]:.1f}")

        # Show session weights if time cols are in HMM obs
        session_info = ""
        asian_col  = next((i for i, c in enumerate(hmm_cols) if c == 'is_asian'),  None)
        london_col = next((i for i, c in enumerate(hmm_cols) if c == 'is_london'), None)
        ny_col     = next((i for i, c in enumerate(hmm_cols) if c == 'is_ny'),     None)
        if asian_col is not None:
            session_info = (
                f" | Asian: {state_means[s, asian_col]:.2f}"
                f" London: {state_means[s, london_col]:.2f}"
                f" NY: {state_means[s, ny_col]:.2f}"
            )

        print(f"  State {s} → {r:<16} "
              f"(ATR: {atr_means[s]:.3f}"
              f" | ADX: {state_means[s, adx_col]:.1f}"
              f" | Mom: {state_means[s, mom_col]:.4f}"
              f"{di_info}{session_info})")
    return state_map


# ================================================================
# STAGE 1d: VITERBI DECODE
# ================================================================
def decode_regimes(hmm_model, obs, state_map):
    # Sanitize full sequence (same clipping used during training)
    obs_safe = np.nan_to_num(obs, nan=0.0, posinf=4.0, neginf=-4.0)
    obs_safe = np.clip(obs_safe, -4.0, 4.0)
    _, states = hmm_model.decode(obs_safe, lengths=[len(obs_safe)], algorithm='viterbi')
    return np.array([state_map[s] for s in states])


# ================================================================
# STAGE 2: XGBOOST — Time-Series Training (70 / 15 / 15)
# ================================================================
def train_xgboost(X, y, le, xgb_params_override=None):
    """
    Trains XGBoost on GMM-HMM discovered regime labels.

    SPLIT STRATEGY — chronological 70 / 15 / 15:
        X_train = first 70%   — model learns from the past
        X_val   = next 15%    — early stopping uses this (never seen by train)
        X_test  = last 15%    — final holdout, most recent data (never seen by anything)

        CRITICAL: no shuffling. Time order is preserved exactly.
        Early stopping watches X_val only — X_test stays completely blind
        until the final evaluation line. This is proper walk-forward discipline.

    CLASS IMBALANCE:
        compute_class_weight('balanced') → per-class weights
        then mapped row-by-row to sample_weights vector.
        REVERSAL and COMPRESSION (< 5% of candles each) get up-weighted
        so XGBoost cannot ignore them by predicting LOW_VOL_RANGE always.

    XGBOOST PARAMS:
        objective = multi:softprob  — outputs full probability vector per candle
        num_class = 5               — explicit, matches 5 regime labels
        lr = 0.03                   — slower learning, better generalisation than 0.05
        gamma = 0.2                 — minimum loss reduction to make a split
        min_child_weight = 5        — prevents tiny leaves on rare regimes

    Args:
        X:  pd.DataFrame — full feature matrix, rolling z-scored, chronological
        y:  pd.Series   — regime labels aligned to X
        le: LabelEncoder fit on ALL_REGIMES

    Returns:
        (final_model, test_accuracy, results_dict)
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("\n❌ xgboost not installed. Run: pip install xgboost\n")
        return None, 0.0, {}

    from sklearn.metrics import (classification_report, accuracy_score,
                                 confusion_matrix)
    from sklearn.utils.class_weight import compute_class_weight

    print("\n" + "═" * 65)
    print("  STAGE 2 — XGBoost Training  (70 / 15 / 15 time-series split)")
    print("═" * 65)

    n       = len(X)
    y_enc   = le.transform(y)
    X_arr   = X.values
    X_cols  = X.columns.tolist()

    # ── 70 / 15 / 15 chronological split ─────────────────────────
    split1 = int(n * 0.70)
    split2 = int(n * 0.85)

    X_train, y_train = X_arr[:split1],        y_enc[:split1]
    X_val,   y_val   = X_arr[split1:split2],  y_enc[split1:split2]
    X_test,  y_test  = X_arr[split2:],        y_enc[split2:]

    print(f"  Total samples : {n:,}")
    print(f"  Train  (0–70%): {len(X_train):,}  "
          f"[{y.index[0].date()} → {y.index[split1-1].date()}]")
    print(f"  Val  (70–85%) : {len(X_val):,}  "
          f"[{y.index[split1].date()} → {y.index[split2-1].date()}]")
    print(f"  Test (85–100%): {len(X_test):,}  "
          f"[{y.index[split2].date()} → {y.index[-1].date()}]")
    print(f"  Regime dist (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ── Class weights → sample weights (row-level) ───────────────
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_map = dict(zip(classes, weights))
    sample_weights   = np.array([class_weight_map[yi] for yi in y_train])

    print(f"\n  Class weights (balanced):")
    for cls_enc, w in sorted(class_weight_map.items()):
        cls_name = le.inverse_transform([cls_enc])[0]
        print(f"    {cls_name:<18}: {w:.3f}")

    # ── XGBoost params ────────────────────────────────────────────
    xgb_params = dict(
        objective             = 'multi:softprob',
        num_class             = len(le.classes_),
        n_estimators          = 1000,
        max_depth             = 6,
        learning_rate         = 0.03,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        gamma                 = 0.2,
        min_child_weight      = 5,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        eval_metric           = 'mlogloss',
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 1,
        early_stopping_rounds = 50,
    )

    # Step 4 — Load Optuna-tuned params if available (override defaults)
    try:
        from paths import OPTUNA_PARAMS_PATH
        if os.path.exists(OPTUNA_PARAMS_PATH):
            with open(OPTUNA_PARAMS_PATH) as _f:
                _optuna = json.load(_f)
            _tuned = _optuna.get("best_params", {})
            if _tuned:
                xgb_params.update(_tuned)
                print(f"  [Trainer] Loaded Optuna-tuned params: {_tuned}")
    except Exception:
        pass   # Optuna params are optional

    # Explicit caller override (walk-forward, testing) — highest priority
    if xgb_params_override:
        xgb_params.update(xgb_params_override)
        print(f"  [Trainer] xgb_params_override applied: {xgb_params_override}")

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n  Training XGBoost (lr=0.03, max_depth=6, early_stop=50)...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,
        eval_set      = [(X_val, y_val)],   # val only — test stays blind
        verbose       = True,
    )

    n_trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') else 600
    print(f"\n  Early stopped at tree {n_trees} / {xgb_params['n_estimators']}")

    # ── Evaluate on blind test set ────────────────────────────────
    y_test_pred   = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n{'─' * 65}")
    print(f"  TEST SET RESULTS  (last 15% — model never saw this)")
    print(f"{'─' * 65}")
    print(f"  Accuracy: {test_accuracy:.3f}")
    print(f"\n  Per-regime report:")
    print(classification_report(
        y_test, y_test_pred,
        target_names = le.classes_,
        digits       = 3,
        zero_division= 0,
    ))

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_test, y_test_pred)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print("  ⚠  Watch: BULL↔BEAR confusion should be low | "
          "COMPRESSION often overlaps LOW_VOL")
    header = "  " + "".join(f"{r[:8]:>10}" for r in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i][:8]:<10}" +
              "".join(f"{v:>10}" for v in row))

    # ── Feature importance ────────────────────────────────────────
    imps = sorted(zip(X_cols, model.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    print("\n  Top 25 features by importance:")
    print("  ⚠  If random features dominate — something is wrong")
    for feat, imp in imps[:25]:
        print(f"  {feat:<40}: {imp:.4f} {'█' * int(imp * 400)}")

    results = {
        "n_train":          len(X_train),
        "n_val":            len(X_val),
        "n_test":           len(X_test),
        "test_accuracy":    round(float(test_accuracy), 4),
        "n_trees_used":     n_trees,
        "top_features":     [(f, round(float(i), 4)) for f, i in imps[:10]],
    }
    return model, test_accuracy, results


# ================================================================
# FULL PIPELINE
# ================================================================
def train(date_from=None, date_to=None, rebuild_features=True,
          n_states=None, n_init=DEFAULT_N_INIT,
          staging_dir=None, xgb_params_override=None):
    """
    Full training pipeline:
        1. Load / build features from historical OHLC data
        2. Scale features for HMM observations
        3. BIC selection of optimal n_states (if n_states=None)
        4. Fit GMM-HMM with n_init random initialisations — pick best LL
        5. Viterbi decode → regime labels
        6. Train XGBoost on discovered labels (walk-forward CV)
        7. Save all artefacts

    Args:
        n_states:            if None, BIC auto-selects. If int, uses that directly.
        n_init:              number of random GMM-HMM initialisations (default 15).
        staging_dir:         if set, ALL model files are written here instead of the
                             live REGIME_MODEL paths.  Used by auto_retrainer for
                             champion/challenger — new model stays in staging until
                             model_evaluator.compare() approves promotion.
        xgb_params_override: dict of XGBoost params to merge over defaults.
                             Used by walk-forward trainer and Optuna tuner.
    """
    # ── Resolve save paths ────────────────────────────────────────
    # When staging_dir is provided, write to staging; otherwise write to
    # the standard live paths defined in paths.py.
    if staging_dir:
        os.makedirs(staging_dir, exist_ok=True)
        _HMM_PATH   = os.path.join(staging_dir, "gmmhmm_model.joblib")
        _HMM_SCALER = os.path.join(staging_dir, "hmm_scaler.joblib")
        _XGB_PATH   = os.path.join(staging_dir, "regime_model.ubj")
        _LE_PATH    = os.path.join(staging_dir, "label_encoder.joblib")
        _META_PATH  = os.path.join(staging_dir, "model_meta.json")
        print(f"[Trainer] ⚡ Staging mode — writing to: {staging_dir}")
    else:
        _HMM_PATH   = HMM_PATH
        _HMM_SCALER = HMM_SCALER
        _XGB_PATH   = XGB_PATH
        _LE_PATH    = LE_PATH
        _META_PATH  = META_PATH
    try:
        from sklearn.preprocessing import StandardScaler, LabelEncoder
    except ImportError:
        print("[Trainer] scikit-learn not installed. Run: pip install scikit-learn")
        return False
    print("=" * 65)
    print("  Antigravity Bridge — GMM-HMM (5-state) + XGBoost Trainer v3")
    print("=" * 65)

    # ── Load / build features ─────────────────────────────────────
    if rebuild_features or not os.path.exists(FEATURES_PATH):
        print("\n[Trainer] Loading historical data...")
        dfs = {tf: load_data(tf, date_from, date_to)
               for tf in ("M5", "H1", "H4", "D1")}
        for tf, df in dfs.items():
            if df.empty:
                print(f"[Trainer] ERROR: No {tf} data. Run data_downloader.py first.")
                return False
            print(f"[Trainer] {tf}: {len(df):,} candles "
                  f"{df.index[0].date()} → {df.index[-1].date()}")

        features = build_multi_tf_features(
            dfs["M5"], dfs["H1"], dfs["H4"], dfs["D1"])
        if features.empty:
            print("[Trainer] ERROR: Empty feature matrix.")
            return False
        save_features(features, FEATURES_PATH)
    else:
        print(f"\n[Trainer] Loading cached features from {FEATURES_PATH}")
        features = load_features(FEATURES_PATH)
        if features is None or features.empty:
            print("[Trainer] Cached features invalid. Re-run without --no-rebuild.")
            return False

    print(f"\n[Trainer] Feature matrix: {len(features):,} × {len(features.columns)}")

    # ── GMM-HMM observation prep ──────────────────────────────────
    hmm_cols = [c for c in HMM_OBS_COLS if c in features.columns]
    if len(hmm_cols) < 4:
        hmm_cols = [c for c in features.columns
                    if any(k in c for k in
                           ['atr_ratio', 'adx', 'momentum_20', 'bb_width'])][:9]
    print(f"\n[GMM-HMM] Observation features ({len(hmm_cols)}): {hmm_cols}")

    hmm_scaler = StandardScaler()
    hmm_obs    = hmm_scaler.fit_transform(features[hmm_cols].values)

    # ── FIX: fit HMM on training rows only (first 70%) ───────────
    # Previously the HMM was fitted on a random subsample of ALL 463K rows,
    # which included the test period. This contaminated test labels because
    # Viterbi then decoded the test period using an HMM shaped by that data.
    # Fix: fit HMM on the chronological first 70% of observations only,
    # then decode the full sequence with that train-only model.
    # The test labels are now generated by a model that never saw test data.
    hmm_train_size = int(len(hmm_obs) * 0.70)
    hmm_obs_train  = hmm_obs[:hmm_train_size]
    print(f"\n[GMM-HMM] HMM will fit on first 70% of data ({hmm_train_size:,} rows) — test period excluded")

    # ── BIC n_states selection (if not forced) ────────────────────
    # FIX: cap BIC to range (3, 6) = tests [3, 4, 5] only.
    # Previous range (3, 7) tested up to n=6, which could return 6 in edge
    # cases producing duplicate BULL_TREND states. Gold is always 5-state.
    if n_states is None:
        n_states = select_n_states(hmm_obs_train, state_range=(3, 6), n_init=5)
        if n_states != DEFAULT_N_STATES:
            print(f"[Trainer] ⚠  BIC returned {n_states} — overriding to "
                  f"{DEFAULT_N_STATES} (Gold regime count is fixed at 5).")
            n_states = DEFAULT_N_STATES
    else:
        print(f"\n[Trainer] n_states={n_states} (user-specified, skipping BIC)")

    # ── Fit GMM-HMM with multiple initialisations ─────────────────
    hmm_model = fit_gmmhmm_best(hmm_obs_train, n_states=n_states,
                                n_mix=DEFAULT_N_MIX, n_iter=DEFAULT_N_ITER,
                                n_init=n_init)
    if hmm_model is None:
        return False

    # ── Map states → regimes ──────────────────────────────────────
    state_map = map_states_to_regimes(hmm_model, hmm_cols)

    # ── Decode full sequence ──────────────────────────────────────
    print("\n[GMM-HMM] Viterbi decoding full sequence...")
    raw_labels       = decode_regimes(hmm_model, hmm_obs, state_map)
    labels           = pd.Series(raw_labels, index=features.index, name='regime')
    labels.to_csv(LABELS_PATH, header=['regime'])

    counts = labels.value_counts()
    print(f"\n[GMM-HMM] Discovered distribution ({len(labels):,} candles):")
    for r in ALL_REGIMES:
        c = counts.get(r, 0)
        print(f"  {r:<20}: {c:>7,}  ({c/len(labels)*100:.1f}%)")

    # ── XGBoost training ──────────────────────────────────────────
    # Hard-check: warn loudly if expected features are missing
    base_avail   = [c for c in FEATURE_COLS if c in features.columns]
    base_missing = [c for c in FEATURE_COLS if c not in features.columns]
    if len(base_missing) > 5:
        print(f"\n[Trainer] ❌ HARD CHECK FAILED: {len(base_missing)} FEATURE_COLS missing "
              f"from feature matrix. This means feature_engineer.py and FEATURE_COLS "
              f"are out of sync. Cannot train reliably.\n"
              f"  Missing: {base_missing[:10]}...")
        return False
    elif base_missing:
        print(f"[Trainer] ⚠  {len(base_missing)} feature col(s) missing — will be "
              f"zero-filled: {base_missing}")

    X = features[base_avail].copy()

    # ── Rolling z-score normalisation ────────────────────────────
    # Applied to TF price features only. Time + persistence left raw.
    # First ~250 rows become NaN (warmup window) — dropped in valid mask below.
    print("\n[Trainer] Applying rolling z-score normalisation (window=500)...")
    X = apply_rolling_zscore(X, cols=[c for c in TF_FEATURE_COLS if c in X.columns],
                             window=500)

    # ── Persistence + transition features (post-HMM) ─────────────
    # Depends on HMM labels AND transmat — pass both.
    print("[Trainer] Computing persistence + transition features...")
    persistence_df = compute_persistence_features(labels,
                                                  hmm_model=hmm_model,
                                                  state_map=state_map)
    X = pd.concat([X, persistence_df.reindex(X.index)], axis=1)
    print(f"[Trainer] Persistence cols: {PERSISTENCE_FEATURE_COLS}")

    # ── FIX: exclude leaked columns from XGBoost training ────────
    # 'previous_regime_encoded' and 'regime_transition_prob' are derived
    # directly from the HMM label sequence that XGBoost is trying to predict.
    # Including them causes the model to trivially reconstruct its own targets
    # (tautology), producing fake 1.000 accuracy.
    # They are still computed and used at inference time in regime_detector.py
    # via _update_persistence() — that's correct because at inference time
    # the previous regime is a genuine past observation, not the current label.
    SAFE_PERSISTENCE_COLS = [
        'candles_since_regime_start',   # safe: count of elapsed candles
        'regime_duration_mean',         # safe: rolling mean of past durations
        # 'previous_regime_encoded'     # REMOVED: derived from HMM labels → leakage
        # 'regime_transition_prob'      # REMOVED: derived from HMM transmat → leakage
    ]

    # Final column list: technical + time + safe persistence only
    avail = base_avail + [c for c in SAFE_PERSISTENCE_COLS if c in X.columns]

    y     = labels.reindex(X.index)
    valid = X[avail].notna().all(axis=1) & y.notna()
    X, y  = X[avail][valid], y[valid]
    print(f"[Trainer] After z-score + dropna: {len(X):,} rows "
          f"(dropped {(~valid).sum():,} warmup rows)")

    le = LabelEncoder()
    le.fit(ALL_REGIMES)

    # ── FISF: Feature Importance Stability Filter ─────────────────
    # Removes features whose importance swings wildly across time windows.
    # Applied BEFORE final XGBoost — model trains only on stable features.
    # Uses loose thresholds for primary model (regime-specific signal is real).
    # ── FIX: save pre-FISF matrix for reversal detector ──────────
    # FISF drops low-importance features including h1_bb_width, h1_atr_ratio
    # etc. which the reversal pre-filter hard-requires. We save X_prefisf here
    # (before any columns are dropped) so the reversal detector can use it.
    # X_final (post-FISF) is still used for the main XGBoost.
    X_prefisf = X[avail].copy()

    print("\n[Trainer] Running FISF — Feature Importance Stability Filter...")
    try:
        from feature_stability import run_full_fisf_pipeline, filter_features
        stable_cols = run_full_fisf_pipeline(
            X[avail], y,
            weights=None,      # no sample weights for primary model
            mode="primary",
            use_shap=False,    # fast mode — SHAP is optional
            n_windows=6,
            verbose=True,
        )
        if len(stable_cols) >= 10:
            # Use FISF-filtered feature set for final XGBoost
            avail = stable_cols
            X_final = X[avail]
            print(f"[Trainer] FISF: using {len(avail)} stable features "
                  f"(was {len(X_prefisf.columns)})")
        else:
            print(f"[Trainer] FISF returned too few features "
                  f"({len(stable_cols)}) — using full set.")
            X_final = X[avail]
    except Exception as e:
        print(f"[Trainer] FISF skipped ({e}) — using full feature set.")
        X_final = X[avail]

    # No StandardScaler — rolling z-score already normalised TF features.
    # XGBoost tree splits are invariant to monotonic transforms anyway.
    xgb_model, accuracy, xgb_results = train_xgboost(
        X_final, y, le, xgb_params_override=xgb_params_override)
    if xgb_model is None:
        return False

    # ── Save ──────────────────────────────────────────────────────
    joblib.dump(hmm_model,  _HMM_PATH,   compress=3)
    joblib.dump(hmm_scaler, _HMM_SCALER, compress=3)
    joblib.dump(le,         _LE_PATH,    compress=3)
    print(f"\n✅ Saved (joblib): GMM-HMM   → {_HMM_PATH}")
    print(f"✅ Saved (joblib): HMM scaler → {_HMM_SCALER}")
    print(f"✅ Saved (joblib): LE         → {_LE_PATH}")

    xgb_model.save_model(_XGB_PATH)
    print(f"✅ Saved (native): XGBoost   → {_XGB_PATH}")

    n_trees = (xgb_model.best_iteration + 1
               if hasattr(xgb_model, 'best_iteration') else 600)

    meta = {
        "trained_date":        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        "architecture":        "GMM-HMM (5-state, time-conditioned) + XGBoost v5",
        "regime_system":       "REVERSAL/BULL_TREND/BEAR_TREND/COMPRESSION/LOW_VOL_RANGE",
        "normalisation":       "rolling_zscore_window500_tf_cols_only",
        "split_strategy":      "chronological_70_15_15",
        "hmm_n_states":        n_states,
        "hmm_n_mix":           DEFAULT_N_MIX,
        "hmm_n_init":          n_init,
        "hmm_obs_cols":        hmm_cols,
        "hmm_obs_price_cols":  [c for c in hmm_cols if c not in
                                ('hour_sin','hour_cos','is_asian','is_london',
                                 'is_ny','dow_sin','dow_cos','is_monday','is_friday')],
        "hmm_obs_time_cols":   [c for c in hmm_cols if c in
                                ('hour_sin','hour_cos','is_asian','is_london',
                                 'is_ny','dow_sin','dow_cos','is_monday','is_friday')],
        "state_map":           {str(k): v for k, v in state_map.items()},
        "xgb_results":         xgb_results,
        "xgb_trees_used":      n_trees,
        "training_rows":       xgb_results.get("n_train", 0),
        "val_rows":            xgb_results.get("n_val",   0),
        "test_rows":           xgb_results.get("n_test",  0),
        "test_accuracy":       round(float(accuracy), 4),
        "top_features":        xgb_results.get("top_features", []),
        "feature_cols":        avail,
        "n_feature_cols":      len(avail),
        "persistence_cols":    PERSISTENCE_FEATURE_COLS,
        "regimes":             ALL_REGIMES,
        "date_from":           str(date_from) if date_from else "all",
        "date_to":             str(date_to)   if date_to   else "all",
        "label_distribution":  {r: int(counts.get(r, 0)) for r in ALL_REGIMES},
        "save_formats": {
            "hmm":       "joblib",
            "xgb":       "native_ubj",
            "hmm_scaler":"joblib",
            "le":        "joblib",
        },
        "staging_dir": staging_dir or "live",
    }
    with open(_META_PATH, 'w') as f:
        json.dump(meta, f, indent=4)

    print(f"✅ Saved: Meta     → {_META_PATH}")

    # ── Session-Regime Duration Profiles ─────────────────────────
    # Build after HMM decode so labels are available.
    # Profiles power: session timing in predict(), adaptive thresholds in meta_labeller.
    try:
        from session_profiler import build_session_profiles
        build_session_profiles(labels, features)
        print("✅ Saved: Session profiles → Data/Models/Regime/session_profiles.json")
    except Exception as e:
        print(f"⚠  Session profiler failed (non-fatal): {e}")

    # ── REVERSAL binary pre-filter ────────────────────────────────
    # Trained on the same features + labels immediately after main model.
    # Uses only the 14 volatility/context features it needs — fast (~60s).
    # Saved to REVERSAL_DETECTOR_PATH — loaded by regime_detector at startup.
    print("\n[Trainer] Training REVERSAL binary pre-filter (Stage 1)...")
    try:
        from reversal_detector import train as train_reversal
        # Pass X_prefisf (pre-FISF full feature matrix) — NOT X_final.
        # FISF drops h1_bb_width, h1_atr_ratio etc. which the reversal
        # detector hard-requires. X_prefisf has all columns intact.
        rev_results = train_reversal(X_prefisf, y, verbose=True)
        if rev_results:
            print(f"✅ REVERSAL pre-filter: "
                  f"recall={rev_results.get('recall',0):.3f} "
                  f"precision={rev_results.get('precision',0):.3f} "
                  f"F2={rev_results.get('f2',0):.3f}")
        else:
            print("⚠  REVERSAL pre-filter training skipped (insufficient REVERSAL candles).")
    except Exception as e:
        print(f"⚠  REVERSAL pre-filter failed (non-fatal): {e}")

    print("\n✅ Training complete.")
    print(f"   {n_states} regimes discovered | {n_init} initialisations used")
    print(f"   Run backtest_engine.py or main_bot.py next.")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train GMM-HMM (5-state, multi-init) + XGBoost regime detector')
    parser.add_argument('--from',        dest='date_from',   type=str,  default=None)
    parser.add_argument('--to',          dest='date_to',     type=str,  default=None)
    parser.add_argument('--no-rebuild',                      action='store_true')
    parser.add_argument('--n-states',    dest='n_states',    type=int,  default=None,
                        help='Force n_states (default: BIC auto-select). Recommended: 5')
    parser.add_argument('--n-init',      dest='n_init',      type=int,  default=DEFAULT_N_INIT,
                        help=f'GMM-HMM random initialisations (default: {DEFAULT_N_INIT})')
    parser.add_argument('--staging-dir', dest='staging_dir', type=str,  default=None,
                        help='Write model files here instead of live paths (champion/challenger)')
    args = parser.parse_args()

    date_from = datetime.strptime(args.date_from, '%Y-%m-%d') if args.date_from else None
    date_to   = datetime.strptime(args.date_to,   '%Y-%m-%d') if args.date_to   else None
    train(date_from, date_to,
          rebuild_features = not args.no_rebuild,
          n_states         = args.n_states,
          n_init           = args.n_init,
          staging_dir      = args.staging_dir)