"""
paths.py — Single Source of Truth for All Data File Locations
==============================================================
Every file in the project that reads or writes data imports from here.

To relocate the Data folder (e.g. to an external drive, NAS, or server):
    Change DATA_ROOT below — nothing else needs to change anywhere.

Data/ folder sits OUTSIDE the code tree, at the project root:

  Gold_AI_Bridge/          ← project root (where main_bot.py lives)
  │
  ├── Data/                ← all runtime data (this file controls it)
  │   ├── Market/          ← OHLCV CSV files downloaded by data_downloader.py
  │   ├── Models/
  │   │   ├── Regime/      ← GMM-HMM, XGBoost, scaler, label_encoder,
  │   │   │                   model_meta.json, session_profiles.json,
  │   │   │                   fisf_primary.json, fisf_meta.json, bic_selection.json
  │   │   └── Meta/        ← meta_model.ubj, meta_model_meta.json
  │   ├── Memory/          ← all runtime memory JSON files
  │   │   └── Filter/      ← wisdom.json, ai_lessons.json, human_rules.json,
  │   │                       keywords.json, misswish_keywords.json, wisdom_tracker.json
  │   ├── Regime/          ← features.csv, labels.csv, drift_log.json,
  │   │                       reload_flag.json, retrain_history.json
  │   ├── Episodes/        ← episodes.json, active_episode.json  (RL trade data)
  │   ├── Meta/            ← meta_wisdom_log.json
  │   ├── Backtest/        ← backtest_tracker.json, download_progress.json
  │   └── Logs/            ← execution_errors.log, other runtime logs
  │
  ├── main_bot.py
  ├── master_controls.py
  ├── ai_client.py
  ├── run_backtest.py
  ├── paths.py             ← this file
  ├── AI/
  ├── Backtest/
  ├── Integration/
  ├── Memory/
  ├── Python Files/
  ├── Quant/
  ├── Strategy/
  └── Strategy_AI/
"""

import os

# ══════════════════════════════════════════════════════════════════════
# PROJECT ROOT
# ══════════════════════════════════════════════════════════════════════
# This file lives at the project root (same level as main_bot.py).
# All code paths that need PROJECT_ROOT can import it from here.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════
# DATA ROOT — CHANGE THIS ONE LINE TO RELOCATE EVERYTHING
# ══════════════════════════════════════════════════════════════════════
DATA_ROOT = os.path.join(PROJECT_ROOT, 'Data')

# ══════════════════════════════════════════════════════════════════════
# SUBDIRECTORIES
# ══════════════════════════════════════════════════════════════════════
MARKET_DIR      = os.path.join(DATA_ROOT, 'Market')
REGIME_MODEL    = os.path.join(DATA_ROOT, 'Models', 'Regime')
META_MODEL      = os.path.join(DATA_ROOT, 'Models', 'Meta')

# Step 1 — Champion/Challenger staging area
# Challenger models are written here; promoted to REGIME_MODEL only on approval.
CHALLENGER_DIR        = os.path.join(DATA_ROOT, 'Models', 'Challenger')
CHAMPION_LOG_PATH     = os.path.join(DATA_ROOT, 'Regime', 'champion_challenger_log.json')

# Step 2 — Versioned model archive (rollback support)
# Layout: Data/Models/Versions/v{YYYY_MM_DD_HHMM}/  — keeps last 5 by default.
VERSIONS_DIR          = os.path.join(DATA_ROOT, 'Models', 'Versions')
CURRENT_VERSION_PATH  = os.path.join(DATA_ROOT, 'Models', 'Regime', 'current_version.txt')

# Step 4 — Optuna best-params cache
# trainer.py loads this if it exists; otherwise uses hardcoded defaults.
OPTUNA_PARAMS_PATH    = os.path.join(DATA_ROOT, 'Models', 'Regime', 'best_xgb_params.json')
OPTUNA_STUDY_PATH     = os.path.join(DATA_ROOT, 'Models', 'Regime', 'optuna_study.json')

MEMORY_DIR      = os.path.join(DATA_ROOT, 'Memory')
FILTER_DIR      = os.path.join(DATA_ROOT, 'Memory', 'Filter')
REGIME_DATA     = os.path.join(DATA_ROOT, 'Regime')
EPISODES_DIR    = os.path.join(DATA_ROOT, 'Episodes')
META_DATA       = os.path.join(DATA_ROOT, 'Meta')
BACKTEST_OUT    = os.path.join(DATA_ROOT, 'Backtest')
LOGS_DIR        = os.path.join(DATA_ROOT, 'Logs')

# ══════════════════════════════════════════════════════════════════════
# MARKET DATA
# ══════════════════════════════════════════════════════════════════════
MARKET_DATA_DIR = MARKET_DIR   # root folder — individual CSVs live here

# ══════════════════════════════════════════════════════════════════════
# REGIME MODEL ARTEFACTS  (Data/Models/Regime/)
# ══════════════════════════════════════════════════════════════════════
HMM_PATH              = os.path.join(REGIME_MODEL, 'gmmhmm_model.joblib')
HMM_SCALER_PATH       = os.path.join(REGIME_MODEL, 'hmm_scaler.joblib')
REGIME_XGB_PATH       = os.path.join(REGIME_MODEL, 'regime_model.ubj')
REGIME_SCALER_PATH    = os.path.join(REGIME_MODEL, 'scaler.joblib')
LABEL_ENCODER_PATH    = os.path.join(REGIME_MODEL, 'label_encoder.joblib')
MODEL_META_PATH       = os.path.join(REGIME_MODEL, 'model_meta.json')
SESSION_PROFILES_PATH = os.path.join(REGIME_MODEL, 'session_profiles.json')
FISF_PRIMARY_PATH     = os.path.join(REGIME_MODEL, 'fisf_primary.json')
FISF_META_PATH        = os.path.join(REGIME_MODEL, 'fisf_meta.json')
BIC_PATH              = os.path.join(REGIME_MODEL, 'bic_selection.json')

# REVERSAL binary pre-filter — trained alongside the main model
# Separate binary XGBoost: "is this REVERSAL or not"
# Runs first in predict(); overrides 5-class model when confident
REVERSAL_DETECTOR_PATH     = os.path.join(REGIME_MODEL, 'reversal_detector.ubj')
REVERSAL_DETECTOR_META_PATH= os.path.join(REGIME_MODEL, 'reversal_detector_meta.json')

# ══════════════════════════════════════════════════════════════════════
# REGIME TRAINING DATA  (Data/Regime/)
# ══════════════════════════════════════════════════════════════════════
FEATURES_PATH         = os.path.join(REGIME_DATA, 'features.csv')
LABELS_PATH           = os.path.join(REGIME_DATA, 'labels.csv')
DRIFT_LOG_PATH        = os.path.join(REGIME_DATA, 'drift_log.json')
RELOAD_FLAG_PATH      = os.path.join(REGIME_DATA, 'reload_flag.json')
RETRAIN_HISTORY_PATH  = os.path.join(REGIME_DATA, 'retrain_history.json')

# ══════════════════════════════════════════════════════════════════════
# META-LABELLER MODEL  (Data/Models/Meta/)
# ══════════════════════════════════════════════════════════════════════
META_MODEL_PATH       = os.path.join(META_MODEL, 'meta_model.ubj')
META_MODEL_META_PATH  = os.path.join(META_MODEL, 'meta_model_meta.json')

# ══════════════════════════════════════════════════════════════════════
# META DATA  (Data/Meta/)
# ══════════════════════════════════════════════════════════════════════
META_WISDOM_LOG_PATH  = os.path.join(META_DATA, 'meta_wisdom_log.json')

# ══════════════════════════════════════════════════════════════════════
# MEMORY  (Data/Memory/)
# ══════════════════════════════════════════════════════════════════════
TRADE_MEMORY_PATH       = os.path.join(MEMORY_DIR, 'trade_memory.json')
CONTINUATION_MEM_PATH   = os.path.join(MEMORY_DIR, 'continuation_memory.json')
RISK_STATE_PATH         = os.path.join(MEMORY_DIR, 'risk_state.json')
SHADOW_JOURNAL_PATH     = os.path.join(MEMORY_DIR, 'shadow_journal.json')
MISSWISH_MEMORY_PATH    = os.path.join(MEMORY_DIR, 'misswish_memory.json')
SCOUT_LOG_PATH          = os.path.join(MEMORY_DIR, 'strategy_scout_log.json')
COUNTERFACTUAL_LOG_PATH = os.path.join(MEMORY_DIR, 'counterfactual_log.json')
SHADOW_GATE_AUDIT_PATH  = os.path.join(MEMORY_DIR, 'shadow_gate_audit.json')
POST_MORTEM_TRACKER     = os.path.join(MEMORY_DIR, 'last_pm_date.txt')
TRADE_MEMORY_ARCHIVE_DIR = MEMORY_DIR   # archives written here with dated filename

# ══════════════════════════════════════════════════════════════════════
# WISDOM / FILTER  (Data/Memory/Filter/)
# ══════════════════════════════════════════════════════════════════════
WISDOM_PATH           = os.path.join(FILTER_DIR, 'wisdom.json')
AI_LESSONS_PATH       = os.path.join(FILTER_DIR, 'ai_lessons.json')
HUMAN_RULES_PATH      = os.path.join(FILTER_DIR, 'human_rules.json')
KEYWORDS_PATH         = os.path.join(FILTER_DIR, 'keywords.json')
MISSWISH_KW_PATH      = os.path.join(FILTER_DIR, 'misswish_keywords.json')
WISDOM_TRACKER_PATH   = os.path.join(FILTER_DIR, 'wisdom_tracker.json')

# ══════════════════════════════════════════════════════════════════════
# EPISODES  (Data/Episodes/)
# ══════════════════════════════════════════════════════════════════════
EPISODES_PATH         = os.path.join(EPISODES_DIR, 'episodes.json')
ACTIVE_EPISODE_PATH   = os.path.join(EPISODES_DIR, 'active_episode.json')

# ══════════════════════════════════════════════════════════════════════
# BACKTEST OUTPUTS  (Data/Backtest/)
# ══════════════════════════════════════════════════════════════════════
BACKTEST_TRACKER_PATH    = os.path.join(BACKTEST_OUT, 'backtest_tracker.json')
DOWNLOAD_PROGRESS_PATH   = os.path.join(BACKTEST_OUT, 'download_progress.json')
PNL_STATS_PATH           = os.path.join(BACKTEST_OUT, 'pnl_stats.json')

# News calendar — real ForexFactory data (populated by ff_fetcher.py)
# Falls back to seeded approximations in news_history.py if not yet fetched.
FF_NEWS_CACHE_PATH       = os.path.join(BACKTEST_OUT, 'ff_news_calendar.json')
FF_FETCH_META_PATH       = os.path.join(BACKTEST_OUT, 'ff_fetch_meta.json')

# ══════════════════════════════════════════════════════════════════════
# LOGS  (Data/Logs/)
# ══════════════════════════════════════════════════════════════════════
EXECUTION_LOG_PATH    = os.path.join(LOGS_DIR, 'execution_errors.log')

# ══════════════════════════════════════════════════════════════════════
# AI CONTEXT (project-relative — not in Data/)
# latest_context.txt is a scratch file written each cycle, not persisted
# between deployments, so it stays in the AI/ source folder.
# ══════════════════════════════════════════════════════════════════════
AI_CONTEXT_PATH       = os.path.join(PROJECT_ROOT, 'AI', 'latest_context.txt')

# ══════════════════════════════════════════════════════════════════════
# STRATEGY_AI  (project-relative — user reviews and promotes these)
# These are code-adjacent files (user manually promotes proposals to
# confirmed/). They stay in the project tree, not in Data/.
# ══════════════════════════════════════════════════════════════════════
STRATEGY_AI_DIR       = os.path.join(PROJECT_ROOT, 'Strategy_AI')
PROPOSALS_DIR         = os.path.join(STRATEGY_AI_DIR, 'proposals')
CONFIRMED_DIR         = os.path.join(STRATEGY_AI_DIR, 'confirmed')


# ══════════════════════════════════════════════════════════════════════
# create_all_dirs()
# ══════════════════════════════════════════════════════════════════════
def create_all_dirs():
    """
    Creates the full Data/ directory tree.
    Safe to call multiple times (exist_ok=True).
    Called automatically at bot startup and from run_backtest.py preflight.
    """
    dirs = [
        MARKET_DIR,
        REGIME_MODEL,
        META_MODEL,
        CHALLENGER_DIR,          # Step 1 — challenger staging
        VERSIONS_DIR,            # Step 2 — versioned archive
        MEMORY_DIR,
        FILTER_DIR,
        REGIME_DATA,
        EPISODES_DIR,
        META_DATA,
        BACKTEST_OUT,
        LOGS_DIR,
        PROPOSALS_DIR,
        CONFIRMED_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    create_all_dirs()
    print(f"Data root: {DATA_ROOT}")
    print("\nDirectory tree created:")
    for root, dirs, files in os.walk(DATA_ROOT):
        level  = root.replace(DATA_ROOT, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
    print("\nAll paths ready. You can now run main_bot.py or run_backtest.py.")
