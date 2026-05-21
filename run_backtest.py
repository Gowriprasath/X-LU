"""
run_backtest.py — Antigravity Bridge
======================================
Master orchestrator. Run this ONE file and everything happens in order:

    Step 1 → data_downloader.py   — pulls 9 years of XAUUSD from MT5
    Step 2 → trainer.py           — trains GMM-HMM + XGBoost regime detector
    Step 3 → backtest_engine.py   — replays full pipeline on historical data

Usage:
    python run_backtest.py                          # full 9-year run
    python run_backtest.py --from 2024-01-01 --to 2024-03-31   # test run
    python run_backtest.py --resume                 # resume interrupted backtest
    python run_backtest.py --skip-download          # skip if data already downloaded
    python run_backtest.py --skip-download --skip-train  # jump straight to backtest

Pre-flight requirements:
    1. MT5 open + logged in + XAUUSD visible in Market Watch
    2. pip install pytz scikit-learn pandas numpy python-dotenv hmmlearn xgboost joblib anthropic
    3. CLAUDE_API_KEY_1 set in .env file (see .env.example)

BUG-3 FIX: preflight now checks CLAUDE_API_KEY_1 (not GEMINI_API_KEY).
           The old check caused sys.exit(1) on every Claude-only deployment,
           making the entire backtest system non-functional.
"""

import os
import sys
import argparse
import importlib
import importlib.util
import subprocess
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.join(SCRIPT_DIR, 'Backtest')
QUANT_DIR    = os.path.join(SCRIPT_DIR, 'Quant', 'regime_detector')

DOWNLOADER   = os.path.join(BACKTEST_DIR, 'data_downloader.py')
TRAINER      = os.path.join(QUANT_DIR,    'trainer.py')
ENGINE       = os.path.join(BACKTEST_DIR, 'backtest_engine.py')


def banner():
    print()
    print("=" * 65)
    print("  ⚡ ANTIGRAVITY BRIDGE — BACKTEST ORCHESTRATOR")
    print("=" * 65)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Root    : {SCRIPT_DIR}")
    print("=" * 65)
    print()


def step(number, title):
    print()
    print(f"{'─' * 65}")
    print(f"  STEP {number}: {title}")
    print(f"{'─' * 65}")
    print()


def run(script, extra_args=None):
    """Runs a Python script in a subprocess. Exits on failure."""
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {script}")
        print(f"   Return code: {result.returncode}")
        print(f"   Fix the error above and re-run.")
        sys.exit(result.returncode)
    print(f"\n✅ {os.path.basename(script)} completed successfully.")


def preflight_check(skip_download):
    """Checks environment before starting."""
    print("  Pre-flight checks...")
    ok = True

    # ── API key (BUG-3 FIX: check CLAUDE_API_KEY_1, not GEMINI_API_KEY) ──
    env_file = os.path.join(SCRIPT_DIR, '.env')
    _placeholder_prefixes = ('your_claude_api_key', 'sk-ant-placeholder', 'your_key_here')

    def _has_claude_key(text):
        """True if any CLAUDE_API_KEY_N is set and not a placeholder."""
        for i in range(1, 4):
            tag = f'CLAUDE_API_KEY_{i}'
            if tag in text:
                # Extract value from "CLAUDE_API_KEY_1=sk-ant-..."
                for line in text.splitlines():
                    if line.strip().startswith(tag):
                        val = line.split('=', 1)[-1].strip().strip('"').strip("'")
                        if val and not any(val.startswith(p) for p in _placeholder_prefixes):
                            return True
        return False

    if os.path.exists(env_file):
        with open(env_file, encoding='utf-8', errors='replace') as f:
            content = f.read()
        if _has_claude_key(content):
            print("  ✅ .env found with CLAUDE_API_KEY")
        else:
            print("  ❌ .env missing or CLAUDE_API_KEY_1 not set")
            ok = False
    else:
        # Check environment variables directly
        if any(os.getenv(f'CLAUDE_API_KEY_{i}', '') for i in range(1, 4)):
            print("  ✅ CLAUDE_API_KEY found in environment")
        else:
            print("  ❌ CLAUDE_API_KEY_1 not set — add to .env file")
            ok = False

    # ── Required scripts ───────────────────────────────────────────
    # Verify keys are actually readable
    from dotenv import load_dotenv
    load_dotenv()
    _k1 = os.getenv('CLAUDE_API_KEY_1', '')
    _k2 = os.getenv('CLAUDE_API_KEY_2', '')
    _k3 = os.getenv('CLAUDE_API_KEY_3', '')
    _keys_found = sum(1 for k in [_k1, _k2, _k3] if k.strip())
    if _keys_found == 0:
        print("[Backtest] ❌ CRITICAL: No Claude API keys "
              "found in .env — all AI calls will fail.")
        print("           Set CLAUDE_API_KEY_1 in .env "
              "and restart.")
        sys.exit(1)
    else:
        print(f"[Backtest] ✓ {_keys_found} API key(s) "
              f"found in .env")

    for path, label in [
        (DOWNLOADER, 'data_downloader.py'),
        (TRAINER,    'trainer.py'),
        (ENGINE,     'backtest_engine.py'),
    ]:
        if os.path.exists(path):
            print(f"  ✅ {label} found")
        else:
            print(f"  ❌ {label} not found at {path}")
            ok = False

    # ── Python package dependencies ────────────────────────────────
    # (hmmlearn and xgboost are new requirements for GMM-HMM + XGBoost trainer)
    required_packages = [
        ('sklearn',   'scikit-learn'),
        ('pandas',    'pandas'),
        ('numpy',     'numpy'),
        ('pytz',      'pytz'),
        ('dotenv',    'python-dotenv'),
        ('joblib',    'joblib'),
        ('hmmlearn',  'hmmlearn'),
        ('xgboost',   'xgboost'),
        ('shap',      'shap'),          # FISF SHAP stability + meta explanations
    ]
    for import_name, install_name in required_packages:
        if importlib.util.find_spec(import_name):
            print(f"  ✅ {install_name}")
        else:
            print(f"  ❌ {install_name} not installed — run: pip install {install_name}")
            ok = False

    # ── Data check ─────────────────────────────────────────────────
    if skip_download:
        from paths import MARKET_DATA_DIR as _market_dir
        csv_files = [f for f in os.listdir(_market_dir) if f.endswith('.csv')] \
            if os.path.exists(_market_dir) else []
        if csv_files:
            print(f"  ✅ Data/Market/ has {len(csv_files)} CSV files (--skip-download OK)")
        else:
            print(f"  ⚠️  --skip-download set but no CSV files found in Data/Market/")
            print(f"     Will run download anyway to get data.")

    # ── Model check ────────────────────────────────────────────────
    # XGBoost now saved as .ubj (native format)
    from paths import REGIME_XGB_PATH as _xgb_path
    if os.path.exists(_xgb_path):
        print(f"  ✅ Trained model found (Data/Models/Regime/) — can skip with --skip-train")
    else:
        print(f"  ℹ️  No trained model yet (Data/Models/Regime/) — training required")

    print()
    if not ok:
        print("  ❌ Pre-flight failed. Fix issues above before running.")
        sys.exit(1)
    print("  ✅ All checks passed. Starting...\n")


def main():
    parser = argparse.ArgumentParser(
        description='Antigravity Bridge — Backtest Orchestrator')

    parser.add_argument('--from', dest='date_from', type=str, default=None,
                        help='Backtest start date YYYY-MM-DD (default: all available)')
    parser.add_argument('--to', dest='date_to', type=str, default=None,
                        help='Backtest end date YYYY-MM-DD (default: all available)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume backtest from last checkpoint')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download (use existing CSV files)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip regime detector training (use existing model)')
    args = parser.parse_args()

    banner()

    # ── Pre-flight ────────────────────────────────────────────────
    preflight_check(args.skip_download)

    # ── Step 0: Fetch/update ForexFactory news calendar ──────────
    # Runs automatically before every backtest so news gate dates
    # are always accurate. Only fetches MISSING weeks — already-cached
    # weeks are skipped. First run: ~35 min. Subsequent runs: seconds.
    step(0, "UPDATING FOREX FACTORY NEWS CALENDAR")
    FF_FETCHER = os.path.join(BACKTEST_DIR, 'ff_fetcher.py')
    if os.path.exists(FF_FETCHER):
        print("  Fetching any missing weeks from ForexFactory...")
        print("  (Already-cached weeks are skipped — only new weeks are fetched)")
        print("  First run takes ~35 min. Re-runs are near-instant.\n")
        try:
            result = subprocess.run(
                [sys.executable, FF_FETCHER,
                 '--from', args.date_from or '2017-01-01',
                 '--to',   args.date_to   or datetime.now().strftime('%Y-%m-%d')],
                cwd=SCRIPT_DIR, timeout=3600   # 1 hour max
            )
            if result.returncode == 0:
                print("\n✅ ff_fetcher.py completed — news calendar up to date.")
            else:
                print("\n⚠️  FF fetcher returned non-zero — continuing with cached/seeded data.")
        except subprocess.TimeoutExpired:
            print("\n⚠️  FF fetcher timed out — continuing with cached/seeded data.")
        except Exception as e:
            print(f"\n⚠️  FF fetcher error ({e}) — continuing with cached/seeded data.")
    else:
        print("  ⚠️  ff_fetcher.py not found — using seeded news approximation.")

    # ── Step 1: Download historical data ─────────────────────────
    if not args.skip_download:
        step(1, "DOWNLOADING HISTORICAL DATA (M5 / H1 / H4 / D1)")
        print("  This pulls 9 years of XAUUSD from MT5.")
        print("  Resumes automatically if interrupted.")
        print("  Estimated time: 10–20 minutes depending on MT5 speed.\n")
        run(DOWNLOADER)
    else:
        step(1, "DATA DOWNLOAD — SKIPPED (--skip-download)")
        print("  Using existing CSV files in Backtest/data/")

    # ── Step 2: Train regime detector ────────────────────────────
    if not args.skip_train:
        step(2, "TRAINING REGIME DETECTOR (GMM-HMM + XGBoost)")
        print("  Stage 1 — GMM-HMM: unsupervised regime discovery from price data.")
        print("  Stage 2 — XGBoost: fast single-candle classifier (<2ms inference).")
        print("  Saves: GMM-HMM (joblib) + XGBoost (native .ubj) + scalers.")
        print("  Estimated time: 5–15 minutes depending on data size.\n")

        train_args = []
        if args.date_from:
            train_args += ['--from', args.date_from]
        if args.date_to:
            train_args += ['--to', args.date_to]

        run(TRAINER, train_args)
    else:
        step(2, "REGIME DETECTOR TRAINING — SKIPPED (--skip-train)")
        print("  Using existing trained model.")

    # ── Step 3: Run backtest ──────────────────────────────────────
    step(3, "RUNNING BACKTEST ENGINE")

    if args.resume:
        print("  Resuming from last checkpoint in backtest_tracker.json\n")
    elif args.date_from or args.date_to:
        print(f"  Date range: {args.date_from or 'earliest'} → {args.date_to or 'latest'}\n")
    else:
        print("  Full 9-year run. Estimated time: several hours.\n")
        print("  💡 Tip: If interrupted, re-run with --resume to continue.\n")
        print("  💰 Estimated API cost: $80–100 on Claude 2.5 Flash.\n")

    # â”€â”€ Reset circuit breaker for clean backtest run â”€â”€
    # The circuit breaker is designed for live trading.
    # In backtest mode it must always start fresh so
    # every valid signal gets a Claude evaluation.
    try:
        from ai_client import _cb_lock
        import ai_client as _ac
        with _cb_lock:
            _ac._cb_tripped           = False
            _ac._cb_consecutive_fails = 0
            _ac._cb_tripped_at        = 0.0
        print("[Backtest] Circuit breaker reset — clean AI state.")
    except Exception as _cb_e:
        print(f"[Backtest] Could not reset circuit breaker: {_cb_e}")

    engine_args = []
    if args.date_from: engine_args += ['--from', args.date_from]
    if args.date_to:   engine_args += ['--to',   args.date_to]
    if args.resume:    engine_args += ['--resume']

    run(ENGINE, engine_args)

    # ── Done ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  🏁 BACKTEST COMPLETE")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()
    print("  Results are in:")
    print("  📊 Data/Backtest/backtest_tracker.json  — stats + trade count")
    print("  🧠 Data/Memory/trade_memory.json         — all trade records")
    print("  📝 Data/Memory/Filter/wisdom.json        — distilled AI lessons")
    print("  🎯 Quant/regime_detector/            — GMM-HMM + XGBoost model")
    print("  📝 Memory/wisdom.json               — distilled lessons")
    print("  🎯 Quant/regime_detector/data/model — GMM-HMM + XGBoost model")
    print()
    print("  Next step: python main_bot.py  — go live!")
    print()


if __name__ == '__main__':
    main()
