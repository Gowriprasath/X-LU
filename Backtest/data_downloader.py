"""
Backtest/data_downloader.py — Antigravity Bridge
=================================================
Downloads historical XAUUSD OHLCV data from MT5 in 30-day chunks
and saves to CSV files in Backtest/data/.

Run this ONCE before running the backtest engine.
Run again any time you want to extend the data range.

Usage:
    python data_downloader.py

Output files (Data/Market/):
    XAUUSD_M5_YYYY-MM.csv   — 5-minute candles  (backtest execution)
    XAUUSD_H1_YYYY-MM.csv   — 1-hour candles    (regime detector + prompt context)
    XAUUSD_H4_YYYY-MM.csv   — 4-hour candles    (Elliott Wave / structure context)
    XAUUSD_D1_YYYY-MM.csv   — daily candles     (HTF bias + daily close rule)

Progress is saved to Data/Backtest/download_progress.json.
If download is interrupted, re-running resumes from last completed chunk.
"""

import os
import sys
import json
import time
import pytz
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────
BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))
_root_dl = os.path.normpath(os.path.join(BACKTEST_DIR, '..'))
import sys as _sys_dl
if _root_dl not in _sys_dl.path: _sys_dl.path.insert(0, _root_dl)
from paths import MARKET_DATA_DIR as DATA_DIR, DOWNLOAD_PROGRESS_PATH as PROGRESS_FILE,                   create_all_dirs as _cad_dl
_cad_dl()

# ── Config ─────────────────────────────────────────────────────────
SYMBOL     = "XAUUSD"
NY_TZ      = pytz.timezone('America/New_York')

# Target range: ~9 years of data
# Adjust START_DATE if you want more or less history
START_DATE = datetime(2016, 1, 1)   # 9 years back from 2025
END_DATE   = datetime.now()

# Timeframes to download
# (mt5_constant, label, chunk_days)
# Larger TFs need bigger chunks — M5 produces ~8,640 rows/month
TIMEFRAMES = [
    (mt5.TIMEFRAME_M5,  "M5",  30),   # 30-day chunks (~8,640 rows each)
    (mt5.TIMEFRAME_H1,  "H1",  90),   # 90-day chunks (~2,160 rows each)
    (mt5.TIMEFRAME_H4,  "H4",  180),  # 180-day chunks (~1,080 rows each)
    (mt5.TIMEFRAME_D1,  "D1",  365),  # 1-year chunks  (~252 rows each)
]

# MT5 rate-limit safety — pause between chunks to avoid connection drops
CHUNK_PAUSE_SECONDS = 0.5


# ================================================================
# PROGRESS TRACKER
# ================================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4, default=str)


def mark_chunk_done(progress, tf_label, chunk_key):
    if tf_label not in progress:
        progress[tf_label] = {"completed_chunks": [], "last_updated": ""}
    if chunk_key not in progress[tf_label]["completed_chunks"]:
        progress[tf_label]["completed_chunks"].append(chunk_key)
    progress[tf_label]["last_updated"] = datetime.now().isoformat()
    save_progress(progress)


def is_chunk_done(progress, tf_label, chunk_key):
    return chunk_key in progress.get(tf_label, {}).get("completed_chunks", [])


# ================================================================
# DATE CHUNK GENERATOR
# ================================================================
def generate_chunks(start, end, chunk_days):
    """
    Yields (chunk_start, chunk_end) datetime pairs covering start→end
    in chunks of chunk_days days.
    """
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


# ================================================================
# MT5 DATA FETCHER
# ================================================================
def fetch_mt5_data(tf_constant, symbol, date_from, date_to):
    """
    Fetches OHLCV data from MT5 for a given timeframe and date range.
    Returns a DataFrame or None on failure.
    """
    rates = mt5.copy_rates_range(symbol, tf_constant, date_from, date_to)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df.rename(columns={
        'time':      'datetime',
        'open':      'open',
        'high':      'high',
        'low':       'low',
        'close':     'close',
        'tick_volume': 'volume',
        'spread':    'spread',
        'real_volume': 'real_volume',
    }, errors='ignore', inplace=True)

    df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert(NY_TZ)
    df.set_index('datetime', inplace=True)

    # Keep only the columns the backtest engine needs
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df[cols]


# ================================================================
# CSV WRITER — appends to monthly files
# ================================================================
def get_csv_path(tf_label, year, month):
    filename = f"{SYMBOL}_{tf_label}_{year:04d}-{month:02d}.csv"
    return os.path.join(DATA_DIR, filename)


def save_chunk_to_csv(df, tf_label):
    """
    Splits a DataFrame by month and appends each month's data
    to its corresponding CSV file (creates if not exists).
    Deduplicates on datetime index before saving.
    """
    if df is None or df.empty:
        return 0

    rows_saved = 0
    for (year, month), group in df.groupby([df.index.year, df.index.month]):
        csv_path = get_csv_path(tf_label, year, month)

        if os.path.exists(csv_path):
            # Load existing, merge, deduplicate
            existing = pd.read_csv(csv_path, index_col='datetime', parse_dates=True)
            existing.index = pd.to_datetime(existing.index, utc=True).tz_convert(NY_TZ)
            combined = pd.concat([existing, group])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            combined.to_csv(csv_path)
        else:
            group.to_csv(csv_path)

        rows_saved += len(group)

    return rows_saved


# ================================================================
# MAIN DOWNLOAD LOOP
# ================================================================
def run_download():
    print("=" * 60)
    print("  Antigravity Bridge — Historical Data Downloader")
    print("=" * 60)
    print(f"  Symbol   : {SYMBOL}")
    print(f"  Range    : {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Output   : {DATA_DIR}")
    print("=" * 60)

    # Connect to MT5
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
        print("Make sure MT5 is open and logged in before running this.")
        return False

    print(f"[MT5] Connected. Terminal: {mt5.terminal_info().name}")
    print(f"[MT5] Account: {mt5.account_info().login}\n")

    progress = load_progress()
    total_rows_downloaded = 0

    for tf_constant, tf_label, chunk_days in TIMEFRAMES:
        print(f"\n── Downloading {tf_label} data ──────────────────────────")

        chunks = list(generate_chunks(START_DATE, END_DATE, chunk_days))
        total_chunks = len(chunks)

        completed = len(progress.get(tf_label, {}).get("completed_chunks", []))
        print(f"   Total chunks : {total_chunks}")
        print(f"   Already done : {completed}")
        print(f"   Remaining    : {total_chunks - completed}\n")

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_key = f"{chunk_start.date()}_{chunk_end.date()}"

            if is_chunk_done(progress, tf_label, chunk_key):
                continue

            print(f"   [{i+1:03d}/{total_chunks}] {chunk_start.date()} → {chunk_end.date()} ... ",
                  end='', flush=True)

            df = fetch_mt5_data(tf_constant, SYMBOL, chunk_start, chunk_end)

            if df is None or df.empty:
                print("NO DATA (weekend/holiday gap — skipping)")
                # Still mark as done so we don't retry empty ranges forever
                mark_chunk_done(progress, tf_label, chunk_key)
                continue

            rows = save_chunk_to_csv(df, tf_label)
            total_rows_downloaded += rows
            mark_chunk_done(progress, tf_label, chunk_key)

            print(f"{rows:,} rows saved")
            time.sleep(CHUNK_PAUSE_SECONDS)

        print(f"\n   ✅ {tf_label} complete.")

    mt5.shutdown()

    print("\n" + "=" * 60)
    print(f"  Download complete.")
    print(f"  Total rows saved : {total_rows_downloaded:,}")
    print(f"  Data location    : {DATA_DIR}")
    print(f"  Progress file    : {PROGRESS_FILE}")
    print("=" * 60)

    _print_data_summary()
    return True


# ================================================================
# DATA SUMMARY — shows what we have after download
# ================================================================
def _print_data_summary():
    print("\n── Data Summary ─────────────────────────────────────────")
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

    if not csv_files:
        print("  No CSV files found.")
        return

    by_tf = {}
    for f in csv_files:
        parts = f.replace('.csv', '').split('_')
        tf = parts[1]
        if tf not in by_tf:
            by_tf[tf] = []
        by_tf[tf].append(f)

    for tf, files in sorted(by_tf.items()):
        total_rows = 0
        for fname in files:
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, fname))
                total_rows += len(df)
            except Exception:
                pass
        date_range = f"{files[0].split('_')[2]} → {files[-1].split('_')[2]}"
        print(f"  {tf:4s} : {len(files):3d} files | {total_rows:>10,} rows | {date_range}")

    print("")


# ================================================================
# UTILITY — load data for a given TF and date range
# Called by backtest_engine.py
# ================================================================
def load_data(tf_label, date_from=None, date_to=None):
    """
    Loads all CSV files for a given timeframe label (e.g. 'M5', 'H1')
    and returns a single sorted DataFrame.

    Optionally filter by date_from / date_to (datetime objects).

    Returns:
        pd.DataFrame with DatetimeIndex (NY timezone), columns: open/high/low/close/volume
        Empty DataFrame if no data found.
    """
    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"{SYMBOL}_{tf_label}_") and f.endswith('.csv')
    ])

    if not csv_files:
        print(f"[DataLoader] No {tf_label} CSV files found in {DATA_DIR}")
        return pd.DataFrame()

    frames = []
    for fname in csv_files:
        try:
            df = pd.read_csv(
                os.path.join(DATA_DIR, fname),
                index_col='datetime',
                parse_dates=True
            )
            frames.append(df)
        except Exception as e:
            print(f"[DataLoader] Warning: could not load {fname}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined.index = pd.to_datetime(combined.index, utc=True)

    # Convert to NY timezone
    if combined.index.tzinfo is None:
        combined.index = combined.index.tz_localize('UTC')
    combined.index = combined.index.tz_convert(NY_TZ)

    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)

    # Apply date filters
    if date_from:
        if date_from.tzinfo is None:
            date_from = NY_TZ.localize(date_from)
        combined = combined[combined.index >= date_from]

    if date_to:
        if date_to.tzinfo is None:
            date_to = NY_TZ.localize(date_to)
        combined = combined[combined.index <= date_to]

    return combined


def get_available_date_range(tf_label="M5"):
    """
    Returns (earliest_date, latest_date) available in the CSV data
    for a given timeframe. Useful for backtest_engine to know its range.
    """
    df = load_data(tf_label)
    if df.empty:
        return None, None
    return df.index[0], df.index[-1]


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    success = run_download()
    if not success:
        sys.exit(1)
