"""
ff_fetcher.py — MT5 Economic Calendar Builder (HIGH IMPACT / RED ONLY)
=======================================================================

WHY THIS EXISTS
────────────────
Replaces the old ForexFactory web scraper with a direct, blazing-fast 
connection to MetaTrader 5's internal economic calendar database.
This completely bypasses Cloudflare, lazy-loading issues, and rate limits.

HOW IT WORKS
─────────────
1. Connects to the local MT5 Terminal.
2. Fetches all historical USD calendar events.
3. Filters strictly for Importance = 3 (High Impact / Red Folders).
4. Converts UTC timestamps to New York time to match your backtest engine.
5. Saves to the exact same ff_news_calendar.json format as before.
"""

import os
import sys
import json
import argparse
from datetime import datetime, date

import pytz
import MetaTrader5 as mt5

# ── Path setup ─────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from paths import FF_NEWS_CACHE_PATH, FF_FETCH_META_PATH, create_all_dirs
create_all_dirs()

NY_TZ = pytz.timezone("America/New_York")
UTC_TZ = pytz.utc

# ── Configuration ──────────────────────────────────────────────────
DEFAULT_FROM = date(2017, 1, 1)
DEFAULT_TO   = date.today()

# ================================================================
# CALENDAR MANAGEMENT
# ================================================================

def _save_cache(calendar: dict):
    os.makedirs(os.path.dirname(FF_NEWS_CACHE_PATH), exist_ok=True)
    with open(FF_NEWS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(calendar, f, indent=2, sort_keys=True)

def _save_meta(meta: dict):
    os.makedirs(os.path.dirname(FF_FETCH_META_PATH), exist_ok=True)
    with open(FF_FETCH_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ================================================================
# MAIN FETCH ENGINE (MT5)
# ================================================================

def fetch_mt5_calendar(from_date: date = DEFAULT_FROM, to_date: date = DEFAULT_TO):
    print(f"\n{'='*65}\n  MT5 Economic Calendar Fetcher (HIGH IMPACT ONLY)\n{'='*65}")
    print(f"  Range: {from_date} → {to_date}")
    
    # 1. Initialize MT5 Connection
    if not mt5.initialize():
        print("  ❌ Failed to initialize MetaTrader 5.")
        print("  Please ensure the MT5 terminal is open and installed on this machine.")
        return
    
    print("  ✅ Connected to MetaTrader 5 Terminal.")

    # 2. Fetch all USD Event Definitions
    # This gets the names and impact levels (0=None, 1=Low, 2=Medium, 3=High)
    # FIX: calendar_get was added in MetaTrader5 >= 5.0.37. Older versions raise
    # AttributeError. Check before calling and fall back gracefully so the rest
    # of the pipeline can continue with whatever cached data already exists.
    if not hasattr(mt5, 'calendar_get'):
        print("  ⚠  mt5.calendar_get is not available in your MetaTrader5 package.")
        print("  ℹ  Upgrade with: pip install --upgrade MetaTrader5")
        print("  ℹ  Continuing with existing cached news data.")
        mt5.shutdown()
        return

    events = mt5.calendar_get(currency="USD")
    if not events:
        print("  ❌ Failed to retrieve calendar events from MT5.")
        mt5.shutdown()
        return

    # Filter for strictly High Impact (Importance == 3) and map by Event ID
    high_impact_events = {}
    for ev in events:
        if ev.importance == 3:
            high_impact_events[ev.id] = ev.name

    print(f"  ✅ Found {len(high_impact_events)} unique High-Impact USD event types.")

    # 3. Fetch Historical Scheduled Values (The actual dates/times)
    # Convert dates to datetime objects for MT5 API
    dt_from = datetime(from_date.year, from_date.month, from_date.day)
    dt_to = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)

    print("  ⏳ Downloading historical timestamps from MT5 database... (This is instant)")
    values = mt5.calendar_value_history(dt_from, dt_to, currency="USD")
    
    if not values:
        print("  ❌ No calendar values found for this date range.")
        mt5.shutdown()
        return

    # 4. Process and Format Data
    calendar = {}
    total_events = 0

    for val in values:
        # Only process if this specific timestamp belongs to a High Impact event
        if val.event_id in high_impact_events:
            
            # MT5 stores time in UTC (seconds since epoch). 
            # We MUST convert to NY Time to match your backtest engine's expectations.
            utc_dt = datetime.fromtimestamp(val.time, tz=UTC_TZ)
            ny_dt = utc_dt.astimezone(NY_TZ)
            
            d_key = ny_dt.strftime("%Y-%m-%d")
            t_str = ny_dt.strftime("%H:%M")
            name = high_impact_events[val.event_id]

            if d_key not in calendar:
                calendar[d_key] = []
                
            entry = {"time": t_str, "name": name, "impact": "High"}
            
            # Prevent duplicates
            if entry not in calendar[d_key]:
                calendar[d_key].append(entry)
                total_events += 1

    # 5. Save outputs exactly where the old scraper did
    _save_cache(calendar)
    
    meta = {
        "source": "MetaTrader 5 API",
        "last_run": datetime.now().isoformat(),
        "total_days_cached": len(calendar),
        "total_events_cached": total_events
    }
    _save_meta(meta)

    mt5.shutdown()

    print(f"\n{'─'*65}")
    print(f"  ✅ Extraction Complete!")
    print(f"  Total Red Folders  : {total_events} events")
    print(f"  Total Active Days  : {len(calendar)} days with news")
    print(f"  Saved JSON to      : {FF_NEWS_CACHE_PATH}")
    print(f"{'─'*65}\n")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MT5 historical news calendar for backtesting.")
    parser.add_argument("--from", dest="from_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    from_d = date.fromisoformat(args.from_date) if args.from_date else DEFAULT_FROM
    to_d   = date.fromisoformat(args.to_date)   if args.to_date   else DEFAULT_TO

    fetch_mt5_calendar(from_d, to_d)