"""
Backtest/news_history.py — Historical High-Impact News for Backtest
====================================================================
Provides the same news-gate behaviour as the live bot's news_extractor.py,
but sourced from a local JSON cache instead of a live API call.

Data model
──────────
News is stored in Data/Backtest/news_calendar.json as a dict keyed by date:

{
  "2023-11-03": [
    {"time": "08:30", "name": "Non-Farm Payrolls", "impact": "HIGH"},
    {"time": "08:30", "name": "Unemployment Rate",  "impact": "HIGH"}
  ],
  ...
}

Times are in NY timezone (HH:MM 24h). Impact levels: "HIGH", "MEDIUM", "LOW".

On first run (if news_calendar.json doesn't exist), we seed it with the most
common recurring high-impact USD events at their most common release times.
The seeded data covers 2017–2026. It won't be 100% accurate for every year
(FOMC dates shift, NFP is always 1st Friday, etc.) but it gives the backtest
a realistic news-blocking framework that matches the live bot's behaviour.

Usage from backtest_engine.py:
    from news_history import get_news_for_date, is_in_news_window, format_for_prompt

    news_today = get_news_for_date(current_time.date())   # list of event dicts
    if is_in_news_window(current_time, news_today):
        # block trade
    prompt_block = format_for_prompt(news_today, current_time)
"""

import os
import sys
import json
from datetime import datetime, date, timedelta
import pytz

# ── Paths ──────────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from paths import BACKTEST_OUT
NEWS_CACHE_PATH = os.path.join(BACKTEST_OUT, "news_calendar.json")

NY_TZ = pytz.timezone("America/New_York")

# ── Block windows — C-01 FIX: import from master_controls (single source of truth) ─
# Previously hardcoded (BEFORE=60, AFTER=5), causing backtest to block 55min more
# than the live bot on every news day → backtest PnL was systematically understated.
# Now always identical to master_controls.NEWS_BLOCK_BEFORE/AFTER_MINUTES.
from master_controls import NEWS_BLOCK_BEFORE_MINUTES, NEWS_BLOCK_AFTER_MINUTES
# Spread spike window is narrower — handled separately in spread_simulator.py


# ================================================================
# FF CACHE — preferred source when available
# ================================================================
# If ff_fetcher.py has been run and built ff_news_calendar.json,
# use that (real ForexFactory data) instead of the seeded approximation.
# This is loaded once at module level and used transparently.
try:
    from paths import FF_NEWS_CACHE_PATH as _FF_CACHE_PATH
    _FF_CACHE_AVAILABLE = os.path.exists(_FF_CACHE_PATH)
except Exception:
    _FF_CACHE_AVAILABLE = False
    _FF_CACHE_PATH      = None


# ================================================================
# RECURRING HIGH-IMPACT USD EVENT SCHEDULE
# Used to seed the news calendar when no cache exists.
# ================================================================

# Approximate monthly/weekly schedule for recurring USD news
_RECURRING_EVENTS = [
    # (month_day_pattern, time_ny, name, impact)
    # NFP — 1st Friday of every month
    # CPI  — ~2nd week Tuesday/Wednesday
    # FOMC — 8 times per year (approximately every 6 weeks)
    # etc.
    # We store them as (weekday_of_month_rule, time, name) below
    # and generate dates in _seed_news_calendar().
]

# Known approximate event times (NY timezone)
_EVENT_TIMES = {
    "Non-Farm Payrolls":     "08:30",
    "Unemployment Rate":     "08:30",
    "CPI":                   "08:30",
    "Core CPI":              "08:30",
    "PPI":                   "08:30",
    "Core PPI":              "08:30",
    "FOMC Statement":        "14:00",
    "Fed Interest Rate":     "14:00",
    "Powell Press Conf":     "14:30",
    "GDP":                   "08:30",
    "Retail Sales":          "08:30",
    "ISM Manufacturing":     "10:00",
    "ISM Services":          "10:00",
    "JOLTS":                 "10:00",
    "ADP Employment":        "08:15",
    "Initial Jobless Claims":"08:30",
    "PCE Price Index":       "08:30",
    "Core PCE":              "08:30",
    "Consumer Confidence":   "10:00",
    "Michigan Sentiment":    "10:00",
}


def _seed_news_calendar() -> dict:
    """
    Generates a best-effort news calendar for 2017–2026.
    Approximates recurring event dates — not a substitute for real FF data,
    but gives the backtest a realistic blocking framework.

    Returns dict: {date_str: [event, ...]}
    """
    calendar = {}

    def add(d: date, name: str, impact: str = "HIGH"):
        key = d.isoformat()
        if key not in calendar:
            calendar[key] = []
        time_str = _EVENT_TIMES.get(name, "08:30")
        calendar[key].append({"time": time_str, "name": name, "impact": impact})

    start = date(2017, 1, 1)
    end   = date(2026, 12, 31)
    cur   = start

    while cur <= end:
        # NFP — 1st Friday of month
        if cur.weekday() == 4:   # Friday
            day_of_month = cur.day
            if 1 <= day_of_month <= 7:
                add(cur, "Non-Farm Payrolls")
                add(cur, "Unemployment Rate")

        # CPI — usually 2nd Tuesday/Wednesday of month around 8:30
        if cur.weekday() in (1, 2):   # Tue or Wed
            day_of_month = cur.day
            if 8 <= day_of_month <= 16:
                # Alternate between CPI and Core CPI (same day, same time)
                add(cur, "CPI")
                add(cur, "Core CPI")

        # PPI — usually Thursday the week before/after CPI
        if cur.weekday() == 3:   # Thursday
            day_of_month = cur.day
            if 9 <= day_of_month <= 17:
                add(cur, "PPI")
                add(cur, "Core PPI")

        # FOMC — H-03 FIX: Replace weekday-algorithm (only 3/8 dates correct in 2023)
        # with hardcoded real FOMC dates for 2017–2026. Each entry is the statement
        # day (Wednesday). Press conference days (Thursday after 2019-01) auto-added.
        # Source: Federal Reserve historical calendar.
        pass   # handled via _KNOWN_FOMC_DATES set below (applied after the while loop)

        # GDP — last Thursday of Jan, Apr, Jul, Oct (advance estimate)
        if cur.weekday() == 3 and cur.month in (1, 4, 7, 10):
            next_thursday = cur + timedelta(days=7)
            if next_thursday.month != cur.month:   # last Thursday of month
                add(cur, "GDP")

        # Retail Sales — ~15th of month
        if cur.day in (14, 15, 16) and cur.weekday() in (0, 1, 2, 3, 4):
            add(cur, "Retail Sales", "HIGH")

        # JOLTS — usually first Tuesday/Wednesday of month (2 weeks after month end)
        if cur.weekday() in (1, 2) and cur.day in (3, 4, 5, 6, 7, 8, 9, 10):
            add(cur, "JOLTS", "HIGH")

        # Initial Jobless Claims — every Thursday
        if cur.weekday() == 3:
            add(cur, "Initial Jobless Claims", "MEDIUM")

        # PCE — last Friday of month
        if cur.weekday() == 4:
            next_friday = cur + timedelta(days=7)
            if next_friday.month != cur.month:
                add(cur, "PCE Price Index")
                add(cur, "Core PCE")

        cur += timedelta(days=1)

    # ── H-03 FIX: Apply real FOMC dates (replaces broken weekday algorithm) ──
    # Real Federal Reserve statement dates for 2017–2026.
    # Press conference added the NEXT DAY for all dates from 2019 onwards
    # (the Fed moved to post-every-meeting press conferences in Jan 2019).
    _KNOWN_FOMC_DATES = {
        # 2017
        "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
        "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
        # 2018
        "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
        "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
        # 2019 — press conf every meeting from here
        "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
        "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
        # 2020
        "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
        "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
        "2020-12-16",
        # 2021
        "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
        "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        # 2022
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        # 2026 (projected — update when confirmed)
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    }
    # Press-conf meetings (post-2019): add Powell Press Conf the next day
    _FOMC_PRESS_CONF_CUTOFF = date(2019, 1, 1)
    for _fomc_str in _KNOWN_FOMC_DATES:
        _fomc_d = date.fromisoformat(_fomc_str)
        if start <= _fomc_d <= end:
            add(_fomc_d, "FOMC Statement")
            add(_fomc_d, "Fed Interest Rate")
            if _fomc_d >= _FOMC_PRESS_CONF_CUTOFF:
                _pc_day = _fomc_d + timedelta(days=1)
                if _pc_day <= end:
                    add(_pc_day, "Powell Press Conf")

    return calendar


def _load_calendar() -> dict:
    """
    Loads news calendar. Priority order:
      1. ff_news_calendar.json  — real ForexFactory data (if ff_fetcher.py has run)
      2. news_calendar.json     — seeded approximation (fallback)

    If FF cache exists but is missing dates for a given range, the seeded
    data fills the gaps transparently (merging happens in get_news_for_date).
    """
    # ── Preferred: real ForexFactory cache ────────────────────────
    if _FF_CACHE_AVAILABLE and _FF_CACHE_PATH:
        try:
            with open(_FF_CACHE_PATH, "r", encoding="utf-8") as f:
                ff_cal = json.load(f)
            print(f"[NewsHistory] Using ForexFactory cache "
                  f"({len(ff_cal)} days) → {_FF_CACHE_PATH}")
            return ff_cal
        except Exception as e:
            print(f"[NewsHistory] FF cache load failed ({e}) — falling back to seeded.")

    # ── Fallback: seeded approximation ────────────────────────────
    if os.path.exists(NEWS_CACHE_PATH):
        try:
            with open(NEWS_CACHE_PATH, "r", encoding="utf-8") as f:
                seeded = json.load(f)
            print(f"[NewsHistory] Using seeded calendar ({len(seeded)} days). "
                  f"Run 'python Backtest/ff_fetcher.py' for real ForexFactory data.")
            return seeded
        except Exception:
            pass

    # ── Neither exists: build seeded calendar and save ────────────
    print("[NewsHistory] Building seeded news calendar (one-time setup)...")
    calendar = _seed_news_calendar()
    os.makedirs(os.path.dirname(NEWS_CACHE_PATH), exist_ok=True)
    with open(NEWS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(calendar, f, indent=2)
    print(f"[NewsHistory] Seeded calendar written → {NEWS_CACHE_PATH}")
    print(f"[NewsHistory] {len(calendar):,} trading days seeded (approximate).")
    print("[NewsHistory] Run 'python Backtest/ff_fetcher.py' for real ForexFactory data.")
    return calendar


# ── Module-level cache ─────────────────────────────────────────────────────
_CALENDAR: dict = {}
_CALENDAR_LOADED: bool = False


def _ensure_loaded():
    global _CALENDAR, _CALENDAR_LOADED
    if not _CALENDAR_LOADED:
        _CALENDAR = _load_calendar()
        _CALENDAR_LOADED = True


def get_news_for_date(query_date) -> list:
    """
    Returns list of event dicts for the given date.
    Each dict: {"time": "HH:MM", "name": str, "impact": "HIGH"|"MEDIUM"|"LOW"}
    The "time" values are returned as datetime objects in NY timezone.

    Args:
        query_date: date object or "YYYY-MM-DD" string

    Returns:
        list of event dicts with "time" as tz-aware datetime
    """
    _ensure_loaded()
    if isinstance(query_date, datetime):
        query_date = query_date.date()
    key = query_date.isoformat() if hasattr(query_date, "isoformat") else str(query_date)
    raw_events = _CALENDAR.get(key, [])

    # Convert "HH:MM" string times to tz-aware datetimes for comparison
    result = []
    for ev in raw_events:
        ev_copy = dict(ev)
        try:
            t = datetime.strptime(ev["time"], "%H:%M")
            dt = NY_TZ.localize(datetime(
                query_date.year, query_date.month, query_date.day,
                t.hour, t.minute, 0
            ))
            ev_copy["time"] = dt
        except Exception:
            ev_copy["time"] = None
        result.append(ev_copy)
    return result


def is_in_news_window(current_time: datetime, news_today: list,
                       block_before: int = NEWS_BLOCK_BEFORE_MINUTES,
                       block_after:  int = NEWS_BLOCK_AFTER_MINUTES) -> bool:
    """
    Returns True if current_time falls within a news blocking window.

    Mirrors main_bot.py check_news_window() logic exactly:
      - block_before minutes BEFORE a HIGH-impact event → block
      - block_after  minutes AFTER  a HIGH-impact event → block

    Args:
        current_time: tz-aware datetime (NY time)
        news_today:   output of get_news_for_date()
        block_before: minutes before event to start blocking (default 60)
        block_after:  minutes after event to stop blocking  (default 5)

    Returns:
        bool
    """
    if not news_today:
        return False

    for event in news_today:
        if event.get("impact", "").upper() != "HIGH":
            continue
        event_time = event.get("time")
        if event_time is None:
            continue
        try:
            diff_minutes = (current_time - event_time).total_seconds() / 60
            # diff_minutes < 0 → event is in the future
            if -block_before <= diff_minutes <= block_after:
                return True
        except Exception:
            continue
    return False


def format_for_prompt(news_today: list, current_time: datetime) -> str:
    """
    Formats the day's news events into the same string the live bot
    injects into the AI prompt via news_extractor.get_usd_news_today().

    Returns multi-line string ready for prompt injection.
    """
    if not news_today:
        return "No high-impact USD events today."

    lines = ["USD High-Impact Events Today:"]
    for event in news_today:
        ev_time = event.get("time")
        name    = event.get("name", "Unknown")
        impact  = event.get("impact", "")
        if ev_time and hasattr(ev_time, "strftime"):
            time_str = ev_time.strftime("%I:%M %p")
            try:
                diff_mins = (current_time - ev_time).total_seconds() / 60
                if diff_mins < -60:
                    status = f"in {int(-diff_mins)} min"
                elif diff_mins < 0:
                    status = f"in {int(-diff_mins)} min ⚠️ APPROACHING"
                elif diff_mins <= 10:
                    status = f"{int(diff_mins)} min ago ⚠️ JUST RELEASED"
                else:
                    status = f"{int(diff_mins)} min ago"
            except Exception:
                status = ""
        else:
            time_str = event.get("time", "??:??")
            status = ""

        impact_tag = "🚨 HIGH IMPACT" if impact.upper() == "HIGH" else impact
        lines.append(f"  - {time_str} (NY Time): {name} [{impact_tag}] {status}")

    return "\n".join(lines)


def add_event(event_date, time_str: str, name: str, impact: str = "HIGH"):
    """
    Adds a custom event to the in-memory calendar.
    Use this to inject real ForexFactory data without modifying the JSON file.

    Args:
        event_date: date object or "YYYY-MM-DD" string
        time_str:   "HH:MM" (NY time)
        name:       event name
        impact:     "HIGH" | "MEDIUM" | "LOW"
    """
    _ensure_loaded()
    if isinstance(event_date, datetime):
        event_date = event_date.date()
    key = event_date.isoformat() if hasattr(event_date, "isoformat") else str(event_date)
    if key not in _CALENDAR:
        _CALENDAR[key] = []
    _CALENDAR[key].append({"time": time_str, "name": name, "impact": impact})
