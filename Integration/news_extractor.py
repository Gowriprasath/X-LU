"""
news_extractor.py — Live USD News Feed with Daily Cache
=========================================================

WHAT CHANGED
─────────────
Previously: Called the ForexFactory JSON API on EVERY 5-minute cycle.
  - ~288 API calls per day for a static dataset that doesn't change
  - If the API was down at 08:25 (5 min before NFP), the gate went inactive
    right at the worst possible moment

Now: Fetches once per day at 06:30 NY, caches in memory.
  - Every subsequent cycle reads from cache — zero network cost
  - Daemon thread started by main_bot.py at boot
  - If 06:30 fetch fails → retries every 10 min until 08:00 NY
    (1.5 hour window before NY session opens)
  - If all retries fail → keeps yesterday's cache, warns loudly

PUBLIC API (unchanged — all callers work without modification)
──────────────────────────────────────────────────────────────
    get_usd_news_today()       → str (formatted for Claude prompt)
    start_daily_news_daemon()  → starts background refresh thread
    force_refresh()            → manual refresh now
    get_cache_status()         → dict: cache age, last fetch, etc.
"""

import os, sys, time, threading, pytz, dateutil.parser
from datetime import datetime, date, timedelta

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

NY_TZ         = pytz.timezone('America/New_York')
_NEWS_URL     = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_MAX_RETRIES  = 3
_RETRY_DELAYS = (2, 5, 10)

# Daily refresh config
_REFRESH_HOUR_NY   = 6
_REFRESH_MINUTE_NY = 30
_RETRY_INTERVAL_S  = 600   # 10 min between retries after 06:30 failure
_RETRY_UNTIL_HOUR  = 8     # stop retrying at 08:00 NY

# Module-level cache
_cache_lock      = threading.Lock()
_cached_news_str = ""
_cached_date     = None
_last_fetch_time = None
_fetch_failed    = False
_daemon_started  = False


def _fetch_from_api():
    """Raw fetch — returns formatted str on success, None on failure."""
    import requests
    last_error = None
    events     = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(_NEWS_URL, timeout=10)
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}"
                time.sleep(_RETRY_DELAYS[attempt] if attempt < len(_RETRY_DELAYS) else 10)
                continue
            events = r.json()
            break
        except Exception as e:
            last_error = str(e)
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAYS[attempt] if attempt < len(_RETRY_DELAYS) else 10
                print(f"[NewsExtractor] Attempt {attempt+1}/{_MAX_RETRIES} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    if events is None:
        print(f"[NewsExtractor] API fetch failed. Last error: {last_error}")
        return None

    today_ny  = datetime.now(NY_TZ).date()
    news_list = []
    for event in events:
        if event.get('country') != 'USD':
            continue
        impact = event.get('impact')
        if impact not in ['High', 'Medium']:
            continue
        try:
            dt_ny = dateutil.parser.isoparse(event.get('date', '')).astimezone(NY_TZ)
        except Exception:
            continue
        if dt_ny.date() != today_ny:
            continue
        time_str    = dt_ny.strftime("%I:%M %p (NY Time)")
        impact_flag = "🚨 HIGH IMPACT" if impact == 'High' else "⚠️ MEDIUM IMPACT"
        news_list.append(f"- {time_str}: {event.get('title', 'Unknown Event')} [{impact_flag}]")

    return "\n".join(news_list) if news_list else "No High/Medium impact USD news scheduled for today."


def _update_cache(news_str):
    global _cached_news_str, _cached_date, _last_fetch_time, _fetch_failed
    with _cache_lock:
        _cached_news_str = news_str
        _cached_date     = datetime.now(NY_TZ).date()
        _last_fetch_time = datetime.now(NY_TZ)
        _fetch_failed    = False
    print(f"[NewsExtractor] ✅ Daily news cached for {_cached_date} "
          f"at {_last_fetch_time.strftime('%H:%M')} NY")


def _daily_refresh_loop():
    global _fetch_failed
    print(f"[NewsExtractor] 📅 Daily news refresh daemon started. "
          f"Fetches at {_REFRESH_HOUR_NY:02d}:{_REFRESH_MINUTE_NY:02d} NY every morning.")
    
    def _is_shutting_down():
        try:
            import main_bot as _mb
            return _mb._shutdown_event.is_set()
        except Exception:
            return False

    def _sleep_with_shutdown_check(seconds):
        slept = 0
        while slept < seconds and not _is_shutting_down():
            time.sleep(min(5, seconds - slept))
            slept += 5
        return not _is_shutting_down()

    while not _is_shutting_down():
        now_ny = datetime.now(NY_TZ)
        target = now_ny.replace(hour=_REFRESH_HOUR_NY, minute=_REFRESH_MINUTE_NY,
                                second=0, microsecond=0)
        if now_ny >= target:
            target += timedelta(days=1)
        sleep_s = (target - now_ny).total_seconds()
        print(f"[NewsExtractor] Next refresh: {target.strftime('%Y-%m-%d %H:%M')} NY "
              f"(in {sleep_s/3600:.1f}h)")
        
        # Sleep until 06:30 NY daily, checking shutdown event every 5 seconds
        if not _sleep_with_shutdown_check(sleep_s):
            break

        # It's 06:30 — fetch with retry window until 08:00
        retry_cutoff = datetime.now(NY_TZ).replace(
            hour=_RETRY_UNTIL_HOUR, minute=0, second=0)
        fetched  = False
        attempt  = 0
        while not fetched and not _is_shutting_down():
            print(f"[NewsExtractor] 🔄 Daily fetch attempt {attempt+1} "
                  f"at {datetime.now(NY_TZ).strftime('%H:%M')} NY...")
            result = _fetch_from_api()
            if result is not None:
                _update_cache(result)
                fetched = True
            else:
                now_ny = datetime.now(NY_TZ)
                if now_ny >= retry_cutoff:
                    with _cache_lock:
                        _fetch_failed = True
                        stale = _cached_date
                    print(f"\n{'⚠️  ' * 8}\n"
                          f"[NewsExtractor] WARNING: All retries failed before 08:00 NY.\n"
                          f"  Running on {'stale data from ' + str(stale) if stale else 'NO data'}.\n"
                          f"  The bot may trade through high-impact events today.\n"
                          f"{'⚠️  ' * 8}\n")
                    break
                remaining = (retry_cutoff - now_ny).total_seconds() / 60
                print(f"[NewsExtractor] Retry in {_RETRY_INTERVAL_S//60} min "
                      f"({remaining:.0f} min until cutoff)...")
                
                # Sleep between retries with shutdown checks
                if not _sleep_with_shutdown_check(_RETRY_INTERVAL_S):
                    break
                attempt += 1

    print("[NewsExtractor] Shutdown signal received. Exiting loop cleanly.")


def start_daily_news_daemon():
    """
    Call once at bot startup (main_bot.py __main__ block).
    1. Fetches news immediately so the first cycle isn't blind.
    2. Starts background thread to refresh at 06:30 NY every morning.
    """
    global _daemon_started
    if _daemon_started:
        return
    print("[NewsExtractor] 🔄 Startup news fetch...")
    result = _fetch_from_api()
    if result is not None:
        _update_cache(result)
    else:
        print("[NewsExtractor] ⚠ Startup fetch failed — will retry at 06:30 NY.")
    t = threading.Thread(target=_daily_refresh_loop,
                         name="NewsRefreshDaemon", daemon=True)
    t.start()
    _daemon_started = True


def get_usd_news_today() -> str:
    """
    Returns today's USD High/Medium news string.
    Reads from cache — zero network calls per cycle after daemon is running.
    Falls back to direct fetch if cache is empty or stale.
    """
    with _cache_lock:
        cached_date = _cached_date
        cached_str  = _cached_news_str
        failed      = _fetch_failed

    today_ny = datetime.now(NY_TZ).date()

    if cached_date == today_ny and cached_str:
        if failed:
            return (f"⚠️ WARNING: News refresh failed today. Data may be from {cached_date}.\n\n"
                    f"{cached_str}")
        return cached_str

    # Cache miss — direct fetch (first cycle before daemon runs, or midnight crossover)
    print("[NewsExtractor] Cache miss — fetching directly...")
    result = _fetch_from_api()
    if result is not None:
        _update_cache(result)
        return result

    if cached_str:
        print("[NewsExtractor] ⚠ Direct fetch failed — using stale cache.")
        return (f"⚠️ News Feed Unavailable. Using stale data from {cached_date}.\n\n{cached_str}")

    print(f"\n{'⚠️ ' * 10}\n"
          f"[NewsExtractor] WARNING: No news data — news gate INACTIVE.\n"
          f"{'⚠️ ' * 10}\n")
    return "⚠️ News Feed Unavailable (Connection Error — news gate inactive)."


def force_refresh() -> str:
    """Forces an immediate fresh fetch, updates cache. Returns new string."""
    print("[NewsExtractor] 🔄 Force refresh triggered...")
    result = _fetch_from_api()
    if result is not None:
        _update_cache(result)
        return result
    print("[NewsExtractor] ⚠ Force refresh failed — cache unchanged.")
    with _cache_lock:
        return _cached_news_str or "⚠️ News Feed Unavailable."


def get_cache_status() -> dict:
    """Returns cache state dict for monitoring."""
    with _cache_lock:
        cached_date = _cached_date
        last_fetch  = _last_fetch_time
        failed      = _fetch_failed
    today_ny = datetime.now(NY_TZ).date()
    age_min  = None
    if last_fetch:
        age_min = round((datetime.now(NY_TZ) - last_fetch).total_seconds() / 60, 1)
    return {
        "cached_date":     str(cached_date) if cached_date else None,
        "last_fetch_time": last_fetch.strftime("%Y-%m-%d %H:%M NY") if last_fetch else None,
        "cache_age_min":   age_min,
        "is_today":        cached_date == today_ny,
        "fetch_failed":    failed,
        "daemon_running":  _daemon_started,
        "refresh_time_ny": f"{_REFRESH_HOUR_NY:02d}:{_REFRESH_MINUTE_NY:02d}",
    }
