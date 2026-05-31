"""
health_agent.py — Daily Self-Healing System Health Agent
==========================================================

Runs once per day at 06:00 NY (30 minutes before news refresh,
1 hour before London→NY overlap). Checks every subsystem in the
bot, produces a structured report, and uses Claude to interpret
findings and suggest actions.

WHAT IT CHECKS
──────────────
 1. Model files          — all 9 required files exist + freshness
 2. Reversal detector    — separate binary model present + age
 3. API connectivity     — Claude keys reachable, rotation healthy
 4. MT5 connectivity     — MT5 terminal live, XAUUSD visible
 5. News feed            — cache age, FF calendar coverage
 6. Training data        — features.csv + labels.csv exist, non-empty
 7. Drift state          — confidence drift log, reload flag status
 8. Risk state           — is bot halted? daily drawdown hit?
 9. Memory integrity     — all JSON files valid + size within caps
10. Shadow journal       — gate distribution (WAIT ratio, block ratio)
11. Post-mortem tracker  — did it run today / last run age
12. Wisdom freshness     — last rebuild date vs WISDOM_REBUILD_DAYS
13. Auto-retrainer       — last retrain date, trigger status
14. FF news calendar     — coverage for upcoming backtest range

OUTPUT
──────
 - Console: coloured summary table (✅ / ⚠️ / ❌ per check)
 - File:    Data/Logs/health_report_YYYY-MM-DD.json
 - Claude:  AI interpretation + prioritised action list (if issues found)

USAGE
──────
# As daemon (called from main_bot.py startup — recommended):
    from health_agent import start_health_daemon
    start_health_daemon()

# Manual one-shot run:
    python health_agent.py

# Verbose mode (shows all checks, not just failures):
    python health_agent.py --verbose

# Skip Claude interpretation (faster, no API cost):
    python health_agent.py --no-ai
"""

import os
import sys
import json
import time
import threading
import argparse
from datetime import datetime, date, timedelta

import pytz

# ── Paths ───────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_integration_path = os.path.join(_THIS_DIR, "Integration")
if _integration_path not in sys.path:
    sys.path.append(_integration_path)
_wisdom_path = os.path.join(_THIS_DIR, "Integration", "Wisdom_Worker")
if _wisdom_path not in sys.path:
    sys.path.append(_wisdom_path)

NY_TZ = pytz.timezone("America/New_York")

# Run health check at this time every morning
_CHECK_HOUR_NY   = 6
_CHECK_MINUTE_NY = 0

_daemon_started = False


# ================================================================
# RESULT HELPERS
# ================================================================

def _ok(label, detail=""):
    return {"status": "OK", "label": label, "detail": detail}

def _warn(label, detail=""):
    return {"status": "WARN", "label": label, "detail": detail}

def _fail(label, detail=""):
    return {"status": "FAIL", "label": label, "detail": detail}

def _days_old(path) -> float | None:
    """Returns age of file in days, or None if missing."""
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) / 86400


def _load_json(path) -> tuple:
    """Returns (dict_or_list, error_str). error_str is None on success."""
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


# ================================================================
# INDIVIDUAL CHECKS
# ================================================================

def check_model_files() -> list:
    results = []
    try:
        from paths import (HMM_PATH, HMM_SCALER_PATH, REGIME_XGB_PATH,
                           LABEL_ENCODER_PATH, MODEL_META_PATH, SESSION_PROFILES_PATH,
                           REVERSAL_DETECTOR_PATH, REVERSAL_DETECTOR_META_PATH)
    except Exception as e:
        return [_fail("Model paths", f"paths.py import error: {e}")]

    required = {
        "GMM-HMM model":       HMM_PATH,
        "HMM scaler":          HMM_SCALER_PATH,
        "XGBoost regime":      REGIME_XGB_PATH,
        "Label encoder":       LABEL_ENCODER_PATH,
        "Model meta":          MODEL_META_PATH,
        "Session profiles":    SESSION_PROFILES_PATH,
        "Reversal detector":   REVERSAL_DETECTOR_PATH,
        "Reversal meta":       REVERSAL_DETECTOR_META_PATH,
    }

    for name, path in required.items():
        age = _days_old(path)
        if age is None:
            results.append(_fail(name, "FILE MISSING — run trainer.py"))
        elif age > 90:
            results.append(_warn(name, f"{age:.0f} days old — consider retraining"))
        else:
            results.append(_ok(name, f"{age:.0f} days old"))

    # Check model_meta.json for training details
    meta, err = _load_json(MODEL_META_PATH)
    if meta and not err:
        trained_on = meta.get("trained_on", "unknown")
        n_states   = meta.get("n_states", "?")
        results.append(_ok("Model meta content",
                           f"n_states={n_states}, trained={trained_on}"))
    elif err and os.path.exists(MODEL_META_PATH):
        results.append(_fail("Model meta content", f"JSON corrupt: {err}"))

    return results


def check_api_connectivity() -> list:
    results = []
    try:
        import anthropic
        from ai_client import _load_keys, AI_MODEL
    except ImportError as e:
        return [_fail("Claude SDK", f"anthropic not installed: {e}")]

    # Check keys exist
    try:
        keys = _load_keys()
        results.append(_ok("API keys", f"{len(keys)} key(s) configured"))
    except EnvironmentError as e:
        return [_fail("API keys", str(e))]

    # Lightweight connectivity test — models.list is cheaper than a completion
    key = keys[0]
    try:
        client = anthropic.Anthropic(api_key=key)
        # Minimal completion to verify key works
        r = client.messages.create(
            model=AI_MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}]
        )
        results.append(_ok("Claude API reachable", f"model={AI_MODEL}"))
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg.lower():
            results.append(_fail("Claude API key 1", "INVALID KEY — check .env"))
        elif "429" in msg or "rate" in msg.lower():
            results.append(_warn("Claude API", "Rate limited — keys may be exhausted"))
        else:
            results.append(_warn("Claude API", f"Connectivity issue: {msg[:120]}"))

    return results


def check_mt5_connectivity() -> list:
    results = []
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return [_warn("MT5", "MetaTrader5 not installed (expected in live env)")]

    try:
        if not mt5.initialize():
            return [_fail("MT5 connection", "initialize() failed — is MT5 open?")]

        info = mt5.terminal_info()
        if info is None:
            results.append(_fail("MT5 terminal", "terminal_info() returned None"))
        else:
            connected = getattr(info, "connected", False)
            results.append(
                _ok("MT5 terminal", "connected") if connected
                else _fail("MT5 terminal", "NOT connected to broker")
            )

        sym = mt5.symbol_info("XAUUSD")
        if sym is None:
            results.append(_fail("XAUUSD symbol", "Not found in Market Watch"))
        else:
            results.append(_ok("XAUUSD symbol",
                               f"bid={sym.bid:.2f} ask={sym.ask:.2f}"))
        mt5.shutdown()
    except Exception as e:
        results.append(_fail("MT5", f"Error: {e}"))

    return results


def check_news_feed() -> list:
    results = []
    try:
        from news_extractor import get_cache_status
        status = get_cache_status()
        if status["daemon_running"]:
            if status["is_today"]:
                age = status["cache_age_min"] or 0
                results.append(_ok("Live news cache",
                                   f"fresh, fetched {age:.0f} min ago"))
            else:
                results.append(_warn("Live news cache",
                                     f"stale (from {status['cached_date']}) — "
                                     f"daemon will refresh at {status['refresh_time_ny']} NY"))
        else:
            results.append(_warn("Live news daemon",
                                 "Not running — start_daily_news_daemon() not called yet"))
    except Exception as e:
        results.append(_warn("Live news", f"Could not check: {e}"))

    # Check FF backtest calendar
    try:
        from paths import FF_NEWS_CACHE_PATH, FF_FETCH_META_PATH
        if os.path.exists(FF_NEWS_CACHE_PATH):
            data, err = _load_json(FF_NEWS_CACHE_PATH)
            if err:
                results.append(_fail("FF news calendar", f"JSON corrupt: {err}"))
            else:
                n_days = len(data) if data else 0
                meta, _ = _load_json(FF_FETCH_META_PATH)
                fetched_weeks = len(meta.get("fetched_weeks", [])) if meta else 0
                last_run = (meta.get("last_run") or "never") if meta else "never"
                results.append(_ok("FF news calendar",
                                   f"{n_days} days, {fetched_weeks} FF-fetched weeks, "
                                   f"last run: {last_run[:10] if last_run != 'never' else 'never'}"))
        else:
            results.append(_warn("FF news calendar",
                                 "Not built yet — run: python Backtest/ff_fetcher.py"))
    except Exception as e:
        results.append(_warn("FF news calendar", f"Check failed: {e}"))

    return results


def check_training_data() -> list:
    results = []
    try:
        from paths import FEATURES_PATH, LABELS_PATH
    except Exception as e:
        return [_fail("Training data paths", str(e))]

    for name, path in [("features.csv", FEATURES_PATH), ("labels.csv", LABELS_PATH)]:
        if not os.path.exists(path):
            results.append(_fail(name, "FILE MISSING — run trainer.py to generate"))
            continue
        try:
            size_kb = os.path.getsize(path) / 1024
            # Count rows quickly without pandas
            with open(path) as f:
                n_rows = sum(1 for _ in f) - 1   # subtract header
            age = _days_old(path)
            results.append(_ok(name, f"{n_rows:,} rows, {size_kb:.0f}KB, {age:.0f}d old"))
        except Exception as e:
            results.append(_warn(name, f"Could not read: {e}"))

    return results


def check_drift_state() -> list:
    results = []
    try:
        from paths import DRIFT_LOG_PATH, RELOAD_FLAG_PATH, RETRAIN_HISTORY_PATH
    except Exception as e:
        return [_fail("Drift paths", str(e))]

    # Reload flag
    if os.path.exists(RELOAD_FLAG_PATH):
        flag, _ = _load_json(RELOAD_FLAG_PATH)
        if flag and flag.get("reload_needed"):
            results.append(_warn("Model reload flag",
                                 "Reload pending — model retrained but not yet picked up by bot"))
        else:
            results.append(_ok("Model reload flag", "Clear"))
    else:
        results.append(_ok("Model reload flag", "Not present (normal)"))

    # Drift log
    drift, err = _load_json(DRIFT_LOG_PATH)
    if err:
        results.append(_warn("Drift log", f"Not found or corrupt: {err}"))
    else:
        if isinstance(drift, list) and drift:
            recent = drift[-20:]   # last 20 readings
            confs  = [r.get("confidence", 1.0) for r in recent
                      if isinstance(r, dict) and "confidence" in r]
            if confs:
                avg_conf = sum(confs) / len(confs)
                if avg_conf < 0.50:
                    results.append(_fail("Recent model confidence",
                                         f"avg={avg_conf:.2f} — well below threshold, "
                                         f"retraining strongly advised"))
                elif avg_conf < 0.60:
                    results.append(_warn("Recent model confidence",
                                         f"avg={avg_conf:.2f} — slightly low"))
                else:
                    results.append(_ok("Recent model confidence",
                                       f"avg={avg_conf:.2f} over last {len(confs)} readings"))

    # Retrain history
    history, _ = _load_json(RETRAIN_HISTORY_PATH)
    if history and isinstance(history, list) and history:
        last = history[-1]
        last_date = last.get("date", "unknown")
        trigger   = last.get("trigger", "unknown")
        results.append(_ok("Last retrain", f"{last_date} (trigger: {trigger})"))
    else:
        results.append(_warn("Retrain history",
                             "No retrain history — model may never have auto-retrained"))

    return results


def check_risk_state() -> list:
    results = []
    try:
        from paths import RISK_STATE_PATH
    except Exception as e:
        return [_fail("Risk state path", str(e))]

    risk, err = _load_json(RISK_STATE_PATH)
    if err:
        results.append(_ok("Risk state", "No risk state file (normal at startup)"))
        return results

    if isinstance(risk, dict):
        halted  = risk.get("halted", False)
        consec  = risk.get("consecutive_losses", 0)
        dd_hit  = risk.get("daily_drawdown_hit", False)
        dd_pct  = risk.get("daily_drawdown_pct", 0)

        if halted:
            results.append(_fail("Risk state", "BOT HALTED — manual review required"))
        elif dd_hit:
            results.append(_warn("Daily drawdown",
                                 f"Daily drawdown cap hit ({dd_pct:.1%}) — "
                                 f"bot will not trade until reset"))
        else:
            results.append(_ok("Risk state",
                               f"Active | consecutive_losses={consec} | "
                               f"daily_dd={dd_pct:.1%}"))

        if consec >= 2:
            results.append(_warn("Consecutive losses",
                                 f"{consec} in a row — 1 more triggers halt"))

    return results


def check_memory_integrity() -> list:
    results = []
    try:
        from paths import (TRADE_MEMORY_PATH, SHADOW_JOURNAL_PATH,
                           WISDOM_PATH, AI_LESSONS_PATH,
                           MISSWISH_MEMORY_PATH, CONTINUATION_MEM_PATH)
        from master_controls import (MEMORY_CAP_SHADOW, MEMORY_CAP_MISSWISH)
    except Exception as e:
        return [_fail("Memory paths", str(e))]

    checks = [
        ("trade_memory.json",        TRADE_MEMORY_PATH,     5000, None),
        ("shadow_journal.json",       SHADOW_JOURNAL_PATH,   MEMORY_CAP_SHADOW, None),
        ("wisdom.json",              WISDOM_PATH,            None, None),
        ("ai_lessons.json",          AI_LESSONS_PATH,        None, None),
        ("misswish_memory.json",     MISSWISH_MEMORY_PATH,   MEMORY_CAP_MISSWISH, None),
        ("continuation_memory.json", CONTINUATION_MEM_PATH,  None, None),
    ]

    for name, path, cap, _ in checks:
        if not os.path.exists(path):
            results.append(_warn(name, "Not yet created (normal if bot is new)"))
            continue
        data, err = _load_json(path)
        if err:
            results.append(_fail(name, f"JSON CORRUPT — {err}"))
            continue
        n = len(data) if isinstance(data, (list, dict)) else 0
        size_kb = os.path.getsize(path) / 1024
        if cap and n >= cap * 0.90:
            results.append(_warn(name,
                                 f"{n} entries ({size_kb:.0f}KB) — "
                                 f"approaching cap of {cap}"))
        else:
            results.append(_ok(name, f"{n} entries, {size_kb:.0f}KB"))

    return results


def check_shadow_journal() -> list:
    results = []
    try:
        from paths import SHADOW_JOURNAL_PATH
    except Exception:
        return [_warn("Shadow journal", "paths import failed")]

    data, err = _load_json(SHADOW_JOURNAL_PATH)
    if err or not data:
        results.append(_warn("Shadow journal", "Empty or missing — normal if bot is new"))
        return results

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = list(data.values())
    else:
        results.append(_warn("Shadow journal", "Unexpected format"))
        return results

    if not entries:
        return [_warn("Shadow journal", "No entries yet")]

    # Count gate distribution for recent 100 entries
    recent  = entries[-100:]
    n       = len(recent)
    waits   = sum(1 for e in recent
                  if isinstance(e, dict) and
                  e.get("gate_blocked_by", "").upper() in ("", "NONE", "WAIT") and
                  e.get("signal", "").upper() == "WAIT")
    blocks  = sum(1 for e in recent
                  if isinstance(e, dict) and
                  e.get("gate_blocked_by") not in (None, "", "NONE"))
    trades  = n - waits - blocks

    wait_pct  = waits  / n * 100
    block_pct = blocks / n * 100

    if wait_pct > 80:
        results.append(_warn("Shadow journal gate dist",
                             f"WAIT={wait_pct:.0f}% over last {n} — "
                             f"bot may be over-filtered"))
    else:
        results.append(_ok("Shadow journal gate dist",
                           f"last {n}: TRADE={trades} WAIT={waits} BLOCKED={blocks}"))

    return results


def check_post_mortem() -> list:
    results = []
    try:
        from paths import POST_MORTEM_TRACKER
    except Exception:
        return [_warn("Post-mortem", "paths import failed")]

    if POST_MORTEM_TRACKER.endswith(".txt"):
        try:
            with open(POST_MORTEM_TRACKER, 'r', encoding='utf-8') as _f:
                _last_run = _f.read().strip()
            tracker = {"last_run": _last_run} if _last_run else {}
            err = None
        except FileNotFoundError:
            tracker, err = {}, "File not found"
        except Exception as _e:
            tracker, err = {}, str(_e)
    else:
        tracker, err = _load_json(POST_MORTEM_TRACKER)
    if err or not tracker:
        results.append(_warn("Post-mortem", "Never run yet — runs daily at 17:00 NY"))
        return results

    last_run = tracker.get("last_run_date", tracker.get("last_run", "never"))
    today    = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    yesterday = (datetime.now(NY_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")

    if last_run == today:
        results.append(_ok("Post-mortem", f"Ran today ({today})"))
    elif last_run == yesterday:
        results.append(_ok("Post-mortem", f"Ran yesterday ({yesterday}) — will run today at 17:00 NY"))
    else:
        age = "never" if last_run == "never" else f"last: {last_run}"
        results.append(_warn("Post-mortem",
                             f"Not run recently ({age}) — check daily_post_mortem.py"))

    return results


def check_wisdom_freshness() -> list:
    results = []
    try:
        from paths import WISDOM_PATH, WISDOM_TRACKER_PATH
        from master_controls import WISDOM_REBUILD_DAYS
    except Exception as e:
        return [_warn("Wisdom", f"Import error: {e}")]

    tracker, _ = _load_json(WISDOM_TRACKER_PATH)
    wisdom, _  = _load_json(WISDOM_PATH)

    if not wisdom:
        results.append(_warn("Wisdom", "Not built yet — will build after first post-mortem"))
        return results

    n_entries = len(wisdom) if isinstance(wisdom, list) else len(wisdom.get("entries", []))

    if tracker:
        last_rebuild = tracker.get("last_rebuild", "never")
        try:
            last_dt = datetime.fromisoformat(last_rebuild)
            age_d   = (datetime.now() - last_dt).days
            if age_d > WISDOM_REBUILD_DAYS * 2:
                results.append(_warn("Wisdom freshness",
                                     f"Last rebuild {age_d}d ago "
                                     f"(threshold={WISDOM_REBUILD_DAYS}d)"))
            else:
                results.append(_ok("Wisdom",
                                   f"{n_entries} entries, rebuilt {age_d}d ago"))
        except Exception:
            results.append(_ok("Wisdom", f"{n_entries} entries"))
    else:
        results.append(_ok("Wisdom", f"{n_entries} entries (no tracker yet)"))

    return results


def check_master_controls() -> list:
    """Validates key master_controls values are sane for live trading."""
    results = []
    try:
        from master_controls import (
            NEWS_BLOCK_BEFORE_MINUTES, NEWS_BLOCK_AFTER_MINUTES,
            GATE_MIN_CONFIDENCE, RISK_BASE_PCT, RISK_DAILY_DRAWDOWN_PCT,
            GATE_MIN_RR, GATE_MAX_SPREAD_DOLLARS, RISK_MAX_CONSECUTIVE_LOSSES
        )
    except Exception as e:
        return [_fail("master_controls", f"Import error: {e}")]

    if NEWS_BLOCK_BEFORE_MINUTES < 15:
        results.append(_fail("NEWS_BLOCK_BEFORE_MINUTES",
                             f"{NEWS_BLOCK_BEFORE_MINUTES} min is dangerously low "
                             f"— minimum safe value is 15 (recommend 30)"))
    else:
        results.append(_ok("NEWS_BLOCK_BEFORE_MINUTES",
                           f"{NEWS_BLOCK_BEFORE_MINUTES} min"))

    if RISK_BASE_PCT > 0.02:
        results.append(_warn("RISK_BASE_PCT",
                             f"{RISK_BASE_PCT:.1%} — above 2% is aggressive"))
    else:
        results.append(_ok("RISK_BASE_PCT", f"{RISK_BASE_PCT:.1%}"))

    if RISK_DAILY_DRAWDOWN_PCT < RISK_BASE_PCT * 2:
        results.append(_warn("RISK_DAILY_DRAWDOWN_PCT",
                             f"{RISK_DAILY_DRAWDOWN_PCT:.1%} — only "
                             f"{RISK_DAILY_DRAWDOWN_PCT/RISK_BASE_PCT:.1f}× base risk, "
                             f"1-2 losses halts the day"))
    else:
        results.append(_ok("RISK_DAILY_DRAWDOWN_PCT",
                           f"{RISK_DAILY_DRAWDOWN_PCT:.1%}"))

    results.append(_ok("GATE_MIN_RR", f"{GATE_MIN_RR}"))
    results.append(_ok("GATE_MAX_SPREAD", f"${GATE_MAX_SPREAD_DOLLARS}"))
    results.append(_ok("GATE_MIN_CONFIDENCE", f"{GATE_MIN_CONFIDENCE:.0%}"))

    return results


# ================================================================
# REPORT BUILDER
# ================================================================

SECTIONS = [
    ("🤖 Model Files",          check_model_files),
    ("🔑 API Connectivity",     check_api_connectivity),
    ("📡 MT5 Connectivity",     check_mt5_connectivity),
    ("📰 News Feed",            check_news_feed),
    ("📊 Training Data",        check_training_data),
    ("📉 Drift State",          check_drift_state),
    ("🛡️  Risk State",           check_risk_state),
    ("🧠 Memory Integrity",     check_memory_integrity),
    ("📓 Shadow Journal",       check_shadow_journal),
    ("📋 Post-Mortem",          check_post_mortem),
    ("💡 Wisdom",               check_wisdom_freshness),
    ("⚙️  Master Controls",     check_master_controls),
]


def run_health_check(verbose: bool = False, use_ai: bool = True) -> dict:
    """
    Runs all health checks and returns a structured report dict.
    Also prints a formatted console summary.
    """
    now_ny = datetime.now(NY_TZ)
    report = {
        "timestamp":  now_ny.isoformat(),
        "date":       now_ny.strftime("%Y-%m-%d"),
        "sections":   {},
        "summary":    {"ok": 0, "warn": 0, "fail": 0},
        "ai_analysis": None,
    }

    print(f"\n{'═' * 65}")
    print(f"  🏥  GOLD AI BRIDGE — SYSTEM HEALTH REPORT")
    print(f"  {now_ny.strftime('%Y-%m-%d %H:%M')} NY")
    print(f"{'═' * 65}")

    all_issues = []

    for section_name, check_fn in SECTIONS:
        print(f"\n  {section_name}")
        print(f"  {'─' * 55}")
        try:
            results = check_fn()
        except Exception as e:
            results = [_fail(section_name, f"Check crashed: {e}")]

        report["sections"][section_name] = results

        for r in results:
            status = r["status"]
            icon   = "✅" if status == "OK" else ("⚠️ " if status == "WARN" else "❌")
            label  = r["label"]
            detail = r["detail"]

            if status == "OK":
                report["summary"]["ok"] += 1
                if verbose:
                    print(f"    {icon} {label:<35} {detail}")
            elif status == "WARN":
                report["summary"]["warn"] += 1
                print(f"    {icon} {label:<35} {detail}")
                all_issues.append({"severity": "WARN", "check": label, "detail": detail})
            else:
                report["summary"]["fail"] += 1
                print(f"    {icon} {label:<35} {detail}")
                all_issues.append({"severity": "FAIL", "check": label, "detail": detail})

    # Summary line
    s = report["summary"]
    print(f"\n{'─' * 65}")
    print(f"  SUMMARY: ✅ {s['ok']} OK  |  ⚠️  {s['warn']} WARN  |  ❌ {s['fail']} FAIL")

    if s["fail"] == 0 and s["warn"] == 0:
        print(f"  🟢 ALL SYSTEMS HEALTHY — bot is ready to trade")
    elif s["fail"] == 0:
        print(f"  🟡 WARNINGS PRESENT — review above before trading")
    else:
        print(f"  🔴 FAILURES DETECTED — action required before trading")
    print(f"{'═' * 65}\n")

    # ── Claude interpretation ──────────────────────────────────────
    if use_ai and all_issues:
        ai_analysis = _get_ai_analysis(all_issues, report["summary"])
        report["ai_analysis"] = ai_analysis
        if ai_analysis:
            print(f"\n{'─' * 65}")
            print(f"  🤖 CLAUDE ANALYSIS & ACTION PLAN")
            print(f"{'─' * 65}")
            print(ai_analysis)
            print(f"{'─' * 65}\n")
    elif use_ai and not all_issues:
        report["ai_analysis"] = "All systems healthy — no analysis needed."

    # ── Save report ───────────────────────────────────────────────
    _save_report(report)

    return report


def _get_ai_analysis(issues: list, summary: dict) -> str | None:
    """Calls Claude to interpret findings and return a prioritised action plan."""
    try:
        from ai_client import call_ai
    except ImportError:
        return None

    issues_text = "\n".join(
        f"  [{i['severity']}] {i['check']}: {i['detail']}"
        for i in issues
    )

    prompt = f"""You are a technical operations assistant for a live XAUUSD algorithmic trading bot.

The daily health check has found the following issues:

{issues_text}

Summary: {summary['fail']} FAIL(s), {summary['warn']} WARN(ings), {summary['ok']} OK

Your job:
1. Identify which issues are CRITICAL (must fix before trading today)
2. Identify which are MINOR (can trade, but should fix soon)
3. Give a numbered action list — most urgent first
4. For each action, give the exact command or file to edit
5. Be concise — max 300 words

Format:
CRITICAL ACTIONS (fix before trading):
1. [action]

MINOR ACTIONS (fix soon):
1. [action]

If everything is minor or there are no criticals, say so clearly at the top.
"""

    try:
        return call_ai(prompt=prompt, max_tokens=500)
    except Exception as e:
        return f"[AI analysis failed: {e}]"


def _save_report(report: dict):
    """Saves health report to Data/Logs/health_report_YYYY-MM-DD.json"""
    try:
        from paths import LOGS_DIR
        os.makedirs(LOGS_DIR, exist_ok=True)
        fname = f"health_report_{report['date']}.json"
        fpath = os.path.join(LOGS_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  [HealthAgent] Report saved → {fpath}")
    except Exception as e:
        print(f"  [HealthAgent] Could not save report: {e}")


# ================================================================
# DAEMON
# ================================================================

def _health_daemon_loop(verbose: bool, use_ai: bool):
    """Background loop — runs health check at 06:00 NY every morning."""
    print(f"[HealthAgent] 🏥 Daily health check daemon started. "
          f"Runs at {_CHECK_HOUR_NY:02d}:{_CHECK_MINUTE_NY:02d} NY every morning.")

    while True:
        now_ny = datetime.now(NY_TZ)
        target = now_ny.replace(hour=_CHECK_HOUR_NY, minute=_CHECK_MINUTE_NY,
                                second=0, microsecond=0)
        if now_ny >= target:
            target += timedelta(days=1)

        sleep_s = (target - now_ny).total_seconds()
        print(f"[HealthAgent] Next health check: "
              f"{target.strftime('%Y-%m-%d %H:%M')} NY "
              f"(in {sleep_s/3600:.1f}h)")

        slept = 0
        while slept < sleep_s:
            time.sleep(min(60, sleep_s - slept))
            slept += 60

        try:
            run_health_check(verbose=verbose, use_ai=use_ai)
        except Exception as e:
            print(f"[HealthAgent] Health check crashed: {e}")


def start_health_daemon(verbose: bool = False, use_ai: bool = True):
    """
    Starts the health check daemon thread.
    Called once from main_bot.py at startup.

    Also runs an immediate check at startup so you know the system
    state the moment the bot boots.
    """
    global _daemon_started
    if _daemon_started:
        return

    # Immediate startup check
    print("[HealthAgent] 🏥 Running startup health check...")
    try:
        run_health_check(verbose=verbose, use_ai=use_ai)
    except Exception as e:
        print(f"[HealthAgent] Startup check failed: {e}")

    # Background daemon for daily checks
    t = threading.Thread(
        target=_health_daemon_loop,
        args=(verbose, use_ai),
        name="HealthAgentDaemon",
        daemon=True
    )
    t.start()
    _daemon_started = True


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gold AI Bridge — System Health Agent"
    )
    parser.add_argument("--verbose",  action="store_true",
                        help="Show all checks, not just failures/warnings")
    parser.add_argument("--no-ai",    action="store_true",
                        help="Skip Claude AI interpretation (faster)")
    args = parser.parse_args()

    run_health_check(verbose=args.verbose, use_ai=not args.no_ai)
