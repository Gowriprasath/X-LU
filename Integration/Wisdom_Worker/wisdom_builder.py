"""
wisdom_builder.py
==================
Runs as a background thread from main_bot.py.

FIX C4: Removed ALL direct genai.Client() calls. All AI now via call_ai().
FIX M5: Both sys.path blocks now point to the correct root directory so
         standalone execution works without crashing.

ISSUE 5 FIX: Wisdom Drift — AI editing AI memory without constraint.

New design — Two-file separation:
    human_rules.json  — YOU write this. The bot NEVER touches it.
    ai_lessons.json   — The bot writes this. AI can only ADD new lessons.
                        Each lesson has a 'confidence' counter.
    wisdom.json       — DEPRECATED. Left in place for backward compat only.
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # FIX C4: replaces genai.Client

import json
import os
import re
import time
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

current_dir  = os.path.dirname(os.path.abspath(__file__))
base_dir     = os.path.dirname(os.path.dirname(current_dir))
import sys as _sys_wb
if base_dir not in _sys_wb.path: _sys_wb.path.insert(0, base_dir)
from paths import (TRADE_MEMORY_PATH, WISDOM_PATH, AI_LESSONS_PATH,
                   HUMAN_RULES_PATH, WISDOM_TRACKER_PATH, create_all_dirs as _cad_wb)
_cad_wb()
MEMORY_FILE      = TRADE_MEMORY_PATH
WISDOM_FILE      = WISDOM_PATH
AI_LESSONS_FILE  = AI_LESSONS_PATH
HUMAN_RULES_FILE = HUMAN_RULES_PATH
TRACKER_FILE     = WISDOM_TRACKER_PATH

from master_controls import WISDOM_REBUILD_DAYS as TRADING_DAYS_THRESHOLD
LESSON_MIN_CONFIDENCE    = 1
LESSON_GRADUATE_CONFIDENCE = 3
LESSON_STALE_REBUILDS    = 10


# ================================================================
# FILE HELPERS
# ================================================================

def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WisdomBuilder] WARNING: Could not read {path}: {e}")
    return default


def _write_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"[WisdomBuilder] WARNING: Could not write {path}: {e}")
        return False


# ================================================================
# TRADING DAY CALCULATOR
# ================================================================

def _count_trading_days(from_date_str, to_date):
    if not from_date_str:
        return 999
    try:
        from_date = datetime.strptime(from_date_str, '%Y-%m-%d').date()
        count, current = 0, from_date + timedelta(days=1)
        while current <= to_date:
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        return count
    except Exception as e:
        print(f"[WisdomBuilder] Date calculation error: {e}")
        return 0


# ================================================================
# UNPROCESSED TRADE RETRIEVAL
# ================================================================

def _get_unprocessed_trades(last_ticket):
    memory = _read_json(MEMORY_FILE, [])
    reviewed = [
        t for t in memory
        if t.get('status') in ('CLOSED', 'REVIEWED')
        and t.get('result', '').strip() != ''
        and t.get('hindsight_feedback', '').strip() != ''
    ]
    if not last_ticket:
        return reviewed
    found, unprocessed = False, []
    for trade in reviewed:
        if found:
            unprocessed.append(trade)
        if str(trade.get('ticket')) == str(last_ticket):
            found = True
    return unprocessed if found else reviewed


# ================================================================
# ISSUE 5 FIX: ADDITIVE-ONLY LESSON EXTRACTION
# FIX C4: Uses call_ai() — no genai.Client
# ================================================================

def _extract_new_lessons(new_trades: list, existing_lessons: dict,
                         rebuild_count: int) -> dict:
    """
    Calls Claude to extract lessons from new trades.
    FIX C4: Replaced genai.Client with call_ai().
    """
    meta_stats_text = "No meta-labelling data yet."
    try:
        import sys
        meta_dir = os.path.join(base_dir, 'Quant', 'meta_labeller')  # code dir only
        if meta_dir not in sys.path:
            sys.path.insert(0, meta_dir)
        from meta_labeller import get_regime_meta_stats
        stats = get_regime_meta_stats()
        if stats:
            lines = ["Meta-model accuracy by regime (last 200 trades):"]
            for regime, s in stats.items():
                lines.append(f"  {regime}: {s['accuracy']:.0%} accuracy "
                             f"over {s['sample_size']} trades")
            meta_stats_text = "\n".join(lines)
    except Exception:
        pass

    if existing_lessons:
        existing_text_parts = []
        for k, v in existing_lessons.items():
            if isinstance(v, dict):
                conf = v.get("confidence", 1)
                text = v.get("text", '')
            else:
                conf = 1
                text = str(v)
            status = "CONFIRMED" if conf >= LESSON_GRADUATE_CONFIDENCE else "tentative"
            existing_text_parts.append(f"  [{k}] (conf={conf}, {status}): {text}")
        existing_text = "\n".join(existing_text_parts)
    else:
        existing_text = "None yet."

    prompt = f"""
You are the long-term memory manager for a Gold (XAUUSD) trading bot.

IMPORTANT CONSTRAINT: You may only ADD new lessons. You may NOT rewrite,
delete, or modify any existing lesson. Existing lessons are shown for
context only — do NOT include them in your output.

--- EXISTING LESSONS (READ-ONLY — do not include in output) ---
{existing_text}

--- META-MODEL PERFORMANCE STATISTICS ---
{meta_stats_text}

--- NEW TRADE RECORDS TO ANALYSE ---
{json.dumps(new_trades, indent=2)}

Your task:
1. Read each new trade's result, reasoning, and hindsight_feedback.
2. Extract ONLY lessons that are GENUINELY NEW and not already covered
   by the existing lessons above.
3. Also check: does any new trade CORROBORATE an existing lesson?
   If yes, list the corroborated lesson keys in a "corroborations" array.
4. Format new lessons as short snake_case keys with 2-4 sentence values.
5. Focus on structural market lessons. Not generic advice.

Output ONLY valid JSON with this exact structure:
{{
  "new_lessons": {{
    "lesson_key_here": "2-4 sentence lesson text here.",
    "another_lesson": "Another lesson."
  }},
  "corroborations": ["existing_lesson_key_1", "existing_lesson_key_2"]
}}

If there are no new lessons, return: {{"new_lessons": {{}}, "corroborations": []}}
"""
    try:
        raw = call_ai(prompt=prompt)  # FIX C4
        if raw is None:
            print("[WisdomBuilder] AI call failed — skipping this rebuild.")
            return existing_lessons

        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*',     '', raw)
        result = json.loads(raw)
        if not isinstance(result, dict):
            return existing_lessons
    except Exception as e:
        print(f"[WisdomBuilder] Lesson extraction failed: {e}")
        return existing_lessons

    new_lessons_raw = result.get("new_lessons", {})
    corroborations  = result.get("corroborations", [])

    for key in corroborations:
        if key in existing_lessons:
            entry = existing_lessons[key]
            if isinstance(entry, dict):
                entry['confidence'] = entry.get('confidence', 1) + 1
                entry['last_corroborated'] = datetime.now().strftime('%Y-%m-%d')
                print(f"[WisdomBuilder] Corroborated: '{key}' "
                      f"(confidence now {entry['confidence']})")
            else:
                existing_lessons[key] = {
                    'text':               str(entry),
                    'confidence':         2,
                    'added_date':         'legacy',
                    'last_corroborated':  datetime.now().strftime('%Y-%m-%d'),
                    'rebuild_at_add':     rebuild_count,
                }

    added = 0
    for key, text in new_lessons_raw.items():
        if key.startswith('_'):
            continue
        if key not in existing_lessons:
            existing_lessons[key] = {
                'text':              str(text),
                'confidence':        1,
                'added_date':        datetime.now().strftime('%Y-%m-%d'),
                'last_corroborated': None,
                'rebuild_at_add':    rebuild_count,
            }
            added += 1
            print(f"[WisdomBuilder] New lesson added: '{key}' (tentative, conf=1)")
        else:
            print(f"[WisdomBuilder] Skipped duplicate: '{key}' already exists.")

    stale_keys = [
        k for k, v in existing_lessons.items()
        if isinstance(v, dict)
        and v.get('confidence', 1) == 1
        and (rebuild_count - v.get('rebuild_at_add', 0)) >= LESSON_STALE_REBUILDS
    ]
    for k in stale_keys:
        del existing_lessons[k]
        print(f"[WisdomBuilder] Expired stale lesson: '{k}'")

    print(f"[WisdomBuilder] Lesson update: "
          f"+{added} new | {len(corroborations)} corroborated | "
          f"{len(stale_keys)} expired | {len(existing_lessons)} total")
    return existing_lessons


# ================================================================
# MAIN ENTRY POINT
# ================================================================

def check_and_run_if_needed(simulated_time=None):
    """
    Live mode  : called as a background thread from main_bot.py.
                 Checks every real hour via time.sleep(3600).
    Backtest   : called from check_simulated_wisdom() every 5 simulated
                 trading days with simulated_time set.

    FIX B2 — Backtest sleep bomb:
        Previously this function always ended with time.sleep(3600),
        which froze the backtest for a real hour every 5 simulated days
        (~300 freeze events × 1h = 300h of real sleeping across a
        2016-2022 backtest).

        Fix: when simulated_time is supplied, use the simulated date
        instead of date.today() and return immediately after the rebuild
        check — no sleeping at all.
    """

    # ----------------------------------------------------------------
    # BACKTEST MODE — one-shot check using the simulated date; no sleep.
    # ----------------------------------------------------------------
    if simulated_time is not None:
        try:
            sim_date = simulated_time.date() if hasattr(simulated_time, 'date') else simulated_time
            tracker     = _read_json(TRACKER_FILE, {})
            last_date   = tracker.get('last_rebuild_date')
            last_ticket = tracker.get('last_ticket_processed')
            elapsed     = _count_trading_days(last_date, sim_date)

            if elapsed >= TRADING_DAYS_THRESHOLD:
                print(f"\n[WisdomBuilder] Backtest: {elapsed} simulated trading days "
                      f"elapsed. Triggering lesson rebuild...")
                _run_rebuild(tracker, last_ticket, sim_date)
            else:
                remaining = TRADING_DAYS_THRESHOLD - elapsed
                print(f"[WisdomBuilder] Backtest: {elapsed}/{TRADING_DAYS_THRESHOLD} "
                      f"simulated days. {remaining} day(s) until next rebuild.")
        except Exception as e:
            print(f"[WisdomBuilder] Backtest error in check: {e}")
        return   # ← no sleep, no loop

    # ----------------------------------------------------------------
    # LIVE MODE — long-lived loop sleeping 1h between checks.
    # ----------------------------------------------------------------
    print("[WisdomBuilder] Background thread started.")
    while True:
        try:
            tracker     = _read_json(TRACKER_FILE, {})
            last_date   = tracker.get('last_rebuild_date')
            last_ticket = tracker.get('last_ticket_processed')
            today       = date.today()
            elapsed     = _count_trading_days(last_date, today)

            if elapsed >= TRADING_DAYS_THRESHOLD:
                print(f"\n[WisdomBuilder] {elapsed} trading days elapsed. "
                      f"Triggering lesson rebuild...")
                _run_rebuild(tracker, last_ticket, today)
            else:
                remaining = TRADING_DAYS_THRESHOLD - elapsed
                print(f"[WisdomBuilder] {elapsed}/{TRADING_DAYS_THRESHOLD} "
                      f"trading days. {remaining} day(s) until next rebuild.")
        except Exception as e:
            print(f"[WisdomBuilder] Error in check loop: {e}")
        time.sleep(3600)


def _run_rebuild(tracker, last_ticket, today):
    new_trades = _get_unprocessed_trades(last_ticket)
    if not new_trades:
        print("[WisdomBuilder] No new completed trades. Resetting counter.")
        tracker['last_rebuild_date'] = today.strftime('%Y-%m-%d')
        _write_json(TRACKER_FILE, tracker)
        return

    print(f"[WisdomBuilder] Processing {len(new_trades)} new trade(s)...")
    rebuild_count    = tracker.get('total_rebuilds', 0) + 1
    existing_lessons = _read_json(AI_LESSONS_FILE, {})

    updated_lessons = _extract_new_lessons(new_trades, existing_lessons, rebuild_count)

    if not _write_json(AI_LESSONS_FILE, updated_lessons):
        print("[WisdomBuilder] Failed to write ai_lessons.json. Retrying next cycle.")
        return

    last_processed = str(new_trades[-1].get('ticket', ''))
    tracker['last_ticket_processed'] = last_processed
    tracker['last_rebuild_date']     = today.strftime('%Y-%m-%d')
    tracker['total_rebuilds']        = rebuild_count
    _write_json(TRACKER_FILE, tracker)

    confirmed = sum(
        1 for v in updated_lessons.values()
        if isinstance(v, dict) and v.get('confidence', 0) >= LESSON_GRADUATE_CONFIDENCE
    )
    tentative = len(updated_lessons) - confirmed
    print(f"[WisdomBuilder] Rebuild #{rebuild_count} complete. "
          f"Lessons: {confirmed} confirmed + {tentative} tentative | "
          f"Last ticket: {last_processed}")