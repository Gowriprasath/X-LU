"""
misswish_analyser.py — MissWish: Ideal Setup Extractor
========================================================
FIX C3/C7: All AI calls now use call_ai() — no more genai.Client bypass.
            run_analysis() no longer requires a client parameter.
            Always runs every day even with no taken trades.

Called by daily_post_mortem.py as STEP 5.
Fetches full day M5+H1 chart, sends to Claude with shadow journal context,
extracts every ideal setup that existed today into misswish_memory.json.
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # replaces genai.Client

import json
import os
import re
from datetime import datetime, timedelta
import pytz
import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

current_dir   = os.path.dirname(os.path.abspath(__file__))
base_dir      = os.path.dirname(current_dir)
import sys as _sys_mw; import os as _os_mw
_root_mw = _os_mw.path.normpath(_os_mw.path.join(_os_mw.path.dirname(_os_mw.path.abspath(__file__)), '..'))
if _root_mw not in _sys_mw.path: _sys_mw.path.insert(0, _root_mw)
from paths import (MISSWISH_MEMORY_PATH, SHADOW_JOURNAL_PATH,
                   TRADE_MEMORY_PATH, create_all_dirs as _cad_mw)
_cad_mw()
MISSWISH_FILE = MISSWISH_MEMORY_PATH
SHADOW_FILE   = SHADOW_JOURNAL_PATH
MEMORY_FILE   = TRADE_MEMORY_PATH

NY_TZ  = pytz.timezone('America/New_York')
SYMBOL = "XAUUSD"


# ================================================================
# FILE HELPERS
# ================================================================

def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[MissWish] Read error {path}: {e}")
    return default


def _write_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[MissWish] Write error {path}: {e}")
        return False


# ================================================================
# DATA FETCHING
# ================================================================

def _df_to_table(df: "pd.DataFrame", label: str, date_str: str) -> str:
    """Formats a DataFrame slice into the candle table string MissWish sends to Claude."""
    if df is None or df.empty:
        return f"No {label} data."
    MAX_ROWS = {"M5": 300, "H1": 48}
    cap = MAX_ROWS.get(label, 300)
    if len(df) > cap:
        df = df.tail(cap)
    lines = [f"--- XAUUSD {label} | {date_str} ({len(df)} candles) ---",
             "Time(UTC) | Open | High | Low | Close | Volume"]
    for dt, row in df.iterrows():
        lines.append(
            f"{dt.strftime('%H:%M')} | "
            f"{row.get('open', 0):.2f} | {row.get('high', 0):.2f} | "
            f"{row.get('low', 0):.2f} | {row.get('close', 0):.2f} | "
            f"{int(row.get('tick_volume', row.get('volume', 0)))}"
        )
    return "\n".join(lines)


def _fetch_day_candles(date_str: str,
                       m5_df=None,
                       h1_df=None) -> dict:
    """
    Returns M5 + H1 candle tables for date_str as a dict.

    FIX B4 — Backtest MT5 dependency:
        In live mode (m5_df/h1_df are None) fetch from MT5 as before.
        In backtest mode the engine passes the already-loaded DataFrames,
        so MT5 is never touched and the analysis runs on real historical data.
    """
    try:
        start_dt = datetime.strptime(date_str, "%Y-%m-%d")
        end_dt   = start_dt + timedelta(days=1)
    except ValueError:
        print(f"[MissWish] Invalid date format: {date_str}")
        return None

    # ── BACKTEST PATH — use pre-loaded DataFrames ────────────────────
    if m5_df is not None and h1_df is not None:
        s_dt = NY_TZ.localize(start_dt) if start_dt.tzinfo is None else start_dt
        e_dt = NY_TZ.localize(end_dt) if end_dt.tzinfo is None else end_dt
        day_m5 = m5_df[(m5_df.index >= s_dt) & (m5_df.index < e_dt)]
        day_h1 = h1_df[(h1_df.index >= s_dt) & (h1_df.index < e_dt)]
        if day_m5.empty:
            print(f"[MissWish] No M5 data in DataFrame for {date_str}.")
            return None
        return {
            "m5": _df_to_table(day_m5, "M5", date_str),
            "h1": _df_to_table(day_h1, "H1", date_str),
        }

    # ── LIVE PATH — fetch from MT5 ───────────────────────────────────
    if not mt5.initialize():
        print("[MissWish] MT5 not connected — cannot fetch candles.")
        return None

    try:
        # Retrieve the broker timezone dynamically before shutting down MT5
        try:
            sys.path.append(os.path.join(base_dir, "Python Files"))
            from data_extractor import _get_broker_tz
            broker_tz = _get_broker_tz()
        except Exception:
            broker_tz = pytz.timezone("Etc/GMT-3")  # Fallback to GMT+3 (broker time)

        # Localize naive dates to New York time (representing the NY trading day)
        ny_tz = pytz.timezone('America/New_York')
        s_dt_ny = ny_tz.localize(start_dt)
        e_dt_ny = ny_tz.localize(end_dt)

        # Convert to broker timezone and make naive (as copy_rates_range expects naive broker time)
        s_dt_broker = s_dt_ny.astimezone(broker_tz).replace(tzinfo=None)
        e_dt_broker = e_dt_ny.astimezone(broker_tz).replace(tzinfo=None)

        m5_rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, s_dt_broker, e_dt_broker)
        h1_rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, s_dt_broker, e_dt_broker)
    finally:
        mt5.shutdown()

    def _to_table(rates, label):
        if rates is None or len(rates) == 0:
            return f"No {label} data."
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        return _df_to_table(df, label, date_str)

    return {
        "m5": _to_table(m5_rates, "M5"),
        "h1": _to_table(h1_rates, "H1"),
    }


def _get_today_shadow_context(date_str: str) -> str:
    entries = _read_json(SHADOW_FILE, [])
    today_entries = [e for e in entries if e.get("timestamp", "").startswith(date_str)]

    if not today_entries:
        return "No shadow journal entries for today."

    lines = [f"--- BOT DECISION LOG ({date_str}) ---",
             "What the bot saw and did (or didn't do) today:"]

    for e in today_entries:
        gate    = e.get("gate_blocked_by", "?")
        signal  = e.get("signal", "?")
        entry_p = e.get("entry_price", 0)
        status  = e.get("status", "PENDING")
        regime  = e.get("regime", "?")
        session = e.get("session", "?")
        outcome = e.get("outcome_r")
        max_fav = e.get("max_favorable", 0)
        ts      = e.get("timestamp", "?")[11:16]

        action      = "TOOK THE TRADE" if gate == "TAKEN" else f"BLOCKED by {gate}"
        outcome_str = ""
        if status in ("TP_HIT", "SL_HIT", "EXPIRED"):
            outcome_str = (f" → Outcome: {status} "
                           f"({outcome:+.2f}R, max_fav: {max_fav:.2f}R)")

        lines.append(
            f"  {ts} {session} | {signal} @ {entry_p:.2f} | "
            f"Regime:{regime} | {action}{outcome_str}"
        )

    return "\n".join(lines)


def _get_today_taken_trades(date_str: str) -> str:
    trades = _read_json(MEMORY_FILE, [])
    today  = [t for t in trades if t.get("timestamp", "").startswith(date_str)]
    if not today:
        return "No trades taken today."

    lines = ["--- TRADES ACTUALLY TAKEN TODAY ---"]
    for t in today:
        lines.append(
            f"  Ticket:{t.get('ticket')} | {t.get('signal')} @ {t.get('entry')} | "
            f"SL:{t.get('sl')} TP:{t.get('tp')} | "
            f"Result:{t.get('result','')} | Regime:{t.get('regime','')} | "
            f"Session:{t.get('session','')}"
        )
        if t.get('hindsight_feedback', '').strip():
            lines.append(f"    Hindsight: {t['hindsight_feedback'][:200]}")
    return "\n".join(lines)


# ================================================================
# CLAUDE ANALYSIS
# ================================================================

def _analyse_with_claude(
    date_str:       str,
    m5_data:        str,
    h1_data:        str,
    shadow_context: str,
    taken_trades:   str,
) -> list:
    """
    Core MissWish prompt. Sends the full chart to Claude and asks it
    to identify every valid setup that existed — taken or not.
    Uses call_ai() — no raw API key needed.
    """
    prompt = f"""
You are a Senior ICT/Price Action Analyst reviewing a complete Gold (XAUUSD)
trading day after the fact. You have access to the full M5 chart and H1 context.

Your job is to identify EVERY valid, high-probability trading setup that
existed today — regardless of whether the bot traded it or not.

=== H1 STRUCTURE (Macro Context) ===
{h1_data}

=== M5 CHART (Full Day — Entry Precision) ===
{m5_data}

=== WHAT THE BOT DID TODAY ===
{taken_trades}

=== BOT DECISION LOG (gates fired + actual outcomes) ===
{shadow_context}

================================================================
TASK: Find every valid setup in the M5 chart.

For EACH setup you identify, provide ALL of the following fields.
A setup is valid if it has: a clear structural basis (OB, FVG, liquidity
sweep, BOS, CHoCH), a logical entry, a protected SL, and a realistic TP.

MANAGEMENT PATH: 3-4 sentences: where to move SL to breakeven, whether
to take a partial close, where to trail, final exit reason.

STRUCTURAL PATTERN KEYWORDS: 3-6 lowercase snake_case strings describing
the structural pattern mechanically (not direction or session).
Examples: "fvg_fill", "ob_retest", "liquidity_sweep_reversal",
"bos_continuation", "equal_highs_sweep", "displacement_into_ob",
"asian_high_break", "mitigation_block_tap", "choch_confirmation"

Respond ONLY with a valid JSON array. No explanation outside the JSON.
Each element must have EXACTLY these fields:

[
  {{
    "session":                    "Asian|London|NY|Dead",
    "signal":                     "BUY|SELL",
    "ideal_entry":                2345.67,
    "ideal_sl":                   2339.00,
    "ideal_tp":                   2363.00,
    "rr":                         2.6,
    "regime_at_signal":           "BULL_TREND|BEAR_TREND|LOW_VOL_RANGE|REVERSAL|COMPRESSION|UNKNOWN",
    "management_path":            "Enter on M5 close above OB at 2345. Move SL to breakeven at 1R (2352). Take 50% partial at 2355. Trail remainder below each M5 swing low. Exit full at TP 2363.",
    "why_it_worked":              "Price displaced from a valid H1 order block after sweeping Asian lows. BOS on M5 confirmed the reversal. FVG above entry provided first TP target.",
    "structural_pattern_keywords": ["asian_low_sweep", "h1_ob_retest", "bos_confirmation", "fvg_target"],
    "bot_action":                 "TAKEN|BLOCKED|MISSED",
    "bot_gate":                   "META_GATE|CONFIDENCE_GATE|CLAUDE_WAIT|DUAL_GATE|LONDON_GATE|NEWS_BLOCK|CONSECUTIVE_WAIT|LLM_CACHE|NONE",
    "setup_time_utc":             "09:35",
    "actual_outcome":             "TP_HIT|SL_HIT|EXPIRED|UNKNOWN",
    "outcome_r":                  2.4
  }}
]

If there are no valid setups today, return an empty array: []
"""
    raw_text = call_ai(prompt=prompt, max_tokens=4096)
    if raw_text is None:
        return []

    try:
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*',     '', raw_text)
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            setups = json.loads(match.group())
            if isinstance(setups, list):
                return setups
        setups = json.loads(raw_text)
        if isinstance(setups, list):
            return setups
    except Exception as e:
        print(f"[MissWish] JSON parse error: {e}")

    return []


# ================================================================
# WRITE MissWish ENTRIES
# ================================================================

def _build_entry(setup: dict, date_str: str) -> dict:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return {
        "id":                        f"mw_{date_str.replace('-','')}_{ts[-6:]}",
        "date":                      date_str,
        "setup_time_utc":            str(setup.get("setup_time_utc", "")),
        "session":                   str(setup.get("session", "UNKNOWN")),
        "signal":                    str(setup.get("signal", "?")).upper(),
        "ideal_entry":               float(setup.get("ideal_entry", 0)),
        "ideal_sl":                  float(setup.get("ideal_sl", 0)),
        "ideal_tp":                  float(setup.get("ideal_tp", 0)),
        "rr":                        float(setup.get("rr", 0)),
        "regime_at_signal":          str(setup.get("regime_at_signal", "UNKNOWN")),
        "management_path":           str(setup.get("management_path", ""))[:600],
        "why_it_worked":             str(setup.get("why_it_worked", ""))[:400],
        "structural_pattern_keywords": list(setup.get("structural_pattern_keywords", [])),
        "bot_action":                str(setup.get("bot_action", "UNKNOWN")),
        "bot_gate":                  str(setup.get("bot_gate", "NONE")),
        "actual_outcome":            str(setup.get("actual_outcome", "UNKNOWN")),
        "outcome_r":                 float(setup.get("outcome_r", 0) or 0),
        "tagged":                    False,
        "created_at":                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def _append_entries(new_entries: list):
    existing = _read_json(MISSWISH_FILE, [])

    existing_keys = set()
    for e in existing:
        key = f"{e.get('date')}_{e.get('signal')}_{e.get('ideal_entry')}"
        existing_keys.add(key)

    added = 0
    for entry in new_entries:
        key = f"{entry.get('date')}_{entry.get('signal')}_{entry.get('ideal_entry')}"
        if key in existing_keys:
            print(f"[MissWish] Duplicate skipped: {key}")
            continue
        existing.append(entry)
        existing_keys.add(key)
        added += 1

    if len(existing) > 1000:
        existing = existing[-1000:]

    _write_json(MISSWISH_FILE, existing)
    return added


# ================================================================
# PUBLIC ENTRY POINT
# ================================================================

def run_analysis(date_str: str,
                 m5_df=None,
                 h1_df=None) -> list:
    """
    Main entry point. Called from daily_post_mortem.py as STEP 5.

    FIX C3/C7: No longer requires a client parameter.
    FIX B4:    Accepts optional m5_df / h1_df from the backtest engine.
               When supplied, candles are sliced from those DataFrames
               instead of fetching from MT5 (which is never connected
               during a backtest).

    Args:
        date_str : "YYYY-MM-DD" — the trading day to analyse
        m5_df    : full M5 DataFrame from backtest (None in live mode)
        h1_df    : full H1 DataFrame from backtest (None in live mode)

    Returns:
        list of new MissWish entry dicts (for downstream tagging)
    """
    print(f"\n[MissWish] ══════════════════════════════════════")
    print(f"[MissWish] Analysing {date_str} for missed/ideal setups...")

    candles = _fetch_day_candles(date_str, m5_df=m5_df, h1_df=h1_df)
    if not candles:
        print(f"[MissWish] Could not fetch candles for {date_str}. Skipping.")
        return []

    shadow_context = _get_today_shadow_context(date_str)
    taken_trades   = _get_today_taken_trades(date_str)

    print(f"[MissWish] Sending full day chart to Claude for setup extraction...")
    setups = _analyse_with_claude(
        date_str       = date_str,
        m5_data        = candles["m5"],
        h1_data        = candles["h1"],
        shadow_context = shadow_context,
        taken_trades   = taken_trades,
    )

    if not setups:
        print(f"[MissWish] No valid setups identified for {date_str}.")
        return []

    print(f"[MissWish] {len(setups)} setup(s) identified by Claude.")

    new_entries = [_build_entry(s, date_str) for s in setups
                   if s.get("signal") in ("BUY", "SELL")
                   and float(s.get("ideal_entry", 0)) > 0]

    added = _append_entries(new_entries)
    print(f"[MissWish] {added} new entries written to misswish_memory.json")

    print(f"\n{'='*55}")
    print(f" MISSWISH — TODAY'S IDEAL SETUPS ({date_str})")
    print(f"{'='*55}")
    for e in new_entries:
        status_icon = "✓" if e["bot_action"] == "TAKEN" else \
                      "✗" if e["bot_action"] == "BLOCKED" else "○"
        print(f"  {status_icon} {e['session']} {e['signal']} @ {e['ideal_entry']:.2f} | "
              f"SL:{e['ideal_sl']:.2f} TP:{e['ideal_tp']:.2f} "
              f"(RR:{e['rr']:.1f}) | {e['actual_outcome']} "
              f"({e['outcome_r']:+.1f}R) | {e['bot_action']}"
              + (f" [{e['bot_gate']}]" if e['bot_gate'] != 'NONE' else ""))
    print(f"{'='*55}\n")

    return new_entries