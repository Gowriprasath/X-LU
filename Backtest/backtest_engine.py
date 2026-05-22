"""
Backtest/backtest_engine.py — Antigravity Bridge v16
=====================================================
Replays the EXACT live bot pipeline on historical CSV data.

VERIFIED IDENTICAL TO LIVE BOT (v16)
─────────────────────────────────────
  ✅  AI provider       → Claude Sonnet 4.6 via call_ai() (same as live)
  ✅  Session gating    → Asian 19–01 / London 02–05 / NY_AM 07–12
  ✅  Session windows   → derived from master_controls.SESSIONS (BUG-02 FIX)
  ✅  London 3/3 gate
  ✅  Pre-filter        → HIGH_VOL + consecutive WAIT block
  ✅  News gate         → historical FF calendar, same windows as live
  ✅  News partial close→ open position 50% partial 2–7min before HI news (GAP-01 FIX)
  ✅  Spread gate       → rejects trades when spread > GATE_MAX_SPREAD_DOLLARS (BUG-06 FIX)
  ✅  Spread + slippage → session-aware + news-spike widening
  ✅  Regime detection  → same GMM-HMM + XGBoost slice (no lookahead)
  ✅  Regime router     → dual gate, confidence gate, adapt_live_trade
  ✅  SL/TP adjuster    → rr.adjust_sl_tp() applied before execution (BUG-04 FIX)
  ✅  Hallucination guard → validate_trade_logic()
  ✅  Position sizing   → news-aware (halved near HI events) (BUG-05 FIX)
  ✅  Memory system     → post-mortem, wisdom builder, keyword tagger
  ✅  Memory logging    → session= field populated, no TypeError (BUG-03 FIX)
  ✅  Risk manager      → same clearance + consecutive loss rules
  ✅  Episode recorder
  ✅  Trade management  → partial close (50%@1R) + break-even + M5 trail
                          + HYBRID dispatch: REVERSAL close (≥65%),
                          TRAIL_AGGRESSIVE, SKIP_PARTIAL, PARTIAL_75, TIGHT_BE
  ✅  Overnight guard   → REVERSAL close, CLOSE_PARTIAL, counter-trend close (GAP-02 FIX)
  ✅  Trade timeout     → force-close after 288 candles (24h) (GAP-04 FIX)
  ✅  format_gate_context → 3-arg call matching live bot exactly (GAP-03 FIX)

INTENTIONAL DIFFERENCES (do not fix — by design)
─────────────────────────────────────────────────
  D-10  Per-candle AI HOLD/CLOSE call while trade is open
        Live: AI called every 5-min cycle and can signal CLOSE at any time.
        Backtest: no per-candle AI call on open trade — would cost ~200 API
        calls per trade. Reversal coverage provided by hybrid dispatch instead.

  D-11  adapt_live_trade TIGHTEN_TP / WIDEN_SL (mid-trade minor adjustments)
        These don't affect win/loss outcome significantly and need per-candle
        regime calls. CLOSE_PARTIAL is now implemented (GAP-02 FIX). Only the
        minor TP/SL nudges (TIGHTEN_TP, WIDEN_SL) remain omitted.

  D-13  shadow_journal
        Records every signal (taken + blocked) for live analysis dashboards.
        Not applicable in backtest context.

  D-14  _decision_cache
        Caches identical WAIT/HOLD to avoid redundant API calls on live bot.
        Backtest runs all candles sequentially — cache would never hit.

  D-15  strategy_selector
        Live bot routes prompts through a strategy classifier. Backtest uses
        the same base prompt; adding strategy_selector would not materially
        change signal accuracy but would slow the run.

  D-18  black_swan_monitor
        Live-only: scans Reuters/Bloomberg RSS at runtime. No historical
        equivalent. Black swan events during the backtest period are
        unaccounted for — treat backtest PnL in crisis periods with caution.

WHAT IS DIFFERENT (data layer only — unavoidable)
──────────────────────────────────────────────────
  - Data source   → CSV files instead of MT5 live ticks
  - Trade fills   → simulated at AI's quoted price + spread + slippage
  - time.sleep()  → removed (runs as fast as possible)
  - Post-mortem   → fires at simulated 5PM NY day boundaries
  - Wisdom builder → fires at simulated 5-trading-day boundaries

Usage:
    python backtest_engine.py
    python backtest_engine.py --from 2020-01-01 --to 2023-12-31
    python backtest_engine.py --resume
"""

import os
import sys
import json
import pytz
import re
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────
BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKTEST_DIR)

sys.path.append(os.path.join(PROJECT_ROOT, "Python Files"))
sys.path.append(os.path.join(PROJECT_ROOT, "Strategy"))
sys.path.append(os.path.join(PROJECT_ROOT, "Memory"))
sys.path.append(os.path.join(PROJECT_ROOT, "Integration"))
sys.path.insert(0, PROJECT_ROOT)
from paths import BACKTEST_TRACKER_PATH, MARKET_DATA_DIR as _MARKET_DIR, create_all_dirs as _create_all_dirs
_create_all_dirs()
sys.path.append(os.path.join(PROJECT_ROOT, "Integration", "Wisdom_Worker"))
sys.path.append(os.path.join(PROJECT_ROOT, "Quant", "regime_detector"))
sys.path.append(os.path.join(PROJECT_ROOT, "Quant", "rl_manager"))
sys.path.append(os.path.join(PROJECT_ROOT, "Quant", "regime_router"))
sys.path.append(os.path.join(PROJECT_ROOT, "Quant", "meta_labeller"))
sys.path.append(BACKTEST_DIR)

import risk_manager
import strategy_rules
import strategy_logic
import memory_manager
import thought_logger
import daily_post_mortem
import regime_detector as rd
import regime_router as rr
import episode_recorder as er
# D-01 FIX: use same AI provider as live bot (Claude Sonnet 4.6 via call_ai)
# instead of Claude 2.5-Flash via google.genai. Backtest and live now call
# identical model with identical prompts — results are directly comparable.
from ai_client import call_ai, AI_MODEL, AI_DISPLAY_NAME
import pnl_tracker as _pnlt
from wisdom_builder import check_and_run_if_needed as wisdom_check
from context_retriever import get_full_memory_context
from data_downloader import load_data, get_available_date_range
from feature_engineer import build_features, build_time_features
from spread_simulator import SpreadSimulator, get_spread as _get_spread_fn
from news_history import (get_news_for_date, is_in_news_window,
                           format_for_prompt as format_news_for_prompt)

_spread_sim = SpreadSimulator()

# XU-L meta-labelling layer
try:
    import meta_labeller as ml
    _META_ENABLED = True
except ImportError:
    _META_ENABLED = False
    print("[Backtest] meta_labeller not available — running without meta gate.")

# Session profiler (non-fatal if not yet built)
try:
    from session_profiler import (enrich_regime_result,
                                  format_session_profile_for_prompt,
                                  compute_norm_vol)
    _PROFILER_ENABLED = True
except ImportError:
    _PROFILER_ENABLED = False

# ── Config ─────────────────────────────────────────────────────────
# load_dotenv() called at the top of the file to fix import-order validation bug
# Signal to risk_manager that this is a backtest run — suppresses per-candle
# "Risk cleared" prints and skips MT5 calls (which hit every active candle).
# Only use mock if no real AI key is available
import os
_has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
_has_claude = any(
    os.getenv(f"CLAUDE_API_KEY_{i}", "").strip()
    for i in range(1, 4)
)
if not _has_gemini and not _has_claude:
    os.environ["BACKTEST_MODE"] = "1"
    print("[Backtest] No AI keys found — running in mock mode.")
else:
    print("[Backtest] AI key found — running with real AI decisions.")

# D-01 FIX: GEMINI_API_KEY removed — backtest now uses the same Anthropic
# keys as the live bot, loaded automatically by ai_client.py from .env
SYMBOL   = "XAUUSD"
NY_TZ    = pytz.timezone('America/New_York')

TRACKER_FILE      = BACKTEST_TRACKER_PATH
SAVE_EVERY_N      = 50
TICKET_PREFIX     = "BACKTEST_"
H1_LOOKBACK       = 300
MAGIC_NUMBER      = 99999

# M-01 FIX: ACTIVE_WINDOWS was a hardcoded static list. Any future edit to
# master_controls.SESSIONS propagated to the live bot automatically but NOT
# to the backtest, causing silent session divergence. Now derived dynamically
# from the same master_controls.SESSIONS as the live bot — one edit propagates
# to both. Logic mirrors main_bot._build_active_windows() exactly.
from master_controls import SESSIONS as _BT_SESSIONS

def _build_active_windows(sessions: dict) -> list:
    windows = []
    for label, cfg in sessions.items():
        s = int(cfg["start"].split(":")[0])
        e = int(cfg["end"].split(":")[0])   # end is EXCLUSIVE
        win_label = cfg.get("label", label)
        if s >= e:          # midnight-crossing session
            windows.append((s, 24, win_label))
            windows.append((0, e,  win_label))
        else:
            windows.append((s, e, win_label))
    return windows

ACTIVE_WINDOWS = _build_active_windows(_BT_SESSIONS)

# ── Active trade state ─────────────────────────────────────────────
# D-02/D-03/D-04 FIX: added management fields so _bt_manage_trade() can
# replicate the live bot's partial close / break-even / trailing loop.
_active_trade = {
    "ticket":           None,
    "type":             None,
    "entry":            0.0,
    "sl":               0.0,
    "tp":               0.0,
    "lot":              0.0,
    "remaining_lot":    0.0,   # starts = lot; drops to lot/2 after partial close
    "management_stage": 0,
    "partial_taken":    False,
    "partial_pnl":      0.0,   # PnL already banked from the partial close
    "partial_cycle":    False,  # True for 1 cycle after partial → defer BE
    "spread_at_entry":  0.0,
    "regime_at_entry":  "",
}

# ── Backtest stats ─────────────────────────────────────────────────
_stats = {
    "total_trades": 0, "wins": 0, "losses": 0,
    "total_pnl": 0.0, "api_calls": 0,
    "skipped_dead_zone": 0, "skipped_prefilter": 0,
    "skipped_consecutive_wait": 0,
    "skipped_news_window": 0,       # NEW — news gate applied to backtest
    "london_blocked": 0,
    "dual_gate_blocked": 0,
    "meta_gate_blocked": 0,
    "confidence_gate_blocked": 0,
    "claude_wait_blocked":     0,
    "hallucination_blocked":   0,
    "rr_blocked":              0,
    "total_spread_cost": 0.0,       # cumulative spread paid across all trades
    "total_slippage_cost": 0.0,     # cumulative slippage paid across all trades
}

# ── Consecutive WAIT tracker ──────────────────────────────────────
# Mirror of main_bot.py — prevents redundant AI calls when market is
# unchanged and AI already said WAIT. Resets on non-WAIT or regime change.
_consecutive_waits = 0
_last_wait_regime  = None


# ================================================================
# PROGRESS TRACKER
# ================================================================
def load_tracker():
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_processed_candle": None,
        "candles_processed": 0,
        "started_at": None,
        "last_saved": None,
        "stats": _stats.copy(),
    }


def save_tracker(tracker):
    tracker["last_saved"]  = datetime.now().isoformat()
    tracker["stats"]       = _stats.copy()
    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=4, default=str)


# ================================================================
# SESSION GATING (identical to live bot)
# ================================================================
def get_session(dt):
    hour = dt.hour
    for start, end, name in ACTIVE_WINDOWS:
        if start <= hour < end:
            return name
    return None


def is_active(dt):
    return get_session(dt) is not None


# ================================================================
# SIMULATED TRADE EXECUTION
# No real MT5 orders — outcome is determined from future candles
# ================================================================
def simulate_trade_outcome(signal, entry, sl, tp, candles_after, spread=0.0):
    """
    Given entry/sl/tp and subsequent M5 candles, determines trade outcome.

    Walk forward through candles_after until:
      - Price hits TP  → WIN
      - Price hits SL  → LOSS
      - 200 candles pass with no resolution → timeout (LOSS at current close)

    spread is in USD/oz (same units as price). Applied via apply_spread_cost()
    so every round-trip pays exactly one spread — identical to live execution.

    Returns:
        dict: {
            "result":      "WIN" | "LOSS" | "TIMEOUT",
            "close_price": float,
            "pnl_pips":    float,   ← AFTER spread deduction
            "pnl_pips_raw": float,  ← before spread (for diagnostics)
            "spread_paid": float,
            "candles_held": int,
            "close_time":  datetime,
        }
    """
    for i, (candle_dt, candle) in enumerate(candles_after.iterrows()):
        high  = candle['high']
        low   = candle['low']
        close = candle['close']

        if signal == "BUY":
            if low <= sl:
                raw_pips = sl - entry
                result   = "LOSS"
                close_price = sl
                candles_held = i + 1
                close_time   = candle_dt
                break
            if high >= tp:
                raw_pips = tp - entry
                result   = "WIN"
                close_price = tp
                candles_held = i + 1
                close_time   = candle_dt
                break

        elif signal == "SELL":
            if high >= sl:
                raw_pips = entry - sl
                result   = "LOSS"
                close_price = sl
                candles_held = i + 1
                close_time   = candle_dt
                break
            if low <= tp:
                raw_pips = entry - tp
                result   = "WIN"
                close_price = tp
                candles_held = i + 1
                close_time   = candle_dt
                break

        # Timeout after 200 candles (~16 hours of M5)
        if i >= 199:
            raw_pips   = (close - entry) if signal == "BUY" else (entry - close)
            result     = "WIN" if raw_pips > 0 else "LOSS"
            close_price  = close
            candles_held = i + 1
            close_time   = candle_dt
            break
    else:
        # Ran out of candles (end of data)
        if candles_after.empty:
            raw_pips    = 0.0
            close_price = entry
            close_time  = None
        else:
            last_close  = candles_after.iloc[-1]['close']
            raw_pips    = (last_close - entry) if signal == "BUY" else (entry - last_close)
            close_price = last_close
            close_time  = candles_after.index[-1]
        result       = "WIN" if raw_pips > 0 else "LOSS"
        candles_held = len(candles_after)

    # Apply spread cost (always one full spread per round-trip)
    adjusted_pips = SpreadSimulator.apply_spread_cost(raw_pips, spread)

    return {
        "result":       result,
        "close_price":  close_price,
        "pnl_pips":     adjusted_pips,
        "pnl_pips_raw": raw_pips,
        "spread_paid":  spread,
        "candles_held": candles_held,
        "close_time":   close_time,
    }


def calculate_pnl_dollars(pnl_pips, lot, contract_size=100):
    """Convert pip PnL to dollar PnL. Gold: 1 pip = $1 per 0.01 lot."""
    return pnl_pips * lot * contract_size


# ================================================================
# HALLUCINATION GUARD (identical to live bot)
# ================================================================
def validate_trade_logic(decision, current_price):
    signal = decision.get("signal", "WAIT").upper()
    if signal not in ["BUY", "SELL"]:
        return True
    entry = float(decision.get("entry", 0))
    sl    = float(decision.get("sl",    0))
    tp    = float(decision.get("tp",    0))
    if entry <= 0 or sl <= 0 or tp <= 0:
        return False
    if abs(entry - current_price) > 20:
        return False
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return False
    rr = abs(tp - entry) / sl_dist
    if rr < 1.5:
        return False
    if signal == "BUY" and not (sl < entry < tp):
        return False
    if signal == "SELL" and not (tp < entry < sl):
        return False
    return True


def _diagnose_trade_failure(decision, current_price):
    """
    Called only when validate_trade_logic() returns False.
    Returns one of: "hallucination", "rr", "levels"
    so the caller can increment the right counter.

    "hallucination" — entry price too far from live price
    "rr"            — risk:reward below minimum
    "levels"        — sl/tp/entry arrangement invalid
    """
    signal = decision.get("signal", "WAIT").upper()
    if signal not in ["BUY", "SELL"]:
        return "levels"

    entry = float(decision.get("entry", 0))
    sl    = float(decision.get("sl",    0))
    tp    = float(decision.get("tp",    0))

    if entry <= 0 or sl <= 0 or tp <= 0:
        return "levels"

    # Check hallucination first (entry vs current price)
    if abs(entry - current_price) > 20:
        return "hallucination"

    # Check RR
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return "levels"
    rr = abs(tp - entry) / sl_dist
    if rr < 1.5:
        return "rr"

    # Check price arrangement
    if signal == "BUY" and not (sl < entry < tp):
        return "levels"
    if signal == "SELL" and not (tp < entry < sl):
        return "levels"

    return "levels"  # fallback


# ================================================================
# REGIME DETECTION (from historical H1 data slice)
# ================================================================
def get_regime_from_slice(m5_df, h1_df, h4_df, d1_df, current_idx):
    """
    Runs multi-TF regime detector using only candles up to current_idx.
    No lookahead — each TF only sees data confirmed before current_time.
    """
    try:
        def past_slice(df, n):
            if df is None or df.empty:
                return None
            s = df[df.index <= current_idx].tail(n)
            return s if len(s) > 50 else None

        h1_slice = past_slice(h1_df, H1_LOOKBACK)
        if h1_slice is not None and len(h1_slice) > 200:
            result = rd.predict(
                h1_df=h1_slice,
                m5_df=past_slice(m5_df, 500),
                h4_df=past_slice(h4_df, 250),
                d1_df=past_slice(d1_df, 250),
            )
            return result, rd.format_for_prompt(result)
    except Exception as e:
        print(f"[Regime] Error: {e}")
    return {"regime": "LOW_VOL_RANGE", "guidance": "", "confidence": None}, ""   # FIX Bug 3: "RANGING" is a v1 name — never matches any live regime label


# ================================================================
# PRE-FILTER (same as live bot)
# Gate 1: LOW_VOL_RANGE + conf<25% — degenerate input, no signal probability.
# Gate 2: Consecutive WAIT ×3+ in same regime + conf≥65%  — proven dead.
# ================================================================
def should_call_ai(regime_result, session):
    regime     = regime_result.get("regime", "LOW_VOL_RANGE")   # FIX Bug 3: was "RANGING" — v1 name, never matched
    confidence = regime_result.get("confidence") or 0.0

    # Gate 1: model has near-zero conviction in structureless market
    if regime == "LOW_VOL_RANGE" and confidence < 0.25:   # FIX Bug 3: was "RANGING" — gate never fired
        return False, "ranging_low_conf"

    from master_controls import GATE_MIN_CONFIDENCE, GATE_MAX_CONSECUTIVE_WAITS
    conf = regime_result.get("confidence") or 0
    
    # FIX: Implement missing reset conditions! If regime changes or confidence drops, reset the wait tracker.
    global _consecutive_waits, _last_wait_regime
    if regime != _last_wait_regime or confidence < 0.65:
        _consecutive_waits = 0
        _last_wait_regime = None

    if conf < GATE_MIN_CONFIDENCE:
        return False, "confidence_gate"

    # Gate 2: AI already said WAIT repeatedly in exact same regime state
    if (_consecutive_waits >= GATE_MAX_CONSECUTIVE_WAITS
            and regime == _last_wait_regime
            and confidence >= 0.65):
        return False, "consecutive_wait"

    return True, ""


# ================================================================
# BUILD MARKET CONTEXT STRING FROM HISTORICAL CANDLES
# Replaces data_extractor.get_live_market_data() for backtest
# ================================================================
def build_market_context(m5_df, h1_df, h4_df, d1_df, current_time):
    """
    Builds the same market context string as data_extractor,
    using only candles available at current_time (no lookahead).
    """
    def get_slice(df, n):
        if df is None or df.empty:
            return pd.DataFrame()
        return df[df.index <= current_time].tail(n)

    m5  = get_slice(m5_df,  20)
    h1  = get_slice(h1_df,  48)
    h4  = get_slice(h4_df,  20)
    d1  = get_slice(d1_df,  10)

    def fmt(df, label):
        if df.empty:
            return f"{label}: No data\n"
        lines = [f"\n{label} (last {len(df)} candles):"]
        for dt, row in df.iterrows():
            lines.append(
                f"  {dt.strftime('%Y-%m-%d %H:%M')} | "
                f"O:{row['open']:.2f} H:{row['high']:.2f} "
                f"L:{row['low']:.2f} C:{row['close']:.2f} "
                f"V:{int(row.get('volume', 0))}"
            )
        return "\n".join(lines)

    current_price = m5.iloc[-1]['close'] if not m5.empty else 0.0

    context = (
        f"Current Price : {current_price:.2f}\n"
        f"Simulated Time: {current_time.strftime('%Y-%m-%d %H:%M %Z')}\n"
        + fmt(m5,  "M5 Candles")
        + fmt(h1,  "H1 Candles")
        + fmt(h4,  "H4 Candles")
        + fmt(d1,  "D1 Candles")
    )
    return context, current_price


# ================================================================
# SIMULATED POST-MORTEM TRIGGER
# In live bot this fires at 5PM NY every day.
# In backtest it fires when simulated clock crosses 5PM.
# ================================================================
_last_postmortem_date = None

def check_simulated_postmortem(current_time, m5_df=None, h1_df=None):
    """
    FIX B4: accepts m5_df + h1_df so MissWish can slice the day's candles
    from the already-loaded DataFrames instead of trying to connect to MT5.
    """
    global _last_postmortem_date
    today = current_time.date()
    if (current_time.hour >= 17 and
            _last_postmortem_date != today and
            current_time.weekday() < 5):   # Mon-Fri only
        daily_post_mortem.check_and_run_if_needed(
            force=True,
            simulated_time=current_time,
            m5_df=m5_df,
            h1_df=h1_df)
        _last_postmortem_date = today


# ================================================================
# SIMULATED WISDOM BUILDER TRIGGER
# Fires every 5 simulated trading days
# ================================================================
_trading_day_count  = 0
_last_wisdom_date   = None

def check_simulated_wisdom(current_time):
    global _trading_day_count, _last_wisdom_date
    today = current_time.date()
    if today != _last_wisdom_date and current_time.weekday() < 5:
        _trading_day_count += 1
        _last_wisdom_date   = today
        if _trading_day_count % 5 == 0:
            try:
                wisdom_check(simulated_time=current_time)
            except Exception as e:
                print(f"[Wisdom] Error: {e}")


# ================================================================
# RESET ACTIVE TRADE
# ================================================================
def _reset_active_trade():
    global _active_trade
    _active_trade = {
        "ticket":           None,
        "type":             None,
        "entry":            0.0,
        "sl":               0.0,
        "tp":               0.0,
        "lot":              0.0,
        "remaining_lot":    0.0,
        "management_stage": 0,
        "partial_taken":    False,
        "partial_pnl":      0.0,
        "partial_cycle":    False,
        "spread_at_entry":  0.0,
        "regime_at_entry":  "",
        "bars_open":        0,   # GAP-04 FIX: candle counter for trade timeout
    }


# ================================================================
# BACKTEST TRADE MANAGEMENT CYCLE
# D-02/D-03/D-04/D-05/D-06/D-07/D-08/D-09 FIX
#
# Pure-Python replica of trade_manager.manage_trade() + hybrid dispatch.
# No MT5 — uses candle data instead of live tick. Modifies _active_trade
# in place every candle exactly as the live bot does every 5-minute cycle.
#
# Stage machine mirrors live bot exactly:
#   Stage 0 → Watching for 1R. Partial close fires here.
#             Partial: 50% lot closed at 1R price.
#             BE SL: deferred 1 candle (partial_cycle flag).
#   Stage 1 → BE set. Trailing begins on M5 swing.
#   Stage 2 → Trailing active. SL follows lowest low - $0.30 (BUY).
#
# Hybrid dispatch (fires BEFORE stage machine, same priority as live):
#   REVERSAL ≥ 0.65 → immediate full close, no stage machine.
#   BULL/BEAR > 0.70 → TRAIL_AGGRESSIVE ($0.10 buffer) or SKIP_PARTIAL.
#   COMPRESSION     → PARTIAL_75 (0.75R) or TIGHT_BE (0.5R).
# ================================================================

# Hybrid constants — must match trade_manager.py exactly
_BT_REVERSAL_THRESHOLD    = 0.65
_BT_TREND_THRESHOLD       = 0.70
_BT_COMPRESSION_REGIMES   = {"COMPRESSION", "LOW_VOL_RANGE", "LOW_VOLATILITY"}
_BT_TREND_REGIMES         = {"BULL_TREND", "BEAR_TREND"}
_BT_AGGRESSIVE_BUFFER     = 0.10
_BT_DEFAULT_BUFFER        = 0.30
_BT_COMPRESSION_PARTIAL_R = 0.75
_BT_COMPRESSION_PARTIAL_P = 0.75
_BT_COMPRESSION_BE_R      = 0.50


def _bt_manage_trade(current_candle: dict, current_time, m5_df, candle_idx: int,
                      regime_result: dict = None) -> str:
    """
    Backtest management cycle — called every candle while a trade is open.
    Mirrors manage_trade() in trade_manager.py but uses candle data not MT5.

    Returns:
        str — action taken this cycle (for logging)
    """
    global _active_trade, _stats

    if not _active_trade["ticket"]:
        return "NO_TRADE"

    # GAP-04 FIX: trade timeout — identical to simulate_trade_outcome()'s 200-candle cap.
    # Without this, a trade with no SL/TP hit and no REVERSAL could stay open indefinitely
    # across sessions, weekends, and corrupt multi-month backtest runs.
    # 288 candles = 24h of M5. Any trade still open at 24h is closed at market.
    MAX_BARS_OPEN = 288
    _active_trade["bars_open"] = _active_trade.get("bars_open", 0) + 1
    if _active_trade["bars_open"] >= MAX_BARS_OPEN:
        close_px_timeout = float(current_candle["close"])
        _force_close_trade(current_time, close_px_timeout,
                           f"Timeout: trade open {_active_trade['bars_open']} candles (>{MAX_BARS_OPEN})")
        return f"TIMEOUT after {MAX_BARS_OPEN} bars"

    direction    = _active_trade["type"]
    entry        = _active_trade["entry"]
    sl           = _active_trade["sl"]
    tp           = _active_trade["tp"]
    stage        = _active_trade["management_stage"]
    lot          = _active_trade["lot"]
    rem_lot      = _active_trade.get("remaining_lot", lot)
    partial_taken = _active_trade.get("partial_taken", False)
    spread       = _active_trade.get("spread_at_entry", 0.0)

    close_px = float(current_candle["close"])
    risk     = abs(entry - sl)
    if risk == 0:
        return "ZERO_RISK"

    profit_dist = (close_px - entry) if direction == "BUY" else (entry - close_px)
    profit_r    = profit_dist / risk

    # ── HYBRID DISPATCH ─────────────────────────────────────────────
    if regime_result:
        regime = regime_result.get("regime", "")
        conf   = float(regime_result.get("confidence") or 0)

        # 1. REVERSAL — immediate full close (D-05 FIX)
        if regime == "REVERSAL" and conf >= _BT_REVERSAL_THRESHOLD:
            _force_close_trade(current_time, close_px,
                               f"REVERSAL {conf:.0%} (hybrid)", spread=spread)
            return f"REVERSAL_CLOSE ({conf:.0%})"

        # 2. TREND — TRAIL_AGGRESSIVE or SKIP_PARTIAL (D-06/D-07 FIX)
        if regime in _BT_TREND_REGIMES and conf > _BT_TREND_THRESHOLD and stage >= 1:
            candles_slice = m5_df.iloc[max(0, candle_idx - 6):candle_idx]
            if len(candles_slice) >= 2:
                if direction == "BUY":
                    recent_low = float(candles_slice["low"].min())
                    cand_sl    = round(recent_low - _BT_AGGRESSIVE_BUFFER, 2)
                    if cand_sl > sl:
                        _active_trade["sl"]               = cand_sl
                        _active_trade["management_stage"] = 2
                        return f"TRAIL_AGGRESSIVE BUY SL→{cand_sl:.2f}"
                elif direction == "SELL":
                    recent_high = float(candles_slice["high"].max())
                    cand_sl     = round(recent_high + _BT_AGGRESSIVE_BUFFER, 2)
                    if cand_sl < sl:
                        _active_trade["sl"]               = cand_sl
                        _active_trade["management_stage"] = 2
                        return f"TRAIL_AGGRESSIVE SELL SL→{cand_sl:.2f}"

        # SKIP_PARTIAL: in TREND with high conf, skip the 50% close at 1R
        if regime in _BT_TREND_REGIMES and conf > _BT_TREND_THRESHOLD and stage == 0:
            if profit_r >= 1.0 and not partial_taken:
                _active_trade["partial_taken"] = True  # mark taken without closing
                return f"SKIP_PARTIAL ({regime} {conf:.0%}) — full position held"

        # 3. COMPRESSION — PARTIAL_75 or TIGHT_BE (D-08/D-09 FIX)
        if regime in _BT_COMPRESSION_REGIMES and stage == 0 and not partial_taken:
            if profit_r >= _BT_COMPRESSION_PARTIAL_R:
                # 75% partial at 0.75R
                partial_lot = round(rem_lot * _BT_COMPRESSION_PARTIAL_P, 2)
                partial_lot = max(0.01, partial_lot)
                p_price     = (entry + risk * _BT_COMPRESSION_PARTIAL_R) if direction == "BUY" \
                              else (entry - risk * _BT_COMPRESSION_PARTIAL_R)
                raw_p       = abs(p_price - entry)
                p_pips      = SpreadSimulator.apply_spread_cost(raw_p, spread)
                p_pnl       = calculate_pnl_dollars(p_pips, partial_lot)
                _active_trade["partial_taken"]  = True
                _active_trade["partial_pnl"]   += p_pnl
                _active_trade["remaining_lot"]  = round(rem_lot - partial_lot, 2)
                _active_trade["partial_cycle"]  = True
                _stats["total_pnl"] += p_pnl
                return f"PARTIAL_75 ({regime}) {partial_lot}L PnL:${p_pnl:+.2f}"

            if profit_r >= _BT_COMPRESSION_BE_R:
                # Early break-even
                _active_trade["sl"]               = entry
                _active_trade["management_stage"] = 1
                return f"TIGHT_BE ({regime}) SL→{entry:.2f}"

    # ── MECHANICAL STAGE MACHINE ─────────────────────────────────────

    # Stage 0: watch for 1R
    if stage == 0:
        if profit_r < 1.0:
            return f"WATCHING {profit_r:.2f}R"

        # --- Partial close at 1R (50%) ---
        if not partial_taken:
            partial_lot = round(rem_lot * 0.5, 2)
            partial_lot = max(0.01, partial_lot)
            p_price     = (entry + risk) if direction == "BUY" else (entry - risk)
            raw_p       = abs(p_price - entry)
            p_pips      = SpreadSimulator.apply_spread_cost(raw_p, spread)
            p_pnl       = calculate_pnl_dollars(p_pips, partial_lot)
            _active_trade["partial_taken"]  = True
            _active_trade["partial_pnl"]   += p_pnl
            _active_trade["remaining_lot"]  = round(rem_lot - partial_lot, 2)
            _active_trade["partial_cycle"]  = True   # defer BE 1 candle
            _stats["total_pnl"] += p_pnl
            print(f"  [BT_Mgmt] 1R hit! Partial {partial_lot}L @ {p_price:.2f} "
                  f"PnL:${p_pnl:+.2f}. BE deferred 1 cycle.")
            return f"PARTIAL_CLOSE 50% PnL:${p_pnl:+.2f}"

        # --- Deferred BE: fires the cycle after partial ---
        if _active_trade.get("partial_cycle"):
            _active_trade["sl"]               = entry
            _active_trade["management_stage"] = 1
            _active_trade["partial_cycle"]    = False
            print(f"  [BT_Mgmt] Break-even SL set → {entry:.2f}")
            return f"BE_SET SL→{entry:.2f}"

    # Stage 1/2: trail on M5 swing
    if stage >= 1:
        candles_slice = m5_df.iloc[max(0, candle_idx - 6):candle_idx]
        if len(candles_slice) < 2:
            return f"TRAIL stage={stage} not enough candles"

        if direction == "BUY":
            recent_low = float(candles_slice["low"].min())
            cand_sl    = round(recent_low - _BT_DEFAULT_BUFFER, 2)
            if cand_sl > sl:
                _active_trade["sl"]               = cand_sl
                _active_trade["management_stage"] = 2
                return f"TRAIL BUY SL→{cand_sl:.2f}"
        elif direction == "SELL":
            recent_high = float(candles_slice["high"].max())
            cand_sl     = round(recent_high + _BT_DEFAULT_BUFFER, 2)
            if cand_sl < sl:
                _active_trade["sl"]               = cand_sl
                _active_trade["management_stage"] = 2
                return f"TRAIL SELL SL→{cand_sl:.2f}"

        return f"TRAIL stage={stage} SL={sl:.2f} no new swing"

    return f"WATCHING {profit_r:.2f}R"



# ================================================================
# BACKTEST RISK CHECK — bypasses MT5, uses in-memory PnL state
# Replaces risk_manager.check_risk_clearance() in the hot candle loop.
# MT5-based version calls mt5.initialize()/shutdown() on EVERY candle
# (~200,000+ round-trips for a full run) — completely unusable in backtest.
# ================================================================
_bt_daily_start_pnl: float = 0.0
_bt_last_risk_day = None

def _bt_check_risk_clearance(current_time) -> bool:
    global _bt_daily_start_pnl, _bt_last_risk_day
    try:
        from master_controls import RISK_DAILY_DRAWDOWN_PCT as _ddpct
    except ImportError:
        _ddpct = 0.015
    today = current_time.date()
    if _bt_last_risk_day != today:
        _bt_daily_start_pnl = _stats["total_pnl"]
        _bt_last_risk_day   = today
    daily_pnl = _stats["total_pnl"] - _bt_daily_start_pnl
    initial   = float(os.getenv("BACKTEST_INITIAL_BALANCE", "10000"))
    cap       = initial * _ddpct
    if daily_pnl < -cap:
        print(f"[RiskManager] DAILY DRAWDOWN CAP HIT: "
              f"Loss ${abs(daily_pnl):.2f} > Cap ${cap:.2f}. Skipping {today}.")
        return False
    return True

# ================================================================
# MAIN BACKTEST LOOP
# ================================================================
def run_backtest(date_from=None, date_to=None, resume=False):
    global _active_trade, _stats

    print("=" * 65)
    print("  Antigravity Bridge — Backtest Engine")
    print("=" * 65)

    # ── Load all timeframe data ────────────────────────────────────
    print("[Data] Loading historical data...")
    from datetime import timedelta

    # Load data with lookback so D1/H4/H1 features
    # have enough warmup data before date_from.
    # Trades are still only evaluated from date_from.
    _D1_LOOKBACK_DAYS = 400   # 250 trading days + buffer
    _H4_LOOKBACK_DAYS = 60    # 250 H4 bars + buffer
    _H1_LOOKBACK_DAYS = 30    # 500 H1 bars + buffer
    _M5_LOOKBACK_DAYS = 5     # 500 M5 bars + buffer

    if date_from is not None:
        _data_from_m5 = date_from - timedelta(days=_M5_LOOKBACK_DAYS)
        _data_from_h1 = date_from - timedelta(days=_H1_LOOKBACK_DAYS)
        _data_from_h4 = date_from - timedelta(days=_H4_LOOKBACK_DAYS)
        _data_from_d1 = date_from - timedelta(days=_D1_LOOKBACK_DAYS)
    else:
        _data_from_m5 = None
        _data_from_h1 = None
        _data_from_h4 = None
        _data_from_d1 = None

    m5_df  = load_data("M5",  _data_from_m5, date_to)
    h1_df  = load_data("H1",  _data_from_h1, date_to)
    h4_df  = load_data("H4",  _data_from_h4, date_to)
    d1_df  = load_data("D1",  _data_from_d1, date_to)

    print(f"[Backtest] Data loaded with lookback:")
    print(f"  M5 : {len(m5_df):,} rows from {_data_from_m5.date() if _data_from_m5 is not None else 'start'}")
    print(f"  H1 : {len(h1_df):,} rows from {_data_from_h1.date() if _data_from_h1 is not None else 'start'}")
    print(f"  H4 : {len(h4_df):,} rows from {_data_from_h4.date() if _data_from_h4 is not None else 'start'}")
    print(f"  D1 : {len(d1_df):,} rows from {_data_from_d1.date() if _data_from_d1 is not None else 'start'}")

    if m5_df.empty:
        print("[ERROR] No M5 data found. Run data_downloader.py first.")
        return

    print(f"[Data] M5 : {len(m5_df):,} candles | "
          f"{m5_df.index[0].date()} → {m5_df.index[-1].date()}")
    print(f"[Data] H1 : {len(h1_df):,} candles")
    print(f"[Data] H4 : {len(h4_df):,} candles")
    print(f"[Data] D1 : {len(d1_df):,} candles")

    # ── Load or init tracker ───────────────────────────────────────
    tracker = load_tracker()

    # ── Init PnL tracker ──────────────────────────────────────────
    # INITIAL_BALANCE can be set via env var; default $10,000
    _initial_bal = float(os.getenv("BACKTEST_INITIAL_BALANCE", "10000"))
    _pnlt.init_tracker(initial_balance=_initial_bal)
    print(f"[PnLTracker] Initialised — starting balance ${_initial_bal:,.0f}")

    # Determine starting candle index
    start_idx = 0
    if resume and tracker.get("last_processed_candle"):
        last_dt = pd.to_datetime(tracker["last_processed_candle"])
        if last_dt.tzinfo is None:
            last_dt = NY_TZ.localize(last_dt)
        # Find the index AFTER the last processed candle
        positions = m5_df.index.searchsorted(last_dt, side='right')
        start_idx = int(positions)
        _stats = tracker.get("stats", _stats)
        print(f"\n[Resume] Resuming from candle {start_idx:,} "
              f"({last_dt.strftime('%Y-%m-%d %H:%M')})")
    else:
        tracker["started_at"] = datetime.now().isoformat()
        print(f"\n[Start] Running full backtest from candle 0")

    total_candles = len(m5_df)
    tracker["total_candles"] = total_candles
    print(f"[Start] {total_candles - start_idx:,} candles to process")
    print(f"[AI]    Provider: {AI_DISPLAY_NAME} ({AI_MODEL})\n")

    _NY_TZ = pytz.timezone("America/New_York")
    date_from_aware = (
        _NY_TZ.localize(date_from) if (date_from is not None and date_from.tzinfo is None)
        else date_from
    )

    # ── Main candle loop ───────────────────────────────────────────
    candle_idx = start_idx
    while candle_idx < total_candles:
        current_time  = m5_df.index[candle_idx]
        current_candle = m5_df.iloc[candle_idx]
        session        = get_session(current_time)

        # Skip lookback warmup period — evaluate only from
        # the requested date_from
        if date_from_aware is not None and current_time < date_from_aware:
            candle_idx += 1
            continue

        # ── Dead zone handling ──────────────────────────────────
        if not session:
            if _active_trade["ticket"]:
                # Overnight regime guard
                _run_overnight_guard(current_time, m5_df, h1_df, h4_df, d1_df, current_candle)
            else:
                _stats["skipped_dead_zone"] += 1
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1  # FIX: must increment before continue or loop is infinite
            continue

        # ── Active window ───────────────────────────────────────

        # Check if open trade was closed by SL/TP on this candle
        if _active_trade["ticket"]:
            _check_sl_tp_hit(current_candle, current_time, m5_df, candle_idx)
            if not _active_trade["ticket"]:
                _save_progress_if_needed(tracker, candle_idx, current_time)
                candle_idx += 1
                continue

        # GAP-01 FIX: open-position news partial close — mirrors live bot's
        # check_open_position_news_rule(). If high-impact news is 2–7 min away
        # and a trade is open, partially close 50% of remaining position.
        if _active_trade["ticket"]:
            news_today_for_partial = get_news_for_date(current_time.date())
            if news_today_for_partial:
                for _ev in news_today_for_partial:
                    if _ev.get("impact", "").upper() != "HIGH":
                        continue
                    _ev_time = _ev.get("time")
                    if _ev_time is None:
                        continue
                    try:
                        _diff = (_ev_time - current_time).total_seconds() / 60
                        if 2 <= _diff <= 7:
                            _tag = f"news_partial_{_ev_time.strftime('%H%M')}_{current_time.date()}"
                            if _active_trade.get("_last_news_partial_tag") != _tag:
                                _rem = _active_trade.get("remaining_lot") or _active_trade["lot"]
                                if _rem > 0.01:
                                    _p_lot   = max(0.01, round(_rem * 0.5, 2))
                                    _spread  = _active_trade.get("spread_at_entry", 0.0)
                                    _raw_p   = abs(float(current_candle["close"]) - _active_trade["entry"])
                                    _p_pips  = SpreadSimulator.apply_spread_cost(_raw_p, _spread)
                                    _p_pnl   = calculate_pnl_dollars(_p_pips, _p_lot)
                                    _active_trade["partial_pnl"]   = _active_trade.get("partial_pnl", 0.0) + _p_pnl
                                    _active_trade["remaining_lot"] = round(_rem - _p_lot, 2)
                                    _active_trade["partial_taken"] = True
                                    _active_trade["_last_news_partial_tag"] = _tag
                                    _stats["total_pnl"] += _p_pnl
                                    print(f"  [NewsPartialClose] {current_time.strftime('%H:%M')} "
                                          f"HI news in {_diff:.0f}min — closed {_p_lot}L PnL:${_p_pnl:+.2f}")
                            break
                    except Exception:
                        continue

        # D-02/D-03/D-04/D-05 FIX: per-candle hybrid management for open trade.
        # Mirrors the live bot's every-5-min manage_trade() cycle.
        # Regime detection runs before management so hybrid dispatch has context.
        # Skips AI entry logic entirely when a trade is open (D-10 intentional gap).
        if _active_trade["ticket"]:
            _mgmt_regime, _ = get_regime_from_slice(m5_df, h1_df, h4_df, d1_df, current_time)
            mgmt_action = _bt_manage_trade(
                current_candle=current_candle.to_dict(),
                current_time=current_time,
                m5_df=m5_df,
                candle_idx=candle_idx,
                regime_result=_mgmt_regime,
            )
            if mgmt_action:
                print(f"  [BT_Mgmt/{current_time.strftime('%H:%M')}] {mgmt_action}")
            # If REVERSAL_CLOSE fired inside _bt_manage_trade, trade is now closed.
            # Check again — if still open, skip AI and move to next candle.
            if not _active_trade["ticket"]:
                _save_progress_if_needed(tracker, candle_idx, current_time)
                candle_idx += 1
                continue
            # Trade still open: no AI entry call this cycle (D-10 by design)
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1
            continue

        # Friday rule
        if current_time.weekday() == 4 and current_time.hour >= 17:
            if _active_trade["ticket"]:
                _force_close_trade(current_time, current_candle['close'],
                                   "Friday 5PM close")
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1  # FIX: must increment before continue or loop is infinite
            continue

        # Risk clearance — use lightweight BT check (no MT5 round-trip per candle)
        if not _bt_check_risk_clearance(current_time):
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1
            continue
        if not risk_manager.check_consecutive_losses(SYMBOL):
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1
            continue

        # ── News gate — fetch once per trading day, cache in loop ──
        # Identical logic to live bot: HIGH=60min block, MEDIUM=30min, post=5min
        news_today = get_news_for_date(current_time.date())
        if is_in_news_window(current_time, news_today) and not _active_trade["ticket"]:
            _stats["skipped_news_window"] += 1
            print(f"  [NewsGate] {current_time.strftime('%H:%M')} blocked by news window.")
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1
            continue

        # Regime detection
        regime_result, regime_context = get_regime_from_slice(m5_df, h1_df, h4_df, d1_df, current_time)

        # ── Enrich with session-timing intelligence ───────────────
        if _PROFILER_ENABLED:
            try:
                # FIX Bug 6: pass m5 slice so compute_norm_vol can use
                # vol_ratio_20_100 / atr_percentile features.
                # compute_norm_vol() with no args always returns 1.0 (hardcoded
                # fallback) — vol adjustments on adaptive thresholds never fired.
                _m5_slice_vol = m5_df[m5_df.index <= current_time].tail(120)
                regime_result = enrich_regime_result(
                    regime_result, session, signal=None,
                    norm_vol=compute_norm_vol(m5_df=_m5_slice_vol))
            except Exception:
                pass

        # Pre-filter
        should_proceed, skip_reason = should_call_ai(regime_result, session)
        if not should_proceed:
            if skip_reason == "consecutive_wait":
                _stats["skipped_consecutive_wait"] += 1
            elif skip_reason == "confidence_gate":
                _stats["confidence_gate_blocked"] += 1
            else:
                _stats["skipped_prefilter"] += 1
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1  # FIX: must increment before continue or loop is infinite
            continue

        # Build market context
        market_context, current_price = build_market_context(
            m5_df, h1_df, h4_df, d1_df, current_time)

        # ── News block — real historical events, identical format to live bot ──
        news_block = format_news_for_prompt(news_today, current_time)

        # Build prompt (identical to live bot)
        past_lessons    = get_full_memory_context(market_context, current_time=current_time)
        current_state   = thought_logger.get_current_state()
        logic_framework = strategy_logic.get_analytical_framework()
        execution_rules = strategy_rules.get_execution_rules()

        # Session timing block (non-empty once session_profiler has profiles built)
        session_timing_block = ""
        if _PROFILER_ENABLED:
            try:
                session_timing_block = format_session_profile_for_prompt(regime_result)
            except Exception:
                pass

        # HMM transition forecast — same block as live bot
        transition_forecast_block = ""
        try:
            transition_forecast_block = rd.format_transition_forecast_for_prompt(regime_result)
        except Exception:
            pass

        # XGBoost signal quality — same block as live bot
        signal_quality_block = ""
        try:
            signal_quality_block = rd.format_model_signal_quality_for_prompt(regime_result)
        except Exception:
            pass

        # Regime router guidance block
        # GAP-03 FIX: call format_gate_context with same 3-arg signature as main_bot.py
        # Previous call passed 2-tuple (gate_result only) — live bot passes (allowed, reason, size_mult).
        router_block = ""
        try:
            gate_result  = rr.check_confidence_gate(regime_result)
            gate_allowed = gate_result[0] if isinstance(gate_result, (tuple, list)) else bool(gate_result)
            gate_reason  = gate_result[1] if isinstance(gate_result, (tuple, list)) and len(gate_result) > 1 else ""
            router_block = rr.format_gate_context(
                regime_result,
                (gate_allowed, gate_reason, 1.0),   # size_mult placeholder — matches main_bot.py
                None
            )
        except Exception:
            router_block = ""

        prompt = f"""
        You are a Multi-Disciplinary Gold Trader with a continuous memory.
        NOTE: This is a BACKTEST simulation. Simulated time: {current_time.strftime('%Y-%m-%d %H:%M %Z')}

        --- YOUR PREVIOUS INTERNAL MONOLOGUE ---
        Your last bias was: {current_state.get('current_bias', 'NEUTRAL')}
        Your ongoing thesis was: {current_state.get('active_thesis', 'Searching for setup.')}

        --- MACROECONOMIC NEWS TODAY ---
        {news_block}

        --- LIVE CALENDAR ---
        Today is strictly: {current_time.strftime("%A")}
        Current session: {session}

        --- REGIME DETECTOR ---
        {regime_context}

        --- REGIME ROUTER ---
        {router_block}

        --- SESSION TIMING INTELLIGENCE ---
        {session_timing_block if session_timing_block else "[Session profiles not yet built — will populate after first training run]"}

        --- HMM TRANSITION FORECAST (WHAT TYPICALLY COMES NEXT) ---
        {transition_forecast_block if transition_forecast_block else "[Transition forecast unavailable — model not trained yet]"}

        --- MODEL SIGNAL QUALITY (HOW CONFIDENT THE MODEL IS) ---
        {signal_quality_block if signal_quality_block else "[Signal quality unavailable — model not trained yet]"}

        --- MARKET DATA ---
        {market_context}

        --- ANALYTICAL FRAMEWORK (HOW TO THINK) ---
        {logic_framework}

        --- EXECUTION RULES (WHEN TO TRADE) ---
        {execution_rules}

        --- HISTORICAL LESSONS & HINDSIGHT ---
        {past_lessons}

        --- TASK ---
        1. Analyze the market using ICT, Classic TA, and Elliott Wave.
        2. Calculate the Confluence Score (0-3).
        3. Decide: BUY, SELL, WAIT, HOLD, or CLOSE.
        4. If a trade signal, provide entry, sl, tp.
        5. Output ONLY valid JSON as specified in the execution rules.
        """

        # ── Call AI (D-01 FIX: Claude Sonnet via call_ai, same as live) ─
        decision = {}
        try:
            raw_text   = call_ai(prompt=prompt)
            if not raw_text:
                raise RuntimeError("call_ai returned None")
            raw        = re.sub(r'```json\s*', '', raw_text)
            raw        = re.sub(r'```\s*',     '', raw)
            candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            for candidate in reversed(candidates):
                try:
                    decision = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
            if not decision:
                decision = json.loads(raw.strip())
            _stats["api_calls"] += 1
        except Exception as e:
            print(f"[AI] Error at {current_time}: {e}")
            _save_progress_if_needed(tracker, candle_idx, current_time)
            candle_idx += 1  # FIX: must increment before continue or loop is infinite
            continue

        signal = decision.get("signal", "WAIT").upper()
        if signal not in ["BUY", "SELL", "WAIT", "HOLD", "CLOSE"]:
            signal = "WAIT"

        # ── Count Claude WAIT ─────────────────────────────
        if signal in ("WAIT", "HOLD"):
            _stats["claude_wait_blocked"] += 1

        # Print cycle summary
        print(f"[{current_time.strftime('%Y-%m-%d %H:%M')}] {session:6s} | "
              f"{signal:5s} | Score: {decision.get('confluence_score', '?')} | "
              f"Trades: {_stats['total_trades']} | "
              f"PnL: ${_stats['total_pnl']:+.0f}")

        # ── CLOSE ───────────────────────────────────────────────
        if signal == "CLOSE" and _active_trade["ticket"]:
            _force_close_trade(
                current_time, current_price,
                f"AI reversal signal at {current_time.strftime('%Y-%m-%d %H:%M')}"
            )

        # ── BUY / SELL ──────────────────────────────────────────
        elif signal in ["BUY", "SELL"] and not _active_trade["ticket"]:

            # London gate — Rule 1a
            if session == "London" and decision.get("confluence_score", 0) < 3:
                _stats["london_blocked"] += 1
                print(f"  [LondonGate] Score {decision.get('confluence_score', 0)}/3 — blocked.")

            elif validate_trade_logic(decision, current_price):
                entry   = float(decision.get("entry", 0))
                sl      = float(decision.get("sl",    0))
                tp      = float(decision.get("tp",    0))

                # BUG-04 FIX: RULE 3 — SL/TP Regime Adjuster.
                # Live bot applies rr.adjust_sl_tp() before every execution.
                # Backtest was missing this entirely — same signal produced
                # different trade levels in live vs backtest.
                if entry > 0 and sl > 0 and tp > 0:
                    try:
                        _raw_sl, _raw_tp = sl, tp
                        adjust = rr.adjust_sl_tp(regime_result, entry, sl, tp, signal)
                        sl = adjust["sl"]
                        tp = adjust["tp"]
                        print(f"  [RegimeRouter] SL/TP ADJUSTED | "
                              f"SL: {_raw_sl:.2f}→{sl:.2f} ({adjust['sl_moved']:+.2f}pts) | "
                              f"TP: {_raw_tp:.2f}→{tp:.2f} ({adjust['tp_moved']:+.2f}pts) | "
                              f"RR: {adjust['rr']:.2f}")
                    except Exception as _adj_err:
                        pass  # non-fatal; use AI's levels if adjuster fails

                sl_dist = abs(entry - sl)

                # ── Regime router: dual confirmation gate + size multiplier ──
                size_multiplier = 1.0
                dual_pass       = True
                dual_reason     = ""   # FIX: pre-init so log_trade never sees NameError
                try:
                    dual_allowed, dual_reason, dual_size = rr.check_dual_confirmation(
                        regime_result, signal)
                    dual_pass       = dual_allowed
                    size_multiplier = dual_size
                    if not dual_pass:
                        _stats["dual_gate_blocked"] += 1
                        print(f"  [DualGate] BLOCKED — {dual_reason}")
                except Exception:
                    pass   # regime router non-fatal

                if not dual_pass:
                    pass   # already logged above, skip to next candle

                else:
                    # ── XU-L Meta gate ────────────────────────────────────
                    meta_pass   = True
                    final_size  = size_multiplier
                    meta_result = {}

                    if _META_ENABLED:
                        try:
                            # Build minimal market_features for meta model
                            mf_slice = m5_df[m5_df.index <= current_time].tail(100)
                            primary_signal = {
                                "signal":           signal,
                                "regime":           regime_result.get("regime", "UNKNOWN"),
                                "confidence":       regime_result.get("confidence", 0.5),
                                "confluence_score": decision.get("confluence_score", 0),
                            }
                            meta_result = ml.predict_meta(
                                primary_signal, regime_result, mf_slice)
                            meta_pass   = meta_result.get("should_trade", True)

                            if not meta_pass:
                                _stats["meta_gate_blocked"] += 1
                                print(f"  [MetaGate] BLOCKED — "
                                      f"p={meta_result.get('meta_prob',0):.2f} < "
                                      f"thresh={meta_result.get('threshold_used',0):.2f}")
                                # Log blocked trade for threshold calibration
                                norm_vol = (regime_result.get("session_profile", {})
                                            .get("norm_vol", 1.0))
                                ml.log_blocked_trade(
                                    signal=signal,
                                    regime=regime_result.get("regime", "UNKNOWN"),
                                    session=session,
                                    meta_prob=meta_result.get("meta_prob", 0),
                                    threshold_used=meta_result.get(
                                        "threshold_used", ml.META_MIN_THRESHOLD),
                                    norm_vol=norm_vol,
                                    shap_reason=meta_result.get("shap_reason", ""),
                                )
                            else:
                                # Combined size: min(meta size, regime size)
                                meta_size  = meta_result.get("size", 1.0)
                                final_size = min(meta_size, size_multiplier)
                                print(f"  [MetaGate] PASS — "
                                      f"p={meta_result.get('meta_prob',0):.2f} | "
                                      f"size={final_size:.2f} "
                                      f"(meta={meta_size:.2f} regime={size_multiplier:.2f})")
                        except Exception as me:
                            print(f"  [MetaGate] Error ({me}) — proceeding without meta filter.")

                    if meta_pass:
                        # BUG-05 FIX: pass news_today so high-impact news reduces
                        # position size by 50%, identical to live bot behaviour.
                        # Live bot passes news_data (string); risk_manager expects a
                        # string — convert the list-of-dicts format used by backtest.
                        _news_str = ""
                        if news_today:
                            try:
                                _news_str = " ".join(
                                    f"{e.get('name','')} [{e.get('impact','')}]"
                                    for e in news_today if e.get("impact","").upper() == "HIGH"
                                )
                            except Exception:
                                pass
                        lot = risk_manager.calculate_position_size(SYMBOL, sl_dist,
                                                                   news_data=_news_str)

                        # Apply size multiplier to lot
                        if lot and lot > 0 and final_size < 1.0:
                            import math
                            lot = max(0.01, round(lot * final_size, 2))

                        if lot and lot > 0:
                            # BUG-06 FIX: Spread gate — live bot calls risk_manager.check_spread()
                            # and REJECTS the trade if spread > GATE_MAX_SPREAD_DOLLARS ($1.50).
                            # Backtest was missing this gate entirely — trades were executing at
                            # news-spike spreads of $5–$12 that live bot would have rejected.
                            _pre_spread = _spread_sim.get_spread(session, current_time, news_today)
                            try:
                                from master_controls import GATE_MAX_SPREAD_DOLLARS as _MAX_SPR
                            except ImportError:
                                _MAX_SPR = 1.50
                            if _pre_spread > _MAX_SPR:
                                _stats.setdefault("spread_blocked", 0)
                                _stats["spread_blocked"] += 1
                                print(f"  [SpreadGate] BLOCKED — spread ${_pre_spread:.2f} > "
                                      f"${_MAX_SPR:.2f} limit (news spike?)")
                                _save_progress_if_needed(tracker, candle_idx, current_time)
                                candle_idx += 1
                                continue

                            ticket = f"{TICKET_PREFIX}{current_time.strftime('%Y%m%d%H%M%S')}"
                            _stats["total_trades"] += 1

                            # ── Spread + Slippage: realistic fill model ───────────────
                            # Spread   = broker bid-ask gap (always paid, session + news-spike aware)
                            # Slippage = extra adverse fill beyond quoted price (market impact)
                            # Both always move against the trade; slippage is 0.3–1.5 pts normally,
                            # up to 3 pts during news — the single largest source of live vs backtest gap.
                            trade_spread    = _pre_spread   # reuse the gate check value (same candle)
                            trade_slippage  = _spread_sim.get_slippage(session, current_time, news_today)
                            filled_entry    = _spread_sim.adjusted_entry(
                                                  signal, entry, trade_spread, trade_slippage)
                            spread_cost_usd   = trade_spread   * lot * 100
                            slippage_cost_usd = trade_slippage * lot * 100
                            _stats["total_spread_cost"]   += spread_cost_usd
                            _stats["total_slippage_cost"] += slippage_cost_usd

                            print(f"  ▶ TRADE #{ticket} | {signal} | "
                                  f"Entry:{entry:.2f}→fill:{filled_entry:.2f} "
                                  f"(slip:{trade_slippage:.2f}) "
                                  f"SL:{sl:.2f} TP:{tp:.2f} | "
                                  f"Lot:{lot} | Size:{final_size:.2f}x | "
                                  f"Spread:${trade_spread:.2f} Slip:${trade_slippage:.2f} "
                                  f"(-${spread_cost_usd + slippage_cost_usd:.2f} total drag)")

                            # D-02/D-03/D-04 FIX: open trade in _active_trade.
                            # simulate_trade_outcome() is REMOVED — management now
                            # happens candle-by-candle via _bt_manage_trade(), which
                            # fires partial close, break-even, trailing stop, and hybrid
                            # dispatch (REVERSAL/TRAIL_AGGRESSIVE/SKIP_PARTIAL/TIGHT_BE)
                            # identically to the live bot's every-5-min cycle.
                            # Close is detected by _check_sl_tp_hit() on each candle.
                            _active_trade.update({
                                "ticket":           ticket,
                                "type":             signal,
                                "entry":            filled_entry,
                                "sl":               sl,
                                "tp":               tp,
                                "lot":              lot,
                                "remaining_lot":    lot,
                                "management_stage": 0,
                                "partial_taken":    False,
                                "partial_pnl":      0.0,
                                "partial_cycle":    False,
                                "spread_at_entry":  trade_spread,
                                "regime_at_entry":  regime_result.get("regime", ""),
                                "bars_open":        0,   # GAP-04 FIX: timeout counter
                                # Stash for meta wisdom lesson when trade closes
                                "_meta_result":     meta_result,
                                "_dual_reason":     dual_reason,
                                "_norm_vol":        regime_result.get("session_profile", {}).get("norm_vol", 1.0),
                            })

                            # Log trade open to memory (outcome filled at close by _check_sl_tp_hit)
                            # BUG-03 FIX: removed signal_alignment= (not in log_trade signature —
                            # caused TypeError on every trade). Added session=session (was missing,
                            # caused all backtest trades to log with empty session field).
                            memory_manager.log_trade(
                                ticket=ticket, signal=signal,
                                reasoning=decision.get("reasoning", ""),
                                entry_price=entry, sl=sl, tp=tp,
                                conf_score=decision.get("confluence_score", 0),
                                ict_logic=decision.get("analysis_ict", ""),
                                classic_logic=decision.get("analysis_classic", ""),
                                elliott_logic=decision.get("analysis_elliott", ""),
                                regime=regime_result.get("regime", ""),
                                regime_confidence=float(regime_result.get("confidence") or 0.0),
                                session=session,
                                meta_prob=float(meta_result.get("meta_prob", 0.0)) if meta_result else None,
                                gate_decisions={
                                    "dual_gate":  dual_reason,
                                    "meta_gate":  meta_result.get("shap_reason", "") if meta_result else "meta_disabled",
                                    "meta_prob":  float(meta_result.get("meta_prob", 0.0)) if meta_result else 0.0,
                                    "signal_aligned": (
                                        (regime_result.get("regime", "") == "BULL_TREND" and signal == "BUY") or
                                        (regime_result.get("regime", "") == "BEAR_TREND" and signal == "SELL")
                                    ),
                                },
                            )

                            # Episode start (close called by _check_sl_tp_hit / _force_close_trade)
                            try:
                                er.start_episode(
                                    ticket=ticket, entry_price=entry,
                                    sl_price=sl, tp_price=tp,
                                    direction=signal, lot_size=lot,
                                    regime=regime_result.get("regime", "UNKNOWN"),
                                    session=session,
                                )
                            except Exception as ep_err:
                                print(f"  [Episode] start_episode error: {ep_err}")

        # ── Update thought state ─────────────────────────────────
            else:
                # validate_trade_logic failed — diagnose why
                _failure = _diagnose_trade_failure(
                               decision, current_price)
                if _failure == "hallucination":
                    _stats["hallucination_blocked"] += 1
                    print(f"  [HallucinationGate] BLOCKED — "
                          f"entry {decision.get('entry',0):.2f} "
                          f"vs live {current_price:.2f} "
                          f"(>{20:.0f}pt deviation)")
                elif _failure == "rr":
                    _stats["rr_blocked"] += 1
                    _e = float(decision.get("entry", 0))
                    _s = float(decision.get("sl",    0))
                    _t = float(decision.get("tp",    0))
                    _d = abs(_e - _s)
                    _r = (abs(_t - _e) / _d) if _d > 0 else 0
                    print(f"  [RRGate] BLOCKED — "
                          f"RR {_r:.2f} < 1.5 minimum")
                else:
                    print(f"  [ValidateGate] BLOCKED — "
                          f"invalid price levels "
                          f"(entry/sl/tp arrangement)")

        thought_logger.update_state(
            decision.get("bias",      "NEUTRAL"),
            decision.get("thesis",    "No active thesis."),
            decision.get("reasoning", ""),
            bool(_active_trade["ticket"])
        )

        # ── Simulated post-mortem + wisdom triggers ──────────────
        check_simulated_postmortem(current_time, m5_df=m5_df, h1_df=h1_df)
        check_simulated_wisdom(current_time)

        # ── Update consecutive WAIT tracker
        global _consecutive_waits, _last_wait_regime
        if signal in ("WAIT", "HOLD"):
            _consecutive_waits += 1
            _last_wait_regime   = regime_result.get("regime", "UNKNOWN")
        else:
            _consecutive_waits = 0
            _last_wait_regime  = None

        _save_progress_if_needed(tracker, candle_idx, current_time)
        candle_idx += 1

    # ── Backtest complete ──────────────────────────────────────────
    _print_final_report()
    save_tracker(tracker)


# ================================================================
# OVERNIGHT GUARD (backtest version — same logic, no MT5)
# ================================================================
def _run_overnight_guard(current_time, m5_df, h1_df, h4_df, d1_df, current_candle):
    """
    Regime guard for open trades in dead zones and off-session gaps — no API call.
    GAP-02 FIX: CLOSE_PARTIAL now implemented (was a 'no lot splitting yet' stub).
    Mirrors live bot's run_overnight_trade_guard() logic:
      REVERSAL ≥ 0.65  → full close
      CLOSE_PARTIAL    → 50% partial close (banks half at current price)
      BULL/BEAR counter-direction ≥ 0.80 → full close
    """
    if not _active_trade["ticket"]:
        return

    regime_result, _ = get_regime_from_slice(m5_df, h1_df, h4_df, d1_df, current_time)
    regime     = regime_result.get("regime", "LOW_VOL_RANGE")
    confidence = regime_result.get("confidence") or 0.0
    direction  = _active_trade["type"]
    close_px   = float(current_candle["close"])

    # 1. REVERSAL full close (threshold matches live hybrid dispatch)
    if regime == "REVERSAL" and confidence >= 0.65:
        print(f"  [OvernightGuard] {current_time.date()} REVERSAL {confidence:.0%} — full close.")
        _force_close_trade(current_time, close_px,
                           f"Overnight guard: REVERSAL (conf {confidence:.2f})")
        return

    # 2. Regime router CLOSE_PARTIAL — GAP-02 FIX: now actually executes 50% partial
    try:
        adapt = rr.adapt_live_trade(regime_result, _active_trade)
        if adapt["action"] == "CLOSE_PARTIAL":
            rem_lot = _active_trade.get("remaining_lot") or _active_trade["lot"]
            if rem_lot > 0.01:
                partial_lot = round(rem_lot * 0.5, 2)
                partial_lot = max(0.01, partial_lot)
                spread      = _active_trade.get("spread_at_entry", 0.0)
                entry       = _active_trade["entry"]
                raw_p       = abs(close_px - entry)
                p_pips      = SpreadSimulator.apply_spread_cost(raw_p, spread)
                p_pnl       = calculate_pnl_dollars(p_pips, partial_lot)
                _active_trade["partial_pnl"]  = _active_trade.get("partial_pnl", 0.0) + p_pnl
                _active_trade["remaining_lot"] = round(rem_lot - partial_lot, 2)
                _active_trade["partial_taken"] = True
                _stats["total_pnl"] += p_pnl
                print(f"  [OvernightGuard] CLOSE_PARTIAL {partial_lot}L @ {close_px:.2f} "
                      f"PnL:${p_pnl:+.2f} | {adapt['reason'][:80]}")
    except Exception as e:
        pass  # adapt_live_trade non-fatal

    # 3. Counter-direction BULL/BEAR trend full close (matches live overnight logic)
    if regime in ("BULL_TREND", "BEAR_TREND") and confidence >= 0.80:
        trend_against = (
            (direction == "BUY"  and regime == "BEAR_TREND") or
            (direction == "SELL" and regime == "BULL_TREND")
        )
        if trend_against:
            print(f"  [OvernightGuard] {regime} counter-direction to {direction} "
                  f"(conf {confidence:.0%}) — full close.")
            _force_close_trade(current_time, close_px,
                               f"Overnight guard: {regime} counter-direction (conf {confidence:.2f})")


def _check_sl_tp_hit(candle, candle_time, m5_df, candle_idx):
    """
    Checks if current candle's high/low hit SL or TP of the open trade.
    D-02/D-03 FIX: uses remaining_lot (after partial close) not full lot,
    and adds partial_pnl already banked to compute total trade PnL.
    """
    if not _active_trade["ticket"]:
        return

    direction = _active_trade["type"]
    sl        = _active_trade["sl"]
    tp        = _active_trade["tp"]
    entry     = _active_trade["entry"]
    # D-02 FIX: use remaining_lot — partial has already been banked
    lot       = _active_trade.get("remaining_lot") or _active_trade["lot"]
    high      = candle['high']
    low       = candle['low']

    hit = None
    close_price = None

    if direction == "BUY":
        if low <= sl:
            hit, close_price = "LOSS", sl
        elif high >= tp:
            hit, close_price = "WIN", tp
    elif direction == "SELL":
        if high >= sl:
            hit, close_price = "LOSS", sl
        elif low <= tp:
            hit, close_price = "WIN", tp

    if hit:
        session       = get_session(candle_time)
        stored_spread = _active_trade.get("spread_at_entry", 0.0)
        spread        = stored_spread if stored_spread else _spread_sim.get_spread(session, candle_time)
        raw_pips      = (close_price - entry) if direction == "BUY" else (entry - close_price)
        pnl_pips      = SpreadSimulator.apply_spread_cost(raw_pips, spread)
        pnl_remainder = calculate_pnl_dollars(pnl_pips, lot)

        # D-02 FIX: total PnL = partial already banked + remainder just closed
        partial_pnl = _active_trade.get("partial_pnl", 0.0)
        total_pnl   = partial_pnl + pnl_remainder

        # Only add remainder to running total (partial was added when it fired)
        _stats["total_pnl"]         += pnl_remainder
        _stats["total_spread_cost"] += spread * lot * 100

        # Win/loss determined by TOTAL PnL (not just remainder)
        final_result = "WIN" if total_pnl > 0 else "LOSS"
        if final_result == "WIN":
            _stats["wins"] += 1
        else:
            _stats["losses"] += 1

        print(f"  [TradeClose] {candle_time.strftime('%Y-%m-%d %H:%M')} "
              f"#{_active_trade['ticket']} → {final_result} | "
              f"Partial:${partial_pnl:+.2f} + Remainder:${pnl_remainder:+.2f} "
              f"= Total:${total_pnl:+.2f}")

        memory_manager.update_final_review(
            _active_trade["ticket"], final_result,
            f"SL/TP hit {hit}. Total PnL:${total_pnl:+.2f} "
            f"(partial:${partial_pnl:+.2f} + rem:${pnl_remainder:+.2f})"
        )
        risk_manager.update_trade_result(final_result)
        er.close_episode(_active_trade["ticket"], final_result, total_pnl)
        _pnlt.record_trade(
            ticket    = _active_trade["ticket"],
            pnl       = total_pnl,
            result    = final_result,
            regime    = _active_trade.get("regime_at_entry", ""),
            session   = session or "",
            signal    = _active_trade.get("type", ""),
            timestamp = str(candle_time),
        )

        # Meta wisdom lesson — fires at close (needs actual outcome)
        _stored_meta   = _active_trade.get("_meta_result")
        _stored_signal = _active_trade.get("type", "")
        _stored_regime = _active_trade.get("regime_at_entry", "UNKNOWN")
        _stored_vol    = _active_trade.get("_norm_vol", 1.0)
        if _META_ENABLED and _stored_meta:
            try:
                ml.generate_wisdom_lesson(
                    ticket=_active_trade["ticket"],
                    meta_prob=_stored_meta.get("meta_prob", 0.5),
                    actual_outcome=final_result,
                    regime=_stored_regime,
                    session=session or "",
                    signal=_stored_signal,
                    shap_reason=_stored_meta.get("shap_reason", ""),
                    threshold_used=_stored_meta.get("threshold_used", ml.META_MIN_THRESHOLD),
                    norm_vol=_stored_vol,
                    trade_taken=True,
                )
            except Exception:
                pass

        _reset_active_trade()


def _force_close_trade(close_time, close_price, reason, spread=0.0):
    """Force closes the active trade at a specific price (Friday, REVERSAL, AI CLOSE)."""
    if not _active_trade["ticket"]:
        return
    direction    = _active_trade["type"]
    entry        = _active_trade["entry"]
    # D-02 FIX: use remaining_lot — partial already banked
    lot          = _active_trade.get("remaining_lot") or _active_trade["lot"]
    raw_pips     = (close_price - entry) if direction == "BUY" else (entry - close_price)
    pnl_pips     = SpreadSimulator.apply_spread_cost(raw_pips, spread)
    pnl_rem      = calculate_pnl_dollars(pnl_pips, lot)
    partial_pnl  = _active_trade.get("partial_pnl", 0.0)
    total_pnl    = partial_pnl + pnl_rem
    result       = "WIN" if total_pnl > 0 else "LOSS"

    if result == "WIN":
        _stats["wins"] += 1
    else:
        _stats["losses"] += 1
    _stats["total_pnl"]         += pnl_rem   # partial already added when it fired
    _stats["total_spread_cost"] += spread * lot * 100

    print(f"  [ForceClose] {close_time} #{_active_trade['ticket']} {reason} | "
          f"Partial:${partial_pnl:+.2f} + Rem:${pnl_rem:+.2f} = Total:${total_pnl:+.2f}")

    memory_manager.update_final_review(
        _active_trade["ticket"], "CLOSED_BY_AI",
        f"Force close: {reason}. Total PnL: ${total_pnl:+.2f}"
    )
    risk_manager.update_trade_result("CLOSED_BY_AI")
    er.close_episode(_active_trade["ticket"], "CLOSED_BY_AI", total_pnl)
    _pnlt.record_trade(
        ticket    = _active_trade["ticket"],
        pnl       = total_pnl,
        result    = result,
        regime    = _active_trade.get("regime_at_entry", ""),
        session   = "",
        signal    = direction,
        timestamp = str(close_time),
    )

    # Meta wisdom lesson at close
    _stored_meta   = _active_trade.get("_meta_result")
    _stored_regime = _active_trade.get("regime_at_entry", "UNKNOWN")
    _stored_vol    = _active_trade.get("_norm_vol", 1.0)
    if _META_ENABLED and _stored_meta:
        try:
            ml.generate_wisdom_lesson(
                ticket=_active_trade["ticket"],
                meta_prob=_stored_meta.get("meta_prob", 0.5),
                actual_outcome=result,
                regime=_stored_regime,
                session="",
                signal=direction,
                shap_reason=_stored_meta.get("shap_reason", ""),
                threshold_used=_stored_meta.get("threshold_used", ml.META_MIN_THRESHOLD),
                norm_vol=_stored_vol,
                trade_taken=True,
            )
        except Exception:
            pass

    _reset_active_trade()


# ================================================================
# PROGRESS SAVER
# ================================================================
def _save_progress_if_needed(tracker, candle_idx, current_time):
    tracker["candles_processed"]     = candle_idx + 1
    tracker["last_processed_candle"] = str(current_time)
    if candle_idx % SAVE_EVERY_N == 0:
        save_tracker(tracker)
    # Heartbeat every 1000 candles — visible during dead-zone / warmup
    # periods so the operator can confirm the engine is not hung.
    if candle_idx % 1000 == 0 and candle_idx > 0:
        print(f"  [Progress] Candle {candle_idx:,} / {tracker.get('total_candles', '?'):,} "
              f"| {current_time.strftime('%Y-%m-%d %H:%M')} "
              f"| Trades: {_stats['total_trades']} | PnL: ${_stats['total_pnl']:+.0f}")
    # Print PnL progress line every 500 candles so the operator
    # can track equity, drawdown and win rate mid-run
    if candle_idx % 500 == 0 and candle_idx > 0:
        _pnlt.print_progress_line()


# ================================================================
# FINAL REPORT
# ================================================================
def _print_final_report():
    total  = _stats["total_trades"]
    wins   = _stats["wins"]
    losses = _stats["losses"]
    pnl    = _stats["total_pnl"]
    wr     = (wins / total * 100) if total > 0 else 0

    signals_generated = (
        total
        + _stats["london_blocked"]
        + _stats["dual_gate_blocked"]
        + _stats["meta_gate_blocked"]
        + _stats["confidence_gate_blocked"]
        + _stats["claude_wait_blocked"]
        + _stats["hallucination_blocked"]
        + _stats["rr_blocked"]
        + _stats.get("spread_blocked", 0)
    )

    # Total cycles evaluated by AI (reached Claude call)
    ai_cycles = (
        _stats["api_calls"]
        + _stats["claude_wait_blocked"]
    )

    # -- Helper: percentage formatter ------------------
    def _pct(n, total):
        return f"{n/total*100:.1f}%" if total > 0 else "0.0%"

    total_cycles = (
        signals_generated
        + _stats["skipped_dead_zone"]
        + _stats["skipped_news_window"]
        + _stats["skipped_prefilter"]
        + _stats["skipped_consecutive_wait"]
        + _stats["confidence_gate_blocked"]
    )

    print()
    print(f"  -- Gate Funnel (full cycle breakdown) ------")
    print(f"  Total cycles run       : {total_cycles:,}")
    print(f"  Dead zone skips        : "
          f"{_stats['skipped_dead_zone']:,}"
          f"  ({_pct(_stats['skipped_dead_zone'], total_cycles)})")
    print(f"  News blocked           : "
          f"{_stats['skipped_news_window']:,}"
          f"  ({_pct(_stats['skipped_news_window'], total_cycles)})")
    print(f"  Pre-filter (HIGH_VOL)  : "
          f"{_stats['skipped_prefilter']:,}"
          f"  ({_pct(_stats['skipped_prefilter'], total_cycles)})")
    print(f"  Consec-WAIT skips      : "
          f"{_stats['skipped_consecutive_wait']:,}"
          f"  ({_pct(_stats['skipped_consecutive_wait'], total_cycles)})")
    print()
    print(f"  -- Cycles that reached AI ------------------")
    print(f"  AI cycles              : {ai_cycles:,}")
    print(f"  Confidence gate block  : "
          f"{_stats['confidence_gate_blocked']:,}"
          f"  ({_pct(_stats['confidence_gate_blocked'], ai_cycles)})"
          f"  ? regime model too uncertain")
    print(f"  Claude said WAIT       : "
          f"{_stats['claude_wait_blocked']:,}"
          f"  ({_pct(_stats['claude_wait_blocked'], ai_cycles)})"
          f"  ? AI found no edge")
    print()
    print(f"  -- Claude said BUY/SELL --------------------")
    directional = (
        total
        + _stats["london_blocked"]
        + _stats["dual_gate_blocked"]
        + _stats["meta_gate_blocked"]
        + _stats.get("spread_blocked", 0)
        + _stats["hallucination_blocked"]
        + _stats["rr_blocked"]
    )
    print(f"  Directional signals    : {directional:,}")
    print(f"  London gate blocked    : "
          f"{_stats['london_blocked']:,}"
          f"  ({_pct(_stats['london_blocked'], directional)})")
    print(f"  Dual gate blocked      : "
          f"{_stats['dual_gate_blocked']:,}"
          f"  ({_pct(_stats['dual_gate_blocked'], directional)})")
    print(f"  Meta gate blocked      : "
          f"{_stats['meta_gate_blocked']:,}"
          f"  ({_pct(_stats['meta_gate_blocked'], directional)})"
          f"  ? meta-labeller confidence too low")
    print(f"  Hallucination blocked  : "
          f"{_stats['hallucination_blocked']:,}"
          f"  ({_pct(_stats['hallucination_blocked'], directional)})"
          f"  ? Claude entry >20pt from live price")
    print(f"  RR gate blocked        : "
          f"{_stats['rr_blocked']:,}"
          f"  ({_pct(_stats['rr_blocked'], directional)})"
          f"  ? risk:reward below 1.5 minimum")
    print(f"  Spread blocked         : "
          f"{_stats.get('spread_blocked',0):,}"
          f"  ({_pct(_stats.get('spread_blocked',0), directional)})")
    print()
    print(f"  -- RESULT ----------------------------------")
    print(f"  Trades executed        : {total:,}")
    print(f"  Win rate               : {wr:.1f}%")
    print(f"  Total PnL              : ${pnl:+,.2f}")
    print()
    gate_counts = {
        "Confidence gate"  : _stats["confidence_gate_blocked"],
        "Claude WAIT"      : _stats["claude_wait_blocked"],
        "Meta gate"        : _stats["meta_gate_blocked"],
        "London gate"      : _stats["london_blocked"],
        "Dual gate"        : _stats["dual_gate_blocked"],
        "Hallucination"    : _stats["hallucination_blocked"],
        "RR gate"          : _stats["rr_blocked"],
        "Spread gate"      : _stats.get("spread_blocked", 0),
        "News block"       : _stats["skipped_news_window"],
        "Consec-WAIT"      : _stats["skipped_consecutive_wait"],
    }
    dominant = max(gate_counts, key=gate_counts.get)
    dominant_count = gate_counts[dominant]
    print(f"  ?  DOMINANT BLOCKER: {dominant} "
          f"({dominant_count:,} blocks)")
    print(f"     ? This gate is your primary friction point.")
    print(f"     ? Review this gate first in debugging.")
    print()
    print(f"  -- Engine Stats ----------------------------")
    print(f"  API calls made    : {_stats['api_calls']:,}")
    print("=" * 65)

    # ── Extended PnL analytics ────────────────────────────────────
    _pnlt.print_full_report()
    _pnlt.print_equity_curve_summary(buckets=10)

    # ── Save PnL stats to Data/Backtest/pnl_stats.json ────────────
    from paths import PNL_STATS_PATH
    _pnlt.save(PNL_STATS_PATH)


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Antigravity Bridge Backtest Engine")
    parser.add_argument('--from', dest='date_from', type=str, default=None,
                        help='Start date YYYY-MM-DD (default: earliest available)')
    parser.add_argument('--to', dest='date_to', type=str, default=None,
                        help='End date YYYY-MM-DD (default: latest available)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint in backtest_tracker.json')
    args = parser.parse_args()

    date_from = datetime.strptime(args.date_from, '%Y-%m-%d') if args.date_from else None
    date_to   = datetime.strptime(args.date_to,   '%Y-%m-%d') if args.date_to   else None

    run_backtest(date_from=date_from, date_to=date_to, resume=args.resume)
