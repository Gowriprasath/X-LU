"""
main_bot.py — Gold AI Bridge | Gold AI Trading Bot v18
=======================================================
Entry point. Run this file to start live trading.
Usage: python main_bot.py

SESSION GATING (NY Time):
    Asian  : 19:00 – 00:00  — Liquidity mapping + entries
    London : 02:00 – 05:00  — Thesis building + rare 3/3 entries (regime-gated)
    NY     : 07:00 – 12:00  — Primary execution window
    Dead   : 00:00–02:00, 05:00–07:00, 12:00–19:00 — No AI calls

OVERNIGHT TRADE MANAGEMENT (when trade open outside active windows):
    Regime detector (local ML, zero API cost) guards the trade.
    REVERSAL         → partial or full close via trade_executor
    TRENDING against → full close
    Everything else  → trade_manager handles SL/BE/trail mechanically

REGIME ROUTER (new v12):
    Four-layer system between regime sensor and Claude:
    1. Confidence gate  — skip Claude if regime confidence < 40%
    2. Dual confirmation — both regime + Claude must agree (size scaling)
    3. SL/TP adjuster   — regime volatility multipliers on Claude's levels
    4. Live trade adapter — regime shifts adjust SL/TP mid-trade (zero API)
"""

import sys
import os
import time
import json
import pytz
import re
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
# FIX: removed 'from google import genai' — all AI calls go through ai_client.py
import MetaTrader5 as mt5
import pandas as pd

# --- Dynamic Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "Python Files"))
sys.path.append(os.path.join(current_dir, "Strategy"))
sys.path.append(os.path.join(current_dir, "Memory"))
sys.path.append(os.path.join(current_dir, "Integration"))
_stability_dir = os.path.join(current_dir, "Stability")
if _stability_dir not in sys.path:
    sys.path.insert(0, _stability_dir)

import data_extractor
import risk_manager
import trade_executor
import strategy_rules
from strategy_rules import get_adaptive_timing_rule as _adaptive_timing_rule
import strategy_logic
import strategy_selector   # ISSUE 3 FIX: regime-to-strategy routing
import memory_manager
import shadow_journal     # captures every signal — taken AND blocked
import thought_logger
from thought_logger import _state_lock as _partial_close_lock  # BUG-03 FIX: share the same
                                                                 # lock that thought_logger uses
                                                                 # for CONTINUATION_MEM_PATH.
                                                                 # Without this, concurrent writes
                                                                 # from the post-mortem daemon and
                                                                 # _set_last_partial_close_event()
                                                                 # corrupted the JSON.
import daily_post_mortem
import black_swan_monitor

# Wisdom Worker
sys.path.append(os.path.join(current_dir, 'Integration', 'Wisdom_Worker'))
from wisdom_builder import check_and_run_if_needed as wisdom_check
from context_retriever import get_full_memory_context

# Quant — regime detector + RL + trade manager
sys.path.append(os.path.join(current_dir, 'Quant', 'regime_detector'))
sys.path.append(os.path.join(current_dir, 'Quant', 'rl_manager'))
sys.path.append(os.path.join(current_dir, 'Quant', 'regime_router'))
import regime_detector as rd
import episode_recorder as er
from trade_manager import manage_trade
import regime_router as rr          # NEW v12 — regime router
import auto_retrainer               # auto GMM-HMM + XGBoost retrainer

# XU-L Meta-labelling layer
sys.path.append(os.path.join(current_dir, 'Quant', 'meta_labeller'))
try:
    import meta_labeller as ml
    _META_ENABLED = True
    print("[MetaLabeller] ✓ Meta-labelling layer active.")
except ImportError:
    _META_ENABLED = False
    print("[MetaLabeller] ✗ Not available — running without meta layer.")

# ── Stability Layer ──────────────────────────────
from startup_validator import run_validation
from file_lock_registry import read_json, write_json
from console_display import (
    print_boot_banner, print_critical, print_warning,
    print_trade_open, print_trade_close,
    print_sl_update, print_partial_close,
    print_memory_event, print_retrain,
    print_pnl_update, print_cycle,
    print_outage_banner, print_recovery_banner,
    print_session_summary, print_wisdom_update
)
from telegram_notifier import (
    send_startup_alert, send_halt_alert,
    send_trade_opened, send_trade_closed,
    send_sl_update, send_partial_close,
    send_session_summary, send_pnl_update,
    send_wisdom_updated, send_postmortem
)
from mt5_heartbeat import check_mt5_health
from ai_client import is_ai_available, get_ai_status

# --- Config ---
# load_dotenv() called at the top of the file to fix import-order validation bug
from master_controls import (
    GATE_MIN_CONFIDENCE, GATE_MAX_CONSECUTIVE_WAITS, GATE_LONDON_MIN_CONFLUENCE,
    GATE_MIN_RR, GATE_MAX_ENTRY_DEVIATION, GATE_MAX_SPREAD_DOLLARS,
    NEWS_BLOCK_BEFORE_MINUTES, NEWS_BLOCK_AFTER_MINUTES,
    FRIDAY_CLOSE_HOUR, FRIDAY_NO_TRADE_MINS,
    ALLOW_STACKING, CYCLE_INTERVAL_SECONDS, AI_DISPLAY_NAME,
    SESSIONS,   # BUG-01 FIX: was missing — caused NameError crash at startup
)
from ai_client import call_ai, get_client, AI_MODEL
# API_KEY and MODEL_ID now managed by ai_client.py / master_controls.py
SYMBOL   = "XAUUSD"
NY_TZ    = pytz.timezone('America/New_York')
from paths import CONTINUATION_MEM_PATH as STATE_FILE, create_all_dirs as _cad_mb
from Stability.file_lock_registry import read_json, write_json
_cad_mb()

_active_trade = {
    "ticket": None, "type": None, "entry": 0.0,
    "sl": 0.0, "tp": 0.0, "lot": 0.0, "management_stage": 0,
    "regime_at_entry": None,    # NEW v12 — stored for live trade adapter
}

# FIX #6 — Graceful shutdown flag.
# Set to True on SIGINT / KeyboardInterrupt. Daemon threads check this flag
# before/after file I/O so they can finish an in-progress write before dying,
# preventing truncated/corrupted wisdom.json, post_mortem.json, etc.
# C-02 FIX — Max consecutive loop error counter.
# If mt5.initialize() or any critical component fails permanently (e.g. MT5
# terminal closed), the loop was previously an infinite 60-second spam loop.
# After _MAX_LOOP_ERRORS consecutive failures the bot performs a hard stop,
# triggering the same graceful KeyboardInterrupt cleanup path.
import threading as _threading
_shutdown_event: _threading.Event = _threading.Event()   # set() → all daemons exit cleanly
_MAX_LOOP_ERRORS    = 10
_consecutive_errors = 0

# ── Consecutive WAIT tracker ───────────────────────────────────────
# Prevents redundant Claude calls when market is structurally unchanged
# and AI already said WAIT. Resets on any non-WAIT signal or regime change.
_consecutive_waits = 0
_last_wait_regime  = None

# ADAPTIVE CYCLE TIMING
# Claude returns next_check_mins in its JSON response.  The bot uses this
# instead of the hard-coded 300-second cycle when in an active window with
# no open trade.  Floor = 300s (5 min).  Cap = 3600s (60 min).
# Resets to 300 (default) at the start of each cycle so a missing field
# never causes a stale long sleep to persist beyond one cycle.
_AI_SLEEP_FLOOR_SECS: int = 300   # 5 min hard minimum — never go below this
_AI_SLEEP_CAP_SECS:   int = 3600  # 60 min hard max in active sessions
_ai_next_check_secs:  int = _AI_SLEEP_FLOOR_SECS  # updated by run_trading_cycle()

# ISSUE 4 FIX: LLM decision cache
# Caches WAIT/HOLD decisions for 1 cycle when market context is byte-identical.
# NEVER caches BUY/SELL — directional trades must always be fresh.
import hashlib as _hashlib
_decision_cache: dict = {}   # {context_hash: {"signal": str, "decision": dict}}

# ================================================================
# SESSION GATING
# ================================================================
# B-04 FIX: ACTIVE_WINDOWS was a hard-coded static list. Editing session times in
# master_controls.SESSIONS had zero effect. Now ACTIVE_WINDOWS is derived from
# master_controls.SESSIONS so a single edit in master_controls propagates here.
#
# BUG-02 FIX: Previous version added +1 to every end hour unconditionally, making
# London run until 05:59 and NY until 12:59 — one extra hour each. End times in
# master_controls are now EXCLUSIVE hour boundaries (e.g. "05:00" means stop AT
# 05:00, so hour 5 is NOT active). No +1 is applied.
# Generic midnight-crossing check (s >= e) replaces the fragile (s==19 and e<=2).
def _build_active_windows(sessions: dict) -> list:
    windows = []
    for label, cfg in sessions.items():
        s = int(cfg["start"].split(":")[0])
        e = int(cfg["end"].split(":")[0])   # BUG-02 FIX: no +1; end is exclusive
        win_label = cfg.get("label", label)
        if s >= e:          # generic midnight-crossing (covers any session spanning midnight)
            windows.append((s, 24, win_label))
            windows.append((0, e, win_label))
        else:
            windows.append((s, e, win_label))
    return windows

ACTIVE_WINDOWS = _build_active_windows(SESSIONS)

def get_ny_time():
    return datetime.now(NY_TZ)

def get_current_session():
    hour = get_ny_time().hour
    for start, end, name in ACTIVE_WINDOWS:
        if start <= hour < end:
            return name
    return None

def is_in_active_window():
    return get_current_session() is not None

def seconds_until_next_window():
    now = get_ny_time()
    candidates = []
    for start, _, name in ACTIVE_WINDOWS:
        candidate = now.replace(hour=start, minute=0, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        candidates.append((candidate, name))
    candidates.sort(key=lambda x: x[0])
    next_open, next_name = candidates[0]
    return max((next_open - now).total_seconds(), 0), next_name

# ================================================================
# PARTIAL CLOSE EVENT GUARD
# ================================================================
def _get_last_partial_close_event():
    try:
        state = read_json(STATE_FILE)
        if state:
            return state.get('last_partial_close_event', '')
    except Exception:
        pass
    return ''

def _set_last_partial_close_event(event_tag):
    # BUG-03 FIX: Acquire the shared _partial_close_lock (imported from thought_logger)
    # before reading/writing STATE_FILE.  thought_logger.update_state() also acquires
    # this lock before touching the same file, so the two concurrent writers
    # (main thread partial-close guard + post-mortem daemon) can no longer overlap.
    # Without this lock, a partial-close write and a thought_logger write issued
    # within the same ~1ms window produced a truncated or event-tag-less JSON,
    # causing the duplicate-close guard to fire again on the very next cycle.
    with _partial_close_lock:
        state = read_json(STATE_FILE) or {}
        state['last_partial_close_event'] = event_tag
        try:
            write_json(STATE_FILE, state)
        except Exception as e:
            print(f"Could not persist partial close guard: {e}")

# ================================================================
# SHARED HELPERS
# ================================================================
# MEM-01 FIX: _get_regime() cache.
# _get_regime() was called 2-3 times per cycle (Step 5 pre-manage_trade + Step 8
# regime detection + overnight guard).  Each call: initialize MT5, pull 1300 candles
# across 4 timeframes, create DataFrames, run full XGBoost pipeline.  With a 5-min
# cycle this was doubling/tripling API and MT5 load for zero benefit.
# Cache design: store the last result and its timestamp.  Any call within 60 seconds
# returns the cached result without touching MT5 or XGBoost.
# BUY/SELL decisions NEVER use a stale cache — each call within the same cycle sees
# the same result, which is correct: the regime doesn't change within one cycle.
_regime_cache_result:    dict  = {}
_regime_cache_timestamp: float = 0.0
_REGIME_CACHE_TTL_SECS:  int   = 60   # one full cycle — regime can't change faster than this


def _get_regime():
    """
    Pulls M5/H1/H4/D1 from MT5 and runs multi-TF regime detection.
    Zero API calls — pure local XGBoost/GMM-HMM.
    Falls back to H1-only if higher TFs unavailable.
    FIX Bug 1: mt5.shutdown() now in finally block — guaranteed even on exception.
    MEM-01 FIX: results cached for 60 seconds — multiple calls within one
    5-minute cycle all share the same result, eliminating 2-3× redundant
    MT5 data pulls and XGBoost inference per cycle.
    """
    global _regime_cache_result, _regime_cache_timestamp
    now_ts = time.time()
    if _regime_cache_result and (now_ts - _regime_cache_timestamp) < _REGIME_CACHE_TTL_SECS:
        return _regime_cache_result   # hot path — most intra-cycle calls take this
    try:
        if mt5.initialize():
            try:
                m5_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 1000)
                h1_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 300)
                h4_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 250)
                d1_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 250)
            finally:
                mt5.shutdown()

            def to_df(rates):
                if rates is None or len(rates) < 50:
                    return None
                df = pd.DataFrame(rates)
                df.columns = [c.lower() for c in df.columns]
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.set_index("time", inplace=True)
                return df

            h1_df = to_df(h1_rates)
            if h1_df is not None:
                result = rd.predict(
                    h1_df=h1_df,
                    m5_df=to_df(m5_rates),
                    h4_df=to_df(h4_rates),
                    d1_df=to_df(d1_rates),
                )
                _regime_cache_result    = result          # MEM-01 FIX: store in cache
                _regime_cache_timestamp = time.time()
                return result
    except Exception as e:
        print(f"[RegimeDetector] Error: {e}")
    fallback = {"regime": "LOW_VOL_RANGE", "guidance": "", "mode": "error_fallback",
                "confidence": None, "probabilities": None}
    _regime_cache_result    = fallback                    # MEM-01 FIX: cache fallback too
    _regime_cache_timestamp = time.time()
    return fallback


def _reset_active_trade():
    global _active_trade
    _active_trade = {
        "ticket": None, "type": None, "entry": 0.0,
        "sl": 0.0, "tp": 0.0, "lot": 0.0, "management_stage": 0,
        "regime_at_entry": None,
    }

def _handle_broker_close():
    global _active_trade
    try:
        if mt5.initialize():
            try:
                # FIX Bug 8: Use broker tick time (not server local time) for SOD
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick:
                    now_dt = datetime.fromtimestamp(tick.time)
                else:
                    now_dt = datetime.now(NY_TZ).replace(tzinfo=None)
                sod   = datetime(now_dt.year, now_dt.month, now_dt.day)
                deals = mt5.history_deals_get(sod, now_dt)
            finally:
                mt5.shutdown()  # FIX Bug 1: guaranteed shutdown
            trade_result = "WIN"
            pnl_dollars  = 0.0
            close_price  = _active_trade.get("entry", 0.0)
            if deals:
                for deal in reversed(deals):
                    if (deal.position_id == _active_trade["ticket"] and
                            deal.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)):
                        pnl_dollars  = float(deal.profit) + float(deal.commission) + float(deal.swap)
                        close_price  = float(getattr(deal, "price", _active_trade.get("entry", 0.0)))
                        trade_result = "WIN" if pnl_dollars >= 0 else "LOSS"
                        break
            print(f"[Main] Broker closed trade. Result: {trade_result}")
            _pnl_pips = 0.0
            try:
                _entry = float(_active_trade.get("entry", 0.0))
                if _entry > 0 and _active_trade.get("type") == "BUY":
                    _pnl_pips = (close_price - _entry) * 10.0
                elif _entry > 0 and _active_trade.get("type") == "SELL":
                    _pnl_pips = (_entry - close_price) * 10.0
            except Exception:
                _pnl_pips = 0.0
            _tg_close = {
                "direction": _active_trade.get("type", "N/A"),
                "entry": _active_trade.get("entry", 0),
                "exit_price": close_price,
                "lot": _active_trade.get("lot", 0),
                "session": get_current_session() or "N/A",
                "ticket": _active_trade.get("ticket", 0),
            }
            print_trade_close(_tg_close, pnl_dollars, _pnl_pips)
            send_trade_closed(_tg_close, pnl_dollars, _pnl_pips)
            risk_manager.update_trade_result(trade_result)
            memory_manager.update_final_review(
                _active_trade["ticket"], trade_result,
                f"Position closed by broker. Result: {trade_result}"
            )
            er.close_episode(_active_trade["ticket"], trade_result, pnl_dollars)
    except Exception as e:
        print(f"[Main] Broker close detection error: {e}")
    _reset_active_trade()

def _record_episode_candle(mgmt_result):
    try:
        if mt5.initialize():
            try:
                m5_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 2)
                tick     = mt5.symbol_info_tick(SYMBOL)
                pos_list = mt5.positions_get() or []
            finally:
                mt5.shutdown()  # FIX Bug 1: guaranteed shutdown
            current_price = (tick.bid + tick.ask) / 2 if tick else _active_trade["entry"]
            unreal_pnl    = next(
                (float(p.profit) for p in pos_list if p.ticket == _active_trade["ticket"]), 0.0
            )
            candle_ohlc = {}
            if m5_rates is not None and len(m5_rates) > 0:
                last = m5_rates[-1]
                candle_ohlc = {"open": float(last[1]), "high": float(last[2]),
                               "low":  float(last[3]), "close": float(last[4])}
            er.record_candle(
                ticket=_active_trade["ticket"], current_price=current_price,
                unrealised_pnl=unreal_pnl,
                management_stage=_active_trade["management_stage"],
                candle_ohlc=candle_ohlc, atr=0.0,
                action_taken=mgmt_result.get("action_taken", "HOLD"),
            )
    except Exception as e:
        print(f"[Main] Episode record error: {e}")

# ================================================================
# OVERNIGHT REGIME GUARD
# Protects open trade during dead zones — zero API cost
# ================================================================
def run_overnight_trade_guard():
    global _active_trade
    if not _active_trade["ticket"]:
        return

    print(f"[OvernightGuard] Trade #{_active_trade['ticket']} active. "
          f"Regime guard running (no AI call).")

    # Check if position still exists
    live_positions = trade_executor.get_open_positions(SYMBOL)
    live_tickets   = [p.ticket for p in live_positions] if live_positions else []
    if _active_trade["ticket"] not in live_tickets:
        print(f"[OvernightGuard] Position closed by broker overnight.")
        _handle_broker_close()
        return

    # Mechanical trade management — always runs
    # Overnight guard already has regime_result from the _get_regime() call below,
    # but we compute it first here so manage_trade can use it for hybrid dispatch.
    regime_result = _get_regime()
    regime        = regime_result.get("regime", "LOW_VOL_RANGE")
    confidence    = regime_result.get("confidence") or 0.0
    direction     = _active_trade["type"]
    today_tag     = get_ny_time().strftime('%Y%m%d')

    mgmt_result = manage_trade(
        symbol=SYMBOL, ticket=_active_trade["ticket"],
        entry_price=_active_trade["entry"], sl_price=_active_trade["sl"],
        tp_price=_active_trade["tp"], direction=_active_trade["type"],
        stage=_active_trade["management_stage"],
        regime_result=regime_result,   # ← hybrid dispatch: REVERSAL fires immediately here too
    )
    _active_trade["management_stage"] = mgmt_result["stage"]
    _active_trade["sl"]               = mgmt_result["sl_price"]
    _path = mgmt_result.get("hybrid_action", "MECHANICAL")
    print(f"[OvernightGuard/{_path}] {mgmt_result['action_taken']}")

    # REVERSAL hardcoded close already executed inside manage_trade — clean up here
    if mgmt_result.get("force_close"):
        # ── Notify: Force Close ──────────────────────────
        try:
            import MetaTrader5 as _mt5_fc
            _fc_close_price = 0.0
            _fc_pnl_dollars = 0.0
            _fc_pnl_pips    = 0.0
            _fc_ticket      = _active_trade.get("ticket", 0)
            
            if _fc_ticket and _mt5_fc.initialize():
                _fc_deals = _mt5_fc.history_deals_get(
                    position=_fc_ticket
                )
                if _fc_deals and len(_fc_deals) > 0:
                    # Last deal is the closing deal
                    _last_deal       = _fc_deals[-1]
                    _fc_close_price  = _last_deal.price
                    _fc_pnl_dollars  = _last_deal.profit
                    _fc_entry        = _active_trade.get("entry", 0)
                    _fc_direction    = _active_trade.get("type", "")
                    if _fc_entry and _fc_close_price:
                        if _fc_direction == "BUY":
                            _fc_pnl_pips = (
                              (_fc_close_price - _fc_entry) * 10.0
                            )
                        elif _fc_direction == "SELL":
                            _fc_pnl_pips = (
                              (_fc_entry - _fc_close_price) * 10.0
                            )
                _mt5_fc.shutdown()
            
            _fc_trade_dict = {
                "direction"  : _active_trade.get("type",   "N/A"),
                "entry"      : _active_trade.get("entry",  0),
                "exit_price" : _fc_close_price,
                "lot"        : _active_trade.get("lot",    0),
                "session"    : get_current_session() or "N/A",
                "ticket"     : _fc_ticket,
            }
            
            print_trade_close(_fc_trade_dict, 
                              _fc_pnl_dollars, 
                              _fc_pnl_pips)
            send_trade_closed(_fc_trade_dict, 
                              _fc_pnl_dollars, 
                              _fc_pnl_pips)
        except Exception:
            pass   # never let notification crash trade mgmt
        memory_manager.update_final_review(
            _active_trade["ticket"], "CLOSED_BY_AI",
            f"Overnight guard hybrid REVERSAL close. {mgmt_result['action_taken']}"
        )
        risk_manager.update_trade_result("CLOSED_BY_AI")
        er.close_episode(_active_trade["ticket"], "CLOSED_BY_AI", 0.0)
        _reset_active_trade()
        return

    # ── Live trade adapter (TIGHTEN_TP / WIDEN_SL only — REVERSAL is now handled ──
    # by hybrid dispatch inside manage_trade() above, so we only process
    # the remaining adapt actions that hybrid dispatch doesn't cover.
    adapt = rr.adapt_live_trade(regime_result, _active_trade)
    if adapt["action"] == "CLOSE_PARTIAL":
        event_tag = f"overnight_adapt_partial_{today_tag}"
        if _get_last_partial_close_event() != event_tag:
            print(f"[OvernightGuard] RegimeRouter: {adapt['reason']}")
            trade_executor.close_partial_position(SYMBOL, close_pct=0.5)
            _lots_closed = float(_active_trade.get("lot", 0.0)) * 0.5
            _lots_remaining = max(float(_active_trade.get("lot", 0.0)) - _lots_closed, 0.0)
            print_partial_close(
                ticket=_active_trade.get("ticket", 0),
                lots_closed=_lots_closed,
                remaining=_lots_remaining
            )
            send_partial_close(
                ticket=_active_trade.get("ticket", 0),
                lots_closed=_lots_closed,
                remaining=_lots_remaining
            )
            _set_last_partial_close_event(event_tag)
    elif adapt["action"] == "TIGHTEN_TP" and adapt.get("new_tp"):
        print(f"[OvernightGuard] RegimeRouter TIGHTEN_TP → {adapt['new_tp']:.2f}")
        _active_trade["tp"] = adapt["new_tp"]
    elif adapt["action"] == "WIDEN_SL" and adapt.get("new_sl"):
        print(f"[OvernightGuard] RegimeRouter WIDEN_SL → {adapt['new_sl']:.2f}")
        from trade_executor import modify_position_sl
        if modify_position_sl(_active_trade["ticket"], SYMBOL, adapt["new_sl"]):
            _active_trade["sl"] = adapt["new_sl"]
            _new_sl = adapt["new_sl"]
            _sl_reason = adapt.get("reason", "RegimeRouter WIDEN_SL")
            print_sl_update(
                ticket=_active_trade.get("ticket", 0),
                new_sl=_new_sl,
                reason=_sl_reason
            )
            send_sl_update(
                ticket=_active_trade.get("ticket", 0),
                new_sl=_new_sl,
                reason=_sl_reason
            )

    # BULL/BEAR counter-direction check (overnight only — not covered by hybrid dispatch)
    if regime in ("BULL_TREND", "BEAR_TREND") and confidence >= 0.80:
        try:
            if mt5.initialize():
                try:
                    h1_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 300)
                finally:
                    mt5.shutdown()
                if h1_rates is not None and len(h1_rates) > 200:
                    h1_df = pd.DataFrame(h1_rates)
                    h1_df.columns = [c.lower() for c in h1_df.columns]
                    h1_df['time'] = pd.to_datetime(h1_df['time'], unit='s')
                    h1_df.set_index('time', inplace=True)
                    from feature_engineer import build_features
                    feats = build_features(h1_df, prefix='h1')
                    if not feats.empty:
                        ema_stack = feats.iloc[-1].get('h1_ema_stack', 0)
                        trend_against = (
                            (direction == "BUY"  and ema_stack < 0) or
                            (direction == "SELL" and ema_stack > 0)
                        )
                        if trend_against:
                            event_tag = f"overnight_trend_against_{today_tag}"
                            if _get_last_partial_close_event() != event_tag:
                                print(f"[OvernightGuard] {regime} counter-direction to {direction} "
                                      f"(conf {confidence:.2f}) — FULL CLOSE.")
                                trade_executor.close_all_positions(SYMBOL)
                                memory_manager.update_final_review(
                                    _active_trade["ticket"], "CLOSED_BY_AI",
                                    f"Overnight guard: {regime} counter-direction to {direction} "
                                    f"(conf {confidence:.2f})."
                                )
                                risk_manager.update_trade_result("CLOSED_BY_AI")
                                er.close_episode(_active_trade["ticket"], "CLOSED_BY_AI", 0.0)
                                _set_last_partial_close_event(event_tag)
                                _reset_active_trade()
                                return
        except Exception as e:
            print(f"[OvernightGuard] EMA stack check error: {e}")
    elif regime not in ("BULL_TREND", "BEAR_TREND", "REVERSAL"):
        print(f"[OvernightGuard] Regime {regime} — hybrid/mechanical management only. Holding.")

    _record_episode_candle(mgmt_result)

# ================================================================
# PRE-TRADE FILTERS
# ================================================================
def check_friday_rule():
    now = get_ny_time()
    if now.weekday() == 4:
        if now.hour >= FRIDAY_CLOSE_HOUR:
            print("Friday 5PM — closing all positions.")
            trade_executor.close_all_positions(SYMBOL)
            return False
        minutes_to_close = (FRIDAY_CLOSE_HOUR * 60) - (now.hour * 60 + now.minute)
        if minutes_to_close <= FRIDAY_NO_TRADE_MINS:
            print(f"{minutes_to_close}min to weekly close (cutoff {FRIDAY_NO_TRADE_MINS}min). No new trades.")
            return False
    return True

def check_news_window(news_data):
    if not news_data:
        return False
    try:
        pattern = r'(\d{1,2}:\d{2} [AP]M) \(NY Time\).*HIGH IMPACT'
        matches = re.findall(pattern, news_data, re.IGNORECASE)
        now = get_ny_time()
        for match in matches:
            event_time = datetime.strptime(match, '%I:%M %p').replace(
                year=now.year, month=now.month, day=now.day, tzinfo=NY_TZ)
            diff_minutes = (now - event_time).total_seconds() / 60
            if -NEWS_BLOCK_BEFORE_MINUTES <= diff_minutes <= NEWS_BLOCK_AFTER_MINUTES:
                print("Blocked by high impact news window.")
                return True
    except Exception:
        pass
    return False

def check_open_position_news_rule(news_data):
    if not news_data or not _active_trade["ticket"]:
        return
    try:
        pattern = r'(\d{1,2}:\d{2} [AP]M) \(NY Time\).*HIGH IMPACT'
        matches = re.findall(pattern, news_data, re.IGNORECASE)
        now = get_ny_time()
        for match in matches:
            event_time = datetime.strptime(match, '%I:%M %p').replace(
                year=now.year, month=now.month, day=now.day, tzinfo=NY_TZ)
            diff_minutes = (event_time - now).total_seconds() / 60
            if 2 <= diff_minutes <= 7:
                event_tag = f"news_{match}"
                if _get_last_partial_close_event() != event_tag:
                    print("Partial close triggered (one-time per news event).")
                    trade_executor.close_partial_position(SYMBOL, close_pct=0.5)
                    _lots_closed = float(_active_trade.get("lot", 0.0)) * 0.5
                    _lots_remaining = max(float(_active_trade.get("lot", 0.0)) - _lots_closed, 0.0)
                    print_partial_close(
                        ticket=_active_trade.get("ticket", 0),
                        lots_closed=_lots_closed,
                        remaining=_lots_remaining
                    )
                    send_partial_close(
                        ticket=_active_trade.get("ticket", 0),
                        lots_closed=_lots_closed,
                        remaining=_lots_remaining
                    )
                    _set_last_partial_close_event(event_tag)
    except Exception:
        pass

def check_no_stacking():
    if not ALLOW_STACKING:
        positions = trade_executor.get_open_positions(SYMBOL)
        if positions:
            print("Position already open. Stacking blocked.")
            return False
    return True

def get_live_price():
    # B-06 FIX: added try/finally so mt5.shutdown() is guaranteed even if
    # symbol_info_tick() raises (e.g. broker disconnect mid-cycle).
    if not mt5.initialize():
        return None
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        return tick.bid if tick else None
    finally:
        mt5.shutdown()

def validate_trade_logic(decision, live_price):
    signal = decision.get("signal", "WAIT").upper()
    if signal not in ["BUY", "SELL"]:
        return True
    entry = float(decision.get("entry", 0))
    sl    = float(decision.get("sl",    0))
    tp    = float(decision.get("tp",    0))
    if entry <= 0 or sl <= 0 or tp <= 0:
        print("Hallucination Guard: entry/sl/tp contains zero. Rejected.")
        return False
    if live_price and abs(entry - live_price) > GATE_MAX_ENTRY_DEVIATION:
        print(f"Hallucination Guard: Entry ${entry} vs live ${live_price} "
              f"(gap ${abs(entry-live_price):.2f}). Rejected.")
        return False
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        print("SL distance zero. Rejected.")
        return False
    rr = abs(tp - entry) / sl_dist
    if rr < GATE_MIN_RR:
        print(f"RR {rr:.2f} below {GATE_MIN_RR} minimum. Rejected.")
        return False
    if signal == "BUY" and not (sl < entry < tp):
        print("BUY structure invalid. Rejected.")
        return False
    if signal == "SELL" and not (tp < entry < sl):
        print("SELL structure invalid. Rejected.")
        return False
    return True

# ================================================================
# MAIN TRADING CYCLE
# ================================================================
def run_trading_cycle(high_precautions=False):
    global _active_trade, _consecutive_waits, _last_wait_regime
    global _regime_cache_result, _regime_cache_timestamp   # MEM-01 FIX: invalidate cache each cycle
    global _ai_next_check_secs                             # ADAPTIVE TIMING: reset to floor each cycle

    # MEM-01 FIX: Force a fresh regime read at the start of every cycle.
    # Within the cycle, repeated calls return the cached result.
    # This prevents the first Step-5 call from using a 55-min-old cache entry
    # left over from the previous cycle.
    _regime_cache_result    = {}
    _regime_cache_timestamp = 0.0

    # ADAPTIVE TIMING: Reset to floor at the start of each cycle.
    # If Claude doesn't return next_check_mins (e.g. gate blocks before AI call),
    # the bot falls back to the standard 5-min cycle — never a stale long sleep.
    _ai_next_check_secs = _AI_SLEEP_FLOOR_SECS

    # ── AUTO-RETRAIN RELOAD CHECK ─────────────────────────────────
    # auto_retrainer.py sets reload_flag.json when a retrain completes.
    # This call is a single os.path.exists() — essentially zero overhead.
    # When the flag is found, new model files are loaded in-process and
    # the flag is cleared. Bot misses zero candles during the swap.
    rd.reload_if_needed()

    now     = get_ny_time()
    session = get_current_session()
    print(f"\n--- CYCLE: {now.strftime('%H:%M:%S')} NY | "
          f"Session: {session or 'DEAD ZONE'} ---")

    # ── MT5 Heartbeat ────────────────────────────────
    if not check_mt5_health():
        print_warning("MT5 unhealthy — skipping this cycle.")
        return True   # skip cycle, main loop will retry

    # Strategy Scout: non-blocking pending proposal ping every cycle
    try:
        import sys as _sys
        _mem_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Memory')
        _mem_dir = os.path.normpath(_mem_dir)
        if _mem_dir not in _sys.path:
            _sys.path.insert(0, _mem_dir)
        from strategy_scout import _ping_pending_proposals
        _ping_pending_proposals()
    except Exception:
        pass

    # ── DEAD ZONE ────────────────────────────────────────────────────
    if not session:
        if _active_trade["ticket"]:
            run_overnight_trade_guard()
        else:
            secs, next_name = seconds_until_next_window()
            print(f"[SessionGate] Dead zone. No trade open. "
                  f"Next: {next_name} in {secs/3600:.1f}h.")
        return True

    # ── ACTIVE WINDOW ────────────────────────────────────────────────

    # 1. Black Swan
    halt, reason, precautions = black_swan_monitor.is_market_in_panic()
    if halt:
        print(f"PANIC LOCKDOWN: {reason}. Liquidating & halting.")
        trade_executor.close_all_positions(SYMBOL)
        return False
    if precautions:
        high_precautions = True
        print(f"HIGH PRECAUTIONS: {reason}. Lot will be halved.")

    # 2. Friday rule
    if not check_friday_rule():
        return True

    # 3. BUG FIX v12 — NEWS FETCHED FIRST so market data receives it correctly
    #    (Previously: market data was fetched before news → compute_news_gate always got "" → AI never saw gate)
    news_data = ""
    try:
        from news_extractor import get_usd_news_today
        news_data = get_usd_news_today()
    except Exception:
        pass

    # 4. Market data — news_today param now correctly populated
    try:
        market_context = data_extractor.get_live_market_data(news_today=news_data)
    except Exception as e:
        print(f"Market data error: {e}")
        return True
    if not market_context:
        return True

    # Shadow journal: advance forward-price tracking on all open shadow entries
    # FIX #7: pass last completed M5 candle so tick() can detect SL-before-TP
    # correctly when price wicks through SL and recovers to TP in the same bar.
    try:
        live_px = get_live_price()
        if live_px:
            _sj_candle = None
            try:
                from trade_manager import get_recent_m5_candles as _get_m5
                _m5s = _get_m5(SYMBOL, count=3)
                if _m5s and len(_m5s) >= 2:
                    _sj_candle = _m5s[-2]   # last *completed* bar ([-1] is still forming)
            except Exception:
                pass
            shadow_journal.tick(live_px, candle=_sj_candle)
    except Exception:
        pass

    # 5. Trade management if position open
    if _active_trade["ticket"]:
        live_positions = trade_executor.get_open_positions(SYMBOL)
        live_tickets   = [p.ticket for p in live_positions] if live_positions else []
        if _active_trade["ticket"] not in live_tickets:
            print(f"[Main] Position #{_active_trade['ticket']} closed by broker.")
            _handle_broker_close()
            return True
        check_open_position_news_rule(news_data)

        # Regime detection is needed BEFORE manage_trade so hybrid dispatch can use it.
        # We do an early regime check here (lightweight — cached by _get_regime if same cycle).
        _mgmt_regime = _get_regime()

        mgmt_result = manage_trade(
            symbol=SYMBOL, ticket=_active_trade["ticket"],
            entry_price=_active_trade["entry"], sl_price=_active_trade["sl"],
            tp_price=_active_trade["tp"], direction=_active_trade["type"],
            stage=_active_trade["management_stage"],
            regime_result=_mgmt_regime,    # ← hybrid dispatch uses this
        )
        _active_trade["management_stage"] = mgmt_result["stage"]
        _active_trade["sl"]               = mgmt_result["sl_price"]
        # ── Notify: SL Updated ───────────────────────────
        try:
            _notify_new_sl     = mgmt_result.get("sl_price", 0)
            _notify_sl_reason  = mgmt_result.get(
                                   "action_taken",
                                   "Trade manager SL update"
                                 )
            if _notify_new_sl and _notify_new_sl != 0:
                print_sl_update(
                    ticket = _active_trade.get("ticket", 0),
                    new_sl = _notify_new_sl,
                    reason = _notify_sl_reason
                )
                send_sl_update(
                    ticket = _active_trade.get("ticket", 0),
                    new_sl = _notify_new_sl,
                    reason = _notify_sl_reason
                )
        except Exception:
            pass   # never let notification crash trade mgmt
        _path = mgmt_result.get("hybrid_action", "MECHANICAL")
        print(f"[TradeManager/{_path}] {mgmt_result['action_taken']}")

        # REVERSAL hardcoded close: manage_trade already sent the close order.
        # We still need to update memory + reset _active_trade here.
        if mgmt_result.get("force_close"):
            # ── Notify: Force Close ──────────────────────────
            try:
                import MetaTrader5 as _mt5_fc
                _fc_close_price = 0.0
                _fc_pnl_dollars = 0.0
                _fc_pnl_pips    = 0.0
                _fc_ticket      = _active_trade.get("ticket", 0)
                
                if _fc_ticket and _mt5_fc.initialize():
                    _fc_deals = _mt5_fc.history_deals_get(
                        position=_fc_ticket
                    )
                    if _fc_deals and len(_fc_deals) > 0:
                        # Last deal is the closing deal
                        _last_deal       = _fc_deals[-1]
                        _fc_close_price  = _last_deal.price
                        _fc_pnl_dollars  = _last_deal.profit
                        _fc_entry        = _active_trade.get("entry", 0)
                        _fc_direction    = _active_trade.get("type", "")
                        if _fc_entry and _fc_close_price:
                            if _fc_direction == "BUY":
                                _fc_pnl_pips = (
                                  (_fc_close_price - _fc_entry) * 10.0
                                )
                            elif _fc_direction == "SELL":
                                _fc_pnl_pips = (
                                  (_fc_entry - _fc_close_price) * 10.0
                                )
                    _mt5_fc.shutdown()
                
                _fc_trade_dict = {
                    "direction"  : _active_trade.get("type",   "N/A"),
                    "entry"      : _active_trade.get("entry",  0),
                    "exit_price" : _fc_close_price,
                    "lot"        : _active_trade.get("lot",    0),
                    "session"    : get_current_session() or "N/A",
                    "ticket"     : _fc_ticket,
                }
                
                print_trade_close(_fc_trade_dict, 
                                  _fc_pnl_dollars, 
                                  _fc_pnl_pips)
                send_trade_closed(_fc_trade_dict, 
                                  _fc_pnl_dollars, 
                                  _fc_pnl_pips)
            except Exception:
                pass   # never let notification crash trade mgmt
            memory_manager.update_final_review(
                _active_trade["ticket"], "CLOSED_BY_AI",
                f"Hybrid manager REVERSAL close. {mgmt_result['action_taken']}"
            )
            risk_manager.update_trade_result("CLOSED_BY_AI")
            er.close_episode(_active_trade["ticket"], "CLOSED_BY_AI", 0.0)
            _reset_active_trade()
            return True

        _record_episode_candle(mgmt_result)

    # 6. News window block (no open trade)
    if check_news_window(news_data) and not _active_trade["ticket"]:
        print("News window active. No open trade. Skipping AI call.")
        return True

    # 7. Risk clearance
    if not risk_manager.check_risk_clearance(SYMBOL):
        return True
    if not risk_manager.check_consecutive_losses(SYMBOL):
        return True

    # 8. Regime detection
    regime_result  = _get_regime()
    regime_context = rd.format_for_prompt(regime_result)

    # ── NEW v12: RULE 1 — REGIME CONFIDENCE GATE ─────────────────
    gate_allowed, gate_reason = rr.check_confidence_gate(regime_result)
    if not gate_allowed:
        print(f"[RegimeRouter] CONFIDENCE GATE: {gate_reason}")
        # Shadow journal: no Claude signal yet — log with live price, no TP/SL
        try:
            _live_px = get_live_price() or 0
            if _live_px > 0:
                # Log both directions when confidence gate fires — regime direction unknown
                _reg = regime_result.get("regime", "UNKNOWN")
                _dir = "BUY" if _reg == "BULL_TREND" else ("SELL" if _reg == "BEAR_TREND" else None)
                if _dir:
                    shadow_journal.log_signal(
                        signal=_dir, entry_price=_live_px,
                        sl_price=0, tp_price=0,
                        regime=_reg,
                        regime_confidence=regime_result.get("confidence") or 0,
                        session=session,
                        gate_blocked_by="CONFIDENCE_GATE",
                        block_reason=gate_reason[:200],
                        strategy="",
                    )
        except Exception:
            pass
        return True   # skip Claude call entirely — zero API cost
    # ─────────────────────────────────────────────────────────────

    # ── CONSECUTIVE WAIT GATE ─────────────────────────────────────
    # If AI said WAIT 3+ cycles in a row AND regime hasn't changed AND
    # model is confident (≥65%) — market is genuinely doing nothing.
    # Calling Claude again immediately is pure token waste.
    # Resets the moment any regime change, conf drop, or non-WAIT fires.
    _cw_regime = regime_result.get("regime", "UNKNOWN")
    _cw_conf   = regime_result.get("confidence") or 0.0
    
    # FIX: Implement missing reset conditions! If regime changes or confidence drops, reset the wait tracker.
    if _cw_regime != _last_wait_regime or _cw_conf < 0.65:
        _consecutive_waits = 0
        _last_wait_regime = None
        
    if (_consecutive_waits >= GATE_MAX_CONSECUTIVE_WAITS
            and _cw_regime == _last_wait_regime
            and _cw_conf >= 0.65):
        print(f"[PreFilter] Consecutive WAIT ×{_consecutive_waits} in "
              f"{_cw_regime} (conf {_cw_conf:.2f}) — "
              f"regime unchanged, skipping Claude call.")
        try:
            _live_px = get_live_price() or 0
            _dir = "BUY" if _cw_regime == "BULL_TREND" else ("SELL" if _cw_regime == "BEAR_TREND" else None)
            if _live_px > 0 and _dir:
                shadow_journal.log_signal(
                    signal=_dir, entry_price=_live_px,
                    sl_price=0, tp_price=0,
                    regime=_cw_regime,
                    regime_confidence=_cw_conf,
                    session=session,
                    gate_blocked_by="CONSECUTIVE_WAIT",
                    block_reason=f"Waited {_consecutive_waits}× in {_cw_regime}",
                    strategy="",
                )
        except Exception:
            pass
        return True
    # ─────────────────────────────────────────────────────────────

    # ── NEW v12: LIVE TRADE ADAPTER (mid-trade regime shift) ──────
    # B-09 FIX: this block was duplicated — once here and once earlier (step 8).
    # The earlier call (at step 8 after regime detection) is the correct placement
    # because it fires before the AI prompt is built. This second copy is removed
    # to prevent TIGHTEN_TP and WIDEN_SL from firing twice per cycle.
    # ─────────────────────────────────────────────────────────────

    # 9. Build prompt
    past_lessons    = get_full_memory_context(market_context, current_time=get_ny_time())
    current_state   = thought_logger.get_current_state()
    logic_framework   = strategy_logic.get_analytical_framework()
    execution_rules   = strategy_rules.get_execution_rules()
    adaptive_timing   = _adaptive_timing_rule()

    # ISSUE 3 FIX + AI Strategy routing: pass session + cached keywords
    # context_retriever already extracted current_keywords (cached) in past_lessons call.
    # We pass them here so strategy_selector can check AI-discovered strategies first.
    try:
        from context_retriever import _get_current_context_keywords as _get_kws
        _current_kws = _get_kws(market_context, current_time=get_ny_time())
    except Exception:
        _current_kws = []
    strategy_result = strategy_selector.select_strategy(
        regime_result, session=session, current_keywords=_current_kws)
    strategy_rules_text = strategy_result["rules"]
    strategy_task       = strategy_result["task_override"]
    print(strategy_selector.format_strategy_header(strategy_result))

    if strategy_result["strategy"] == "BLOCKED":
        print("[StrategySelector] BLOCKED regime — skipping Claude call.")
        return True

    # ISSUE 4 FIX: Cache WAIT/HOLD decisions — BUY/SELL never cached
    # B-12 FIX: upgraded from MD5[:16] (64-bit) to SHA-256[:32] (128-bit) to
    # eliminate hash collision risk. Over years of operation a 64-bit key with
    # 100-entry cache could serve a stale WAIT for a genuine BUY setup.
    # INFO-03 FIX: include today's date in the hash so a WAIT cached at session
    # close can never suppress a fresh signal the following morning.
    _cache_date = get_ny_time().strftime('%Y%m%d')
    _ctx_hash = _hashlib.sha256((_cache_date + market_context[:600]).encode()).hexdigest()[:32]
    _cached   = _decision_cache.get(_ctx_hash)
    if _cached and _cached.get("signal") in ("WAIT", "HOLD"):
        print(f"[LLMCache] HIT ({_ctx_hash[:8]}) — context unchanged, "
              f"last signal was {_cached['signal']}. Skipping Claude call.")
        try:
            _live_px = get_live_price() or 0
            _reg = regime_result.get("regime", "UNKNOWN")
            _dir = "BUY" if _reg == "BULL_TREND" else ("SELL" if _reg == "BEAR_TREND" else None)
            if _live_px > 0 and _dir:
                shadow_journal.log_signal(
                    signal=_dir, entry_price=_live_px,
                    sl_price=0, tp_price=0,
                    regime=_reg,
                    regime_confidence=regime_result.get("confidence") or 0,
                    session=session,
                    gate_blocked_by="LLM_CACHE",
                    block_reason="Identical market context — WAIT reused",
                    strategy=strategy_result.get("strategy", ""),
                )
        except Exception:
            pass
        _consecutive_waits += 1
        _last_wait_regime   = regime_result.get("regime", "UNKNOWN")
        return True

    # ── NEW v12: Regime router context injected into prompt ───────
    # Claude now sees: regime state, confidence, size permission, SL/TP note
    # For the prompt we build the gate context without an adjust_result yet —
    # that gets built after Claude returns a signal (so Claude sees it next cycle)
    router_context = rr.format_gate_context(
        regime_result,
        (gate_allowed, gate_reason, 1.0),   # size_mult placeholder for prompt
        None
    )

    # ── Session timing intelligence ───────────────────────────────
    # FIX Bug 6: Removed duplicate enrich_regime_result() call.
    # regime_detector.predict() already calls enrich_regime_result internally.
    # We only call format_session_profile_for_prompt here to build the prompt text.
    session_timing_context = ""
    try:
        from session_profiler import format_session_profile_for_prompt
        session_timing_context = format_session_profile_for_prompt(regime_result)
    except Exception:
        pass

    # ── HMM transition forecast ────────────────────────────────────
    # "Given the current regime, what does the HMM say typically comes next?"
    # Gives Claude a statistical prior about the next move before it reads price.
    transition_forecast_context = ""
    try:
        transition_forecast_context = rd.format_transition_forecast_for_prompt(regime_result)
    except Exception:
        pass

    # ── XGBoost signal quality ─────────────────────────────────────
    # Exposes internal decision process: vote tally, directional gap, downgrade flag.
    # Lets Claude know HOW confident the model is, not just what it decided.
    signal_quality_context = ""
    try:
        signal_quality_context = rd.format_model_signal_quality_for_prompt(regime_result)
    except Exception:
        pass
    # ─────────────────────────────────────────────────────────────

    prompt = f"""
    You are a Multi-Disciplinary Gold Trader with a continuous memory.

    --- YOUR PREVIOUS INTERNAL MONOLOGUE ---
    Your last bias was: {current_state.get('current_bias', 'NEUTRAL')}
    Your ongoing thesis was: {current_state.get('active_thesis', 'Searching for setup.')}

    --- MACROECONOMIC NEWS TODAY ---
    {news_data}

    --- LIVE CALENDAR ---
    Today is strictly: {now.strftime("%A")}
    Current session: {session}

    --- REGIME ROUTER (OBJECTIVE MARKET STATE SENSOR) ---
    {router_context}

    --- SESSION-REGIME TIMING INTELLIGENCE ---
    {session_timing_context}

    --- HMM TRANSITION FORECAST (WHAT TYPICALLY COMES NEXT) ---
    {transition_forecast_context}

    --- MODEL SIGNAL QUALITY (HOW CONFIDENT THE MODEL IS) ---
    {signal_quality_context}

    --- DETAILED REGIME GUIDANCE ---
    {regime_context}

    --- MARKET DATA ---
    {market_context}

    --- ANALYTICAL FRAMEWORK (HOW TO THINK) ---
    {logic_framework}

    --- EXECUTION RULES (RISK + POSITION SIZING) ---
    {execution_rules}

    --- ACTIVE STRATEGY (WHAT TO DO THIS CYCLE) ---
    {strategy_rules_text}

    --- HISTORICAL LESSONS & HINDSIGHT ---
    {past_lessons}

    --- ADAPTIVE CYCLE TIMING ---
    {adaptive_timing}

    --- TASK ---
    {strategy_task}
    """

    # 10. Call AI  (provider + model set in master_controls.py)
    decision = {}
    # ── AI Availability Gate ─────────────────────────
    _ai_available_now = is_ai_available()
    if not _ai_available_now:
        print_outage_banner()
        # Do not return — trade management has already run above.
    try:
        if _ai_available_now:
            raw_text = call_ai(prompt=prompt)
            if raw_text is None:
                raise RuntimeError(f"{AI_DISPLAY_NAME} call returned None")
            raw      = re.sub(r'```json\s*', '', raw_text)
            raw      = re.sub(r'```\s*',     '', raw)
            candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            for candidate in reversed(candidates):
                try:
                    decision = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
            if not decision:
                decision = json.loads(raw.strip())
        else:
            decision = {
                "signal": "WAIT",
                "bias": "NEUTRAL",
                "confluence_score": 0,
                "thesis": "AI unavailable — outage mode (manage-only).",
                "reasoning": "Circuit breaker active or AI outage."
            }
    except Exception as e:
        print(f"AI call / JSON parse error: {e}")
        return True

    signal = decision.get("signal", "WAIT").upper()
    if signal not in ["BUY", "SELL", "WAIT", "HOLD", "CLOSE"]:
        print(f"Unrecognised signal '{signal}' — defaulting to WAIT.")
        signal = "WAIT"

    # ── ADAPTIVE CYCLE TIMING ─────────────────────────────────────
    # Extract Claude's suggested wait time and store it globally.
    # get_sleep_duration() reads _ai_next_check_secs after this cycle ends.
    # Floor = 5 min, Cap = 60 min.  BUY/SELL resets to floor (act immediately
    # next cycle to manage the just-opened trade).
    if signal in ("BUY", "SELL"):
        _ai_next_check_secs = _AI_SLEEP_FLOOR_SECS   # new trade → check again ASAP
    else:
        _raw_mins = decision.get("next_check_mins", 5)
        try:
            _raw_mins = int(_raw_mins)
        except (TypeError, ValueError):
            _raw_mins = 5
        _ai_next_check_secs = max(_AI_SLEEP_FLOOR_SECS,
                                  min(_raw_mins * 60, _AI_SLEEP_CAP_SECS))
    print(f"[AdaptiveSleep] Next check in {_ai_next_check_secs // 60}min "
          f"(Claude suggested: {decision.get('next_check_mins', '—')}min)")
    # ─────────────────────────────────────────────────────────────

    # ISSUE 4 FIX: Cache WAIT/HOLD decisions — BUY/SELL never cached
    if signal in ("WAIT", "HOLD"):
        _decision_cache[_ctx_hash] = {"signal": signal, "decision": decision}
        if len(_decision_cache) > 100:
            del _decision_cache[next(iter(_decision_cache))]
        # Shadow journal: Claude itself chose WAIT — log what the regime implied
        try:
            _live_px = get_live_price() or 0
            _reg = regime_result.get("regime", "UNKNOWN")
            _dir = "BUY" if _reg == "BULL_TREND" else ("SELL" if _reg == "BEAR_TREND" else None)
            if _live_px > 0 and _dir:
                shadow_journal.log_signal(
                    signal=_dir, entry_price=_live_px,
                    sl_price=0, tp_price=0,
                    regime=_reg,
                    regime_confidence=regime_result.get("confidence") or 0,
                    session=session,
                    gate_blocked_by="CLAUDE_WAIT",   # ZOM-01 FIX: was "GEMINI_WAIT" — stale
                                                       # Gemini-era label. All WAIT-signal gate
                                                       # analytics were filed under the wrong
                                                       # provider name, corrupting post-mortem
                                                       # step 3b accuracy measurements.
                    block_reason=f"Claude said {signal}. Thesis: {decision.get('thesis','')[:150]}",
                    meta_prob=None,
                    confluence_score=decision.get("confluence_score", 0),
                    claude_thesis=decision.get("thesis", ""),
                    strategy=strategy_result.get("strategy", ""),
                )
        except Exception:
            pass

    print_cycle(
        regime=regime_result.get("regime", "N/A"),
        confidence=(regime_result.get("confidence") or 0) * 100,
        session=session or "N/A",
        result=signal,
        reason=decision.get("reasoning", decision.get("thesis", ""))
    )
    print(f"Thesis: {decision.get('thesis', 'No thesis.')}")

    # 11. CLOSE
    if signal == "CLOSE":
        if _active_trade["ticket"]:
            print("REVERSAL DETECTED — closing position.")
            trade_executor.close_all_positions(SYMBOL)
            memory_manager.update_final_review(
                _active_trade["ticket"], "CLOSED_BY_AI", "AI detected reversal.")
            risk_manager.update_trade_result("CLOSED_BY_AI")
            er.close_episode(_active_trade["ticket"], "CLOSED_BY_AI", 0.0)
            _reset_active_trade()
        else:
            print("CLOSE signal but no active trade.")

    # 12. BUY / SELL
    elif signal in ["BUY", "SELL"]:

        # ── NEW v12: RULE 2 — DUAL CONFIRMATION GATE ─────────────
        dual_allowed, dual_reason, size_multiplier = rr.check_dual_confirmation(
            regime_result, signal)
        if not dual_allowed:
            print(f"[RegimeRouter] DUAL GATE BLOCKED: {dual_reason}")
            # Shadow journal: Claude had a directional signal but regime disagreed
            try:
                shadow_journal.log_signal(
                    signal=signal,
                    entry_price=float(decision.get("entry", 0)) or (get_live_price() or 0),
                    sl_price=float(decision.get("sl", 0)),
                    tp_price=float(decision.get("tp", 0)),
                    regime=regime_result.get("regime", "UNKNOWN"),
                    regime_confidence=regime_result.get("confidence") or 0,
                    session=session,
                    gate_blocked_by="DUAL_GATE",
                    block_reason=dual_reason[:200],
                    meta_prob=None,
                    confluence_score=decision.get("confluence_score", 0),
                    claude_thesis=decision.get("thesis", ""),
                    strategy=strategy_result.get("strategy", ""),
                )
            except Exception:
                pass
            thought_logger.update_state(
                decision.get("bias", "NEUTRAL"),
                decision.get("thesis", "Regime blocked entry."),
                dual_reason, bool(_active_trade["ticket"])
            )
            return True
        if size_multiplier < 1.0:
            print(f"[RegimeRouter] DUAL GATE: size capped at {int(size_multiplier*100)}%. "
                  f"Reason: {dual_reason}")
        # ─────────────────────────────────────────────────────────

        # ── XU-L META GATE — trade quality filter ─────────────────
        meta_result = None
        if _META_ENABLED:
            try:
                primary_signal_ctx = {
                    "signal":      signal,
                    "confidence":  regime_result.get("confidence", 0.5),
                    "session":     session,
                    "entry_price": float(decision.get("entry", 0)),
                }
                meta_result = ml.predict_meta(
                    primary_signal_ctx, regime_result, None)
                meta_prob   = meta_result["meta_prob"]
                trade_size  = meta_result["size"]          # min(Kelly, edge)
                kelly_size  = meta_result["kelly_size"]
                edge_size   = meta_result["edge_size"]
                size_method = meta_result.get("size_method", "kelly")
                print(f"[MetaLabeller] meta_p={meta_prob:.2f} | "
                      f"size={trade_size:.2f} "
                      f"(kelly={kelly_size:.2f} edge={edge_size:.2f} "
                      f"→ using {size_method}) | "
                      f"{'✓ TRADE' if meta_result['should_trade'] else '✗ SKIP'}")
                print(f"[MetaLabeller] {meta_result['shap_reason']}")

                if not meta_result["should_trade"]:
                    print(f"[MetaLabeller] META GATE BLOCKED — "
                          f"trade quality too low (p={meta_prob:.2f} < "
                          f"{meta_result.get('threshold_used', ml.META_MIN_THRESHOLD):.2f})")
                    # Shadow journal: full Claude signal but meta said no
                    try:
                        shadow_journal.log_signal(
                            signal=signal,
                            entry_price=float(decision.get("entry", 0)) or (get_live_price() or 0),
                            sl_price=float(decision.get("sl", 0)),
                            tp_price=float(decision.get("tp", 0)),
                            regime=regime_result.get("regime", "UNKNOWN"),
                            regime_confidence=regime_result.get("confidence") or 0,
                            session=session,
                            gate_blocked_by="META_GATE",
                            block_reason=f"meta_p={meta_prob:.2f} < threshold. {meta_result.get('shap_reason','')[:150]}",
                            meta_prob=meta_prob,
                            confluence_score=decision.get("confluence_score", 0),
                            claude_thesis=decision.get("thesis", ""),
                            strategy=strategy_result.get("strategy", ""),
                        )
                    except Exception:
                        pass
                    # Log blocked trade — critical for threshold calibration
                    try:
                        ml.log_blocked_trade(
                            signal=signal,
                            regime=regime_result.get("regime", "UNKNOWN"),
                            session=session,
                            meta_prob=meta_prob,
                            threshold_used=meta_result.get(
                                "threshold_used", ml.META_MIN_THRESHOLD),
                            norm_vol=regime_result.get(
                                "session_profile", {}).get("norm_vol", 1.0),
                            shap_reason=meta_result.get("shap_reason", ""),
                        )
                    except Exception:
                        pass
                    thought_logger.update_state(
                        decision.get("bias", "NEUTRAL"),
                        decision.get("thesis", "Meta gate blocked entry."),
                        meta_result["shap_reason"],
                        bool(_active_trade["ticket"])
                    )
                    return True

                # Final size = min(meta combined size, regime size multiplier)
                if trade_size < size_multiplier:
                    print(f"[MetaLabeller] Combined size ({trade_size:.2f}) < "
                          f"regime size ({size_multiplier:.2f}) — using combined.")
                    size_multiplier = trade_size

            except Exception as e:
                print(f"[MetaLabeller] predict_meta error: {e} — meta gate skipped.")
        # ──────────────────────────────────────────────────────────

        # London gate — Rule 1a: only execute if 3/3 confluence
        if session == "London" and decision.get("confluence_score", 0) < GATE_LONDON_MIN_CONFLUENCE:
            print(f"[LondonGate] Confluence {decision.get('confluence_score', 0)}/{GATE_LONDON_MIN_CONFLUENCE} "
                  f"— below 3/3 required for London. Standing aside.")
            # FIX H3: log LONDON_GATE to shadow journal
            try:
                shadow_journal.log_signal(
                    signal=signal,
                    entry_price=float(decision.get("entry", 0)) or (get_live_price() or 0),
                    sl_price=float(decision.get("sl", 0)),
                    tp_price=float(decision.get("tp", 0)),
                    regime=regime_result.get("regime", "UNKNOWN"),
                    regime_confidence=regime_result.get("confidence") or 0,
                    session=session,
                    gate_blocked_by="LONDON_GATE",
                    block_reason=f"Confluence {decision.get('confluence_score',0)}/{GATE_LONDON_MIN_CONFLUENCE} in London session",
                    meta_prob=meta_result.get("meta_prob") if meta_result else None,
                    confluence_score=decision.get("confluence_score", 0),
                    claude_thesis=decision.get("thesis", ""),
                    strategy=strategy_result.get("strategy", ""),
                )
            except Exception:
                pass
        elif not check_no_stacking():
            # FIX H3: log STACKING_BLOCK to shadow journal
            try:
                shadow_journal.log_signal(
                    signal=signal,
                    entry_price=float(decision.get("entry", 0)) or (get_live_price() or 0),
                    sl_price=float(decision.get("sl", 0)),
                    tp_price=float(decision.get("tp", 0)),
                    regime=regime_result.get("regime", "UNKNOWN"),
                    regime_confidence=regime_result.get("confidence") or 0,
                    session=session,
                    gate_blocked_by="STACKING_BLOCK",
                    block_reason="Position already open — stacking not allowed",
                    meta_prob=meta_result.get("meta_prob") if meta_result else None,
                    confluence_score=decision.get("confluence_score", 0),
                    claude_thesis=decision.get("thesis", ""),
                    strategy=strategy_result.get("strategy", ""),
                )
            except Exception:
                pass
        elif check_news_window(news_data):
            print("Blocked by news rule.")
            # FIX H3: log NEWS_BLOCK to shadow journal
            try:
                shadow_journal.log_signal(
                    signal=signal,
                    entry_price=float(decision.get("entry", 0)) or (get_live_price() or 0),
                    sl_price=float(decision.get("sl", 0)),
                    tp_price=float(decision.get("tp", 0)),
                    regime=regime_result.get("regime", "UNKNOWN"),
                    regime_confidence=regime_result.get("confidence") or 0,
                    session=session,
                    gate_blocked_by="NEWS_BLOCK",
                    block_reason="High-impact news window active",
                    meta_prob=meta_result.get("meta_prob") if meta_result else None,
                    confluence_score=decision.get("confluence_score", 0),
                    claude_thesis=decision.get("thesis", ""),
                    strategy=strategy_result.get("strategy", ""),
                )
            except Exception:
                pass
        else:
            entry   = float(decision.get("entry", 0))
            sl      = float(decision.get("sl",    0))
            tp      = float(decision.get("tp",    0))

            # ── NEW v12: RULE 3 — SL/TP REGIME ADJUSTER ──────────
            if entry > 0 and sl > 0 and tp > 0:
                adjust  = rr.adjust_sl_tp(regime_result, entry, sl, tp, signal)
                adj_sl  = adjust["sl"]
                adj_tp  = adjust["tp"]
                print(f"[RegimeRouter] SL/TP ADJUSTED | "
                      f"SL: {sl:.2f}→{adj_sl:.2f} ({adjust['sl_moved']:+.2f}pts) | "
                      f"TP: {tp:.2f}→{adj_tp:.2f} ({adjust['tp_moved']:+.2f}pts) | "
                      f"RR: {adjust['rr']:.2f}")
                sl = adj_sl
                tp = adj_tp
            # ─────────────────────────────────────────────────────

            if not validate_trade_logic(
                {**decision, "sl": sl, "tp": tp, "signal": signal}, get_live_price()):
                print("Validation failed after regime adjustment. Order not sent.")
                # FIX H3: log VALIDATION_FAIL to shadow journal
                try:
                    shadow_journal.log_signal(
                        signal=signal,
                        entry_price=float(decision.get("entry", 0)) or (get_live_price() or 0),
                        sl_price=sl, tp_price=tp,
                        regime=regime_result.get("regime", "UNKNOWN"),
                        regime_confidence=regime_result.get("confidence") or 0,
                        session=session,
                        gate_blocked_by="VALIDATION_FAIL",
                        block_reason="Hallucination guard or RR check rejected levels after regime adjustment",
                        meta_prob=meta_result.get("meta_prob") if meta_result else None,
                        confluence_score=decision.get("confluence_score", 0),
                        claude_thesis=decision.get("thesis", ""),
                        strategy=strategy_result.get("strategy", ""),
                    )
                except Exception:
                    pass
            elif not risk_manager.check_spread(SYMBOL):
                # FIX H3: log SPREAD_BLOCK to shadow journal
                try:
                    shadow_journal.log_signal(
                        signal=signal,
                        entry_price=float(decision.get("entry", 0)) or (get_live_price() or 0),
                        sl_price=sl, tp_price=tp,
                        regime=regime_result.get("regime", "UNKNOWN"),
                        regime_confidence=regime_result.get("confidence") or 0,
                        session=session,
                        gate_blocked_by="SPREAD_BLOCK",
                        block_reason=f"Spread too wide (>{GATE_MAX_SPREAD_DOLLARS}$)",
                        meta_prob=meta_result.get("meta_prob") if meta_result else None,
                        confluence_score=decision.get("confluence_score", 0),
                        claude_thesis=decision.get("thesis", ""),
                        strategy=strategy_result.get("strategy", ""),
                    )
                except Exception:
                    pass
            else:
                sl_dist = abs(entry - sl)
                lot     = risk_manager.calculate_position_size(
                    SYMBOL, sl_dist, news_data=news_data)

                # Apply dual gate size multiplier
                lot = round(lot * size_multiplier, 2)

                if high_precautions:
                    lot = round(lot * 0.5, 2)
                    print(f"High Precautions: lot reduced to {lot}")

                if not lot or lot <= 0:
                    print("Invalid lot size.")
                else:
                    ticket = trade_executor.execute_trade(
                        symbol=SYMBOL, action=signal, lot=lot,
                        sl=sl, tp=tp,
                        order_type=decision.get("order_type", "MARKET"),
                        entry_price=entry
                    )
                    if ticket:
                        print(f"POSITION OPENED: Ticket #{ticket}")
                        er.start_episode(
                            ticket=ticket, entry_price=entry, sl_price=sl, tp_price=tp,
                            direction=signal, lot_size=lot,
                            regime=regime_result.get("regime", "UNKNOWN"),
                            session=session,
                        )
                        # FIX Bug 6 (signal_aligned) + Memory Gap:
                        # Now we have the actual Claude signal — compute real alignment
                        # and refresh adaptive_threshold before logging.
                        _regime_str = regime_result.get("regime", "UNKNOWN")
                        _signal_aligned = (
                            (signal == "BUY"  and _regime_str == "BULL_TREND") or
                            (signal == "SELL" and _regime_str == "BEAR_TREND")
                        )
                        try:
                            from session_profiler import get_adaptive_threshold, compute_norm_vol
                            _sp = regime_result.get("session_profile", {})
                            _thr = get_adaptive_threshold(
                                _regime_str, session,
                                _sp.get("norm_vol", 1.0),
                                _sp.get("age_status", "MATURE"),
                                _signal_aligned,
                            )
                            if "session_profile" in regime_result:
                                regime_result["session_profile"]["adaptive_threshold"] = _thr.get("threshold", 0.55)
                                regime_result["session_profile"]["signal_aligned"]     = _signal_aligned
                        except Exception:
                            pass

                        memory_manager.log_trade(
                            ticket=ticket, signal=signal,
                            reasoning=decision.get("reasoning", ""),
                            entry_price=entry, sl=sl, tp=tp,
                            conf_score=decision.get("confluence_score", 0),
                            ict_logic=decision.get("analysis_ict", ""),
                            classic_logic=decision.get("analysis_classic", ""),
                            elliott_logic=decision.get("analysis_elliott", ""),
                            # MEMORY GAP FIX — regime context at entry
                            regime=_regime_str,
                            regime_confidence=regime_result.get("confidence"),
                            session=session,
                            meta_prob=meta_result.get("meta_prob") if meta_result else None,
                            gate_decisions={
                                "confidence_gate": gate_reason,
                                "dual_gate":       dual_reason,
                                "signal_aligned":  _signal_aligned,
                                "size_multiplier": size_multiplier,
                            },
                        )
                        _active_trade = {
                            "ticket":           ticket,
                            "type":             signal,
                            "entry":            entry,
                            "sl":               sl,
                            "tp":               tp,
                            "lot":              lot,
                            "management_stage": 0,
                            "regime_at_entry":  regime_result.get("regime"),   # NEW v12
                        }
                        # ── Notify: Trade Opened ─────────────────────────
                        _tg_trade = {
                            "direction": _active_trade.get("type", "N/A"),
                            "entry": _active_trade.get("entry", 0),
                            "sl": _active_trade.get("sl", 0),
                            "tp": _active_trade.get("tp", 0),
                            "lot": _active_trade.get("lot", 0),
                            "regime_confidence": round((regime_result.get("confidence") or 0) * 100),
                            "session": session if session else "N/A",
                            "ticket": _active_trade.get("ticket", 0),
                        }
                        _tg_regime = regime_result.get("regime", "N/A")
                        _tg_meta = (meta_result.get("meta_prob", 0.0) if meta_result else 0.0)
                        print_trade_open(_tg_trade)
                        send_trade_opened(_tg_trade, _tg_regime, _tg_meta)

    # 13. Update thought state
    thought_logger.update_state(
        decision.get("bias",      "NEUTRAL"),
        decision.get("thesis",    "No active thesis."),
        decision.get("reasoning", ""),
        bool(_active_trade["ticket"])
    )

    # 14. Update consecutive WAIT tracker
    # Any BUY/SELL/CLOSE resets — means market changed, fresh eyes next cycle.
    # WAIT/HOLD in same regime increments — we already know nothing is happening.
    if signal in ("WAIT", "HOLD"):
        _consecutive_waits += 1
        _last_wait_regime   = regime_result.get("regime", "UNKNOWN")
    else:
        _consecutive_waits = 0
        _last_wait_regime  = None

    return True


# ================================================================
# STARTUP: Orphan sync + Ghost Ship recovery
# ================================================================
def startup_sync():
    global _active_trade
    print("[Startup] Syncing memory with live MT5 positions...")
    if not mt5.initialize():
        print("[Startup] WARNING: Could not connect to MT5.")
        return
    # BUG-05 FIX: live_tickets initialised BEFORE the try block.
    # If mt5.positions_get() raises (e.g. MT5 not connected), the finally
    # still calls mt5.shutdown() correctly, but live_tickets would have been
    # undefined — causing NameError at memory_manager.sync_orphan_trades(live_tickets)
    # and crashing startup entirely.  Initialising here guarantees sync_orphan_trades
    # always receives a valid list (empty at worst).
    live_tickets = []
    try:
        positions    = mt5.positions_get(symbol=SYMBOL)
        live_tickets = [pos.ticket for pos in positions] if positions else []
    finally:
        mt5.shutdown()
    memory_manager.sync_orphan_trades(live_tickets)
    print(f"[Startup] Live positions: {live_tickets}")
    if not positions:
        print("[Startup] No open positions. Starting fresh.")
        return
    for pos in positions:
        if pos.magic != trade_executor.MAGIC_NUMBER:
            print(f"[Startup] Skipping #{pos.ticket} — magic {pos.magic} not ours.")
            continue
        direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        from trade_manager import _read_mgmt_state
        disk_state  = _read_mgmt_state(pos.ticket)
        saved_stage = disk_state.get("management_stage", 0)
        saved_sl    = disk_state.get("current_sl") or float(pos.sl)
        _active_trade = {
            "ticket":           pos.ticket,
            "type":             direction,
            "entry":            float(pos.price_open),
            "sl":               saved_sl,
            "tp":               float(pos.tp),
            "lot":              float(pos.volume),
            "management_stage": saved_stage,
            "regime_at_entry":  None,   # unknown after restart — adapter falls back to HOLD
        }
        print(f"[Startup] GHOST SHIP RECOVERED: #{pos.ticket} | {direction} | "
              f"Entry: {pos.price_open} | Stage: {saved_stage}")
        break


# ================================================================
# SMART SLEEP
# ================================================================
def get_sleep_duration():
    # ── DEAD ZONE: always sleep to next window boundary ───────────
    # AI suggestion is irrelevant — bot has nothing to do until the
    # next active window opens.
    if not is_in_active_window():
        secs, next_name = seconds_until_next_window()
        sleep_secs = max(secs - 30, 60)
        print(f"[Sleep] Dead zone. Sleeping {sleep_secs/3600:.2f}h until {next_name}.")
        return sleep_secs

    # ── ACTIVE WINDOW — TRADE OPEN ────────────────────────────────
    # Trade management runs on its own mechanical cadence; we still honour
    # Claude's suggestion but cap it at 10 minutes to stay responsive to
    # rapid price moves while a position is live.
    if _active_trade["ticket"]:
        ai_secs    = _ai_next_check_secs
        capped     = min(ai_secs, 600)   # 10-min cap with open trade
        capped     = max(capped, _AI_SLEEP_FLOOR_SECS)
        if capped != 300:
            print(f"[AdaptiveSleep] Trade open — AI suggested "
                  f"{ai_secs//60}min, capped to {capped//60}min.")
        return capped

    # ── ACTIVE WINDOW — NO TRADE ──────────────────────────────────
    # Claude fully controls the interval within [5 min, 60 min].
    ai_secs  = _ai_next_check_secs
    enforced = max(_AI_SLEEP_FLOOR_SECS, min(ai_secs, _AI_SLEEP_CAP_SECS))
    if enforced != 300:
        print(f"[AdaptiveSleep] Claude suggested {ai_secs//60}min → sleeping {enforced//60}min.")
    return enforced


# ================================================================
# MAIN LOOP
# ================================================================
if __name__ == "__main__":
    # ── Startup Validation ───────────────────────────
    if not run_validation():
        sys.exit(1)

    try:
        from Bootlogo import boot_trident_79
        boot_trident_79()
    except Exception as e:
        print(f"[Boot] Logo skipped ({e})")
        print("⚡ Gold AI Bridge v18 | Regime Router Active")

    startup_sync()

    try:
        from ai_client import _load_keys
        _n_keys = len(_load_keys())
    except Exception:
        _n_keys = 3

    print_boot_banner(
        model=AI_MODEL,
        symbol=SYMBOL,
        keys_loaded=_n_keys,
        regime_mode="AUTO"
    )
    send_startup_alert(
        symbol=SYMBOL,
        model=AI_MODEL,
        regime_mode="AUTO"
    )

    # ── Daily news refresh daemon ──────────────────────────────────
    # Fetches today's USD news immediately at startup, then refreshes
    # automatically at 06:30 NY every morning. All cycles read from
    # the in-memory cache — zero API calls mid-session.
    try:
        from news_extractor import start_daily_news_daemon
        start_daily_news_daemon()
    except Exception as _ne:
        print(f"[Boot] News daemon failed to start ({_ne}) — "
              f"news will be fetched per-cycle as fallback.")

    # ── Health agent daemon ────────────────────────────────────────
    # Runs a full system health check at startup + every 06:00 NY.
    # Checks models, API keys, MT5, news, memory, risk state etc.
    # Prints a report and uses Claude to prioritise any issues found.
    try:
        from health_agent import start_health_daemon
        start_health_daemon(verbose=False, use_ai=True)
    except Exception as _ha:
        print(f"[Boot] Health agent failed to start ({_ha})")

    pm_thread = threading.Thread(
        target=daily_post_mortem.check_and_run_if_needed, daemon=True)
    pm_thread.start()

    wisdom_thread = threading.Thread(target=wisdom_check, daemon=True)
    wisdom_thread.start()

    retrainer_thread = threading.Thread(
        target=auto_retrainer.check_and_retrain_if_needed, daemon=True)
    retrainer_thread.start()
    print("🔄 Auto-retrainer daemon started. Checks every 4h.")

    print("\n📅 Active windows (NY Time):")
    print("   Asian  : 19:00 – 01:00  (19:00–23:59 + 00:00–00:59)")
    print("   London : 02:00 – 05:00  (3/3 confluence + regime gate)")
    print("   NY     : 07:00 – 12:00")
    print("   Off-session gaps (01–02, 05–07, 12–13): overnight guard if trade open, sleep otherwise")
    print("   Dead zone (13:00–19:00): no AI calls, no entries. Overnight guard if trade open.")
    print("\n🧠 Regime Router: confidence gate | dual confirmation | SL/TP adjuster | live adapter")
    print("   Train:  python Quant/regime_detector/trainer.py --n-states 5")
    print("\nBot running. Ctrl+C to stop safely.\n")

    try:
        while True:
            try:
                should_continue = run_trading_cycle()
                if not should_continue:
                    print("Bot halted by risk system.")
                    break
                _consecutive_errors = 0   # C-02 FIX: reset on any successful cycle
                time.sleep(get_sleep_duration())
            except KeyboardInterrupt:
                raise   # propagate to outer handler
            except Exception as e:
                _consecutive_errors += 1
                print(f"Critical loop error #{_consecutive_errors}/{_MAX_LOOP_ERRORS}: {e}")
                if _consecutive_errors >= _MAX_LOOP_ERRORS:
                    # C-02 FIX: Hard stop — likely MT5 terminal is permanently closed.
                    # Raise KeyboardInterrupt to trigger the graceful shutdown block below
                    # (daemon joins, MT5 cleanup, file I/O protection).
                    print(f"\n🛑 HARD STOP: {_MAX_LOOP_ERRORS} consecutive critical errors. "
                          f"MT5 terminal likely offline. Triggering graceful shutdown.")
                    raise KeyboardInterrupt
                time.sleep(60)

    except KeyboardInterrupt:
        # FIX #6 — Graceful daemon shutdown.
        # Set the event FIRST so all daemon threads see it immediately.
        # Then give them up to 8 seconds to finish in-progress file I/O
        # (wisdom.json, post_mortem.json, trade_memory.json writes).
        # Without this, Ctrl+C during a write truncates the JSON to 0 bytes.
        print("\n\n⛔ Ctrl+C received. Signalling daemon threads to finish safely...")
        _shutdown_event.set()

        # Wait for daemon threads to finish (max 8s each)
        _daemon_threads = [
            ("PostMortem",  pm_thread),
            ("Wisdom",      wisdom_thread),
            ("Retrainer",   retrainer_thread),
        ]
        for _name, _t in _daemon_threads:
            _t.join(timeout=8)
            status = "done" if not _t.is_alive() else "still running (forced kill)"
            print(f"   [{_name}] {status}")

        # Clean up MT5 connection
        send_halt_alert("Manual shutdown (KeyboardInterrupt)")
        print_critical("Bot shutting down — Telegram notified.")
        try:
            if mt5.initialize():
                mt5.shutdown()
        except Exception:
            pass

        print("✅ Shutdown complete. All file I/O should be safe.")
