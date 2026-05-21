"""
risk_manager.py — Position Sizing + Risk Controls
===================================================
FIX H4: calculate_position_size() now wraps MT5 in a try/finally block
         so mt5.shutdown() is ALWAYS called, even if _get_or_init_daily_state
         or any intermediate step throws an exception.
"""

import os
import json
import re
import MetaTrader5 as mt5  # type: ignore
import pytz
from datetime import datetime

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
# Use normcase comparison to handle Windows case/separator differences
if not any(_os.path.normcase(p) == _os.path.normcase(_mc_dir) for p in _sys.path):
    _sys.path.insert(0, _mc_dir)
from master_controls import (
    RISK_BASE_PCT               as BASE_RISK_PCT,
    RISK_DAILY_DRAWDOWN_PCT     as DAILY_DRAWDOWN_PCT,
    RISK_MAX_CONSECUTIVE_LOSSES as MAX_CONSECUTIVE_LOSSES,
    RISK_MIN_PCT                as MIN_RISK_PCT,
    GATE_MAX_SPREAD_DOLLARS,
)

current_dir     = os.path.dirname(os.path.abspath(__file__))
base_dir        = os.path.dirname(current_dir)
import sys as _sys_r
_root_r = os.path.normpath(os.path.join(base_dir))
if _root_r not in _sys_r.path: _sys_r.insert(0, _root_r)
from paths import RISK_STATE_PATH, create_all_dirs as _cad_r
_cad_r()
RISK_STATE_FILE = RISK_STATE_PATH

NY_TZ = pytz.timezone('America/New_York')

HIGH_IMPACT_EVENTS = [
    "CPI", "PPI", "Non-Farm", "NFP", "Unemployment Claims",
    "JOLTS", "Job Openings",
]
NEWS_REDUCE_WINDOW_BEFORE = 60
NEWS_REDUCE_WINDOW_AFTER  = 30


# ================================================================
# RISK STATE PERSISTENCE
# ================================================================

from Stability.file_lock_registry import read_json, write_json, modify_json

def _read_risk_state():
    try:
        if os.path.exists(RISK_STATE_FILE):
            return read_json(RISK_STATE_FILE) or {}
    except Exception as e:
        print(f"[RiskManager] Could not read risk state: {e}")
    return {}


def _write_risk_state(state):
    try:
        write_json(RISK_STATE_FILE, state)
    except Exception as e:
        print(f"[RiskManager] Could not write risk state: {e}")


def _get_or_init_daily_state(balance):
    today_str = datetime.now(NY_TZ).strftime('%Y-%m-%d')
    initialized = [None]
    
    def update(state):
        if not state:
            state = {}
        if state.get('date') != today_str:
            state = {
                'date':               today_str,
                'opening_balance':    round(float(balance), 2),
                'daily_drawdown_cap': round(float(balance) * DAILY_DRAWDOWN_PCT, 2),
                'base_risk_pct':      BASE_RISK_PCT,
                'current_risk_pct':   BASE_RISK_PCT,
                'consecutive_losses': 0,
                'halted':             False,
                'daily_drawdown_hit': False,
                'daily_drawdown_pct': 0.0,
            }
            initialized[0] = state
        return state

    modify_json(RISK_STATE_FILE, update)
    
    if initialized[0] is not None:
        state = initialized[0]
        print(f"[RiskManager] New day. Opening balance locked: "
              f"${state['opening_balance']:.2f} | "
              f"Drawdown cap: ${state['daily_drawdown_cap']:.2f} | "
              f"Risk: {state['current_risk_pct']*100:.2f}%")
        
    return _read_risk_state()


# ================================================================
# TRADE RESULT UPDATE
# ================================================================

def update_trade_result(result):
    result = result.upper()
    
    def update(state):
        if not state:
            return state
            
        if result == 'WIN':
            if state['current_risk_pct'] < state['base_risk_pct']:
                new_risk = min(
                    round(state['current_risk_pct'] * 2, 5),
                    state['base_risk_pct']
                )
                print(f"[RiskManager] WIN (recovering). Risk: "
                      f"{state['current_risk_pct']*100:.3f}% → {new_risk*100:.3f}%")
                state['current_risk_pct'] = new_risk
            state['consecutive_losses'] = 0
    
        elif result == 'CLOSED_BY_AI':
            print(f"[RiskManager] CLOSED_BY_AI — neutral result. "
                  f"Consecutive losses unchanged: {state['consecutive_losses']}")
    
        elif result == 'LOSS':
            state['consecutive_losses'] += 1
            new_risk = round(state['current_risk_pct'] * 0.5, 5)
            if new_risk < MIN_RISK_PCT:
                new_risk = MIN_RISK_PCT
                print(f"[RiskManager] LOSS #{state['consecutive_losses']}. "
                      f"Risk floor reached ({MIN_RISK_PCT*100:.1f}%).")
            else:
                print(f"[RiskManager] LOSS #{state['consecutive_losses']}. Risk reduced: "
                      f"{state['current_risk_pct']*100:.2f}% → {new_risk*100:.2f}%")
            state['current_risk_pct'] = new_risk
            
        return state

    modify_json(RISK_STATE_FILE, update)
    
    state = _read_risk_state()
    if state:
        print(f"[RiskManager] State updated — "
              f"Consecutive losses: {state.get('consecutive_losses')} | "
              f"Current risk: {state.get('current_risk_pct', 0.0)*100:.2f}%")


# ================================================================
# HIGH-IMPACT NEWS DETECTION
# ================================================================

def _get_news_volatility_multiplier(news_data, now_ny=None):
    if not news_data or not news_data.strip():
        return 1.0
    if now_ny is None:
        now_ny = datetime.now(NY_TZ)
    try:
        now_naive = now_ny.replace(tzinfo=None)
    except Exception:
        now_naive = now_ny

    pattern = r'(\d{1,2}:\d{2} [AP]M) \(NY Time\): (.+?) \['
    matches  = re.findall(pattern, news_data)

    for time_str, event_title in matches:
        is_target = any(
            keyword.lower() in event_title.lower()
            for keyword in HIGH_IMPACT_EVENTS
        )
        if not is_target:
            continue
        try:
            event_dt = datetime.strptime(time_str, '%I:%M %p').replace(
                year=now_naive.year, month=now_naive.month, day=now_naive.day
            )
        except Exception:
            continue
        diff_minutes = (now_naive - event_dt).total_seconds() / 60
        if -NEWS_REDUCE_WINDOW_BEFORE <= diff_minutes <= NEWS_REDUCE_WINDOW_AFTER:
            print(f"[RiskManager] HIGH-IMPACT EVENT WINDOW ACTIVE: '{event_title}' "
                  f"({diff_minutes:+.0f} min). Risk halved.")
            return 0.5
    return 1.0


# ================================================================
# POSITION SIZING
# FIX H4: MT5 wrapped in try/finally — shutdown always called.
# ================================================================

def calculate_position_size(symbol, sl_dist, risk_pct=None, news_data=""):
    if not mt5.initialize():
        return 0.01

    try:   # FIX H4: everything after initialize is inside try/finally
        account_info = mt5.account_info()
        symbol_info  = mt5.symbol_info(symbol)

        if not account_info or not symbol_info:
            return 0.01

        balance = float(account_info.balance)

        # This used to be outside try — FIX H4: now safely inside
        state = _get_or_init_daily_state(balance)

        effective_risk_pct = state.get('current_risk_pct', BASE_RISK_PCT)
        news_multiplier    = _get_news_volatility_multiplier(news_data)
        effective_risk_pct = round(effective_risk_pct * news_multiplier, 4)

        print(f"[RiskManager] Effective risk: {effective_risk_pct*100:.2f}% "
              f"(state: {state.get('current_risk_pct',BASE_RISK_PCT)*100:.2f}% × "
              f"news multiplier: {news_multiplier})")

        risk_amount = balance * effective_risk_pct

        tick_size  = float(symbol_info.trade_tick_size)
        tick_value = float(symbol_info.trade_tick_value)

        if tick_size == 0 or tick_value == 0:
            return 0.01

        ticks_at_risk = sl_dist / tick_size
        lot_size      = risk_amount / (ticks_at_risk * tick_value)

        step     = float(symbol_info.volume_step)
        lot_size = round(lot_size / step) * step

        min_vol  = float(symbol_info.volume_min)
        max_vol  = float(symbol_info.volume_max)
        lot_size = max(min_vol, min(lot_size, max_vol))

        return float(lot_size)

    except Exception as e:
        print(f"[RiskManager] Position size error: {e}")
        return 0.01

    finally:
        mt5.shutdown()   # FIX H4: ALWAYS runs, even on exception


# ================================================================
# DAILY DRAWDOWN CHECK
# ================================================================

def check_risk_clearance(symbol):
    if not mt5.initialize():
        print("[RiskManager] MT5 init failed.")
        return False
    try:
        account_info = mt5.account_info()
        if account_info is None:
            return False

        balance = float(account_info.balance)
        state   = _get_or_init_daily_state(balance)
        drawdown_cap = state.get('daily_drawdown_cap', balance * DAILY_DRAWDOWN_PCT)

        broker_time = mt5.symbol_info_tick(symbol)
        if broker_time:
            now = datetime.fromtimestamp(broker_time.time)
        else:
            now = datetime.now(NY_TZ).replace(tzinfo=None)

        start_of_day = datetime(now.year, now.month, now.day)
        deals        = mt5.history_deals_get(start_of_day, now)

        today_profit = 0.0
        if deals:
            for deal in deals:
                if deal.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                    today_profit += (float(deal.profit) +
                                     float(deal.commission) +
                                     float(deal.swap))

        positions       = mt5.positions_get()
        floating_profit = sum(float(p.profit) for p in positions) if positions else 0.0
        total_daily_pnl = today_profit + floating_profit

        if total_daily_pnl < -drawdown_cap:
            print(f"[RiskManager] DAILY DRAWDOWN CAP HIT: "
                  f"Loss ${abs(total_daily_pnl):.2f} > Cap ${drawdown_cap:.2f}.")
            current_dd_pct = round(abs(total_daily_pnl) / balance, 5) if balance > 0 else 0.0
            
            def update_cap_hit(state):
                if not state:
                    state = {}
                if not state.get("halted") or not state.get("daily_drawdown_hit"):
                    state["halted"] = True
                    state["daily_drawdown_hit"] = True
                    state["daily_drawdown_pct"] = current_dd_pct
                return state
                
            modify_json(RISK_STATE_FILE, update_cap_hit)
            return False

        # If drawdown cap not hit, make sure daily_drawdown_hit is False
        def update_cap_clear(state):
            if not state:
                state = {}
            if state.get("daily_drawdown_hit"):
                state["daily_drawdown_hit"] = False
                state["daily_drawdown_pct"] = 0.0
                if state.get("consecutive_losses", 0) < MAX_CONSECUTIVE_LOSSES:
                    state["halted"] = False
            return state
            
        modify_json(RISK_STATE_FILE, update_cap_clear)

        if not __import__("os").getenv("BACKTEST_MODE"):
            print(f"[RiskManager] Risk cleared. "
              f"Today PnL: ${total_daily_pnl:.2f} | Cap: -${drawdown_cap:.2f}")
        return True
    finally:
        mt5.shutdown()


# ================================================================
# CONSECUTIVE LOSS CHECK
# ================================================================

def check_consecutive_losses(symbol):
    state = _read_risk_state()
    if not state:
        return True
    today_str = datetime.now(NY_TZ).strftime('%Y-%m-%d')
    if state.get('date') != today_str:
        return True
    consecutive = state.get('consecutive_losses', 0)
    if consecutive >= MAX_CONSECUTIVE_LOSSES:
        print(f"[RiskManager] {MAX_CONSECUTIVE_LOSSES} CONSECUTIVE LOSSES TODAY. "
              f"Trading stopped for the day.")
        
        def update_loss_halt(state):
            if not state:
                state = {}
            if not state.get("halted"):
                state["halted"] = True
            return state
            
        modify_json(RISK_STATE_FILE, update_loss_halt)
        return False
    
    def update_loss_clear(state):
        if not state:
            state = {}
        if state.get("halted"):
            # Make sure we only unhalt if daily drawdown didn't hit it too
            if not state.get("daily_drawdown_hit"):
                state["halted"] = False
        return state
        
    modify_json(RISK_STATE_FILE, update_loss_clear)
    return True


# ================================================================
# SPREAD CHECK
# ================================================================

def check_spread(symbol, max_spread_dollars=None):
    """
    XAUUSD spread context:
      Asian session (quiet):  ~$0.20 – $0.60
      London open:            ~$0.50 – $1.20
      NY open (07:00–09:00):  ~$0.80 – $1.50  primary trading window
      News spikes:            $2.00+           correctly blocked here
    """
    if max_spread_dollars is None:
        max_spread_dollars = GATE_MAX_SPREAD_DOLLARS
    if not mt5.initialize():
        return False
    try:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        current_spread = tick.ask - tick.bid
        if current_spread > max_spread_dollars:
            print(f"[RiskManager] SPREAD TOO HIGH: "
                  f"${current_spread:.2f} (max ${max_spread_dollars:.2f}). Trade aborted.")
            return False
        return True
    finally:
        mt5.shutdown()