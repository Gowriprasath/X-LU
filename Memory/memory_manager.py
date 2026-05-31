"""
memory_manager.py — Trade Memory Persistence
=============================================
FIX H5: sync_orphan_trades() now safely handles non-numeric and None
         ticket values without crashing at bot startup.
BUG-14 FIX: Rolling cap of 800 entries — archives oldest to Data/Memory/.
PATHS: All paths now sourced from paths.py (Data/Memory/).
"""

import json
import os
import threading
from datetime import datetime

import sys as _sys
_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _root not in _sys.path: _sys.path.insert(0, _root)
from paths import TRADE_MEMORY_PATH, MEMORY_DIR, create_all_dirs
create_all_dirs()
from file_lock_registry import read_json, write_json, modify_json

MEMORY_FILE = TRADE_MEMORY_PATH
_memory_lock = threading.Lock()


def _init_memory():
    if not os.path.exists(MEMORY_FILE):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        write_json(MEMORY_FILE, [])


def get_recent_memories(limit=5):
    _init_memory()
    try:
        memory_data = read_json(MEMORY_FILE)
        if not memory_data:
            return "No previous trades in memory."
        reviewed_trades = [t for t in memory_data
                           if t.get('status') in ['CLOSED', 'REVIEWED']]
        if not reviewed_trades:
            return "No detailed reviews or hindsight lessons available yet."
        recent_lessons = reviewed_trades[-limit:]
        memory_string = "--- DEEP-DIVE TRADING LESSONS & HINDSIGHT ---\n"
        for trade in recent_lessons:
            memory_string += (f"Ticket: {trade.get('ticket')} | "
                              f"Result: {trade.get('result', 'UNKNOWN')}\n")
            memory_string += f"Original Logic: {trade.get('reasoning')}\n"
            if trade.get('detailed_review'):
                memory_string += f"IMMEDIATE POST-MORTEM: {trade.get('detailed_review')}\n"
            if trade.get('hindsight_feedback'):
                memory_string += f"HINDSIGHT LESSON: {trade.get('hindsight_feedback')}\n"
            memory_string += "-" * 30 + "\n"
        return memory_string
    except Exception as e:
        return f"Warning: Could not load memory. Error: {e}"


def log_trade(ticket, signal, conf_score, ict_logic, classic_logic, elliott_logic,
              reasoning, entry_price, sl, tp,
              regime="UNKNOWN", regime_confidence=None, session="",
              meta_prob=None, gate_decisions=None):
    _init_memory()

    # M-03 FIX: Compute "did the AI follow the regime direction?" at log time.
    _REGIME_DIR_MAP = {
        "BULL_TREND":    "BUY",
        "BEAR_TREND":    "SELL",
        "LOW_VOL_RANGE": None,   # neutral — both BUY and SELL are valid (mean reversion)
        "COMPRESSION":   None,   # blocked regime — no direction
        "REVERSAL":      None,   # spike — direction unclear
        "UNCERTAIN":     None,   # model unreadable
    }
    _regime_direction = _REGIME_DIR_MAP.get(regime, None)
    _followed_regime  = (signal == _regime_direction) if _regime_direction else None
    _counter_regime   = (signal != _regime_direction) if _regime_direction else None

    def update(memory_data):
        if memory_data is None:
            memory_data = []
        new_entry = {
            "ticket":            ticket,
            "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal":            signal,
            "confluence_score":  conf_score,
            "analysis_ict":      ict_logic,
            "analysis_classic":  classic_logic,
            "analysis_elliott":  elliott_logic,
            "reasoning":         reasoning,
            "entry":             entry_price,
            "sl":                sl,
            "tp":                tp,
            "status":            "OPEN",
            "result":            "",
            "detailed_review":   "",
            "hindsight_feedback": "",
            "regime":            regime,
            "regime_confidence": regime_confidence,
            "session":           session,
            "meta_prob":         meta_prob,
            "gate_decisions":    gate_decisions or {},
            "pnl_r":             None,
            "regime_direction":  _regime_direction,   # "BUY" / "SELL" / None
            "followed_regime":   _followed_regime,    # True / False / None
            "counter_regime":    _counter_regime,     # True / False / None
        }
        memory_data.append(new_entry)

        # BUG-14 FIX: Rolling cap — archive oldest when file exceeds 800 trades
        _MEMORY_CAP = 800
        if len(memory_data) > _MEMORY_CAP:
            # Batch optimization: archive a chunk of 100 items at once rather than thrashing disk 1-by-1
            archive_count = max(100, len(memory_data) - _MEMORY_CAP)
            to_archive    = memory_data[:archive_count]
            memory_data   = memory_data[archive_count:]
            archive_date  = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_path  = os.path.join(MEMORY_DIR, f'trade_memory_archive_{archive_date}.json')
            try:
                write_json(archive_path, to_archive)
                print(f"[MemoryManager] Archived {archive_count} old trades → {os.path.basename(archive_path)}")
            except Exception as arch_err:
                print(f"[MemoryManager] Warning: could not write archive: {arch_err}")

        return memory_data

    return modify_json(MEMORY_FILE, update)


def update_trade_status(ticket, status):
    _init_memory()
    def update(memory_data):
        if not memory_data:
            memory_data = []
        for trade in memory_data:
            if str(trade.get('ticket')) == str(ticket):
                trade['status'] = status
                break
        return memory_data
    modify_json(MEMORY_FILE, update)


def update_trade_result(ticket, result, review=""):
    _init_memory()
    def update(memory_data):
        if not memory_data:
            memory_data = []
        for trade in memory_data:
            if str(trade.get('ticket')) == str(ticket):
                trade['result'] = result
                trade['status'] = 'CLOSED'
                if review:
                    trade['detailed_review'] = review
                break
        return memory_data
    modify_json(MEMORY_FILE, update)


def update_hindsight_review(ticket, hindsight_text):
    _init_memory()
    def update(memory_data):
        if not memory_data:
            memory_data = []
        for trade in memory_data:
            if str(trade.get('ticket')) == str(ticket):
                trade['hindsight_feedback'] = hindsight_text
                trade['status'] = 'REVIEWED'
                break
        return memory_data
    modify_json(MEMORY_FILE, update)


def update_pnl_r(ticket, pnl_r):
    _init_memory()
    def update(memory_data):
        if not memory_data:
            memory_data = []
        for trade in memory_data:
            if str(trade.get('ticket')) == str(ticket):
                trade['pnl_r'] = pnl_r
                break
        return memory_data
    modify_json(MEMORY_FILE, update)


# B-02 FIX: main_bot.py calls memory_manager.update_final_review() in 4 places
# but only update_trade_result() is defined. Add alias so all call sites work.
update_final_review = update_trade_result


def sync_orphan_trades(live_tickets):
    """
    FIX H5: Safely handles non-numeric and None ticket values.
    Marks any OPEN trades not in live_tickets as CLOSED (orphaned).
    """
    _init_memory()
    def update(memory_data):
        if not memory_data:
            memory_data = []

        live_set = set(str(t) for t in (live_tickets or []) if t is not None)
        for trade in memory_data:
            raw = trade.get('ticket')
            if raw is None:
                continue
            try:
                ticket_str = str(int(float(str(raw))))
            except (ValueError, TypeError):
                continue
            if trade.get('status') == 'OPEN' and ticket_str not in live_set:
                trade['status'] = 'CLOSED'
                if not trade.get('result'):
                    trade['result'] = 'CLOSED_UNKNOWN'
        return memory_data
        
    modify_json(MEMORY_FILE, update)
