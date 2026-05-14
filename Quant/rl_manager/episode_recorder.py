"""
episode_recorder.py
====================
Records every 5-minute candle while a trade is live into
Data/Episodes/episodes.json.

FIX Bug 5: Active episode is written to a separate lightweight file
(active_episode.json) while the trade is open. On close, the completed
episode is appended to episodes.json ONCE. This replaces the previous
design of reading + writing the full history on every single candle.

Old design:  full episodes.json read+write every 5 min → O(N) disk I/O
             grows to ~20MB+ at 1000 episodes, freezes for seconds
New design:  active_episode.json (~50KB max) read+write every 5 min
             episodes.json written only once on close

Called from main_bot.py:
  - episode_recorder.start_episode(ticket, metadata)  — on trade open
  - episode_recorder.record_candle(ticket, state)     — every 5-min cycle
  - episode_recorder.close_episode(ticket, result)    — on trade close
"""

import os
import json
import threading
from datetime import datetime

current_dir          = os.path.dirname(os.path.abspath(__file__))
base_dir             = os.path.dirname(os.path.dirname(current_dir))
import sys as _sys_ep
_root_ep = os.path.normpath(os.path.join(base_dir))
if _root_ep not in _sys_ep.path: _sys_ep.path.insert(0, _root_ep)
from paths import EPISODES_PATH as EPISODES_FILE, ACTIVE_EPISODE_PATH as ACTIVE_EPISODE_FILE,                   create_all_dirs as _cad_ep
_cad_ep()

_lock = threading.Lock()


# ================================================================
# FILE HELPERS
# ================================================================

def _read_episodes() -> list:
    try:
        if os.path.exists(EPISODES_FILE):
            with open(EPISODES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[EpisodeRecorder] Read error (episodes.json): {e}")
    return []


def _append_episode(episode: dict):
    """Appends ONE episode to the history file. Called only on close."""
    try:
        os.makedirs(os.path.dirname(EPISODES_FILE), exist_ok=True)
        episodes = _read_episodes()
        episodes.append(episode)
        with open(EPISODES_FILE, 'w', encoding='utf-8') as f:
            json.dump(episodes, f, indent=4)
    except Exception as e:
        print(f"[EpisodeRecorder] Append error: {e}")


def _read_active() -> dict | None:
    """Reads the active (in-progress) episode. Returns None if no trade open."""
    try:
        if os.path.exists(ACTIVE_EPISODE_FILE):
            with open(ACTIVE_EPISODE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[EpisodeRecorder] Read error (active_episode.json): {e}")
    return None


def _write_active(episode: dict):
    """Writes the active episode to its dedicated lightweight file."""
    try:
        os.makedirs(os.path.dirname(ACTIVE_EPISODE_FILE), exist_ok=True)
        with open(ACTIVE_EPISODE_FILE, 'w', encoding='utf-8') as f:
            json.dump(episode, f, indent=2)
    except Exception as e:
        print(f"[EpisodeRecorder] Write error (active_episode.json): {e}")


def _clear_active():
    """Removes the active episode file after the trade is closed and merged."""
    try:
        if os.path.exists(ACTIVE_EPISODE_FILE):
            os.remove(ACTIVE_EPISODE_FILE)
    except Exception as e:
        print(f"[EpisodeRecorder] Could not clear active episode: {e}")


# ================================================================
# PUBLIC API
# ================================================================

def start_episode(ticket, entry_price, sl_price, tp_price,
                  direction, lot_size, regime, session=""):
    """
    Called when a new trade is opened.
    Creates a new active episode in active_episode.json.
    """
    with _lock:
        # Guard: if another active episode exists, close it first
        existing = _read_active()
        if existing and str(existing.get('ticket')) != str(ticket):
            print(f"[EpisodeRecorder] WARNING: Orphaned active episode for "
                  f"#{existing.get('ticket')} — merging to history before starting new.")
            existing['result']     = 'UNKNOWN'
            existing['close_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            existing['total_steps'] = len(existing.get('candle_states', []))
            _append_episode(existing)
            _clear_active()

        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(entry_price - tp_price)
        rr      = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0

        episode = {
            "ticket":        str(ticket),
            "open_time":     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "close_time":    None,
            "direction":     direction,
            "entry_price":   entry_price,
            "sl_price":      sl_price,
            "tp_price":      tp_price,
            "lot_size":      lot_size,
            "sl_dist":       round(sl_dist, 5),
            "tp_dist":       round(tp_dist, 5),
            "risk_reward":   rr,
            "regime":        regime,
            "session":       session,
            "result":        None,
            "final_pnl_r":   None,
            "candle_states": [],
        }
        _write_active(episode)
        print(f"[EpisodeRecorder] Episode started for Ticket #{ticket} | "
              f"Direction: {direction} | Regime: {regime} | R:R {rr}")


def record_candle(ticket, current_price, unrealised_pnl,
                  management_stage, candle_ohlc, atr,
                  action_taken="HOLD"):
    """
    Called every 5-minute cycle while a trade is open.
    FIX Bug 5: Only reads/writes the tiny active_episode.json (~50KB),
    never touches the full episodes.json history during an open trade.
    """
    with _lock:
        ep = _read_active()
        if not ep or str(ep.get('ticket')) != str(ticket):
            print(f"[EpisodeRecorder] No active episode for Ticket #{ticket}.")
            return

        entry   = ep.get('entry_price', current_price)
        sl_dist = ep.get('sl_dist', 1)

        if ep['direction'] == 'BUY':
            pnl_r = (current_price - entry) / sl_dist if sl_dist > 0 else 0
        else:
            pnl_r = (entry - current_price) / sl_dist if sl_dist > 0 else 0

        state = {
            "timestamp":        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "current_price":    round(current_price, 5),
            "unrealised_pnl":   round(unrealised_pnl, 2),
            "unrealised_pnl_r": round(pnl_r, 4),
            "management_stage": management_stage,
            "candle_open":      candle_ohlc.get('open',  0),
            "candle_high":      candle_ohlc.get('high',  0),
            "candle_low":       candle_ohlc.get('low',   0),
            "candle_close":     candle_ohlc.get('close', 0),
            "atr":              round(atr, 5),
            "action_taken":     action_taken,
        }
        ep['candle_states'].append(state)
        _write_active(ep)


def close_episode(ticket, result, final_pnl_dollars=0.0):
    """
    Called when a trade closes.
    FIX Bug 5: Loads active_episode.json, marks it complete, appends it
    to episodes.json ONCE, then deletes active_episode.json.
    """
    with _lock:
        ep = _read_active()
        if not ep or str(ep.get('ticket')) != str(ticket):
            print(f"[EpisodeRecorder] No active episode for #{ticket} to close.")
            # Fallback: check if it's already in history (duplicate close call)
            return

        # BUG-06 FIX: pnl_r formula verified for XAUUSD.
        # sl_dist is stored in price terms ($/oz), e.g. 2.5 for a 2.5-dollar SL.
        # For XAUUSD: 1 standard lot = 100 troy oz, so:
        #   dollar_risk = sl_dist ($/oz) × lot_size (lots) × 100 (oz/lot)
        #   pnl_r = final_pnl_dollars / dollar_risk
        # Example: sl_dist=2.5, lot=0.1 → dollar_risk = 2.5 × 0.1 × 100 = $25
        # If final_pnl = -$25 → pnl_r = -1.0R (correct full loss)
        # If final_pnl = +$50 → pnl_r = +2.0R (correct 2R win)
        # The * 100 is the oz-per-lot multiplier — NOT a pip convention assumption.
        lot_size = ep.get('lot_size', 1) or 1
        dollar_risk = sl_dist * lot_size * 100   # XAUUSD: 100 oz per standard lot
        pnl_r = round(final_pnl_dollars / dollar_risk, 4) if dollar_risk > 0 else 0

        ep['result']       = result
        ep['close_time']   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ep['final_pnl_r']  = pnl_r
        ep['total_steps']  = len(ep.get('candle_states', []))

        # Append completed episode to history ONCE
        _append_episode(ep)
        # Remove the active file
        _clear_active()

        print(f"[EpisodeRecorder] Episode closed: Ticket #{ticket} | "
              f"Result: {result} | Steps: {ep['total_steps']} | "
              f"P&L: {pnl_r:+.2f}R")


def get_episode_stats():
    """Returns summary stats for the current episode dataset."""
    episodes    = _read_episodes()
    active      = _read_active()
    closed      = [e for e in episodes if e.get('result') is not None]
    wins        = [e for e in closed if e.get('result') == 'WIN']
    losses      = [e for e in closed if e.get('result') == 'LOSS']
    total_steps = sum(len(e.get('candle_states', [])) for e in closed)

    print(f"\n[EpisodeRecorder] Dataset stats:")
    print(f"  Total closed   : {len(closed)}")
    print(f"  Wins           : {len(wins)}")
    print(f"  Losses         : {len(losses)}")
    print(f"  Total steps    : {total_steps}")
    print(f"  Active trade   : {'YES (#' + str(active.get('ticket')) + ')' if active else 'No'}")
    print(f"  (Need ~1000+ closed episodes to train RL agent)")
    return {
        "total":       len(closed),
        "wins":        len(wins),
        "losses":      len(losses),
        "total_steps": total_steps,
        "active":      active is not None,
    }
