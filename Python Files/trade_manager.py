"""
trade_manager.py
=================
Hybrid trade management: regime-aware AI decisions + mechanical fallback.
Called every cycle from main_bot.py while a position is open.

HYBRID DISPATCH TABLE (priority order)
───────────────────────────────────────
1. REVERSAL (any confidence ≥ 0.65)
       → Hardcoded: immediate FULL CLOSE. No AI call. No mechanical logic.
         Rationale: speed > deliberation when structure is breaking.

2. BULL_TREND / BEAR_TREND (confidence > 0.70)
       → AI management call. Prompt asks: "Hold? Trail aggressively? Skip partial?"
         AI responds with one of: HOLD | TRAIL_AGGRESSIVE | SKIP_PARTIAL
         Fallback: mechanical trail if AI call fails or times out.

3. COMPRESSION / LOW_VOL_RANGE (any confidence)
       → AI management call. Prompt asks: "Early partial? Tight BE? Hold?"
         AI responds with: PARTIAL_75 | TIGHT_BE | HOLD
         Rationale: range-bound price → capture profit before mean reversion.
         Fallback: mechanical 1R partial if AI fails.

4. Everything else (RANGING, BULL_FLAG, BEAR_FLAG, etc.)
       → Pure mechanical: 1R partial + break-even + M5 trailing stop.
         No AI call. Deterministic, zero API cost.

STATE MACHINE (all paths share the same stage machine)
  Stage 0  → Watching for profit threshold. Partial + BE logic fires here.
  Stage 1  → Partial taken. SL at break-even. Trailing begins.
  Stage 2  → Trailing active. SL follows M5 swing structure.

FIX (Bug 2 - Amnesia Loop):
  Stage is recovered from disk (trade_memory.json) not in-memory variable.
  On restart, the correct stage + SL are restored automatically.
"""

import MetaTrader5 as mt5
import os
import sys
import time
import json
import threading
import re

# Dynamic path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.dirname(current_dir)
sys.path.append(current_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from trade_executor import modify_position_sl, partial_close, close_position, MAGIC_NUMBER

# FIX Bug 3: Import _memory_lock from memory_manager so both modules share
# the same file lock and can't corrupt trade_memory.json concurrently.
sys.path.append(os.path.join(base_dir, 'Memory'))
from memory_manager import _memory_lock as _state_lock

from Stability.file_lock_registry import read_json, write_json
from paths import TRADE_MEMORY_PATH
MEMORY_FILE = TRADE_MEMORY_PATH

# ── Lazy import of call_ai to avoid circular import at module load time ──
_call_ai_fn = None

def _get_call_ai():
    """Lazy-loads call_ai from ai_client.py. Returns None if unavailable."""
    global _call_ai_fn
    if _call_ai_fn is None:
        try:
            _ai_path = base_dir
            if _ai_path not in sys.path:
                sys.path.insert(0, _ai_path)
            from ai_client import call_ai
            _call_ai_fn = call_ai
        except Exception as e:
            print(f"[HybridManager] WARNING: Could not import call_ai: {e}")
    return _call_ai_fn


# ==============================================================
# PERSISTENT STATE READ / WRITE
# ==============================================================

def _read_mgmt_state(ticket):
    """
    Reads the management state for a given ticket from trade_memory.json.
    Returns a dict with keys: partial_taken, break_even_set, current_sl, stage.
    If no state exists yet (first run), returns safe defaults (stage 0).
    """
    defaults = {
        "partial_taken":   False,
        "break_even_set":  False,
        "current_sl":      None,   # None = not yet updated from disk
        "management_stage": 0,
    }
    try:
        memory_data = read_json(MEMORY_FILE)
        if not memory_data:
            return defaults
        for trade in memory_data:
            if str(trade.get('ticket')) == str(ticket):
                mgmt = trade.get('management_state', {})
                # Merge with defaults so any missing keys are safe
                return {**defaults, **mgmt}
    except Exception as e:
        print(f"[TradeManager] WARNING: Could not read management state: {e}")
    return defaults


def _write_mgmt_state(ticket, state_update):
    """
    Persists management state fields into the trade's record in trade_memory.json.
    state_update is a dict — only the keys provided are updated (partial merge).
    """
    with _state_lock:
        try:
            memory_data = read_json(MEMORY_FILE)
            if not memory_data:
                return

            for trade in memory_data:
                if str(trade.get('ticket')) == str(ticket):
                    existing = trade.get('management_state', {})
                    existing.update(state_update)
                    trade['management_state'] = existing
                    break

            write_json(MEMORY_FILE, memory_data)

        except Exception as e:
            print(f"[TradeManager] WARNING: Could not persist management state: {e}")


# ==============================================================
# MT5 HELPERS
# ==============================================================

def get_live_position(ticket, symbol):
    """Returns the live MT5 position object or None."""
    # B-07 FIX: try/finally guarantees mt5.shutdown() even if positions_get raises
    if not mt5.initialize():
        return None
    try:
        positions = mt5.positions_get(ticket=ticket)
        return positions[0] if positions else None
    finally:
        mt5.shutdown()


def get_recent_m5_candles(symbol, count=10):
    """Fetches recent M5 OHLC data for swing structure detection."""
    # B-07 FIX: try/finally guarantees mt5.shutdown() even if copy_rates raises
    if not mt5.initialize():
        return None
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
        return rates if (rates is not None and len(rates) > 0) else None
    finally:
        mt5.shutdown()


_digits_cache = {}

def _get_digits(symbol):
    """Returns the symbol's decimal precision."""
    global _digits_cache
    if symbol in _digits_cache:
        return _digits_cache[symbol]

    # B-07 FIX: try/finally guarantees mt5.shutdown() even if symbol_info raises
    if not mt5.initialize():
        return 2
    try:
        si = mt5.symbol_info(symbol)
        digits = si.digits if si else 2
        _digits_cache[symbol] = digits
        return digits
    finally:
        mt5.shutdown()


# ==============================================================
# HYBRID MANAGEMENT CONSTANTS
# ==============================================================

REVERSAL_CLOSE_THRESHOLD    = 0.65   # REVERSAL ≥ this conf → immediate full close
TREND_AI_THRESHOLD          = 0.70   # BULL/BEAR ≥ this conf → AI management call
COMPRESSION_REGIMES         = {"COMPRESSION", "LOW_VOL_RANGE", "LOW_VOLATILITY"}
TREND_REGIMES               = {"BULL_TREND", "BEAR_TREND"}

# FIX #4 — ATR-based trail buffer replaces static dollar amounts.
# Gold's M5 ATR during NY open = $1.50–$4.00. A static $0.30 buffer is < 1 ATR
# and gets wicked out constantly on liquidity sweeps before the trend continues.
#
# New dynamic formula:
#   MECHANICAL path : recent_low - (ATR * ATR_TRAIL_MULT_DEFAULT)   ← 0.5× ATR
#   TRAIL_AGGRESSIVE: recent_low - (ATR * ATR_TRAIL_MULT_AGGRESSIVE) ← 0.3× ATR (tighter)
#
# Floor values prevent absurdly tight stops during very low-vol periods.
ATR_TRAIL_MULT_DEFAULT      = 0.5    # mechanical path: half-ATR buffer beyond swing low/high
ATR_TRAIL_MULT_AGGRESSIVE   = 0.3    # AI TRAIL_AGGRESSIVE: tighter but still ATR-proportional
ATR_TRAIL_FLOOR_DEFAULT     = 0.30   # minimum $0.30 floor — prevents < 1pt stop on dead markets
ATR_TRAIL_FLOOR_AGGRESSIVE  = 0.15   # minimum $0.15 floor for aggressive path

# Early partial threshold for COMPRESSION AI path
COMPRESSION_PARTIAL_R       = 0.75   # Take 75% partial at 0.75R (not 1R)
COMPRESSION_PARTIAL_PCT     = 0.75   # 75% of position closed
COMPRESSION_BE_R            = 0.50   # Move SL to BE at 0.5R (not 1R)


def _compute_atr(candles: list, period: int = 10) -> float:
    """
    FIX #4: Compute Average True Range from a list of candle dicts.
    Each dict must have 'high', 'low', 'close' keys.
    Returns ATR in price units (e.g. 1.50 = $1.50 on XAUUSD).
    Falls back to ATR_TRAIL_FLOOR_DEFAULT if candles are insufficient.
    """
    if not candles or len(candles) < 2:
        return ATR_TRAIL_FLOOR_DEFAULT
    try:
        trs = []
        for i in range(1, min(period + 1, len(candles))):
            h    = float(candles[i]['high'])
            lo   = float(candles[i]['low'])
            prev = float(candles[i - 1]['close'])
            trs.append(max(h - lo, abs(h - prev), abs(lo - prev)))
        if not trs:
            return ATR_TRAIL_FLOOR_DEFAULT
        return max(ATR_TRAIL_FLOOR_DEFAULT, sum(trs) / len(trs))
    except Exception:
        return ATR_TRAIL_FLOOR_DEFAULT


# ==============================================================
# AI MANAGEMENT CALL
# Builds a focused management-only prompt (NOT an entry prompt).
# Returns a parsed action dict or None on failure.
# ==============================================================

def _call_management_ai(regime: str, confidence: float, direction: str,
                         current_price: float, entry: float, sl: float, tp: float,
                         stage: int, profit_r: float, sensory_math: dict = None) -> dict:
    """
    Calls the AI for a trade management decision only.
    The prompt is minimal by design — the AI should not re-analyse entry logic,
    only decide HOW to manage the open position given the current regime.

    Returns dict with "action" key, or None if the AI call fails/times out.

    Valid actions:
      Reversal path:    CLOSE | HOLD
      Trend path:       HOLD | TRAIL_AGGRESSIVE | SKIP_PARTIAL
      Compression path: HOLD | PARTIAL_75 | TIGHT_BE

    Never raises — always returns None on any failure so mechanical fallback fires.
    """
    call_ai = _get_call_ai()
    if call_ai is None:
        return None

    # Load dynamic RAG memory block
    memory_context = ""
    try:
        from Integration.Wisdom_Worker.context_retriever import get_full_memory_context
        # Use "trade_management" as dummy context to fetch all Layer 1 rules/lessons
        memory_context = get_full_memory_context("trade_management")
    except Exception as e:
        print(f"[HybridManager] Warning: could not load memory context: {e}")

    # Format sensory math block if available
    sensory_text = ""
    if sensory_math:
        sensory_text = (
            "\n--- REGIME SENSORY MATH ---\n"
            f"Binary Reversal Prob: {sensory_math.get('reversal_prob', 0.0):.1%}\n"
            f"Trend Strength (ADX): {sensory_math.get('h1_adx', 0.0):.1f}\n"
            f"BB Width/Expansion  : {sensory_math.get('h1_bb_width', 0.0):.2f}\n"
            f"Short-Term ATR Ratio: {sensory_math.get('h1_atr_ratio', 0.0):.2f}\n"
            f"Upper Wick Rejection: {sensory_math.get('h1_upper_wick_ratio', 0.0):.1%}\n"
            f"Lower Wick Rejection: {sensory_math.get('h1_lower_wick_ratio', 0.0):.1%}\n"
        )

    risk        = abs(entry - sl)
    to_tp_r     = abs(tp - current_price) / risk if risk > 0 else 0

    if regime == "REVERSAL":
        action_options = "CLOSE, HOLD"
        guidance = (
            "- CLOSE: the reversal is structurally real and represents high risk; close full position immediately\n"
            "- HOLD: the reversal is likely a short-lived fakeout/pullback; do NOT close, let the trade run"
        )
        valid_actions = {"CLOSE", "HOLD"}
    elif regime in TREND_REGIMES:
        action_options = "HOLD, TRAIL_AGGRESSIVE, SKIP_PARTIAL"
        guidance = (
            "- HOLD: price structure intact, keep current SL and TP unchanged\n"
            "- TRAIL_AGGRESSIVE: trend is strong, tighten trailing stop to ~0.3× ATR "
            "  beyond the last M5 swing low/high (typically $0.45–$1.20 on Gold M5). "
            "  Lock in maximum gains while giving the trade room to breathe.\n"
            "- SKIP_PARTIAL: skip the 50% partial close at 1R — let full position "
            "  run to TP because trend momentum justifies it"
        )
        valid_actions = {"HOLD", "TRAIL_AGGRESSIVE", "SKIP_PARTIAL"}
    else:
        action_options = "HOLD, PARTIAL_75, TIGHT_BE"
        guidance = (
            "- HOLD: structure still valid, keep current management plan\n"
            "- PARTIAL_75: close 75% of position NOW (price is near 0.75R) — "
            "  range/compression will revert before TP; lock in most of the profit\n"
            "- TIGHT_BE: move SL to break-even immediately (at current price) — "
            "  compression means TP is unlikely; protect capital"
        )
        valid_actions = {"HOLD", "PARTIAL_75", "TIGHT_BE"}

    prompt = f"""You are managing an OPEN XAUUSD trade. Do NOT suggest a new entry.
Only decide how to manage the existing position.

{memory_context}

--- OPEN TRADE STATE ---
Direction  : {direction}
Entry      : {entry:.2f}
SL         : {sl:.2f}
TP         : {tp:.2f}
Current    : {current_price:.2f}
Profit     : {profit_r:.2f}R  ({'+' if profit_r >= 0 else ''}{(current_price - entry) if direction == 'BUY' else (entry - current_price):.2f} pts)
Remaining  : {to_tp_r:.2f}R to TP
Stage      : {stage} (0=watching, 1=BE set, 2=trailing)

--- REGIME CONTEXT ---
Regime     : {regime}
Confidence : {confidence:.0%}
{sensory_text}
--- MANAGEMENT OPTIONS ---
{guidance}

--- TASK ---
Choose exactly ONE action from: {action_options}

Respond ONLY with valid JSON. No other text.
Example: {{"action": "HOLD", "reasoning": "Trend intact, no reason to adjust."}}
"""

    try:
        raw = call_ai(prompt=prompt)
        if not raw:
            return None
        # Strip markdown fences if present
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*',     '', raw)
        # Find JSON object in response
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            action = parsed.get("action", "").upper().strip()
            if action in valid_actions:
                print(f"[HybridManager] AI decision: {action} — {parsed.get('reasoning','')[:120]}")
                return parsed
    except Exception as e:
        print(f"[HybridManager] AI call failed ({e}) — using mechanical fallback.")
    return None


# ==============================================================
# HYBRID REGIME DISPATCHER
# Entry point called by manage_trade() when regime_result is provided.
# Returns a management result dict OR None (= fall through to mechanical).
# ==============================================================

def _hybrid_dispatch(regime_result: dict, symbol: str, ticket, pos,
                      entry_price: float, sl_price: float, tp_price: float,
                      direction: str, disk_state: dict) -> dict | None:
    """
    Decides whether a regime-specific action should override the mechanical path.

    Returns:
        dict  → use this result directly (action taken, skip mechanical)
        None  → no override, let manage_trade() run its mechanical logic

    Priority:
      1. REVERSAL  → hardcoded full close (fastest, no AI)
      2. TREND     → AI management call → execute AI decision
      3. COMPRESSION → AI management call → execute AI decision
      4. Anything else → None (mechanical)
    """
    if not regime_result:
        return None

    regime     = regime_result.get("regime", "")
    confidence = float(regime_result.get("confidence") or 0.0)
    true_stage = disk_state.get("management_stage", 0)

    current_price = float(pos.price_current)
    risk          = abs(entry_price - sl_price)
    profit_dist   = (current_price - entry_price) if direction == "BUY" \
                    else (entry_price - current_price)
    profit_r      = (profit_dist / risk) if risk > 0 else 0.0

    digits = _get_digits(symbol)

    # ── 1. REVERSAL — dynamic override route ────────────────────
    if regime == "REVERSAL" and confidence >= REVERSAL_CLOSE_THRESHOLD:
        print(f"[HybridManager] 🔴 REVERSAL ({confidence:.0%}) detected — Calling AI for dynamic exit override.")
        sens_math = regime_result.get("sensory_math", {})
        ai_result = _call_management_ai(
            regime, confidence, direction,
            current_price, entry_price, sl_price, tp_price,
            true_stage, profit_r, sensory_math=sens_math
        )

        action = "CLOSE"  # default fallback if AI fails or times out
        if ai_result:
            action = ai_result.get("action", "CLOSE").upper().strip()

        if action == "HOLD":
            print(f"[HybridManager] 🤖 AI Overrode REVERSAL! Holding position based on dynamic sensory math.")
            return {
                "stage":         true_stage,
                "sl_price":      sl_price,
                "action_taken":  f"REVERSAL HELD (AI Override): {regime} {confidence:.0%} conf. "
                                 f"Position held based on structural math.",
                "hybrid_action": "REVERSAL_HOLD_OVERRIDE",
            }
        else:
            print(f"[HybridManager] 🔴 REVERSAL Close Confirmed (AI choice: {action}).")
            result = close_position(ticket, symbol,
                                    comment=f"HybridMgr: REVERSAL ({confidence:.0%})")
            if result and hasattr(result, 'retcode') and \
               result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    "stage":         true_stage,
                    "sl_price":      sl_price,
                    "action_taken":  f"REVERSAL CLOSE: regime={regime} "
                                     f"conf={confidence:.0%}. Full position closed.",
                    "hybrid_action": "REVERSAL_CLOSE",
                    "force_close":   True,        # main_bot checks this to reset _active_trade
                }
            else:
                # close_position failed — fall through to mechanical (better than nothing)
                print(f"[HybridManager] WARNING: close_position() failed for REVERSAL. "
                      f"Falling through to mechanical management.")
                return None

    # ── 2. TREND regime with high confidence — AI management call ──────
    if regime in TREND_REGIMES and confidence > TREND_AI_THRESHOLD:
        print(f"[HybridManager] 📈 {regime} ({confidence:.0%}) — Calling AI for management.")
        sens_math = regime_result.get("sensory_math", {})
        ai_result = _call_management_ai(
            regime, confidence, direction,
            current_price, entry_price, sl_price, tp_price,
            true_stage, profit_r, sensory_math=sens_math
        )

        if ai_result is None:
            print(f"[HybridManager] AI unavailable — mechanical fallback for {regime}.")
            return None   # fall through to mechanical

        action = ai_result.get("action", "HOLD").upper()

        if action == "SKIP_PARTIAL":
            # Don't take the 50% partial at 1R — mark partial_taken so it won't fire
            if true_stage == 0 and profit_r >= 1.0 and \
               not disk_state.get("partial_taken"):
                _write_mgmt_state(ticket, {"partial_taken": True,
                                           "skip_partial_reason": "AI: TREND momentum"})
                print(f"[HybridManager] SKIP_PARTIAL — full position stays open "
                      f"({regime} {confidence:.0%} conf).")
            return {
                "stage":         true_stage,
                "sl_price":      sl_price,
                "action_taken":  f"SKIP_PARTIAL (AI): {regime} {confidence:.0%} conf. "
                                 f"Full position held through 1R.",
                "hybrid_action": "SKIP_PARTIAL",
            }

        elif action == "TRAIL_AGGRESSIVE":
            # FIX #4: ATR-proportional buffer (was fixed $0.10).
            # ATR_TRAIL_MULT_AGGRESSIVE × M5 ATR, floored at ATR_TRAIL_FLOOR_AGGRESSIVE.
            if true_stage >= 1:
                candles = get_recent_m5_candles(symbol, count=10)
                if candles is not None:
                    completed = candles[:-1]
                    if len(completed) < 2:
                        return {"stage": true_stage, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                                "action_taken": "M5 candles insufficient (< 2 completed). Skipping trail this cycle."}
                    atr       = _compute_atr(completed)
                    dyn_buf   = max(ATR_TRAIL_FLOOR_AGGRESSIVE,
                                    round(atr * ATR_TRAIL_MULT_AGGRESSIVE, 2))
                    if direction == "BUY":
                        recent_low   = min(c['low'] for c in completed[-5:])
                        candidate_sl = round(recent_low - dyn_buf, digits)
                        if candidate_sl > sl_price:
                            if modify_position_sl(ticket, symbol, candidate_sl):
                                _write_mgmt_state(ticket, {
                                    "current_sl":       candidate_sl,
                                    "management_stage": 2,
                                })
                                print(f"[HybridManager] TRAIL_AGGRESSIVE BUY → SL {candidate_sl:.2f} "
                                      f"(ATR={atr:.2f} × {ATR_TRAIL_MULT_AGGRESSIVE} = ${dyn_buf:.2f} buffer)")
                                return {
                                    "stage":         2,
                                    "sl_price":      candidate_sl,
                                    "action_taken":  f"TRAIL_AGGRESSIVE (AI): SL → {candidate_sl:.2f} "
                                                     f"(ATR-buffer ${dyn_buf:.2f}).",
                                    "hybrid_action": "TRAIL_AGGRESSIVE",
                                }
                    elif direction == "SELL":
                        recent_high  = max(c['high'] for c in completed[-5:])
                        candidate_sl = round(recent_high + dyn_buf, digits)
                        if candidate_sl < sl_price:
                            if modify_position_sl(ticket, symbol, candidate_sl):
                                _write_mgmt_state(ticket, {
                                    "current_sl":       candidate_sl,
                                    "management_stage": 2,
                                })
                                print(f"[HybridManager] TRAIL_AGGRESSIVE SELL → SL {candidate_sl:.2f} "
                                      f"(ATR={atr:.2f} × {ATR_TRAIL_MULT_AGGRESSIVE} = ${dyn_buf:.2f} buffer)")
                                return {
                                    "stage":         2,
                                    "sl_price":      candidate_sl,
                                    "action_taken":  f"TRAIL_AGGRESSIVE (AI): SL → {candidate_sl:.2f} "
                                                     f"(ATR-buffer ${dyn_buf:.2f}).",
                                    "hybrid_action": "TRAIL_AGGRESSIVE",
                                }
                # Trail not possible this cycle (no new swing improvement) — fall through to mechanical
                return None

            else:
                # UB-03 FIX: AI returned TRAIL_AGGRESSIVE at stage 0 (no partial yet).
                # The intent is "let the full position run — skip the 50% partial".
                # The OLD code hit `return None` here which fell through to the mechanical
                # block, which at stage 0 executes the 50% partial + break-even — the
                # EXACT OPPOSITE of what TRAIL_AGGRESSIVE means.
                #
                # Fix: immediately mark partial_taken=True so mechanical NEVER fires
                # the stage-0 partial close, then return a holding result so the
                # position is untouched this cycle.  Trailing activates naturally
                # when the position moves past stage 1.
                _write_mgmt_state(ticket, {
                    "partial_taken":       True,
                    "skip_partial_reason": "AI: TRAIL_AGGRESSIVE at stage 0 — holding full position",
                    "management_stage":    0,
                })
                print(f"[HybridManager] TRAIL_AGGRESSIVE at stage 0 — partial suppressed. "
                      f"Full position held, mechanical partial guard disabled.")
                return {
                    "stage":         0,
                    "sl_price":      sl_price,
                    "action_taken":  "TRAIL_AGGRESSIVE (AI): Stage 0 — 50% partial suppressed. Full position held.",
                    "hybrid_action": "TRAIL_AGGRESSIVE",
                }

        else:  # HOLD
            return {
                "stage":         true_stage,
                "sl_price":      sl_price,
                "action_taken":  f"HOLD (AI): {regime} {confidence:.0%} conf. "
                                 f"Position unchanged this cycle.",
                "hybrid_action": "HOLD",
            }

    # ── 3. COMPRESSION / LOW_VOL — AI management call ──────────────────
    if regime in COMPRESSION_REGIMES:
        print(f"[HybridManager] 📉 {regime} ({confidence:.0%}) — Calling AI for management.")
        sens_math = regime_result.get("sensory_math", {})
        ai_result = _call_management_ai(
            regime, confidence, direction,
            current_price, entry_price, sl_price, tp_price,
            true_stage, profit_r, sensory_math=sens_math
        )

        if ai_result is None:
            print(f"[HybridManager] AI unavailable — mechanical fallback for {regime}.")
            return None

        action = ai_result.get("action", "HOLD").upper()

        if action == "PARTIAL_75" and true_stage == 0:
            # Early 75% partial at COMPRESSION_PARTIAL_R (0.75R) — before 1R
            if profit_r >= COMPRESSION_PARTIAL_R and \
               not disk_state.get("partial_taken"):
                if mt5.initialize():
                    try:
                        si = mt5.symbol_info(symbol)
                    finally:
                        mt5.shutdown()
                    if si:
                        pos_live = get_live_position(ticket, symbol)
                        if pos_live:
                            step      = float(si.volume_step)
                            vol_close = round((pos_live.volume * COMPRESSION_PARTIAL_PCT)
                                             / step) * step
                            vol_close = max(float(si.volume_min), vol_close)
                            close_result = partial_close(ticket, symbol, vol_close)
                            if close_result and hasattr(close_result, 'retcode') and \
                               close_result.retcode == mt5.TRADE_RETCODE_DONE:
                                _write_mgmt_state(ticket, {"partial_taken": True,
                                                            "partial_close_time": time.time()})
                                print(f"[HybridManager] PARTIAL_75: closed {vol_close} lots "
                                      f"at {profit_r:.2f}R ({COMPRESSION_PARTIAL_R}R threshold).")
                                return {
                                    "stage":         0,
                                    "sl_price":      sl_price,
                                    "action_taken":  f"PARTIAL_75 (AI): {COMPRESSION_PARTIAL_PCT*100:.0f}% "
                                                     f"closed at {profit_r:.2f}R ({regime}).",
                                    "hybrid_action": "PARTIAL_75",
                                }
            # Not enough profit for early partial yet — nothing to do this cycle
            return {
                "stage":         true_stage,
                "sl_price":      sl_price,
                "action_taken":  f"PARTIAL_75 (AI): waiting for {COMPRESSION_PARTIAL_R}R "
                                 f"(currently {profit_r:.2f}R).",
                "hybrid_action": "PARTIAL_75_WAITING",
            }

        elif action == "TIGHT_BE" and true_stage == 0:
            # Move SL to break-even NOW (at COMPRESSION_BE_R, before 1R)
            if profit_r >= COMPRESSION_BE_R:
                new_sl = round(float(entry_price), digits)
                if modify_position_sl(ticket, symbol, new_sl):
                    _write_mgmt_state(ticket, {
                        "break_even_set":   True,
                        "current_sl":       new_sl,
                        "management_stage": 1,
                    })
                    print(f"[HybridManager] TIGHT_BE: SL → {new_sl:.2f} "
                          f"at {profit_r:.2f}R (early BE for {regime}).")
                    return {
                        "stage":         1,
                        "sl_price":      new_sl,
                        "action_taken":  f"TIGHT_BE (AI): SL → BE at {profit_r:.2f}R "
                                         f"({regime}, early protection).",
                        "hybrid_action": "TIGHT_BE",
                    }
            return {
                "stage":         true_stage,
                "sl_price":      sl_price,
                "action_taken":  f"TIGHT_BE (AI): waiting for {COMPRESSION_BE_R}R "
                                 f"(currently {profit_r:.2f}R).",
                "hybrid_action": "TIGHT_BE_WAITING",
            }

        else:  # HOLD
            return {
                "stage":         true_stage,
                "sl_price":      sl_price,
                "action_taken":  f"HOLD (AI): {regime} {confidence:.0%}. "
                                 f"No action this cycle.",
                "hybrid_action": "HOLD",
            }

    # ── 4. All other regimes → mechanical ──────────────────────────────
    return None


# ==============================================================
# MAIN MANAGEMENT FUNCTION
# ==============================================================

def manage_trade(symbol, ticket, entry_price, sl_price, tp_price,
                 direction, stage, regime_result=None):
    """
    Hybrid trade management. Called every 5-minute cycle from main_bot.py.

    Parameters:
        symbol        : e.g. "XAUUSD"
        ticket        : MT5 position ticket (int)
        entry_price   : original entry price (float)
        sl_price      : the SL price stored in main_bot's _active_trade dict
        tp_price      : TP price (float)
        direction     : "BUY" or "SELL"
        stage         : the stage stored in main_bot's _active_trade dict
                        (used only as a fallback if disk state is missing)
        regime_result : dict from regime_detector.predict() — optional.
                        When provided, enables hybrid regime-aware management.
                        When None, runs pure mechanical management (legacy mode).

    Returns:
        dict: {
            "stage"        : updated management stage (int),
            "sl_price"     : updated SL price (float),
            "action_taken" : human-readable description (str),
            "hybrid_action": which path fired (str) — for logging/backtest,
            "force_close"  : True if a hardcoded close was executed (bool, optional)
        }
    """
    # -----------------------------------------------------------
    # STEP 1: Recover true state from disk (BUG 2 FIX)
    # -----------------------------------------------------------
    disk_state     = _read_mgmt_state(ticket)
    true_stage     = disk_state["management_stage"]
    partial_taken  = disk_state["partial_taken"]
    break_even_set = disk_state["break_even_set"]

    if disk_state["current_sl"] is not None:
        sl_price = disk_state["current_sl"]

    # -----------------------------------------------------------
    # STEP 2: Confirm position is still live
    # -----------------------------------------------------------
    pos = get_live_position(ticket, symbol)
    if pos is None:
        return {"stage": true_stage, "sl_price": sl_price,
                "action_taken": "Position not found in MT5 (already closed or TP/SL hit by broker).",
                "hybrid_action": "NONE"}

    current_price = pos.price_current
    risk          = abs(entry_price - sl_price)

    if risk == 0:
        return {"stage": true_stage, "sl_price": sl_price,
                "action_taken": "SL distance is zero — skipping management to avoid division error.",
                "hybrid_action": "NONE"}

    # -----------------------------------------------------------
    # STEP 3: HYBRID REGIME DISPATCH (runs before mechanical)
    # Returns a result dict → caller uses it directly.
    # Returns None → fall through to mechanical logic below.
    # -----------------------------------------------------------
    if regime_result:
        hybrid_result = _hybrid_dispatch(
            regime_result=regime_result,
            symbol=symbol, ticket=ticket, pos=pos,
            entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
            direction=direction, disk_state=disk_state,
        )
        if hybrid_result is not None:
            return hybrid_result
        # None → continue to mechanical management below

    action_taken = "WATCHING — price has not yet reached 1R profit."

    # ===========================================================
    # STAGE 0 — Wait for 1R profit, then: partial close + break-even
    # ===========================================================
    if true_stage == 0:

        profit_distance = (current_price - entry_price) if direction == "BUY" \
                          else (entry_price - current_price)

        if profit_distance < risk:
            # Not at 1R yet — just report floating profit
            profit_r = profit_distance / risk
            return {"stage": 0, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                    "action_taken": f"WATCHING — {profit_r:.2f}R in profit. Waiting for 1R to trigger break-even."}

        # -------------------------------------------------------
        # 1R REACHED
        # -------------------------------------------------------

        # --- PARTIAL CLOSE (only if not already taken) ---
        if not partial_taken:
            if mt5.initialize():
                try:        # H-01 FIX: try/finally guarantees mt5.shutdown()
                    si = mt5.symbol_info(symbol)
                finally:
                    mt5.shutdown()
                if si:
                    step       = float(si.volume_step)
                    vol_close  = round((pos.volume * 0.5) / step) * step
                    vol_close  = max(float(si.volume_min), vol_close)

                    close_result = partial_close(ticket, symbol, vol_close)

                    if close_result and hasattr(close_result, 'retcode') and \
                       close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"[TradeManager] 1R hit! Partial closed {vol_close} lots at {current_price:.2f}.")
                        _write_mgmt_state(ticket, {"partial_taken": True})
                        partial_taken = True
                    else:
                        err = close_result.comment if close_result else "No response"
                        print(f"[TradeManager] WARNING: Partial close failed: {err}. Will retry next cycle.")
                        return {"stage": 0, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                                "action_taken": f"1R hit but partial close failed ({err}). Retrying next cycle."}
        else:
            print(f"[TradeManager] Partial already taken (recovered from disk). Skipping.")

        # -------------------------------------------------------
        # BUG-13 FIX: Replaced blocking time.sleep(1.5) with a
        # non-blocking timestamp check.
        # The broker still needs ~1.5s to process the partial close
        # before accepting an SLTP modification, but sleeping blocks
        # the entire main trading loop (and compounds on every call).
        # Instead: record the partial_close_time, and skip the SL
        # modification this cycle if not enough time has elapsed.
        # The SL modification will succeed on the next cycle (~5 min).
        # -------------------------------------------------------
        import time as _time
        # B-01 FIX: was _state.get() — variable is disk_state (re-read to pick up partial_close_time)
        disk_state   = _read_mgmt_state(ticket)
        partial_time = disk_state.get("partial_close_time")
        now_ts       = _time.time()

        if partial_time is None:
            # First time through — record timestamp and come back next cycle
            _write_mgmt_state(ticket, {"partial_close_time": now_ts})
            print("[TradeManager] Partial close recorded. SL modification deferred 1 cycle for broker processing.")
            # B-11 FIX: include sl_price so caller's _active_trade["sl"] assignment doesn't KeyError
            return {"stage": 0, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                    "action_taken": "Partial closed. SL move deferred to next cycle."}

        if now_ts - partial_time < 2.0:
            # Not enough time elapsed yet — skip this cycle
            print("[TradeManager] Waiting for broker to process partial (< 2s elapsed). Skipping SL move this cycle.")
            # B-11 FIX: include sl_price
            return {"stage": 0, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                    "action_taken": "Waiting for broker partial confirmation."}

        # Sufficient time has passed — proceed with SL modification

        # --- MOVE SL TO BREAK-EVEN ---
        digits = _get_digits(symbol)
        new_sl = round(float(entry_price), digits)

        sl_success = modify_position_sl(ticket, symbol, new_sl)

        if sl_success:
            print(f"[TradeManager] Break-even SL set at {new_sl:.2f}")
            _write_mgmt_state(ticket, {
                "break_even_set":   True,
                "current_sl":       new_sl,
                "management_stage": 1,
            })
            return {"stage": 1, "sl_price": new_sl, "hybrid_action": "MECHANICAL",
                    "action_taken": f"1R hit. 50% closed. SL moved to break-even ({new_sl:.2f})."}
        else:
            print(f"[TradeManager] WARNING: Break-even SL modification failed. Will retry next cycle.")
            return {"stage": 0, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                    "action_taken": "Partial closed. Break-even SL failed — will retry next cycle."}

    # ===========================================================
    # STAGE 1 / 2 — Trailing Stop on M5 swing structure
    # ===========================================================
    if true_stage >= 1:

        candles = get_recent_m5_candles(symbol, count=10)
        if candles is None:
            return {"stage": true_stage, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                    "action_taken": "Could not pull M5 candles for trailing stop. Skipping cycle."}

        # Use last 5 *completed* candles (exclude index -1 which is still forming)
        completed = candles[:-1]
        if len(completed) < 2:
            return {"stage": true_stage, "sl_price": sl_price, "hybrid_action": "MECHANICAL",
                    "action_taken": "M5 candles insufficient (< 2 completed). Skipping trail this cycle."}
        # FIX #4: ATR-proportional buffer (was fixed $0.30).
        # Gold M5 ATR during NY open is $1.50–$4.00. A $0.30 buffer is < 1 ATR
        # and gets wicked on every liquidity sweep before the trend continues.
        # Dynamic formula: max(floor, ATR × multiplier)
        atr    = _compute_atr(completed)
        buffer = max(ATR_TRAIL_FLOOR_DEFAULT,
                     round(atr * ATR_TRAIL_MULT_DEFAULT, 2))

        if direction == "BUY":
            recent_low   = min(c['low'] for c in completed[-5:])
            candidate_sl = round(recent_low - buffer, _get_digits(symbol))

            if candidate_sl > sl_price:   # Only move SL in the profitable direction
                success = modify_position_sl(ticket, symbol, candidate_sl)
                if success:
                    print(f"[TradeManager] Trailing SL raised to {candidate_sl:.2f} (BUY) "
                          f"[ATR={atr:.2f} × {ATR_TRAIL_MULT_DEFAULT} = ${buffer:.2f} buffer]")
                    _write_mgmt_state(ticket, {
                        "current_sl":       candidate_sl,
                        "management_stage": 2,
                    })
                    return {"stage": 2, "sl_price": candidate_sl, "hybrid_action": "MECHANICAL",
                            "action_taken": f"Trailing SL raised to {candidate_sl:.2f} (ATR-buffer ${buffer:.2f})."}

        elif direction == "SELL":
            recent_high  = max(c['high'] for c in completed[-5:])
            candidate_sl = round(recent_high + buffer, _get_digits(symbol))

            if candidate_sl < sl_price:   # Only move SL in the profitable direction
                success = modify_position_sl(ticket, symbol, candidate_sl)
                if success:
                    print(f"[TradeManager] Trailing SL lowered to {candidate_sl:.2f} (SELL) "
                          f"[ATR={atr:.2f} × {ATR_TRAIL_MULT_DEFAULT} = ${buffer:.2f} buffer]")
                    _write_mgmt_state(ticket, {
                        "current_sl":       candidate_sl,
                        "management_stage": 2,
                    })
                    return {"stage": 2, "sl_price": candidate_sl, "hybrid_action": "MECHANICAL",
                            "action_taken": f"Trailing SL lowered to {candidate_sl:.2f} (ATR-buffer ${buffer:.2f})."}

        action_taken = (f"Trailing active (Stage {true_stage}). "
                        f"Current SL: {sl_price:.2f}. No new swing to trail to yet.")

    return {"stage": true_stage, "sl_price": sl_price, "action_taken": action_taken,
            "hybrid_action": "MECHANICAL"}
