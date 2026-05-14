"""
regime_router.py — v12 Regime-Aware Trade Router
==================================================
Four-layer system that sits between the regime sensor (regime_detector.py)
and Claude.  Evaluates every potential trade through objective regime rules
BEFORE burning an API token.

LAYER 1 — CONFIDENCE GATE
    Skips Claude entirely if regime confidence < GATE_MIN_CONFIDENCE.
    Low-confidence regimes mean the model is uncertain about market state —
    trading on uncertain regime data degrades edge.

LAYER 2 — DUAL CONFIRMATION GATE
    Both regime AND Claude's signal must agree directionally.
    BULL_TREND + BUY  → full size allowed
    BULL_TREND + SELL → counter-regime → BLOCKED (or 50% size if relaxed)
    LOW_VOL_RANGE     → any direction allowed (mean-reversion context)
    REVERSAL          → allowed but size capped at 50%

LAYER 3 — SL/TP REGIME ADJUSTER
    Widens or tightens Claude's SL/TP levels based on regime volatility.
    REVERSAL / BULL_TREND / BEAR_TREND at high confidence → wider SL
    COMPRESSION / LOW_VOL_RANGE → tighter TP (capture profit before MR)

LAYER 4 — LIVE TRADE ADAPTER
    Mid-trade regime shifts trigger SL/TP modifications without an AI call.
    REVERSAL during open trade  → request partial or full close
    COMPRESSION during BUY/SELL → tighten TP to capture before reversal
    No cost: zero API calls — pure local ML.
"""

import os
import sys

# ── Path setup ────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.normpath(os.path.join(current_dir, '..', '..'))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from master_controls import (
        GATE_MIN_CONFIDENCE,
        REGIME_ROUTER_DUAL_STRICT,
        REGIME_ROUTER_REVERSAL_SIZE,
        REGIME_ROUTER_SL_MULTIPLIER_TREND,
        REGIME_ROUTER_SL_MULTIPLIER_REVERSAL,
        REGIME_ROUTER_TP_MULTIPLIER_RANGE,
    )
except ImportError:
    # Safe defaults when master_controls is unavailable or missing keys
    GATE_MIN_CONFIDENCE                = 0.40
    REGIME_ROUTER_DUAL_STRICT          = True    # False = allow counter-regime at 50% size
    REGIME_ROUTER_REVERSAL_SIZE        = 0.50    # REVERSAL regime max size multiplier
    REGIME_ROUTER_SL_MULTIPLIER_TREND  = 1.10   # widen SL 10% in strong trends
    REGIME_ROUTER_SL_MULTIPLIER_REVERSAL = 1.25  # widen SL 25% in REVERSAL (wicks)
    REGIME_ROUTER_TP_MULTIPLIER_RANGE  = 0.85   # tighten TP 15% in range/compression

# Regime constants
REGIME_BULL_TREND    = "BULL_TREND"
REGIME_BEAR_TREND    = "BEAR_TREND"
REGIME_LOW_VOL_RANGE = "LOW_VOL_RANGE"
REGIME_COMPRESSION   = "COMPRESSION"
REGIME_REVERSAL      = "REVERSAL"
REGIME_UNCERTAIN     = "UNCERTAIN"

_DIRECTIONAL_REGIMES = {REGIME_BULL_TREND, REGIME_BEAR_TREND}
_BLOCKED_REGIMES     = {REGIME_COMPRESSION, REGIME_UNCERTAIN}


# ================================================================
# LAYER 1 — CONFIDENCE GATE
# ================================================================

def check_confidence_gate(regime_result: dict) -> tuple:
    """
    Returns (allowed: bool, reason: str).

    Blocks if:
      - Regime is UNCERTAIN (model confidence below UNCERTAIN_THRESHOLD)
      - Confidence < GATE_MIN_CONFIDENCE
      - Regime is COMPRESSION (hard block — no entry in pre-breakout squeeze)

    Allows if:
      - Regime is any other state and confidence >= GATE_MIN_CONFIDENCE
      - Regime is LOW_VOL_RANGE (mean-reversion context — confidence not required)
    """
    regime     = regime_result.get("regime", REGIME_LOW_VOL_RANGE)
    confidence = regime_result.get("confidence")

    # Hard block — COMPRESSION and UNCERTAIN are always blocked regardless of confidence
    if regime == REGIME_COMPRESSION:
        return (False,
                f"COMPRESSION regime — market is coiling pre-breakout. "
                f"No entry until expansion candle confirms direction.")
    if regime == REGIME_UNCERTAIN:
        conf_str = f"{confidence:.0%}" if confidence else "N/A"
        return (False,
                f"UNCERTAIN regime — model confidence {conf_str} below "
                f"UNCERTAIN_THRESHOLD. No trade until clarity.")

    # LOW_VOL_RANGE — mean-reversion context, confidence is informational only
    if regime == REGIME_LOW_VOL_RANGE:
        return (True,
                f"LOW_VOL_RANGE — mean-reversion context, confidence gate relaxed.")

    # Directional and REVERSAL regimes — require minimum confidence
    if confidence is None:
        # Fallback mode (no XGBoost model) — allow but note
        return (True,
                f"Fallback mode (no model confidence). Proceeding with caution.")

    if confidence < GATE_MIN_CONFIDENCE:
        return (False,
                f"{regime} confidence {confidence:.0%} < "
                f"minimum {GATE_MIN_CONFIDENCE:.0%}. "
                f"Regime sensor is uncertain — skipping Claude call to save tokens.")

    return (True,
            f"{regime} confidence {confidence:.0%} ≥ {GATE_MIN_CONFIDENCE:.0%}. "
            f"Confidence gate passed.")


# ================================================================
# LAYER 2 — DUAL CONFIRMATION GATE
# ================================================================

def check_dual_confirmation(regime_result: dict, signal: str) -> tuple:
    """
    Returns (allowed: bool, reason: str, size_multiplier: float).

    Validates that Claude's signal agrees with the regime direction.

    BULL_TREND + BUY   → full size (1.0)
    BULL_TREND + SELL  → blocked if REGIME_ROUTER_DUAL_STRICT else 50% size
    BEAR_TREND + SELL  → full size (1.0)
    BEAR_TREND + BUY   → blocked if REGIME_ROUTER_DUAL_STRICT else 50% size
    REVERSAL  + any    → 50% size (volatility cap)
    LOW_VOL   + any    → full size (mean-reversion, any direction valid)
    COMPRESSION        → should have been blocked by confidence gate; block here too
    """
    regime     = regime_result.get("regime", REGIME_LOW_VOL_RANGE)
    confidence = regime_result.get("confidence") or 0.0
    signal     = signal.upper()

    if regime in _BLOCKED_REGIMES:
        return (False,
                f"{regime} regime is blocked — no entry allowed.",
                0.0)

    if regime == REGIME_REVERSAL:
        return (True,
                f"REVERSAL regime — all directions valid. "
                f"Size capped at {int(REGIME_ROUTER_REVERSAL_SIZE*100)}% (volatility risk).",
                REGIME_ROUTER_REVERSAL_SIZE)

    if regime == REGIME_LOW_VOL_RANGE:
        return (True,
                f"LOW_VOL_RANGE — mean-reversion context, both directions valid.",
                1.0)

    # Directional regimes
    aligned = (
        (regime == REGIME_BULL_TREND and signal == "BUY") or
        (regime == REGIME_BEAR_TREND and signal == "SELL")
    )

    if aligned:
        conf_str = f"{confidence:.0%}" if confidence else "N/A"
        return (True,
                f"DUAL CONFIRMATION: {regime} + {signal} aligned. "
                f"Confidence: {conf_str}. Full size allowed.",
                1.0)

    # Counter-regime signal
    if REGIME_ROUTER_DUAL_STRICT:
        return (False,
                f"DUAL GATE BLOCKED: {signal} is counter-regime to {regime}. "
                f"Regime sensor and Claude disagree — standing aside.",
                0.0)
    else:
        # Relaxed mode: allow but halve size
        return (True,
                f"Counter-regime {signal} in {regime}. "
                f"Size reduced to 50% (dual confirmation not met).",
                0.50)


# ================================================================
# LAYER 3 — SL/TP REGIME ADJUSTER
# ================================================================

def adjust_sl_tp(regime_result: dict,
                 entry: float, sl: float, tp: float, signal: str) -> dict:
    """
    Adjusts Claude's SL and TP levels based on regime characteristics.

    Returns dict: {sl, tp, sl_moved, tp_moved, rr}

    TREND regimes:
        SL widened by REGIME_ROUTER_SL_MULTIPLIER_TREND (default 1.10 = 10% wider)
        TP unchanged — trend can run further than Claude's estimate
    REVERSAL:
        SL widened by REGIME_ROUTER_SL_MULTIPLIER_REVERSAL (default 1.25 = 25%)
        Reason: wicks can be violent — SL placed at body level gets stopped before wick reversal
    COMPRESSION / LOW_VOL_RANGE:
        TP tightened by REGIME_ROUTER_TP_MULTIPLIER_RANGE (default 0.85 = 15% closer)
        Reason: range-bound market likely to mean-revert before Claude's TP is hit
    """
    regime = regime_result.get("regime", REGIME_LOW_VOL_RANGE)
    signal = signal.upper()

    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp)

    if sl_dist == 0:
        return {"sl": sl, "tp": tp, "sl_moved": 0.0, "tp_moved": 0.0, "rr": 0.0}

    new_sl = sl
    new_tp = tp

    if regime in _DIRECTIONAL_REGIMES:
        # Widen SL to absorb trend wicks
        mult      = REGIME_ROUTER_SL_MULTIPLIER_TREND
        new_sl_dist = sl_dist * mult
        if signal == "BUY":
            new_sl = round(entry - new_sl_dist, 2)
        else:
            new_sl = round(entry + new_sl_dist, 2)

    elif regime == REGIME_REVERSAL:
        # Wider SL — REVERSAL candles often wick hard before reversing
        mult        = REGIME_ROUTER_SL_MULTIPLIER_REVERSAL
        new_sl_dist = sl_dist * mult
        if signal == "BUY":
            new_sl = round(entry - new_sl_dist, 2)
        else:
            new_sl = round(entry + new_sl_dist, 2)

    elif regime in (REGIME_COMPRESSION, REGIME_LOW_VOL_RANGE):
        # Tighter TP — don't let profits evaporate in a ranging market
        mult        = REGIME_ROUTER_TP_MULTIPLIER_RANGE
        new_tp_dist = tp_dist * mult
        if signal == "BUY":
            new_tp = round(entry + new_tp_dist, 2)
        else:
            new_tp = round(entry - new_tp_dist, 2)

    new_sl_dist = abs(entry - new_sl)
    new_tp_dist = abs(entry - new_tp)
    rr          = round(new_tp_dist / new_sl_dist, 2) if new_sl_dist > 0 else 0.0

    return {
        "sl":       new_sl,
        "tp":       new_tp,
        "sl_moved": round(new_sl - sl, 2),
        "tp_moved": round(new_tp - tp, 2),
        "rr":       rr,
    }


# ================================================================
# LAYER 4 — LIVE TRADE ADAPTER
# ================================================================

def adapt_live_trade(regime_result: dict, active_trade: dict) -> dict:
    """
    Called every cycle while a position is open (from run_overnight_trade_guard
    and the main active-window cycle).

    Returns an action dict:
        {"action": "HOLD"}
        {"action": "CLOSE_PARTIAL", "reason": str}
        {"action": "TIGHTEN_TP", "new_tp": float, "reason": str}
        {"action": "WIDEN_SL",  "new_sl": float, "reason": str}

    Rules (applied in priority order):
        1. REVERSAL at high confidence → CLOSE_PARTIAL (50%)
        2. Regime flipped AGAINST open trade direction → suggest CLOSE_PARTIAL
        3. COMPRESSION opened during BUY/SELL → TIGHTEN_TP
        4. Strong trend aligned with trade → WIDEN_SL (let trade breathe)
        5. Everything else → HOLD
    """
    regime     = regime_result.get("regime", REGIME_LOW_VOL_RANGE)
    confidence = regime_result.get("confidence") or 0.0
    direction  = active_trade.get("type", "BUY")
    entry      = active_trade.get("entry",  0.0)
    sl         = active_trade.get("sl",     0.0)
    tp         = active_trade.get("tp",     0.0)

    # ── 1. REVERSAL at high confidence ──────────────────────────
    if regime == REGIME_REVERSAL and confidence >= 0.70:
        return {
            "action": "CLOSE_PARTIAL",
            "reason": f"REVERSAL detected (conf {confidence:.0%}) while trade open. "
                      f"Partial close to reduce exposure to spike risk.",
        }

    # ── 2. Regime flipped counter-direction ──────────────────────
    if regime == REGIME_BULL_TREND and direction == "SELL":
        return {
            "action": "CLOSE_PARTIAL",
            "reason": f"BULL_TREND (conf {confidence:.0%}) confirmed while short open. "
                      f"Regime now counter-directional — partial close to reduce risk.",
        }
    if regime == REGIME_BEAR_TREND and direction == "BUY":
        return {
            "action": "CLOSE_PARTIAL",
            "reason": f"BEAR_TREND (conf {confidence:.0%}) confirmed while long open. "
                      f"Regime now counter-directional — partial close to reduce risk.",
        }

    # ── 3. COMPRESSION — tighten TP to capture before MR ─────────
    if regime == REGIME_COMPRESSION and tp > 0:
        tp_dist    = abs(tp - entry)
        new_tp_dist = tp_dist * 0.70   # bring TP 30% closer
        if direction == "BUY":
            new_tp = round(entry + new_tp_dist, 2)
            if new_tp < tp:
                return {
                    "action": "TIGHTEN_TP",
                    "new_tp": new_tp,
                    "reason": f"COMPRESSION regime — market likely to revert. "
                              f"Tightening TP {tp:.2f} → {new_tp:.2f}.",
                }
        else:
            new_tp = round(entry - new_tp_dist, 2)
            if new_tp > tp:
                return {
                    "action": "TIGHTEN_TP",
                    "new_tp": new_tp,
                    "reason": f"COMPRESSION regime — market likely to revert. "
                              f"Tightening TP {tp:.2f} → {new_tp:.2f}.",
                }

    # ── 4. Strong aligned trend — widen SL to let trade run ──────
    if confidence >= 0.85 and sl > 0:
        aligned = (
            (regime == REGIME_BULL_TREND and direction == "BUY") or
            (regime == REGIME_BEAR_TREND and direction == "SELL")
        )
        if aligned:
            sl_dist     = abs(entry - sl)
            new_sl_dist = sl_dist * 1.10  # 10% wider
            if direction == "BUY":
                new_sl = round(entry - new_sl_dist, 2)
                if new_sl < sl:
                    return {
                        "action": "WIDEN_SL",
                        "new_sl": new_sl,
                        "reason": f"{regime} (conf {confidence:.0%}) strongly aligned. "
                                  f"Widening SL {sl:.2f} → {new_sl:.2f} to absorb trend wicks.",
                    }
            else:
                new_sl = round(entry + new_sl_dist, 2)
                if new_sl > sl:
                    return {
                        "action": "WIDEN_SL",
                        "new_sl": new_sl,
                        "reason": f"{regime} (conf {confidence:.0%}) strongly aligned. "
                                  f"Widening SL {sl:.2f} → {new_sl:.2f} to absorb trend wicks.",
                    }

    # ── 5. No action needed ───────────────────────────────────────
    return {"action": "HOLD"}


# ================================================================
# FORMAT GATE CONTEXT (for Claude prompt)
# ================================================================

def format_gate_context(regime_result: dict,
                         gate_result:   tuple,
                         adjust_result: dict = None) -> str:
    """
    Formats regime router state for injection into the Claude prompt.

    Claude sees:
      - Current regime + confidence
      - Gate 1 result (confidence gate passed/blocked)
      - Gate 2 hint (what dual confirmation will require)
      - SL/TP adjustment note (if Claude returns a directional signal)
    """
    regime     = regime_result.get("regime",     REGIME_LOW_VOL_RANGE)
    confidence = regime_result.get("confidence")
    conf_str   = f"{confidence:.0%}" if confidence else "N/A"

    gate_allowed, gate_reason, *rest = gate_result
    size_mult = rest[0] if rest else 1.0

    lines = [
        "╔═══════════════════════════════════════════════════════════╗",
        "║  REGIME ROUTER — OBJECTIVE MARKET STATE (v12)           ║",
        "╚═══════════════════════════════════════════════════════════╝",
        f"Current Regime   : {regime}",
        f"Confidence       : {conf_str}",
        "",
        f"Gate 1 (Confidence): {'✓ PASSED' if gate_allowed else '✗ BLOCKED'}",
        f"  → {gate_reason}",
        "",
    ]

    # Gate 2 hint — tell Claude what the dual gate will check
    if regime == REGIME_BULL_TREND:
        lines.append("Gate 2 (Dual Confirm): BULL_TREND active.")
        lines.append("  → BUY signals: ALIGNED → full size.")
        lines.append("  → SELL signals: COUNTER-REGIME → blocked (or 50% if relaxed).")
    elif regime == REGIME_BEAR_TREND:
        lines.append("Gate 2 (Dual Confirm): BEAR_TREND active.")
        lines.append("  → SELL signals: ALIGNED → full size.")
        lines.append("  → BUY signals: COUNTER-REGIME → blocked (or 50% if relaxed).")
    elif regime == REGIME_REVERSAL:
        lines.append("Gate 2 (Dual Confirm): REVERSAL — both directions possible.")
        lines.append(f"  → Any signal allowed. Size capped at "
                     f"{int(REGIME_ROUTER_REVERSAL_SIZE*100)}% (spike risk).")
    elif regime == REGIME_LOW_VOL_RANGE:
        lines.append("Gate 2 (Dual Confirm): LOW_VOL_RANGE — mean-reversion context.")
        lines.append("  → Any direction valid. Size unrestricted.")
    else:
        lines.append(f"Gate 2 (Dual Confirm): {regime} — check signal alignment.")

    lines.append("")

    # SL/TP note
    if regime in _DIRECTIONAL_REGIMES:
        lines.append(f"Gate 3 (SL/TP Adjust): Trend regime detected.")
        lines.append(f"  → SL will be widened {int((REGIME_ROUTER_SL_MULTIPLIER_TREND-1)*100)}% "
                     f"to absorb trend continuation wicks.")
    elif regime == REGIME_REVERSAL:
        lines.append(f"Gate 3 (SL/TP Adjust): REVERSAL regime — wide wicks expected.")
        lines.append(f"  → SL will be widened {int((REGIME_ROUTER_SL_MULTIPLIER_REVERSAL-1)*100)}%. "
                     f"Do not place SL at candle body — use wick extreme.")
    elif regime in (REGIME_COMPRESSION, REGIME_LOW_VOL_RANGE):
        lines.append(f"Gate 3 (SL/TP Adjust): Range/Compression regime.")
        lines.append(f"  → TP will be tightened {int((1-REGIME_ROUTER_TP_MULTIPLIER_RANGE)*100)}% "
                     f"to capture profit before mean-reversion.")

    if adjust_result and (adjust_result.get("sl_moved", 0) != 0 or
                          adjust_result.get("tp_moved", 0) != 0):
        lines.append("")
        lines.append(f"Gate 3 Applied: SL moved {adjust_result['sl_moved']:+.2f} pts | "
                     f"TP moved {adjust_result['tp_moved']:+.2f} pts | "
                     f"RR → {adjust_result['rr']:.2f}")

    lines.append("═══════════════════════════════════════════════════════════")
    return "\n".join(lines)
