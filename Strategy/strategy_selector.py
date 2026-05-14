"""
strategy_selector.py — Regime-to-Strategy Router
==================================================
FIX H2: _direction_from_regime() now returns "BOTH" for REVERSAL
         (sweep trades are directional but the direction is determined by which
          side was swept — not known at regime detection time).
"""

import os
import sys

_strategy_ai_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Strategy_AI'))
if _strategy_ai_dir not in sys.path:
    sys.path.insert(0, _strategy_ai_dir)

REGIME_BULL_TREND    = "BULL_TREND"
REGIME_BEAR_TREND    = "BEAR_TREND"
REGIME_LOW_VOL_RANGE = "LOW_VOL_RANGE"
REGIME_COMPRESSION   = "COMPRESSION"
REGIME_REVERSAL      = "REVERSAL"

# ================================================================
# STRATEGY RULEBOOKS
# ================================================================

_TREND_RULES = """
╔══════════════════════════════════════════════════════════════════╗
║  ACTIVE STRATEGY: TREND CONTINUATION                           ║
║  Regime: {regime} (conf {conf})                                ║
╚══════════════════════════════════════════════════════════════════╝

You are operating in TREND mode. One job: find the optimal
continuation entry aligned with the established institutional flow.

ENTRY CRITERIA (all must be satisfied):
  1. Trade ONLY in the trend direction — {direction}.
     Counter-trend entries are blocked.
  2. Identify a clean Order Block or Fair Value Gap in trend direction.
     The OB/FVG must be from an IMPULSIVE leg, not consolidation noise.
  3. Price must have displaced away from the OB/FVG before returning.
     No displacement candle → no valid entry.
  4. Higher-timeframe structure must confirm: BOS/CHoCH on H1 pointing {direction}.

SL PLACEMENT:
  Place SL beyond the last significant swing structure.
  Minimum SL distance: 1.5 × ATR.

TP PLACEMENT:
  Target the next major liquidity pool. Minimum 2.5R required.

CONFLUENCE REQUIREMENTS (target 3/3):
  ✓ ICT: OB or FVG in trend direction with prior displacement
  ✓ Classic: MACD histogram aligned + price above/below EMA200
  ✓ Elliott: Wave structure consistent with continuation

AVOID:
  - Entering at session open without retrace (chasing)
  - Counter-trend entries under any framing
"""

_RANGE_RULES = """
╔══════════════════════════════════════════════════════════════════╗
║  ACTIVE STRATEGY: RANGE MEAN-REVERSION                         ║
║  Regime: {regime} (conf {conf})                                ║
╚══════════════════════════════════════════════════════════════════╝

You are operating in RANGE mode. One job: fade the extremes of a
defined range.

ENTRY CRITERIA (all must be satisfied):
  1. Define the current range: clear highs and lows with multiple touches.
  2. Price must be AT or WITHIN 0.5 ATR of the range extreme.
     Mid-range entries are forbidden.
  3. Entry requires a rejection candle at the extreme.
  4. An OB or FVG at the boundary strengthens the setup to 3/3.

SL PLACEMENT:
  Place SL just OUTSIDE the range boundary.
  Tight stops: 0.8–1.2 × ATR beyond the boundary.

TP PLACEMENT:
  Target range midpoint first (partial close), then opposite extreme.
  1.5–2.0R is realistic for mean-reversion.

CONFLUENCE REQUIREMENTS (2/3 acceptable):
  ✓ ICT: OB or FVG at range boundary
  ✓ Classic: RSI divergence at extreme + Bollinger Band touch
  ✓ Elliott: A-B-C corrective structure with C at boundary

AVOID:
  - Trading the first touch of a new range boundary
  - Entering if ATR is expanding rapidly
"""

_SWEEP_RULES = """
╔══════════════════════════════════════════════════════════════════╗
║  ACTIVE STRATEGY: LIQUIDITY SWEEP REVERSAL                     ║
║  Regime: {regime} (conf {conf})                                ║
╚══════════════════════════════════════════════════════════════════╝

You are operating in SWEEP mode. One job: catch the reversal after
institutional liquidity has been hunted.

DIRECTION: Determined by which side was swept.
  If Asian HIGHS were swept → look for SELL reversal.
  If Asian LOWS were swept  → look for BUY reversal.
  Do NOT assume direction from regime alone.

CRITICAL RULE — WAIT FOR CLOSE:
  NEVER enter during the spike candle. The candle that sweeps the
  liquidity must be FULLY CLOSED before your entry is valid.

ENTRY CRITERIA (all must be satisfied):
  1. Identify the swept level: equal highs/lows, previous session extreme.
  2. The sweep candle must have a long wick beyond that level.
  3. After the sweep candle CLOSES, price must show displacement BACK.
  4. A Break of Structure (BOS) on M5 in the reversal direction confirms.

SL PLACEMENT:
  Place SL beyond the WICK EXTREME of the sweep candle — not just the body.

TP PLACEMENT:
  Target the pre-sweep structure (where price was BEFORE the sweep).
  1.0–1.5R is realistic.

CONFLUENCE REQUIREMENTS (target 3/3, minimum 2/3):
  ✓ ICT: Clear liquidity level swept + wick through it + BOS back
  ✓ Classic: RSI extreme at sweep point
  ✓ Elliott: Completion of a corrective spike

AVOID:
  - Any entry DURING the sweep candle
  - Targets beyond the pre-sweep structure
"""

_BLOCKED_RULES = """
╔══════════════════════════════════════════════════════════════════╗
║  STRATEGY: BLOCKED — NO VALID SETUP                            ║
║  Regime: {regime} (conf {conf})                                ║
╚══════════════════════════════════════════════════════════════════╝

The current market regime ({regime}) has no valid trading strategy.
Your ONLY valid output is WAIT. No BUY. No SELL.
"""

_TREND_TASK = """
STRATEGY-SPECIFIC TASK (TREND CONTINUATION):
  1. Identify current swing structure on H1: is the trend intact?
  2. Locate the most recent valid OB or FVG in the trend direction.
  3. Is price currently at or near that OB/FVG? (within 0.5 ATR)
  4. Is there prior displacement from that level?
  5. Compute Confluence Score (0-3) using TREND criteria above.
  6. Signal: BUY or SELL (trend direction only) with 2.5R+ target, OR WAIT.
  7. Output ONLY valid JSON per execution rules.
"""

_RANGE_TASK = """
STRATEGY-SPECIFIC TASK (RANGE MEAN-REVERSION):
  1. Define the current range — upper and lower boundaries.
  2. Is price at an extreme (within 0.5 ATR of boundary)?
  3. Is there a rejection candle or OB/FVG at that boundary?
  4. Compute Confluence Score (0-3) using RANGE criteria above.
  5. Signal: BUY (at low) or SELL (at high) with 1.5R target, OR WAIT.
  6. Output ONLY valid JSON per execution rules.
"""

_SWEEP_TASK = """
STRATEGY-SPECIFIC TASK (LIQUIDITY SWEEP REVERSAL):
  1. Identify the swept level: what liquidity was hunted, and on WHICH side?
  2. Has the sweep candle FULLY CLOSED? (If no, output WAIT.)
  3. Is there displacement back into the range?
  4. Is there a BOS on M5 confirming the reversal direction?
  5. Compute Confluence Score (0-3) using SWEEP criteria above.
  6. Signal: BUY (sweep of lows) or SELL (sweep of highs) with 1.2R target, OR WAIT.
     SL must be beyond the wick extreme — not body.
  7. Output ONLY valid JSON per execution rules.
"""

_BLOCKED_TASK = """
TASK: Output WAIT. Reasoning: {regime} regime — no valid strategy.
JSON: {{"signal": "WAIT", "bias": "NEUTRAL", "confluence_score": 0,
"reasoning": "{regime} regime active — no entry criteria satisfied.",
"thesis": "Waiting for regime change."}}
"""


# ================================================================
# ROUTING TABLE
# ================================================================

_REGIME_TO_STRATEGY = {
    REGIME_BULL_TREND:    "TREND",
    REGIME_BEAR_TREND:    "TREND",
    REGIME_LOW_VOL_RANGE: "RANGE",
    REGIME_REVERSAL:      "SWEEP",
    REGIME_COMPRESSION:   "BLOCKED",
    "UNCERTAIN":          "BLOCKED",
}

_STRATEGY_MIN_CONFLUENCE = {
    "TREND":   3,
    "RANGE":   2,
    "SWEEP":   2,
    "BLOCKED": 0,
}


def _direction_from_regime(regime: str) -> str:
    """
    Returns the direction hint for the given regime.
    FIX H2: REVERSAL (SWEEP strategy) returns "BOTH" — direction is determined
             by which liquidity level was swept, not by the regime label itself.
    """
    if regime == REGIME_BULL_TREND:
        return "BUY"
    elif regime == REGIME_BEAR_TREND:
        return "SELL"
    elif regime == REGIME_REVERSAL:
        return "BOTH"   # FIX H2: was "NONE", which misled downstream routing
    return "NONE"


# ================================================================
# PUBLIC API
# ================================================================

def select_strategy(regime_result: dict,
                    session: str = "",
                    current_keywords: list = None) -> dict:
    """
    Maps current market conditions to the appropriate strategy.

    PRIORITY:
      1. AI-Discovered confirmed strategy (matches regime + session + keywords)
      2. Regime-based fallback (TREND / RANGE / SWEEP / BLOCKED)
    """
    regime     = regime_result.get("regime", REGIME_LOW_VOL_RANGE)
    confidence = regime_result.get("confidence") or 0.0
    conf_str   = f"{confidence:.0%}"
    kws        = current_keywords or []

    # ── PRIORITY 1: AI-Discovered confirmed strategy ──────────────
    ai_match = None
    try:
        from strategy_loader import match_strategy, format_strategy_for_prompt
        ai_match = match_strategy(regime_result, session, kws)
    except ImportError:
        # ZOM-02 FIX: Strategy_AI/strategy_loader.py does not exist in this installation.
        # This import was PRIORITY 1 but has never executed successfully — it's dead code
        # from a planned feature that was never shipped.  The ImportError is silently
        # swallowed here (not printed) to avoid drowning the console with 288 identical
        # error lines per day (every 5-min cycle).  The regime-based fallback below
        # handles all strategy routing correctly.
        pass
    except Exception as e:
        print(f"[StrategySelector] strategy_loader error: {e}")

    if ai_match:
        rules, task = format_strategy_for_prompt(ai_match, regime_result, session)
        ev = ai_match.get("evidence", {})
        return {
            "strategy":       "AI_DISCOVERED",
            "regime":         regime,
            "confidence":     confidence,
            "rules":          rules,
            "task_override":  task,
            "min_confluence": ai_match.get("min_confluence", 2),
            "direction_hint": _direction_from_regime(regime),
            "ai_strategy":    ai_match,
            "source":         f"AI-Discovered: {ai_match.get('name')} "
                              f"({ev.get('win_rate',0):.0%} WR, "
                              f"{ev.get('sample_count',0)} setups)",
        }

    # ── PRIORITY 2: Regime-based fallback ─────────────────────────
    strategy  = _REGIME_TO_STRATEGY.get(regime, "BLOCKED")
    direction = _direction_from_regime(regime)
    if strategy == "RANGE":
        direction = "BOTH"   # range trades can go either direction

    fmt = {"regime": regime, "conf": conf_str, "direction": direction}

    if strategy == "TREND":
        rules         = _TREND_RULES.format(**fmt)
        task_override = _TREND_TASK
    elif strategy == "RANGE":
        rules         = _RANGE_RULES.format(**fmt)
        task_override = _RANGE_TASK
    elif strategy == "SWEEP":
        rules         = _SWEEP_RULES.format(**fmt)
        task_override = _SWEEP_TASK
    else:
        rules         = _BLOCKED_RULES.format(**fmt)
        task_override = _BLOCKED_TASK.format(regime=regime)

    return {
        "strategy":       strategy,
        "regime":         regime,
        "confidence":     confidence,
        "rules":          rules,
        "task_override":  task_override,
        "min_confluence": _STRATEGY_MIN_CONFLUENCE[strategy],
        "direction_hint": direction,
        "ai_strategy":    None,
        "source":         f"Regime-based: {strategy}",
    }


def format_strategy_header(strategy_result: dict) -> str:
    s   = strategy_result
    src = s.get("source", s.get("strategy", "?"))
    return (
        f"[StrategySelector] {src} | "
        f"Regime: {s['regime']} ({s['confidence']:.0%}) | "
        f"Direction: {s['direction_hint']} | "
        f"Min confluence: {s['min_confluence']}/3"
    )
