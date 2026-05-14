"""
strategy_loader.py — Confirmed AI Strategy Loader
===================================================

Reads all fully-structured JSON files from Strategy_AI/confirmed/.
These are strategy proposals that were reviewed by the user and promoted
from Strategy_AI/proposals/ to Strategy_AI/confirmed/.

On every call to match_strategy(), checks whether the current market
conditions (regime + session + keywords) satisfy any confirmed strategy's
activation conditions. If a match is found, returns that strategy's full
ruleset for injection into the Claude prompt and live routing.

PROMOTION WORKFLOW:
    Claude discovers a pattern (10+ occurrences)
    → writes proposal to Strategy_AI/proposals/name_YYYYMMDD.json
    → console pings you until the file is confirmed on disk
    → you review, then copy/move to Strategy_AI/confirmed/name.json
    → strategy_loader picks it up on the next cycle (no restart needed)

CONFIRMED STRATEGY JSON SCHEMA (must match this exactly):
{
    "name":              "ob_retest_bull_continuation",
    "display_name":      "OB Retest Bull Continuation",
    "version":           1,
    "created_date":      "2026-03-09",
    "promoted_date":     "2026-03-10",
    "status":            "ACTIVE",       // ACTIVE | PAUSED | RETIRED

    "activation": {
        "regimes":       ["BULL_TREND"],  // list of regimes where valid
        "sessions":      ["NY", "London"],
        "keywords":      ["ob_retest", "bos_continuation", "displacement_into_ob"],
        "keyword_match_threshold": 2,     // how many keywords must match to activate
        "min_confidence": 0.55            // regime confidence floor
    },

    "evidence": {
        "sample_count":       14,
        "win_rate":           0.71,
        "avg_r":              2.3,
        "source_mix":         {"taken": 6, "misswish": 8},
        "date_range":         ["2026-01-10", "2026-03-08"]
    },

    "entry_criteria": [
        "Confirm BULL_TREND regime with confidence ≥ 55%.",
        "Identify a clear Order Block from an impulsive leg on M15 or H1.",
        "Price must have displaced away from the OB (clear FVG left behind).",
        "Wait for price to return to the OB. Enter on M5 close inside the OB.",
        "A BOS on M5 above the most recent swing high confirms the setup."
    ],

    "sl_placement": "Below the Order Block by 0.5 × ATR. Never inside the OB body.",
    "tp_placement":  "Next institutional liquidity pool (equal highs, prior session high). Minimum 2.5R.",
    "min_rr":        2.5,
    "min_confluence": 2,

    "management_path": [
        "At 1R profit: move SL to breakeven.",
        "At 1.5R: take 30% partial close, trail remainder below each M5 swing low.",
        "At 2.5R (TP): close remaining position unless strong momentum continues.",
        "If regime shifts to REVERSAL mid-trade: tighten TP to current price + 0.5R."
    ],

    "avoid": [
        "OB formed during low-volume consolidation (no displacement = no valid OB).",
        "Entry if price closed below the OB before returning to it (OB mitigated).",
        "Session open first 15 minutes — spread spikes invalidate entry logic."
    ],

    "prompt_injection": "Discovered strategy OB_RETEST_BULL_CONTINUATION is active. This pattern has shown {win_rate:.0%} win rate over {sample_count} setups (avg {avg_r:.1f}R). Apply its specific entry criteria and management path. See ACTIVE STRATEGY block below."
}
"""

import os
import json
import glob
from datetime import datetime

current_dir   = os.path.dirname(os.path.abspath(__file__))
import sys as _sys_sl
_root_sl = os.path.normpath(os.path.join(current_dir, '..'))
if _root_sl not in _sys_sl.path: _sys_sl.path.insert(0, _root_sl)
from paths import CONFIRMED_DIR, create_all_dirs as _cad_sl
_cad_sl()

# Cache loaded strategies for the session (reload only when files change)
_cache: dict = {}          # {filepath: strategy_dict}
_cache_mtime: dict = {}    # {filepath: mtime float}


# ================================================================
# LOAD & CACHE
# ================================================================

def _load_confirmed_strategies() -> list:
    """
    Loads all ACTIVE confirmed strategy JSONs.
    Uses file mtime for cache invalidation — new promotions picked up
    on the next cycle automatically, no restart needed.
    """
    global _cache, _cache_mtime

    if not os.path.isdir(CONFIRMED_DIR):
        return []

    strategies = []
    for path in glob.glob(os.path.join(CONFIRMED_DIR, "*.json")):
        try:
            mtime = os.path.getmtime(path)
            if path not in _cache or _cache_mtime.get(path) != mtime:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                _cache[path]       = data
                _cache_mtime[path] = mtime
                print(f"[StrategyLoader] Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"[StrategyLoader] Failed to load {path}: {e}")

    # Return only ACTIVE ones
    return [s for s in _cache.values()
            if s.get("status", "ACTIVE") == "ACTIVE"]


# ================================================================
# MATCHING
# ================================================================

def _keyword_overlap(current_keywords: list, strategy_keywords: list,
                     threshold: int) -> bool:
    """
    Returns True if at least `threshold` strategy keywords are
    present (substring match) in current_keywords.
    """
    matches = 0
    for sk in strategy_keywords:
        for ck in current_keywords:
            if sk in ck or ck in sk:
                matches += 1
                break
    return matches >= threshold


def match_strategy(regime_result: dict, session: str,
                   current_keywords: list) -> dict | None:
    """
    Checks all confirmed strategies against current market conditions.

    Returns the best-matching strategy dict if one activates, else None.
    If multiple strategies match, returns the one with highest win_rate.

    Args:
        regime_result:     output of regime_detector.predict()
        session:           current session string ("Asian","London","NY")
        current_keywords:  list of keyword strings from context_retriever

    Returns:
        strategy dict (full JSON content) or None
    """
    strategies = _load_confirmed_strategies()
    if not strategies:
        return None

    regime     = regime_result.get("regime", "")
    confidence = regime_result.get("confidence") or 0.0

    candidates = []
    for strat in strategies:
        act = strat.get("activation", {})

        # Regime check
        if regime not in act.get("regimes", []):
            continue

        # Session check — empty list means any session
        allowed_sessions = act.get("sessions", [])
        if allowed_sessions and session not in allowed_sessions:
            continue

        # Confidence floor
        min_conf = act.get("min_confidence", 0.45)
        if confidence < min_conf:
            continue

        # Keyword match
        kw_threshold = act.get("keyword_match_threshold", 2)
        if not _keyword_overlap(current_keywords,
                                act.get("keywords", []),
                                kw_threshold):
            continue

        candidates.append(strat)

    if not candidates:
        return None

    # Pick highest win_rate if multiple match
    best = max(candidates,
               key=lambda s: s.get("evidence", {}).get("win_rate", 0))

    ev = best.get("evidence", {})
    print(f"[StrategyLoader] ✓ ACTIVE STRATEGY MATCHED: "
          f"'{best.get('name')}' | "
          f"win_rate={ev.get('win_rate',0):.0%} | "
          f"avg_R={ev.get('avg_r',0):.1f} | "
          f"n={ev.get('sample_count',0)}")
    return best


# ================================================================
# PROMPT FORMATTING
# ================================================================

def format_strategy_for_prompt(strategy: dict, regime_result: dict,
                                session: str) -> tuple[str, str]:
    """
    Formats a matched confirmed strategy into (rules_block, task_block)
    for injection into the Claude prompt — same shape as strategy_selector
    outputs so main_bot.py needs no branching.

    Returns:
        (rules_text: str, task_text: str)
    """
    ev   = strategy.get("evidence", {})
    name = strategy.get("display_name", strategy.get("name", "AI Strategy"))
    conf = regime_result.get("confidence") or 0.0

    # Build rules block
    entry_lines = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(strategy.get("entry_criteria", []))
    )
    avoid_lines = "\n".join(
        f"  - {a}" for a in strategy.get("avoid", [])
    )
    mgmt_lines = "\n".join(
        f"  {i+1}. {m}" for i, m in enumerate(strategy.get("management_path", []))
    )

    rules = f"""
╔══════════════════════════════════════════════════════════════════╗
║  ACTIVE STRATEGY: {name.upper():<43}║
║  Source: AI-Discovered | Regime: {regime_result.get('regime','?'):<29}║
║  Evidence: {ev.get('sample_count',0)} setups | Win rate: {ev.get('win_rate',0):.0%} | Avg R: {ev.get('avg_r',0):.1f}   ║
╚══════════════════════════════════════════════════════════════════╝

This strategy was discovered by the bot through pattern recognition
across {ev.get('sample_count',0)} repeated setups (taken + MissWish).
Apply it precisely — do not mix it with generic TREND/RANGE/SWEEP rules.

ENTRY CRITERIA (all must be satisfied):
{entry_lines}

SL PLACEMENT: {strategy.get('sl_placement', 'Per standard regime rules.')}
TP PLACEMENT: {strategy.get('tp_placement', 'Per standard regime rules.')}
MINIMUM RR: {strategy.get('min_rr', 1.5)}

MANAGEMENT PATH:
{mgmt_lines}

AVOID:
{avoid_lines}
"""

    task = f"""
STRATEGY-SPECIFIC TASK ({name.upper()}):
  This is an AI-discovered strategy with {ev.get('win_rate',0):.0%} historical win rate.
  1. Verify ALL entry criteria above are currently satisfied.
  2. If satisfied: signal BUY or SELL with levels matching the strategy's SL/TP rules.
  3. If any criteria are not satisfied: signal WAIT — do not force the setup.
  4. Minimum confluence score: {strategy.get('min_confluence', 2)}/3.
  5. Output ONLY valid JSON per execution rules.
"""

    return rules, task


def get_confirmed_strategy_list() -> list:
    """Returns summary of all confirmed strategies for context_retriever."""
    strategies = _load_confirmed_strategies()
    return [
        {
            "name":        s.get("name"),
            "display_name": s.get("display_name"),
            "regimes":     s.get("activation", {}).get("regimes", []),
            "sessions":    s.get("activation", {}).get("sessions", []),
            "keywords":    s.get("activation", {}).get("keywords", []),
            "win_rate":    s.get("evidence", {}).get("win_rate", 0),
            "sample_count": s.get("evidence", {}).get("sample_count", 0),
            "avg_r":       s.get("evidence", {}).get("avg_r", 0),
        }
        for s in strategies
    ]
