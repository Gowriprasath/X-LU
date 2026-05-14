"""
strategy_scout.py — Emerging Strategy Pattern Detector
========================================================
FIX M3: Removed direct genai import that would crash Claude-only installs.
         All AI calls now use call_ai() — no client param needed.
FIX C7 integration: run_scout() no longer requires a client argument.
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # FIX M3: replaces genai.Client

import os
import json
import re
import threading
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

current_dir    = os.path.dirname(os.path.abspath(__file__))
base_dir       = os.path.dirname(current_dir)
import sys as _sys_ss; import os as _os_ss
_root_ss = _os_ss.path.normpath(_os_ss.path.join(_os_ss.path.dirname(_os_ss.path.abspath(__file__)), '..'))
if _root_ss not in _sys_ss.path: _sys_ss.path.insert(0, _root_ss)
from paths import (TRADE_MEMORY_PATH, MISSWISH_MEMORY_PATH, SCOUT_LOG_PATH,
                   PROPOSALS_DIR, CONFIRMED_DIR, create_all_dirs as _cad_ss)
_cad_ss()
TRADE_MEM_FILE = TRADE_MEMORY_PATH
MISSWISH_FILE  = MISSWISH_MEMORY_PATH
SCOUT_LOG_FILE = SCOUT_LOG_PATH

from master_controls import SCOUT_PATTERN_THRESHOLD as PATTERN_THRESHOLD

_pending_proposals: list = []
_pending_lock = threading.Lock()


def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[StrategyScout] Read error {path}: {e}")
    return default


def _write_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[StrategyScout] Write error {path}: {e}")
        return False


def _expand_abbreviations(text: str) -> str:
    """
    BUG-18 FIX: Expand common ICT/SMC abbreviations before token matching.
    AI-generated reasoning often uses 'OB', 'FVG', 'CHoCH' etc. interchangeably
    with full names. Without this, a significant fraction of valid pattern keywords
    are silently missed, degrading pattern-frequency analysis.
    """
    _ABBREV_MAP = {
        r'\bob\b':   'order block',
        r'\bfvg\b':  'fair value gap',
        r'\bchoch\b': 'change of character',
        r'\bbos\b':  'break of structure',
        r'\bssl\b':  'sell-side liquidity',
        r'\bbsl\b':  'buy-side liquidity',
        r'\bmss\b':  'market structure shift',
        r'\bifc\b':  'inducement',
        r'\bpoi\b':  'point of interest',
        r'\bpd\b':   'premium discount',
        r'\berl\b':  'external range liquidity',
        r'\birl\b':  'internal range liquidity',
        r'\bbb\b':   'breaker block',
        r'\bsmt\b':  'smart money technique',
        r'\bmmxm\b': 'market maker model',
    }
    import re as _re
    result = text
    for pattern, replacement in _ABBREV_MAP.items():
        result = _re.sub(pattern, replacement, result, flags=_re.IGNORECASE)
    return result


def _extract_keywords_from_trade(trade: dict) -> list:
    raw_text = " ".join([
        str(trade.get('analysis_ict', '') or ''),
        str(trade.get('reasoning', '') or ''),
        str(trade.get('hindsight_feedback', '') or ''),
    ]).lower()

    # BUG-18 FIX: expand abbreviations before matching
    text = _expand_abbreviations(raw_text)

    _PATTERN_TOKENS = [
        "order block", "ob retest", "order_block", "ob_retest",
        "fair value gap", "fvg", "fvg_fill", "fvg fill",
        "liquidity sweep", "equal highs", "equal lows",
        "break of structure", "bos", "bos_", "_bos",
        "choch", "change of character",
        "displacement", "imbalance",
        "mitigation", "breaker block",
        "asian high", "asian low", "asian_high", "asian_low",
        "london open", "ny open", "session open",
        "swing high", "swing low",
        "inducement", "stop hunt",
        "premium", "discount", "equilibrium",
        # BUG-18 FIX: additional expanded forms now matched after abbreviation expansion
        "sell-side liquidity", "buy-side liquidity",
        "market structure shift", "point of interest",
        "breaker block", "external range liquidity", "internal range liquidity",
    ]
    found = []
    for token in _PATTERN_TOKENS:
        if token in text:
            found.append(token.replace(" ", "_"))
    return list(set(found))


def _count_patterns() -> dict:
    patterns = defaultdict(lambda: {
        "keywords": [], "regime": "", "session": "",
        "count": 0, "sources": []
    })

    trades = _read_json(TRADE_MEM_FILE, [])
    for t in trades:
        if t.get("result", "").upper() != "WIN":
            continue
        kws = (t.get("structural_pattern_keywords") or _extract_keywords_from_trade(t))
        if not kws:
            continue
        regime     = str(t.get("regime", "UNKNOWN"))
        session    = str(t.get("session", "UNKNOWN"))
        group_kws  = tuple(sorted(kws[:4]))
        key        = (group_kws, regime, session)
        patterns[key]["keywords"] = list(group_kws)
        patterns[key]["regime"]   = regime
        patterns[key]["session"]  = session
        patterns[key]["count"]   += 1
        patterns[key]["sources"].append({
            "id":        str(t.get("ticket", "?")),
            "type":      "taken",
            "date":      str(t.get("timestamp", ""))[:10],
            "entry":     t.get("entry"),
            "sl":        t.get("sl"),
            "tp":        t.get("tp"),
            "result":    t.get("result"),
            "reasoning": str(t.get("reasoning", ""))[:200],
            "hindsight": str(t.get("hindsight_feedback", ""))[:200],
        })

    mw_entries = _read_json(MISSWISH_FILE, [])
    for e in mw_entries:
        if e.get("actual_outcome", "") != "TP_HIT":
            continue
        kws = list(e.get("structural_pattern_keywords") or [])
        if not kws:
            continue
        regime    = str(e.get("regime_at_signal", "UNKNOWN"))
        session   = str(e.get("session", "UNKNOWN"))
        group_kws = tuple(sorted(kws[:4]))
        key       = (group_kws, regime, session)
        patterns[key]["keywords"] = list(group_kws)
        patterns[key]["regime"]   = regime
        patterns[key]["session"]  = session
        patterns[key]["count"]   += 1
        patterns[key]["sources"].append({
            "id":        str(e.get("id", "?")),
            "type":      "misswish",
            "date":      str(e.get("date", "")),
            "entry":     e.get("ideal_entry"),
            "sl":        e.get("ideal_sl"),
            "tp":        e.get("ideal_tp"),
            "result":    "TP_HIT",
            "outcome_r": e.get("outcome_r"),
            "reasoning": str(e.get("why_it_worked", ""))[:200],
            "management": str(e.get("management_path", ""))[:200],
        })

    return {str(k): v for k, v in patterns.items()}


def _already_proposed(pattern_key: str) -> bool:
    log = _read_json(SCOUT_LOG_FILE, {"proposed": []})
    return pattern_key in log.get("proposed", [])


def _mark_proposed(pattern_key: str, proposal_path: str):
    log = _read_json(SCOUT_LOG_FILE, {"proposed": [], "proposals": []})
    if pattern_key not in log["proposed"]:
        log["proposed"].append(pattern_key)
    log["proposals"].append({
        "pattern_key":   pattern_key,
        "proposal_path": proposal_path,
        "proposed_at":   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "promoted":      False,
    })
    _write_json(SCOUT_LOG_FILE, log)


def _is_confirmed(pattern_name: str) -> bool:
    confirmed = os.path.join(CONFIRMED_DIR, f"{pattern_name}.json")
    return os.path.exists(confirmed)


def _write_strategy_proposal(pattern: dict) -> str | None:
    """
    Calls Claude to write a fully-structured strategy JSON.
    FIX M3: Uses call_ai() — no genai client needed.
    Returns the proposal file path, or None on failure.
    """
    sources_text = json.dumps(pattern["sources"][:15], indent=2)
    n            = pattern["count"]
    keywords     = pattern["keywords"]
    regime       = pattern["regime"]
    session      = pattern["session"]

    taken_wins = sum(1 for s in pattern["sources"] if s.get("type") == "taken")
    mw_hits    = sum(1 for s in pattern["sources"] if s.get("type") == "misswish")
    r_vals     = [s.get("outcome_r") for s in pattern["sources"]
                  if s.get("outcome_r") is not None]
    avg_r      = round(sum(r_vals) / len(r_vals), 2) if r_vals else 0.0
    win_rate   = round((taken_wins + mw_hits) / max(n, 1), 2)

    prompt = f"""
You are writing a fully-structured trading strategy definition for a Gold (XAUUSD)
algorithmic trading bot. This strategy was discovered automatically by detecting
{n} repeated winning occurrences of the same structural pattern.

PATTERN SUMMARY:
  Keywords    : {keywords}
  Regime      : {regime}
  Session     : {session}
  Sample count: {n} ({taken_wins} taken trades + {mw_hits} MissWish ideal setups)
  Win rate    : {win_rate:.0%}
  Average R   : {avg_r:.1f}

SOURCE EVIDENCE (sample — all are WIN or TP_HIT outcomes):
{sources_text}

Write a complete strategy definition as a single valid JSON object.
Synthesise the specific entry criteria, SL placement, TP placement, and
management path from the evidence. Be precise.

The JSON must follow this EXACT schema:
{{
    "name":              "snake_case_strategy_name",
    "display_name":      "Human Readable Strategy Name",
    "version":           1,
    "created_date":      "{datetime.now().strftime('%Y-%m-%d')}",
    "promoted_date":     null,
    "status":            "ACTIVE",
    "activation": {{
        "regimes":                  ["{regime}"],
        "sessions":                 ["{session}"],
        "keywords":                 {json.dumps(keywords)},
        "keyword_match_threshold":  2,
        "min_confidence":           0.55
    }},
    "evidence": {{
        "sample_count":  {n},
        "win_rate":      {win_rate},
        "avg_r":         {avg_r},
        "source_mix":    {{"taken": {taken_wins}, "misswish": {mw_hits}}},
        "date_range":    ["earliest_date_from_sources", "latest_date_from_sources"]
    }},
    "entry_criteria": ["Step 1.", "Step 2.", "Step 3.", "Step 4."],
    "sl_placement":  "Precise SL rule",
    "tp_placement":  "Precise TP rule",
    "min_rr":        2.0,
    "min_confluence": 2,
    "management_path": ["At 1R: action.", "At 1.5R: action.", "At TP: action.", "If regime shifts: action."],
    "avoid": ["Specific invalidation condition.", "Another condition."],
    "prompt_injection": "One sentence summary of strategy and its stats."
}}

RESPOND ONLY WITH THE JSON OBJECT. No explanation, no markdown.
"""
    raw_text = call_ai(prompt=prompt)   # FIX M3
    if raw_text is None:
        print("[StrategyScout] AI call failed. Cannot write proposal.")
        return None

    try:
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*',     '', raw_text)
        strategy = json.loads(raw_text)
        if not isinstance(strategy, dict):
            print("[StrategyScout] AI returned non-dict. Aborting.")
            return None

        name     = strategy.get("name", "unnamed_strategy").replace(" ", "_")
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{name}_{date_str}.json"
        path     = os.path.join(PROPOSALS_DIR, filename)

        os.makedirs(PROPOSALS_DIR, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2)
        return path

    except Exception as e:
        print(f"[StrategyScout] Strategy write failed: {e}")
        return None


def _ping_pending_proposals():
    log     = _read_json(SCOUT_LOG_FILE, {"proposed": [], "proposals": []})
    pending = [
        p for p in log.get("proposals", [])
        if not p.get("promoted")
        and not _is_confirmed(
            os.path.splitext(os.path.basename(p.get("proposal_path", "")))[0]
               .rsplit("_", 1)[0]
        )
    ]
    if not pending:
        return

    print("\n" + "█" * 60)
    print("█  ⚠  STRATEGY PROPOSAL(S) AWAITING YOUR REVIEW")
    print("█" * 60)
    for p in pending:
        path = p.get("proposal_path", "?")
        name = os.path.basename(path) if path != "?" else "?"
        print(f"█  📄  {name}")
        print(f"█      Pattern: {p.get('pattern_key','')[:70]}")
        print(f"█      Proposed: {p.get('proposed_at','?')}")
        print(f"█  To activate:")
        print(f"█    1. Review: {path}")
        print(f"█    2. Approve: copy to Strategy_AI/confirmed/{name.rsplit('_',1)[0]}.json")
        print(f"█    3. Bot picks it up next cycle automatically.")
    print("█" * 60 + "\n")


def run_scout() -> int:
    """
    Main entry point. Called from daily_post_mortem.py as STEP 6.
    FIX M3 + FIX C7: No longer requires a client parameter.

    Returns: number of new proposals written this run.
    """
    print(f"\n[StrategyScout] ══════════════════════════════════════")
    print(f"[StrategyScout] Scanning for emerging strategy patterns...")

    _ping_pending_proposals()

    patterns   = _count_patterns()
    qualifying = {
        k: v for k, v in patterns.items()
        if v["count"] >= PATTERN_THRESHOLD
        and not _already_proposed(k)
        and v["regime"] not in ("UNKNOWN", "")
        and v["session"] not in ("UNKNOWN", "")
    }

    if not qualifying:
        total   = len(patterns)
        closest = max((v["count"] for v in patterns.values()), default=0)
        print(f"[StrategyScout] No new patterns at threshold. "
              f"Total patterns tracked: {total}. "
              f"Closest to threshold: {closest}/{PATTERN_THRESHOLD}.")
        return 0

    print(f"[StrategyScout] {len(qualifying)} pattern(s) hit threshold "
          f"({PATTERN_THRESHOLD}). Writing proposals...")

    new_proposals = 0
    for pattern_key, pattern in qualifying.items():
        print(f"\n[StrategyScout] Writing strategy for:")
        print(f"  Keywords: {pattern['keywords']}")
        print(f"  Regime:   {pattern['regime']}")
        print(f"  Session:  {pattern['session']}")
        print(f"  Count:    {pattern['count']} occurrences")

        proposal_path = _write_strategy_proposal(pattern)   # FIX M3: no client param

        if proposal_path:
            _mark_proposed(pattern_key, proposal_path)
            new_proposals += 1
            print(f"\n[StrategyScout] ✓ Proposal written: {proposal_path}")
        else:
            print(f"[StrategyScout] ✗ Proposal write failed for {pattern_key}")

    if new_proposals > 0:
        _ping_pending_proposals()

    return new_proposals
