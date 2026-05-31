"""
context_retriever.py
=====================
Called by main_bot.py before every trade decision.
Assembles all five memory layers into one formatted block
ready to inject into the trading prompt.

FIX C6: Removed direct genai.Client() call in _get_current_context_keywords().
         All AI calls now go through call_ai() — provider-agnostic, key-rotating.
FIX M2: Cold-cache now correctly shares the result within the same call chain
         so no double API call occurs.

Layer 1 — Wisdom
Layer 2 — Last 3 recent complete trade records
Layer 3 — Keyword-matched historical trades (keywords.json)
Layer 4 — MissWish ideal setup matches (misswish_keywords.json)
Layer 5 — Confirmed AI-discovered strategies (Strategy_AI/confirmed/*.json)
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # FIX C6: replaces genai.Client

import json
import os
import re
import hashlib
from dotenv import load_dotenv

load_dotenv()

current_dir   = os.path.dirname(os.path.abspath(__file__))
base_dir      = os.path.dirname(os.path.dirname(current_dir))
import sys as _sys_cr
if base_dir not in _sys_cr.path: _sys_cr.path.insert(0, base_dir)
from paths import (TRADE_MEMORY_PATH, KEYWORDS_PATH, WISDOM_PATH,
                   HUMAN_RULES_PATH, AI_LESSONS_PATH, MISSWISH_KW_PATH,
                   CONFIRMED_DIR, create_all_dirs as _cad_cr)
_cad_cr()
MEMORY_FILE      = TRADE_MEMORY_PATH
KEYWORDS_FILE    = KEYWORDS_PATH
WISDOM_FILE      = WISDOM_PATH
HUMAN_RULES_FILE = HUMAN_RULES_PATH
AI_LESSONS_FILE  = AI_LESSONS_PATH
MISSWISH_KW_FILE = MISSWISH_KW_PATH

RECENT_TRADE_COUNT = 3
MAX_MATCHED_TRADES = 4

# FIX Bug 4 / FIX B3:
# Cache keyword extraction by simulated/live time bucket.
# Keywords are re-extracted at most once per H1 period (60 minutes).
# The old hash-based approach always missed because the rolling OHLCV
# window shifts every M5 candle, making every context string unique.
_KEYWORD_CACHE_MINUTES = 60          # re-extract at most once per hour
_keyword_cache_entry:  dict  = {}    # {bucket_key, keywords, expires_bucket}


# ================================================================
# FILE HELPERS
# ================================================================

def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[ContextRetriever] WARNING: Could not read {path}: {e}")
    return default


# ================================================================
# LAYER 1 — WISDOM FILE
# ================================================================

def _get_wisdom_block():
    block = ""

    human_rules = _read_json(HUMAN_RULES_FILE, {})
    rules = {k: v for k, v in human_rules.items() if not k.startswith("_")}
    if rules:
        block += "--- VERIFIED TRADING RULES (human-confirmed, non-negotiable) ---\n"
        for key, rule in rules.items():
            block += f"  ✓ [{key}]: {rule}\n"
        block += "\n"
    else:
        block += "--- VERIFIED TRADING RULES ---\nNo human rules defined yet.\n\n"

    ai_lessons = _read_json(AI_LESSONS_FILE, {})
    if not ai_lessons:
        legacy = _read_json(WISDOM_FILE, {})
        if legacy:
            ai_lessons = {k: {"text": v, "confidence": 1} for k, v in legacy.items()
                         if not k.startswith("_")}

    if not ai_lessons:
        block += ("--- AI-OBSERVED LESSONS ---\n"
                  "No lessons yet. Will populate after first trading week.\n")
        return block

    confirmed = {}
    tentative = {}

    for k, v in ai_lessons.items():
        if k.startswith("_"):
            continue
        if not isinstance(v, dict):
            tentative[k] = {
                "text": str(v),
                "total_trials": 1,
                "success_rate": 1.0
            }
        else:
            corrobs = v.get("corroborations", v.get("confidence", 1))
            viols   = v.get("violations", 0)
            trials  = v.get("total_trials", corrobs + viols)
            rate    = v.get("success_rate", 1.0 if trials == 0 else corrobs / trials)
            
            entry = {
                "text":         v.get("text", ""),
                "total_trials": trials,
                "success_rate": rate
            }
            
            if trials >= 3 and rate < 0.50:
                continue  # Filter out low-success lessons
                
            if trials >= 3 and rate >= 0.65:
                confirmed[k] = entry
            else:
                tentative[k] = entry

    if confirmed:
        block += "--- CONFIRMED LESSONS (corroborated by 3+ trades with >=65% success) ---\n"
        for key, entry in confirmed.items():
            text   = entry["text"]
            trials = entry["total_trials"]
            rate   = entry["success_rate"]
            block += f"  ✓ [{key}] (trials={trials}, success={rate:.0%}): {text}\n"
        block += "\n"

    if tentative:
        block += ("--- TENTATIVE OBSERVATIONS (not yet corroborated or lower success rate) ---\n")
        for key, entry in tentative.items():
            text   = entry["text"]
            trials = entry["total_trials"]
            rate   = entry["success_rate"]
            block += f"  ? [{key}] (trials={trials}, success={rate:.0%}): {text}\n"
        block += "\n"

    return block


# ================================================================
# LAYER 2 — RECENT FULL TRADE RECORDS
# ================================================================

def _get_recent_trades_block(count=RECENT_TRADE_COUNT):
    memory   = _read_json(MEMORY_FILE, [])
    reviewed = [
        t for t in memory
        if t.get('status') in ('CLOSED', 'REVIEWED')
        and t.get('result', '').strip() != ''
    ]

    if not reviewed:
        return "--- RECENT TRADE CONTEXT ---\nNo completed trades yet.\n"

    recent = reviewed[-count:]
    block  = f"--- RECENT TRADE CONTEXT (Last {len(recent)} completed trades) ---\n"
    for trade in recent:
        block += (
            f"Ticket : {trade.get('ticket')} | "
            f"Signal : {trade.get('signal')} | "
            f"Result : {trade.get('result')} | "
            f"Date   : {str(trade.get('timestamp', 'N/A'))[:10]}\n"
        )
        block += f"Reasoning  : {trade.get('reasoning', 'N/A')}\n"
        if trade.get('detailed_review', '').strip():
            block += f"Post-Mortem: {trade.get('detailed_review')}\n"
        if trade.get('hindsight_feedback', '').strip():
            block += f"Hindsight  : {trade.get('hindsight_feedback')}\n"
        block += "-" * 40 + "\n"
    return block


# ================================================================
# LAYER 3 — KEYWORD MATCHED HISTORICAL TRADES
# FIX C6: Uses call_ai() instead of genai.Client
# ================================================================

def _get_current_context_keywords(market_context: str,
                                   current_time=None) -> list:
    """
    Sends current market context to Claude and gets structural keywords.

    FIX C6:  Removed genai.Client. Uses call_ai() with key rotation.
    FIX Bug4: Results cached to avoid redundant API calls each candle.
    FIX B3:  Cache keyed by TIME BUCKET, not content hash.

        Root cause of the always-miss bug: the market_context string
        contains a rolling window of 20 M5 + 48 H1 + 20 H4 + 10 D1
        candles.  Every new M5 bar shifts that window — one candle
        enters, one drops — so the content (and any hash of it) changes
        on every single bar.  Content-hashing can never produce a HIT.

        Fix: key the cache on floor(current_time, 60min).  Keywords are
        re-extracted at most once per hour regardless of how many M5
        candles arrive in that hour (12 candles = 12x fewer API calls).
        Falls back to a single global slot when current_time is None
        (live bot without explicit clock injection).
    """
    global _keyword_cache_entry

    # Build bucket key: "YYYY-MM-DD HH:00" — stable for a full hour
    if current_time is not None:
        try:
            bucket_key = current_time.strftime("%Y-%m-%d %H:00")
        except Exception:
            bucket_key = "live"
    else:
        bucket_key = "live"

    cached = _keyword_cache_entry
    if cached.get("bucket") == bucket_key and cached.get("keywords"):
        remaining = cached.get("remaining", 0)
        print(f"[ContextRetriever] Keyword cache HIT "
              f"(bucket={bucket_key}, {remaining} candles left this hour): "
              f"{cached['keywords']}")
        cached["remaining"] = max(0, remaining - 1)
        return cached["keywords"]

    prompt = f"""
You are analysing the current market setup for a Gold (XAUUSD) trading bot.

--- CURRENT MARKET CONTEXT ---
{market_context}

Extract 3-6 short keywords describing the STRUCTURAL PATTERN of the current setup.
Describe what is happening mechanically — the setup type, the condition, the key structure.
Do NOT include direction (buy/sell) or session names.
Use the same style as these examples:
    "fvg_fill", "breaker_block_entry", "liquidity_sweep", "asian_high_sweep",
    "displacement_candle", "bos_retest", "wick_rejection", "order_block_tap",
    "equal_highs_sweep", "mitigation_block", "news_spike"

Respond ONLY with a JSON array of strings. No explanation. No markdown.
Example: ["fvg_fill", "asian_high_sweep", "displacement_candle"]
"""
    try:
        raw = call_ai(prompt=prompt)  # FIX C6
        if raw is None:
            print("[ContextRetriever] Keyword extraction AI call failed.")
            return []

        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*',     '', raw)
        keywords = json.loads(raw)
        if isinstance(keywords, list):
            kws = [str(k).lower().strip() for k in keywords]
            _keyword_cache_entry = {
                "bucket":    bucket_key,
                "keywords":  kws,
                "remaining": 11,   # 12 M5 candles per hour; first used this call
            }
            print(f"[ContextRetriever] Keyword cache MISS — Claude called. "
                  f"Cached for bucket {bucket_key} (~12 candles).")
            return kws
    except Exception as e:
        print(f"[ContextRetriever] Keyword extraction failed: {e}")
    return []


def _substring_match(current_keywords, stored_entry):
    stored = stored_entry.get('keywords', [])
    count  = 0
    for ck in current_keywords:
        for sk in stored:
            if ck in sk or sk in ck:
                count += 1
                break
    return count


def _get_matched_trades_block(market_context, precomputed_keywords=None):
    """BUG-16 FIX: accepts pre-computed keywords to avoid double AI call."""
    keywords_data = _read_json(KEYWORDS_FILE, [])
    if not keywords_data:
        return "--- HISTORICAL PATTERN MATCHES ---\nNo keyword history yet.\n"

    # BUG-16 FIX: use pre-computed keywords if provided, else compute once here
    current_keywords = precomputed_keywords if precomputed_keywords is not None \
                       else _get_current_context_keywords(market_context)
    if not current_keywords:
        return "--- HISTORICAL PATTERN MATCHES ---\nCould not extract current context keywords.\n"

    print(f"[ContextRetriever] Current setup keywords: {current_keywords}")

    scored = []
    for entry in keywords_data:
        score = _substring_match(current_keywords, entry)
        if score > 0:
            scored.append((score, entry))

    if not scored:
        return (f"--- HISTORICAL PATTERN MATCHES ---\n"
                f"No historical trades match current keywords: {current_keywords}\n")

    scored.sort(key=lambda x: x[0], reverse=True)
    top_entries = scored[:MAX_MATCHED_TRADES]

    memory          = _read_json(MEMORY_FILE, [])
    memory_by_ticket = {str(t.get('ticket')): t for t in memory}

    block = f"--- HISTORICAL PATTERN MATCHES (Keywords: {current_keywords}) ---\n"
    for score, entry in top_entries:
        ticket = str(entry.get('ticket'))
        trade  = memory_by_ticket.get(ticket)
        if not trade:
            continue
        block += (
            f"Ticket : {ticket} | Result : {trade.get('result','?')} | "
            f"Match : {score} keyword(s) | Tags: {entry.get('keywords')}\n"
        )
        block += f"Lesson     : {entry.get('lesson', 'N/A')}\n"
        block += f"Reasoning  : {trade.get('reasoning', 'N/A')}\n"
        if trade.get('analysis_ict', '').strip():
            block += f"ICT        : {trade.get('analysis_ict')}\n"
        if trade.get('analysis_classic', '').strip():
            block += f"Classic PA : {trade.get('analysis_classic')}\n"
        if trade.get('detailed_review', '').strip():
            block += f"Post-Mortem: {trade.get('detailed_review')}\n"
        if trade.get('hindsight_feedback', '').strip():
            block += f"Hindsight  : {trade.get('hindsight_feedback')}\n"
        block += "-" * 40 + "\n"
    return block


# ================================================================
# LAYER 4 — MISSWISH IDEAL SETUP MATCHES
# ================================================================

def _get_misswish_block(market_context: str, precomputed_keywords=None) -> str:
    """BUG-16 FIX: accepts pre-computed keywords to avoid double AI call."""
    mw_keywords = _read_json(MISSWISH_KW_FILE, [])
    if not mw_keywords:
        return ""

    # BUG-16 FIX: use pre-computed keywords if provided — no extra API call
    current_keywords = precomputed_keywords if precomputed_keywords is not None \
                       else _get_current_context_keywords(market_context)
    if not current_keywords:
        return ""

    scored = []
    for entry in mw_keywords:
        score = _substring_match(current_keywords, entry)
        if score > 0:
            scored.append((score, entry))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored[:3]

    block  = "=" * 60 + "\n"
    block += "IDEAL SETUPS FROM SIMILAR STRUCTURES\n"
    block += "(Source: post-mortem MissWish analysis — aspirational patterns,\n"
    block += " NOT execution history. Use as a structural reference, not a signal.)\n"
    block += "=" * 60 + "\n\n"

    for score, entry in top_matches:
        outcome_str = ""
        if entry.get('actual_outcome') and entry.get('actual_outcome') != 'UNKNOWN':
            r = entry.get('outcome_r', 0) or 0
            outcome_str = f" | Verified outcome: {entry['actual_outcome']} ({r:+.1f}R)"

        bot_str = ""
        if entry.get('bot_action') == 'BLOCKED':
            bot_str = f" [Bot blocked by {entry.get('bot_gate', '?')}]"
        elif entry.get('bot_action') == 'MISSED':
            bot_str = " [Bot was in dead zone — setup not seen]"
        elif entry.get('bot_action') == 'TAKEN':
            bot_str = " [Bot took this trade]"

        block += (
            f"Setup    : {entry.get('signal','?')} | "
            f"Session: {entry.get('session','?')} | "
            f"Regime: {entry.get('regime_at_signal','?')} | "
            f"RR: {entry.get('rr', 0):.1f} | "
            f"Match: {score} keyword(s){bot_str}{outcome_str}\n"
        )
        block += f"Entry    : {entry.get('ideal_entry',0):.2f} | "
        block += f"SL: {entry.get('ideal_sl',0):.2f} | "
        block += f"TP: {entry.get('ideal_tp',0):.2f}\n"
        block += f"Pattern  : {entry.get('keywords', [])}\n"
        block += f"Why      : {entry.get('why_it_worked', 'N/A')}\n"
        block += f"Manage   : {entry.get('management_path', 'N/A')}\n"
        block += f"Lesson   : {entry.get('lesson', 'N/A')}\n"
        block += "-" * 50 + "\n"

    return block


# ================================================================
# LAYER 5 — CONFIRMED AI-DISCOVERED STRATEGIES
# ================================================================

def _get_confirmed_strategies_block() -> str:
    try:
        import sys as _sys
        # _strat_dir now resolved via CONFIRMED_DIR from paths.py
        _strat_dir = str(CONFIRMED_DIR)
        if _strat_dir not in _sys.path:
            _sys.path.insert(0, _strat_dir)
        from strategy_loader import get_confirmed_strategy_list
        strategies = get_confirmed_strategy_list()
    except Exception:
        return ""

    if not strategies:
        return ""

    block  = "=" * 60 + "\n"
    block += "CONFIRMED AI-DISCOVERED STRATEGIES\n"
    block += "(These strategies fire automatically when conditions match.\n"
    block += " strategy_selector routes through them before regime rules.)\n"
    block += "=" * 60 + "\n\n"

    for s in strategies:
        block += (
            f"  [{s['name']}]\n"
            f"  Regimes: {s['regimes']} | Sessions: {s['sessions']}\n"
            f"  Keywords: {s['keywords']}\n"
            f"  Evidence: {s['sample_count']} setups | "
            f"WR: {s['win_rate']:.0%} | Avg R: {s['avg_r']:.1f}\n\n"
        )
    return block


# ================================================================
# PUBLIC API
# ================================================================

def get_full_memory_context(market_context: str,
                            current_time=None) -> str:
    """
    Called from main_bot.py / backtest_engine.py before every trade decision.
    Returns all memory layers as one formatted string.

    BUG-16 FIX: keywords extracted once and shared across layers 3 & 4.
    FIX B3:     current_time forwarded to _get_current_context_keywords()
                so the time-bucket cache can function correctly.
                Without current_time the cache falls back to a single
                global slot (acceptable for live bot; ideal for backtest).
    """
    if os.environ.get("BACKTEST_MODE") != "1":
        print("[ContextRetriever] Assembling memory context...")
    layer1 = _get_wisdom_block()
    layer2 = _get_recent_trades_block()

    # BUG-16 / FIX B3: compute keywords once, pass bucket clock + to both layers
    shared_keywords = _get_current_context_keywords(
        market_context, current_time=current_time)

    layer3 = _get_matched_trades_block(market_context, precomputed_keywords=shared_keywords)
    layer4 = _get_misswish_block(market_context,       precomputed_keywords=shared_keywords)

    combined = (
        "=" * 60 + "\n"
        "MEMORY SYSTEM — FIVE LAYER CONTEXT\n"
        "=" * 60 + "\n\n"
        + layer1 + "\n"
        + layer2 + "\n"
        + layer3
    )

    if layer4:
        combined += "\n" + layer4

    layer5 = _get_confirmed_strategies_block()
    if layer5:
        combined += "\n" + layer5

    return combined