"""
keyword_tagger.py
==================
Triggered automatically after daily_post_mortem.py finishes writing
hindsight_feedback for each completed trade.

FIX C5: Removed direct genai.Client() call in _ask_claude(). All AI
         calls now go through call_ai() — provider-agnostic, key-rotating.

Asks Claude one question per trade:
compared to what is already in wisdom.json, is this lesson:
    DUPLICATE  -> already known, nothing to do, stop
    EXTENSION  -> same core lesson, new conditions/evidence, update wisdom entry
    NEW        -> genuinely new lesson, create keyword entry in keywords.json
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # FIX C5: replaces genai.Client

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

current_dir   = os.path.dirname(os.path.abspath(__file__))
base_dir      = os.path.dirname(os.path.dirname(current_dir))
import sys as _sys_kt
if base_dir not in _sys_kt.path: _sys_kt.path.insert(0, base_dir)
from paths import TRADE_MEMORY_PATH, KEYWORDS_PATH, WISDOM_PATH, create_all_dirs as _cad_kt
_cad_kt()
MEMORY_FILE   = TRADE_MEMORY_PATH
KEYWORDS_FILE = KEYWORDS_PATH
WISDOM_FILE   = WISDOM_PATH


# ================================================================
# FILE HELPERS
# ================================================================

def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[KeywordTagger] WARNING: Could not read {path}: {e}")
    return default


def _write_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"[KeywordTagger] WARNING: Could not write {path}: {e}")
        return False


def _get_trade_record(ticket):
    memory = _read_json(MEMORY_FILE, [])
    for trade in memory:
        if str(trade.get('ticket')) == str(ticket):
            return trade
    return None


# ================================================================
# AI CALL — FIX C5: uses call_ai() not genai.Client
# ================================================================

def _ask_ai(prompt: str) -> str | None:
    """
    Single AI call. Returns raw text or None on failure.
    FIX C5: Uses call_ai() — no raw API key or genai.Client needed.
    """
    return call_ai(prompt=prompt)


def _parse_json_response(text):
    if not text:
        return None
    try:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*',     '', text)
        candidates = re.findall(
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL
        )
        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return json.loads(text)
    except Exception:
        return None


# ================================================================
# MAIN TAGGER
# ================================================================

def tag_trade(ticket):
    """
    Main entry point. Called by daily_post_mortem.py after per-trade
    hindsight feedback has been written for a completed trade.
    FIX C5: Uses call_ai() via _ask_ai() — no raw genai.Client.
    """
    print(f"\n[KeywordTagger] Starting tag process for Ticket #{ticket}...")

    trade = _get_trade_record(ticket)
    if not trade:
        print(f"[KeywordTagger] Ticket #{ticket} not found in memory. Aborting.")
        return

    if not trade.get('hindsight_feedback', '').strip():
        print(f"[KeywordTagger] Ticket #{ticket} has no hindsight feedback yet. Aborting.")
        return

    wisdom = _read_json(WISDOM_FILE, {})
    wisdom_text = json.dumps(wisdom, indent=2) if wisdom else "No wisdom entries yet."

    trade_summary = json.dumps({
        "ticket":             trade.get('ticket'),
        "signal":             trade.get('signal'),
        "result":             trade.get('result'),
        "reasoning":          trade.get('reasoning'),
        "analysis_ict":       trade.get('analysis_ict'),
        "analysis_classic":   trade.get('analysis_classic'),
        "analysis_elliott":   trade.get('analysis_elliott'),
        "detailed_review":    trade.get('detailed_review'),
        "hindsight_feedback": trade.get('hindsight_feedback'),
    }, indent=2)

    prompt = f"""
You are a trading knowledge manager for a Gold (XAUUSD) algorithmic trading bot.

A trade has just completed with a full post-mortem. Your job is to classify
whether the lesson from this trade is already captured in the existing wisdom file.

--- COMPLETED TRADE ---
{trade_summary}

--- EXISTING WISDOM FILE ---
{wisdom_text}

Compare the lesson from this trade against every entry in the wisdom file.
Classify it as exactly one of:
    DUPLICATE  - The core lesson is already fully captured. Nothing new.
    EXTENSION  - The same core lesson exists but this trade adds new conditions,
                 new evidence, or a new edge case not yet captured.
                 Provide the updated wisdom entry that merges old + new.
    NEW        - This is a genuinely different lesson not present at all.
                 Provide 3-6 short keywords describing the structural pattern
                 (e.g. "failed_breaker", "premature_FVG_entry", "news_spike_reversal").
                 Keywords describe WHAT happened structurally, not direction or session.
                 Use underscores. Lowercase only.

Respond ONLY with valid JSON in one of these three formats:

DUPLICATE:
{{"classification": "DUPLICATE", "reason": "brief explanation"}}

EXTENSION:
{{"classification": "EXTENSION", "wisdom_key": "existing_key_to_update", "updated_entry": "full updated lesson text merging old and new"}}

NEW:
{{"classification": "NEW", "keywords": ["keyword1", "keyword2", "keyword3"], "lesson": "one clear sentence describing the lesson"}}
"""

    raw    = _ask_ai(prompt)   # FIX C5
    result = _parse_json_response(raw)

    if not result:
        print(f"[KeywordTagger] Could not parse AI response. Raw: {raw}")
        return

    classification = result.get('classification', '').upper()
    print(f"[KeywordTagger] Classification: {classification}")

    if classification == 'DUPLICATE':
        print(f"[KeywordTagger] DUPLICATE — already in wisdom. "
              f"Reason: {result.get('reason', 'N/A')}. Nothing to save.")

    elif classification == 'EXTENSION':
        wisdom_key   = result.get('wisdom_key', '').strip()
        updated_text = result.get('updated_entry', '').strip()

        if not wisdom_key or not updated_text:
            print(f"[KeywordTagger] EXTENSION missing wisdom_key or updated_entry. Aborting.")
            return

        wisdom_data = wisdom if isinstance(wisdom, dict) else {}
        wisdom_data[wisdom_key] = updated_text
        _write_json(WISDOM_FILE, wisdom_data)
        print(f"[KeywordTagger] EXTENSION — wisdom entry '{wisdom_key}' updated.")

    elif classification == 'NEW':
        keywords = result.get('keywords', [])
        lesson   = result.get('lesson', '').strip()

        if not keywords:
            print(f"[KeywordTagger] NEW response missing keywords. Aborting.")
            return

        keywords_data = _read_json(KEYWORDS_FILE, [])
        new_entry = {
            "ticket":    str(ticket),
            "keywords":  keywords,
            "lesson":    lesson,
            "result":    trade.get('result', ''),
            "signal":    trade.get('signal', ''),
            "timestamp": trade.get('timestamp', ''),
        }
        keywords_data.append(new_entry)
        keywords_data = keywords_data[-500:]   # rolling cap
        _write_json(KEYWORDS_FILE, keywords_data)
        print(f"[KeywordTagger] NEW — keywords saved: {keywords}")
        print(f"[KeywordTagger] Lesson: {lesson}")

    else:
        print(f"[KeywordTagger] Unknown classification '{classification}'. Aborting.")
