"""
misswish_tagger.py — MissWish Pattern Classifier
==================================================
FIX: Removed dead `from google import genai` import (BUG M3).
FIX: Removed unused AI_LESSONS_FILE variable (BUG A9).
All AI calls use call_ai() — no raw API key or genai client needed.
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # FIX: replaces genai.Client; removed AI_LESSONS_FILE (dead)

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

current_dir      = os.path.dirname(os.path.abspath(__file__))
base_dir         = os.path.dirname(os.path.dirname(current_dir))
import sys as _sys_mwt
if base_dir not in _sys_mwt.path: _sys_mwt.path.insert(0, base_dir)
from paths import MISSWISH_MEMORY_PATH, MISSWISH_KW_PATH, create_all_dirs as _cad_mwt
_cad_mwt()
MISSWISH_FILE    = MISSWISH_MEMORY_PATH
MISSWISH_KW_FILE = MISSWISH_KW_PATH
# NOTE: AI_LESSONS_FILE removed — it was imported but never used (BUG A9 fix)


def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[MissWishTagger] WARNING: Could not read {path}: {e}")
    return default


def _write_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"[MissWishTagger] WARNING: Could not write {path}: {e}")
        return False


def _get_entry(entry_id: str) -> dict | None:
    entries = _read_json(MISSWISH_FILE, [])
    for e in entries:
        if e.get('id') == entry_id:
            return e
    return None


def _mark_tagged(entry_id: str):
    entries = _read_json(MISSWISH_FILE, [])
    for e in entries:
        if e.get('id') == entry_id:
            e['tagged'] = True
            break
    _write_json(MISSWISH_FILE, entries)


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*',     '', text)
        candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return json.loads(text)
    except Exception:
        return None


def tag_misswish_entry(entry_id: str):
    """
    Main entry point. Called from daily_post_mortem.py for each new
    MissWish entry written by misswish_analyser.py.
    Uses call_ai() — no genai client needed.
    """
    print(f"\n[MissWishTagger] Tagging entry {entry_id}...")

    entry = _get_entry(entry_id)
    if not entry:
        print(f"[MissWishTagger] Entry {entry_id} not found. Aborting.")
        return

    existing_kw   = _read_json(MISSWISH_KW_FILE, [])
    existing_text = json.dumps(
        [{"id": e.get("entry_id"), "keywords": e.get("keywords"),
          "lesson": e.get("lesson")}
         for e in existing_kw[-100:]],
        indent=2
    ) if existing_kw else "No existing MissWish patterns yet."

    entry_summary = json.dumps({
        "session":                    entry.get('session'),
        "signal":                     entry.get('signal'),
        "regime_at_signal":           entry.get('regime_at_signal'),
        "ideal_entry":                entry.get('ideal_entry'),
        "ideal_sl":                   entry.get('ideal_sl'),
        "ideal_tp":                   entry.get('ideal_tp'),
        "rr":                         entry.get('rr'),
        "management_path":            entry.get('management_path'),
        "why_it_worked":              entry.get('why_it_worked'),
        "structural_pattern_keywords": entry.get('structural_pattern_keywords'),
        "bot_action":                 entry.get('bot_action'),
        "bot_gate":                   entry.get('bot_gate'),
        "actual_outcome":             entry.get('actual_outcome'),
        "outcome_r":                  entry.get('outcome_r'),
    }, indent=2)

    prompt = f"""
You are a trading pattern knowledge manager for a Gold (XAUUSD) trading bot.

A new ideal setup has been identified during post-mortem analysis.
Classify whether the structural PATTERN of this setup is already known,
or whether it adds new knowledge to the pattern library.

--- NEW MISSWISH SETUP ---
{entry_summary}

--- EXISTING KNOWN PATTERNS (misswish_keywords library) ---
{existing_text}

A pattern is defined by its MECHANICS — the sequence of market events
(sweep → displacement → OB tap → BOS) — NOT by direction, session, or price level.

Classify as exactly one of:
    DUPLICATE  — Structural mechanics are already fully captured.
    EXTENSION  — Same core mechanics, but this entry adds new context.
                 Provide the updated lesson that merges old and new.
    NEW        — Genuinely different structural mechanics not yet in the library.
                 Provide 3-6 keywords + a one-sentence lesson.

Respond ONLY with valid JSON in one of these three formats:

DUPLICATE:
{{"classification": "DUPLICATE", "reason": "brief explanation"}}

EXTENSION:
{{"classification": "EXTENSION", "entry_id": "existing_entry_id_to_update", "updated_lesson": "merged lesson text"}}

NEW:
{{"classification": "NEW", "keywords": ["kw1", "kw2", "kw3"], "lesson": "one clear sentence describing the structural pattern"}}
"""
    raw    = call_ai(prompt=prompt)   # FIX: uses call_ai, no genai
    result = _parse_json(raw)

    if not result:
        print(f"[MissWishTagger] Could not parse AI response for {entry_id}.")
        return

    classification = result.get('classification', '').upper()
    print(f"[MissWishTagger] Classification: {classification}")

    if classification == 'DUPLICATE':
        print(f"[MissWishTagger] DUPLICATE — pattern already known. "
              f"Reason: {result.get('reason', 'N/A')}.")
        _mark_tagged(entry_id)

    elif classification == 'EXTENSION':
        target_id    = result.get('entry_id', '').strip()
        updated_text = result.get('updated_lesson', '').strip()
        if not target_id or not updated_text:
            print(f"[MissWishTagger] EXTENSION missing entry_id or updated_lesson. Aborting.")
            return
        updated = False
        for e in existing_kw:
            if e.get('entry_id') == target_id:
                e['lesson'] = updated_text
                updated = True
                break
        if not updated:
            print(f"[MissWishTagger] EXTENSION target '{target_id}' not found — "
                  f"saving as NEW instead.")
            _save_new_entry(entry_id, entry,
                            entry.get('structural_pattern_keywords', []),
                            updated_text, existing_kw)
        else:
            _write_json(MISSWISH_KW_FILE, existing_kw)
            print(f"[MissWishTagger] EXTENSION — entry '{target_id}' lesson updated.")
            _mark_tagged(entry_id)

    elif classification == 'NEW':
        keywords = result.get('keywords', [])
        lesson   = result.get('lesson', '').strip()
        if not keywords:
            keywords = entry.get('structural_pattern_keywords', [])
            print(f"[MissWishTagger] Using entry's own keywords: {keywords}")
        if not keywords:
            print(f"[MissWishTagger] NEW — no keywords available. Aborting.")
            return
        _save_new_entry(entry_id, entry, keywords, lesson, existing_kw)

    else:
        print(f"[MissWishTagger] Unknown classification '{classification}'. Aborting.")


def _save_new_entry(entry_id: str, entry: dict, keywords: list,
                    lesson: str, existing_kw: list):
    new_kw_entry = {
        "entry_id":       entry_id,
        "date":           entry.get('date'),
        "keywords":       [str(k).lower().strip() for k in keywords],
        "lesson":         lesson,
        "signal":         entry.get('signal'),
        "session":        entry.get('session'),
        "regime_at_signal": entry.get('regime_at_signal'),
        "rr":             entry.get('rr'),
        "management_path": entry.get('management_path'),
        "why_it_worked":  entry.get('why_it_worked'),
        "ideal_entry":    entry.get('ideal_entry'),
        "ideal_sl":       entry.get('ideal_sl'),
        "ideal_tp":       entry.get('ideal_tp'),
        "bot_action":     entry.get('bot_action'),
        "actual_outcome": entry.get('actual_outcome'),
        "outcome_r":      entry.get('outcome_r'),
    }
    existing_kw.append(new_kw_entry)
    # FIX BUG C5 rolling cap: use a local variable so the write gets the trimmed list
    if len(existing_kw) > 500:
        existing_kw = existing_kw[-500:]
    _write_json(MISSWISH_KW_FILE, existing_kw)
    _mark_tagged(entry_id)
    print(f"[MissWishTagger] NEW entry saved. Keywords: {keywords}")
    print(f"[MissWishTagger] Lesson: {lesson}")
