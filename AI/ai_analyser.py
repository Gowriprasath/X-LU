"""
ai_analyser.py — Standalone Market Analysis Tool
==================================================
RENAMED from claude_analyzer.py → ai_analyser.py (Claude migration).

FIX C2: Removed bare `API_KEY` reference — was undefined, caused NameError.
         All AI calls now go through call_ai() / ai_client.py.
         The file still imports from ai_client at the top (correct) and now
         ACTUALLY USES it (the old code imported it but then ignored it).

This file is a MANUAL diagnostic tool — run it directly from the terminal
for a one-shot market analysis snapshot.

It is NOT connected to main_bot.py and does NOT execute trades.
Use it to: manually inspect AI reasoning, debug memory, or test prompts.

To run: python AI/ai_analyser.py
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL, AI_DISPLAY_NAME  # FIX C2: actually used now

import sys
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(base_dir, "Python Files"))
sys.path.append(os.path.join(base_dir, "Strategy"))
sys.path.append(os.path.join(base_dir, "Memory"))

import data_extractor
import memory_manager
import strategy_rules


def run_smart_analysis():
    print(f"🔄 Initializing XAU/USD Algorithmic Pipeline ({AI_DISPLAY_NAME} — {AI_MODEL})...")

    market_context = data_extractor.get_live_market_data()
    if not market_context:
        print("❌ Could not get live market data.")
        return

    from paths import AI_CONTEXT_PATH as _ctx_path
    context_path = _ctx_path
    try:
        with open(context_path, "w", encoding="utf-8") as file:
            file.write(market_context)
    except Exception as e:
        print(f"[Analyser] Could not save context file: {e}")

    print("📖 Searching Memory Bank for past setups...")
    past_lessons = memory_manager.get_recent_memories(limit=3)

    print(f"🧠 Sending Live Data + Memories to {AI_DISPLAY_NAME} Logic Engine...")

    my_strategy_rules = strategy_rules.get_execution_rules()
    current_day       = datetime.now().strftime("%A")

    prompt = f"""
    You are an expert algorithmic ICT trader. Review the precise mathematical data below.

    --- LIVE CALENDAR ---
    Today is strictly: {current_day}

    {market_context}

    --- MY STRATEGY RULES ---
    {my_strategy_rules}

    --- PAST LESSONS & MEMORY ---
    Read these past trades to avoid making the same mistakes and check where
    we can enter to get profit instead of loss. Adjust your current bias and entry.
    If today's setup closely matches a failed setup from the past.
    {past_lessons}

    Based ONLY on the live data, my rules, and the past lessons, provide a clear
    breakdown of the current setup. If a valid trade exists, output the entry,
    stop loss, and target.
    """

    # FIX C2: call_ai() — handles provider, model, and 3-key rotation automatically
    response_text = call_ai(prompt=prompt)

    print("\n" + "=" * 60)
    print(f" 🤖 ICT GOLD ANALYSIS ({AI_DISPLAY_NAME.upper()} — MEMORY ENHANCED)")
    print("=" * 60)
    if response_text:
        print(response_text)
    else:
        print("❌ AI call failed. Check API keys in .env and try again.")
    print("=" * 60)


if __name__ == "__main__":
    run_smart_analysis()
