"""
trade_reviewer.py — Manual Post-Trade Deep Review
===================================================
BUG FIX: Removed `from google import genai` and direct genai.Client() call.
All AI calls now go through call_ai() via ai_client.py — same as every
other file in the bot.
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL

import sys
import os
import json

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.dirname(current_dir)
sys.path.append(os.path.join(base_dir, 'Memory'))

import thought_logger
import memory_manager

from dotenv import load_dotenv
load_dotenv()


def run_detailed_review():
    print("🔬 Starting Detailed ICT Post-Mortem...")

    # 1. Load market context
    from paths import AI_CONTEXT_PATH as _ctx_path
    try:
        with open(_ctx_path, "r", encoding="utf-8") as f:
            market_math = f.read()
    except FileNotFoundError:
        print("⚠️ latest_context.txt not found. Run main_bot.py at least once first.")
        market_math = "No market context available. The bot has not run yet."
    except Exception as e:
        print(f"⚠️ Could not read market context: {e}")
        market_math = f"Error reading context: {e}"

    # 2. Load continuation memory
    narrative = thought_logger.get_current_state()

    print("\n--- INPUT TRADE DATA ---")
    ticket  = input("MT5 Ticket Number: ")
    outcome = input("Result (WIN/LOSS/BE): ").upper()

    # 3. Ask Claude for deep analysis
    prompt = f"""
    Analyze this Gold Trade for my learning model.

    MARKET MATH AT ENTRY: {market_math}
    AI THOUGHTS DURING TRADE: {narrative['active_thesis']}
    FINAL OUTCOME: {outcome}

    Provide a 3-part 'Statement of Review':
    1. MECHANICAL ERROR: Did the AI miss a higher timeframe level?
    2. PSYCHOLOGICAL/LOGIC ERROR: Was the bias premature?
    3. IMPROVEMENT: One specific mathematical rule to prevent this in the future.
    """

    detailed_statement = call_ai(prompt=prompt)
    if not detailed_statement:
        detailed_statement = "[AI review unavailable — API call failed]"

    print(f"\n🤖 AI REVIEW STATEMENT:\n{detailed_statement}")

    # 4. Save to memory bank
    memory_manager.update_final_review(ticket, outcome, detailed_statement)
    print("\n✅ Learning Model Updated.")


if __name__ == "__main__":
    run_detailed_review()
