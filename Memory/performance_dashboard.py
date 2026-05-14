"""
performance_dashboard.py
========================
BUG-4 FIX: Removed `from google import genai` import and the dead
           `genai.Client(api_key=API_KEY)` call (API_KEY was also
           undefined, causing a NameError before genai was even reached).
           Migration was started (call_ai imported) but never completed.
           Now fully wired: call_ai() used for all AI calls.
"""
import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # BUG-4 FIX: now actually used (was imported but bypassed)

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.dirname(current_dir)
from paths import TRADE_MEMORY_PATH
MEMORY_FILE = TRADE_MEMORY_PATH


def run_dashboard():
    print("📊 Booting up the Multi-Disciplinary Performance Dashboard...")

    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            trades = json.load(f)
    except Exception as e:
        print(f"❌ Could not load trade memory: {e}")
        return

    completed_trades = [t for t in trades if t.get('result') in ['WIN', 'LOSS', 'BREAK EVEN']]

    if not completed_trades:
        print("ℹ️ Not enough completed trades (WIN/LOSS/BREAK EVEN) to generate a statistical report yet.")
        print("Let the bot run for a few days and log some closed trades first!")
        return

    trade_data_str = ""
    for t in completed_trades:
        trade_data_str += f"Ticket: {t.get('ticket')} | Result: {t.get('result')}\n"
        trade_data_str += f"Reasoning: {t.get('reasoning')}\n"
        trade_data_str += "-" * 30 + "\n"

    # BUG-4 FIX: Was genai.Client(api_key=API_KEY) — API_KEY was undefined → NameError
    prompt = f"""
    You are a Quantitative Data Scientist for an algorithmic trading desk.
    I am providing you with a list of historical trades. Our bot uses a "Rule of Two" Confluence model based on:
    1. ICT / SMC
    2. Classic TA (Support/Resistance/Trendlines)
    3. Elliott Wave Theory

    --- HISTORICAL TRADE DATA ---
    {trade_data_str}

    --- TASK ---
    1. Read the 'Reasoning' for each trade to deduce which specific combination of strategies (e.g., 'ICT + Classic TA' or 'Classic TA + Elliott Wave') triggered the entry.
    2. Generate a statistical 'Performance Matrix' report showing:
       - The Win Rate percentage for each specific combination (WIN / total, treating BREAK EVEN as 0.5 of a win).
       - Total Wins, Losses, and Break Evens per combination.
    3. Provide a brief, ruthless recommendation on which combination the bot should prioritize or ignore in the future.

    Format it beautifully as a terminal output. Keep it clean, mathematical, and highly structured.
    """

    try:
        print("🧠 AI Data Scientist is analyzing your trade history and calculating win rates...\n")
        response_text = call_ai(prompt=prompt)  # BUG-4 FIX: replaced genai.Client block

        if response_text:
            print("==================================================")
            print("📈 QUANTITATIVE CONFLUENCE MATRIX")
            print("==================================================")
            print(response_text.strip())
            print("==================================================")
        else:
            print("❌ AI returned no response for dashboard analysis.")

    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")


if __name__ == "__main__":
    run_dashboard()
