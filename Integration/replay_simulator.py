"""
replay_simulator.py — Historical Day Replay Engine
====================================================
Replays a single trading day against MT5 historical data.
Uses ai_client.py for all AI calls → Claude Sonnet 4.6, 3-key rotation.

FIX C1: Removed bare API_KEY reference. All AI calls now go through call_ai().
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL, AI_DISPLAY_NAME   # FIX C1: replaces genai.Client(api_key=API_KEY)

import MetaTrader5 as mt5
import pandas as pd
import pytz
import time
import json
import re
import os
import sys
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.dirname(current_dir)
sys.path.append(os.path.join(base_dir, "Python Files"))
sys.path.append(os.path.join(base_dir, "Strategy"))
sys.path.append(os.path.join(base_dir, "Memory"))

import strategy_rules
import strategy_logic
import memory_manager
import thought_logger

from dotenv import load_dotenv
load_dotenv()

SYMBOL = "XAUUSD"
NY_TZ  = pytz.timezone('America/New_York')


def _get_broker_tz():
    """Dynamic broker timezone detection (DST aware). Replaces hardcoded offset."""
    try:
        if not mt5.initialize():
            return pytz.timezone('Etc/GMT-3')
        tick = mt5.symbol_info_tick(SYMBOL)
        mt5.shutdown()
        if tick:
            broker_ts = tick.time
            utc_now   = datetime.utcnow()
            broker_dt = datetime.utcfromtimestamp(broker_ts)
            offset_hrs = round((broker_dt - utc_now).total_seconds() / 3600)
            if offset_hrs == 3:
                return pytz.timezone('Etc/GMT-3')
    except Exception:
        pass
    return pytz.timezone('Etc/GMT-2')


def get_historical_data_for_replay(target_date_str):
    """Pulls historical data and formats it for the replay loop."""
    print(f"Pulling MT5 Data for {target_date_str}...")
    if not mt5.initialize():
        return None, None

    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    start_time  = target_date - timedelta(days=5)
    end_time    = target_date + timedelta(days=1)

    rates_m5  = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5,  start_time, end_time)
    rates_m15 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, start_time, end_time)
    mt5.shutdown()

    if rates_m5 is None or rates_m15 is None or len(rates_m5) == 0 or len(rates_m15) == 0:
        return None, None

    broker_tz = _get_broker_tz()

    def format_df(rates):
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['ny_time'] = (
            df['time']
            .dt.tz_localize(broker_tz)
            .dt.tz_convert(NY_TZ)
            .dt.tz_localize(None)
        )
        df.set_index('ny_time', inplace=True)
        return df

    return format_df(rates_m5), format_df(rates_m15)


def build_simulated_context(df_m5, df_m15, current_time):
    """Builds the market context string for a specific point in history."""
    past_m5  = df_m5.loc[:current_time]
    past_m15 = df_m15.loc[:current_time]

    if past_m5.empty or past_m15.empty:
        return None, None

    current_price = past_m5.iloc[-1]['close']

    context = f"""
    --- SIMULATED LIVE MARKET FEED ---
    Current NY Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
    Current Live Price: {current_price}

    --- LAST 15 M15 CANDLES (Structure) ---
    {past_m15.tail(15)[['open', 'high', 'low', 'close']].to_csv(date_format='%H:%M')}

    --- LAST 15 M5 CANDLES (Execution) ---
    {past_m5.tail(15)[['open', 'high', 'low', 'close']].to_csv(date_format='%H:%M')}
    """
    return context, current_price


def ask_ai(context, current_state, past_lessons):
    """
    Sends the snapshot to Claude and returns a parsed JSON decision.
    FIX C1: Uses call_ai() — no more direct API_KEY / genai usage.
    """
    logic_framework = strategy_logic.get_analytical_framework()
    execution_rules = strategy_rules.get_execution_rules()

    prompt = f"""
    You are a Multi-Disciplinary Gold Trader in a time-compressed simulation.

    --- YOUR PREVIOUS INTERNAL MONOLOGUE ---
    Your last bias was: {current_state.get('current_bias', 'NEUTRAL')}
    Your ongoing thesis was: {current_state.get('active_thesis', 'Searching for setup.')}

    --- MARKET DATA ---
    {context}

    --- ANALYTICAL FRAMEWORK ---
    {logic_framework}

    --- EXECUTION RULES ---
    {execution_rules}

    --- PAST LESSONS & HINDSIGHT ---
    {past_lessons}

    Analyze the data. Calculate Confluence Score (0-3). Decide to BUY, SELL, WAIT, HOLD, or CLOSE.

    RESPOND ONLY IN VALID JSON:
    {{
        "signal": "BUY" | "SELL" | "WAIT" | "HOLD" | "CLOSE",
        "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
        "confluence_score": 0-3,
        "reasoning": "Technical breakdown...",
        "thesis": "Overall summary...",
        "entry": price, "sl": price, "tp": price
    }}
    """
    # FIX C1: call_ai() handles key rotation automatically
    raw_text = call_ai(prompt=prompt)
    if raw_text is None:
        return {"signal": "WAIT"}

    try:
        candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[Replay] JSON parse error: {e}")

    return {"signal": "WAIT"}


def run_day_replay(target_date_str):
    print(f"STARTING FAST-FORWARD REPLAY FOR: {target_date_str}")
    df_m5, df_m15 = get_historical_data_for_replay(target_date_str)

    if df_m5 is None:
        print("Could not get data.")
        return

    day_candles = df_m5[df_m5.index.date == datetime.strptime(target_date_str, "%Y-%m-%d").date()]
    ny_session  = day_candles.between_time('07:00', '16:00')

    active_trade = None

    for current_time, candle in ny_session.iterrows():
        print(f"\n[SIMULATED TIME: {current_time.strftime('%H:%M')}] | Price: {candle['close']}")

        if active_trade:
            sl         = active_trade['sl']
            tp         = active_trade['tp']
            trade_type = active_trade['type']

            hit_sl = (candle['low']  <= sl and trade_type == 'BUY') or \
                     (candle['high'] >= sl and trade_type == 'SELL')
            hit_tp = (candle['high'] >= tp and trade_type == 'BUY') or \
                     (candle['low']  <= tp and trade_type == 'SELL')

            if hit_sl:
                print(f"STOP LOSS HIT at {sl}!")
                memory_manager.update_final_review(active_trade['ticket'], "LOSS", "Hit SL during replay.")
                active_trade = None
                continue

            if hit_tp:
                print(f"TAKE PROFIT HIT at {tp}!")
                memory_manager.update_final_review(active_trade['ticket'], "WIN", "Hit TP during replay.")
                active_trade = None
                continue

            print(f"   Trade active ({trade_type}). Floating... skipping AI call.")
            continue

        context, live_price = build_simulated_context(df_m5, df_m15, current_time)
        current_state       = thought_logger.get_current_state()
        past_lessons        = memory_manager.get_recent_memories()

        decision = ask_ai(context, current_state, past_lessons)
        signal   = decision.get("signal", "WAIT").upper()

        print(f"AI Decision: {signal} | Bias: {decision.get('bias')} | "
              f"Score: {decision.get('confluence_score')}")

        if signal == "CLOSE":
            if active_trade:
                print("AI ordered CLOSE. Exiting active trade.")
                memory_manager.update_final_review(
                    active_trade['ticket'], "CLOSED_BY_AI",
                    f"AI detected reversal at {candle['close']} and ordered exit."
                )
                active_trade = None
            else:
                print("CLOSE signal received but no active trade.")

        elif signal in ["BUY", "SELL"]:
            entry  = candle['close']
            sl_val = float(decision.get('sl', 0))
            tp_val = float(decision.get('tp', 0))

            print(f"   Executing Virtual Trade: {signal} at {entry} (SL: {sl_val}, TP: {tp_val})")

            ticket = f"REPLAY_{current_time.strftime('%Y%m%d%H%M')}"
            memory_manager.log_trade(
                ticket=ticket, signal=signal,
                reasoning=decision.get('reasoning'),
                entry_price=entry, sl=sl_val, tp=tp_val
            )
            active_trade = {
                "ticket": ticket,
                "type":   signal,
                "entry":  entry,
                "sl":     sl_val,
                "tp":     tp_val,
            }

        elif signal in ["WAIT", "HOLD"]:
            print(f"   {signal}. No action taken.")

        thought_logger.update_state(
            decision.get('bias',      'NEUTRAL'),
            decision.get('thesis',    ''),
            decision.get('reasoning', ''),
            bool(active_trade)
        )

        time.sleep(4)   # API throttle

    if active_trade:
        print(f"\nEnd of session. Trade {active_trade['ticket']} still open — marking as OPEN.")


if __name__ == "__main__":
    run_day_replay("2026-03-04")
