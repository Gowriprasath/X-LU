"""
historical_data.py — MT5 Historical Data Fetcher
=================================================
FIX A10: mt5.shutdown() is now in a try/finally block so it is
         always called, even if copy_rates_range throws.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta


def get_historical_day_data(symbol, date_str):
    """
    Connects to MT5 and pulls the full 15-Minute chart for a specific past date.
    date_str must be in 'YYYY-MM-DD' format.
    """
    if not mt5.initialize():
        print("❌ Failed to initialize MT5 for historical data.")
        return "ERROR: MT5 not connected."

    try:   # FIX A10: everything after initialize is inside try/finally
        try:
            start_dt = datetime.strptime(date_str, "%Y-%m-%d")
            end_dt   = start_dt + timedelta(days=1)
        except ValueError:
            return "ERROR: Invalid date format. Use YYYY-MM-DD."

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            return (f"⚠️ No market data found for {symbol} on {date_str}. "
                    f"Was it a weekend?")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        formatted_data  = f"--- {symbol} 15-MINUTE CHART FOR {date_str} ---\n"
        formatted_data += "Time (Broker) | Open | High | Low | Close\n"
        formatted_data += "-" * 50 + "\n"
        for _, row in df.iterrows():
            time_str = row['time'].strftime("%H:%M")
            formatted_data += (f"{time_str} | {row['open']:.3f} | "
                                f"{row['high']:.3f} | {row['low']:.3f} | "
                                f"{row['close']:.3f}\n")
        return formatted_data

    finally:
        mt5.shutdown()   # FIX A10: ALWAYS runs


if __name__ == "__main__":
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching data for {yesterday}...")
    print(get_historical_day_data("XAUUSD", yesterday))
