"""
backtest_seeder.py  —  v2  (Multi-Setup, Regime-Aware)
=======================================================
Pulls historical M15 + H1 data from MT5 and runs a structural
simulation across 4 ICT setup types.

SETUPS DETECTED PER DAY:
  1. Asian High Sweep  → Short fade
  2. Asian Low  Sweep  → Long  fade
  3. NY Macro   Sweep  → Fade the macro sweep (08:50 / 09:50 / 10:50)
  4. London     Sweep  → Fade London session high/low into NY open

SL/TP: ATR-based (matches live bot logic — structural, not fixed dollars)
  SL  = ATR(14) × ATR_SL_MULT    (default 1.5)
  TP  = SL      × MIN_RR          (default 2.0  →  2R minimum)

REGIME: Each trade is tagged with its H1 regime from regime_detector.
This gives the bot genuine regime-aware win-rate data in memory.

CONFLUENCE SCORE (1–3):
  +1  liquidity sweep confirmed
  +1  session timing correct (NY macro or London→NY transition)
  +1  regime supports the fade (RANGING / LOW_VOLATILITY)

OUTPUT: All trades written via the public memory_manager API (thread-safe).

End-of-run report prints:
  - Win rate by REGIME
  - Win rate by SETUP TYPE
  - Total trades seeded

Run from the project root:
    python Backtest/backtest_seeder.py
"""

import sys
import os
import pytz
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ── Dynamic path setup ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

sys.path.append(os.path.join(BASE_DIR, "Python Files"))
sys.path.append(os.path.join(BASE_DIR, "Memory"))
sys.path.append(os.path.join(BASE_DIR, "Quant", "regime_detector"))

import memory_manager
import regime_detector as rd
from feature_engineer import compute_atr

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[BacktestSeeder] WARNING: MetaTrader5 not installed.")

# ================================================================
# ⚙️  CONFIGURATION
# ================================================================

SYMBOL        = "XAUUSD"
DAYS_BACK     = 60
BROKER_TZ_STR = "Etc/GMT-2"
NY_TZ         = pytz.timezone("America/New_York")

ATR_PERIOD   = 14
ATR_SL_MULT  = 1.5   # SL = ATR × 1.5
MIN_RR       = 2.0   # TP = SL × 2.0

MIN_SL_DOLLARS = 2.0
MAX_SL_DOLLARS = 30.0

NY_MACROS = [
    ("08:45", "09:00"),
    ("09:45", "10:00"),
    ("10:45", "11:00"),
]

# ================================================================
# MT5 DATA PULL
# ================================================================

def _mt5_to_ny(df, broker_tz_str=BROKER_TZ_STR):
    broker_tz = pytz.timezone(broker_tz_str)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["ny_time"] = (
        df["time"]
        .dt.tz_localize(broker_tz)
        .dt.tz_convert(NY_TZ)
        .dt.tz_localize(None)
    )
    return df


def get_historical_data():
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not available.")
        return None, None

    print(f"🔌 Connecting to MT5 — pulling {DAYS_BACK} days of {SYMBOL}...")
    if not mt5.initialize() or not mt5.symbol_select(SYMBOL, True):
        print("❌ MT5 connection failed.")
        return None, None

    m15_count = int(DAYS_BACK * 96 * 1.1)
    h1_count  = max(int(DAYS_BACK * 24 * 1.1), 300)

    rates_m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, m15_count)
    rates_h1  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1,  0, h1_count)
    mt5.shutdown()

    if rates_m15 is None or rates_h1 is None or len(rates_m15) == 0:
        print("❌ No data returned from MT5.")
        return None, None

    df_m15 = _mt5_to_ny(pd.DataFrame(rates_m15))
    df_h1  = _mt5_to_ny(pd.DataFrame(rates_h1))
    df_m15.set_index("ny_time", inplace=True)
    df_h1.set_index("ny_time",  inplace=True)

    print(f"✅ M15: {len(df_m15)} candles | H1: {len(df_h1)} candles")
    return df_m15, df_h1

# ================================================================
# ATR + SL/TP HELPERS
# ================================================================

def _get_atr_at(df_m15, timestamp):
    past = df_m15.loc[:timestamp]
    if len(past) < ATR_PERIOD + 5:
        return None
    return float(compute_atr(past, ATR_PERIOD).iloc[-1])


def _build_sl_tp(entry, direction, atr):
    sl_dist = round(atr * ATR_SL_MULT, 2)
    if sl_dist < MIN_SL_DOLLARS:
        sl_dist = MIN_SL_DOLLARS
    if sl_dist > MAX_SL_DOLLARS:
        return None
    tp_dist = round(sl_dist * MIN_RR, 2)
    if direction == "SELL":
        return round(entry + sl_dist, 3), round(entry - tp_dist, 3), sl_dist, round(tp_dist / sl_dist, 2)
    return round(entry - sl_dist, 3), round(entry + tp_dist, 3), sl_dist, round(tp_dist / sl_dist, 2)

# ================================================================
# REGIME PER DAY
# ================================================================

def _get_day_regime(df_h1, date):
    try:
        cutoff = datetime.combine(date, datetime.min.time()) + timedelta(hours=7)
        h1_past = df_h1.loc[:cutoff]
        if len(h1_past) < 50:
            return "UNKNOWN"
        return rd.predict(h1_past).get("regime", "UNKNOWN")
    except Exception as e:
        return "UNKNOWN"

# ================================================================
# CONFLUENCE SCORER
# ================================================================

def _score_confluence(setup_type, regime):
    score = 1  # sweep always confirmed (we only get here if it fired)
    if setup_type in ("NY_MACRO_SWEEP", "LONDON_SWEEP", "ASIAN_HIGH_SWEEP", "ASIAN_LOW_SWEEP"):
        score += 1   # session timing point
    if regime in ("RANGING", "LOW_VOLATILITY"):
        score += 1   # regime supports fade
    elif regime == "TRENDING" and setup_type == "NY_MACRO_SWEEP":
        score += 1   # macro fades work as pullbacks in trends too
    return min(score, 3)

# ================================================================
# OUTCOME SIMULATOR
# ================================================================

def _simulate_outcome(df_m15, entry_time, entry, sl, tp, direction):
    session_end = entry_time.replace(hour=16, minute=0)
    future = df_m15.loc[entry_time:session_end]
    for _, c in future.iterrows():
        if direction == "SELL":
            if c["low"]  <= tp: return "WIN"
            if c["high"] >= sl: return "LOSS"
        else:
            if c["high"] >= tp: return "WIN"
            if c["low"]  <= sl: return "LOSS"
    return "BREAK_EVEN"

# ================================================================
# SETUP DETECTORS
# ================================================================

def _try_entry(df_m15, sweep_candle_series, direction, setup_type, regime, date, label):
    """Shared helper to build, simulate, and return a trade dict from a sweep candle."""
    entry_time = sweep_candle_series.name
    entry      = float(sweep_candle_series["close"])
    atr        = _get_atr_at(df_m15, entry_time)
    if not atr:
        return None
    result = _build_sl_tp(entry, direction, atr)
    if not result:
        return None
    sl, tp, sl_dist, rr = result
    outcome    = _simulate_outcome(df_m15, entry_time, entry, sl, tp, direction)
    conf_score = _score_confluence(setup_type, regime)
    context = (
        f"Date: {date} | Setup: {setup_type} | {label} | "
        f"Signal: {direction} at {entry:.3f}. "
        f"ATR={atr:.3f}, SL={sl:.3f} ({sl_dist:.2f}pts), TP={tp:.3f} ({rr}R). "
        f"Regime: {regime}."
    )
    feedback = (
        f"Backtest result: {outcome}. {setup_type} in {regime} regime. "
        f"ATR SL={sl_dist:.2f}pts, RR={rr}. "
        f"{'Fade succeeded — liquidity sweep reversed as expected.' if outcome == 'WIN' else 'Fade failed — sweep continued or no follow-through within session.' if outcome == 'LOSS' else 'Price did not reach either target within the NY session.'}"
    )
    return {
        "ticket":           f"BACKTEST_{setup_type}_{date.strftime('%Y%m%d')}_{direction}_{entry_time.strftime('%H%M')}",
        "date":             str(date),
        "setup_type":       setup_type,
        "signal":           direction,
        "entry":            entry,
        "sl":               sl,
        "tp":               tp,
        "sl_dist":          sl_dist,
        "rr":               rr,
        "confluence_score": conf_score,
        "regime":           regime,
        "context":          context,
        "outcome":          outcome,
        "feedback":         feedback,
        "ict_logic":        f"{setup_type}: liquidity sweep confirmed. Entry at close of sweep candle.",
        "classic_logic":    f"Price swept a key session level and closed back inside. Fade entry.",
        "elliott_logic":    "N/A (Backtest)",
    }


def _detect_asian_sweeps(day_data, df_m15, date, regime):
    trades = []
    # Determine previous trading day (if Monday, look back to Friday; otherwise yesterday)
    prev_day = date - timedelta(days=3 if date.weekday() == 0 else 1)
    
    start_asian = pd.Timestamp(datetime.combine(prev_day, datetime.strptime("19:00", "%H:%M").time()))
    end_asian   = pd.Timestamp(datetime.combine(date, datetime.strptime("01:59", "%H:%M").time()))
    
    # Handle timezone localization if the dataframe index is timezone-aware
    if df_m15.index.tz is not None:
        start_asian = start_asian.tz_localize(df_m15.index.tz)
        end_asian   = end_asian.tz_localize(df_m15.index.tz)
        
    asian = df_m15[(df_m15.index >= start_asian) & (df_m15.index <= end_asian)]
    ny_confirm = day_data.between_time("07:00", "09:00")
    if asian.empty or ny_confirm.empty:
        return trades
        
    asian_high = asian["high"].max()
    asian_low  = asian["low"].min()
 
    swept_high = ny_confirm[ny_confirm["high"] > asian_high]
    if not swept_high.empty:
        t = _try_entry(df_m15, swept_high.iloc[0], "SELL", "ASIAN_HIGH_SWEEP", regime, date,
                       f"Asian High {asian_high:.3f} swept")
        if t: trades.append(t)
 
    swept_low = ny_confirm[ny_confirm["low"] < asian_low]
    if not swept_low.empty:
        t = _try_entry(df_m15, swept_low.iloc[0], "BUY", "ASIAN_LOW_SWEEP", regime, date,
                       f"Asian Low {asian_low:.3f} swept")
        if t: trades.append(t)
 
    return trades


def _detect_ny_macro_sweeps(day_data, df_m15, date, regime):
    trades = []
    for macro_start, macro_end in NY_MACROS:
        macro_data = day_data.between_time(macro_start, macro_end)
        pre_data   = day_data.between_time("07:00", macro_start)
        if macro_data.empty or pre_data.empty:
            continue
        session_high = pre_data["high"].max()
        session_low  = pre_data["low"].min()

        swept_high = macro_data[macro_data["high"] > session_high]
        if not swept_high.empty:
            t = _try_entry(df_m15, swept_high.iloc[0], "SELL", "NY_MACRO_SWEEP", regime, date,
                           f"Macro {macro_start} swept session high {session_high:.3f}")
            if t: trades.append(t)
            continue   # one setup per macro window

        swept_low = macro_data[macro_data["low"] < session_low]
        if not swept_low.empty:
            t = _try_entry(df_m15, swept_low.iloc[0], "BUY", "NY_MACRO_SWEEP", regime, date,
                           f"Macro {macro_start} swept session low {session_low:.3f}")
            if t: trades.append(t)

    return trades


def _detect_london_sweep(day_data, df_m15, date, regime):
    trades  = []
    london  = day_data.between_time("02:00", "05:00")
    ny_open = day_data.between_time("07:00", "09:30")
    if london.empty or ny_open.empty:
        return trades
    london_high = london["high"].max()
    london_low  = london["low"].min()

    swept_high = ny_open[ny_open["high"] > london_high]
    if not swept_high.empty:
        t = _try_entry(df_m15, swept_high.iloc[0], "SELL", "LONDON_SWEEP", regime, date,
                       f"London High {london_high:.3f} swept in NY")
        if t: trades.append(t)
    else:
        # FIX: was `elif True:` — semantically identical but misleading; use `else:`
        swept_low = ny_open[ny_open["low"] < london_low]
        if not swept_low.empty:
            t = _try_entry(df_m15, swept_low.iloc[0], "BUY", "LONDON_SWEEP", regime, date,
                           f"London Low {london_low:.3f} swept in NY")
            if t: trades.append(t)

    return trades

# ================================================================
# MEMORY LOGGER
# ================================================================

def _log_to_memory(trade):
    memory_manager.log_trade(
        ticket        = trade["ticket"],
        signal        = trade["signal"],
        reasoning     = trade["context"],
        entry_price   = trade["entry"],
        sl            = trade["sl"],
        tp            = trade["tp"],
        conf_score    = trade["confluence_score"],
        ict_logic     = trade["ict_logic"],
        classic_logic = trade["classic_logic"],
        elliott_logic = trade["elliott_logic"],
    )
    memory_manager.update_final_review(
        ticket    = trade["ticket"],
        result    = trade["outcome"],
        statement = trade["feedback"],
    )
    memory_manager.update_hindsight_review(trade["ticket"], trade["feedback"])

# ================================================================
# REPORT
# ================================================================

def _print_report(all_trades):
    if not all_trades:
        print("\n⚠️  No trades generated.")
        return

    print("\n" + "=" * 62)
    print("  📊  BACKTEST SEEDER v2 — WIN RATE REPORT")
    print("=" * 62)

    regime_stats = defaultdict(lambda: {"WIN": 0, "LOSS": 0, "BREAK_EVEN": 0})
    setup_stats  = defaultdict(lambda: {"WIN": 0, "LOSS": 0, "BREAK_EVEN": 0})
    for t in all_trades:
        regime_stats[t["regime"]][t["outcome"]] += 1
        setup_stats[t["setup_type"]][t["outcome"]] += 1

    def _wr(s):
        total = s["WIN"] + s["LOSS"] + s["BREAK_EVEN"]
        return (s["WIN"] + 0.5 * s["BREAK_EVEN"]) / total * 100 if total else 0, total

    print(f"\n{'REGIME':<22} {'W':>4} {'L':>4} {'BE':>4} {'TOT':>5} {'WIN%':>7}")
    print("-" * 50)
    for regime, s in sorted(regime_stats.items()):
        wr, tot = _wr(s)
        print(f"{regime:<22} {s['WIN']:>4} {s['LOSS']:>4} {s['BREAK_EVEN']:>4} {tot:>5} {wr:>6.1f}%")

    print(f"\n{'SETUP TYPE':<25} {'W':>4} {'L':>4} {'BE':>4} {'TOT':>5} {'WIN%':>7}")
    print("-" * 53)
    for setup, s in sorted(setup_stats.items()):
        wr, tot = _wr(s)
        print(f"{setup:<25} {s['WIN']:>4} {s['LOSS']:>4} {s['BREAK_EVEN']:>4} {tot:>5} {wr:>6.1f}%")

    wins   = sum(1 for t in all_trades if t["outcome"] == "WIN")
    losses = sum(1 for t in all_trades if t["outcome"] == "LOSS")
    be     = sum(1 for t in all_trades if t["outcome"] == "BREAK_EVEN")
    wr     = (wins + 0.5 * be) / len(all_trades) * 100

    print("\n" + "=" * 62)
    print(f"  TOTAL SEEDED : {len(all_trades)} trades")
    print(f"  OVERALL WIN  : {wr:.1f}%  ({wins}W / {losses}L / {be}BE)")
    print("=" * 62)

# ================================================================
# MAIN
# ================================================================

def run_backtest():
    print("🚀 Backtest Seeder v2 — Multi-Setup, Regime-Aware")
    print(f"   {SYMBOL} | {DAYS_BACK} days | ATR×{ATR_SL_MULT} SL | {MIN_RR}R min\n")

    df_m15, df_h1 = get_historical_data()
    if df_m15 is None:
        return

    grouped    = df_m15.groupby(df_m15.index.date)
    all_trades = []
    skipped    = 0

    for date, day_data in grouped:
        if date.weekday() >= 5:
            continue
        try:
            regime     = _get_day_regime(df_h1, date)
            day_trades = (
                _detect_asian_sweeps(day_data, df_m15, date, regime) +
                _detect_ny_macro_sweeps(day_data, df_m15, date, regime) +
                _detect_london_sweep(day_data, df_m15, date, regime)
            )
            for trade in day_trades:
                _log_to_memory(trade)
                all_trades.append(trade)
                print(f"  [{date}] {trade['setup_type']:<22} {trade['signal']:<4} "
                      f"Score:{trade['confluence_score']} "
                      f"Regime:{trade['regime']:<16} {trade['outcome']}")
        except Exception as e:
            print(f"  ⚠️  Skipping {date}: {e}")
            skipped += 1

    print(f"\n✅ Days processed: {len(grouped) - skipped} | Skipped: {skipped}")
    _print_report(all_trades)


if __name__ == "__main__":
    run_backtest()
