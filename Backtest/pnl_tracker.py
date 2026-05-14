"""
Backtest/pnl_tracker.py — Full PnL Analytics Engine
=====================================================
Plugs into backtest_engine.py via 3 calls:
    1. init_tracker(initial_balance)         ← once at start
    2. record_trade(...)                     ← once per completed trade
    3. print_full_report()                   ← once at end (or mid-run)

All 8 requested features implemented here:
    ✅  1. Equity curve (ASCII chart — no external libs needed)
    ✅  2. Slippage model       (in spread_simulator.py — cost reflected in PnL)
    ✅  3. Spread widening during news (in spread_simulator.py)
    ✅  4. Per-regime PnL breakdown
    ✅  5. Session PnL breakdown
    ✅  6. Walk-forward validation
    ✅  7. Max consecutive loss tracker
    ✅  8. Monte Carlo simulation (1000 shuffles)

None of these touch the backtest flow logic — they are purely
analytical functions called after trades are recorded.
"""

import json
import math
import random
import os
import sys
from datetime import datetime
from collections import defaultdict

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from paths import PNL_STATS_PATH

# ── Internal state ─────────────────────────────────────────────────────────
_initial_balance: float  = 10_000.0
_equity_curve:   list    = []   # [(timestamp_str, balance)]
_trades:         list    = []   # full trade records

# ── Regime / session accumulators ─────────────────────────────────────────
_regime_stats:  dict = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
_session_stats: dict = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
_monthly_stats: dict = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})


# ================================================================
# INITIALISATION
# ================================================================
def init_tracker(initial_balance: float = 10_000.0):
    """Call once at backtest start to set the opening balance."""
    global _initial_balance, _equity_curve, _trades
    global _regime_stats, _session_stats, _monthly_stats
    _initial_balance = initial_balance
    _equity_curve    = [(None, initial_balance)]
    _trades          = []
    _regime_stats    = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    _session_stats   = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    _monthly_stats   = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})


# ================================================================
# RECORD TRADE
# ================================================================
def record_trade(ticket: str, pnl: float, result: str,
                 regime: str = "", session: str = "",
                 signal: str = "", timestamp: str = ""):
    """
    Records a completed trade and updates all accumulators.
    Call this once per completed trade from backtest_engine.py.

    Args:
        ticket:    trade ticket string
        pnl:       dollar PnL (positive = profit, negative = loss)
        result:    "WIN" | "LOSS" | "TIMEOUT"
        regime:    regime label at trade entry (e.g. "BULL_TREND")
        session:   session label at trade entry (e.g. "NY_AM")
        signal:    "BUY" or "SELL"
        timestamp: ISO string or datetime of trade close
    """
    current_balance = _equity_curve[-1][1] + pnl
    _equity_curve.append((timestamp, round(current_balance, 2)))

    trade_rec = {
        "ticket":    ticket,
        "pnl":       round(pnl, 2),
        "result":    result,
        "regime":    regime or "UNKNOWN",
        "session":   session or "UNKNOWN",
        "signal":    signal,
        "timestamp": timestamp,
        "balance":   round(current_balance, 2),
    }
    _trades.append(trade_rec)

    # ── Regime accumulator
    rs = _regime_stats[regime or "UNKNOWN"]
    rs["pnl"] += pnl
    if result == "WIN":
        rs["wins"] += 1
    elif result in ("LOSS", "TIMEOUT"):
        rs["losses"] += 1

    # ── Session accumulator
    ss = _session_stats[session or "UNKNOWN"]
    ss["pnl"] += pnl
    if result == "WIN":
        ss["wins"] += 1
    elif result in ("LOSS", "TIMEOUT"):
        ss["losses"] += 1

    # ── Monthly accumulator
    month_key = _parse_month(timestamp)
    ms = _monthly_stats[month_key]
    ms["trades"] += 1
    ms["pnl"]    += pnl
    if result == "WIN":
        ms["wins"] += 1
    elif result in ("LOSS", "TIMEOUT"):
        ms["losses"] += 1


# ================================================================
# CORE METRICS
# ================================================================
def _pnl_list() -> list:
    """Returns list of per-trade PnL values in chronological order."""
    return [t["pnl"] for t in _trades]


def _balance_series() -> list:
    """Returns balance at each point in time (including starting balance)."""
    return [pt[1] for pt in _equity_curve]


def compute_max_drawdown(balances: list = None) -> dict:
    """
    Computes maximum drawdown from peak.
    Returns:
        {"abs": float, "pct": float, "peak": float, "trough": float}
    """
    if balances is None:
        balances = _balance_series()
    if len(balances) < 2:
        return {"abs": 0.0, "pct": 0.0, "peak": balances[0] if balances else 0, "trough": 0}

    peak      = balances[0]
    max_dd    = 0.0
    peak_val  = balances[0]
    trough_val = balances[0]

    for b in balances:
        if b > peak:
            peak = b
        dd = peak - b
        if dd > max_dd:
            max_dd     = dd
            peak_val   = peak
            trough_val = b

    max_dd_pct = (max_dd / peak_val * 100) if peak_val > 0 else 0.0
    return {"abs": round(max_dd, 2), "pct": round(max_dd_pct, 2),
            "peak": round(peak_val, 2), "trough": round(trough_val, 2)}


def compute_sharpe(pnl_list: list = None, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio from per-trade PnL.
    Assumes ~1,000 trades per year (adjustable via TRADES_PER_YEAR).
    """
    TRADES_PER_YEAR = 1000
    if pnl_list is None:
        pnl_list = _pnl_list()
    if len(pnl_list) < 2:
        return 0.0
    n    = len(pnl_list)
    mean = sum(pnl_list) / n
    var  = sum((x - mean) ** 2 for x in pnl_list) / (n - 1)
    std  = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return round((mean - risk_free_rate) / std * math.sqrt(TRADES_PER_YEAR), 3)


def compute_profit_factor(pnl_list: list = None) -> float:
    """Gross profit / gross loss. >1.0 = profitable system."""
    if pnl_list is None:
        pnl_list = _pnl_list()
    gross_profit = sum(p for p in pnl_list if p > 0)
    gross_loss   = sum(abs(p) for p in pnl_list if p < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 3)


def compute_expectancy(pnl_list: list = None) -> float:
    """Average dollar return per trade (expectancy)."""
    if pnl_list is None:
        pnl_list = _pnl_list()
    if not pnl_list:
        return 0.0
    return round(sum(pnl_list) / len(pnl_list), 2)


# ================================================================
# FEATURE 7 — MAX CONSECUTIVE LOSS TRACKER
# ================================================================
def compute_consecutive_stats(trades: list = None) -> dict:
    """
    Computes max consecutive wins and losses, plus current streak.

    Returns dict:
        max_consec_losses:  int — worst losing streak
        max_consec_wins:    int — best winning streak
        current_streak:     int — current streak (+ = winning, - = losing)
        avg_loss_streak:    float — average length of losing streaks
        losing_streaks:     list[int] — all losing streak lengths
    """
    if trades is None:
        trades = _trades

    if not trades:
        return {
            "max_consec_losses": 0, "max_consec_wins": 0,
            "current_streak": 0, "avg_loss_streak": 0.0,
            "losing_streaks": [],
        }

    max_losses = 0
    max_wins   = 0
    cur_losses = 0
    cur_wins   = 0
    losing_streaks = []

    for t in trades:
        is_win = t["result"] == "WIN"
        if is_win:
            if cur_losses > 0:
                losing_streaks.append(cur_losses)
            cur_losses = 0
            cur_wins  += 1
            max_wins   = max(max_wins, cur_wins)
        else:
            if cur_wins > 0:
                cur_wins = 0
            cur_losses += 1
            max_losses  = max(max_losses, cur_losses)

    if cur_losses > 0:
        losing_streaks.append(cur_losses)

    # Current streak: positive = winning, negative = losing
    last = trades[-1]["result"]
    if last == "WIN":
        current_streak = cur_wins
    else:
        current_streak = -cur_losses

    avg_loss = (sum(losing_streaks) / len(losing_streaks)) if losing_streaks else 0.0

    return {
        "max_consec_losses": max_losses,
        "max_consec_wins":   max_wins,
        "current_streak":    current_streak,
        "avg_loss_streak":   round(avg_loss, 1),
        "losing_streaks":    losing_streaks,
    }


# ================================================================
# FEATURE 8 — MONTE CARLO SIMULATION
# ================================================================
def run_monte_carlo(n_sims: int = 1000, trades: list = None,
                    initial_balance: float = None) -> dict:
    """
    Shuffles trade order N times to build distribution of possible outcomes.

    This answers: "Is our backtest result a lucky ordering, or is the edge
    robust regardless of trade sequence?"

    Returns dict:
        median_final_balance:   float
        pct_profitable:         float  (% of sims that ended above start)
        worst_drawdown_median:  float  (median max drawdown across sims)
        worst_drawdown_95pct:   float  (95th-percentile max drawdown)
        final_balance_5pct:     float  (5th-percentile final balance — tail risk)
        final_balance_95pct:    float  (95th-percentile final balance)
        final_balances:         list   (all N final balances, sorted)
        max_drawdowns:          list   (all N max drawdowns)
        actual_final_balance:   float
        actual_max_drawdown:    float
    """
    if trades is None:
        trades = _trades
    if initial_balance is None:
        initial_balance = _initial_balance

    if not trades:
        return {"error": "No trades to simulate."}

    pnl_list = [t["pnl"] for t in trades]
    actual_balances = _balance_series()
    actual_dd       = compute_max_drawdown(actual_balances)

    final_balances = []
    max_drawdowns  = []

    for _ in range(n_sims):
        shuffled = pnl_list[:]
        random.shuffle(shuffled)

        balance = initial_balance
        peak    = initial_balance
        max_dd  = 0.0
        for pnl in shuffled:
            balance += pnl
            if balance > peak:
                peak = balance
            dd = peak - balance
            if dd > max_dd:
                max_dd = dd

        final_balances.append(round(balance, 2))
        max_drawdowns.append(round(max_dd, 2))

    final_balances.sort()
    max_drawdowns.sort()

    n = len(final_balances)
    profitable = sum(1 for b in final_balances if b > initial_balance)

    return {
        "n_simulations":            n_sims,
        "median_final_balance":     round(_percentile(final_balances, 50), 2),
        "pct_profitable":           round(profitable / n_sims * 100, 1),
        "worst_drawdown_median":    round(_percentile(max_drawdowns, 50), 2),
        "worst_drawdown_95pct":     round(_percentile(max_drawdowns, 95), 2),
        "final_balance_5pct":       round(_percentile(final_balances, 5), 2),
        "final_balance_95pct":      round(_percentile(final_balances, 95), 2),
        "final_balances":           final_balances,
        "max_drawdowns":            max_drawdowns,
        "actual_final_balance":     round(actual_balances[-1] if actual_balances else initial_balance, 2),
        "actual_max_drawdown":      actual_dd["abs"],
    }


def _percentile(sorted_list: list, pct: float) -> float:
    """Returns the p-th percentile of a sorted list."""
    if not sorted_list:
        return 0.0
    idx = (len(sorted_list) - 1) * pct / 100
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_list) - 1)
    frac = idx - lo
    return sorted_list[lo] * (1 - frac) + sorted_list[hi] * frac


# ================================================================
# FEATURE 6 — WALK-FORWARD VALIDATION
# ================================================================
def run_walk_forward(split_pct: float = 0.7, trades: list = None,
                     initial_balance: float = None) -> dict:
    """
    Splits trades into in-sample (IS) and out-of-sample (OOS) windows.

    The IS window trains us on what the strategy looked like in the past.
    The OOS window shows how it would have performed on unseen data.

    Healthy signs:
      - OOS win rate within 5% of IS win rate → model generalises
      - OOS Sharpe > 0.5 → positive edge on unseen data
      - OOS profit factor > 1.0 → positive expectancy out-of-sample

    Warning signs:
      - OOS win rate much lower than IS → overfitting
      - OOS max drawdown >> IS max drawdown → real-world conditions harder

    Args:
        split_pct: fraction of trades used as in-sample (default 0.7 = 70/30)
        trades:    list of trade dicts (uses module-level _trades by default)
        initial_balance: starting balance (uses module-level _initial_balance)

    Returns dict with IS and OOS metrics.
    """
    if trades is None:
        trades = _trades
    if initial_balance is None:
        initial_balance = _initial_balance

    if len(trades) < 10:
        return {"error": "Not enough trades for walk-forward split (need ≥ 10)."}

    split_idx = int(len(trades) * split_pct)
    is_trades  = trades[:split_idx]
    oos_trades = trades[split_idx:]

    def metrics_for(tlist, starting_balance):
        if not tlist:
            return {}
        pnl_list = [t["pnl"] for t in tlist]
        wins     = sum(1 for t in tlist if t["result"] == "WIN")
        losses   = sum(1 for t in tlist if t["result"] in ("LOSS", "TIMEOUT"))
        total    = wins + losses

        # Equity curve for this window
        bal = starting_balance
        balances = [bal]
        for p in pnl_list:
            bal += p
            balances.append(round(bal, 2))

        dd = compute_max_drawdown(balances)
        return {
            "trades":          len(tlist),
            "wins":            wins,
            "losses":          losses,
            "win_rate":        round(wins / total * 100, 1) if total > 0 else 0.0,
            "total_pnl":       round(sum(pnl_list), 2),
            "final_balance":   round(balances[-1], 2),
            "profit_factor":   compute_profit_factor(pnl_list),
            "sharpe":          compute_sharpe(pnl_list),
            "expectancy":      compute_expectancy(pnl_list),
            "max_drawdown":    dd["abs"],
            "max_drawdown_pct":dd["pct"],
        }

    is_end_balance = initial_balance + sum(t["pnl"] for t in is_trades)
    is_metrics     = metrics_for(is_trades,  initial_balance)
    oos_metrics    = metrics_for(oos_trades, is_end_balance)

    # Consistency score: how close is OOS to IS performance?
    is_wr  = is_metrics.get("win_rate",  0)
    oos_wr = oos_metrics.get("win_rate", 0)
    wr_gap = abs(is_wr - oos_wr)
    if wr_gap < 5:
        consistency = "✅ CONSISTENT — OOS win rate within 5% of IS"
    elif wr_gap < 10:
        consistency = "⚠️  MODERATE DRIFT — OOS win rate within 10% of IS"
    else:
        consistency = "🔴 HIGH DRIFT — OOS win rate deviates >10% from IS (possible overfit)"

    return {
        "split_pct":       split_pct,
        "is_window_size":  split_idx,
        "oos_window_size": len(trades) - split_idx,
        "in_sample":       is_metrics,
        "out_of_sample":   oos_metrics,
        "consistency":     consistency,
        "win_rate_gap_pct": round(wr_gap, 1),
    }


# ================================================================
# FEATURE 1 — EQUITY CURVE (ASCII)
# ================================================================
def build_ascii_equity_curve(balances: list = None, width: int = 60,
                              height: int = 15) -> str:
    """
    Renders an ASCII equity curve chart.
    No external libraries needed — pure Python.

    Returns multi-line string ready for print().
    """
    if balances is None:
        balances = _balance_series()
    if len(balances) < 2:
        return "  [Equity curve: not enough data]\n"

    min_b = min(balances)
    max_b = max(balances)
    rng   = max_b - min_b if max_b != min_b else 1.0

    # Sample down to `width` points
    step     = max(1, len(balances) // width)
    sampled  = balances[::step]
    if sampled[-1] != balances[-1]:
        sampled.append(balances[-1])

    # Normalise to [0, height-1]
    rows = [[" "] * len(sampled) for _ in range(height)]
    for col, val in enumerate(sampled):
        row = int((val - min_b) / rng * (height - 1))
        row = max(0, min(height - 1, row))
        rows[height - 1 - row][col] = "█"

    # Add filled area below the curve
    for col, val in enumerate(sampled):
        top_row = height - 1 - int((val - min_b) / rng * (height - 1))
        for r in range(top_row + 1, height):
            rows[r][col] = "░"

    # Y-axis labels
    lines = []
    label_positions = {0: max_b, height // 2: (max_b + min_b) / 2, height - 1: min_b}
    for i, row in enumerate(rows):
        prefix = f"  ${label_positions[i]:>8,.0f} │" if i in label_positions else "            │"
        lines.append(prefix + "".join(row))

    x_axis = "            └" + "─" * len(sampled)
    lines.append(x_axis)

    start_str = f"  Start: ${balances[0]:,.2f}"
    end_str   = f"  End: ${balances[-1]:,.2f}"
    pct_chg   = (balances[-1] - balances[0]) / balances[0] * 100
    gain_str  = f"  Change: {pct_chg:+.1f}%"
    lines.append(start_str + "   " + end_str + "   " + gain_str)

    return "\n".join(lines)


# ================================================================
# FEATURE 4 — PER-REGIME BREAKDOWN
# ================================================================
def build_regime_report() -> str:
    """Returns formatted per-regime PnL breakdown string."""
    if not _regime_stats:
        return "  No regime data recorded.\n"

    lines = []
    lines.append(f"  {'Regime':<22} {'Trades':>7} {'Win%':>6} {'PnL':>10} {'Exp/trade':>10}")
    lines.append("  " + "─" * 60)

    # Sort by total PnL descending
    sorted_regimes = sorted(_regime_stats.items(),
                            key=lambda x: x[1]["pnl"], reverse=True)
    for regime, stats in sorted_regimes:
        wins   = stats["wins"]
        losses = stats["losses"]
        total  = wins + losses
        if total == 0:
            continue
        wr     = wins / total * 100
        pnl    = stats["pnl"]
        exp    = pnl / total
        bar    = "▓" * int(wr / 10)   # visual WR bar (10 chars = 100%)
        lines.append(
            f"  {regime:<22} {total:>7} {wr:>5.1f}% ${pnl:>+9,.0f} ${exp:>+9,.1f}"
        )
    return "\n".join(lines)


# ================================================================
# FEATURE 5 — SESSION BREAKDOWN
# ================================================================
def build_session_report() -> str:
    """Returns formatted per-session PnL breakdown string."""
    if not _session_stats:
        return "  No session data recorded.\n"

    lines = []
    lines.append(f"  {'Session':<12} {'Trades':>7} {'Win%':>6} {'PnL':>10} {'Exp/trade':>10}")
    lines.append("  " + "─" * 50)

    sorted_sessions = sorted(_session_stats.items(),
                             key=lambda x: x[1]["pnl"], reverse=True)
    for session, stats in sorted_sessions:
        wins  = stats["wins"]
        losses = stats["losses"]
        total = wins + losses
        if total == 0:
            continue
        wr    = wins / total * 100
        pnl   = stats["pnl"]
        exp   = pnl / total
        lines.append(
            f"  {session:<12} {total:>7} {wr:>5.1f}% ${pnl:>+9,.0f} ${exp:>+9,.1f}"
        )
    return "\n".join(lines)


# ================================================================
# MONTHLY PnL TABLE
# ================================================================
def build_monthly_report() -> str:
    """Returns a monthly PnL table (rows = months, cols = metrics)."""
    if not _monthly_stats:
        return "  No monthly data recorded.\n"

    lines = []
    lines.append(f"  {'Month':<10} {'Trades':>7} {'Win%':>6} {'PnL':>10} {'Running':>12}")
    lines.append("  " + "─" * 50)

    sorted_months = sorted(_monthly_stats.keys())
    running = _initial_balance
    for month in sorted_months:
        stats  = _monthly_stats[month]
        total  = stats["trades"]
        wins   = stats["wins"]
        losses = stats["losses"]
        pnl    = stats["pnl"]
        wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        running += pnl
        marker = " 🔴" if pnl < -200 else (" ✅" if pnl > 200 else "")
        lines.append(
            f"  {month:<10} {total:>7} {wr:>5.1f}% ${pnl:>+9,.0f} ${running:>11,.0f}{marker}"
        )
    return "\n".join(lines)


# ================================================================
# HELPERS
# ================================================================
def _parse_month(timestamp_str) -> str:
    """Extracts 'YYYY-MM' from a timestamp string or datetime object."""
    try:
        if hasattr(timestamp_str, "strftime"):
            return timestamp_str.strftime("%Y-%m")
        dt = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m")
    except Exception:
        return "????-??"


# ================================================================
# PRINT HELPERS (called from backtest_engine.py)
# ================================================================
def print_progress_line():
    """Prints a one-line progress update. Called every 500 candles."""
    if len(_trades) == 0:
        return
    balances = _balance_series()
    dd       = compute_max_drawdown(balances)
    pnl_list = _pnl_list()
    wins     = sum(1 for t in _trades if t["result"] == "WIN")
    wr       = wins / len(_trades) * 100 if _trades else 0
    print(
        f"  [PnL] Balance: ${balances[-1]:>10,.2f} | "
        f"Trades: {len(_trades)} | WR: {wr:.1f}% | "
        f"MaxDD: -${dd['abs']:.0f} ({dd['pct']:.1f}%) | "
        f"PF: {compute_profit_factor(pnl_list):.2f}"
    )


def print_full_report():
    """
    Prints the complete analytics report at backtest end.
    Called by backtest_engine._print_final_report().
    """
    balances = _balance_series()
    pnl_list = _pnl_list()
    trades   = _trades

    if not trades:
        print("\n  [PnLTracker] No trades recorded.\n")
        return

    wins   = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] in ("LOSS", "TIMEOUT"))
    total  = wins + losses
    wr     = wins / total * 100 if total > 0 else 0

    dd      = compute_max_drawdown(balances)
    sharpe  = compute_sharpe(pnl_list)
    pf      = compute_profit_factor(pnl_list)
    exp     = compute_expectancy(pnl_list)
    consec  = compute_consecutive_stats(trades)

    sep = "─" * 65

    # ── Core metrics ────────────────────────────────────────────────
    print(f"\n  {'═'*65}")
    print(f"  {'FULL PnL ANALYTICS REPORT':^65}")
    print(f"  {'═'*65}")
    print(f"  Starting balance : ${_initial_balance:>12,.2f}")
    print(f"  Final balance    : ${balances[-1]:>12,.2f}")
    net_pnl = balances[-1] - _initial_balance
    print(f"  Net PnL          : ${net_pnl:>+12,.2f}  "
          f"({net_pnl/_initial_balance*100:+.1f}%)")
    print(f"  Total trades     : {total}")
    print(f"  Win rate         : {wr:.1f}%")
    print(f"  Profit factor    : {pf:.3f}")
    print(f"  Sharpe ratio     : {sharpe:.3f}")
    print(f"  Expectancy/trade : ${exp:+.2f}")
    print(f"  Max drawdown     : -${dd['abs']:,.2f}  (-{dd['pct']:.1f}%)")
    print(f"    Peak → trough  : ${dd['peak']:,.2f} → ${dd['trough']:,.2f}")

    # ── Feature 7: Consecutive stats ──────────────────────────────
    print(f"\n  {sep}")
    print(f"  CONSECUTIVE LOSS TRACKER")
    print(f"  {sep}")
    print(f"  Max consecutive losses : {consec['max_consec_losses']:>4}  ← "
          "check live risk rules can handle this")
    print(f"  Max consecutive wins   : {consec['max_consec_wins']:>4}")
    print(f"  Current streak         : {consec['current_streak']:>+4}  "
          f"({'winning' if consec['current_streak'] > 0 else 'losing' if consec['current_streak'] < 0 else 'flat'})")
    print(f"  Avg losing streak len  : {consec['avg_loss_streak']:.1f}")
    if consec["losing_streaks"]:
        streaks = sorted(consec["losing_streaks"], reverse=True)[:5]
        print(f"  Worst streaks          : {streaks}")

    # ── Feature 4: Per-regime breakdown ──────────────────────────
    print(f"\n  {sep}")
    print(f"  PER-REGIME PnL BREAKDOWN")
    print(f"  {sep}")
    print(build_regime_report())

    # ── Feature 5: Session breakdown ─────────────────────────────
    print(f"\n  {sep}")
    print(f"  SESSION PnL BREAKDOWN")
    print(f"  {sep}")
    print(build_session_report())

    # ── Monthly PnL table ─────────────────────────────────────────
    print(f"\n  {sep}")
    print(f"  MONTHLY PnL TABLE")
    print(f"  {sep}")
    print(build_monthly_report())

    # ── Feature 1: Equity curve ───────────────────────────────────
    print(f"\n  {sep}")
    print(f"  EQUITY CURVE")
    print(f"  {sep}")
    print(build_ascii_equity_curve(balances))

    # ── Feature 6: Walk-forward validation ───────────────────────
    print(f"\n  {sep}")
    print(f"  WALK-FORWARD VALIDATION  (70% in-sample / 30% out-of-sample)")
    print(f"  {sep}")
    wf = run_walk_forward(split_pct=0.70, trades=trades,
                          initial_balance=_initial_balance)
    if "error" in wf:
        print(f"  {wf['error']}")
    else:
        _is  = wf["in_sample"]
        _oos = wf["out_of_sample"]
        print(f"  {'Metric':<22} {'In-Sample':>14} {'Out-of-Sample':>14}")
        print(f"  {'─'*52}")
        for key, label in [
            ("trades",          "Trades"),
            ("win_rate",        "Win Rate (%)"),
            ("profit_factor",   "Profit Factor"),
            ("sharpe",          "Sharpe Ratio"),
            ("total_pnl",       "Total PnL ($)"),
            ("max_drawdown",    "Max Drawdown ($)"),
            ("max_drawdown_pct","Max Drawdown (%)"),
        ]:
            iv  = _is.get(key, "N/A")
            ov  = _oos.get(key, "N/A")
            fmt = f"  {label:<22} {iv:>14} {ov:>14}"
            print(fmt)
        print(f"\n  {wf['consistency']}")
        print(f"  Win-rate gap: {wf['win_rate_gap_pct']}%")

    # ── Feature 8: Monte Carlo ────────────────────────────────────
    print(f"\n  {sep}")
    print(f"  MONTE CARLO SIMULATION  (1,000 random shuffles)")
    print(f"  {sep}")
    mc = run_monte_carlo(n_sims=1000, trades=trades,
                         initial_balance=_initial_balance)
    if "error" in mc:
        print(f"  {mc['error']}")
    else:
        print(f"  Actual final balance   : ${mc['actual_final_balance']:>10,.2f}")
        print(f"  Median sim balance     : ${mc['median_final_balance']:>10,.2f}")
        print(f"  5th  percentile balance: ${mc['final_balance_5pct']:>10,.2f}  ← "
              "worst-case tail")
        print(f"  95th percentile balance: ${mc['final_balance_95pct']:>10,.2f}  ← "
              "best-case tail")
        print(f"  % sims profitable      : {mc['pct_profitable']:>9.1f}%")
        print(f"  Actual max drawdown    : ${mc['actual_max_drawdown']:>10,.2f}")
        print(f"  Median sim max drawdown: ${mc['worst_drawdown_median']:>10,.2f}")
        print(f"  95th-pctile max DD     : ${mc['worst_drawdown_95pct']:>10,.2f}  ← "
              "stress test")

        # Interpret result
        pct_p = mc["pct_profitable"]
        if pct_p >= 80:
            mc_verdict = "✅ ROBUST — >80% of random orderings are profitable"
        elif pct_p >= 60:
            mc_verdict = "⚠️  MODERATE — 60-80% profitable; edge exists but sequencing matters"
        else:
            mc_verdict = "🔴 FRAGILE — <60% profitable; result may be lucky ordering"
        print(f"\n  Monte Carlo verdict: {mc_verdict}")

    print(f"\n  {'═'*65}\n")


def print_equity_curve_summary(buckets: int = 10):
    """
    Prints a bucketed equity curve summary (called from backtest_engine).
    Each bucket = 1/N of total trades.
    """
    if len(_trades) < buckets:
        return

    bucket_size = len(_trades) // buckets
    print(f"\n  EQUITY CURVE — {buckets}-bucket summary:")
    print(f"  {'Bucket':<10} {'Trades':>7} {'PnL':>10} {'Balance':>12}")
    print(f"  {'─'*44}")

    running = _initial_balance
    for i in range(buckets):
        start = i * bucket_size
        end   = start + bucket_size if i < buckets - 1 else len(_trades)
        bucket_pnl = sum(t["pnl"] for t in _trades[start:end])
        running   += bucket_pnl
        label = f"{start+1}–{end}"
        marker = " 🔴" if bucket_pnl < -_initial_balance * 0.02 else (
            " ✅" if bucket_pnl > _initial_balance * 0.02 else "")
        print(f"  {label:<10} {end-start:>7} ${bucket_pnl:>+9,.0f} ${running:>11,.0f}{marker}")


# ================================================================
# SAVE / LOAD
# ================================================================
def save(path: str = None):
    """Saves full analytics to JSON for external tooling (Excel, Python plots)."""
    if path is None:
        path = PNL_STATS_PATH

    balances = _balance_series()
    pnl_list = _pnl_list()
    dd       = compute_max_drawdown(balances)
    consec   = compute_consecutive_stats()
    mc       = run_monte_carlo(n_sims=500)   # faster for save
    wf       = run_walk_forward()

    data = {
        "generated_at":     datetime.now().isoformat(),
        "initial_balance":  _initial_balance,
        "final_balance":    round(balances[-1], 2) if balances else 0,
        "total_pnl":        round(sum(pnl_list), 2),
        "total_trades":     len(_trades),
        "win_rate":         round(sum(1 for t in _trades if t["result"] == "WIN")
                                  / len(_trades) * 100, 2) if _trades else 0,
        "profit_factor":    compute_profit_factor(pnl_list),
        "sharpe":           compute_sharpe(pnl_list),
        "expectancy":       compute_expectancy(pnl_list),
        "max_drawdown":     dd,
        "consecutive_stats":consec,
        "regime_breakdown": {k: v for k, v in _regime_stats.items()},
        "session_breakdown":{k: v for k, v in _session_stats.items()},
        "monthly_breakdown":{k: v for k, v in _monthly_stats.items()},
        "monte_carlo":      {k: v for k, v in mc.items()
                             if k not in ("final_balances", "max_drawdowns")},
        "walk_forward":     wf,
        "equity_curve":     [(str(ts), bal) for ts, bal in _equity_curve],
        "trades":           _trades,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  [PnLTracker] Analytics saved → {path}")
    print(f"  Load in Excel: Data > Get Data > From JSON")
    print(f"  Plot equity curve: python -c \""
          f"import json,matplotlib.pyplot as plt; "
          f"d=json.load(open('{path}')); "
          f"plt.plot([b for _,b in d['equity_curve']]); plt.show()\"")
