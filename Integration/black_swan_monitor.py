"""
black_swan_monitor.py — Black Swan / Panic Event Guard
=======================================================
Monitors for extreme market conditions that warrant either a full halt
(immediate liquidation + no new trades) or high-precaution mode
(halve position sizes, extra confirmation required).

Called every cycle from main_bot.py Step 1 before any other logic.

is_market_in_panic() returns:
    (halt: bool, reason: str, precautions: bool)

    halt        = True  → close all positions immediately, stop bot
    precautions = True  → continue but halve lot sizes
    Both False          → normal trading conditions

Detection signals (checked in priority order):
    1. HALT: Extreme spread — broker spread > BLACK_SWAN_SPREAD_HALT_DOLLARS
       Indicates thin liquidity, potential flash crash or circuit breaker.
    2. HALT: Price gap > BLACK_SWAN_GAP_HALT_PCT from previous close
       Weekend/holiday gap or major geopolitical shock.
    3. PRECAUTION: Elevated spread > BLACK_SWAN_SPREAD_CAUTION_DOLLARS
       Pre-event liquidity thinning — reduce size, don't halt.
    4. PRECAUTION: Large but sub-halt gap > BLACK_SWAN_GAP_CAUTION_PCT
       Significant overnight move — trade with caution.

All thresholds are configurable in master_controls.py.
If MT5 is unavailable, returns (False, "MT5 unavailable", False) — safe default.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

SYMBOL = "XAUUSD"

# ── Thresholds ────────────────────────────────────────────────────
# Attempt to load from master_controls; fall back to safe defaults.
try:
    from master_controls import (
        BLACK_SWAN_SPREAD_HALT_DOLLARS,
        BLACK_SWAN_SPREAD_CAUTION_DOLLARS,
        BLACK_SWAN_GAP_HALT_PCT,
        BLACK_SWAN_GAP_CAUTION_PCT,
    )
except ImportError:
    BLACK_SWAN_SPREAD_HALT_DOLLARS    = 8.0    # halt if spread > $8 (extreme illiquidity)
    BLACK_SWAN_SPREAD_CAUTION_DOLLARS = 3.5    # caution if spread > $3.50
    BLACK_SWAN_GAP_HALT_PCT           = 0.015  # halt if gap > 1.5% (e.g., $30 on $2000 gold)
    BLACK_SWAN_GAP_CAUTION_PCT        = 0.005  # caution if gap > 0.5% ($10 on $2000 gold)


def is_market_in_panic() -> tuple:
    """
    Main entry point — called every cycle from main_bot.py.

    Returns (halt: bool, reason: str, precautions: bool).
    """
    if not _MT5_AVAILABLE:
        return (False, "MT5 library not available — black swan monitor passive.", False)

    try:
        if not mt5.initialize():
            return (False, "MT5 unavailable — black swan monitor skipped.", False)

        try:
            tick        = mt5.symbol_info_tick(SYMBOL)
            symbol_info = mt5.symbol_info(SYMBOL)
        finally:
            mt5.shutdown()

        if tick is None or symbol_info is None:
            return (False, f"Could not read {SYMBOL} tick data.", False)

        # ── Compute spread in dollars ─────────────────────────────
        spread_pts     = tick.ask - tick.bid
        contract_size  = getattr(symbol_info, 'trade_contract_size', 100)  # oz per lot
        # Spread in USD per point (XAU: 1 point = $1 per lot of 100oz, so per oz = $0.01/pt)
        # We measure spread in dollars of potential slippage on a 1-lot trade
        spread_dollars = spread_pts  # for XAUUSD, spread in price = spread in $/oz

        # ── 1. HALT — extreme spread ──────────────────────────────
        if spread_dollars > BLACK_SWAN_SPREAD_HALT_DOLLARS:
            return (True,
                    f"EXTREME SPREAD: ${spread_dollars:.2f} > "
                    f"${BLACK_SWAN_SPREAD_HALT_DOLLARS:.2f} halt threshold. "
                    f"Market illiquid — flash crash or circuit breaker in progress.",
                    False)

        # ── 2. HALT — large price gap ─────────────────────────────
        mid_price = (tick.bid + tick.ask) / 2
        if mid_price > 0:
            # Use the last completed daily candle to estimate previous close
            try:
                if mt5.initialize():
                    try:
                        d1_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 3)
                    finally:
                        mt5.shutdown()
                    if d1_rates is not None and len(d1_rates) >= 2:
                        prev_close = float(d1_rates[-2]['close'])
                        gap_pct    = abs(mid_price - prev_close) / prev_close
                        if gap_pct > BLACK_SWAN_GAP_HALT_PCT:
                            return (True,
                                    f"EXTREME GAP: {gap_pct:.1%} from previous close "
                                    f"({prev_close:.2f} → {mid_price:.2f}). "
                                    f"Halt threshold: {BLACK_SWAN_GAP_HALT_PCT:.1%}. "
                                    f"Major geopolitical/economic shock suspected.",
                                    False)
                        if gap_pct > BLACK_SWAN_GAP_CAUTION_PCT:
                            return (False,
                                    f"ELEVATED GAP: {gap_pct:.1%} from previous close. "
                                    f"Significant overnight move — high precautions active.",
                                    True)
            except Exception:
                pass  # Gap check is non-fatal — spread check already done above

        # ── 3. CAUTION — elevated spread ─────────────────────────
        if spread_dollars > BLACK_SWAN_SPREAD_CAUTION_DOLLARS:
            return (False,
                    f"ELEVATED SPREAD: ${spread_dollars:.2f} > "
                    f"${BLACK_SWAN_SPREAD_CAUTION_DOLLARS:.2f} caution threshold. "
                    f"Thin liquidity — high precautions active. Lot halved.",
                    True)

        # ── All clear ─────────────────────────────────────────────
        return (False, "", False)

    except Exception as e:
        # Never let the monitor crash the bot — safe default is to allow trading
        print(f"[BlackSwanMonitor] Error (non-fatal): {e}")
        return (False, f"Monitor error: {e}", False)
