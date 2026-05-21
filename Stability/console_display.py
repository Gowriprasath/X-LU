"""
Thread-safe console display gateway for the trading bot.
"""

from colorama import init, Fore, Back, Style
import datetime
import pytz
import threading


init(autoreset=True)

_NY_TZ = pytz.timezone("America/New_York")
_print_lock = threading.Lock()


def _now_ny() -> str:
    return datetime.datetime.now(_NY_TZ).strftime("%H:%M NY")


def _box(lines: list[str], colour: str) -> None:
    top = "╔" + "═" * 54 + "╗"
    bottom = "╚" + "═" * 54 + "╝"
    with _print_lock:
        print(colour + top)
        for line in lines:
            text = str(line)
            if len(text) > 52:
                text = text[:49] + "..."
            print(colour + "║  " + text.ljust(52) + "║")
        print(colour + bottom)
        print()


def _plain(message: str) -> None:
    with _print_lock:
        print(f"  → {_now_ny()} | {message}")


def print_critical(message: str) -> None:
    _box(["🔴  " + message], Fore.RED)


def print_warning(message: str) -> None:
    _box(["⚠️   " + message], Fore.YELLOW)


def print_trade_open(trade: dict) -> None:
    _box([
        "🟡  TRADE OPENED",
        f"Direction  : {trade.get('direction', 'N/A')}",
        f"Entry      : {trade.get('entry', 'N/A')}",
        f"SL         : {trade.get('sl', 'N/A')}",
        f"TP         : {trade.get('tp', 'N/A')}",
        f"Lot        : {trade.get('lot', 'N/A')}",
        f"Regime     : {trade.get('regime', 'N/A')} ({trade.get('regime_confidence', 'N/A')}%)",
        f"Meta Score : {trade.get('meta_score', 'N/A')}",
        f"Session    : {trade.get('session', 'N/A')}",
    ], Fore.CYAN)


def print_trade_close(trade: dict, pnl_dollars: float, pnl_pips: float) -> None:
    colour = Fore.GREEN if pnl_dollars >= 0 else Fore.RED
    emoji = "🟢" if pnl_dollars >= 0 else "🔴"
    dollar_sign = "+" if pnl_dollars >= 0 else ""
    pip_sign = "+" if pnl_pips >= 0 else ""
    _box([
        f"{emoji}  TRADE CLOSED",
        f"Direction  : {trade.get('direction', 'N/A')}",
        f"Entry      : {trade.get('entry', 'N/A')}",
        f"Exit       : {trade.get('exit_price', 'N/A')}",
        f"Lot        : {trade.get('lot', 'N/A')}",
        f"P&L        : {dollar_sign}{pnl_dollars:.2f} USD",
        f"Pips       : {pip_sign}{pnl_pips:.1f}",
        f"Session    : {trade.get('session', 'N/A')}",
    ], colour)


def print_sl_update(ticket: int, new_sl: float, reason: str) -> None:
    _box([
        "⚡  SL UPDATED",
        f"Ticket     : {ticket}",
        f"New SL     : {new_sl}",
        f"Reason     : {reason}",
    ], Fore.YELLOW)


def print_partial_close(ticket: int, lots_closed: float, remaining: float) -> None:
    _box([
        "⚡  PARTIAL CLOSE",
        f"Ticket     : {ticket}",
        f"Closed     : {lots_closed} lots",
        f"Remaining  : {remaining} lots",
    ], Fore.YELLOW)


def print_memory_event(message: str) -> None:
    _box(["🧠  " + message], Fore.MAGENTA)


def print_retrain(message: str, status: str) -> None:
    mapping = {
        "STARTED": (Fore.BLUE, "⚙️"),
        "COMPLETED": (Fore.GREEN, "✅"),
        "FAILED": (Fore.RED, "❌"),
    }
    colour, emoji = mapping.get(status, (Fore.YELLOW, "⚙️"))
    _box([f"{emoji}  RETRAINING {status}", message], colour)


def print_pnl_update(trade: dict, current_pnl: float) -> None:
    pnl_sign = "+" if current_pnl >= 0 else ""
    _plain(
        f"📊 PnL Update | Ticket: {trade.get('ticket', 'N/A')} | "
        f"Floating: {pnl_sign}{current_pnl:.2f} USD"
    )


def print_cycle(regime: str, confidence: float, session: str, result: str, reason: str) -> None:
    _plain(f"{regime} {confidence:.0f}% | {session} | {result} — {reason}")


def print_outage_banner() -> None:
    lines = [
        "🔴  AI OUTAGE — PROTECTION MODE ACTIVE",
        "No new entries. Managing open trades only.",
        "Check your API keys or Anthropic status.",
        f"Time: {_now_ny()}",
    ]
    _box(lines, Fore.RED)


def print_recovery_banner() -> None:
    _box([
        "🟢  AI RECOVERED — NORMAL MODE RESUMED",
        "Claude is reachable. Resuming full operation.",
        f"Time: {_now_ny()}",
    ], Fore.GREEN)


def print_boot_banner(model: str, symbol: str, keys_loaded: int, regime_mode: str) -> None:
    line = "═" * 56
    with _print_lock:
        print(Fore.CYAN + line)
        print(Fore.CYAN + f"  GOLD AI BRIDGE — {symbol} Trading Bot")
        print(Fore.CYAN + f"  Model    : {model}")
        print(Fore.CYAN + f"  Keys     : {keys_loaded} API key(s) loaded")
        print(Fore.CYAN + f"  Regime   : {regime_mode}")
        print(Fore.CYAN + f"  Started  : {_now_ny()}")
        print(Fore.CYAN + line)
        print()


def print_wisdom_update(beliefs_added: int, beliefs_removed: int, key_insight: str) -> None:
    _box([
        "🧠  WISDOM UPDATED",
        f"Beliefs added   : {beliefs_added}",
        f"Beliefs removed : {beliefs_removed}",
        f"Insight : {key_insight}",
    ], Fore.MAGENTA)


def print_session_summary(summary: dict) -> None:
    net_pnl = summary.get("net_pnl", 0)
    pnl_sign = "+" if net_pnl >= 0 else ""
    _box([
        f"📋  SESSION SUMMARY — {summary.get('session', 0)}",
        f"Trades     : {summary.get('trades_taken', 0)}",
        f"Win / Loss : {summary.get('wins', 0)}W / {summary.get('losses', 0)}L",
        f"Win Rate   : {summary.get('win_rate', 0):.0f}%",
        f"Net P&L    : {pnl_sign}{net_pnl:.2f} USD",
        f"Regime     : {summary.get('regime', 0)}",
    ], Fore.CYAN)


if __name__ == "__main__":
    trade = {
        "direction": "BUY",
        "entry": 2345.60,
        "exit_price": 2358.00,
        "sl": 2338.00,
        "tp": 2358.00,
        "lot": 0.02,
        "regime": "BULL_TREND",
        "regime_confidence": 89,
        "meta_score": 0.74,
        "session": "NY",
        "ticket": 123456,
    }
    print_critical("Critical visual test")
    print_warning("Warning visual test")
    print_trade_open(trade)
    print_trade_close(trade, 24.35, 12.4)
    print_trade_close(trade, -9.10, -4.6)
    print_sl_update(123456, 2348.0, "Break-even")
    print_partial_close(123456, 0.01, 0.01)
    print_memory_event("Memory updated")
    print_retrain("Testing retrain start", "STARTED")
    print_retrain("Testing retrain complete", "COMPLETED")
    print_retrain("Testing retrain fail", "FAILED")
    print_pnl_update(trade, 12.34)
    print_cycle("BULL_TREND", 89, "NY", "WAIT", "No clean setup")
    print_cycle("BULL_TREND", 89, "NY", "BUY", "Breakout confirmed")
    print_cycle("LOW_VOL_RANGE", 20, "London", "BLOCKED", "Confidence gate")
    print_outage_banner()
    print_recovery_banner()
    print_boot_banner("claude-sonnet-4-6", "XAUUSD", 3, "AUTO")
    print_wisdom_update(2, 1, "Avoid low-volume fakeouts")
    print_session_summary({
        "trades_taken": 4,
        "wins": 3,
        "losses": 1,
        "net_pnl": 52.4,
        "win_rate": 75,
        "session": "NY",
        "regime": "BULL_TREND",
    })
    print("Visual test complete — verify output above.")
