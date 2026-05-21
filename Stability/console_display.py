from colorama import init, Fore, Back, Style
import datetime
import pytz
import threading

init(autoreset=True)

_NY_TZ = pytz.timezone("America/New_York")
_print_lock = threading.Lock()


def _now_ny() -> str:
    return datetime.datetime.now(_NY_TZ).strftime("%H:%M NY")


def _truncate_and_pad(line: str) -> str:
    text = str(line)
    if len(text) > 52:
        text = text[:49] + "..."
    return text.ljust(52)


def _box(lines: list[str], colour: str) -> None:
    top = "╔" + ("═" * 54) + "╗"
    bottom = "╚" + ("═" * 54) + "╝"
    with _print_lock:
        print(f"{colour}{top}")
        for line in lines:
            print(f"{colour}║  {_truncate_and_pad(line)}║")
        print(f"{colour}{bottom}")
        print()


def _plain(message: str) -> None:
    with _print_lock:
        print(f"  → {_now_ny()} | {message}")


def print_critical(message: str) -> None:
    _box([f"🔴  {message}"], Fore.RED)


def print_warning(message: str) -> None:
    _box([f"⚠️   {message}"], Fore.YELLOW)


def print_trade_open(trade: dict) -> None:
    _box(
        [
            "🟡  TRADE OPENED",
            f"Direction  : {trade.get('direction', 'N/A')}",
            f"Entry      : {trade.get('entry', 'N/A')}",
            f"SL         : {trade.get('sl', 'N/A')}",
            f"TP         : {trade.get('tp', 'N/A')}",
            f"Lot        : {trade.get('lot', 'N/A')}",
            (
                f"Regime     : {trade.get('regime', 'N/A')} "
                f"({trade.get('regime_confidence', 'N/A')}%)"
            ),
            f"Meta Score : {trade.get('meta_score', 'N/A')}",
            f"Session    : {trade.get('session', 'N/A')}",
        ],
        Fore.CYAN,
    )


def print_trade_close(trade: dict, pnl_dollars: float, pnl_pips: float) -> None:
    positive = pnl_dollars >= 0
    colour = Fore.GREEN if positive else Fore.RED
    emoji = "🟢" if positive else "🔴"
    plus = "+" if positive else ""
    _box(
        [
            f"{emoji}  TRADE CLOSED",
            f"Direction  : {trade.get('direction', 'N/A')}",
            f"Entry      : {trade.get('entry', 'N/A')}",
            f"Exit       : {trade.get('exit_price', 'N/A')}",
            f"Lot        : {trade.get('lot', 'N/A')}",
            f"P&L        : {plus}{pnl_dollars:.2f} USD",
            f"Pips       : {plus}{pnl_pips:.1f}",
            f"Session    : {trade.get('session', 'N/A')}",
        ],
        colour,
    )


def print_sl_update(ticket: int, new_sl: float, reason: str) -> None:
    _box(
        [
            "⚡  SL UPDATED",
            f"Ticket     : {ticket}",
            f"New SL     : {new_sl}",
            f"Reason     : {reason}",
        ],
        Fore.YELLOW,
    )


def print_partial_close(ticket: int, lots_closed: float, remaining: float) -> None:
    _box(
        [
            "⚡  PARTIAL CLOSE",
            f"Ticket     : {ticket}",
            f"Closed     : {lots_closed} lots",
            f"Remaining  : {remaining} lots",
        ],
        Fore.YELLOW,
    )


def print_memory_event(message: str) -> None:
    _box([f"🧠  {message}"], Fore.MAGENTA)


def print_retrain(message: str, status: str) -> None:
    mapping = {
        "STARTED": (Fore.BLUE, "⚙️"),
        "COMPLETED": (Fore.GREEN, "✅"),
        "FAILED": (Fore.RED, "❌"),
    }
    colour, emoji = mapping.get(status, (Fore.YELLOW, "⚠️"))
    _box([f"{emoji}  RETRAINING {status}", message], colour)


def print_pnl_update(trade: dict, current_pnl: float) -> None:
    plus = "+" if current_pnl >= 0 else ""
    ticket = trade.get("ticket", "N/A")
    _plain(f"📊 PnL Update | Ticket: {ticket} | Floating: {plus}{current_pnl:.2f} USD")


def print_cycle(
    regime: str, confidence: float, session: str, result: str, reason: str
) -> None:
    _plain(f"{regime} {confidence:.0f}% | {session} | {result} — {reason}")


def print_outage_banner() -> None:
    lines = [
        "🔴  AI OUTAGE — PROTECTION MODE ACTIVE",
        "No new entries. Managing open trades only.",
        "Check your API keys or Anthropic status.",
        f"Time: {_now_ny()}",
    ]
    top = "╔" + ("═" * 54) + "╗"
    bottom = "╚" + ("═" * 54) + "╝"
    with _print_lock:
        print(f"{Fore.RED}{top}")
        for line in lines:
            print(f"{Fore.RED}║  {_truncate_and_pad(line)}║")
        print(f"{Fore.RED}{bottom}")


def print_recovery_banner() -> None:
    _box(
        [
            "🟢  AI RECOVERED — NORMAL MODE RESUMED",
            "Claude is reachable. Resuming full operation.",
            f"Time: {_now_ny()}",
        ],
        Fore.GREEN,
    )


def print_boot_banner(model: str, symbol: str, keys_loaded: int, regime_mode: str) -> None:
    line = "═" * 56
    with _print_lock:
        print(f"{Fore.CYAN}{line}")
        print(f"{Fore.CYAN}  GOLD AI BRIDGE — {symbol} Trading Bot")
        print(f"{Fore.CYAN}  Model    : {model}")
        print(f"{Fore.CYAN}  Keys     : {keys_loaded} API key(s) loaded")
        print(f"{Fore.CYAN}  Regime   : {regime_mode}")
        print(f"{Fore.CYAN}  Started  : {_now_ny()}")
        print(f"{Fore.CYAN}{line}")
        print()


def print_wisdom_update(
    beliefs_added: int, beliefs_removed: int, key_insight: str
) -> None:
    _box(
        [
            "🧠  WISDOM UPDATED",
            f"Beliefs added   : {beliefs_added}",
            f"Beliefs removed : {beliefs_removed}",
            f"Insight : {key_insight}",
        ],
        Fore.MAGENTA,
    )


def print_session_summary(summary: dict) -> None:
    net_pnl = summary.get("net_pnl", 0)
    plus = "+" if net_pnl >= 0 else ""
    _box(
        [
            f"📋  SESSION SUMMARY — {summary.get('session', 0)}",
            f"Trades     : {summary.get('trades_taken', 0)}",
            f"Win / Loss : {summary.get('wins', 0)}W / {summary.get('losses', 0)}L",
            f"Win Rate   : {summary.get('win_rate', 0):.0f}%",
            f"Net P&L    : {plus}{net_pnl:.2f} USD",
            f"Regime     : {summary.get('regime', 0)}",
        ],
        Fore.CYAN,
    )


if __name__ == "__main__":
    sample_trade = {
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

    print_boot_banner(
        model="claude-sonnet-4", symbol="XAUUSD", keys_loaded=3, regime_mode="AUTO"
    )
    print_critical("Risk guard triggered on account equity.")
    print_warning("Spread is elevated. Entry filters tightened.")
    print_trade_open(sample_trade)
    print_trade_close(sample_trade, pnl_dollars=24.35, pnl_pips=12.4)
    print_trade_close(sample_trade, pnl_dollars=-9.10, pnl_pips=-4.6)
    print_sl_update(ticket=123456, new_sl=2349.25, reason="Break-even protection")
    print_partial_close(ticket=123456, lots_closed=0.01, remaining=0.01)
    print_memory_event("Merged 3 new beliefs from last losing streak.")
    print_retrain("Model retraining window opened for NY close.", "STARTED")
    print_retrain("Validation metrics stable and improved.", "COMPLETED")
    print_retrain("Data drift exceeded threshold in validation set.", "FAILED")
    print_pnl_update(sample_trade, current_pnl=17.82)
    print_cycle("RANGE", 63, "London", "WAIT", "Low confidence signal")
    print_cycle("BULL_TREND", 88, "NY", "BUY", "Breakout confirmed")
    print_cycle("VOLATILE", 51, "Asia", "BLOCKED", "Protection mode active")
    print_outage_banner()
    print_recovery_banner()
    print_wisdom_update(
        beliefs_added=5,
        beliefs_removed=2,
        key_insight="Avoid entries during post-news volatility spikes.",
    )
    print_session_summary(
        {
            "trades_taken": 7,
            "wins": 4,
            "losses": 3,
            "net_pnl": 36.42,
            "win_rate": 57.1,
            "session": "NY",
            "regime": "BULL_TREND",
        }
    )
    print("Visual test complete — verify output above.")
