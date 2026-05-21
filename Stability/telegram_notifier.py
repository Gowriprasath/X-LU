import os
import threading
import time
import requests
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_ENABLED = bool(_BOT_TOKEN.strip() and _CHAT_ID.strip())
_API_URL = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
_NY_TZ = pytz.timezone("America/New_York")

_last_send_time: float = 0.0
_rate_lock: threading.Lock = threading.Lock()
_MIN_INTERVAL: float = 2.0

if _ENABLED:
    print("[Telegram] ✓ Notifier ready.")
else:
    print(
        "[Telegram] ⚠️  Not configured — set TELEGRAM_BOT_TOKEN and "
        "TELEGRAM_CHAT_ID in .env to enable notifications."
    )


def _now_ny() -> str:
    return datetime.now(_NY_TZ).strftime("%H:%M NY")


def _send(message: str) -> None:
    global _last_send_time
    if not _ENABLED:
        return

    with _rate_lock:
        now = time.time()
        gap = now - _last_send_time
        if gap < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - gap)
        _last_send_time = time.time()

    try:
        response = requests.post(
            _API_URL,
            json={"chat_id": _CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        print(f"[Telegram] Send failed: {exc}")
        return

    if response.status_code != 200:
        print(f"[Telegram] API error {response.status_code}: {response.text[:100]}")


def _send_async(message: str) -> None:
    t = threading.Thread(target=_send, args=(message,), daemon=True)
    t.start()


def send_outage_alert(reason: str) -> None:
    _send_async(
        "🔴 <b>AI OUTAGE — PROTECTION MODE</b>\n\n"
        "<code>Status  :</code> Claude unreachable\n"
        f"<code>Reason  :</code> {reason}\n"
        "<code>Mode    :</code> No new entries. Managing open trades only.\n"
        "<code>Action  :</code> Check API keys or Anthropic status page.\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_recovery_alert() -> None:
    _send_async(
        "🟢 <b>AI RECOVERED — NORMAL MODE</b>\n\n"
        "<code>Status  :</code> Claude reachable\n"
        "<code>Mode    :</code> Full operation resumed.\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_trade_opened(trade: dict, regime: str, meta_score: float) -> None:
    _send_async(
        "🟡 <b>TRADE OPENED</b>\n\n"
        f"<code>Direction   :</code> {trade.get('direction', 'N/A')}\n"
        f"<code>Entry       :</code> {trade.get('entry', 'N/A')}\n"
        f"<code>SL          :</code> {trade.get('sl', 'N/A')}\n"
        f"<code>TP          :</code> {trade.get('tp', 'N/A')}\n"
        f"<code>Lot         :</code> {trade.get('lot', 'N/A')}\n"
        f"<code>Regime      :</code> {regime} ({trade.get('regime_confidence', 'N/A')}%)\n"
        f"<code>Meta Score  :</code> {meta_score}\n"
        f"<code>Session     :</code> {trade.get('session', 'N/A')}\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_trade_closed(trade: dict, pnl_dollars: float, pnl_pips: float) -> None:
    emoji = "🟢" if pnl_dollars >= 0 else "🔴"
    label = "WIN" if pnl_dollars >= 0 else "LOSS"
    pnl_sign = "+" if pnl_dollars >= 0 else ""
    _send_async(
        f"{emoji} <b>TRADE CLOSED — {label}</b>\n\n"
        f"<code>Direction   :</code> {trade.get('direction', 'N/A')}\n"
        f"<code>Entry       :</code> {trade.get('entry', 'N/A')}\n"
        f"<code>Exit        :</code> {trade.get('exit_price', 'N/A')}\n"
        f"<code>Lot         :</code> {trade.get('lot', 'N/A')}\n"
        f"<code>P&L         :</code> {pnl_sign}{pnl_dollars:.2f} USD\n"
        f"<code>Pips        :</code> {pnl_sign}{pnl_pips:.1f}\n"
        f"<code>Session     :</code> {trade.get('session', 'N/A')}\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_pnl_update(trade: dict, current_pnl: float) -> None:
    if abs(current_pnl) <= 0:
        return
    pnl_sign = "+" if current_pnl >= 0 else ""
    _send_async(
        "📊 <b>PnL UPDATE</b>\n\n"
        f"<code>Ticket      :</code> {trade.get('ticket', 'N/A')}\n"
        f"<code>Direction   :</code> {trade.get('direction', 'N/A')}\n"
        f"<code>Floating    :</code> {pnl_sign}{current_pnl:.2f} USD\n"
        f"<code>Session     :</code> {trade.get('session', 'N/A')}\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_sl_update(ticket: int, new_sl: float, reason: str) -> None:
    _send_async(
        "⚡ <b>SL UPDATED</b>\n\n"
        f"<code>Ticket      :</code> {ticket}\n"
        f"<code>New SL      :</code> {new_sl}\n"
        f"<code>Reason      :</code> {reason}\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_partial_close(ticket: int, lots_closed: float, remaining: float) -> None:
    _send_async(
        "⚡ <b>PARTIAL CLOSE</b>\n\n"
        f"<code>Ticket      :</code> {ticket}\n"
        f"<code>Closed      :</code> {lots_closed} lots\n"
        f"<code>Remaining   :</code> {remaining} lots\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_session_summary(summary: dict) -> None:
    net_pnl = summary.get("net_pnl", 0)
    pnl_sign = "+" if net_pnl >= 0 else ""
    _send_async(
        f"📋 <b>SESSION SUMMARY — {summary.get('session', 0)}</b>\n\n"
        f"<code>Trades      :</code> {summary.get('trades_taken', 0)}\n"
        f"<code>Win / Loss  :</code> {summary.get('wins', 0)}W / {summary.get('losses', 0)}L\n"
        f"<code>Win Rate    :</code> {summary.get('win_rate', 0):.0f}%\n"
        f"<code>Net P&L     :</code> {pnl_sign}{net_pnl:.2f} USD\n"
        f"<code>Regime      :</code> {summary.get('regime', 0)}\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_wisdom_updated(beliefs_added: int, beliefs_removed: int, key_insight: str) -> None:
    _send_async(
        "🧠 <b>WISDOM UPDATED</b>\n\n"
        f"<code>Added       :</code> {beliefs_added} beliefs\n"
        f"<code>Removed     :</code> {beliefs_removed} beliefs\n"
        f"<code>Insight     :</code> {key_insight}\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_postmortem(summary: dict) -> None:
    net_pnl = summary.get("net_pnl", 0)
    pnl_sign = "+" if net_pnl >= 0 else ""
    _send_async(
        f"📋 <b>POST-MORTEM — {summary.get('date', 'N/A')}</b>\n\n"
        f"<code>Trades      :</code> {summary.get('trades_taken', 0)}\n"
        f"<code>Win / Loss  :</code> {summary.get('wins', 0)}W / {summary.get('losses', 0)}L\n"
        f"<code>Net P&L     :</code> {pnl_sign}{net_pnl:.2f} USD\n"
        f"<code>Best Gate   :</code> {summary.get('best_gate', 'N/A')}\n"
        f"<code>Top Blocker :</code> {summary.get('worst_blocker', 'N/A')}\n"
        f"<code>Regime Acc  :</code> {summary.get('regime_accuracy', 0)}%\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_retrain_started(trigger_reason: str, data_size: int) -> None:
    _send_async(
        "⚙️ <b>MODEL RETRAINING STARTED</b>\n\n"
        f"<code>Trigger     :</code> {trigger_reason}\n"
        f"<code>Data Size   :</code> {data_size} candles\n"
        "<code>Bot Mode    :</code> Protection only during retrain\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_retrain_completed(old_accuracy: float, new_accuracy: float, duration_seconds: float) -> None:
    duration_mins = duration_seconds / 60
    _send_async(
        "✅ <b>MODEL RETRAINED</b>\n\n"
        f"<code>Old Accuracy:</code> {old_accuracy:.1f}%\n"
        f"<code>New Accuracy:</code> {new_accuracy:.1f}%\n"
        f"<code>Duration    :</code> {duration_mins:.1f} min\n"
        "<code>Status      :</code> New model loaded and active\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_startup_alert(symbol: str, model: str, regime_mode: str) -> None:
    _send_async(
        "🚀 <b>BOT STARTED</b>\n\n"
        f"<code>Symbol      :</code> {symbol}\n"
        f"<code>Model       :</code> {model}\n"
        f"<code>Regime      :</code> {regime_mode}\n"
        "<code>Status      :</code> All systems validated. Live.\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_halt_alert(reason: str) -> None:
    _send_async(
        "🛑 <b>BOT HALTED</b>\n\n"
        f"<code>Reason      :</code> {reason}\n"
        "<code>Action      :</code> Manual review required.\n\n"
        f"<i>{_now_ny()}</i>"
    )


def send_mt5_alert(status: str, detail: str) -> None:
    emoji_map = {
        "STALE": "⚠️",
        "RECONNECTING": "🔄",
        "RECOVERED": "🟢",
        "FAILED": "🔴",
    }
    emoji = emoji_map.get(status, "⚠️")
    _send_async(
        f"{emoji} <b>MT5 CONNECTION — {status}</b>\n\n"
        f"<code>Detail      :</code> {detail}\n\n"
        f"<i>{_now_ny()}</i>"
    )


if __name__ == "__main__":
    sent_messages = []

    import telegram_notifier as tn

    tn._send = lambda msg: sent_messages.append(msg) if _ENABLED else None
    _send = tn._send

    original_enabled = tn._ENABLED
    tn._ENABLED = True
    _ENABLED = True

    send_outage_alert("Timeout after 30s")
    time.sleep(0.1)
    assert len(sent_messages) == 1
    assert "OUTAGE" in sent_messages[0]
    assert "Timeout after 30s" in sent_messages[0]
    print("Test 1 PASSED")

    trade = {
        "direction": "BUY", "entry": 2345.60,
        "sl": 2338.00, "tp": 2358.00, "lot": 0.02,
        "regime_confidence": 89, "session": "NY"
    }
    send_trade_opened(trade, "BULL_TREND", 0.74)
    time.sleep(0.1)
    assert "TRADE OPENED" in sent_messages[-1]
    assert "BULL_TREND" in sent_messages[-1]
    assert "0.74" in sent_messages[-1]
    print("Test 2 PASSED")

    trade = {
        "direction": "BUY", "entry": 2345.60,
        "exit_price": 2358.00, "lot": 0.02,
        "session": "NY"
    }
    send_trade_closed(trade, 24.35, 12.4)
    time.sleep(0.1)
    assert "WIN" in sent_messages[-1]
    assert "+24.35" in sent_messages[-1]
    print("Test 3 PASSED")

    send_trade_closed(trade, -9.10, -4.6)
    time.sleep(0.1)
    assert "LOSS" in sent_messages[-1]
    assert "-9.10" in sent_messages[-1]
    print("Test 4 PASSED")

    tn._ENABLED = False
    _ENABLED = False
    before_count = len(sent_messages)
    send_outage_alert("test")
    time.sleep(0.1)
    assert len(sent_messages) == before_count
    tn._ENABLED = original_enabled
    _ENABLED = original_enabled
    print("Test 5 PASSED — disabled mode blocks send")

    send_recovery_alert()
    send_pnl_update({"ticket": 123456, "direction": "BUY", "session": "NY"}, 12.3)
    send_sl_update(123456, 2349.25, "Break-even lock")
    send_partial_close(123456, 0.01, 0.01)
    send_session_summary({
        "session": "NY", "trades_taken": 7, "wins": 4, "losses": 3,
        "net_pnl": 36.42, "win_rate": 57.1, "regime": "BULL_TREND"
    })
    send_wisdom_updated(5, 2, "Avoid post-news spikes.")
    send_postmortem({
        "date": "2026-05-14", "trades_taken": 9, "wins": 5, "losses": 4,
        "net_pnl": 28.75, "best_gate": "Trend gate",
        "worst_blocker": "Spread spikes", "regime_accuracy": 81
    })
    send_retrain_started("Weekly schedule", 4200)
    send_retrain_completed(71.2, 74.9, 615)
    send_startup_alert("XAUUSD", "claude-sonnet-4", "AUTO")
    send_halt_alert("Critical JSON corrupted")
    send_mt5_alert("STALE", "No ticks for 120s")
    send_mt5_alert("RECONNECTING", "Retry 2/5")
    send_mt5_alert("RECOVERED", "Connection restored")
    send_mt5_alert("FAILED", "Auth rejected by broker")
    time.sleep(0.2)
    print("Test 6 PASSED — all functions callable")

    print("All Telegram tests passed.")
