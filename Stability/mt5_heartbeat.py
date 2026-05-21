"""
MT5 stale-data heartbeat.
"""

import MetaTrader5 as mt5
import pytz
import time
import os
import sys
from datetime import datetime
from console_display import (
    print_warning,
    print_critical,
    print_recovery_banner,
)
from telegram_notifier import send_mt5_alert


_NY_TZ = pytz.timezone("America/New_York")
_STALE_THRESHOLD_SECONDS: int = 180
_RECONNECT_ATTEMPTS: int = 2
_RECONNECT_WAIT_SECONDS: float = 5.0
_SYMBOL: str = "XAUUSD"

_last_status: str = "UNKNOWN"
_last_alert_status: str = ""


def _get_latest_candle_time() -> float | None:
    try:
        result = mt5.copy_rates_from_pos(_SYMBOL, mt5.TIMEFRAME_M1, 0, 1)
        if result is None or len(result) == 0:
            return None
        return float(result[0]["time"])
    except Exception:
        return None


def _attempt_reconnect() -> bool:
    for attempt in range(_RECONNECT_ATTEMPTS):
        print(f"[MT5 Heartbeat] Reconnect attempt {attempt + 1}/{_RECONNECT_ATTEMPTS}...")
        try:
            mt5.shutdown()
        except Exception:
            pass
        time.sleep(_RECONNECT_WAIT_SECONDS)
        try:
            success = mt5.initialize()
            if success:
                print("[MT5 Heartbeat] Reconnect successful.")
                return True
            error = mt5.last_error()
            print(f"[MT5 Heartbeat] Attempt {attempt + 1} failed: {error}")
        except Exception as error:
            print(f"[MT5 Heartbeat] Attempt {attempt + 1} failed: {error}")

    print("[MT5 Heartbeat] ❌ All reconnect attempts failed.")
    return False


def check_mt5_health() -> bool:
    global _last_status, _last_alert_status
    try:
        now = time.time()
        candle_time = _get_latest_candle_time()

        if candle_time is not None:
            age_seconds = now - candle_time
            if age_seconds <= _STALE_THRESHOLD_SECONDS:
                if _last_status != "HEALTHY":
                    print_recovery_banner()
                    if _last_alert_status != "RECOVERED":
                        send_mt5_alert(
                            "RECOVERED",
                            f"Data fresh — last candle {age_seconds:.0f}s ago",
                        )
                        _last_alert_status = "RECOVERED"
                _last_status = "HEALTHY"
                return True
        else:
            age_seconds = None

        age_str = f"{age_seconds:.0f}s old" if candle_time is not None else "no data returned"
        print_warning(f"MT5 stale data detected — {age_str}. Reconnecting...")

        if _last_alert_status != "STALE":
            send_mt5_alert("STALE", f"Last candle: {age_str}")
            send_mt5_alert("RECONNECTING", "Attempting to restore connection...")
            _last_alert_status = "STALE"

        _last_status = "RECONNECTING"

        if _attempt_reconnect():
            fresh_time = _get_latest_candle_time()
            if fresh_time is not None:
                fresh_age = time.time() - fresh_time
                if fresh_age <= _STALE_THRESHOLD_SECONDS:
                    print_recovery_banner()
                    send_mt5_alert("RECOVERED", "Connection restored. Data is fresh.")
                    _last_status = "HEALTHY"
                    _last_alert_status = "RECOVERED"
                    return True

        print_critical("MT5 connection FAILED — skipping this cycle.")
        send_mt5_alert(
            "FAILED",
            "All reconnect attempts exhausted. Check MT5 terminal.",
        )
        _last_status = "FAILED"
        _last_alert_status = "FAILED"
        return False
    except Exception as error:
        print_critical(f"MT5 heartbeat error — skipping this cycle: {error}")
        return False


if __name__ == "__main__":
    import mt5_heartbeat as hb

    real_get_latest = hb._get_latest_candle_time
    real_reconnect = hb._attempt_reconnect

    hb._get_latest_candle_time = lambda: time.time() - 30
    assert hb.check_mt5_health() is True
    print("Mock Test 1 PASSED — fresh data returns True")

    calls = {"n": 0}

    def stale_then_fresh():
        calls["n"] += 1
        return time.time() - 300 if calls["n"] == 1 else time.time() - 20

    hb._get_latest_candle_time = stale_then_fresh
    hb._attempt_reconnect = lambda: True
    assert hb.check_mt5_health() is True
    print("Mock Test 2 PASSED — stale data, reconnect works")

    hb._get_latest_candle_time = lambda: time.time() - 300
    hb._attempt_reconnect = lambda: False
    assert hb.check_mt5_health() is False
    print("Mock Test 3 PASSED — reconnect failure returns False")

    hb._get_latest_candle_time = lambda: None
    hb._attempt_reconnect = lambda: False
    assert hb.check_mt5_health() is False
    print("Mock Test 4 PASSED — None response returns False")

    hb._get_latest_candle_time = real_get_latest
    hb._attempt_reconnect = real_reconnect
    try:
        live_available = mt5.initialize()
    except Exception:
        live_available = False

    if live_available:
        result = hb.check_mt5_health()
        if result:
            print("Live Test 5 PASSED — MT5 healthy")
        else:
            print("Live Test 5 INFO — MT5 not healthy (check terminal)")
        mt5.shutdown()

    print("All MT5 heartbeat tests completed.")
