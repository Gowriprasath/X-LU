import MetaTrader5 as mt5
import os
import sys
from datetime import datetime

_root_te = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _root_te not in sys.path: sys.path.insert(0, _root_te)
from paths import EXECUTION_LOG_PATH, create_all_dirs as _cad_te
_cad_te()

MAGIC_NUMBER = 202603  # FIX: Single source of truth for magic number


def _log_mt5_error(symbol, action, error_msg):
    log_file = EXECUTION_LOG_PATH
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {symbol} {action} REJECTED: {error_msg}\n")
    except Exception:
        pass


def execute_trade(symbol, action, lot, sl, tp, order_type="MARKET", entry_price=0.0, comment="Antigravity_AI"):
    if not mt5.initialize():
        print("Failed to initialize MT5 for execution.")
        return None

    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)

    if tick is None or symbol_info is None:
        print(f"Could not get info for {symbol}.")
        mt5.shutdown()
        return None

    digits = symbol_info.digits
    sl_normalized = round(float(sl), digits)
    tp_normalized = round(float(tp), digits)

    if order_type == "MARKET":
        mt5_action = mt5.TRADE_ACTION_DEAL
        mt5_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action == "BUY" else tick.bid
    elif order_type == "LIMIT":
        mt5_action = mt5.TRADE_ACTION_PENDING
        mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if action == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
        price = round(float(entry_price), digits)
    elif order_type == "STOP":
        mt5_action = mt5.TRADE_ACTION_PENDING
        mt5_type = mt5.ORDER_TYPE_BUY_STOP if action == "BUY" else mt5.ORDER_TYPE_SELL_STOP
        price = round(float(entry_price), digits)
    else:
        print(f"Unknown order type: {order_type}")
        mt5.shutdown()
        return None

    request = {
        "action": mt5_action,
        "symbol": symbol,
        "volume": float(lot),
        "type": mt5_type,
        "price": price,
        "sl": sl_normalized,
        "tp": tp_normalized,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    if order_type == "MARKET":
        request["type_filling"] = mt5.ORDER_FILLING_IOC

    # FIX #1 — RETRY LOOP
    # mt5.order_send() fails silently on requote (10004), price changed (10009),
    # and connection loss (10018) — all very common during XAUUSD NY open and
    # high-volume sweeps. Without retry the trade is completely dropped.
    # Strategy:
    #   - Up to 5 attempts
    #   - Retriable codes: requote (10004), price changed (10009), no connection (10018),
    #     trade context busy (10026), too many requests (10016)
    #   - On each retry: fresh tick price + widen deviation by 5 pts to absorb slippage
    #   - Hard-fail immediately on fatal codes (off quotes, market closed, etc.)
    RETRIABLE = {10004, 10009, 10018, 10016, 10026}
    MAX_RETRIES = 5
    result = None

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            import time as _t
            _t.sleep(0.5)
            # Refresh price from a fresh tick
            fresh_tick = mt5.symbol_info_tick(symbol)
            if fresh_tick and order_type == "MARKET":
                request["price"] = fresh_tick.ask if action == "BUY" else fresh_tick.bid
            # Widen deviation by 5 pts per retry to absorb slippage
            request["deviation"] = 20 + (attempt - 1) * 5

        print(f"Sending {action} {order_type} order for {symbol} at "
              f"{request['price']} (Lot: {lot}, attempt {attempt}/{MAX_RETRIES})...")
        result = mt5.order_send(request)

        if result is None:
            err_code = mt5.last_error()[0] if mt5.last_error() else 0
            print(f"  [Retry] mt5.order_send returned None (attempt {attempt}). "
                  f"MT5 error: {mt5.last_error()}")
            if err_code not in RETRIABLE and attempt > 1:
                break  # non-retriable error
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            break  # success

        print(f"  [Retry] retcode {result.retcode}: {result.comment} (attempt {attempt})")
        if result.retcode not in RETRIABLE:
            break  # fatal error — don't retry

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        error_msg = mt5.last_error() if result is None else result.comment
        print(f"Trade FAILED after {attempt} attempt(s)! Error: {error_msg}")
        _log_mt5_error(symbol, f"{action}_{order_type}", f"[{attempt} attempts] {error_msg}")
        mt5.shutdown()
        return None

    print(f"Trade {result.order} opened successfully (attempt {attempt}).")
    mt5.shutdown()
    return result.order


def close_all_positions(symbol):
    """
    BUG-5 FIX: mt5.shutdown() now guaranteed via try/finally.
    Previously a crash in order_send() left the MT5 handle open,
    causing subsequent initialize() calls to fail silently.
    """
    if not mt5.initialize():
        return False

    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                ticket, volume, pos_type = pos.ticket, pos.volume, pos.type
                close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                request = {
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume),
                    "type": close_type, "position": ticket, "price": price, "deviation": 20,
                    "magic": MAGIC_NUMBER, "comment": "AI_REVERSAL_EXIT",
                    "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
                }
                import time as _t
                _RETRIABLE = {10004, 10009, 10018, 10016, 10026}
                res = None
                for _att in range(1, 4):
                    if _att > 1:
                        _t.sleep(0.5)
                        _tk = mt5.symbol_info_tick(symbol)
                        if _tk:
                            request["price"] = _tk.bid if close_type == mt5.ORDER_TYPE_SELL else _tk.ask
                        request["deviation"] = 20 + (_att - 1) * 5
                    res = mt5.order_send(request)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        break
                    if res and res.retcode not in _RETRIABLE:
                        break
                if not res or res.retcode != mt5.TRADE_RETCODE_DONE:
                    _log_mt5_error(symbol, "CLOSE_POS", res.comment if res else mt5.last_error())

        orders = mt5.orders_get(symbol=symbol)
        if orders:
            for ord in orders:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ord.ticket})

        return True
    finally:
        mt5.shutdown()  # BUG-5 FIX: always called even if order_send raises


def close_partial_position(symbol, close_pct=0.5):
    """
    BUG-5 FIX: mt5.shutdown() now guaranteed via try/finally.
    BUG-PC-01 FIX: Added retry loop for retriable MT5 error codes.
    close_partial_position() is called primarily during high-impact news events
    (check_open_position_news_rule) — exactly when spreads widen and requotes /
    price-changed errors (10004, 10009) are most common. Without a retry the
    partial close would silently fail, leaving full exposure through the news spike.
    The retry pattern mirrors close_all_positions() (3 attempts, 0.5s gap,
    fresh tick + widened deviation on each retry).
    """
    if not mt5.initialize():
        return False

    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return True

        import time as _t
        _RETRIABLE = {10004, 10009, 10018, 10016, 10026}

        for pos in positions:
            ticket, volume, pos_type = pos.ticket, pos.volume, pos.type
            symbol_info = mt5.symbol_info(symbol)
            step = float(symbol_info.volume_step) if symbol_info else 0.01
            partial_vol = round((volume * close_pct) / step) * step
            partial_vol = max(float(symbol_info.volume_min if symbol_info else 0.01), partial_vol)

            close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(partial_vol),
                "type": close_type, "position": ticket, "price": price, "deviation": 20,
                "magic": MAGIC_NUMBER, "comment": "AI_NEWS_PARTIAL",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = None
            for _att in range(1, 4):
                if _att > 1:
                    _t.sleep(0.5)
                    _tk = mt5.symbol_info_tick(symbol)
                    if _tk:
                        request["price"] = _tk.bid if close_type == mt5.ORDER_TYPE_SELL else _tk.ask
                    request["deviation"] = 20 + (_att - 1) * 5
                res = mt5.order_send(request)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    break
                if res and res.retcode not in _RETRIABLE:
                    break
            if not res or res.retcode != mt5.TRADE_RETCODE_DONE:
                _log_mt5_error(symbol, "PARTIAL_CLOSE", res.comment if res else mt5.last_error())

        return True
    finally:
        mt5.shutdown()  # BUG-5 FIX: always called even if order_send raises


def get_open_positions(symbol=None):
    """
    BUG-9 FIX: Returns ONLY filled active positions (mt5.positions_get).
    Previously also included pending orders (mt5.orders_get), which:
      - Have different ticket numbers than filled positions
      - Lack price_current and profit attributes → AttributeError
      - Caused the 'position already open' check to incorrectly pass
    Use get_pending_orders() separately if pending order info is needed.
    """
    if not mt5.initialize():
        return []
    try:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return list(positions) if positions else []
    finally:
        mt5.shutdown()


def get_pending_orders(symbol=None):
    """Returns pending (unfilled) orders. Separate from get_open_positions()."""
    if not mt5.initialize():
        return []
    try:
        orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        return list(orders) if orders else []
    finally:
        mt5.shutdown()


def close_position(ticket, symbol, comment="AI Full Exit"):
    """Closes an entire position by ticket."""
    # B-05 FIX: wrapped in try/finally so mt5.shutdown() is always called,
    # even if positions_get, symbol_info_tick, or order_send raises.
    if not mt5.initialize():
        return False

    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        order_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result
    finally:
        mt5.shutdown()


def partial_close(ticket, symbol, volume_to_close):
    """Closes part of a position to bank profit."""
    # B-05 FIX: wrapped in try/finally so mt5.shutdown() is always called.
    if not mt5.initialize():
        return False

    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        order_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": symbol,
            "volume": volume_to_close,
            "type": order_type,
            "price": price,
            "magic": MAGIC_NUMBER,
            "comment": "Partial Harvest",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result
    finally:
        mt5.shutdown()


def modify_position_sl(ticket, symbol, new_sl):
    """
    Modifies the Stop Loss of an open position.
    Used by the trade manager for break-even and trailing stop logic.
    Returns True on success, False on failure.

    BUG-02 FIX: Wrapped entire body in try/finally so mt5.shutdown() is
    guaranteed even if order_send() raises (e.g. broker disconnect mid-call).
    Without this, a raised exception left the handle open and caused all
    subsequent mt5.initialize() calls to fail silently for the rest of the cycle.
    """
    if not mt5.initialize():
        return False

    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits if symbol_info else 2

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   symbol,
            "sl":       round(float(new_sl), digits),
            "tp":       round(float(pos.tp), digits),
            "magic":    MAGIC_NUMBER,
        }
        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True

        error = result.comment if result else mt5.last_error()
        _log_mt5_error(symbol, "MODIFY_SL", error)
        return False

    finally:
        mt5.shutdown()  # BUG-02 FIX: guaranteed regardless of exception
