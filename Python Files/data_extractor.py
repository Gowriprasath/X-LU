import MetaTrader5 as mt5
import pandas as pd
import pytz
import os
from datetime import datetime

SYMBOL = "XAUUSD" 
NY_TZ = pytz.timezone('America/New_York')

_TZ_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'Logs', 'broker_tz_cache.json'
)

def _get_broker_tz():
    """
    FIX #3 — Weekend-safe broker timezone detection.
    Problem: on weekends mt5.symbol_info_tick() returns a STALE Friday tick.
    The offset calculation (broker_ts vs utcnow) produces a garbage offset
    because the tick timestamp is hours old. The bot then boots with the wrong
    NY session boundary, potentially trading in the wrong hour on Monday open.

    Fix: cache the last known-good offset to disk after each successful
    weekday detection. On weekends (or when the offset looks suspicious),
    load from disk instead. Falls back to GMT-3 (DST / summer) if no cache.
    """
    import json as _j

    def _load_cache():
        try:
            if os.path.exists(_TZ_CACHE_PATH):
                with open(_TZ_CACHE_PATH) as _f:
                    return _j.load(_f).get("offset_hrs")
        except Exception:
            pass
        return None

    def _save_cache(offset_hrs):
        try:
            os.makedirs(os.path.dirname(_TZ_CACHE_PATH), exist_ok=True)
            with open(_TZ_CACHE_PATH, 'w') as _f:
                _j.dump({"offset_hrs": offset_hrs,
                          "updated": datetime.utcnow().isoformat()}, _f)
        except Exception:
            pass

    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick:
            broker_ts  = tick.time
            utc_now    = datetime.utcnow()
            broker_dt  = datetime.utcfromtimestamp(broker_ts)
            age_secs   = abs((utc_now - broker_dt).total_seconds())

            # If tick is < 15 minutes old → market is live → trust it
            if age_secs < 900:
                offset_hrs = round((broker_dt - utc_now).total_seconds() / 3600)
                if offset_hrs in (2, 3):        # valid broker offsets (winter=2, summer=3)
                    _save_cache(offset_hrs)     # persist for weekend boot
                    return pytz.timezone(f'Etc/GMT-{offset_hrs}')

            # Tick is stale (weekend / broker closed) — use cached value
            cached = _load_cache()
            if cached in (2, 3):
                print(f"[DataExtractor] Weekend/stale tick — using cached broker TZ: GMT-{cached}")
                return pytz.timezone(f'Etc/GMT-{cached}')
    except Exception:
        pass

    # Final fallback: GMT-3 (DST/summer offset — safer than GMT-2 for NY calculations)
    print("[DataExtractor] WARNING: Could not determine broker TZ. Defaulting to GMT-3 (DST).")
    return pytz.timezone('Etc/GMT-3')   

def _mt5_to_ny(df, broker_tz=None):
    if broker_tz is None:
        broker_tz = _get_broker_tz()
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ny_time'] = (
        df['time']
        .dt.tz_localize(broker_tz)
        .dt.tz_convert(NY_TZ)
        .dt.tz_localize(None) 
    )
    return df


def _find_session_boundaries(df_m5, gap_threshold_minutes=20):
    """
    Detects session open/close boundaries by finding gaps in M5 data.

    Brokers close and reopen XAUUSD at different times:
      - Some close at 16:45 NY, some at 17:00, some at 18:00
      - Times shift during DST changeovers
      - Weekend gap is always the largest gap

    Strategy: find all M5 index gaps > gap_threshold_minutes.
    Each gap represents a session close → reopen.
    The candle BEFORE the gap = session close (last candle).
    The candle AFTER  the gap = session open  (first candle).

    Returns list of (close_candle_time, open_candle_time) tuples,
    sorted oldest → newest. Excludes the weekend gap (>= 2 days).
    """
    if df_m5.empty or len(df_m5) < 2:
        return []

    idx       = df_m5.index.sort_values()
    gaps      = []
    threshold       = pd.Timedelta(minutes=gap_threshold_minutes)
    weekend_min_gap = pd.Timedelta(hours=20)   # weekend gap is always > 20h

    for i in range(1, len(idx)):
        diff = idx[i] - idx[i - 1]
        if diff >= threshold:   # >= so exactly-20-min gaps are caught
            gaps.append((idx[i - 1], idx[i], diff))

    # Separate daily gaps from weekend gap
    # Daily gap: 20min–20h (broker close/reopen)
    # Weekend gap: > 20h (Friday close → Sunday/Monday open)
    daily_gaps   = [(c, o, d) for c, o, d in gaps if d < weekend_min_gap]
    weekend_gaps = [(c, o, d) for c, o, d in gaps if d >= weekend_min_gap]

    return daily_gaps, weekend_gaps


def _gap_label(gap_abs):
    """Returns a human-readable label for a gap size."""
    if gap_abs < 0.50:
        return "MINIMAL GAP — price opened near previous close"
    elif gap_abs > 10.0:
        return "LARGE GAP — high probability fill target"
    elif gap_abs > 5.0:
        return "SIGNIFICANT GAP — strong overnight move"
    return ""


def compute_ndog_nwog(df_m5):
    """
    Computes NDOG (New Day Opening Gap) and NWOG (New Week Opening Gap).

    Gap detection is broker-agnostic: instead of assuming the daily
    close is at a fixed time (e.g. 17:00 NY), it detects session
    boundaries by finding gaps in the M5 candle stream.

    This handles brokers that close at 14:15, 16:45, 17:00, 18:00
    or any other time — including DST-shifted open/close times.

    NDOG: most recent daily gap (last candle before gap → first after)
    NWOG: most recent weekend gap (Friday last candle → Monday first)

    Returns two formatted strings for injection into the AI prompt.
    """
    try:
        if df_m5.empty or len(df_m5) < 10:
            return ("NDOG: Not enough data (need at least 2 sessions)",
                    "NWOG: Not enough data (need Friday + Monday data)")

        df = df_m5[['open', 'close']].copy()
        df = df.sort_index()

        daily_gaps, weekend_gaps = _find_session_boundaries(df)

        # ── NDOG — most recent daily gap ────────────────────────────
        ndog_str = "NDOG: No daily session boundary detected in available data"

        if daily_gaps:
            # Most recent daily gap
            prev_close_time, today_open_time, gap_duration = daily_gaps[-1]

            prev_close_price = float(df.loc[prev_close_time, 'close'])
            today_open_price = float(df.loc[today_open_time, 'open'])

            gap     = round(today_open_price - prev_close_price, 2)
            gap_dir = "Bullish" if gap > 0 else ("Bearish" if gap < 0 else "Flat")
            gap_abs = abs(gap)
            label   = _gap_label(gap_abs)

            close_str = prev_close_time.strftime('%Y-%m-%d %H:%M')
            open_str  = today_open_time.strftime('%Y-%m-%d %H:%M')

            ndog_str = (
                f"NDOG ({close_str} close → {open_str} open | "
                f"Session gap: {int(gap_duration.total_seconds()//60)}min): "
                f"Close={prev_close_price:.2f} | Open={today_open_price:.2f} | "
                f"Gap={gap:+.2f} ({gap_dir}, {gap_abs:.2f} pts)"
            )
            if label:
                ndog_str += f" | {label}"

        # ── NWOG — most recent weekend gap ──────────────────────────
        nwog_str = "NWOG: No weekend gap detected in available data (need Fri + Mon data)"

        if weekend_gaps:
            fri_close_time, mon_open_time, wgap_duration = weekend_gaps[-1]

            fri_close_price = float(df.loc[fri_close_time, 'close'])
            mon_open_price  = float(df.loc[mon_open_time,  'open'])

            wgap     = round(mon_open_price - fri_close_price, 2)
            wgap_dir = "Bullish" if wgap > 0 else ("Bearish" if wgap < 0 else "Flat")
            wgap_abs = abs(wgap)
            wlabel   = _gap_label(wgap_abs)

            fri_str = fri_close_time.strftime('%Y-%m-%d %H:%M')
            mon_str = mon_open_time.strftime('%Y-%m-%d %H:%M')

            # Weekly-specific label
            if wgap_abs < 0.50:
                wlabel = "MINIMAL WEEKLY GAP"
            elif wgap_abs > 10.0:
                wlabel = "LARGE WEEKLY GAP — high probability fill target"
            elif wgap_abs > 5.0:
                wlabel = "SIGNIFICANT WEEKLY GAP — strong weekend move"
            else:
                wlabel = ""

            nwog_str = (
                f"NWOG ({fri_str} Fri close → {mon_str} Mon open | "
                f"Weekend gap: {int(wgap_duration.total_seconds()//3600)}h): "
                f"Close={fri_close_price:.2f} | Open={mon_open_price:.2f} | "
                f"Gap={wgap:+.2f} ({wgap_dir}, {wgap_abs:.2f} pts)"
            )
            if wlabel:
                nwog_str += f" | {wlabel}"

        return ndog_str, nwog_str

    except Exception as e:
        return f"NDOG: Error computing ({e})", f"NWOG: Error computing ({e})"


def get_live_market_data(news_today=""):
    """
    Pulls live market data from MT5 and builds the full AI prompt context.

    BUG-6 FIX: All mt5.copy_rates_from_pos() calls are now inside a
               try/finally block so mt5.shutdown() is ALWAYS called,
               even if a rates call raises (MT5 disconnected, data gap, etc.)

    BUG-2 FIX: _get_broker_tz() called BEFORE mt5.shutdown() and result
               passed explicitly to _mt5_to_ny(). Previously shutdown() was
               called first, leaving mt5.symbol_info_tick() returning None
               and silently hardcoding GMT-2 for all brokers.

    Args:
        news_today: formatted news string from news_extractor.get_usd_news_today()
    """
    print(f"🔌 Connecting to MT5 and pulling live multi-timeframe data for {SYMBOL}...")

    if not mt5.initialize():
        print("❌ Failed to initialize MT5 connection. Is the terminal open?")
        return None

    try:
        # BUG-2 FIX: detect broker TZ before shutdown (mt5 must still be open)
        broker_tz = _get_broker_tz()

        rates_m5  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5,  0, 1000)
        rates_m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 500)
        rates_h1  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1,  0, 400)
        rates_h4  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4,  0, 100)
    finally:
        mt5.shutdown()  # BUG-6 FIX: guaranteed even if any copy_rates call raises

    if rates_m5 is None or rates_m15 is None or rates_h4 is None or rates_h1 is None:
        print(f"❌ Failed to pull data for {SYMBOL}. Check symbol name or MT5 connection.")
        return None

    # BUG-2 FIX: pass pre-captured broker_tz — no more re-detect after shutdown
    df_m5 = pd.DataFrame(rates_m5)
    df_m5 = _mt5_to_ny(df_m5, broker_tz=broker_tz)
    df_m5.set_index('ny_time', inplace=True)

    df_m15 = pd.DataFrame(rates_m15)
    df_m15 = _mt5_to_ny(df_m15, broker_tz=broker_tz)
    df_m15.set_index('ny_time', inplace=True)
    
    live_time = df_m5.index[-1]
    current_price = df_m5.iloc[-1]['close']

    # Compute NDOG and NWOG from M5 data (Python-precise, not AI-derived)
    ndog_str, nwog_str = compute_ndog_nwog(df_m5)
    
    today_data = df_m5[df_m5.index.date == live_time.date()]
    daily_high = today_data['high'].max() if not today_data.empty else current_price
    daily_low = today_data['low'].min() if not today_data.empty else current_price

    def get_session_data(start_time, end_time, start_time2=None, end_time2=None):
        """
        BUG-19 FIX: Supports two time windows via optional start_time2/end_time2.
        Asian session spans two calendar days (19:00–23:59 + 00:00–01:59).
        between_time() with a single range misses the post-midnight window.
        """
        session_df = df_m5.between_time(start_time, end_time)
        # BUG-19 FIX: merge the second window (post-midnight Asian candles)
        if start_time2 and end_time2:
            session_df2 = df_m5.between_time(start_time2, end_time2)
            session_df  = pd.concat([session_df, session_df2]).sort_index()

        if session_df.empty:
            return "Waiting", "N/A", "N/A", "N/A", "N/A"

        last_date = session_df.index.date[-1]
        latest_session = session_df[session_df.index.date == last_date]

        # For sessions spanning midnight, also include next-day rows for the second window
        if start_time2 and end_time2:
            # UB-06 FIX: Use the second-most-recent UNIQUE date instead of index[-2].
            # date[-2] is the second-to-last row's date — after a bank holiday or
            # broker data gap, this could be 2+ days ago, silently pulling the wrong
            # session range.  For example, if data goes Mon-Fri and we call on Tuesday,
            # the second-to-last ROW might be from Friday (because Saturday/Sunday have
            # no rows).  But date[-2] could also be Monday itself if we have many
            # Monday rows.  Neither is reliable.
            # The correct approach: find all unique dates, take the second-most-recent.
            unique_dates = sorted(set(session_df.index.date), reverse=True)
            if len(unique_dates) >= 2:
                prev_date = unique_dates[1]   # UB-06 FIX: second-most-recent UNIQUE date
                latest_session = session_df[
                    (session_df.index.date == last_date) |
                    (session_df.index.date == prev_date)
                ][-50:]
            else:
                latest_session = session_df[session_df.index.date == last_date][-50:]

        open_price  = latest_session.iloc[0]['open']
        high_price  = latest_session['high'].max()
        low_price   = latest_session['low'].min()
        close_price = latest_session.iloc[-1]['close']

        start_t   = pd.to_datetime(start_time).time()
        end_t     = pd.to_datetime(end_time).time()
        current_t = live_time.time()

        if start_t <= current_t <= end_t:
            status = "Developing (Live)"
        elif start_time2 and end_time2:
            start_t2 = pd.to_datetime(start_time2).time()
            end_t2   = pd.to_datetime(end_time2).time()
            status = "Developing (Live)" if start_t2 <= current_t <= end_t2 else "Closed"
        else:
            status = "Closed"

        return status, open_price, high_price, low_price, close_price

    # BUG-19 FIX: Asian session now uses BOTH windows (19:00-23:59 AND 00:00-01:59)
    asian_stat,  asian_o,  asian_h,  asian_l,  asian_c  = get_session_data('19:00', '23:59', '00:00', '01:59')
    london_stat, london_o, london_h, london_l, london_c = get_session_data('02:00', '05:00')
    ny_stat,     ny_o,     ny_h,     ny_l,     ny_c     = get_session_data('07:00', '12:00')

    m1_stat, m1_o, m1_h, m1_l, m1_c = get_session_data('08:50', '09:10')
    m2_stat, m2_o, m2_h, m2_l, m2_c = get_session_data('09:50', '10:10')
    m3_stat, m3_o, m3_h, m3_l, m3_c = get_session_data('10:50', '11:10')
    m4_stat, m4_o, m4_h, m4_l, m4_c = get_session_data('11:50', '12:10')
    m5_stat, m5_o, m5_h, m5_l, m5_c = get_session_data('13:50', '14:10')

    def process_htf(rates_data, tz=broker_tz):
        # B-03 FIX: broker_tz captured before mt5.shutdown() and passed explicitly.
        # Without this, _mt5_to_ny() calls _get_broker_tz() after shutdown → silently
        # hardcodes GMT-2 for all brokers (H4/H1 timestamps off by 1h for GMT-3 brokers).
        df = pd.DataFrame(rates_data)
        df = _mt5_to_ny(df, broker_tz=tz)
        df.set_index('ny_time', inplace=True)
        return df[['open', 'high', 'low', 'close']].to_csv(date_format='%Y-%m-%d %H:%M')

    h4_csv = process_htf(rates_h4)
    h1_csv = process_htf(rates_h1)

    # 15M Asian session window: capture from 19:00 to 02:00 (covers full Asian range + early London)
    # Body violations are evaluated AFTER 21:00 NY — the AI can determine this from the timestamps
    # ── RULE 54: Asian Range Body Violation (Python-computed) ─────
    def compute_asian_body_violation():
        """
        Rule 54 — Python computes the Asian range and body violation status.
        Sends Claude the result directly — no raw M15 CSV needed.

        Logic:
            Asian Range   = H/L of M15 candles between 19:00–21:00 NY
            Body Violation = any M15 candle AFTER 21:00 whose body
                            (min/max of open/close) closes beyond range

        Returns a formatted string for the AI prompt.
        """
        try:
            if df_m15.empty:
                return "Asian Range: No 15M data available."

            # Asian range definition window: 19:00–21:00 NY
            asian_range_m15 = df_m15.between_time('19:00', '20:59')
            if asian_range_m15.empty:
                return "Asian Range: No 19:00–21:00 candles in M15 data."

            # Use only the most recent Asian session
            last_date       = asian_range_m15.index.date[-1]
            range_candles   = asian_range_m15[asian_range_m15.index.date == last_date]

            if range_candles.empty:
                return "Asian Range: No candles for today's Asian session."

            asian_high = float(range_candles['high'].max())
            asian_low  = float(range_candles['low'].min())
            asian_mid  = round((asian_high + asian_low) / 2, 2)
            asian_rng  = round(asian_high - asian_low, 2)

            # Post-21:00 candles — check for body violations
            post_asian = df_m15.between_time('21:00', '23:59')
            post_asian = post_asian[post_asian.index.date == last_date]

            violation_str  = "No body violation detected — Asian range intact."
            violation_type = "NONE"
            violation_pts  = 0.0
            violation_time = ""

            for ts, row in post_asian.iterrows():
                body_high = max(row['open'], row['close'])
                body_low  = min(row['open'], row['close'])

                if body_high > asian_high:
                    pts = round(body_high - asian_high, 2)
                    violation_type = "BSL_SWEPT"
                    violation_pts  = pts
                    violation_time = ts.strftime('%H:%M')
                    violation_str = (
                        f"⚠️  BSL SWEPT — Body closed ABOVE Asian High at {violation_time}\n"
                        f"   Body close: {body_high:.2f} | Asian High: {asian_high:.2f} "
                        f"| Distance: +{pts:.2f} pts above\n"
                        f"   Context: Buy-Side Liquidity swept → BEARISH reversal bias for NY session\n"
                        f"   Price has likely run stops above Asian High — watch for reversal"
                    )
                    break  # first violation is the signal

                elif body_low < asian_low:
                    pts = round(asian_low - body_low, 2)
                    violation_type = "SSL_SWEPT"
                    violation_pts  = pts
                    violation_time = ts.strftime('%H:%M')
                    violation_str = (
                        f"⚠️  SSL SWEPT — Body closed BELOW Asian Low at {violation_time}\n"
                        f"   Body close: {body_low:.2f} | Asian Low: {asian_low:.2f} "
                        f"| Distance: -{pts:.2f} pts below\n"
                        f"   Context: Sell-Side Liquidity swept → BULLISH reversal bias for NY session\n"
                        f"   Price has likely run stops below Asian Low — watch for reversal"
                    )
                    break

            # Current price position relative to Asian range
            price_position = (
                "ABOVE Asian range" if current_price > asian_high else
                "BELOW Asian range" if current_price < asian_low  else
                f"INSIDE Asian range ({round((current_price - asian_low) / asian_rng * 100, 1)}% from low)"
            )

            return (
                f"Asian Range (19:00–21:00 NY): High={asian_high:.2f} | "
                f"Low={asian_low:.2f} | Mid={asian_mid:.2f} | Range={asian_rng:.2f} pts\n"
                f"Current price is: {price_position}\n"
                f"Body Violation Status: {violation_str}"
            )

        except Exception as e:
            return f"Asian Range: Error computing ({e})"

    asian_body_analysis = compute_asian_body_violation()

    # ── RULE 5: News Gate + Open Position Protocol (Python-computed) ─
    def compute_news_gate(news_text):
        """
        Rule 5 — Python parses news times, checks current time proximity,
        checks if an Asian-session position is open.

        Sends Claude a clear action directive — no raw news parsing needed.

        Returns formatted string for the AI prompt.
        """
        try:
            import re, json as _json, os as _os

            now = live_time  # already in NY time

            # ── Parse HIGH IMPACT events from news_text ───────────
            # news_text format: "- HH:MM AM/PM (NY Time): Event [🚨 HIGH IMPACT]"
            high_impact_times = []
            for line in (news_text or "").split('\n'):
                if 'HIGH IMPACT' not in line:
                    continue
                # Extract time string e.g. "08:30 AM"
                m = re.search(r'(\d{1,2}:\d{2}\s*[AP]M)', line)
                if not m:
                    continue
                try:
                    t_str = m.group(1).strip()
                    event_dt = datetime.strptime(
                        f"{now.strftime('%Y-%m-%d')} {t_str}", '%Y-%m-%d %I:%M %p'
                    )
                    # Extract event name
                    name_m = re.search(r':\s*(.+?)\s*\[', line)
                    name   = name_m.group(1).strip() if name_m else "HIGH IMPACT event"
                    high_impact_times.append((event_dt, name))
                except Exception:
                    continue

            if not high_impact_times:
                return "NEWS GATE: No HIGH IMPACT events today — no news constraints active."

            # ── Check proximity to each event ─────────────────────
            gate_lines = []
            active_gate = False

            for event_dt, name in high_impact_times:
                mins_to_event   = (event_dt - now).total_seconds() / 60
                mins_since_event = (now - event_dt).total_seconds() / 60

                if 0 <= mins_to_event <= 60:
                    # Within 60 min BEFORE event (event is in the FUTURE)
                    gate_lines.append(
                        f"🚨 NEWS GATE ACTIVE: {name} in {int(mins_to_event)} min ({event_dt.strftime('%H:%M')} NY)\n"
                        f"   RULE 5: Do NOT issue BUY or SELL. Return WAIT.\n"
                        f"   Liquidity is being manipulated ahead of this event."
                    )
                    active_gate = True
                elif 0 < mins_since_event <= 5:
                    # Within 5 min AFTER event
                    gate_lines.append(
                        f"🚨 NEWS GATE ACTIVE: {name} released {int(mins_since_event)} min ago ({event_dt.strftime('%H:%M')} NY)\n"
                        f"   RULE 5: Do NOT issue BUY or SELL. Return WAIT.\n"
                        f"   Wait for one full M15 candle after news before evaluating direction."
                    )
                    active_gate = True
                else:
                    # Show upcoming events for awareness
                    if mins_to_event > 0:
                        gate_lines.append(
                            f"📅 Upcoming: {name} at {event_dt.strftime('%H:%M')} NY "
                            f"(in {int(mins_to_event)} min)"
                        )

            # ── Check for open Asian-session position (Rule 5 protocol) ──
            open_position_line = ""
            try:
                project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                import sys as _sys_de
                _root_de = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
                if _root_de not in _sys_de.path: _sys_de.path.insert(0, _root_de)
                from paths import CONTINUATION_MEM_PATH as _cont_path
                mem_path = _cont_path
                if _os.path.exists(mem_path):
                    with open(mem_path, 'r') as f:
                        mem = _json.load(f)
                    open_trade = mem.get('open_trade', {})
                    if open_trade and open_trade.get('ticket'):
                        entry_time_str = open_trade.get('entry_time', '')
                        direction      = open_trade.get('direction', 'UNKNOWN')
                        entry_price    = open_trade.get('entry_price', 0)
                        current_pnl    = round(current_price - entry_price, 2) if direction == 'BUY' else round(entry_price - current_price, 2)

                        # Detect if this was an Asian session entry (between 19:00–02:00)
                        is_asian_trade = False
                        if entry_time_str:
                            try:
                                entry_dt = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
                                is_asian_trade = (entry_dt.hour >= 19 or entry_dt.hour < 2)
                            except Exception:
                                pass

                        if active_gate and is_asian_trade:
                            open_position_line = (
                                f"\n📊 OPEN POSITION DETECTED (Asian session entry):\n"
                                f"   {direction} from {entry_price:.2f} | Running P&L: {current_pnl:+.2f} pts\n"
                                f"   RULE 5 PROTOCOL: Close 50% NOW (5 min pre-news).\n"
                                f"   Leave remaining 50% to hit TP or close on reversal signs post-news."
                            )
                        elif active_gate:
                            open_position_line = (
                                f"\n📊 OPEN POSITION DETECTED:\n"
                                f"   {direction} from {entry_price:.2f} | Running P&L: {current_pnl:+.2f} pts\n"
                                f"   NEWS GATE ACTIVE — no new entries. Monitor existing position carefully."
                            )
            except Exception:
                pass  # Memory not accessible — skip position check

            result = "\n".join(gate_lines)
            if open_position_line:
                result += open_position_line

            return result if result else "NEWS GATE: No constraints active."

        except Exception as e:
            return f"NEWS GATE: Error processing ({e})"

    news_gate_str = compute_news_gate(news_today)

    market_context = f"""
--- LIVE MARKET FEED ---
Symbol: {SYMBOL}
Current NY Time: {live_time.strftime('%Y-%m-%d %H:%M:%S')}
Current Live Price: {current_price}
Today's Daily High: {daily_high}
Today's Daily Low: {daily_low}

--- NEWS GATE & OPEN POSITION PROTOCOL (Rule 5 — Python-computed) ---
{news_gate_str}

--- OPENING GAPS (Python-computed — use these exact levels) ---
{ndog_str}
{nwog_str}
Note: Gap levels are potential price targets / areas of interest (price tends to fill gaps).
If price is currently INSIDE a gap, the unfilled portion is a key magnet level.

--- MAJOR SESSIONS (OHLC) ---
ASIAN (19:00-00:00): {asian_stat} | O: {asian_o} | H: {asian_h} | L: {asian_l} | C: {asian_c}
LONDON (02:00-05:00): {london_stat} | O: {london_o} | H: {london_h} | L: {london_l} | C: {london_c}
NY AM (07:00-12:00): {ny_stat} | O: {ny_o} | H: {ny_h} | L: {ny_l} | C: {ny_c}

--- NY MACROS (OHLC) ---
MACRO 1 (08:50-09:10): {m1_stat} | O: {m1_o} | H: {m1_h} | L: {m1_l} | C: {m1_c}
MACRO 2 (09:50-10:10): {m2_stat} | O: {m2_o} | H: {m2_h} | L: {m2_l} | C: {m2_c}
MACRO 3 (10:50-11:10): {m3_stat} | O: {m3_o} | H: {m3_h} | L: {m3_l} | C: {m3_c}
MACRO 4 (11:50-12:10): {m4_stat} | O: {m4_o} | H: {m4_h} | L: {m4_l} | C: {m4_c}
MACRO 5 (13:50-14:10): {m5_stat} | O: {m5_o} | H: {m5_h} | L: {m5_l} | C: {m5_c}

--- HIGHER TIMEFRAME MARKET STRUCTURE ---
The following is the OHLC data for the last 100 4-Hour (H4) candles to determine macro trend:
{h4_csv}

The following is the OHLC data for the last 400 1-Hour (H1) candles mapping to the H4 period:
{h1_csv}

--- ASIAN SESSION RANGE & BODY VIOLATION (Rule 54 — Python-computed) ---
{asian_body_analysis}
Cross-reference this with any 1H FVG or Breaker zones visible in the H1 data above.
The violated side of the Asian range becomes the day's S/R level for retest entries during NY session.
"""

    try:
        # B-14 FIX: removed redundant os.makedirs(ai_folder_path) — create_all_dirs()
        # at bot startup already ensures this directory exists.
        from paths import AI_CONTEXT_PATH as _ctx_path
        file_path = _ctx_path
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(market_context)
    except Exception as e:
        print(f"⚠️ Warning: Could not save latest_context.txt to AI folder. Error: {e}")

    return market_context