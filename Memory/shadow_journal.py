"""
shadow_journal.py — The Bot's Blind Spot Recorder
===================================================

The core problem this solves:
    Post-mortem only sees taken trades. Every WAIT, every blocked signal,
    every gate that fired is invisible. You cannot measure the cost of your
    filters without knowing what the market did after you stood aside.

What this module does:
    1. LOG every signal at every decision point — taken AND blocked.
       For each signal, record: direction, Claude's entry/SL/TP, regime,
       confidence, session, which gate fired, and why.

    2. TRACK the real move forward in time.
       Every cycle, tick() scans all open shadow entries and checks whether
       price has hit the proposed TP or SL. This captures the real outcome.

    3. REPORT for post-mortem.
       get_stats_for_postmortem() returns structured analysis by gate type,
       by regime, by session — answering: which filters are adding edge and
       which are just costing you profitable trades?

Shadow entry lifecycle:
    PENDING     → signal logged, forward tracking started
    TP_HIT      → price reached Claude's proposed TP (would have been a WIN)
    SL_HIT      → price reached Claude's proposed SL (would have been a LOSS)
    EXPIRED     → max_bars elapsed without hitting either (inconclusive)
    TAKEN       → bot actually executed this trade (truth baseline for comparison)

Gate codes logged at each block point:
    CONFIDENCE_GATE     — regime confidence too low, Claude skipped entirely
    CONSECUTIVE_WAIT    — 3+ WAIT cycles in same regime, Claude skipped
    LLM_CACHE           — identical market context, WAIT reused
    STRATEGY_BLOCKED    — COMPRESSION/UNCERTAIN hit strategy_selector
    CLAUDE_WAIT         — Claude itself returned WAIT/HOLD
    DUAL_GATE           — regime and signal disagreed (or COMPRESSION/UNCERTAIN)
    META_GATE           — meta-labeller probability too low
    LONDON_GATE         — London session requires 3/3 confluence
    STACKING_BLOCK      — position already open
    NEWS_BLOCK          — inside high-impact news window
    VALIDATION_FAIL     — hallucination guard or RR check rejected levels
    SPREAD_BLOCK        — spread too wide at execution time
    TAKEN               — trade was actually executed (baseline reference)
"""

import os
import json
import threading
import uuid        # MEM-04 FIX: uuid4 suffix replaces %f (microseconds) for unique IDs
import sys as _sys
import os as _os
from datetime import datetime

_root_sj = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _root_sj not in _sys.path: _sys.path.insert(0, _root_sj)
from paths import SHADOW_JOURNAL_PATH, create_all_dirs as _cad_sj
_cad_sj()
from master_controls import SHADOW_FORWARD_BARS as FORWARD_BARS

current_dir  = _os.path.dirname(_os.path.abspath(__file__))
JOURNAL_FILE = SHADOW_JOURNAL_PATH
_lock        = threading.Lock()

# Shadow entry statuses
STATUS_PENDING  = "PENDING"
STATUS_TP_HIT   = "TP_HIT"
STATUS_SL_HIT   = "SL_HIT"
STATUS_EXPIRED  = "EXPIRED"
STATUS_TAKEN    = "TAKEN"


# ================================================================
# FILE HELPERS
# ================================================================

# M-02 FIX: In-memory cache — eliminates the full JSON read on every 5-minute
# tick() call.  Before this fix, every cycle did:
#     acquire lock → read ~1MB JSON → iterate 2000 entries → write ~1MB JSON
# That's 96 read+write pairs per 8-hour session even when nothing changed.
#
# After this fix:
#   Cold start   — one disk read, stored in _cache (list of dicts)
#   _read()      — returns shallow copy of _cache; zero disk I/O
#   _write()     — updates _cache then flushes to disk (write path unchanged)
#   tick()       — early-exit if no PENDING in cache (common during dead zone)
#
# Shallow copy (list(_cache)) is intentional: dict entries in the copy are the
# same objects as in _cache, so in-place mutations inside tick() (e["status"] = …)
# propagate back to _cache automatically.  _write() then canonises the state.
_cache: list = None   # None = not yet loaded from disk


def _read() -> list:
    """Return a shallow copy of the in-memory cache, loading from disk on first call."""
    global _cache
    if _cache is not None:
        return list(_cache)      # hot path — zero disk I/O
    # Cold path: first call in this process — populate cache from disk
    try:
        if os.path.exists(JOURNAL_FILE):
            with open(JOURNAL_FILE, 'r', encoding='utf-8') as f:
                _cache = json.load(f)
        else:
            _cache = []
    except Exception as e:
        print(f"[ShadowJournal] Read error: {e}")
        _cache = []
    return list(_cache)


def _write(entries: list):
    """Write entries to disk and keep the in-memory cache in sync."""
    global _cache
    _cache = entries             # sync cache before disk write (callers already hold _lock)
    try:
        os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
        with open(JOURNAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)
    except Exception as e:
        print(f"[ShadowJournal] Write error: {e}")


# ================================================================
# PUBLIC API — called from main_bot.py
# ================================================================

def log_signal(
    signal:            str,           # "BUY" or "SELL"
    entry_price:       float,         # price at signal time (Claude's or live price)
    sl_price:          float,         # Claude's SL (0 if not available — gate fired early)
    tp_price:          float,         # Claude's TP (0 if not available)
    regime:            str,
    regime_confidence: float,
    session:           str,
    gate_blocked_by:   str,           # one of the gate codes above, or "TAKEN"
    block_reason:      str = "",      # human-readable reason from the gate
    meta_prob:         float = None,
    confluence_score:  int   = 0,
    claude_thesis:     str   = "",
    strategy:          str   = "",    # TREND / RANGE / SWEEP / BLOCKED
    was_taken:         bool  = False,
):
    """
    Logs one signal point — whether taken or blocked.

    If sl_price or tp_price are 0 (gate fired before Claude's levels were
    available), we estimate them using ATR-based defaults:
        SL = entry ± 1.5 × 10pts (rough ATR proxy)
        TP = entry ± 3.0 × 10pts (2R minimum)
    These estimates let the forward tracker still compute a meaningful outcome
    for pre-Claude blocks (confidence gate, consecutive wait, etc.).
    """
    if signal not in ("BUY", "SELL"):
        return   # only directional signals are trackable

    if entry_price <= 0:
        return   # no price — can't track

    # Estimate SL/TP if not available (pre-Claude blocks)
    # MEM-02 FIX: atr_proxy corrected from 10.0 → 2.5.
    # Gold M5 ATR during active sessions = $1.50–$3.50 per candle.
    # The old value of 10.0 produced:
    #   estimated SL = entry ± 15pts  (4-10× too wide for XAUUSD)
    #   estimated TP = entry ± 30pts
    # At those levels, price almost NEVER reached the estimated SL in 24 bars,
    # so pre-Claude blocked signals showed as EXPIRED instead of SL_HIT.
    # This made CONFIDENCE_GATE and CONSECUTIVE_WAIT look ~90% accurate when
    # they were actually blocking many would-be losses.  The analytics were
    # meaningless — post-mortem gate verdicts were all wrong.
    # 2.5 $/oz is the realistic mid-session M5 ATR for XAUUSD.
    atr_proxy = 2.5   # MEM-02 FIX: was 10.0 (4-10× too wide for XAUUSD)
    if sl_price <= 0:
        sl_price = (entry_price - 1.5 * atr_proxy) if signal == "BUY" \
                   else (entry_price + 1.5 * atr_proxy)
        sl_estimated = True
    else:
        sl_estimated = False

    if tp_price <= 0:
        tp_price = (entry_price + 3.0 * atr_proxy) if signal == "BUY" \
                   else (entry_price - 3.0 * atr_proxy)
        tp_estimated = True
    else:
        tp_estimated = False

    sl_dist = abs(entry_price - sl_price)
    tp_dist = abs(tp_price   - entry_price)
    rr      = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0

    entry = {
        # MEM-04 FIX: ID now uses uuid4 suffix instead of %f (microseconds).
        # On Windows, datetime.now() microsecond resolution is 10-15ms.
        # Two log_signal() calls within the same 10ms window (e.g. confidence
        # gate + consecutive wait firing together) produced identical IDs,
        # silently overwriting entries or causing analytics collisions.
        # uuid4 is collision-free by construction (128-bit random).
        "id":                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal[:1]}_{uuid.uuid4().hex[:8]}",
        "timestamp":         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "signal":            signal,
        "entry_price":       round(entry_price, 5),
        "sl_price":          round(sl_price, 5),
        "tp_price":          round(tp_price, 5),
        "sl_estimated":      sl_estimated,
        "tp_estimated":      tp_estimated,
        "sl_dist":           round(sl_dist, 5),
        "tp_dist":           round(tp_dist, 5),
        "rr":                rr,
        "regime":            regime,
        "regime_confidence": round(regime_confidence, 4) if regime_confidence else None,
        "session":           session,
        "strategy":          strategy,
        "gate_blocked_by":   gate_blocked_by,
        "block_reason":      block_reason[:300] if block_reason else "",  # trim long reasons
        "meta_prob":         round(meta_prob, 4) if meta_prob else None,
        "confluence_score":  confluence_score,
        "claude_thesis":     claude_thesis[:400] if claude_thesis else "",
        "was_taken":         was_taken,
        # Forward tracking fields (filled by tick())
        "status":            STATUS_TAKEN if was_taken else STATUS_PENDING,
        "bars_tracked":      0,
        "max_forward_bars":  FORWARD_BARS,
        "max_favorable":     0.0,    # max price movement in signal direction (in R)
        "price_at_close":    None,   # price when status changed from PENDING
        "outcome_bars":      None,   # how many bars until TP/SL hit
        "outcome_r":         None,   # P&L in R-multiples at resolution
    }

    with _lock:
        entries = _read()
        entries.append(entry)
        # Rolling cap: keep last 2000 entries (taken + blocked)
        if len(entries) > 2000:
            entries = entries[-2000:]
        _write(entries)

    tag = "✓ TAKEN" if was_taken else f"✗ BLOCKED [{gate_blocked_by}]"
    print(f"[ShadowJournal] {tag} | {signal} @ {entry_price:.2f} | "
          f"SL:{sl_price:.2f} TP:{tp_price:.2f} (RR:{rr}) | "
          f"Regime:{regime} ({regime_confidence:.0%})")


def tick(current_price: float, candle: dict = None):
    """
    Called every cycle from main_bot.py with the current live price.

    FIX #7 — OHLC-aware SL/TP detection.
    PROBLEM: tick() was called with a single live price point. Between two
    5-minute calls, price could wick DOWN through SL and then recover ABOVE TP.
    At the next call, current_price >= TP → TP_HIT logged → FALSE WIN.
    The trade actually stopped out first.

    FIX: accept an optional `candle` dict with 'high' and 'low' keys (the last
    completed M5 candle). When a candle is supplied:
      - BUY:  check candle['low']  <= SL first (SL hit first → worst case)
              then check candle['high'] >= TP
      - SELL: check candle['high'] >= SL first
              then check candle['low']  <= TP
    This matches standard backtest convention: assume adverse move hits first.
    Falls back to current_price comparison if no candle is supplied (backward compat).

    Also tracks max_favorable_excursion in R-multiples (how far the trade
    could have gone in the right direction before reversing).

    Only updates PENDING entries — TAKEN entries are managed by episode_recorder.
    """
    if current_price <= 0:
        return

    with _lock:
        entries = _read()

        # M-02 FIX: early exit — no PENDING entries means nothing to update.
        # Common during the dead zone (13:00–19:00) and after heavy news blocks
        # where all recent signals have already resolved.  Saves a full write
        # cycle (~15ms) on the vast majority of ticks.
        if not any(e.get("status") == STATUS_PENDING for e in entries):
            return

        modified = False

        for e in entries:
            if e.get("status") != STATUS_PENDING:
                continue

            signal     = e["signal"]
            entry_p    = e["entry_price"]
            sl_p       = e["sl_price"]
            tp_p       = e["tp_price"]
            sl_dist    = e.get("sl_dist", 1)
            bars       = e.get("bars_tracked", 0)
            max_fav    = e.get("max_favorable", 0.0)

            # Compute current excursion in R-multiples
            if signal == "BUY":
                excursion_r = (current_price - entry_p) / sl_dist if sl_dist > 0 else 0
            else:  # SELL
                excursion_r = (entry_p - current_price) / sl_dist if sl_dist > 0 else 0

            # Update max favorable excursion
            if excursion_r > max_fav:
                e["max_favorable"] = round(excursion_r, 4)
                modified = True

            # ── FIX #7: OHLC-aware hit detection ──────────────────────────
            # When we have a completed M5 candle, use High/Low to determine
            # which level was touched FIRST within the bar.
            # Convention: always check the ADVERSE direction first (SL),
            # identical to industry-standard backtesting (MetaTrader, TradingView).
            if candle and "high" in candle and "low" in candle:
                bar_high = float(candle["high"])
                bar_low  = float(candle["low"])
                if signal == "BUY":
                    sl_touched = bar_low  <= sl_p
                    tp_touched = bar_high >= tp_p
                else:  # SELL
                    sl_touched = bar_high >= sl_p
                    tp_touched = bar_low  <= tp_p

                if sl_touched and tp_touched:
                    # Both levels inside the same bar — SL wins (worst case)
                    sl_hit, tp_hit = True, False
                else:
                    sl_hit = sl_touched
                    tp_hit = tp_touched
            else:
                # Fallback: single price comparison (backward compat / no candle data)
                if signal == "BUY":
                    tp_hit = current_price >= tp_p
                    sl_hit = current_price <= sl_p
                else:
                    tp_hit = current_price <= tp_p
                    sl_hit = current_price >= sl_p
            # ── end FIX #7 ─────────────────────────────────────────────────

            # Check resolution
            if tp_hit:
                e["status"]       = STATUS_TP_HIT
                e["price_at_close"] = current_price
                e["outcome_bars"] = bars + 1
                e["outcome_r"]    = round(e["tp_dist"] / sl_dist, 4) if sl_dist > 0 else 0
                modified = True
                print(f"[ShadowJournal] TP_HIT shadow {e['id']} | "
                      f"{signal} | entry:{entry_p} tp:{tp_p} | "
                      f"gate:{e['gate_blocked_by']} | {bars+1} bars")

            elif sl_hit:
                e["status"]       = STATUS_SL_HIT
                e["price_at_close"] = current_price
                e["outcome_bars"] = bars + 1
                e["outcome_r"]    = round(-(e["sl_dist"] / sl_dist), 4) if sl_dist > 0 else -1
                modified = True
                print(f"[ShadowJournal] SL_HIT shadow {e['id']} | "
                      f"{signal} | entry:{entry_p} sl:{sl_p} | "
                      f"gate:{e['gate_blocked_by']} | {bars+1} bars")

            elif bars + 1 >= e.get("max_forward_bars", FORWARD_BARS):
                # Time expired — use final price for R calculation
                if signal == "BUY":
                    final_r = (current_price - entry_p) / sl_dist
                else:
                    final_r = (entry_p - current_price) / sl_dist
                e["status"]         = STATUS_EXPIRED
                e["price_at_close"] = current_price
                e["outcome_bars"]   = bars + 1
                e["outcome_r"]      = round(final_r, 4)
                modified = True

            else:
                e["bars_tracked"] = bars + 1
                modified = True

        if modified:
            _write(entries)


# ================================================================
# POST-MORTEM ANALYSIS
# ================================================================

def get_stats_for_postmortem(days_back: int = 1) -> dict:
    """
    Called by daily_post_mortem.py.

    Returns structured stats for the post-mortem prompt covering
    the last `days_back` days of shadow journal entries.

    Key questions answered:
        1. By gate: how often did each gate fire, and what % of those
           would have been TP_HIT vs SL_HIT (edge cost per gate)?
        2. Best missed moves: the blocked signals that moved the most
           in the signal direction (highest max_favorable_excursion).
        3. By regime: in which regimes are blocks most "expensive"?
        4. By session: in which sessions are we leaving the most money?
        5. Gate accuracy: is each gate blocking more winners or losers?
    """
    entries = _read()
    if not entries:
        return {"summary": "No shadow journal entries yet.", "entries": []}

    # Filter to days_back
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=days_back)
    recent = []
    for e in entries:
        try:
            ts = datetime.strptime(e["timestamp"], '%Y-%m-%d %H:%M:%S')
            if ts >= cutoff:
                recent.append(e)
        except Exception:
            pass

    if not recent:
        return {"summary": f"No entries in last {days_back} day(s).", "entries": []}

    resolved  = [e for e in recent if e.get("status") != STATUS_PENDING]
    blocked   = [e for e in resolved if e.get("gate_blocked_by") != STATUS_TAKEN]
    taken     = [e for e in resolved if e.get("gate_blocked_by") == STATUS_TAKEN]

    # ── Stats by gate ──────────────────────────────────────────────
    from collections import defaultdict
    gate_stats = defaultdict(lambda: {
        "count": 0, "tp_hit": 0, "sl_hit": 0, "expired": 0,
        "total_r_lost": 0.0, "max_favorable_sum": 0.0
    })

    for e in blocked:
        gate = e.get("gate_blocked_by", "UNKNOWN")
        gs   = gate_stats[gate]
        gs["count"] += 1
        status = e.get("status")
        if status == STATUS_TP_HIT:
            gs["tp_hit"] += 1
            gs["total_r_lost"] += e.get("outcome_r", 0) or 0
        elif status == STATUS_SL_HIT:
            gs["sl_hit"] += 1
        elif status == STATUS_EXPIRED:
            gs["expired"] += 1
        gs["max_favorable_sum"] += e.get("max_favorable", 0) or 0

    # Gate accuracy: % of blocked that would have been LOSSES (SL_HIT)
    # A gate with 70% SL_HIT is doing its job. 30% SL_HIT means it's
    # blocking winners.
    gate_accuracy = {}
    for gate, gs in gate_stats.items():
        resolved_count = gs["tp_hit"] + gs["sl_hit"]
        accuracy = gs["sl_hit"] / resolved_count if resolved_count > 0 else None
        gate_accuracy[gate] = {
            "fires":         gs["count"],
            "tp_hit":        gs["tp_hit"],
            "sl_hit":        gs["sl_hit"],
            "expired":       gs["expired"],
            "gate_accuracy": round(accuracy, 3) if accuracy else "n/a",
            "r_cost":        round(gs["total_r_lost"], 2),   # R left on table
            "avg_max_fav":   round(gs["max_favorable_sum"] / max(gs["count"], 1), 2),
            "verdict":       _gate_verdict(accuracy, gs["tp_hit"], gs["count"]),
        }

    # ── Top missed moves ───────────────────────────────────────────
    # Signals that moved most in the right direction after being blocked
    missed_moves = sorted(
        [e for e in blocked if (e.get("max_favorable") or 0) > 0.5],
        key=lambda x: x.get("max_favorable", 0),
        reverse=True
    )[:5]

    missed_summary = []
    for e in missed_moves:
        missed_summary.append({
            "timestamp":     e["timestamp"],
            "signal":        e["signal"],
            "regime":        e["regime"],
            "session":       e["session"],
            "gate":          e["gate_blocked_by"],
            "block_reason":  e["block_reason"],
            "meta_prob":     e.get("meta_prob"),
            "entry_price":   e["entry_price"],
            "tp_price":      e["tp_price"],
            "max_favorable": e.get("max_favorable"),
            "status":        e["status"],
            "outcome_r":     e.get("outcome_r"),
            "tp_estimated":  e.get("tp_estimated", False),
        })

    # ── By regime ──────────────────────────────────────────────────
    regime_stats = defaultdict(lambda: {"blocked": 0, "tp_hit": 0, "sl_hit": 0, "r_cost": 0.0})
    for e in blocked:
        rs = regime_stats[e.get("regime", "UNKNOWN")]
        rs["blocked"] += 1
        if e.get("status") == STATUS_TP_HIT:
            rs["tp_hit"] += 1
            rs["r_cost"] += e.get("outcome_r", 0) or 0
        elif e.get("status") == STATUS_SL_HIT:
            rs["sl_hit"] += 1

    # ── Taken trade baseline ───────────────────────────────────────
    taken_tp = sum(1 for e in taken if e.get("status") == STATUS_TP_HIT)
    taken_sl = sum(1 for e in taken if e.get("status") == STATUS_SL_HIT)

    # ── Total R left on table from blocked TP_HITs ─────────────────
    total_r_cost = sum(
        e.get("outcome_r", 0) or 0
        for e in blocked
        if e.get("status") == STATUS_TP_HIT
    )

    return {
        "period_days":      days_back,
        "total_signals":    len(recent),
        "taken":            len(taken),
        "blocked":          len(blocked),
        "pending":          len([e for e in recent if e.get("status") == STATUS_PENDING]),
        "taken_win_rate":   round(taken_tp / max(len(taken), 1), 3),
        "blocked_tp_rate":  round(
            sum(1 for e in blocked if e.get("status") == STATUS_TP_HIT)
            / max(len(blocked), 1), 3
        ),
        "total_r_left_on_table": round(total_r_cost, 2),
        "gate_accuracy":    dict(gate_accuracy),
        "top_missed_moves": missed_summary,
        "by_regime":        {k: dict(v) for k, v in regime_stats.items()},
        "raw_entries":      recent[-50:],   # last 50 for prompt context
    }


def _gate_verdict(accuracy, tp_hits, total) -> str:
    """Simple verdict string for the post-mortem prompt."""
    if accuracy is None or total < 3:
        return "INSUFFICIENT_DATA"
    if accuracy >= 0.65:
        return "WORKING"        # gate blocks more losers than winners — good
    elif accuracy >= 0.45:
        return "MARGINAL"       # roughly random — gate adds uncertainty
    else:
        return "COSTING_EDGE"   # gate blocks more winners than losers — review it


def get_live_summary() -> str:
    """Compact one-liner for main_bot console. Called each cycle."""
    entries = _read()
    pending = sum(1 for e in entries if e.get("status") == STATUS_PENDING)
    tp_hits = sum(1 for e in entries[-100:] if e.get("status") == STATUS_TP_HIT
                  and e.get("gate_blocked_by") != STATUS_TAKEN)
    sl_hits = sum(1 for e in entries[-100:] if e.get("status") == STATUS_SL_HIT
                  and e.get("gate_blocked_by") != STATUS_TAKEN)
    resolved = tp_hits + sl_hits
    if resolved > 0:
        rate = f"{tp_hits/resolved:.0%} win"
    else:
        rate = "n/a"
    return (f"[ShadowJournal] {pending} tracking | "
            f"Last 100 blocked: {tp_hits}TP/{sl_hits}SL ({rate})")
