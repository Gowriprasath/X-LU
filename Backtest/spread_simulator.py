"""
Backtest/spread_simulator.py — Realistic XAUUSD Spread Model
=============================================================
Replaces the flat-spread assumption in the original backtest engine.

Two-layer model
───────────────
Layer 1 — Session spread
    Each session has a typical spread range for XAUUSD based on historical
    broker data. We sample uniformly inside the range so every trade gets
    a slightly different spread, just like live trading.

    Session         Typical range (USD/oz)
    ─────────────────────────────────────
    Asian           $0.20 – $0.60
    London          $0.50 – $1.20
    NY_AM 07–09     $0.80 – $1.50   ← primary execution window
    NY_AM 09–12     $0.40 – $0.90
    Dead zone       $0.20 – $0.50

Layer 2 — News spike multiplier
    During high-impact news (NFP, CPI, FOMC) the spread widens 3–8×.
    This is applied on top of the session spread so a $1.20 London
    spread becomes $3.60–$9.60 during news spikes.

    The window is ±2 minutes around event time — this is the realistic
    window when brokers widen spreads most aggressively.

Slippage model
──────────────
Separate from spread. Slippage is the extra fill deviation beyond the
quoted ask/bid. In live gold trading:
    - Normal conditions : 0.1–0.5 pts slippage on entry
    - London/NY open   : 0.3–1.5 pts
    - News events      : 0.5–3.0 pts (can be much worse)

Slippage always moves AGAINST the trade:
    BUY  → filled ABOVE the quoted ask (paid more)
    SELL → filled BELOW the quoted bid (received less)

Usage (called from backtest_engine.py):
    sim = SpreadSimulator()
    spread = sim.get_spread(session, current_time, news_today)
    entry_filled = sim.adjusted_entry(signal, entry, spread)
    slippage = sim.get_slippage(session, current_time, news_today)
    # slippage already included in adjusted_entry() — don't double-count
"""

import random
from datetime import datetime


# ── Session spread ranges (USD/oz) ────────────────────────────────────────
# (min, max) — sampled uniformly for each trade
SESSION_SPREAD_RANGES = {
    "Asian":  (0.20, 0.60),
    "London": (0.50, 1.20),
    "NY_AM":  (0.60, 1.50),   # blends open (wide) and mid-session (tighter)
    None:     (0.20, 0.50),   # dead zone / unknown
}

# Early NY open (07:00–09:00) is notably wider than mid-session
NY_OPEN_SPREAD_RANGE = (0.80, 1.50)   # 07:00–09:00 NY
NY_MID_SPREAD_RANGE  = (0.40, 0.90)   # 09:00–12:00 NY

# ── News spike multipliers ─────────────────────────────────────────────────
# Applied when current_time is within NEWS_SPIKE_WINDOW_MINS of a high-impact event.
NEWS_SPIKE_WINDOW_MINS = 2     # ±2 min around event time
NEWS_SPIKE_MULTIPLIER  = (3.0, 8.0)   # spread multiplied by 3–8×

# ── Slippage ranges (USD/oz, always against the trade) ────────────────────
SESSION_SLIPPAGE_RANGES = {
    "Asian":  (0.10, 0.50),
    "London": (0.30, 1.20),
    "NY_AM":  (0.30, 1.50),
    None:     (0.10, 0.30),
}
NEWS_SLIPPAGE_RANGE = (0.50, 3.00)   # during news events


class SpreadSimulator:
    """
    Stateless simulator — all methods can be called without shared state.
    Pass a SpreadSimulator instance around for clean testing/mocking.
    """

    def get_spread(self, session: str, current_time: datetime,
                   news_today=None) -> float:
        """
        Returns the simulated spread (USD/oz) for a trade at current_time.

        Applies:
          1. Session-based spread (sampled from range)
          2. NY open widening (07:00–09:00 NY is wider than mid-session)
          3. News spike multiplier if current_time is near a high-impact event

        Args:
            session:      "Asian" | "London" | "NY_AM" | None
            current_time: timezone-aware datetime (NY time)
            news_today:   list of news event dicts from news_history.get_news_for_date()
                          each dict has keys: time (datetime), impact (str), name (str)

        Returns:
            float — spread in USD/oz (same units as XAUUSD price)
        """
        # Layer 1: session base spread
        if session == "NY_AM":
            hour = current_time.hour if hasattr(current_time, "hour") else 9
            lo, hi = NY_OPEN_SPREAD_RANGE if hour < 9 else NY_MID_SPREAD_RANGE
        else:
            lo, hi = SESSION_SPREAD_RANGES.get(session, SESSION_SPREAD_RANGES[None])

        base_spread = round(random.uniform(lo, hi), 2)

        # Layer 2: news spike multiplier
        spike_mult = self._get_news_spike_multiplier(current_time, news_today)
        spread = round(base_spread * spike_mult, 2)

        return spread

    def get_slippage(self, session: str, current_time: datetime,
                     news_today=None) -> float:
        """
        Returns simulated slippage (USD/oz, always positive = adverse).

        Slippage is separate from spread:
          - Spread   = broker's bid-ask gap (always paid)
          - Slippage = extra fill deviation beyond quoted price (market impact)

        Returns:
            float — slippage in USD/oz (always >= 0)
        """
        # News spike → larger slippage
        if self._is_near_news(current_time, news_today):
            lo, hi = NEWS_SLIPPAGE_RANGE
        else:
            lo, hi = SESSION_SLIPPAGE_RANGES.get(
                session, SESSION_SLIPPAGE_RANGES[None])

        return round(random.uniform(lo, hi), 2)

    def adjusted_entry(self, signal: str, quoted_entry: float,
                        spread: float, slippage: float = 0.0) -> float:
        """
        Returns the realistic fill price after applying spread and slippage.

        BUY:  filled above quoted price (paid spread/2 + slippage)
        SELL: filled below quoted price (paid spread/2 + slippage)

        Args:
            signal:        "BUY" or "SELL"
            quoted_entry:  AI's ideal entry price (mid-price)
            spread:        spread in USD/oz from get_spread()
            slippage:      additional adverse fill (default 0, caller can pass
                           result of get_slippage() for full realism)

        Returns:
            float — realistic fill price
        """
        half_spread = spread / 2.0
        total_cost  = half_spread + slippage
        if signal == "BUY":
            return round(quoted_entry + total_cost, 2)
        else:
            return round(quoted_entry - total_cost, 2)

    @staticmethod
    def apply_spread_cost(raw_pips: float, spread: float) -> float:
        """
        Deducts spread from pip PnL for round-trip cost.
        Called on both entry and exit legs (spread paid twice per round trip).

        One round-trip = 1 full spread deducted from raw pips.
        We deduct it once here (the second leg is priced into the fill).
        """
        return round(raw_pips - spread, 4)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _is_near_news(self, current_time: datetime, news_today) -> bool:
        """True if current_time is within NEWS_SPIKE_WINDOW_MINS of a high-impact event."""
        if not news_today:
            return False
        for event in news_today:
            impact = event.get("impact", "").upper()
            if impact != "HIGH":
                continue
            event_time = event.get("time")
            if event_time is None:
                continue
            try:
                diff_mins = abs((current_time - event_time).total_seconds()) / 60
                if diff_mins <= NEWS_SPIKE_WINDOW_MINS:
                    return True
            except Exception:
                continue
        return False

    def _get_news_spike_multiplier(self, current_time: datetime,
                                    news_today) -> float:
        """
        Returns spread multiplier for news events.
        Normal conditions → 1.0 (no change).
        Near high-impact event → 3.0–8.0×.

        The multiplier tapers as time moves away from the event:
          0–2 min:  full spike (3.0–8.0×)
          2–5 min:  partial spike (1.5–3.0×)
          5+ min:   no spike (1.0×)
        """
        if not news_today:
            return 1.0

        best_mult = 1.0
        for event in news_today:
            impact = event.get("impact", "").upper()
            if impact != "HIGH":
                continue
            event_time = event.get("time")
            if event_time is None:
                continue
            try:
                diff_mins = abs((current_time - event_time).total_seconds()) / 60
                if diff_mins <= 2:
                    mult = random.uniform(*NEWS_SPIKE_MULTIPLIER)
                    best_mult = max(best_mult, mult)
                elif diff_mins <= 5:
                    mult = random.uniform(1.5, 3.0)
                    best_mult = max(best_mult, mult)
            except Exception:
                continue

        return round(best_mult, 2)


# ── Convenience function (mirrors old backtest_engine get_spread call) ────
_default_sim = SpreadSimulator()

def get_spread(session: str, current_time: datetime, news_today=None) -> float:
    """Module-level convenience wrapper around SpreadSimulator.get_spread()."""
    return _default_sim.get_spread(session, current_time, news_today)
