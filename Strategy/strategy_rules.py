# ==========================================
# 📜 MASTER STRATEGY RULEBOOK
# ==========================================


def get_execution_rules():
    """Returns the trading framework for the AI to reference — not a hard gate."""

    rules = (
        "IMPORTANT FRAMING — READ BEFORE RULES:\n"
        "These rules are your STRATEGIC FRAMEWORK — a strong reference built from experience.\n"
        "They are NOT a hard gate that blocks valid analysis.\n"
        "If your live analysis shows a high-conviction setup that technically deviates from\n"
        "a guideline, you may act on it IF you explicitly state the deviation in your\n"
        "reasoning field and justify it with concrete price evidence.\n"
        "Rule 5 (news gate) and Rule 54 (Asian body violation) have already been processed\n"
        "by Python — the results are in the market data above. Act on those computed values.\n"
        "The REGIME ROUTER section above shows the current objective market state — "
        "use it as a strong prior. The SL/TP levels shown are regime-adjusted. "
        "You may override them but must state why in your reasoning field.\n"
        "The framework grows as the system learns — do not let it constrain superior analysis.\n\n"

        "1.  TIME CONSTRAINTS: Only execute setups that occur during the Asian Session, "
        "London Session, or the NY AM Session (07:00 AM – 12:00 PM NY Time).\n"

        "1a. LONDON SESSION GATE: London session entries are permitted ONLY when:\n"
        "    (a) Confluence score is 3/3 — all three methods (ICT, Classic PA, Elliott Wave) agree.\n"
        "    (b) AI has very high probability, confirmation, and conviction.\n"
        "    (c) Regime detector shows BULL_TREND or BEAR_TREND — not COMPRESSION or LOW_VOL_RANGE.\n"
        "    If any of these is absent, return WAIT for the London session.\n\n"

        "2.  FRIDAY RISK MANAGEMENT: If today is Friday, do not recommend any new trades "
        "if it is within 1 hour of the weekly market close. If a trade is currently open, "
        "recommend closing it immediately.\n"

        "3.  NEWS AVOIDANCE: Review the MACROECONOMIC NEWS TODAY section. Do not issue a BUY "
        "or SELL signal if the current time is within 60 minutes BEFORE or 5 minutes AFTER "
        "a 🚨 HIGH IMPACT news event. In these windows, liquidity is manipulated. Return WAIT.\n"

        "4.  MEMORY INTEGRATION: Compare the current setup to 'PAST LESSONS'. If memory shows "
        "a similar setup failed, look for a variation that would succeed or take it if it has "
        "very high probability. If memory shows a successful trade, replicate the entry logic "
        "with better risk management.\n"

        "5.  NEWS + OPEN POSITION: If a high impact news event is present and we have an open "
        "position from the same day's Asian session setup that is consolidating, close 50% of "
        "the position 5 minutes before the news and leave the remaining 50% to hit TP or close "
        "on signs of reversal.\n"

        "6.  NO CLEAR DIRECTION: If the market is not showing any clear direction or trend, "
        "it is better to wait for a clear signal rather than forcing a trade.\n"

        "7.  CLEAR TREND: If the market is showing a clear direction or trend, it is better "
        "to take the trade rather than waiting — can enter on any PD array that looks strong.\n"

        "8.  CONFIRMATION REQUIREMENT: Always look for 2+ confirmations before entering a "
        "trade (ICT / Classic / Elliott Wave).\n"

        # ══════════════════════════════════════════════════════════════════
        # ELLIOTT WAVE RULES
        # ══════════════════════════════════════════════════════════════════

        "\n9.  ELLIOTT WAVE — RULE 1 (Hard Law): Wave 2 NEVER retraces more than 100% of Wave 1. "
        "If price breaks below Wave 1 origin on a bullish count (or above on bearish), the count is dead. Return WAIT and recount.\n"

        "10. ELLIOTT WAVE — RULE 2 (Hard Law): Wave 3 is NEVER the shortest impulse wave among Wave 1, Wave 3, and Wave 5. "
        "If current W3 projection is shorter than W1, do NOT enter. Flag count as invalid.\n"

        "11. ELLIOTT WAVE — RULE 3 (Hard Law): Wave 4 NEVER overlaps Wave 1 price territory. "
        "Exception: contracting diagonal triangles only. If overlap occurs outside a diagonal, count is invalid — stand aside.\n"

        "12. ELLIOTT WAVE — WAVE 2 ENTRY (Scoring Model): A Wave 2 entry requires at least 3 of the following 5 conditions: "
        "(1) Price retraces 50%–78.6% of Wave 1 — this is the required Fibonacci window, "
        "(2) Liquidity sweep present — a wick or body has swept a BSL or SSL pool at or near the retracement zone, "
        "(3) Momentum confirmation — at least ONE of: RSI divergence, momentum slowdown on lower timeframe, "
        "volume decrease on the retracement candles, or weak and overlapping counter-trend candles, "
        "(4) Reversal candlestick pattern (pin bar, engulfing, or two-bar reversal) at the retracement zone, "
        "(5) Market structure shift on the 15M or 1H chart — a ChoCH or BOS in the direction of the new trend. "
        "Score 3/5 = valid entry with standard stop below Wave 1 origin. "
        "Score 4/5 = high probability setup. Score 5/5 = execute trade. "
        "Score 2 or below = WAIT — insufficient confirmation.\n"

        "13. ELLIOTT WAVE — WAVE 4 ENTRY: A Wave 4 long setup triggers ONLY when: "
        "(a) price holds above 38.2% retracement of Wave 3 and a corrective structure (triangle, flat, or zigzag) has visibly completed, "
        "(b) Wave 4 has NOT entered Wave 1 price territory — if it has, the count is invalid, stand aside, "
        "(c) entry is taken on the breakout of the Wave 4 consolidation range in the Wave 5 direction, "
        "(d) stop loss is placed BELOW the Wave 4 low (not below Wave 1 high — that creates an oversized stop), "
        "(e) invalidation level: if price subsequently enters Wave 1 territory after entry, exit immediately. "
        "Do NOT enter mid-consolidation before the breakout is confirmed.\n"

        "14. ELLIOTT WAVE — TRIANGLE THRUST ENTRY: When a contracting triangle completes its E-wave: "
        "(a) wait for a breakout candle to close OUTSIDE the triangle boundary, "
        "(b) target = width of the widest part of the triangle projected from the breakout point, "
        "(c) invalidation = price closes back inside the triangle.\n"

        "15. ELLIOTT WAVE — FIBONACCI TARGETS: "
        "Wave 3 target = 161.8% of Wave 1 (minimum), 261.8% (extended). "
        "Wave 5 target = equality with Wave 1 (100%) unless Wave 3 was extended, then use 61.8% of W1. "
        "Wave C target = equality with Wave A (100%). "
        "Always set partial TP at the first Fibonacci extension level.\n"

        "16. ELLIOTT WAVE — INVALIDATION LEVELS: Every Elliott Wave setup MUST have a defined invalidation price BEFORE entry. "
        "Buying Wave 2 bottom → invalidated below Wave 1 origin. "
        "Buying Wave 4 bottom → invalidated if price enters Wave 1 territory. "
        "Buying sub-wave (ii) of Wave 3 → invalidated below Wave 3 start. "
        "Triangle thrust trade → invalidated if price re-enters triangle. "
        "No invalidation level defined = no trade.\n"

        "17. ELLIOTT WAVE — AMBIGUITY FILTER: Before using Elliott Wave in confluence scoring, assess count clarity: "
        "If two or more valid alternative counts exist pointing in opposite directions, "
        "or if the suspected Wave 3 is smaller than Wave 1 and shows low momentum, "
        "or if price is overlapping heavily with no clear 5-wave or 3-wave structure, "
        "then set Elliott Wave score to ZERO and rely on ICT and Classic PA only. "
        "Do NOT force a wave label to justify an entry. A forced count is not a count.\n"

        "18. ELLIOTT WAVE — CONFLUENCE STACK: An Elliott Wave signal only qualifies if at least 4 of 5 conditions align: "
        "(1) Wave count is clearly identifiable with no ambiguity (see Rule 17), "
        "(2) Fibonacci retracement or extension level is hit, "
        "(3) Momentum confirmation — at least ONE of: RSI divergence, momentum slowdown, "
        "volume decrease on corrective candles, or weak follow-through on counter-trend move, "
        "(4) Reversal candlestick pattern at the level, "
        "(5) Higher timeframe (Daily or Weekly) structure agrees with trade direction. "
        "Score 3 or below = WAIT. Score 4 = consider. Score 5 = execute trade.\n"

        "19. ELLIOTT WAVE — WAVE 4 AVOIDANCE: Do NOT enter new trades during Wave 4 consolidation unless a triangle "
        "breakout thrust setup (Rule 14) is present. Choppy Wave 4 action produces false ICT and Classic PA signals — reduce their weight.\n"

        "20. ELLIOTT WAVE — WAVE 5 EXIT PROTOCOL: When in a suspected Wave 5: "
        "(a) do NOT add new positions, "
        "(b) tighten trailing stop aggressively, "
        "(c) momentum exhaustion trigger — at least ONE of: RSI bearish divergence (price HH, RSI LH), "
        "volume declining on the final push, weak and overlapping sub-waves, "
        "or momentum candles visibly smaller than Wave 3 candles, "
        "(d) Wave 5 failure to exceed Wave 3 high = exit immediately, sharp reversal follows.\n"

        "21. ELLIOTT WAVE — GOLD RULE (Deep Wave 2): Wave 2 in gold retraces 61.8–78.6%. "
        "Do NOT treat a shallow 38.2% retracement as Wave 2 completion. Wait for deeper confirmation.\n"

        "22. ELLIOTT WAVE — GOLD RULE (Extended Wave 3): Wave 3 in gold extends to 161.8–261.8% of Wave 1. "
        "Do not close Wave 3 longs prematurely. Trail stop and let Fibonacci extensions guide TP.\n"

        "23. ELLIOTT WAVE — GOLD RULE (Flat B-Wave Trap): B-wave nearly retests Wave 1 high in a flat correction. "
        "Do NOT buy the apparent breakout — it is a trap. Confirm false break, then look for C-wave short.\n"

        "24. ELLIOTT WAVE — GOLD RULE (DXY Inverse): If DXY is in an impulsive Wave 3 advance, "
        "gold is in a corrective structure. Do NOT issue BUY signals for gold. Return WAIT.\n"

        "25. ELLIOTT WAVE — WAVE 5 FAILURE RULE: If gold's Wave 5 reverses before exceeding Wave 3 high, "
        "close all positions immediately and flag a potential trend change.\n"

        # ── ELLIOTT WAVE: MARKET REGIME DETECTION RULES ──
        "\n26. ELLIOTT WAVE — MARKET REGIME GATE: Before assigning any Elliott Wave label, classify the 4H regime: "
        "TRENDING: clear HH+HL or LH+LL structure with expanding candle sizes and ATR expanding → "
        "use impulse wave counts, prioritize Wave 2 and Wave 3 entries, use Fibonacci extensions as targets. "
        "CORRECTIVE: sideways overlapping price, equal highs/lows, ATR flat or contracting → "
        "reduce Elliott weight, count A-B-C only, target Wave C completions, "
        "use ICT liquidity sweeps at range extremes as the primary signal instead. "
        "TRANSITIONAL: first impulse after correction or first overlap after impulse → "
        "mark as hypothesis only, wait for Wave 2 or B retracement before confirming. "
        "Regime mismatch (structure says corrective but count requires impulse) = ambiguity flag, set Elliott score to ZERO.\n"

        "27. ELLIOTT WAVE — REGIME ATR FILTER: If ATR on the 4H chart is low and price is inside a tight range "
        "(price has not made a new swing high or low in the last 8+ 4H candles), "
        "classify as RANGE MARKET. In range market: do NOT use Elliott Wave impulse setups. "
        "Use ICT liquidity sweeps at range highs and lows as the only entry method. "
        "Resume Elliott Wave analysis only after ATR expands and a clear structural break occurs.\n"

        # ── ELLIOTT WAVE: LIQUIDITY-BASED TARGETS ──
        "28. ELLIOTT WAVE — LIQUIDITY TARGET RULE: Every Elliott Wave trade must have TWO targets defined: "
        "(a) 1st TP (partial — 50% of position): the NEARER of either the first Fibonacci extension level "
        "OR the nearest liquidity pool (equal highs/lows, prior session high/low, weekly high/low). "
        "Do not hold for the second target if the first rejects price. "
        "(b) 2nd TP (remaining 50%): the price zone where a Fibonacci extension level AND a liquidity pool "
        "align within the same area — this dual-confluence zone is the institutional delivery target. "
        "Never set a single TP that ignores liquidity pools — Fibonacci alone is incomplete.\n"

        "29. ELLIOTT WAVE — LIQUIDITY OVERSHOOT RULE: If the Wave 3 Fibonacci target (161.8%) sits just below "
        "a BSL pool (equal highs, prior swing high, weekly high), the BSL pool is the actual delivery target — "
        "extend the TP to include it, do not close early at the Fibonacci level. "
        "If the Wave 5 target aligns with equal highs or a prior unswept swing high, "
        "that liquidity level IS the Wave 5 target — do not exit before it is reached unless "
        "momentum exhaustion signals are present (Rule 20).\n"

        # ── ELLIOTT WAVE: DIAGONAL RULES ──
        "30. ELLIOTT WAVE — LEADING DIAGONAL RULE: When a Leading Diagonal is identified in Wave 1 position "
        "(wedge shape, 5-3-5-3-5 internal structure, Wave 4 overlapping Wave 1 territory): "
        "(a) DO NOT invalidate the count based on Wave 4 overlap — overlap is required in a diagonal, "
        "(b) expect the subsequent Wave 2 to be a DEEP retracement (78.6%+ of Wave 1), "
        "(c) the Wave 2 pullback after a Leading Diagonal = the highest-conviction Wave 3 entry available, "
        "(d) treat the deep Wave 2 retracement zone as the primary entry zone for the Wave 3 trade.\n"

        "31. ELLIOTT WAVE — ENDING DIAGONAL EXIT RULE: When an Ending Diagonal is identified in Wave 5 or C position "
        "(wedge, 3-3-3-3-3 internal structure, Wave 4 overlapping Wave 1 territory, declining momentum): "
        "(a) close all existing longs immediately — this is an exhaustion and reversal structure, "
        "(b) wait for a displacement candle body close OUTSIDE the wedge boundary — wick only is not confirmed, "
        "(c) entry for the reversal trade: on the first retest of the broken wedge boundary, "
        "(d) target for the reversal: base of the entire diagonal structure, "
        "(e) stop: just beyond the final wave extreme of the diagonal. "
        "Do NOT counter-trade an Ending Diagonal until the body close outside the wedge is confirmed.\n"

        # ══════════════════════════════════════════════════════════════════
        # CLASSIC PRICE ACTION RULES
        # ══════════════════════════════════════════════════════════════════

        # ── S/R ZONE RULES ──
        "\n32. CLASSIC PA — S/R ZONE VALIDITY: A support or resistance level is only valid if: "
        "(a) it has been tested and respected a minimum of 2 times, "
        "(b) it is drawn as a ZONE — candle bodies define the inner boundary, wicks define the outer extreme — never a single price line, "
        "(c) role reversal is confirmed before use: broken support must hold as resistance on a retest before shorting it, and vice versa. "
        "A level tested only once or drawn as a single line = invalid, return WAIT.\n"

        "33. CLASSIC PA — TIMEFRAME LEVEL WEIGHTING: Weight S/R levels as follows when conflicts arise: "
        "Weekly and Monthly closes = highest priority, never trade against them. "
        "Daily open and close = high priority. "
        "4H swing highs and lows = medium priority, context only. "
        "1H and below = execution refinement only, not standalone entry triggers. "
        "A 1H signal that directly conflicts with a Weekly level must NOT be taken.\n"

        "34. CLASSIC PA — GOLD ROUND NUMBER RULE: Round numbers in gold ($2000, $2500, $3000, etc.) are institutional "
        "order clusters and must be treated as high-weight S/R zones. "
        "Never place a stop loss directly at a round number — offset by at least 10–15 points beyond the zone.\n"

        # ── TREND STRUCTURE RULES ──
        "35. CLASSIC PA — TREND STRUCTURE GATE: Classify market structure before any signal: "
        "Uptrend (HH + HL) → BUY signals only on pullbacks to Higher Low zones. "
        "Downtrend (LH + LL) → SELL signals only on rallies to Lower High zones. "
        "Ranging (equal highs and lows) → buy at range support, sell at range resistance only. "
        "Structure ambiguous or unclassifiable → return WAIT.\n"

        "36. CLASSIC PA — BREAK OF STRUCTURE (BOS): When the last Higher Low is broken by a body close: "
        "(a) suspend all BUY signals immediately, "
        "(b) wait for a Lower High to form to confirm trend change, "
        "(c) only re-engage BUY bias after ChoCH is confirmed (first HH + HL sequence). "
        "Do NOT buy into a BOS candle — it is a warning, not an entry.\n"

        "37. CLASSIC PA — CHANGE OF CHARACTER (ChoCH) ENTRY: A ChoCH long entry triggers when: "
        "(a) confirmed downtrend was active (LH + LL sequence), "
        "(b) first Higher High forms — warning only, do NOT enter, "
        "(c) price pulls back and forms a Higher Low WITHOUT taking out the last Lower Low, "
        "(d) reversal candlestick confirms at the Higher Low zone. "
        "Stop below the Higher Low. This is the earliest valid trend-change entry.\n"

        # ── TRENDLINE RULES ──
        "38. CLASSIC PA — TRENDLINE VALIDITY: A trendline requires a minimum of 3 confirmed touches to be tradeable. "
        "A 2-touch trendline is provisional — do not use as a primary entry trigger. "
        "A trendline break is confirmed ONLY by a candle body close outside the line. "
        "Wick-only penetration = not a break, return WAIT.\n"

        "39. CLASSIC PA — TRENDLINE BREAK & RETEST ENTRY: After confirmed trendline break (body close outside): "
        "(a) do NOT enter on the break candle — wait for the retest of the broken trendline from the new side, "
        "(b) a reversal candlestick must form at the retest zone, "
        "(c) stop placed beyond the retest candle's extreme. "
        "Entering on the break candle without retest = chasing — not permitted.\n"

        "40. CLASSIC PA — TRENDLINE ANGLE RULE: "
        "Trendlines steeper than 45° are unsustainable — do not initiate new trend-direction trades. "
        "Trendlines flatter than 15° indicate ranging conditions — treat as range, not trend. "
        "Only trendlines between 15° and 45° are valid, tradeable trend structures.\n"

        # ── BREAKOUT & RETEST RULES ──
        "41. CLASSIC PA — BREAKOUT CONFIRMATION: A breakout is only confirmed by a candle BODY close beyond the level. "
        "Wick penetration alone = not a breakout. Mark wick-only breaks as potential fakeout, not a signal.\n"

        "42. CLASSIC PA — BREAKOUT RETEST ENTRY: After confirmed breakout body close, entry sequence: "
        "(a) wait for retest of broken level from the new side, "
        "(b) reversal candlestick (pin bar, engulfing, or two-bar reversal) must form at the retest zone, "
        "(c) retest candle must NOT close back through the broken level, "
        "(d) stop below retest candle low (longs) or above retest candle high (shorts). "
        "If price blows through the retest zone with no reversal candle — return WAIT.\n"

        "43. CLASSIC PA — FAKEOUT REVERSAL RULE: If price breaks a key level and closes back inside within 1–2 candles = False Breakout. "
        "Fakeout above resistance + strong bearish body close = high-conviction short signal. "
        "Fakeout below support + strong bullish body close = high-conviction long signal. "
        "Fakeout signals carry HIGHER priority than standard breakout-retest setups — size accordingly.\n"

        # ── CANDLESTICK PATTERN RULES ──
        "44. CLASSIC PA — CANDLESTICK LOCATION RULE: No candlestick pattern generates a signal unless it forms AT "
        "a validated S/R zone, trendline touch, channel boundary, or Fibonacci confluence level. "
        "Patterns in open price space with no structural confluence = IGNORE entirely.\n"

        "45. CLASSIC PA — PIN BAR ENTRY RULE: A pin bar entry triggers only when: "
        "(a) wick is at least 2/3 of total candle length, "
        "(b) wick pierces the key level and body closes back on the opposite side of the level, "
        "(c) candle forms on 1H timeframe or higher — sub-1H pin bars are noise. "
        "Entry: open of next candle or 50% retrace of pin bar. "
        "Stop: beyond the wick tip. "
        "Body larger than 1/3 of total candle = weak pin, do not trade.\n"

        "46. CLASSIC PA — ENGULFING ENTRY RULE: Engulfing entry triggers only when: "
        "(a) engulfing candle's body FULLY covers the prior candle's body (wicks excluded), "
        "(b) forms at a validated S/R zone or structural level, "
        "(c) engulfing body is at least 1.5x the size of the prior candle's body — barely-engulfing = weak, skip. "
        "Entry: open of next candle. Stop: beyond the low of engulfing candle (longs) or high (shorts).\n"

        "47. CLASSIC PA — INSIDE BAR RULE: Inside bar triggers only when: "
        "(a) entire inside candle range (high AND low) is within the prior mother bar range, "
        "(b) forms at a key S/R level or after a strong trend move, "
        "(c) entry is on a body close OUTSIDE the mother bar — wick-only break = false break risk, do NOT enter. "
        "Stop: beyond the opposite extreme of the mother bar.\n"

        "48. CLASSIC PA — DOJI CONFIRMATION RULE: A doji at a key level does NOT trigger an entry alone. "
        "Entry is only valid when the NEXT candle closes decisively in the reversal direction "
        "with a body at least 2x the doji's total range. "
        "Doji followed by another doji or small body = still undecided, return WAIT.\n"

        "49. CLASSIC PA — TWO-BAR REVERSAL RULE: Two-bar reversal entry triggers when: "
        "(a) first candle is a strong trend candle (body > 60% of total range) in the prevailing direction, "
        "(b) second candle is an equal or larger opposing candle closing at or beyond the first candle's open, "
        "(c) both candles form at a key S/R level on 4H or Daily timeframe. "
        "Entry: open of third candle. Stop: beyond the extreme of the two-bar structure.\n"

        # ── CONFLUENCE & MULTI-TIMEFRAME RULES ──
        "50. CLASSIC PA — CONFLUENCE STACK: A Classic PA signal qualifies only if at least 4 of 5 align: "
        "(1) Validated S/R zone with 2+ prior touches, "
        "(2) Market structure agrees (trading with trend direction), "
        "(3) Trendline or channel boundary confluence at the same level, "
        "(4) High-probability candlestick reversal pattern at the level, "
        "(5) Higher timeframe (Daily or Weekly) level aligns with the entry zone. "
        "Score 3 or below = WAIT. Score 4 = reduced size. Score 5 = execute trade.\n"

        "51. CLASSIC PA — MULTI-TIMEFRAME CONFLICT RULE: If Daily or Weekly structure is bearish and a long signal "
        "appears on 1H or 4H, the signal is DOWNGRADED to scalp-only with reduced size. "
        "Never take a full-size counter-trend position based on a lower timeframe PA signal alone. "
        "Higher timeframe structure always overrides lower timeframe signals.\n"

        # ── GOLD-SPECIFIC CLASSIC PA RULES ──
        "52. CLASSIC PA — GOLD DAILY CLOSE RULE: A Daily candle body close above or below a key S/R zone "
        "is the highest-weight intraday confirmation signal in gold. "
        "A daily body close beyond a key level overrides conflicting intraday signals — reassess trade bias immediately.\n"

        "53. CLASSIC PA — GOLD WICK TRAP RULE: Gold regularly generates false wick spikes through major S/R levels before reversing. "
        "Never enter on a wick penetration alone. "
        "Wick spike through a key zone + candle body closes back inside = Wick Trap. "
        "A wick trap is a high-conviction reversal signal in the opposite direction — treat as a fakeout (Rule 36).\n"

        "54. CLASSIC PA — GOLD ASIAN RANGE RULE: The Asian session high and low define the intraday battleground. "
        "Do not assign directional bias until one side of the Asian range is broken by a body close. "
        "The broken side of the Asian range becomes the day's S/R level for retest entries during London and NY sessions.\n"

        "55. CLASSIC PA — GOLD SUNDAY GAP RULE: If gold opens Sunday with a gap, do NOT trade in the gap direction "
        "until the gap is filled or price clearly rejects the fill attempt. "
        "Sunday gap momentum is not a directional signal — gaps fill before trend resumes the majority of the time. "
        "No directional bias assigned based solely on Sunday open gap direction.\n"
    )
    return rules


def get_adaptive_timing_rule():
    """
    Returns the adaptive cycle timing rule injected into every prompt.
    Tells Claude how to populate the next_check_mins field.
    Kept separate so it can be updated without touching the main rulebook.
    """
    return (
        "\n--- ADAPTIVE CYCLE TIMING (REQUIRED FIELD) ---\n"
        "Every JSON response MUST include 'next_check_mins' (integer, minimum 5).\n"
        "This controls how long the bot sleeps before the next analysis cycle.\n"
        "Choose based on how soon the market will reach an actionable state:\n"
        "  5  min — Price is AT a key level (OB/FVG/S&R) RIGHT NOW. Setup imminent.\n"
        "  10 min — Setup developing. Price approaching structure but not there yet.\n"
        "  15 min — Clear bias, but price needs time to retrace to the entry zone.\n"
        "  20 min — Waiting for a candle close confirmation (15M or 1H body close).\n"
        "  30 min — No current setup. Valid regime but market needs time to move.\n"
        "  45 min — Clear consolidation / Wave 4 / range midpoint. Nothing forming soon.\n"
        "  60 min — COMPRESSION regime, deeply ranging, or session is clearly dead.\n"
        "TRADE OPEN GUIDANCE: Trade management runs independently — you only control\n"
        "  when the next AI analysis fires. Suggest 5-10 mins if trade is in danger\n"
        "  (near SL, REVERSAL regime). Suggest 15-30 mins if running cleanly to TP.\n"
        "HARD FLOOR: Never return next_check_mins < 5. The system enforces this.\n"
        "HARD CAP: Never return next_check_mins > 60 during active sessions.\n"
    )
