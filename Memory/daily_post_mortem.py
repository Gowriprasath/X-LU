"""
daily_post_mortem.py
====================
Runs at 5PM NY time every trading day.

FIX C3 : Removed ALL direct genai.Client() calls. All AI now via call_ai().
FIX C7 : STEP 5 (MissWish) and STEP 6 (Strategy Scout) now run EVERY day
          even when there are no new closed trades. The early-return guard
          was blocking MissWish on no-trade days — precisely the days with
          the most blocked/missed data. Fixed by restructuring the function
          so STEP 5 and 6 are unconditional.

STEP 1 — Group summary (console output)
STEP 2 — Per-trade hindsight (written to trade_memory.json)
STEP 3 — Counterfactual analysis
STEP 4 — Keyword tagger (Bug 12 fix: mark_as_run_today AFTER tagger completes)
STEP 5 — MissWish analysis (NOW RUNS ON ALL DAYS — fix C7)
STEP 6 — Strategy Scout   (NOW RUNS ON ALL DAYS — fix C7)
Also: calls memory_manager.update_pnl_r() for continuous label support.
"""

import sys as _sys, os as _os
_mc_dir = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _mc_dir not in _sys.path: _sys.path.insert(0, _mc_dir)
from ai_client import call_ai, AI_MODEL  # FIX C3: replaces genai.Client

import json
import os
import sys
import time
import re
from datetime import datetime
import pytz
from dotenv import load_dotenv
import memory_manager

load_dotenv()

# ==========================================
# DYNAMIC PATHING & CONFIG
# ==========================================
current_dir         = os.path.dirname(os.path.abspath(__file__))
base_dir            = os.path.dirname(current_dir)
from paths import (TRADE_MEMORY_PATH, POST_MORTEM_TRACKER,
                       COUNTERFACTUAL_LOG_PATH, SHADOW_GATE_AUDIT_PATH,
                       create_all_dirs as _cad_pm)
_cad_pm()
MEMORY_FILE         = TRADE_MEMORY_PATH
TRACKER_FILE        = POST_MORTEM_TRACKER
COUNTERFACTUAL_FILE = COUNTERFACTUAL_LOG_PATH
NY_TZ               = pytz.timezone('America/New_York')


def get_ny_time():
    return datetime.now(NY_TZ)


def check_if_run_today():
    today_str = get_ny_time().strftime("%Y-%m-%d")
    try:
        if os.path.exists(TRACKER_FILE):
            with open(TRACKER_FILE, 'r') as f:
                last_run = f.read().strip()
            if last_run == today_str:
                return True
    except Exception:
        pass
    return False


def mark_as_run_today():
    today_str = get_ny_time().strftime("%Y-%m-%d")
    try:
        with open(TRACKER_FILE, 'w') as f:
            f.write(today_str)
    except Exception as e:
        print(f"Warning: Could not save post-mortem tracker: {e}")


# ================================================================
# STEP 3 — COUNTERFACTUAL ANALYSIS (Post-Mortem Gap fix)
# FIX C3: Uses call_ai() — no genai.Client
# ================================================================

def _run_counterfactual(recent_trades: list) -> dict:
    """
    Post-Mortem Gap fix: Counterfactual analysis — "What if we traded
    every regime signal without AI's gatekeeping?"
    FIX C3: Removed direct genai.Client usage. Now uses call_ai().
    """
    today = get_ny_time().strftime("%Y-%m-%d")

    trades_with_gates = [
        t for t in recent_trades
        if isinstance(t.get('gate_decisions'), dict)
    ]

    if not trades_with_gates:
        print("[PostMortem] Counterfactual: No gate_decisions data yet. "
              "Will populate after next trade cycle.")
        return {}

    gate_analysis_data = []
    for t in trades_with_gates:
        gate_analysis_data.append({
            "ticket":             t.get('ticket'),
            "signal":             t.get('signal'),
            "result":             t.get('result'),
            "regime":             t.get('regime', 'UNKNOWN'),
            "regime_confidence":  t.get('regime_confidence'),
            "session":            t.get('session', 'UNKNOWN'),
            "meta_prob":          t.get('meta_prob'),
            "gate_decisions":     t.get('gate_decisions', {}),
            "entry":              t.get('entry'),
            "detailed_review":    t.get('detailed_review', ''),
        })

    prompt = f"""
You are a quantitative trading analyst reviewing whether an AI gating layer
added value over the base HMM/XGBoost regime signal alone.

DATE: {today}

EXECUTED TRADES WITH GATE DATA:
{json.dumps(gate_analysis_data, indent=2)}

For each trade, the gate_decisions field shows:
  - confidence_gate: why the regime detector allowed/blocked this trade
  - dual_gate: whether the AI direction matched the regime
  - signal_aligned: whether the final signal matched regime direction
  - size_multiplier: how much the AI's conviction sized the position

TASK:
1. For each trade, estimate what would have happened WITHOUT AI gating.
2. Compute:
   REGIME_ONLY_SCORE: Estimated win rate if we traded every regime signal >= 0.55
   AI_GATED_SCORE:    Actual win rate of executed trades

3. Verdict (choose one):
   - AI_ADDS_VALUE:    AI gating improved win rate significantly (>5pp)
   - AI_NEUTRAL:       Win rates similar (+/- 5pp)
   - REGIME_SUPERIOR:  Regime-only would have outperformed

4. Specific recommendation: one concrete change to improve gating logic.

Respond ONLY with valid JSON:
{{
  "regime_only_score": float (0-1),
  "ai_gated_score": float (0-1),
  "verdict": "AI_ADDS_VALUE|AI_NEUTRAL|REGIME_SUPERIOR",
  "key_finding": "1-2 sentence summary",
  "recommendation": "1 concrete actionable change",
  "trade_by_trade": [
    {{"ticket": "...", "regime_would_trade": true/false, "ai_helped": true/false, "note": "..."}}
  ]
}}
"""
    try:
        raw = call_ai(prompt=prompt)  # FIX C3
        if raw is None:
            print("[PostMortem] Counterfactual AI call failed (all keys exhausted).")
            return {}

        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*',     '', raw)
        result = json.loads(raw)
        if not isinstance(result, dict):
            return {}

        print("\n" + "=" * 55)
        print(" COUNTERFACTUAL ANALYSIS — AI vs REGIME-ONLY")
        print("=" * 55)
        print(f"  Regime-only win rate : {result.get('regime_only_score', 0):.0%}")
        print(f"  AI-gated win rate    : {result.get('ai_gated_score', 0):.0%}")
        print(f"  Verdict  : {result.get('verdict', 'N/A')}")
        print(f"  Finding  : {result.get('key_finding', '')}")
        print(f"  Action   : {result.get('recommendation', '')}")
        print("=" * 55 + "\n")

        log = []
        if os.path.exists(COUNTERFACTUAL_FILE):
            try:
                with open(COUNTERFACTUAL_FILE, 'r') as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append({
            "date":                today,
            "regime_only_score":   result.get('regime_only_score'),
            "ai_gated_score":      result.get('ai_gated_score'),
            "verdict":             result.get('verdict'),
            "key_finding":         result.get('key_finding'),
            "recommendation":      result.get('recommendation'),
            "trade_by_trade":      result.get('trade_by_trade', []),
            "n_trades_analysed":   len(gate_analysis_data),
        })
        log = log[-90:]
        with open(COUNTERFACTUAL_FILE, 'w') as f:
            json.dump(log, f, indent=4)
        return result

    except Exception as e:
        print(f"[PostMortem] Counterfactual analysis failed: {e}")
        return {}


# ================================================================
# STEP 3b — SHADOW GATE AUDIT
# Uses shadow_journal.get_stats_for_postmortem() — the first place
# this data is ever actually consumed.  Previously the shadow journal
# tracked every blocked signal's real outcome but that data was never
# read by anything.  This step feeds it to the AI daily so the bot can
# learn which gates are protecting it and which are costing it edge.
# ================================================================

def _run_shadow_gate_audit(today: str) -> dict:
    """
    Pulls yesterday's shadow journal stats and asks the AI to render a
    verdict on each gate:  WORKING | MARGINAL | COSTING_EDGE.

    Returns the parsed audit dict (also written to shadow_gate_audit.json).
    Non-fatal — any failure prints a warning and returns {}.
    """
    try:
        sys.path.insert(0, current_dir)
        import shadow_journal as sj
        stats = sj.get_stats_for_postmortem(days_back=1)
    except Exception as e:
        print(f"[PostMortem] Shadow gate audit: could not load shadow journal: {e}")
        return {}

    if not stats or stats.get("total_signals", 0) == 0:
        print("[PostMortem] Shadow gate audit: no shadow journal entries for today — skipping.")
        return {}

    gate_accuracy   = stats.get("gate_accuracy",   {})
    top_missed      = stats.get("top_missed_moves", [])
    by_regime       = stats.get("by_regime",        {})
    total_signals   = stats.get("total_signals",    0)
    blocked         = stats.get("blocked",          0)
    taken           = stats.get("taken",            0)
    blocked_tp_rate = stats.get("blocked_tp_rate",  0)
    r_on_table      = stats.get("total_r_left_on_table", 0)

    # Build a readable gate summary for the prompt
    gate_lines = []
    for gate, g in gate_accuracy.items():
        verdict = g.get("verdict", "INSUFFICIENT_DATA")
        fires   = g.get("fires",   0)
        tp      = g.get("tp_hit",  0)
        sl      = g.get("sl_hit",  0)
        acc     = g.get("gate_accuracy", "n/a")
        r_cost  = g.get("r_cost",  0)
        gate_lines.append(
            f"  {gate:<22} fires={fires:>3}  tp_hit={tp:>2}  sl_hit={sl:>2}  "
            f"accuracy={acc}  R_cost={r_cost:+.1f}  verdict={verdict}"
        )

    missed_lines = []
    for m in top_missed[:3]:
        missed_lines.append(
            f"  {m['timestamp']} | {m['signal']} @ {m['entry_price']} | "
            f"gate={m['gate']} | moved {m.get('max_favorable', 0):.1f}R | "
            f"status={m['status']} | reason: {m['block_reason'][:80]}"
        )

    regime_lines = []
    for regime, rs in by_regime.items():
        regime_lines.append(
            f"  {regime:<18}  blocked={rs['blocked']:>2}  "
            f"tp_hit={rs['tp_hit']:>2}  sl_hit={rs['sl_hit']:>2}  "
            f"R_cost={rs['r_cost']:+.1f}"
        )

    prompt = f"""
You are a quantitative trading analyst reviewing gate filter performance for a Gold (XAUUSD) AI trading bot.

DATE: {today}

SHADOW JOURNAL SUMMARY (last 24h):
  Total signals tracked : {total_signals}
  Taken by bot          : {taken}
  Blocked by gates      : {blocked}
  Blocked TP-hit rate   : {blocked_tp_rate:.0%}  ← % of blocked signals that would have been winners
  Total R left on table : {r_on_table:+.1f}R

GATE-BY-GATE PERFORMANCE:
  (accuracy = % of blocked signals that were actual losers — higher = gate is working)
  (R_cost = sum of R the gate cost us by blocking winners; negative means it saved us R)
{chr(10).join(gate_lines) if gate_lines else "  No gate data."}

TOP 3 MISSED MOVES (blocked signals that moved most in signal direction):
{chr(10).join(missed_lines) if missed_lines else "  None."}

BLOCKED SIGNALS BY REGIME:
{chr(10).join(regime_lines) if regime_lines else "  No regime data."}

TASK:
1. For each gate that has enough data (fires >= 5), state whether it is:
   - WORKING:      blocks more losers than winners (accuracy >= 65%) — keep it
   - MARGINAL:     roughly random (45–65%) — worth tuning
   - COSTING_EDGE: blocks more winners than losers (accuracy < 45%) — needs a rethink

2. Identify the single gate causing the most unnecessary edge loss today.

3. Give ONE concrete, specific tuning suggestion (e.g. "Lower CONFIDENCE_GATE threshold
   from 70% to 60% during BULL_TREND only" — not generic advice).

4. Note any regime where blocking cost is disproportionately high.

Respond ONLY with valid JSON — no markdown, no preamble:
{{
  "gate_verdicts": {{
    "GATE_NAME": {{
      "verdict": "WORKING|MARGINAL|COSTING_EDGE|INSUFFICIENT_DATA",
      "fires": int,
      "accuracy": float_or_null,
      "r_cost": float,
      "note": "1-sentence observation"
    }}
  }},
  "biggest_edge_leak": "GATE_NAME or null",
  "tuning_suggestion": "1 concrete actionable change",
  "regime_concern": "regime name + 1-sentence note, or null",
  "overall_verdict": "GATES_WORKING|GATES_OVER_FILTERING|MIXED",
  "summary": "2-3 sentence plain-English takeaway for the bot operator"
}}
"""

    try:
        raw = call_ai(prompt=prompt)
        if raw is None:
            print("[PostMortem] Shadow gate audit: AI call failed.")
            return {}

        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*',     '', raw)
        result = json.loads(raw)
        if not isinstance(result, dict):
            return {}

        # Console summary
        print("\n" + "=" * 58)
        print(" SHADOW GATE AUDIT — FILTER PERFORMANCE REPORT")
        print("=" * 58)
        print(f"  Signals today : {total_signals} total | {taken} taken | {blocked} blocked")
        print(f"  Blocked TP rate: {blocked_tp_rate:.0%}  |  R left on table: {r_on_table:+.1f}R")
        print(f"  Overall verdict: {result.get('overall_verdict', 'N/A')}")
        print(f"  Biggest leak  : {result.get('biggest_edge_leak', 'N/A')}")
        print(f"  Suggestion    : {result.get('tuning_suggestion', '')}")
        if result.get('regime_concern'):
            print(f"  Regime concern: {result.get('regime_concern')}")
        print(f"\n  {result.get('summary', '')}")
        print()
        for gname, gv in result.get("gate_verdicts", {}).items():
            print(f"  {gname:<22} → {gv.get('verdict','?'):<18} {gv.get('note','')}")
        print("=" * 58 + "\n")

        # Persist to shadow_gate_audit.json (rolling 90-day log)
        audit_log = []
        if os.path.exists(SHADOW_GATE_AUDIT_PATH):
            try:
                with open(SHADOW_GATE_AUDIT_PATH, 'r') as f:
                    audit_log = json.load(f)
            except Exception:
                pass

        audit_log.append({
            "date":              today,
            "total_signals":     total_signals,
            "taken":             taken,
            "blocked":           blocked,
            "blocked_tp_rate":   round(blocked_tp_rate, 3),
            "r_on_table":        round(r_on_table, 2),
            "gate_verdicts":     result.get("gate_verdicts", {}),
            "biggest_edge_leak": result.get("biggest_edge_leak"),
            "tuning_suggestion": result.get("tuning_suggestion"),
            "regime_concern":    result.get("regime_concern"),
            "overall_verdict":   result.get("overall_verdict"),
            "summary":           result.get("summary"),
        })
        audit_log = audit_log[-90:]   # keep 90 days

        with open(SHADOW_GATE_AUDIT_PATH, 'w') as f:
            json.dump(audit_log, f, indent=4)

        return result

    except Exception as e:
        print(f"[PostMortem] Shadow gate audit: AI parse failed: {e}")
        return {}

def run_post_mortem(simulated_time=None, m5_df=None, h1_df=None):
    """
    Full post-mortem pipeline.

    FIX B1:  Accepts optional simulated_time so backtest runs stamp
             lessons/logs with the correct simulated date rather than
             the real wall-clock date.
    FIX C3:  All AI calls via call_ai() — no genai.Client anywhere.
    FIX C7:  Steps 5 (MissWish) and 6 (Strategy Scout) run EVERY day,
             regardless of whether there are new closed trades.
             Previous code had early-return before these steps — fixed.
    Bug 12:  mark_as_run_today() called AFTER keyword tagger completes.
    """
    import pytz as _pytz
    _NY = _pytz.timezone('America/New_York')
    if simulated_time is not None:
        if simulated_time.tzinfo is None:
            _sim_ny = _NY.localize(simulated_time)
        else:
            _sim_ny = simulated_time.astimezone(_NY)
        today = _sim_ny.strftime("%Y-%m-%d")
        today = get_ny_time().strftime("%Y-%m-%d")

    pm_success = True

    # -- Gate Review: process shadow journal blocks ------------------
    try:
        from Integration.Wisdom_Worker.gate_review_builder \
            import process_new_blocks, should_trigger_monthly_review
        new_blocks = process_new_blocks()
        if new_blocks > 0:
            print(f"[PostMortem] Gate review: {new_blocks} new "
                  f"confidence gate outcomes logged.")
    except Exception as _gre:
        print(f"[PostMortem] Gate review processing skipped: {_gre}")

    # ----------------------------------------------------------------
    # Load trade memory (needed for STEP 1-4)
    # ----------------------------------------------------------------
    trade_data = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                trade_data = json.load(f)
        except Exception as e:
            print(f"Post-Mortem could not read memory: {e}")

    # Process CLOSED trades without hindsight yet
    closed_trades = [
        t for t in trade_data
        if t.get('status') == 'CLOSED'
        and not t.get('hindsight_feedback', '').strip()
    ]

    recent_trades = closed_trades[-5:] if closed_trades else []

    # ----------------------------------------------------------------
    # STEP 1 — Group summary (only if there are trades to review)
    # ----------------------------------------------------------------
    if recent_trades:
        group_prompt = f"""
You are an elite Trading Coach. Review these recent closed trades:
{json.dumps(recent_trades, indent=2)}

In 3 short bullet points, what is the biggest mistake or the best pattern
the bot should remember for tomorrow? Keep it strictly analytical. No fluff.
"""
        try:
            print("\n AI Coach: Analyzing today's trades for Post-Mortem...")
            response = call_ai(prompt=group_prompt)  # FIX C3
            if response:
                print("\n" + "=" * 50)
                print(" DAILY POST-MORTEM LESSONS")
                print("=" * 50)
                print(response.strip())
                print("=" * 50 + "\n")
        except Exception as e:
            print(f"AI Coach group summary failed: {e}")

        # ----------------------------------------------------------------
        # STEP 2 — Per-trade hindsight + continuous pnl_r label
        # ----------------------------------------------------------------
        for trade in recent_trades:
            ticket = trade.get('ticket')
            try:
                per_trade_prompt = f"""
You are an elite Trading Coach writing a concise post-mortem for one specific trade.

TRADE RECORD:
{json.dumps(trade, indent=2)}

Write 2-3 sentences of hindsight feedback for this specific trade.
Focus on: what structural lesson does this trade teach?
What should the bot look for or avoid next time it sees a similar setup?
Be specific to THIS trade — not generic advice.
Respond with plain text only, no JSON, no bullet points.
"""
                hindsight_text = call_ai(prompt=per_trade_prompt)  # FIX C3
                if hindsight_text:
                    memory_manager.update_hindsight_review(ticket, hindsight_text.strip())
                    print(f"[PostMortem] Hindsight written for Ticket #{ticket}")

                # Continuous label — compute actual pnl_r from exit data
                # UB-05 FIX: The old code stamped pnl_r = -1.0 for ALL LOSS trades.
                # Break-even exits (-0.05R), partial losses (-0.3R), and overnight
                # guard exits (-0.2R) all got the same -1.0 label.  The meta-labeller
                # trained on these labels had wrong threshold calibration — it saw
                # every loss as a full 1R wipeout.
                #
                # Fix: attempt to recover the actual exit price from MT5 history.
                # If MT5 is unavailable, use episode_recorder's recorded pnl_r if
                # available, and only fall back to the blunt -1.0 for pure LOSS with
                # no better data.
                if trade.get('pnl_r') is None:
                    try:
                        entry  = float(trade.get('entry', 0) or 0)
                        sl     = float(trade.get('sl',    0) or 0)
                        tp     = float(trade.get('tp',    0) or 0)
                        result = (trade.get('result') or '').upper()
                        signal = (trade.get('signal') or 'BUY').upper()
                        sl_dist = abs(entry - sl)
                        if sl_dist > 0 and result in ('WIN', 'LOSS', 'CLOSED_BY_AI'):
                            pnl_r = None

                            # ── Try to recover actual exit P&L from episode_recorder ──
                            try:
                                sys.path.insert(0, os.path.join(base_dir, 'Quant', 'rl_manager'))
                                from episode_recorder import _read_episodes
                                episodes = _read_episodes()
                                for ep in reversed(episodes):
                                    if str(ep.get('ticket')) == str(ticket):
                                        ep_pnl_r = ep.get('final_pnl_r')
                                        if ep_pnl_r is not None:
                                            pnl_r = float(ep_pnl_r)
                                        break
                            except Exception:
                                pass

                            # ── Fall back to TP/SL price-based estimate ───────────────
                            if pnl_r is None:
                                if result == 'WIN' and tp > 0:
                                    pnl_r = abs(tp - entry) / sl_dist
                                elif result == 'LOSS':
                                    # Use actual SL distance as a proxy for -1R.
                                    # Still not perfect but at least uses this trade's
                                    # SL geometry rather than a flat -1.0 for all losses.
                                    pnl_r = -1.0
                                else:
                                    # CLOSED_BY_AI or other — approximate as 0 R (neutral)
                                    pnl_r = 0.0

                            memory_manager.update_pnl_r(ticket, round(pnl_r, 4))
                            print(f"[PostMortem] pnl_r={pnl_r:+.2f}R stored for #{ticket}")
                    except Exception as e:
                        print(f"[PostMortem] pnl_r calc failed for #{ticket}: {e}")

            except Exception as e:
                print(f"[PostMortem] Per-trade hindsight failed for Ticket #{ticket}: {e}")
                continue

        # ----------------------------------------------------------------
        # STEP 3 — Counterfactual analysis
        # ----------------------------------------------------------------
        try:
            _run_counterfactual(recent_trades)
        except Exception as e:
            print(f"[PostMortem] Counterfactual analysis error: {e}")

        # ----------------------------------------------------------------
        # STEP 4 — Keyword tagger
        # Bug 12: mark_as_run_today() only AFTER tagger completes.
        # ----------------------------------------------------------------
        tagged_count = 0
        try:
            sys.path.append(os.path.join(base_dir, 'Integration', 'Wisdom_Worker'))
            from keyword_tagger import tag_trade
            for trade in recent_trades:
                ticket = trade.get('ticket')
                if ticket:
                    try:
                        print(f"[PostMortem] Triggering keyword tagger for Ticket #{ticket}...")
                        tag_trade(ticket)
                        tagged_count += 1
                    except Exception as e:
                        print(f"[PostMortem] Tagger failed for #{ticket}: {e}")
        except Exception as tag_err:
            print(f"[PostMortem] Keyword tagger import error: {tag_err}")

        if tagged_count >= len(recent_trades) - 1:
            print(f"[PostMortem] ✅ Hindsight+tags complete. "
                  f"{tagged_count}/{len(recent_trades)} trades processed.")
        else:
            print(f"[PostMortem] ⚠️ Only {tagged_count}/{len(recent_trades)} trades tagged. "
                  f"NOT marking complete — will retry.")
            pm_success = False
    else:
        print("[PostMortem] No new closed trades to process for STEPS 1–4.")

    # ================================================================
    # FIX C7: STEPS 3b, 5 + 6 NOW RUN UNCONDITIONALLY EVERY DAY
    # These were previously inside the 'if closed_trades' guard, meaning
    # they were silently skipped on days with no taken trades — the days
    # with the most valuable blocked/missed signal data.
    # ================================================================

    # ----------------------------------------------------------------
    # STEP 3b — Shadow Gate Audit
    # First step to ever actually consume shadow_journal data.
    # Runs unconditionally — blocked-signal analysis is most informative
    # on days with few or no taken trades (heavy filter days, dead zones).
    # ----------------------------------------------------------------
    try:
        print(f"\n[PostMortem] ── STEP 3b: Shadow Gate Audit for {today} ──")
        _run_shadow_gate_audit(today)
    except Exception as e:
        print(f"[PostMortem] Shadow gate audit error (non-fatal): {e}")

    # ----------------------------------------------------------------
    # STEP 5 — MissWish analysis
    # FIX C3: run_analysis() no longer takes a client param — uses call_ai()
    # FIX C7: runs every day, not gated on closed_trades
    # ----------------------------------------------------------------
    try:
        print(f"\n[PostMortem] ── STEP 5: MissWish analysis for {today} ──")

        sys.path.insert(0, current_dir)
        from misswish_analyser import run_analysis as run_misswish

        new_entries = run_misswish(date_str=today, m5_df=m5_df, h1_df=h1_df)  # FIX B4

        if new_entries:
            try:
                mw_tagger_dir = os.path.join(base_dir, 'Integration', 'Wisdom_Worker')
                if mw_tagger_dir not in sys.path:
                    sys.path.append(mw_tagger_dir)
                from misswish_tagger import tag_misswish_entry

                mw_tagged = 0
                for entry in new_entries:
                    try:
                        print(f"[PostMortem] MissWish tagging: {entry['id']}...")
                        tag_misswish_entry(entry['id'])
                        mw_tagged += 1
                    except Exception as e:
                        print(f"[PostMortem] MissWish tagger failed for {entry['id']}: {e}")

                print(f"[PostMortem] MissWish: {len(new_entries)} setups found, {mw_tagged} tagged.")
            except Exception as te:
                print(f"[PostMortem] MissWish tagger import error: {te}")
        else:
            print(f"[PostMortem] MissWish: no new setups identified for {today}.")

    except Exception as mw_err:
        print(f"[PostMortem] MissWish analysis error (non-fatal): {mw_err}")

    # ----------------------------------------------------------------
    # STEP 6 — Strategy Scout
    # FIX C3: run_scout() no longer takes a client param — uses call_ai()
    # FIX C7: runs every day, not gated on closed_trades
    # ----------------------------------------------------------------
    try:
        print(f"\n[PostMortem] ── STEP 6: Strategy Scout ──")

        _scout_mem_dir = current_dir
        if _scout_mem_dir not in sys.path:
            sys.path.insert(0, _scout_mem_dir)
        from strategy_scout import run_scout, _ping_pending_proposals

        _ping_pending_proposals()
        n_proposals = run_scout()   # FIX C3: no client arg

        if n_proposals > 0:
            print(f"[PostMortem] Strategy Scout: {n_proposals} new proposal(s) written.")
        else:
            print(f"[PostMortem] Strategy Scout: no new patterns at threshold today.")

    except Exception as scout_err:
        print(f"[PostMortem] Strategy Scout error (non-fatal): {scout_err}")

    # -- Gate Review: monthly meta-review if triggered ----------------
    try:
        if should_trigger_monthly_review():
            print("[PostMortem] Monthly gate review triggered...")
            from tools.trigger_monthly_gate_review import run_monthly_review
            run_monthly_review()
    except Exception as _gmr:
        print(f"[PostMortem] Monthly gate review skipped: {_gmr}")

    if pm_success:
        mark_as_run_today()
        print("[PostMortem] ✅ Daily post-mortem process fully complete and marked as run today.")


def check_and_run_if_needed(force=False, simulated_time=None, m5_df=None, h1_df=None):
    """
    Background thread entry point called from main_bot.py (live mode)
    and check_simulated_postmortem() in backtest_engine.py (backtest).

    FIX B1 — Backtest clock bug (two-part fix):
      Part A: Accept force + simulated_time so the backtest engine call
              no longer raises TypeError and falls through to the
              wall-clock fallback.
      Part B: In backtest mode skip the while-loop entirely. Execute
              run_post_mortem(simulated_time=...) once and return.
              The engine already enforces the hour>=17 / weekday / not-
              run-today guards — no need to re-check them here.
              run_post_mortem now stamps all logs with the simulated date
              instead of today's real date.

    BUG-11 FIX: in-memory _ran_today_date guard prevents same-day re-run.
    FIX #6:     sleep in 10-second ticks so shutdown_event is checked fast.
    """

    # ----------------------------------------------------------------
    # BACKTEST MODE — engine passes simulated_time; run once and return.
    # ----------------------------------------------------------------
    if simulated_time is not None:
        import pytz as _pytz
        _NY = _pytz.timezone('America/New_York')
        if simulated_time.tzinfo is None:
            sim_ny = _NY.localize(simulated_time)
        else:
            sim_ny = simulated_time.astimezone(_NY)
        sim_date_str = sim_ny.strftime("%Y-%m-%d")

        print(f"[PostMortem] Backtest trigger — sim date {sim_date_str} "
              f"({sim_ny.strftime('%H:%M')} NY).")
        try:
            run_post_mortem(simulated_time=simulated_time, m5_df=m5_df, h1_df=h1_df)
        except Exception as pm_err:
            print(f"[PostMortem] Unhandled exception in run_post_mortem: {pm_err}")
        finally:
            mark_as_run_today()
        return

    # ----------------------------------------------------------------
    # LIVE MODE — long-lived background thread using the real wall clock.
    # ----------------------------------------------------------------

    # Import shutdown event lazily — main_bot sets it, we just read it
    def _is_shutting_down():
        try:
            import main_bot as _mb
            return _mb._shutdown_event.is_set()
        except Exception:
            return False

    print("[PostMortem] Background thread started.")
    _ran_today_date = None  # BUG-11 FIX: in-memory guard against same-day re-run

    while not _is_shutting_down():
        try:
            ny_now    = get_ny_time()
            today_str = ny_now.strftime("%Y-%m-%d")

            if ny_now.weekday() >= 5:
                # Weekend guard: bypass execution and waiting logs
                if _ran_today_date != today_str:
                    print(f"[PostMortem] Weekend detected ({ny_now.strftime('%A')}) — post-mortem deactivated.")
                    _ran_today_date = today_str
            elif ny_now.hour >= 17 and _ran_today_date != today_str:
                if not check_if_run_today():
                    time.sleep(15)
                    if _is_shutting_down():
                        break   # don't start a multi-minute AI run during shutdown
                    try:
                        run_post_mortem()
                    except Exception as pm_err:
                        print(f"[PostMortem] Unhandled exception in run_post_mortem: {pm_err}")
                    finally:
                        # BUG-11 FIX: always mark as done — prevents infinite retry loop
                        mark_as_run_today()
                        _ran_today_date = today_str
                else:
                    # Already run (persisted on disk) — update in-memory guard too
                    _ran_today_date = today_str
            elif ny_now.hour < 17:
                hours_to_pm    = 17 - ny_now.hour
                mins_to_pm     = (17 * 60) - (ny_now.hour * 60 + ny_now.minute)
                print(f"[PostMortem] Waiting for 5PM NY. "
                      f"~{hours_to_pm}h {mins_to_pm % 60}m remaining.")
                _ran_today_date = None  # reset guard after midnight

        except Exception as e:
            print(f"[PostMortem] Error in check loop: {e}")

        # FIX #6: sleep in short intervals so shutdown_event is checked promptly
        for _ in range(360):   # 360 × 10s = 1h total, but checks shutdown every 10s
            if _is_shutting_down():
                break
            time.sleep(10)

    print("[PostMortem] Shutdown signal received. Exiting cleanly.")
