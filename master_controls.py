"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MASTER CONTROLS                                   ║
║              Gold AI Bridge — XAUUSD Trading Bot                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  ONE FILE. EVERY SETTING. CHANGE HERE → WHOLE SYSTEM UPDATES.       ║
║                                                                      ║
║  Sections:                                                           ║
║    1.  AI PROVIDER          — model, API keys, 3-key rotation       ║
║    2.  TRADING SYMBOL       — instrument                             ║
║    3.  GATE: CONFIDENCE     — regime confidence floor                ║
║    4.  GATE: META LABELLER  — XU-L meta probability floor           ║
║    5.  GATE: CONSECUTIVE    — max repeated WAITs before pause        ║
║    6.  GATE: LONDON         — London session confluence requirement  ║
║    7.  GATE: STACKING       — single position only                  ║
║    8.  GATE: SPREAD         — max allowed spread in $               ║
║    9.  GATE: RR MINIMUM     — minimum risk:reward before execution  ║
║    10. GATE: NEWS WINDOW    — blackout minutes around high-impact   ║
║    11. GATE: FRIDAY         — Friday cutoff time (NY)               ║
║    12. GATE: HALLUCINATION  — max entry price deviation from live   ║
║    13. RISK: BASE RISK      — base position size (% of balance)     ║
║    14. RISK: DAILY DRAWDOWN — max daily loss cap (% of balance)     ║
║    15. RISK: CONSECUTIVE    — max consecutive losses before halt     ║
║    16. RISK: FLOOR          — minimum risk % (never goes below)     ║
║    17. SESSIONS             — active trading windows (NY time)      ║
║    18. CYCLE INTERVAL       — how often the bot polls in seconds    ║
║    19. SHADOW JOURNAL       — forward-tracking bars for gate eval   ║
║    20. STRATEGY SCOUT       — pattern occurrences to propose strat  ║
║    21. WISDOM BUILDER       — days between wisdom rebuilds          ║
║    22. MEMORY CAPS          — rolling caps on memory files          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — AI PROVIDER
# ══════════════════════════════════════════════════════════════════════
# Provider is fixed to Claude. Model can be changed below.
# API keys are set in .env: CLAUDE_API_KEY_1, _2, _3
# ai_client.py auto-rotates across all 3 keys on rate-limit.

AI_PROVIDER = "CLAUDE"

# BUG-21 FIX: Model string verified against Anthropic API as of March 2026.
# Exact accepted strings: https://docs.anthropic.com/en/docs/models-overview
# To override without a code edit, set CLAUDE_MODEL=<model_string> in .env
#
# Claude model options (verified strings):
#   "claude-sonnet-4-6"         ← recommended (balanced speed + intelligence)
#   "claude-opus-4-6"           ← highest intelligence, most expensive
#   "claude-haiku-4-5-20251001" ← fastest, cheapest
import os as _os
CLAUDE_MODEL = _os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Active model (resolved automatically — do not change this line)
AI_MODEL = CLAUDE_MODEL

# Environment variable names for the 3 rotating API keys
CLAUDE_API_KEY_ENV_1 = "CLAUDE_API_KEY_1"
CLAUDE_API_KEY_ENV_2 = "CLAUDE_API_KEY_2"
CLAUDE_API_KEY_ENV_3 = "CLAUDE_API_KEY_3"

# Legacy single-key fallback (used if _1/_2/_3 not set)
CLAUDE_API_KEY_ENV = "CLAUDE_API_KEY"

# Display name used in logs and boot screen
AI_DISPLAY_NAME = "Claude"




# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — TRADING SYMBOL
# ══════════════════════════════════════════════════════════════════════

SYMBOL = "XAUUSD"


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — GATE: CONFIDENCE (pre-AI)
# ══════════════════════════════════════════════════════════════════════
GATE_MIN_CONFIDENCE = 0.45


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — GATE: META LABELLER
# ══════════════════════════════════════════════════════════════════════
META_MIN_THRESHOLD = 0.55


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — GATE: CONSECUTIVE WAITS
# ══════════════════════════════════════════════════════════════════════
GATE_MAX_CONSECUTIVE_WAITS = 3


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — GATE: LONDON SESSION CONFLUENCE
# ══════════════════════════════════════════════════════════════════════
GATE_LONDON_MIN_CONFLUENCE = 3


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — GATE: STACKING
# ══════════════════════════════════════════════════════════════════════
ALLOW_STACKING = False


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — GATE: SPREAD LIMIT
# ══════════════════════════════════════════════════════════════════════
GATE_MAX_SPREAD_DOLLARS = 1.50


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — GATE: MINIMUM RISK:REWARD
# ══════════════════════════════════════════════════════════════════════
GATE_MIN_RR = 1.5


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — GATE: NEWS WINDOW
# ══════════════════════════════════════════════════════════════════════
NEWS_BLOCK_BEFORE_MINUTES = 30
NEWS_BLOCK_AFTER_MINUTES  = 15


# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — GATE: FRIDAY CUTOFF
# ══════════════════════════════════════════════════════════════════════
FRIDAY_CLOSE_HOUR    = 17
FRIDAY_NO_TRADE_MINS = 15


# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — GATE: HALLUCINATION GUARD
# ══════════════════════════════════════════════════════════════════════
GATE_MAX_ENTRY_DEVIATION = 20.0


# ══════════════════════════════════════════════════════════════════════
# SECTION 13 — RISK: BASE POSITION SIZE
# ══════════════════════════════════════════════════════════════════════
RISK_BASE_PCT = 0.01


# ══════════════════════════════════════════════════════════════════════
# SECTION 14 — RISK: DAILY DRAWDOWN CAP
# ══════════════════════════════════════════════════════════════════════
RISK_DAILY_DRAWDOWN_PCT = 0.015


# ══════════════════════════════════════════════════════════════════════
# SECTION 15 — RISK: CONSECUTIVE LOSS HALT
# ══════════════════════════════════════════════════════════════════════
RISK_MAX_CONSECUTIVE_LOSSES = 3


# ══════════════════════════════════════════════════════════════════════
# SECTION 16 — RISK: MINIMUM RISK FLOOR
# ══════════════════════════════════════════════════════════════════════
RISK_MIN_PCT = 0.002


# ══════════════════════════════════════════════════════════════════════
# SECTION 17 — SESSIONS
# ══════════════════════════════════════════════════════════════════════
# BUG-12 FIX: SESSIONS and ACTIVE_SESSIONS were defined here but NEVER
# imported by main_bot.py. main_bot.py used its own hard-coded ACTIVE_WINDOWS
# list, meaning edits to SESSIONS had zero effect on live trading behaviour.
#
# Resolution: ACTIVE_WINDOWS in main_bot.py now imports from master_controls.
# Edit the windows below — changes propagate to the live bot automatically.
#
# Format: {"session_name": {"start": "HH:MM", "end": "HH:MM", "label": str}}
# Asian spans midnight — main_bot.py handles the two-window split internally.
SESSIONS = {
    "Asian":  {"start": "19:00", "end": "01:00",  "label": "Asian"},   # 19:00–23:59 + 00:00–00:59 (end is EXCLUSIVE)
    "London": {"start": "02:00", "end": "05:00",  "label": "London"},  # 02:00–04:59
    "NY":     {"start": "07:00", "end": "12:00",  "label": "NY_AM"},   # 07:00–11:59
    # Dead zone: 13:00–19:00 — no AI calls, no new entries.
    # Off-session gaps (01:00–02:00, 05:00–07:00, 12:00–13:00):
    #   if trade OPEN  → overnight/off-session regime guard runs (zero API cost)
    #   if no trade    → bot sleeps until next active window
}
ACTIVE_SESSIONS = ["Asian", "London", "NY"]


# ══════════════════════════════════════════════════════════════════════
# SECTION 18 — CYCLE INTERVAL
# ══════════════════════════════════════════════════════════════════════
CYCLE_INTERVAL_SECONDS = 300


# ══════════════════════════════════════════════════════════════════════
# SECTION 19 — SHADOW JOURNAL
# ══════════════════════════════════════════════════════════════════════
SHADOW_FORWARD_BARS = 48


# ══════════════════════════════════════════════════════════════════════
# SECTION 20 — STRATEGY SCOUT
# ══════════════════════════════════════════════════════════════════════
SCOUT_PATTERN_THRESHOLD = 10


# ══════════════════════════════════════════════════════════════════════
# SECTION 21 — WISDOM BUILDER
# ══════════════════════════════════════════════════════════════════════
WISDOM_REBUILD_DAYS = 5


# ══════════════════════════════════════════════════════════════════════
# SECTION 22 — MEMORY CAPS
# ══════════════════════════════════════════════════════════════════════
MEMORY_CAP_KEYWORDS    = 5000
MEMORY_CAP_MISSWISH_KW = 5000
MEMORY_CAP_MISSWISH    = 10000
MEMORY_CAP_SHADOW      = 20000


# ══════════════════════════════════════════════════════════════════════
# VALIDATION — runs at import time
# FIX O4: Only validates the active provider's key(s).
# ══════════════════════════════════════════════════════════════════════

def _validate():
    errors = []

    if not (0.30 <= GATE_MIN_CONFIDENCE <= 0.90):
        errors.append(f"GATE_MIN_CONFIDENCE {GATE_MIN_CONFIDENCE} out of range [0.30–0.90]")
    if not (0.40 <= META_MIN_THRESHOLD <= 0.80):
        errors.append(f"META_MIN_THRESHOLD {META_MIN_THRESHOLD} out of range [0.40–0.80]")
    if not (0 < RISK_BASE_PCT <= 0.05):
        errors.append(f"RISK_BASE_PCT {RISK_BASE_PCT} out of range (0–0.05]")
    if not (0.005 <= RISK_DAILY_DRAWDOWN_PCT <= 0.10):
        errors.append(f"RISK_DAILY_DRAWDOWN_PCT {RISK_DAILY_DRAWDOWN_PCT} out of range [0.005–0.10]")
    if RISK_MIN_PCT >= RISK_BASE_PCT:
        errors.append(f"RISK_MIN_PCT {RISK_MIN_PCT} must be less than RISK_BASE_PCT {RISK_BASE_PCT}")
    if GATE_MIN_RR < 1.0:
        errors.append(f"GATE_MIN_RR {GATE_MIN_RR} must be ≥ 1.0")

    # FIX O4: Only check Claude keys (the active provider)
    has_any_key = any(
        os.getenv(f"CLAUDE_API_KEY_{i}", "").strip()
        for i in range(1, 4)
    ) or bool(os.getenv("CLAUDE_API_KEY", "").strip())

    if not has_any_key:
        errors.append(
            "No Claude API keys found in .env. "
            "Set at least one of: CLAUDE_API_KEY_1, CLAUDE_API_KEY_2, CLAUDE_API_KEY_3"
        )

    if errors:
        print("\n" + "═" * 60)
        print("  MASTER CONTROLS — CONFIGURATION ERROR(S)")
        print("═" * 60)
        for e in errors:
            print(f"  ✗  {e}")
        print("═" * 60)
        raise ValueError(
            f"master_controls: {len(errors)} config error(s). Fix before starting."
        )


_validate()
