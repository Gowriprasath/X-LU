"""
ai_client.py - AI Provider Adapter (Claude Sonnet 4.6 + 3-Key Rotation)
=========================================================================
Single import point for ALL AI calls in the bot.

PROVIDER: Claude Sonnet 4.6 - v16 (claude-sonnet-4-6)

3-KEY AUTO-ROTATION:
  Reads CLAUDE_API_KEY_1, CLAUDE_API_KEY_2, CLAUDE_API_KEY_3 from .env.
  On any rate-limit / overloaded error, silently switches to the next key
  and retries - zero manual intervention needed.
  Rotation order: Key 1 -> Key 2 -> Key 3 -> Key 1 (wraps around).
  The "current" key index is remembered so healthy keys aren't churned.

Usage (same everywhere in the bot):
    from ai_client import call_ai, AI_MODEL, AI_DISPLAY_NAME

    response_text = call_ai(prompt="Your prompt here")
    # Returns plain string on success, None on total failure.

For files that need the model name only:
    from ai_client import AI_MODEL
"""

import os
import sys
import time
import threading
import concurrent.futures
from dotenv import load_dotenv

_STABILITY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Stability")
if _STABILITY_DIR not in sys.path:
    sys.path.insert(0, _STABILITY_DIR)

from console_display import (
    print_critical,
    print_outage_banner,
    print_recovery_banner,
)
from telegram_notifier import (
    send_outage_alert,
    send_recovery_alert,
)

load_dotenv()

# -- Provider Configuration --
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
USE_GEMINI = os.getenv("USE_GEMINI", "False").strip().lower() == "true"
USE_DEEPSEEK = os.getenv("USE_DEEPSEEK", "False").strip().lower() == "true"
USE_NIM = os.getenv("USE_NIM", "False").strip().lower() == "true"

if USE_DEEPSEEK:
    AI_PROVIDER = "DEEPSEEK"
    AI_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    AI_DISPLAY_NAME = "DeepSeek Direct"
elif USE_NIM:
    AI_PROVIDER = "NIM"
    AI_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct")
    AI_DISPLAY_NAME = "NVIDIA NIM"
elif USE_GEMINI:
    AI_PROVIDER = "GEMINI"
    AI_MODEL = "gemini-2.5-flash"
    AI_DISPLAY_NAME = "Gemini"
else:
    AI_PROVIDER = "CLAUDE"
    try:
        from master_controls import CLAUDE_MODEL as AI_MODEL
    except ImportError:
        AI_MODEL = "claude-sonnet-4-6"
    AI_DISPLAY_NAME = "Claude"

_key_lock = threading.Lock()
_current_key_idx = 0          # which key slot we're currently on
# BUG-8 FIX: _key_fail_count was hardcoded to [0, 0, 0] (exactly 3 slots).
# If a 4th key is added or one is removed, the index goes out of bounds.
# Now initialised lazily inside call_ai() based on actual key count.
# Using a dict keyed by index is the simplest collision-proof approach.
_key_fail_count: dict = {}    # {key_index: consecutive_fail_count}
_MAX_KEY_FAILS = 3            # mark a key as "skip" after this many consecutive fails
_RETRY_SLEEP = 2.0            # seconds between retries on rate-limit

# B-10 FIX: cache one Anthropic client per key index instead of creating a new
# connection pool on every call. Over 288 calls/day this prevents port exhaustion
# and TLS handshake overhead. Clients are created lazily and reused.
_client_pool: dict = {}       # {key_index: client}

# -- Circuit Breaker State ------------------------------------------------
_cb_lock = threading.Lock()
_cb_consecutive_fails: int = 0
_cb_tripped: bool = False
_cb_tripped_at: float = 0.0      # time.time()
_CB_FAIL_THRESHOLD: int = 3      # trips after N fails
_CB_COOLDOWN_SECONDS: float = 900.0  # 15 minutes

# -- Call Timeout ---------------------------------------------------------
_CALL_TIMEOUT_SECONDS: float = 150.0


def _load_keys() -> list:
    """
    Reads the 3 Claude API keys from environment.
    Falls back to single CLAUDE_API_KEY if the numbered ones aren't set.
    Always returns a non-empty list (raises if truly nothing configured).
    """
    if USE_GEMINI:
        if not GEMINI_API_KEY:
            raise EnvironmentError("[ai_client] GEMINI_API_KEY is empty in .env.")
        return [GEMINI_API_KEY]

    keys = []
    for i in range(1, 4):
        k = os.getenv(f"CLAUDE_API_KEY_{i}", "").strip()
        if k and k != "your_claude_api_key_1_here" \
               and k != "your_claude_api_key_2_here" \
               and k != "your_claude_api_key_3_here":
            keys.append(k)

    # Legacy single-key fallback
    if not keys:
        k = os.getenv("CLAUDE_API_KEY", "").strip()
        if k:
            keys.append(k)

    if not keys:
        raise EnvironmentError(
            "[ai_client] No Claude API keys found.\n"
            "Set CLAUDE_API_KEY_1 / _2 / _3 in your .env file."
        )
    return keys


def _is_rate_limit_error(exc) -> bool:
    """
    Detects errors that should trigger key rotation rather than hard failure.

    Rotatable errors:
      - 429 rate-limit / too many requests
      - 529 overloaded
      - 400 credit balance too low

    Non-rotatable errors (genuine hard failures - bad prompt, auth error
    on all keys, etc.) still return False so the caller gets None fast.
    """
    try:
        import anthropic
        if isinstance(exc, (anthropic.RateLimitError, anthropic.APIStatusError)):
            msg = str(exc).lower()
            # Rate-limit / capacity errors -> rotate
            if any(t in msg for t in ("rate_limit", "rate limit", "429",
                                       "overloaded", "529", "too many")):
                return True
            # Credit / billing errors -> rotate to next key
            # (the next key may belong to a different account with credit)
            if any(t in msg for t in ("credit balance", "too low",
                                       "billing", "payment", "quota",
                                       "insufficient")):
                return True
            # Check numeric status codes directly
            if hasattr(exc, "status_code") and exc.status_code in (429, 529):
                return True
    except ImportError:
        pass
    return False


# ================================================================
# UNIFIED CALL WITH 3-KEY ROTATION
# ================================================================

_backtest_mode: bool = False

def is_backtest_mode() -> bool:
    global _backtest_mode
    return _backtest_mode or os.environ.get("BACKTEST_MODE") == "1"

def set_backtest_mode(enabled: bool) -> None:
    global _backtest_mode
    _backtest_mode = enabled
    if enabled:
        print("[ai_client] Backtest mode — circuit breaker cooldown: 30s")

def is_ai_available() -> bool:
    global _cb_tripped, _cb_consecutive_fails
    with _cb_lock:
        if not _cb_tripped:
            return True

        # Check if cooldown has expired
        elapsed = time.time() - _cb_tripped_at
        active_cooldown = 30.0 if is_backtest_mode() else _CB_COOLDOWN_SECONDS
        if elapsed >= active_cooldown:
            # Auto-reset the circuit breaker
            _cb_tripped = False
            _cb_consecutive_fails = 0
            print("[ai_client] ✓ Circuit breaker auto-reset after cooldown. Resuming AI calls.")
            send_recovery_alert()
            print_recovery_banner()
            return True

        remaining = active_cooldown - elapsed
        print(f"[ai_client] Circuit breaker active. Resuming in {remaining:.0f}s")
        return False


def get_ai_status() -> dict:
    with _cb_lock:
        remaining = 0.0
        if _cb_tripped:
            remaining = max(
                0.0,
                _CB_COOLDOWN_SECONDS - (time.time() - _cb_tripped_at)
            )
        return {
            "available": not _cb_tripped,
            "circuit_breaker": _cb_tripped,
            "consecutive_fails": _cb_consecutive_fails,
            "cooldown_remaining": remaining,
        }


def _generate_mock_decision(prompt: str) -> str:
    import re
    import json
    import hashlib

    # Default values
    current_price = 1800.0
    session = "London"
    regime = "LOW_VOL_RANGE"
    time_str = "unknown"

    # Parse current price
    price_match = re.search(r"Current Price\s*:\s*([\d.]+)", prompt, re.IGNORECASE)
    if price_match:
        current_price = float(price_match.group(1))

    # Parse session
    session_match = re.search(r"Current session\s*:\s*(\w+)", prompt, re.IGNORECASE)
    if session_match:
        session = session_match.group(1)

    # Parse simulated time
    time_match = re.search(r"Simulated Time\s*:\s*([\d-]+) ([\d:]+)", prompt, re.IGNORECASE)
    if time_match:
        time_str = f"{time_match.group(1)} {time_match.group(2)}"
    else:
        # Fallback to a hash of the prompt itself to be deterministic
        time_str = str(len(prompt))

    # Detect regime
    # Look for the regime name in the prompt
    for r in ["BULL_TREND", "BEAR_TREND", "LOW_VOL_RANGE", "COMPRESSION", "REVERSAL"]:
        if r in prompt:
            regime = r
            break

    # Generate a deterministic hash for decision making
    h = int(hashlib.md5(time_str.encode('utf-8')).hexdigest(), 16)

    decision = {
        "signal": "WAIT",
        "confluence_score": 1,
        "rationale": "Default mock wait state."
    }

    # We want a variety of outcomes based on h % 30
    outcome_key = h % 30

    # Let's map outcomes:
    # 0, 1, 2, 3 -> Valid Buy
    # 4 -> Invalid Buy (R:R low)
    # 5 -> Invalid Buy (wrong levels)
    # 6 -> Invalid Buy (hallucinated entry price)
    # 7, 8, 9, 10 -> Valid Sell
    # 11 -> Invalid Sell (R:R low)
    # 12 -> Invalid Sell (wrong levels)
    # 13 -> Invalid Sell (hallucinated entry price)
    # 14, 15 -> HOLD
    # 16, 17 -> CLOSE
    # 18 -> Compression Buy (tight target)
    # 19 -> Compression Sell (tight target)
    # 20 -> BUY_LIMIT (Pending, valid)
    # 21 -> BUY_LIMIT (Pending, invalid R:R)
    # 22 -> SELL_LIMIT (Pending, valid)
    # 23 -> SELL_LIMIT (Pending, invalid levels layout)
    # 24 -> BUY_STOP (Pending, valid)
    # 25 -> BUY_STOP (Pending, hallucinated entry)
    # 26 -> SELL_STOP (Pending, valid)
    # 27 -> SELL_STOP (Pending, invalid R:R)
    # 28, 29 -> WAIT

    if outcome_key in [0, 1, 2, 3]:
        # Valid BUY setup
        decision = {
            "signal": "BUY",
            "confluence_score": 3 if (h % 2 == 0) else 2,
            "entry": current_price,
            "sl": round(current_price - 8.0, 2),
            "tp": round(current_price + 18.0, 2),
            "rationale": f"Deterministic mock valid BUY setup in {regime} regime | {session} session."
        }
    elif outcome_key == 4:
        # Invalid BUY (R:R below 1.5)
        decision = {
            "signal": "BUY",
            "confluence_score": 3,
            "entry": current_price,
            "sl": round(current_price - 10.0, 2),
            "tp": round(current_price + 8.0, 2), # R:R = 0.8 < 1.5
            "rationale": f"Deterministic mock invalid R:R BUY setup (hallucination guard test)."
        }
    elif outcome_key == 5:
        # Invalid BUY (wrong levels: sl > entry)
        decision = {
            "signal": "BUY",
            "confluence_score": 3,
            "entry": current_price,
            "sl": round(current_price + 5.0, 2), # sl is above entry
            "tp": round(current_price + 15.0, 2),
            "rationale": f"Deterministic mock invalid level layout BUY setup."
        }
    elif outcome_key == 6:
        # Invalid BUY (entry far from current_price: hallucination)
        decision = {
            "signal": "BUY",
            "confluence_score": 3,
            "entry": round(current_price + 35.0, 2), # entry is >20 pts away
            "sl": round(current_price + 27.0, 2),
            "tp": round(current_price + 55.0, 2),
            "rationale": f"Deterministic mock hallucinated entry price BUY setup."
        }
    elif outcome_key in [7, 8, 9, 10]:
        # Valid SELL setup
        decision = {
            "signal": "SELL",
            "confluence_score": 3 if (h % 2 == 0) else 2,
            "entry": current_price,
            "sl": round(current_price + 8.0, 2),
            "tp": round(current_price - 18.0, 2),
            "rationale": f"Deterministic mock valid SELL setup in {regime} regime | {session} session."
        }
    elif outcome_key == 11:
        # Invalid SELL (R:R below 1.5)
        decision = {
            "signal": "SELL",
            "confluence_score": 3,
            "entry": current_price,
            "sl": round(current_price + 10.0, 2),
            "tp": round(current_price - 8.0, 2), # R:R = 0.8 < 1.5
            "rationale": f"Deterministic mock invalid R:R SELL setup (hallucination guard test)."
        }
    elif outcome_key == 12:
        # Invalid SELL (wrong levels: sl < entry)
        decision = {
            "signal": "SELL",
            "confluence_score": 3,
            "entry": current_price,
            "sl": round(current_price - 5.0, 2), # sl is below entry
            "tp": round(current_price - 15.0, 2),
            "rationale": f"Deterministic mock invalid level layout SELL setup."
        }
    elif outcome_key == 13:
        # Invalid SELL (entry far from current_price)
        decision = {
            "signal": "SELL",
            "confluence_score": 3,
            "entry": round(current_price - 35.0, 2), # entry is >20 pts away
            "sl": round(current_price - 27.0, 2),
            "tp": round(current_price - 55.0, 2),
            "rationale": f"Deterministic mock hallucinated entry price SELL setup."
        }
    elif outcome_key in [14, 15]:
        decision = {
            "signal": "HOLD",
            "confluence_score": 1,
            "rationale": f"Deterministic mock HOLD signal generated in {regime}."
        }
    elif outcome_key in [16, 17]:
        decision = {
            "signal": "CLOSE",
            "confluence_score": 2,
            "rationale": f"Deterministic mock CLOSE signal generated in {regime}."
        }
    elif outcome_key == 18:
        # Compression BUY setup (designed to hit partials and tight break-evens easily)
        decision = {
            "signal": "BUY",
            "confluence_score": 3,
            "entry": current_price,
            "sl": round(current_price - 6.0, 2),
            "tp": round(current_price + 12.0, 2),
            "rationale": f"Deterministic mock compression BUY setup in {regime} regime."
        }
    elif outcome_key == 19:
        # Compression SELL setup (designed to hit partials and tight break-evens easily)
        decision = {
            "signal": "SELL",
            "confluence_score": 3,
            "entry": current_price,
            "sl": round(current_price + 6.0, 2),
            "tp": round(current_price - 12.0, 2),
            "rationale": f"Deterministic mock compression SELL setup in {regime} regime."
        }
    elif outcome_key == 20:
        # Valid BUY_LIMIT order (entry below market price)
        entry_price = round(current_price - 4.0, 2)
        decision = {
            "signal": "BUY_LIMIT",
            "confluence_score": 3,
            "entry": entry_price,
            "sl": round(entry_price - 8.0, 2),
            "tp": round(entry_price + 18.0, 2),
            "rationale": f"Deterministic mock valid BUY_LIMIT setup in {regime} regime | {session} session."
        }
    elif outcome_key == 21:
        # Invalid BUY_LIMIT order (R:R below 1.5)
        entry_price = round(current_price - 4.0, 2)
        decision = {
            "signal": "BUY_LIMIT",
            "confluence_score": 3,
            "entry": entry_price,
            "sl": round(entry_price - 10.0, 2),
            "tp": round(entry_price + 5.0, 2), # R:R low
            "rationale": f"Deterministic mock invalid R:R BUY_LIMIT setup (safety test)."
        }
    elif outcome_key == 22:
        # Valid SELL_LIMIT order (entry above market price)
        entry_price = round(current_price + 4.0, 2)
        decision = {
            "signal": "SELL_LIMIT",
            "confluence_score": 3,
            "entry": entry_price,
            "sl": round(entry_price + 8.0, 2),
            "tp": round(entry_price - 18.0, 2),
            "rationale": f"Deterministic mock valid SELL_LIMIT setup in {regime} regime | {session} session."
        }
    elif outcome_key == 23:
        # Invalid SELL_LIMIT order (wrong levels: sl is below entry)
        entry_price = round(current_price + 4.0, 2)
        decision = {
            "signal": "SELL_LIMIT",
            "confluence_score": 3,
            "entry": entry_price,
            "sl": round(entry_price - 5.0, 2),
            "tp": round(entry_price - 15.0, 2),
            "rationale": f"Deterministic mock invalid levels SELL_LIMIT setup."
        }
    elif outcome_key == 24:
        # Valid BUY_STOP order (entry above market price)
        entry_price = round(current_price + 4.0, 2)
        decision = {
            "signal": "BUY_STOP",
            "confluence_score": 3,
            "entry": entry_price,
            "sl": round(entry_price - 8.0, 2),
            "tp": round(entry_price + 18.0, 2),
            "rationale": f"Deterministic mock valid BUY_STOP setup in {regime} regime | {session} session."
        }
    elif outcome_key == 25:
        # Invalid BUY_STOP order (hallucinated entry price too far)
        decision = {
            "signal": "BUY_STOP",
            "confluence_score": 3,
            "entry": round(current_price + 35.0, 2),
            "sl": round(current_price + 27.0, 2),
            "tp": round(current_price + 55.0, 2),
            "rationale": f"Deterministic mock hallucinated entry price BUY_STOP setup."
        }
    elif outcome_key == 26:
        # Valid SELL_STOP order (entry below market price)
        entry_price = round(current_price - 4.0, 2)
        decision = {
            "signal": "SELL_STOP",
            "confluence_score": 3,
            "entry": entry_price,
            "sl": round(entry_price + 8.0, 2),
            "tp": round(entry_price - 18.0, 2),
            "rationale": f"Deterministic mock valid SELL_STOP setup in {regime} regime | {session} session."
        }
    elif outcome_key == 27:
        # Invalid SELL_STOP order (R:R below 1.5)
        entry_price = round(current_price - 4.0, 2)
        decision = {
            "signal": "SELL_STOP",
            "confluence_score": 3,
            "entry": entry_price,
            "sl": round(entry_price + 10.0, 2),
            "tp": round(entry_price - 8.0, 2),
            "rationale": f"Deterministic mock invalid R:R SELL_STOP setup."
        }
    else:
        decision = {
            "signal": "WAIT",
            "confluence_score": 1,
            "rationale": f"Deterministic mock WAIT setup in {regime} regime during {session} session."
        }

    # Wrap the decision dictionary as a JSON block
    return json.dumps(decision, indent=2)


def call_ai(prompt: str, max_tokens: int = 2048) -> str | None:
    """
    Makes a single-turn Claude or Gemini call. Auto-rotates across API keys
    on rate-limit or overloaded errors. Returns plain string or None.


    Args:


    Args:
        prompt:     Full prompt text.
        max_tokens: Max response tokens (default 2048).

    Returns:
        Response text string, or None on total failure.
    """
    if is_backtest_mode():
        if os.getenv("BACKTEST_AI_MODE") == "real":
            provider = os.getenv("BACKTEST_AI_PROVIDER", "mock").strip().lower()
            if provider == "nim":
                from Integration.NIM import call_nim
                return call_nim(prompt, max_tokens)
            elif provider == "deepseek":
                from Integration.DeepSeek import call_deepseek
                res = call_deepseek(prompt, max_tokens)
                if res is None:
                    # Self-healing fallback to NVIDIA NIM
                    print("[ai_client] 🔄 Backtest DeepSeek failed/exhausted. Falling back to NVIDIA NIM...")
                    from Integration.NIM import call_nim
                    return call_nim(prompt, max_tokens)
                return res
            elif provider in ("claude", "gemini"):
                # Bypass mock generator and proceed to real API execution logic below
                pass
            else:
                return _generate_mock_decision(prompt)
        else:
            return _generate_mock_decision(prompt)

    global _current_key_idx, _cb_consecutive_fails, _cb_tripped, _cb_tripped_at

    # -- Circuit breaker guard --
    if not is_ai_available():
        print_outage_banner()
        return None

    if USE_DEEPSEEK:
        from Integration.DeepSeek import call_deepseek
        res = call_deepseek(prompt, max_tokens)
        if res is not None:
            return res
        print("[ai_client] 🔄 Live DeepSeek failed/exhausted. Trying fallback to NVIDIA NIM...")
        from Integration.NIM import call_nim
        res = call_nim(prompt, max_tokens)
        if res is not None:
            return res
        print("[ai_client] 🔄 Live NIM fallback failed. Trying fallback to Claude keys...")
        # fall through to Claude

    elif USE_NIM:
        from Integration.NIM import call_nim
        res = call_nim(prompt, max_tokens)
        if res is not None:
            return res
        print("[ai_client] 🔄 Live NIM failed. Trying fallback to Claude keys...")
        # fall through to Claude

    if USE_GEMINI:
        try:
            from google import genai
            from google.genai import types
            with _key_lock:
                if "gemini" not in _client_pool:
                    _client_pool["gemini"] = genai.Client(
                        api_key=GEMINI_API_KEY,
                        http_options=types.HttpOptions(timeout=150000)
                    )
                client = _client_pool["gemini"]

            def _do_call(model_name):
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

            # Try primary model first (AI_MODEL, which is gemini-3.5-flash)
            success = False
            response = None
            last_err = None

            # Set max_workers=2 so that the fallback call is never blocked in queue by a slow/hanging primary call
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # 1. Primary call
                future = executor.submit(_do_call, AI_MODEL)
                try:
                    response = future.result(timeout=_CALL_TIMEOUT_SECONDS)
                    success = True
                except concurrent.futures.TimeoutError:
                    print(f"[ai_client] ⚠ Gemini primary ({AI_MODEL}) timed out after {_CALL_TIMEOUT_SECONDS}s.")
                    last_err = "TimeoutError"
                except Exception as exc:
                    print(f"[ai_client] ⚠ Gemini primary ({AI_MODEL}) call failed: {type(exc).__name__}: {exc}")
                    last_err = exc

                # 2. Fallback if primary failed
                if not success:
                    fallback_model = "gemini-2.5-flash"
                    print(f"[ai_client] 🔄 Attempting automatic fallback to {fallback_model}...")
                    future_fallback = executor.submit(_do_call, fallback_model)
                    try:
                        response = future_fallback.result(timeout=_CALL_TIMEOUT_SECONDS)
                        success = True
                        print(f"[ai_client] ✓ Fallback to {fallback_model} succeeded!")
                    except concurrent.futures.TimeoutError:
                        print(f"[ai_client] ✗ Gemini fallback ({fallback_model}) also timed out.")
                        last_err = "TimeoutError (fallback)"
                    except Exception as exc:
                        print(f"[ai_client] ✗ Gemini fallback ({fallback_model}) failed: {type(exc).__name__}: {exc}")
                        last_err = exc

            if not success:
                # Both primary and fallback failed
                with _cb_lock:
                    _cb_consecutive_fails += 1
                    if (_cb_consecutive_fails >= _CB_FAIL_THRESHOLD and not _cb_tripped):
                        _cb_tripped = True
                        _cb_tripped_at = time.time()
                        print(f"[ai_client] 🔴 Circuit breaker TRIPPED after {_cb_consecutive_fails} consecutive failures. AI paused.")
                return None

            with _cb_lock:
                if _cb_consecutive_fails > 0 or _cb_tripped:
                    was_tripped = _cb_tripped
                    _cb_consecutive_fails = 0
                    _cb_tripped = False
                    if was_tripped:
                        print("[ai_client] ✓ AI recovered after outage.")
                        send_recovery_alert()
                        print_recovery_banner()

            return response.text.strip()

        except Exception as outer_exc:
            print(f"[ai_client] ✗ Gemini block execution error: {type(outer_exc).__name__}: {outer_exc}")
            return None

    try:
        import anthropic
    except ImportError:
        print("[ai_client] anthropic package not installed. Run: pip install anthropic")
        return None

    keys = _load_keys()
    n = len(keys)

    # BUG-8 FIX: initialise counters for any key slots not yet seen
    with _key_lock:
        for i in range(n):
            if i not in _key_fail_count:
                _key_fail_count[i] = 0
        start_idx = _current_key_idx % n

    _last_err = None
    for attempt in range(n):       # try each key at most once per call
        with _key_lock:
            key_idx = (start_idx + attempt) % n

        key = keys[key_idx]
        key_label = f"Key {key_idx + 1}/{n}"

        try:
            # B-10 FIX: reuse cached client for this key index instead of
            # creating a new Anthropic() (and therefore a new HTTP connection pool)
            # on every single API call.
            with _key_lock:
                if key_idx not in _client_pool:
                    _client_pool[key_idx] = anthropic.Anthropic(api_key=key)
                client = _client_pool[key_idx]

            # -- Timeout wrapper --
            def _do_call():
                return client.messages.create(
                    model=AI_MODEL,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_call)
                try:
                    response = future.result(timeout=_CALL_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    print(f"[ai_client] ⚠ {key_label} timed out "
                          f"after {_CALL_TIMEOUT_SECONDS}s. "
                          f"Rotating to next key...")
                    with _cb_lock:
                        _cb_consecutive_fails += 1
                    time.sleep(_RETRY_SLEEP)
                    continue   # try next key

            # -- SUCCESS --
            with _key_lock:
                _current_key_idx = key_idx
                _key_fail_count[key_idx] = 0

            # -- Circuit breaker: reset on success --
            with _cb_lock:
                if _cb_consecutive_fails > 0 or _cb_tripped:
                    was_tripped = _cb_tripped
                    _cb_consecutive_fails = 0
                    _cb_tripped = False
                    if was_tripped:
                        print("[ai_client] ✓ AI recovered after outage.")
                        send_recovery_alert()
                        print_recovery_banner()

            texts = [block.text for block in response.content if hasattr(block, "text")]
            return "\n".join(texts).strip()

        except Exception as exc:
            _last_err = exc
            if _is_rate_limit_error(exc):
                with _key_lock:
                    _key_fail_count[key_idx] = _key_fail_count.get(key_idx, 0) + 1
                # Identify reason for cleaner logs
                _exc_msg = str(exc).lower()
                if any(t in _exc_msg for t in ("credit", "billing", "payment",
                                                "quota", "insufficient")):
                    _reason = "credit/billing issue"
                else:
                    _reason = "rate-limited/overloaded"
                print(f"[ai_client] ⚠ {key_label} {_reason} "
                      f"(fail #{_key_fail_count[key_idx]}). "
                      f"Rotating to next key...")
                time.sleep(_RETRY_SLEEP)
                continue    # try next key

            else:
                # Non-rate-limit error - log and return None immediately
                print(f"[ai_client] ✗ Call failed ({key_label}): "
                      f"{type(exc).__name__}: {exc}")
                with _cb_lock:
                    _cb_consecutive_fails += 1
                return None

    # All keys exhausted
    print(f"[ai_client] ✗ All {n} API key(s) rate-limited or exhausted. "
          f"Cannot complete AI call. Will retry next cycle.")

    # -- Circuit breaker: count total exhaustion --
    with _cb_lock:
        _cb_consecutive_fails += 1
        if (_cb_consecutive_fails >= _CB_FAIL_THRESHOLD and not _cb_tripped):
            _cb_tripped = True
            _cb_tripped_at = time.time()
            print(
                f"[ai_client] 🔴 Circuit breaker TRIPPED after "
                f"{_cb_consecutive_fails} consecutive failures. "
                f"AI paused for "
                f"{_CB_COOLDOWN_SECONDS/60:.0f} minutes."
            )
            print_critical(
                f"AI CIRCUIT BREAKER TRIPPED — "
                f"paused {_CB_COOLDOWN_SECONDS/60:.0f} min"
            )
            _last_err_str = str(_last_err) if _last_err else "unknown"
            send_outage_alert(
                f"Circuit breaker tripped after "
                f"{_cb_consecutive_fails} consecutive failures. "
                f"Last error: {_last_err_str[:120]}"
            )
            print(
                f"[ai_client] Last API error: {_last_err_str[:200]}"
            )
    return None


# ================================================================
# CLIENT FACTORY (for files that need raw client - not recommended)
# ================================================================

def get_client():
    """
    Returns an Anthropic client using the current best key.
    Prefer call_ai() for all normal use - this is for edge cases only.
    """
    try:
        import anthropic
    except ImportError:
            send_outage_alert(
                f"Circuit breaker tripped after "
                f"{_cb_consecutive_fails} consecutive failures. "
                f"Last error: {_last_err_str[:120]}"
            )
            print(
                f"[ai_client] Last API error: {_last_err_str[:200]}"
            )
    return None


# ================================================================
# CLIENT FACTORY (for files that need raw client - not recommended)
# ================================================================

def get_client():
    """
    Returns an Anthropic client using the current best key.
    Prefer call_ai() for all normal use - this is for edge cases only.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed")

    keys = _load_keys()
    with _key_lock:
        idx = _current_key_idx % len(keys)
    return anthropic.Anthropic(api_key=keys[idx])

# ================================================================
# STARTUP VALIDATION
# ================================================================

def _validate_on_import():
    """Checks keys exist at import time - fails fast before bot starts."""
    try:
        keys = _load_keys()
        try:
            print(f"[ai_client] ✓ {AI_DISPLAY_NAME} ({AI_MODEL}) | "
                  f"{len(keys)} API key(s) loaded.")
        except UnicodeEncodeError:
            print(f"[ai_client] [OK] {AI_DISPLAY_NAME} ({AI_MODEL}) | "
                  f"{len(keys)} API key(s) loaded.")
    except EnvironmentError as e:
        try:
            print(f"[ai_client] ✗ {e}")
        except UnicodeEncodeError:
            print(f"[ai_client] [FAIL] {e}")

_validate_on_import()


if __name__ == "__main__":
    import ai_client as ac

    assert ac.is_ai_available() is True
    print("Test 1 PASSED — AI available by default")

    status = ac.get_ai_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "circuit_breaker" in status
    assert "consecutive_fails" in status
    assert "cooldown_remaining" in status
    assert status["available"] is True
    assert status["circuit_breaker"] is False
    print("Test 2 PASSED — status dict correct")

    with ac._cb_lock:
        ac._cb_consecutive_fails = ac._CB_FAIL_THRESHOLD
        ac._cb_tripped = True
        ac._cb_tripped_at = time.time()
    assert ac.is_ai_available() is False
    print("Test 3 PASSED — circuit breaker trips correctly")

    with ac._cb_lock:
        ac._cb_tripped = True
        ac._cb_tripped_at = time.time() - 1000
    result = ac.is_ai_available()
    assert result is True
    assert ac._cb_tripped is False
    print("Test 4 PASSED — auto-reset after cooldown")

    with ac._cb_lock:
        ac._cb_tripped = True
        ac._cb_tripped_at = time.time()
    result = ac.call_ai("test prompt")
    assert result is None
    print("Test 5 PASSED — call_ai blocked by breaker")

    # Reset breaker after test
    with ac._cb_lock:
        ac._cb_tripped = False
        ac._cb_consecutive_fails = 0

    print("All ai_client tests passed.")
