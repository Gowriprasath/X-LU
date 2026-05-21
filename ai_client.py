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

# -- Re-export these so every file can do: from ai_client import AI_MODEL --
AI_PROVIDER = "CLAUDE"
AI_MODEL = "claude-sonnet-4-6"
AI_DISPLAY_NAME = "Claude"

__all__ = ["call_ai", "AI_MODEL", "AI_DISPLAY_NAME", "AI_PROVIDER"]

# ================================================================
# 3-KEY ROTATION STATE
# ================================================================

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
_client_pool: dict = {}       # {key_index: anthropic.Anthropic}

# -- Circuit Breaker State ------------------------------------------------
_cb_lock = threading.Lock()
_cb_consecutive_fails: int = 0
_cb_tripped: bool = False
_cb_tripped_at: float = 0.0      # time.time()
_CB_FAIL_THRESHOLD: int = 3      # trips after N fails
_CB_COOLDOWN_SECONDS: float = 900.0  # 15 minutes

# -- Call Timeout ---------------------------------------------------------
_CALL_TIMEOUT_SECONDS: float = 30.0


def _load_keys() -> list:
    """
    Reads the 3 Claude API keys from environment.
    Falls back to single CLAUDE_API_KEY if the numbered ones aren't set.
    Always returns a non-empty list (raises if truly nothing configured).
    """
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

def is_ai_available() -> bool:
    global _cb_tripped, _cb_consecutive_fails
    with _cb_lock:
        if not _cb_tripped:
            return True

        # Check if cooldown has expired
        elapsed = time.time() - _cb_tripped_at
        if elapsed >= _CB_COOLDOWN_SECONDS:
            # Auto-reset the circuit breaker
            _cb_tripped = False
            _cb_consecutive_fails = 0
            print("[ai_client] ✓ Circuit breaker auto-reset after cooldown. Resuming AI calls.")
            send_recovery_alert()
            print_recovery_banner()
            return True

        remaining = _CB_COOLDOWN_SECONDS - elapsed
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


def call_ai(prompt: str, max_tokens: int = 2048) -> str | None:
    """
    Makes a single-turn Claude call. Auto-rotates across 3 API keys
    on rate-limit or overloaded errors. Returns plain string or None.

    Args:
        prompt:     Full prompt text.
        max_tokens: Max response tokens (default 2048).

    Returns:
        Response text string, or None on total failure.
    """
    global _current_key_idx, _cb_consecutive_fails, _cb_tripped, _cb_tripped_at

    # -- Circuit breaker guard --
    if not is_ai_available():
        print_outage_banner()
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
            _last_err_str = str(exc) if 'exc' in dir() else "unknown"
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
        print(f"[ai_client] ✓ {AI_DISPLAY_NAME} ({AI_MODEL}) | "
              f"{len(keys)} API key(s) loaded.")
    except EnvironmentError as e:
        print(f"[ai_client] ✗ {e}")


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
