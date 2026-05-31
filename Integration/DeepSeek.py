"""
DeepSeek.py - Direct DeepSeek API Provider Adapter
==================================================
Self-contained adapter for routing Selected backtest AI calls directly through
DeepSeek's official OpenAI-compatible endpoint.
"""

import concurrent.futures
import os
import sys
import threading
import time

from dotenv import load_dotenv

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_STABILITY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Stability")
if _STABILITY_DIR not in sys.path:
    sys.path.insert(0, _STABILITY_DIR)

from console_display import print_critical, print_outage_banner

load_dotenv()

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_DISPLAY_NAME = "DeepSeek V3"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_CALL_TIMEOUT_SECONDS = 120.0
_CB_FAIL_THRESHOLD = 1   # Trip immediately on 1st billing/permanent failure; NIM will take over
_CB_COOLDOWN_SECONDS = 300.0

_cb_lock = threading.Lock()
_cb_consecutive_fails = 0
_cb_tripped = False
_cb_tripped_at = 0.0

_client = None
_client_lock = threading.Lock()

__all__ = ["call_deepseek", "DEEPSEEK_MODEL", "DEEPSEEK_DISPLAY_NAME", "is_deepseek_available"]


def _is_backtest_mode() -> bool:
    return os.getenv("BACKTEST_MODE") == "1"


def _record_failure(reason: str) -> None:
    global _cb_consecutive_fails, _cb_tripped, _cb_tripped_at
    with _cb_lock:
        _cb_consecutive_fails += 1
        if _cb_consecutive_fails >= _CB_FAIL_THRESHOLD and not _cb_tripped:
            _cb_tripped = True
            _cb_tripped_at = time.time()
            print(
                f"[DeepSeek] Circuit breaker TRIPPED after "
                f"{_cb_consecutive_fails} consecutive failures. "
                f"DeepSeek paused for {_CB_COOLDOWN_SECONDS/60:.0f} minutes."
            )
            print_critical(f"DEEPSEEK CIRCUIT BREAKER TRIPPED - {reason}")


def _record_success() -> None:
    global _cb_consecutive_fails, _cb_tripped
    with _cb_lock:
        if _cb_consecutive_fails > 0 or _cb_tripped:
            print("[DeepSeek] Circuit breaker reset after successful call.")
        _cb_consecutive_fails = 0
        _cb_tripped = False


def is_deepseek_available() -> bool:
    global _cb_consecutive_fails, _cb_tripped
    with _cb_lock:
        if not _cb_tripped:
            return True

        elapsed = time.time() - _cb_tripped_at
        if elapsed >= _CB_COOLDOWN_SECONDS:
            _cb_tripped = False
            _cb_consecutive_fails = 0
            print("[DeepSeek] Circuit breaker auto-reset after cooldown. Resuming calls.")
            return True

        remaining = _CB_COOLDOWN_SECONDS - elapsed
        print(f"[DeepSeek] Circuit breaker active. Resuming in {remaining:.0f}s")
        return False


def _get_client():
    global _client
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print_critical("DEEPSEEK_API_KEY missing - add it to .env before using DeepSeek")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("[DeepSeek] openai package not installed. Run: pip install openai")
        return None

    with _client_lock:
        if _client is None:
            _client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        return _client


def call_deepseek(prompt: str, max_tokens: int = 2048) -> str | None:
    if not is_deepseek_available():
        print_outage_banner()
        return None

    client = _get_client()
    if client is None:
        _record_failure("DeepSeek client unavailable")
        if _is_backtest_mode():
            time.sleep(1.0)
        return None

    def _do_call():
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a programmatic trading engine API. You MUST output ONLY valid JSON, "
                    "exactly matching the requested schema. Do NOT output any introductory text, "
                    "conversational explanations, or markdown fences. Just return the raw JSON object."
                )
            },
            {"role": "user", "content": prompt}
        ]
        return client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

    max_retries = 5
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_call)
                response = future.result(timeout=_CALL_TIMEOUT_SECONDS)

            content = response.choices[0].message.content
            _record_success()
            return content.strip() if content else ""

        except concurrent.futures.TimeoutError:
            print(f"[DeepSeek] Call timed out after {_CALL_TIMEOUT_SECONDS}s.")
            _record_failure("DeepSeek timeout")
            return None

        except Exception as exc:
            exc_name = type(exc).__name__
            exc_str = str(exc)

            # 402 Insufficient Balance = permanent billing block, trip immediately, no retries
            is_billing_error = "402" in exc_str or "insufficient balance" in exc_str.lower()
            if is_billing_error:
                print(f"[DeepSeek] Billing block (402 Insufficient Balance). "
                      f"Tripping circuit breaker immediately — will fall back to NIM.")
                _record_failure("402 Insufficient Balance")
                return None

            # Check for 429 rate limit or 503 service congestion
            is_rate_limit = "429" in exc_str or "RateLimit" in exc_name or "503" in exc_str

            if is_rate_limit and attempt < max_retries - 1:
                sleep_time = backoff * (1.5 ** attempt)
                print(f"[DeepSeek] Rate-limit or congestion ({exc_name}) on attempt {attempt+1}/{max_retries}. "
                      f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            print(f"[DeepSeek] Call failed: {exc_name}: {exc}")
            _record_failure(exc_str[:160])
            return None

        finally:
            if _is_backtest_mode():
                time.sleep(0.5)


print(f"[DeepSeek] ✓ {DEEPSEEK_MODEL} | Direct DeepSeek endpoint ready")
