"""
NIM.py - NVIDIA NIM Provider Adapter (DeepSeek V4 Pro)
======================================================
Self-contained adapter for routing selected backtest AI calls through
NVIDIA NIM's OpenAI-compatible endpoint.
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

NIM_MODEL = os.getenv("NIM_MODEL", "deepseek-ai/deepseek-v4-pro")
NIM_DISPLAY_NAME = "NVIDIA NIM"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

_THINKING_MODES = {
    "none": None,
    "non_think": {"enable_thinking": False},
    "think_high": {"enable_thinking": True, "thinking_budget": 4096},
    "think_max": {"enable_thinking": True, "thinking_budget": 16384},
}

NIM_THINKING_MODE = os.getenv("NIM_THINKING_MODE", "think_high").strip().lower()
if NIM_THINKING_MODE not in _THINKING_MODES:
    print(f"[NIM] Unknown NIM_THINKING_MODE={NIM_THINKING_MODE!r}; using think_high")
    NIM_THINKING_MODE = "think_high"

_CALL_TIMEOUT_SECONDS = 150.0
_CB_FAIL_THRESHOLD = 3
_CB_COOLDOWN_SECONDS = 180.0  # 3 min cooldown in backtest (not 15)

_cb_lock = threading.Lock()
_cb_consecutive_fails = 0
_cb_tripped = False
_cb_tripped_at = 0.0

_client = None
_client_lock = threading.Lock()

__all__ = ["call_nim", "NIM_MODEL", "NIM_DISPLAY_NAME", "is_nim_available"]


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
                f"[NIM] Circuit breaker TRIPPED after "
                f"{_cb_consecutive_fails} consecutive failures. "
                f"NIM paused for {_CB_COOLDOWN_SECONDS/60:.0f} minutes."
            )
            print_critical(f"NIM CIRCUIT BREAKER TRIPPED - {reason}")


def _record_success() -> None:
    global _cb_consecutive_fails, _cb_tripped
    with _cb_lock:
        if _cb_consecutive_fails > 0 or _cb_tripped:
            print("[NIM] Circuit breaker reset after successful call.")
        _cb_consecutive_fails = 0
        _cb_tripped = False


def is_nim_available() -> bool:
    global _cb_consecutive_fails, _cb_tripped
    with _cb_lock:
        if not _cb_tripped:
            return True

        elapsed = time.time() - _cb_tripped_at
        if elapsed >= _CB_COOLDOWN_SECONDS:
            _cb_tripped = False
            _cb_consecutive_fails = 0
            print("[NIM] Circuit breaker auto-reset after cooldown. Resuming NIM calls.")
            return True

        remaining = _CB_COOLDOWN_SECONDS - elapsed
        print(f"[NIM] Circuit breaker active. Resuming in {remaining:.0f}s")
        return False


def _get_client():
    global _client
    api_key = os.getenv("NIM_API_KEY", "").strip()
    if not api_key:
        print_critical("NIM_API_KEY missing - add it to .env before using DeepSeek NIM")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("[NIM] openai package not installed. Run: pip install openai")
        return None

    with _client_lock:
        if _client is None:
            _client = OpenAI(api_key=api_key, base_url=NIM_BASE_URL)
        return _client


def call_nim(prompt: str, max_tokens: int = 2048) -> str | None:
    if not is_nim_available():
        print_outage_banner()
        return None

    client = _get_client()
    if client is None:
        _record_failure("NIM client unavailable")
        if _is_backtest_mode():
            time.sleep(1.5)
        return None

    thinking_kwargs = _THINKING_MODES[NIM_THINKING_MODE]

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
        call_kwargs = {
            "model": NIM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if thinking_kwargs is not None:
            call_kwargs["extra_body"] = {"chat_template_kwargs": thinking_kwargs}
        return client.chat.completions.create(**call_kwargs)

    max_retries = 6
    backoff = 5.0  # Start at 5s; doubles each attempt: 5→10→20→40→80s

    for attempt in range(max_retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_call)
                response = future.result(timeout=_CALL_TIMEOUT_SECONDS)

            content = response.choices[0].message.content
            _record_success()
            return content.strip() if content else ""

        except concurrent.futures.TimeoutError:
            print(f"[NIM] Call timed out after {_CALL_TIMEOUT_SECONDS}s.")
            _record_failure("NIM timeout")
            return None

        except Exception as exc:
            exc_name = type(exc).__name__
            exc_str = str(exc)
            
            # Check for 429 rate limit or 503 service congestion
            is_rate_limit = "429" in exc_str or "RateLimit" in exc_name or "503" in exc_str
            
            if is_rate_limit and attempt < max_retries - 1:
                sleep_time = backoff * (2.0 ** attempt)  # 5→10→20→40→80s
                print(f"[NIM] Rate-limit ({exc_name}) attempt {attempt+1}/{max_retries}. "
                      f"Backing off {sleep_time:.0f}s...")
                time.sleep(sleep_time)
                continue
                
            print(f"[NIM] Call failed: {exc_name}: {exc}")
            _record_failure(exc_str[:160])
            return None

        finally:
            if _is_backtest_mode():
                time.sleep(3.5)  # 3.5s between each NIM call — stays under free-tier RPM


print(f"[NIM] ✓ {NIM_MODEL} | NIM endpoint ready | mode: {NIM_THINKING_MODE}")
