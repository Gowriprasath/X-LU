"""
backtest_selector.py - Interactive AI provider selector for backtests.
"""

import os

__all__ = ["run_selector"]


def _ask_choice(prompt: str, options: dict[str, str]) -> str:
    while True:
        print(prompt)
        for key, label in options.items():
            print(f"  {key}. {label}")
        choice = input("> ").strip()
        if choice in options:
            return choice
        print("Invalid choice. Please enter one of: " + ", ".join(options))


def run_selector() -> None:
    mode_choice = _ask_choice(
        "Mock or Real API?",
        {
            "1": "Mock",
            "2": "Real API",
        },
    )

    if mode_choice != "2":
        os.environ["BACKTEST_AI_MODE"] = "mock"
        return

    os.environ["BACKTEST_AI_MODE"] = "real"

    provider_choice = _ask_choice(
        "Claude / Gemini / DeepSeek (NIM)?",
        {
            "1": "Claude",
            "2": "Gemini",
            "3": "DeepSeek (NIM)",
        },
    )

    provider_map = {
        "1": "claude",
        "2": "gemini",
        "3": "nim",
    }
    os.environ["BACKTEST_AI_PROVIDER"] = provider_map[provider_choice]

    if provider_choice == "3":
        thinking_choice = _ask_choice(
            "Non-think / Think High / Think Max?",
            {
                "1": "Non-think",
                "2": "Think High",
                "3": "Think Max",
            },
        )
        thinking_map = {
            "1": "non_think",
            "2": "think_high",
            "3": "think_max",
        }
        os.environ["NIM_THINKING_MODE"] = thinking_map[thinking_choice]
    else:
        os.environ["NIM_THINKING_MODE"] = os.getenv("NIM_THINKING_MODE", "think_high")

    print()
    print("WARNING: Real API mode will make live model calls during backtesting.")
    print("This can be slower and may consume paid API credits or free-tier quota.")
    print()
