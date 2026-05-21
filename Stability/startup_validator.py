import os
import sys
import json
from file_lock_registry import read_json, write_json, register
from console_display import (print_critical, print_warning, print_memory_event)

CRITICAL_FILES = [
    "Memory/continuation_memory.json",
    "Memory/trade_memory.json",
    "Memory/post_mortem_tracker.json",
]

NON_CRITICAL_FILES = [
    ("Memory/Filter/wisdom.json", []),
    ("Memory/Filter/keywords.json", []),
    ("Memory/Filter/wisdom_tracker.json", {}),
    ("Memory/Filter/human_rules.json", {}),
    ("Quant/trade_episodes/episodes.json", []),
    ("Backtest/backtest_tracker.json", {}),
]


def _resolve(relative_path: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def _validate_file(abs_path: str, is_critical: bool, safe_default) -> str:
    register(abs_path)
    result = read_json(abs_path)
    if result is not None:
        return "OK"

    backup_path = abs_path + ".backup"
    if os.path.exists(backup_path):
        backup_content = read_json(backup_path)
        if backup_content is not None:
            if write_json(abs_path, backup_content):
                print_warning(f"Restored from backup: {os.path.basename(abs_path)}")
                return "RESTORED"

    if is_critical:
        return "FAILED"

    if write_json(abs_path, safe_default):
        print_warning(f"Initialised empty: {os.path.basename(abs_path)}")
        return "RESTORED"
    return "FAILED"


def run_validation() -> bool:
    print("=" * 56)
    print("  STARTUP VALIDATION")
    print("=" * 56)

    ok_count = 0
    restored_count = 0
    failed_files = []

    for path in CRITICAL_FILES:
        abs_path = _resolve(path)
        status = _validate_file(abs_path, is_critical=True, safe_default=None)
        if status == "OK":
            print(f"  ✅  {path}")
            ok_count += 1
        elif status == "RESTORED":
            print(f"  ⚠️   {path}  ← restored from backup")
            restored_count += 1
        elif status == "FAILED":
            print(f"  ❌  {path}  ← CORRUPTED, no backup")
            failed_files.append(path)

    for path, safe_default in NON_CRITICAL_FILES:
        abs_path = _resolve(path)
        status = _validate_file(abs_path, is_critical=False, safe_default=safe_default)
        if status == "OK":
            print(f"  ✅  {path}")
            ok_count += 1
        elif status == "RESTORED":
            print(f"  ⚠️   {path}  ← restored from backup")
            restored_count += 1
        elif status == "FAILED":
            print(f"  ❌  {path}  ← FAILED TO INITIALISE")
            failed_files.append(path)

    print("=" * 56)
    print(f"  Validated : {ok_count + restored_count} files")
    print(f"  Restored  : {restored_count} files")
    print(f"  Failed    : {len(failed_files)} files")
    print("=" * 56)

    if failed_files:
        for failed_path in failed_files:
            print_critical(
                f"HALTING — critical file corrupted: {os.path.basename(failed_path)}"
            )
        print("Bot cannot start safely. Fix or delete the corrupted file and restart.")
        return False

    print("  ✅  All systems validated. Bot starting...\n")
    return True


if __name__ == "__main__":
    test_root = _resolve("test_validator_tmp")
    original_critical = list(CRITICAL_FILES)
    original_non_critical = list(NON_CRITICAL_FILES)

    CRITICAL_FILES = [
        "test_validator_tmp/critical_1.json",
        "test_validator_tmp/critical_2.json",
        "test_validator_tmp/critical_3.json",
    ]
    NON_CRITICAL_FILES = [
        ("test_validator_tmp/non_critical_1.json", []),
        ("test_validator_tmp/non_critical_2.json", {}),
        ("test_validator_tmp/non_critical_3.json", []),
    ]

    def _write_raw(path: str, text: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _safe_remove(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _setup_healthy_state() -> None:
        all_files = []
        all_files.extend((_resolve(p), {"status": "critical_ok"}) for p in CRITICAL_FILES)
        all_files.extend((_resolve(p), d) for p, d in NON_CRITICAL_FILES)
        for path, payload in all_files:
            write_json(path, payload)

    try:
        if os.path.exists(test_root):
            for root, dirs, files in os.walk(test_root, topdown=False):
                for name in files:
                    _safe_remove(os.path.join(root, name))
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass
        os.makedirs(test_root, exist_ok=True)

        # Test 1 — All files healthy
        _setup_healthy_state()
        assert run_validation() is True
        print("Test 1 PASSED")

        # Test 2 — Non-critical file corrupted, no backup
        _setup_healthy_state()
        non_critical_path = _resolve(NON_CRITICAL_FILES[0][0])
        _write_raw(non_critical_path, "NOT JSON {{{")
        _safe_remove(non_critical_path + ".backup")
        assert run_validation() is True
        print("Test 2 PASSED")

        # Test 3 — Non-critical file corrupted, backup exists
        _setup_healthy_state()
        non_critical_path = _resolve(NON_CRITICAL_FILES[1][0])
        _write_raw(non_critical_path, "NOT JSON {{{")
        write_json(non_critical_path + ".backup", {"restored": True})
        assert run_validation() is True
        restored = read_json(non_critical_path)
        assert restored == {"restored": True}
        print("Test 3 PASSED")

        # Test 4 — Critical file corrupted, no backup
        _setup_healthy_state()
        critical_path = _resolve(CRITICAL_FILES[0])
        _write_raw(critical_path, "NOT JSON {{{")
        _safe_remove(critical_path + ".backup")
        assert run_validation() is False
        print("Test 4 PASSED — halt triggered correctly")

        # Test 5 — Critical file corrupted, backup exists
        _setup_healthy_state()
        critical_path = _resolve(CRITICAL_FILES[1])
        _write_raw(critical_path, "NOT JSON {{{")
        write_json(critical_path + ".backup", {"critical_restored": 1})
        assert run_validation() is True
        print("Test 5 PASSED — critical file restored")

    finally:
        CRITICAL_FILES = original_critical
        NON_CRITICAL_FILES = original_non_critical
        if os.path.exists(test_root):
            for root, dirs, files in os.walk(test_root, topdown=False):
                for name in files:
                    _safe_remove(os.path.join(root, name))
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass
            try:
                os.rmdir(test_root)
            except Exception:
                pass

    print("All validator tests passed.")
