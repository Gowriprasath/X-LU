"""
Startup validation for critical bot JSON files.
"""

import os
import sys
import json
from file_lock_registry import read_json, write_json, register
from console_display import (
    print_critical,
    print_warning,
    print_memory_event,
)
from paths import (
    CONTINUATION_MEM_PATH,
    TRADE_MEMORY_PATH,
    EPISODES_PATH,
    BACKTEST_TRACKER_PATH,
    WISDOM_PATH,
    KEYWORDS_PATH,
    WISDOM_TRACKER_PATH,
    HUMAN_RULES_PATH,
)
# Post mortem tracker path
import os as _sv_os
_POST_MORTEM_PATH = _sv_os.path.join(
    _sv_os.path.dirname(CONTINUATION_MEM_PATH),
    'post_mortem_tracker.json'
)


CRITICAL_FILES = [
    (CONTINUATION_MEM_PATH, None),
    (TRADE_MEMORY_PATH, None),
    (_POST_MORTEM_PATH, None),
]

NON_CRITICAL_FILES = [
    (WISDOM_PATH,           []),
    (KEYWORDS_PATH,         []),
    (WISDOM_TRACKER_PATH,   {}),
    (HUMAN_RULES_PATH,      {}),
    (EPISODES_PATH,         []),
    (BACKTEST_TRACKER_PATH, {}),
]


def _resolve(relative_path: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            write_json(abs_path, backup_content)
            print_warning(f"Restored from backup: {os.path.basename(abs_path)}")
            return "RESTORED"

    if is_critical:
        return "FAILED"

    write_json(abs_path, safe_default)
    print_warning(f"Initialised empty: {os.path.basename(abs_path)}")
    return "RESTORED"


def run_validation() -> bool:
    print("=" * 56)
    print("  STARTUP VALIDATION")
    print("=" * 56)

    ok_count = 0
    restored_count = 0
    failed_files = []

    for path, safe_default in CRITICAL_FILES:
        abs_path = _resolve(path)
        status = _validate_file(abs_path, is_critical=True, safe_default=safe_default)
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
            print(f"  ❌  {path}  ← CORRUPTED, no backup")
            failed_files.append(path)

    print("=" * 56)
    print(f"  Validated : {ok_count + restored_count} files")
    print(f"  Restored  : {restored_count} files")
    print(f"  Failed    : {len(failed_files)} files")
    print("=" * 56)

    if failed_files:
        for failed_path in failed_files:
            print_critical(
                f"HALTING — critical file corrupted: "
                f"{os.path.basename(failed_path)}"
            )
        print(
            "Bot cannot start safely. Fix or delete the "
            "corrupted file and restart."
        )
        return False

    print("  ✅  All systems validated. Bot starting...\n")
    return True


if __name__ == "__main__":
    import shutil

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(base, "test_validator_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    CRITICAL_FILES = [
        ("test_validator_tmp/critical_a.json", None),
        ("test_validator_tmp/critical_b.json", None),
    ]
    NON_CRITICAL_FILES = [
        ("test_validator_tmp/noncritical_a.json", []),
        ("test_validator_tmp/noncritical_b.json", {}),
    ]

    def _write(rel, data):
        abs_path = _resolve(rel)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _corrupt(rel):
        with open(_resolve(rel), "w", encoding="utf-8") as f:
            f.write("NOT VALID JSON {{{{")

    try:
        for p, d in CRITICAL_FILES:
            _write(p, {"ok": True})
        for p, d in NON_CRITICAL_FILES:
            _write(p, d)
        assert run_validation() is True
        print("Test 1 PASSED")

        _corrupt("test_validator_tmp/noncritical_a.json")
        try:
            os.remove(_resolve("test_validator_tmp/noncritical_a.json") + ".backup")
        except FileNotFoundError:
            pass
        assert run_validation() is True
        print("Test 2 PASSED")

        _corrupt("test_validator_tmp/noncritical_b.json")
        _write("test_validator_tmp/noncritical_b.json.backup", {"restored": True})
        assert run_validation() is True
        assert read_json(_resolve("test_validator_tmp/noncritical_b.json")) == {"restored": True}
        print("Test 3 PASSED")

        _corrupt("test_validator_tmp/critical_a.json")
        try:
            os.remove(_resolve("test_validator_tmp/critical_a.json") + ".backup")
        except FileNotFoundError:
            pass
        assert run_validation() is False
        print("Test 4 PASSED — halt triggered correctly")

        _corrupt("test_validator_tmp/critical_a.json")
        _write("test_validator_tmp/critical_a.json.backup", {"restored": True})
        assert run_validation() is True
        print("Test 5 PASSED — critical file restored")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("All validator tests passed.")
