"""
Central JSON file lock registry.

All shared JSON reads and writes should pass through this module so every
file has one process-local re-entrant lock and atomic write/backup behavior.
"""

import os
import json
import threading
import shutil
from pathlib import Path


_registry = {}
_registry_lock = threading.Lock()


def _get_lock(path: str) -> threading.RLock:
    try:
        key = os.path.abspath(path)
        with _registry_lock:
            if key not in _registry:
                _registry[key] = threading.RLock()
            return _registry[key]
    except Exception:
        return threading.RLock()


def read_json(path: str) -> dict | list | None:
    path = os.path.abspath(path)
    lock = _get_lock(path)
    lock.acquire()
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            print(f"[FileRegistry] WARNING: Failed to read {path} — {error}")
            backup_path = path + ".backup"
            if os.path.exists(backup_path):
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        backup_content = json.load(f)
                    print(f"[FileRegistry] Restored {path} from .backup")
                    return backup_content
                except Exception:
                    print(f"[FileRegistry] CRITICAL: Backup also corrupted for {path}")
                    return None
            return None
        except Exception as error:
            print(f"[FileRegistry] WARNING: Failed to read {path} — {error}")
            return None
    finally:
        lock.release()


def write_json(path: str, data: dict | list) -> bool:
    path = os.path.abspath(path)
    lock = _get_lock(path)
    tmp_path = path + ".tmp"
    lock.acquire()
    try:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            shutil.copy2(path, path + ".backup")
            return True
        except Exception as error:
            print(f"[FileRegistry] ERROR: Failed to write {path} — {error}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False
    finally:
        lock.release()


def register(path: str) -> None:
    path = os.path.abspath(path)
    _get_lock(path)
    print(f"[FileRegistry] Registered: {path}")


def get_registry_status() -> dict:
    with _registry_lock:
        return {
            "registered_files": len(_registry),
            "paths": list(_registry.keys()),
        }


print("[FileRegistry] ✓ Initialised — central file lock registry ready.")


if __name__ == "__main__":
    write_json("test_output.json", {"hello": "world", "count": 42})
    result = read_json("test_output.json")
    assert result == {"hello": "world", "count": 42}
    print("Test 1 PASSED")

    write_json("test_output.json", {"hello": "world"})
    with open("test_output.json", "w", encoding="utf-8") as f:
        f.write("NOT VALID JSON {{{{")
    result = read_json("test_output.json")
    assert result == {"hello": "world"}
    print("Test 2 PASSED — backup restore works")

    results = []

    def worker(i):
        write_json("test_thread.json", {"writer": i})
        val = read_json("test_thread.json")
        results.append(val)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r is not None for r in results)
    print("Test 3 PASSED — no thread corruption")

    for test_file in [
        "test_output.json",
        "test_output.json.backup",
        "test_thread.json",
        "test_thread.json.backup",
    ]:
        try:
            Path(test_file).unlink()
        except FileNotFoundError:
            pass

    print("All tests passed.")
