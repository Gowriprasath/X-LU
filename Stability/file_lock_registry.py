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


def read_json(path: str):
    normalized_path = os.path.abspath(path)
    lock = _get_lock(normalized_path)
    with lock:
        try:
            with open(normalized_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as error:
            print(f"[FileRegistry] WARNING: Failed to read {normalized_path} — {error}")
            backup_path = normalized_path + ".backup"
            if os.path.exists(backup_path):
                try:
                    with open(backup_path, "r", encoding="utf-8") as backup_file:
                        backup_data = json.load(backup_file)
                    print(f"[FileRegistry] Restored {normalized_path} from .backup")
                    return backup_data
                except Exception:
                    print(
                        f"[FileRegistry] CRITICAL: Backup also corrupted for {normalized_path}"
                    )
                    return None
            return None


def write_json(path: str, data):
    normalized_path = os.path.abspath(path)
    lock = _get_lock(normalized_path)
    tmp_path = normalized_path + ".tmp"
    with lock:
        try:
            parent = os.path.dirname(normalized_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, normalized_path)
            shutil.copy2(normalized_path, normalized_path + ".backup")
            return True
        except Exception as error:
            print(f"[FileRegistry] ERROR: Failed to write {normalized_path} — {error}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False


def register(path: str) -> None:
    normalized_path = os.path.abspath(path)
    _get_lock(normalized_path)
    print(f"[FileRegistry] Registered: {normalized_path}")


def get_registry_status() -> dict:
    with _registry_lock:
        paths = list(_registry.keys())
    return {"registered_files": len(paths), "paths": paths}


try:
    print("[FileRegistry] ✓ Initialised — central file lock registry ready.")
except UnicodeEncodeError:
    print("[FileRegistry] Initialised - central file lock registry ready.")


if __name__ == "__main__":
    test_files = [
        Path("test_output.json"),
        Path("test_output.json.backup"),
        Path("test_output.json.tmp"),
        Path("test_thread.json"),
        Path("test_thread.json.backup"),
        Path("test_thread.json.tmp"),
    ]

    try:
        # Test 1 — Basic write and read
        assert write_json("test_output.json", {"hello": "world", "count": 42}) is True
        result = read_json("test_output.json")
        assert result == {"hello": "world", "count": 42}
        print("Test 1 PASSED")

        # Test 2 — Backup restore
        assert write_json("test_output.json", {"hello": "world"}) is True
        with open("test_output.json", "w", encoding="utf-8") as f:
            f.write("NOT VALID JSON {{{{")
        result = read_json("test_output.json")
        assert result == {"hello": "world"}
        print("Test 2 PASSED — backup restore works")

        # Test 3 — Thread safety
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
    finally:
        for file_path in test_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

    print("All tests passed.")
