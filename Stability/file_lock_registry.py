"""
Central JSON file lock registry.

All shared JSON reads and writes should pass through this module so every
file has one process-local re-entrant lock and atomic write/backup behavior.
Enhanced with CrossProcessFileLock for multi-process safety.
"""

import os
import json
import threading
import shutil
import time
from pathlib import Path


_registry = {}
_registry_lock = threading.Lock()


class CrossProcessFileLock:
    """
    A lightweight, robust cross-process file-system lock using atomic directory
    creation. Safe across threads and processes on both Windows and POSIX.
    Handles cleaning up orphaned locks.
    """
    def __init__(self, filepath: str, timeout: float = 5.0, delay: float = 0.05, max_lock_age: float = 10.0):
        self.lock_dir = os.path.abspath(filepath) + ".lockdir"
        self.timeout = timeout
        self.delay = delay
        self.max_lock_age = max_lock_age
        self.acquired = False

    def acquire(self) -> bool:
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                os.mkdir(self.lock_dir)
                self.acquired = True
                return True
            except FileExistsError:
                # Check for orphaned lock
                try:
                    mtime = os.path.getmtime(self.lock_dir)
                    if time.time() - mtime > self.max_lock_age:
                        print(f"[FileRegistry] Warning: Removing orphaned lock for {self.lock_dir}")
                        try:
                            os.rmdir(self.lock_dir)
                        except Exception:
                            pass
                        # retry immediately after cleanup
                        os.mkdir(self.lock_dir)
                        self.acquired = True
                        return True
                except Exception:
                    pass
                time.sleep(self.delay)
            except Exception:
                time.sleep(self.delay)
        return False

    def release(self):
        if self.acquired:
            try:
                if os.path.exists(self.lock_dir):
                    os.rmdir(self.lock_dir)
            except Exception:
                pass
            finally:
                self.acquired = False


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
    t_lock = _get_lock(path)
    p_lock = CrossProcessFileLock(path)
    
    t_lock.acquire()
    if not p_lock.acquire():
        t_lock.release()
        print(f"[FileRegistry] ERROR: Timeout acquiring cross-process lock for reading {path}")
        return None
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
        p_lock.release()
        t_lock.release()


def write_json(path: str, data: dict | list) -> bool:
    path = os.path.abspath(path)
    t_lock = _get_lock(path)
    p_lock = CrossProcessFileLock(path)
    tmp_path = path + ".tmp"
    
    t_lock.acquire()
    if not p_lock.acquire():
        t_lock.release()
        print(f"[FileRegistry] ERROR: Timeout acquiring cross-process lock for writing {path}")
        return False
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
        p_lock.release()
        t_lock.release()


def modify_json(path: str, updater_fn) -> bool:
    """
    Atomically read, modify, and write a JSON file while holding the cross-process lock.
    updater_fn is a callable that receives the current data (dict or list) and modifies it in-place
    or returns the new data.
    """
    path = os.path.abspath(path)
    t_lock = _get_lock(path)
    p_lock = CrossProcessFileLock(path)
    tmp_path = path + ".tmp"
    
    t_lock.acquire()
    if not p_lock.acquire():
        t_lock.release()
        print(f"[FileRegistry] ERROR: Timeout acquiring cross-process lock for modifying {path}")
        return False
    try:
        try:
            # 1. Read
            data = None
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError) as error:
                    print(f"[FileRegistry] WARNING: Failed to read {path} for modify — {error}")
                    backup_path = path + ".backup"
                    if os.path.exists(backup_path):
                        try:
                            with open(backup_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            print(f"[FileRegistry] Restored {path} from .backup for modify")
                        except Exception:
                            print(f"[FileRegistry] CRITICAL: Backup also corrupted for {path} during modify")
            
            # 2. Modify
            updated_data = updater_fn(data)
            if updated_data is None:
                updated_data = data
                
            # 3. Write
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            shutil.copy2(path, path + ".backup")
            return True
        except Exception as error:
            print(f"[FileRegistry] ERROR: Failed to modify {path} — {error}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False
    finally:
        p_lock.release()
        t_lock.release()


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


print("[FileRegistry] OK Initialised - central file lock registry ready.")


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
