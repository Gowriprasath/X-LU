"""
test_concurrency.py — Process Concurrency Synchronization Verification
======================================================================
This script validates that the `CrossProcessFileLock` integrated inside
`file_lock_registry.py` correctly prevents file corruption, data loss,
and simultaneous write collisions across multiple independent operating system
processes writing concurrently to the same shared JSON file.
"""

import sys
import os
import time
from multiprocessing import Process, Barrier, Value

# Dynamic pathing to import file_lock_registry
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Stability.file_lock_registry import read_json, write_json, modify_json

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "concurrency_test_state.json")

def worker_task(worker_id, barrier, num_writes, success_counter, error_counter):
    # Synchronize startup
    barrier.wait()
    
    for i in range(num_writes):
        lock_acquired_successfully = False
        for attempt in range(5): # Retry if lock acquisition times out
            try:
                # Atomic Read-Modify-Write using modify_json
                def update(state):
                    if not isinstance(state, dict) or "writes" not in state:
                        state = {"writes": []}
                    state["writes"].append({
                        "worker": worker_id,
                        "write_idx": i,
                        "timestamp": time.time()
                    })
                    return state

                if modify_json(TEST_FILE, update):
                    lock_acquired_successfully = True
                    break
            except Exception as e:
                print(f"[Worker-{worker_id}] Exception encountered during cycle: {e}")
                
            time.sleep(0.01) # Short backoff
            
        if lock_acquired_successfully:
            with success_counter.get_lock():
                success_counter.value += 1
        else:
            with error_counter.get_lock():
                error_counter.value += 1

def run_concurrency_verification():
    print("=" * 70)
    print(" 🧪 RUNNING MULTI-PROCESS CONCURRENCY VERIFICATION")
    print("=" * 70)

    # Initialize test file
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
    write_json(TEST_FILE, {"writes": []})
    
    num_workers = 4
    writes_per_worker = 10
    total_expected_writes = num_workers * writes_per_worker
    
    # Multiprocessing shared counters
    success_counter = Value('i', 0)
    error_counter = Value('i', 0)
    
    # Barrier to ensure all processes start writing at the exact same millisecond
    barrier = Barrier(num_workers)
    
    processes = []
    for w_id in range(num_workers):
        p = Process(
            target=worker_task,
            args=(w_id, barrier, writes_per_worker, success_counter, error_counter)
        )
        processes.append(p)
        p.start()
        
    print(f"Spawned {num_workers} concurrent processes. Running high-frequency locked writes...")
    
    for p in processes:
        p.join()
        
    print("\n--- RESULTS ---")
    print(f"Total successful writes reported by workers: {success_counter.value} / {total_expected_writes}")
    print(f"Total write timeout errors: {error_counter.value}")
    
    # Verify file integrity and actual written count
    state_final = read_json(TEST_FILE)
    if state_final and isinstance(state_final, dict) and "writes" in state_final:
        actual_writes_in_file = len(state_final["writes"])
        print(f"Actual verified writes in JSON file: {actual_writes_in_file}")
        
        # Check for corruption (should be a valid JSON list of dicts, no duplicate/overlapping items or partial files)
        is_corrupt = False
        for idx, item in enumerate(state_final["writes"]):
            if not isinstance(item, dict) or "worker" not in item or "write_idx" not in item:
                is_corrupt = True
                break
                
        print(f"File integrity check: {'❌ CORRUPT!' if is_corrupt else '✅ 100% HEALTHY'}")
        
        # Check for data losses
        if actual_writes_in_file == total_expected_writes and not is_corrupt:
            print("\n" + "=" * 70)
            print(" 🎉 CONCURRENCY VERIFICATION SUCCESS: Zero data loss, zero file corruption!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print(" ⚠️ CONCURRENCY WARNING: Some writes failed or data was lost due to lock timeout.")
            print("=" * 70)
    else:
        print("❌ CRITICAL: Could not read final test file or it is corrupted.")
        
    # Clean up test file
    if os.path.exists(TEST_FILE):
        try:
            os.remove(TEST_FILE)
            # clean lockdir
            lockdir = TEST_FILE + ".lockdir"
            if os.path.exists(lockdir):
                os.rmdir(lockdir)
        except Exception:
            pass

if __name__ == "__main__":
    # On Windows, multiprocessing requires if __name__ == "__main__"
    run_concurrency_verification()
