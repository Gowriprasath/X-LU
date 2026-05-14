import json
import os
import sys
import threading
from datetime import datetime

# BUG-7 FIX: Add module-level lock for continuation_memory.json.
# main_bot.py calls update_state() from the main thread while
# daily_post_mortem.py runs as a daemon and also accesses the same file.
# Simultaneous writes corrupt the JSON. This lock matches the pattern
# already used correctly in memory_manager.py (_memory_lock).
_state_lock = threading.Lock()

# BUG-TL-01 FIX: CONTINUATION_MEM_PATH was used on the next line but never
# imported. This caused NameError at module-load time, crashing the bot
# on startup before main_bot.py could even call import thought_logger.
# The dead _find_project_root() helper (which was defined but never called to
# build the path) is also removed to eliminate the confusion.
_root_tl = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _root_tl not in sys.path:
    sys.path.insert(0, _root_tl)
from paths import CONTINUATION_MEM_PATH

STATE_FILE = CONTINUATION_MEM_PATH


def get_current_state():
    if not os.path.exists(STATE_FILE):
        return {
            "current_bias": "NEUTRAL", 
            "active_thesis": "No previous data found.",
            "trade_in_progress": False
        }
        
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error reading continuation memory: {e}")
        return {
            "current_bias": "NEUTRAL", 
            "active_thesis": "Error reading state.",
            "trade_in_progress": False
        }

def update_state(bias, thesis, analysis, trade_active):
    # BUG-7 FIX: Lock the entire read-modify-write so concurrent calls
    # (main thread + post-mortem daemon) cannot interleave writes.
    with _state_lock:
        # 🚀 FIX: Read existing state first so we don't wipe partial_close_event
        state = {}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except Exception:
                pass

        # Update only the thought-logger fields
        state["current_bias"]          = bias
        state["active_thesis"]         = thesis
        state["last_analysis_summary"] = analysis
        state["trade_in_progress"]     = trade_active
        state["last_update"]           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            return True
        except Exception as e:
            print(f"❌ Failed to update continuation memory: {e}")
            return False