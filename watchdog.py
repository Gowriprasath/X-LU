"""
watchdog.py — Gold AI Bridge Watchdog Daemon
===========================================================
Monitors and automatically restarts main_bot.py if it exits
or crashes. Enforces process safety, exponential backoff,
and sends Telegram alerts upon crash and recovery.
"""

import sys
import os
import time
import subprocess
from datetime import datetime

# --- Dynamic Path Setup to utilize telegram_notifier ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "Python Files"))
sys.path.append(os.path.join(current_dir, "Strategy"))
sys.path.append(os.path.join(current_dir, "Memory"))
sys.path.append(os.path.join(current_dir, "Integration"))
_stability_dir = os.path.join(current_dir, "Stability")
if _stability_dir not in sys.path:
    sys.path.insert(0, _stability_dir)

try:
    from telegram_notifier import _send_async
    TELEGRAM_ACTIVE = True
except ImportError:
    TELEGRAM_ACTIVE = False

LOG_FILE = os.path.join(current_dir, "Data", "Logs", "watchdog.log")

def log_to_file(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] [Watchdog] {message}")

def notify_telegram(message: str):
    if TELEGRAM_ACTIVE:
        try:
            # Add NY footer styling consistent with codebase notifications
            ny_time_str = datetime.now().strftime("%H:%M NY")
            footer = f"\n\n<i>{ny_time_str}</i>"
            _send_async(message + footer)
        except Exception as e:
            print(f"[Watchdog] Failed to send Telegram alert: {e}")

def get_python_executable():
    # Detect the virtual environment python first
    venv_py = os.path.join(current_dir, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def main():
    python_exe = get_python_executable()
    bot_script = os.path.join(current_dir, "main_bot.py")
    
    log_to_file("==================================================")
    log_to_file("Starting Gold AI Bridge Watchdog Daemon")
    log_to_file(f"Python Executable: {python_exe}")
    log_to_file(f"Target Script: {bot_script}")
    log_to_file("==================================================")
    
    notify_telegram(
        "👁️ <b>WATCHDOG DAEMON ACTIVE</b>\n\n"
        "<code>Status   :</code> Monitoring main_bot.py\n"
        "<code>Engine   :</code> Safe Restart enabled\n"
        "<code>Backoff  :</code> Exponential (5s to 300s)"
    )

    base_delay = 5
    max_delay = 300
    current_delay = base_delay
    
    while True:
        start_time = time.time()
        log_to_file("Launching main_bot.py...")
        
        # Start main_bot.py as a subprocess
        # We run it with unbuffered output (-u) so we get real-time prints
        proc = subprocess.Popen(
            [python_exe, "-u", bot_script],
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=current_dir
        )
        
        # Wait for the process to terminate
        exit_code = proc.wait()
        run_duration = time.time() - start_time
        
        log_to_file(f"main_bot.py exited with code {exit_code} after {run_duration:.1f} seconds.")
        
        # Determine backoff strategy:
        # If the bot ran successfully for more than 5 minutes (300s),
        # reset the delay to base_delay to avoid long wait times on a random rare crash.
        if run_duration > 300:
            current_delay = base_delay
            log_to_file("Process ran successfully for >5 minutes. Resetting backoff delay.")
            
        # Format Telegram alerts
        exit_msg = (
            f"⚠️ <b>BOT TERMINATED</b>\n\n"
            f"<code>Exit Code :</code> {exit_code}\n"
            f"<code>Duration  :</code> {run_duration:.1f}s\n"
            f"<code>Action    :</code> Auto-restarting in {current_delay}s..."
        )
        notify_telegram(exit_msg)
        
        log_to_file(f"Sleeping for {current_delay} seconds before restart...")
        time.sleep(current_delay)
        
        # Exponential backoff increment (max out at 5 minutes)
        current_delay = min(current_delay * 2, max_delay)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_to_file("Watchdog manually terminated by KeyboardInterrupt.")
        notify_telegram(
            "🛑 <b>WATCHDOG TERMINATED</b>\n\n"
            "<code>Status :</code> Manually stopped by operator."
        )
