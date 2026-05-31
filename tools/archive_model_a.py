import glob
import os
import shutil


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
MODEL_A_DIR = os.path.join(DATA_DIR, "ModelVariants", "MODEL_A_HMM_TIME")


def _copy_file(src, dst_dir):
    if not os.path.exists(src):
        print(f"WARNING: missing source file: {src}")
        return 0

    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"Copying {src} -> {dst}")
    return 1


def run():
    copied = 0

    model_src = os.path.join(DATA_DIR, "Models", "Regime")
    model_dst = os.path.join(MODEL_A_DIR, "Models")
    for pattern in ("*.joblib", "*.ubj", "*.json"):
        for src in glob.glob(os.path.join(model_src, pattern)):
            copied += _copy_file(src, model_dst)

    training_dst = os.path.join(MODEL_A_DIR, "TrainingData")
    copied += _copy_file(os.path.join(DATA_DIR, "Regime", "features.csv"), training_dst)
    copied += _copy_file(os.path.join(DATA_DIR, "Regime", "labels.csv"), training_dst)

    backtest_dst = os.path.join(MODEL_A_DIR, "BacktestResults")
    copied += _copy_file(os.path.join(DATA_DIR, "Backtest", "backtest_tracker.json"), backtest_dst)
    copied += _copy_file(os.path.join(DATA_DIR, "Backtest", "pnl_stats.json"), backtest_dst)

    print(f"Archived {copied} files to MODEL_A_HMM_TIME")


if __name__ == "__main__":
    run()
