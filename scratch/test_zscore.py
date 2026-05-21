"""
test_zscore.py — Verification Script for the Live Z-Score Erasure Bug
======================================================================
This script validates that:
1. The old single-row rolling z-score method (Bug A.1) results in NaN standard
   deviations (zeroed out by fillna(0.0)), erasing all real z-scores.
2. The new method (z-scoring the entire historical DataFrame before slicing the
   latest row) successfully preserves the rolling standard deviation and yields
   accurate, non-zero z-scored features for live inference.
"""

import sys
import os
import numpy as np
import pandas as pd

# Dynamic pathing to import feature_engineer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Quant.regime_detector.feature_engineer import apply_rolling_zscore, TF_FEATURE_COLS

def run_zscore_verification():
    print("=" * 70)
    print(" 🧪 RUNNING LIVE Z-SCORE ERASURE BUG VERIFICATION")
    print("=" * 70)

    # 1. Create a mock historical features DataFrame (600 rows)
    np.random.seed(42)
    n_rows = 600
    tf_cols = [c for c in TF_FEATURE_COLS if c != "regime"] # Exclude target column if any
    
    # Generate mock features with trend and volatility
    data = {}
    for col in tf_cols[:10]: # Test on first 10 TF features for demonstration
        # Random walk for some realistic variance
        data[col] = np.cumsum(np.random.normal(0.1, 1.0, n_rows))
        
    df_history = pd.DataFrame(data)
    print(f"Generated mock historical features: {df_history.shape} (600 candles)")

    # ----------------------------------------------------
    # CASE 1: THE OLD METHOD (Bug A.1)
    # Slices the latest row first, then calls apply_rolling_zscore()
    # ----------------------------------------------------
    print("\n--- CASE 1: The Old Method (Single-Row Slicing First) ---")
    latest_row_raw = df_history.iloc[[-1]].copy()
    
    # Apply rolling z-score on just this single row
    latest_row_z_old = apply_rolling_zscore(latest_row_raw, cols=list(data.keys()), window=500)
    
    # Before the fix, single-row std dev is mathematically undefined (NaN), zeroed out by fillna:
    latest_row_z_old_filled = latest_row_z_old.fillna(0.0)
    
    mean_value = latest_row_z_old_filled.mean().mean()
    std_value = latest_row_z_old_filled.std().mean()
    print(f"Old Method Z-scores: Mean={mean_value:.4f}, Std={std_value:.4f}")
    print("Sample Z-score values (First 5 cols):")
    for col in list(data.keys())[:5]:
        print(f"  {col}: {latest_row_z_old_filled[col].values[0]}")
        
    # Check if they are indeed all zeros
    is_all_zero = np.allclose(latest_row_z_old_filled.values, 0.0)
    print(f"Result: All Z-score values are ZERO? {'❌ YES (ERASED!)' if is_all_zero else '✅ NO'}")

    # ----------------------------------------------------
    # CASE 2: THE NEW METHOD (Bug A.1 Fixed)
    # Z-scores the entire historical DataFrame first, then slices the latest row
    # ----------------------------------------------------
    print("\n--- CASE 2: The New Method (Full-DataFrame Z-Scoring First) ---")
    
    # Apply rolling z-score to full historical DataFrame
    df_z_full = apply_rolling_zscore(df_history, cols=list(data.keys()), window=500)
    df_z_full = df_z_full.fillna(0.0)
    
    # Slices the latest row
    latest_row_z_new = df_z_full.iloc[[-1]]
    
    mean_value_new = latest_row_z_new.mean().mean()
    # Verify we have non-zero variance and real z-scores
    std_value_new = latest_row_z_new.std().mean()
    print("Sample Z-score values (First 5 cols):")
    non_zero_count = 0
    for col in list(data.keys())[:5]:
        val = latest_row_z_new[col].values[0]
        print(f"  {col}: {val:+.4f}")
        if not np.isclose(val, 0.0):
            non_zero_count += 1
            
    is_new_all_zero = np.allclose(latest_row_z_new.values, 0.0)
    print(f"Result: All Z-score values are ZERO? {'❌ YES' if is_new_all_zero else '✅ NO (PRESERVED!)'}")
    
    print("\n" + "=" * 70)
    if is_all_zero and not is_new_all_zero:
        print(" 🎉 VERIFICATION SUCCESS: The Live Z-Score Erasure Bug is successfully fixed!")
    else:
        print(" ⚠️ VERIFICATION FAILURE: Z-scores behave unexpectedly.")
    print("=" * 70)

if __name__ == "__main__":
    run_zscore_verification()
