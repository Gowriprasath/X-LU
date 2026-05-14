"""
labeler.py — v2 (5 directional regimes)
=========================================
Algorithmically labels each candle with a market regime.

NEW in v2 — directional Bull/Bear split:
    v1 had: HIGH_VOLATILITY, TRENDING_STRONG, TRENDING_WEAK, LOW_VOLATILITY, RANGING
    v2 has: REVERSAL, BULL_TREND, BEAR_TREND, COMPRESSION, LOW_VOL_RANGE

    KEY UPGRADE: The system now knows DIRECTION, not just strength.
    BULL_TREND + BUY signal = aligned (full size)
    BULL_TREND + SELL signal = counter-regime (50% size)
    This enables true directional dual confirmation in regime_router.py.

5-State Regime Map:
    Priority 1 — REVERSAL:      ATR ratio > 1.5  (spike/news — overrides all)
    Priority 2 — BULL_TREND:    ADX > 22, DI+ > DI-, H4 EMA bullish, H1 structure up
    Priority 3 — BEAR_TREND:    ADX > 22, DI- > DI+, H4 EMA bearish, H1 structure down
    Priority 4 — COMPRESSION:   ATR ratio < 0.55 AND BB width very tight (squeeze)
    Priority 5 — LOW_VOL_RANGE: default (quiet, directionless, no squeeze)

Backward compatibility:
    Old constants kept as aliases so existing code doesn't break.
    REGIME_HIGH_VOLATILITY → REVERSAL
    REGIME_TRENDING_STRONG → BULL_TREND
    REGIME_TRENDING_WEAK   → BEAR_TREND  (closest equivalent)
    REGIME_LOW_VOLATILITY  → COMPRESSION
    REGIME_RANGING         → LOW_VOL_RANGE
"""

import pandas as pd
import numpy as np
import os

# ── New v2 regime constants ───────────────────────────────────────
REGIME_REVERSAL      = "REVERSAL"       # high vol spike / news event
REGIME_BULL_TREND    = "BULL_TREND"     # confirmed upward directional move
REGIME_BEAR_TREND    = "BEAR_TREND"     # confirmed downward directional move
REGIME_COMPRESSION   = "COMPRESSION"   # extreme ATR squeeze, pre-breakout
REGIME_LOW_VOL_RANGE = "LOW_VOL_RANGE" # quiet, directionless, normal range

# ── Backward compatibility aliases (old code still works) ─────────
REGIME_HIGH_VOLATILITY = REGIME_REVERSAL
REGIME_TRENDING_STRONG = REGIME_BULL_TREND
# ZOM-05 FIX: REGIME_TRENDING_WEAK was incorrectly mapped to REGIME_BEAR_TREND.
# v1 "TRENDING_WEAK" meant a weak/maturing UPTREND — the closest v2 equivalent
# is BULL_TREND (a confirmed upward move), not BEAR_TREND (confirmed downward).
# The old mapping meant any code reading REGIME_TRENDING_WEAK was silently
# checking for a bear regime — the exact opposite of the intended direction.
# Mapping to LOW_VOL_RANGE is the safest approximation: "weak trend that isn't
# clearly directional enough to confidently call BULL" aligns with LOW_VOL_RANGE
# semantics better than BEAR_TREND ever did.
REGIME_TRENDING_WEAK   = REGIME_LOW_VOL_RANGE  # ZOM-05 FIX: was REGIME_BEAR_TREND (semantically opposite)
REGIME_TRENDING        = REGIME_BULL_TREND      # old v1 alias
REGIME_LOW_VOLATILITY  = REGIME_COMPRESSION
REGIME_RANGING         = REGIME_LOW_VOL_RANGE

# ── ALL_REGIMES: ordered for consistent LabelEncoder output ──────
ALL_REGIMES = [
    REGIME_REVERSAL,
    REGIME_BULL_TREND,
    REGIME_BEAR_TREND,
    REGIME_COMPRESSION,
    REGIME_LOW_VOL_RANGE,
]

# ── Thresholds ────────────────────────────────────────────────────
ATR_REVERSAL_THRESHOLD    = 1.5    # ATR ratio above this = REVERSAL
ATR_COMPRESSION_THRESHOLD = 0.55   # ATR ratio below this = candidate COMPRESSION
ADX_TREND_THRESHOLD       = 22.0   # ADX above this = trend regime
# BB width percentile below this (within rolling 50-candle window) = COMPRESSION confirmed
BB_COMPRESSION_PERCENTILE = 0.25   # bottom 25% of recent BB width

# DI Margin — minimum DI+ minus DI- (or DI- minus DI+) required to label as
# BULL_TREND or BEAR_TREND.  Without this, DI+=21 vs DI-=20 gets the same
# BULL_TREND label as DI+=35 vs DI-=10.  The HMM then trains on statistically
# indistinguishable candles under two different labels → BULL↔BEAR confusion.
#
# WHY 3.0:
#   ADX period=14, so DI lines have ~14-candle smoothing.
#   A 3-point margin means DI+ is at least 15% relatively stronger than DI-.
#   Below that, the directional signal is inside measurement noise.
#   Candles below the margin fall through to LOW_VOL_RANGE (directionless).
#
# EFFECT ON LABEL DISTRIBUTION:
#   Expect BULL_TREND + BEAR_TREND to each shrink by ~8-12%.
#   LOW_VOL_RANGE grows to absorb those borderline candles.
#   This is correct — those candles were always ambiguous.
DI_MARGIN_MIN = 3.0

# Legacy thresholds (kept for backward compat)
ATR_HIGH_THRESHOLD   = ATR_REVERSAL_THRESHOLD
ATR_LOW_THRESHOLD    = ATR_COMPRESSION_THRESHOLD
ADX_STRONG_THRESHOLD = 30.0
ADX_WEAK_THRESHOLD   = 22.0
ADX_TREND_THRESHOLD_COMPAT = 25.0
ADX_RANGE_THRESHOLD  = 20.0


def label_candles(features_df):
    """
    Labels each row of the multi-TF feature matrix with a regime.

    Priority order (highest wins):
        1. REVERSAL      (ATR spike — overrides everything)
        2. BULL_TREND    (strong ADX + DI+ dominant + H4 EMA bullish + H1 structure up)
        3. BEAR_TREND    (strong ADX + DI- dominant + H4 EMA bearish + H1 structure down)
        4. COMPRESSION   (ATR squeeze + BB squeeze)
        5. LOW_VOL_RANGE (default)

    Returns pd.Series of regime labels, same index as features_df.
    """
    # ── Primary: H1 features ─────────────────────────────────
    if 'h1_atr_ratio' in features_df.columns:
        atr_ratio       = features_df['h1_atr_ratio']
        adx             = features_df['h1_adx']
        plus_di         = features_df['h1_plus_di']
        minus_di        = features_df['h1_minus_di']
        trend_structure = features_df['h1_trend_structure']
        bb_width        = features_df['h1_bb_width']
    else:
        # Single-TF fallback (no prefix)
        atr_ratio       = features_df.get('atr_ratio',       pd.Series(1.0, index=features_df.index))
        adx             = features_df.get('adx',             pd.Series(15.0, index=features_df.index))
        plus_di         = features_df.get('plus_di',         pd.Series(20.0, index=features_df.index))
        minus_di        = features_df.get('minus_di',        pd.Series(20.0, index=features_df.index))
        trend_structure = features_df.get('trend_structure', pd.Series(0,    index=features_df.index))
        bb_width        = features_df.get('bb_width',        pd.Series(0.02, index=features_df.index))

    # ── H4 EMA stack for higher-TF confirmation ───────────────
    if 'h4_ema_stack' in features_df.columns:
        h4_ema_stack = features_df['h4_ema_stack']
    elif 'ema_stack' in features_df.columns:
        h4_ema_stack = features_df['ema_stack']
    else:
        h4_ema_stack = pd.Series(0, index=features_df.index)

    # Default: LOW_VOL_RANGE
    labels = pd.Series(REGIME_LOW_VOL_RANGE, index=features_df.index)

    # ── Priority 4: COMPRESSION ───────────────────────────────
    # ATR compressed AND BB width in bottom quartile of recent 50-candle window
    atr_squeeze_mask = atr_ratio < ATR_COMPRESSION_THRESHOLD
    bb_squeeze_mask  = bb_width < bb_width.rolling(50, min_periods=10).quantile(BB_COMPRESSION_PERCENTILE)
    labels[atr_squeeze_mask & bb_squeeze_mask] = REGIME_COMPRESSION

    # ── Priority 3: BEAR_TREND ────────────────────────────────
    # ADX trending + DI- clearly dominant (margin ≥ DI_MARGIN_MIN)
    # + H4 EMA bearish or neutral + H1 structure down
    #
    # DI_MARGIN_MIN filters out borderline candles where DI- is only
    # marginally above DI+.  Those candles look statistically identical
    # to borderline BULL_TREND candles — labeling them BEAR_TREND creates
    # training noise that causes BULL↔BEAR confusion at inference time.
    bear_mask = (
        (adx > ADX_TREND_THRESHOLD) &
        ((minus_di - plus_di) >= DI_MARGIN_MIN) &
        (h4_ema_stack <= 0) &
        (trend_structure <= 0)
    )
    labels[bear_mask] = REGIME_BEAR_TREND

    # ── Priority 2: BULL_TREND ────────────────────────────────
    # ADX trending + DI+ clearly dominant (margin ≥ DI_MARGIN_MIN)
    # + H4 EMA bullish + H1 structure up
    bull_mask = (
        (adx > ADX_TREND_THRESHOLD) &
        ((plus_di - minus_di) >= DI_MARGIN_MIN) &
        (h4_ema_stack >= 0) &
        (trend_structure >= 0)
    )
    labels[bull_mask] = REGIME_BULL_TREND

    # ── Priority 1: REVERSAL (overrides everything) ───────────
    labels[atr_ratio > ATR_REVERSAL_THRESHOLD] = REGIME_REVERSAL

    return labels


def label_and_save(features_df, labels_path):
    labels = label_candles(features_df)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    labels.to_csv(labels_path, header=['regime'])

    counts = labels.value_counts()
    total  = len(labels)
    print(f"\n[Labeler v2] Regime distribution ({total:,} candles):")
    for regime in ALL_REGIMES:
        count = counts.get(regime, 0)
        bar   = '█' * int(count / total * 40)
        print(f"  {regime:<16}: {count:>8,}  ({count/total*100:>5.1f}%)  {bar}")
    return labels


def load_labels(labels_path):
    if not os.path.exists(labels_path):
        print(f"[Labeler] Labels file not found: {labels_path}")
        return None
    df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
    return df['regime']
