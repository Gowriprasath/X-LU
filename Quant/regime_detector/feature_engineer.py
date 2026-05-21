"""
feature_engineer.py  — v2 (25-feature regime engine)
======================================================
Builds features from multi-timeframe OHLC data + full time decomposition.

WHAT CHANGED FROM v1:
    Added 19 new features per timeframe — full 25-feature regime sensor.

    NEW per-TF features:
        Volatility:
            atr_slope           — 5-candle rate-of-change of ATR
            atr_percentile      — ATR rank within rolling 100-candle window (0-1)
            bb_width_slope      — 5-candle rate-of-change of BB width
        Trend:
            adx_slope           — 5-candle slope of ADX
            ema20_50_ratio      — EMA20 / EMA50
            ema50_200_ratio     — EMA50 / EMA200
        Momentum:
            rsi                 — RSI-14
            rsi_slope           — 3-candle slope of RSI
            macd_hist           — MACD histogram (12/26/9)
            macd_slope          — 3-candle slope of MACD histogram
            roc_10              — Rate-of-change over 10 candles
        Structure:
            body_ratio          — |close-open| / (high-low)
            upper_wick_ratio    — upper wick / candle range
            lower_wick_ratio    — lower wick / candle range
            close_position      — (close-low) / (high-low)
            volume_zscore       — volume deviation from 20-candle mean
        Context:
            rolling_vol_20      — 20-candle returns std
            rolling_vol_100     — 100-candle returns std
            vol_ratio_20_100    — rolling_vol_20 / rolling_vol_100

All features are pure numpy/pandas — zero external TA libraries.
Zero API calls — runs entirely locally.
"""

import pandas as pd
import numpy as np
import os
import pytz

NY_TZ = pytz.timezone('America/New_York')


# ================================================================
# ATR
# ================================================================
def compute_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_atr_ratio(df, atr_period=14, mean_period=50):
    atr      = compute_atr(df, atr_period)
    atr_mean = atr.rolling(mean_period, min_periods=max(1, mean_period // 4)).mean()
    return atr, atr / atr_mean.replace(0, np.nan)


# ================================================================
# ADX
# ================================================================
def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_high, prev_low = high.shift(1), low.shift(1)
    up_move, down_move  = high - prev_high, prev_low - low
    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr      = compute_atr(df, period)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


# ================================================================
# EMA STACK — now returns raw EMA values too
# ================================================================
def compute_emas(df):
    close  = df['close']
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    pos20  = np.sign(close - ema20)
    pos50  = np.sign(close - ema50)
    pos200 = np.sign(close - ema200)
    ema_stack = pd.Series(
        np.where((pos20 == pos50) & (pos50 == pos200), pos20, 0),
        index=df.index)
    return pos20, pos50, pos200, ema_stack, ema20, ema50, ema200


# ================================================================
# BOLLINGER BAND WIDTH
# ================================================================
def compute_bb_width(df, period=20, std_dev=2.0):
    close  = df['close']
    middle = close.rolling(period, min_periods=max(1, period // 4)).mean()
    std    = close.rolling(period, min_periods=max(1, period // 4)).std()
    width  = ((middle + std_dev * std) - (middle - std_dev * std)) / middle.replace(0, np.nan)
    return width


# ================================================================
# MOMENTUM (existing)
# ================================================================
def compute_momentum_features(df, lookback=20):
    close = df['close']
    high, low = df['high'], df['low']
    momentum     = close.pct_change(lookback)
    rolling_range = (
        high.rolling(lookback, min_periods=max(1, lookback // 4)).max()
        - low.rolling(lookback, min_periods=max(1, lookback // 4)).min()
    ) / close
    return momentum, rolling_range


# ================================================================
# TREND STRUCTURE (HH/HL or LH/LL)
# ================================================================
def compute_trend_structure(df, swing_period=5):
    high, low = df['high'], df['low']
    prev_high, prev_low = high.shift(swing_period), low.shift(swing_period)
    hh = (high > prev_high).astype(int)
    hl = (low  > prev_low).astype(int)
    lh = (high < prev_high).astype(int)
    ll = (low  < prev_low).astype(int)
    structure = np.where(
        (hh == 1) & (hl == 1),  1,
        np.where((lh == 1) & (ll == 1), -1, 0)
    )
    return pd.Series(structure, index=df.index)


# ================================================================
# NEW v2: RSI-14
# ================================================================
def compute_rsi(df, period=14):
    close    = df['close']
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ================================================================
# NEW v2: MACD (12/26/9)
# ================================================================
def compute_macd(df, fast=12, slow=26, signal=9):
    close       = df['close']
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ================================================================
# NEW v2: RATE OF CHANGE
# ================================================================
def compute_roc(df, period=10):
    return df['close'].pct_change(period)


# ================================================================
# NEW v2: CANDLE STRUCTURE
# ================================================================
def compute_candle_structure(df):
    """Returns body_ratio, upper_wick, lower_wick, close_position — all 0..1."""
    open_, high, low, close = df['open'], df['high'], df['low'], df['close']
    candle_range = (high - low).replace(0, np.nan)
    top_body     = pd.concat([open_, close], axis=1).max(axis=1)
    bottom_body  = pd.concat([open_, close], axis=1).min(axis=1)
    body_ratio   = (close - open_).abs() / candle_range
    upper_wick   = (high - top_body)    / candle_range
    lower_wick   = (bottom_body - low)  / candle_range
    close_pos    = (close - low)        / candle_range
    return body_ratio, upper_wick, lower_wick, close_pos


# ================================================================
# NEW v2: VOLUME Z-SCORE
# ================================================================
def compute_volume_zscore(df, period=20):
    if 'tick_volume' in df.columns:
        vol = df['tick_volume'].astype(float)
    elif 'volume' in df.columns:
        vol = df['volume'].astype(float)
    else:
        return pd.Series(0.0, index=df.index, dtype=float)
    vol_mean = vol.rolling(period, min_periods=max(1, period // 4)).mean()
    vol_std  = vol.rolling(period, min_periods=max(1, period // 4)).std()
    return (vol - vol_mean) / vol_std.replace(0, np.nan)


# ================================================================
# NEW v2: ROLLING VOLATILITY (returns-based)
# ================================================================
def compute_rolling_volatility(df, period_short=20, period_long=100):
    """Returns std of pct_change at two horizons + ratio. Complements ATR."""
    returns  = df['close'].pct_change()
    rv_short = returns.rolling(period_short, min_periods=max(1, period_short // 4)).std()
    rv_long  = returns.rolling(period_long, min_periods=max(1, period_long // 4)).std()
    vol_ratio = rv_short / rv_long.replace(0, np.nan)
    return rv_short, rv_long, vol_ratio


# ================================================================
# SINGLE-TF FEATURE BUILDER
# ================================================================
def build_features(df, prefix='h1'):
    """
    Builds all 31 technical features from a single OHLC DataFrame.
    Each column is prefixed (e.g. 'h1_rsi', 'h4_macd_hist').

    Args:
        df:     pd.DataFrame with open/high/low/close (+ optional volume)
        prefix: string prefix ('m5', 'h1', 'h4', 'd1')

    Returns:
        pd.DataFrame — one row per candle, NaN warmup rows dropped.
    """
    p = f"{prefix}_"
    f = pd.DataFrame(index=df.index)

    # ── VOLATILITY (5 features) ────────────────────────────────
    atr, atr_ratio = compute_atr_ratio(df)
    # _atr_raw is for HMM observation use only — intentionally excluded from
    # _TF_COLS_TEMPLATE and FEATURE_COLS. Named with underscore to signal this.
    f[p+'_atr_raw']       = atr                    # HMM obs only — NOT in FEATURE_COLS
    f[p+'atr_ratio']      = atr_ratio
    f[p+'atr_slope']      = atr.diff(5) / atr.shift(5).abs().replace(0, np.nan)
    f[p+'atr_percentile'] = atr.rolling(100, min_periods=max(1, 100 // 4)).apply(
        lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1), raw=True)

    bb_width = compute_bb_width(df)
    f[p+'bb_width']       = bb_width
    f[p+'bb_width_slope'] = bb_width.diff(5) / bb_width.shift(5).abs().replace(0, np.nan)

    # ── TREND (10 features) ────────────────────────────────────
    adx, plus_di, minus_di = compute_adx(df)
    f[p+'adx']             = adx
    f[p+'adx_slope']       = adx.diff(5) / 5.0
    f[p+'plus_di']         = plus_di
    f[p+'minus_di']        = minus_di

    pos20, pos50, pos200, ema_stack, ema20, ema50, ema200 = compute_emas(df)
    f[p+'ema_pos_20']      = pos20
    f[p+'ema_pos_50']      = pos50
    f[p+'ema_pos_200']     = pos200
    f[p+'ema_stack']       = ema_stack
    f[p+'ema20_50_ratio']  = ema20 / ema50.replace(0, np.nan)
    f[p+'ema50_200_ratio'] = ema50 / ema200.replace(0, np.nan)

    # ── MOMENTUM (7 features) ──────────────────────────────────
    rsi = compute_rsi(df)
    f[p+'rsi']             = rsi
    f[p+'rsi_slope']       = rsi.diff(3) / 3.0

    _, __, macd_hist = compute_macd(df)
    f[p+'macd_hist']       = macd_hist
    f[p+'macd_slope']      = macd_hist.diff(3)

    f[p+'roc_10']          = compute_roc(df, period=10)

    momentum, rolling_range = compute_momentum_features(df)
    f[p+'momentum_20']     = momentum
    f[p+'rolling_range_20']= rolling_range

    # ── STRUCTURE (6 features) ─────────────────────────────────
    f[p+'trend_structure'] = compute_trend_structure(df)

    body_ratio, upper_wick, lower_wick, close_pos = compute_candle_structure(df)
    f[p+'body_ratio']       = body_ratio
    f[p+'upper_wick_ratio'] = upper_wick
    f[p+'lower_wick_ratio'] = lower_wick
    f[p+'close_position']   = close_pos

    f[p+'volume_zscore']    = compute_volume_zscore(df)

    # ── VOLATILITY CONTEXT (3 features) ────────────────────────
    rv_short, rv_long, vol_ratio = compute_rolling_volatility(df)
    f[p+'rolling_vol_20']    = rv_short
    f[p+'rolling_vol_100']   = rv_long
    f[p+'vol_ratio_20_100']  = vol_ratio

    return f.dropna()


# ================================================================
# TIME FEATURES
# ================================================================
def build_time_features(index):
    try:
        if index.tzinfo is None:
            idx = index.tz_localize('UTC').tz_convert(NY_TZ)
        else:
            idx = index.tz_convert(NY_TZ)
    except Exception:
        idx = index

    tf = pd.DataFrame(index=index)

    tf['hour']          = idx.hour
    tf['minute']        = idx.minute
    tf['day_of_week']   = idx.dayofweek
    tf['day_of_month']  = idx.day
    tf['month']         = idx.month
    tf['week_of_year']  = idx.isocalendar().week.astype(int)
    tf['quarter']       = idx.quarter

    tf['is_monday']      = (idx.dayofweek == 0).astype(int)
    tf['is_friday']      = (idx.dayofweek == 4).astype(int)
    tf['is_month_start'] = idx.is_month_start.astype(int)
    tf['is_month_end']   = idx.is_month_end.astype(int)
    tf['is_quarter_end'] = idx.is_quarter_end.astype(int)

    hour = idx.hour
    tf['is_asian']  = ((hour >= 19) | (hour < 2)).astype(int)
    tf['is_london'] = ((hour >= 2)  & (hour < 7)).astype(int)
    tf['is_ny']     = ((hour >= 7)  & (hour < 17)).astype(int)
    tf['is_dead']   = (
        ~(tf['is_asian'].astype(bool) |
          tf['is_london'].astype(bool) |
          tf['is_ny'].astype(bool))
    ).astype(int)

    tf['hour_sin']          = np.sin(2 * np.pi * idx.hour / 24)
    tf['hour_cos']          = np.cos(2 * np.pi * idx.hour / 24)
    minute_of_day           = idx.hour * 60 + idx.minute
    tf['minute_of_day_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    tf['minute_of_day_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
    tf['dow_sin']           = np.sin(2 * np.pi * idx.dayofweek / 7)
    tf['dow_cos']           = np.cos(2 * np.pi * idx.dayofweek / 7)
    tf['dom_sin']           = np.sin(2 * np.pi * idx.day    / 31)
    tf['dom_cos']           = np.cos(2 * np.pi * idx.day    / 31)
    tf['month_sin']         = np.sin(2 * np.pi * idx.month  / 12)
    tf['month_cos']         = np.cos(2 * np.pi * idx.month  / 12)
    tf['woy_sin']           = np.sin(2 * np.pi * tf['week_of_year'] / 52)
    tf['woy_cos']           = np.cos(2 * np.pi * tf['week_of_year'] / 52)

    return tf


# ================================================================
# MULTI-TF FEATURE BUILDER — primary entry point
# ================================================================
def build_multi_tf_features(m5_df, h1_df, h4_df, d1_df):
    """
    Builds a wide multi-timeframe feature matrix aligned to M5 index.
    All higher TF data forward-filled — no lookahead bias.
    Returns: pd.DataFrame — one row per M5 candle.
    """
    print("[FeatureEngineer] Building M5 features (31 per TF)...")
    m5_feats = build_features(m5_df, prefix='m5')

    print("[FeatureEngineer] Building H1 features...")
    h1_feats = build_features(h1_df, prefix='h1')

    print("[FeatureEngineer] Building H4 features...")
    h4_feats = build_features(h4_df, prefix='h4')

    print("[FeatureEngineer] Building D1 features...")
    d1_feats = build_features(d1_df, prefix='d1')

    print("[FeatureEngineer] Building time features...")
    time_feats = build_time_features(m5_df.index)

    print("[FeatureEngineer] Aligning timeframes to M5 index...")

    def align_to_m5(feats):
        m5_idx   = m5_df.index
        combined = feats.reindex(feats.index.union(m5_idx)).ffill()
        return combined.reindex(m5_idx)

    combined = pd.concat([
        m5_feats,
        align_to_m5(h1_feats),
        align_to_m5(h4_feats),
        align_to_m5(d1_feats),
        time_feats.reindex(m5_df.index),
    ], axis=1)

    combined = combined.dropna()

    print(f"[FeatureEngineer] Matrix: {len(combined):,} rows × {len(combined.columns)} cols "
          f"(31 features × 4 TFs + {len(TIME_FEATURE_COLS)} time features)")
    return combined


# ================================================================
# FEATURE COLUMN DEFINITIONS — single source of truth
# ================================================================

# 31 features per timeframe
_TF_COLS_TEMPLATE = [
    # Volatility (5)
    'atr_ratio', 'atr_slope', 'atr_percentile', 'bb_width', 'bb_width_slope',
    # Trend (10)
    'adx', 'adx_slope', 'plus_di', 'minus_di',
    'ema_pos_20', 'ema_pos_50', 'ema_pos_200', 'ema_stack',
    'ema20_50_ratio', 'ema50_200_ratio',
    # Momentum (7)
    'rsi', 'rsi_slope', 'macd_hist', 'macd_slope', 'roc_10',
    'momentum_20', 'rolling_range_20',
    # Structure (6)
    'trend_structure', 'body_ratio', 'upper_wick_ratio',
    'lower_wick_ratio', 'close_position', 'volume_zscore',
    # Context (3)
    'rolling_vol_20', 'rolling_vol_100', 'vol_ratio_20_100',
]

TF_FEATURE_COLS = (
    [f'm5_{c}' for c in _TF_COLS_TEMPLATE] +
    [f'h1_{c}' for c in _TF_COLS_TEMPLATE] +
    [f'h4_{c}' for c in _TF_COLS_TEMPLATE] +
    [f'd1_{c}' for c in _TF_COLS_TEMPLATE]
)

TIME_FEATURE_COLS = [
    'hour', 'minute', 'day_of_week', 'day_of_month',
    'month', 'week_of_year', 'quarter',
    'is_monday', 'is_friday', 'is_month_start', 'is_month_end', 'is_quarter_end',
    'is_asian', 'is_london', 'is_ny', 'is_dead',
    'hour_sin', 'hour_cos', 'minute_of_day_sin', 'minute_of_day_cos',
    'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
    'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
]

FEATURE_COLS = TF_FEATURE_COLS + TIME_FEATURE_COLS
FEATURE_COLS_H1_ONLY = [f'h1_{c}' for c in _TF_COLS_TEMPLATE]


# ================================================================
# ROLLING Z-SCORE NORMALISATION
# ================================================================
def apply_rolling_zscore(df, cols=None, window=500):
    """
    Applies rolling z-score normalisation to financial features.

    WHY ROLLING Z-SCORE INSTEAD OF GLOBAL STANDARDSCALER:
        A global StandardScaler fit on 2015–2024 data computes mean/std
        across the full 10yr period. In 2024, when gold ATR is 30 points,
        the 2015 ATR mean of 10 points is still baked into the scaler.
        Result: 2024 features look like outliers relative to the global mean.

        Rolling z-score normalises each candle relative to ITS OWN recent
        history (last `window` candles). The model always sees features
        in the context of recent market conditions — exactly what a trader
        does mentally when reading a chart.

        z = (x - rolling_mean) / rolling_std

    Applied only to TF_FEATURE_COLS (price-derived features).
    NOT applied to:
        - Time features (already bounded: sin/cos ∈ [-1,1], flags ∈ {0,1})
        - Persistence features (already relative: candle counts, probabilities)

    Args:
        df:     pd.DataFrame — feature matrix
        cols:   list of columns to z-score. None = TF_FEATURE_COLS ∩ df.columns
        window: rolling window size (default 500 ≈ 2 trading months of M5 candles)

    Returns:
        pd.DataFrame — same shape, TF cols replaced with z-scored values.
        First `window-1` rows will have NaN — caller must dropna().

    BUG FIX (v17): rs.replace(0, np.nan) caused 99.4% of rows to be dropped.
        D1/H4 features are forward-filled from daily/4h bars to M5 frequency,
        making them constant for stretches of 48–288 M5 candles. Rolling std
        of a constant = 0. Replacing 0 with NaN produced NaN z-scores for those
        entire stretches. notna().all(axis=1) in trainer.py then dropped every
        row touching a zero-std window → 460,624 of 463,449 rows silently lost.
        XGBoost trained on only 2,825 rows (0.6% of data).

        Fix: replace(0, 1e-8) instead of replace(0, np.nan).
        When std=0, all values in the window equal the mean, so
        z = (x - mean) / 1e-8 = 0 / 1e-8 = 0 — correct neutral z-score.
        Warmup NaN (first ~249 rows, where min_periods not yet met) are
        preserved correctly and still dropped by the caller's dropna().
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in TF_FEATURE_COLS if c in df.columns]

    for col in cols:
        rm  = df[col].rolling(window, min_periods=max(1, window // 4)).mean()
        rs  = df[col].rolling(window, min_periods=max(1, window // 4)).std()
        df[col] = (df[col] - rm) / rs.replace(0, 1e-8)   # BUG FIX: was replace(0, np.nan)

    return df

# ── Persistence + transition feature columns ──────────────────────
PERSISTENCE_FEATURE_COLS = [
    'candles_since_regime_start',      # how long current regime has run
    'regime_duration_mean',            # rolling mean of last 20 completed durations
    'previous_regime_encoded',         # integer-encoded previous regime (one-hot at predict time)
    'regime_transition_prob',          # HMM transition matrix probability: prev→current state
]

# Master list: everything XGBoost trains on
ALL_FEATURE_COLS = FEATURE_COLS + PERSISTENCE_FEATURE_COLS


# ================================================================
# PERSISTENCE FEATURES — computed from HMM label sequence
# Called once in trainer.py after Viterbi decode.
# ================================================================
def compute_persistence_features(labels, hmm_model=None, state_map=None):
    """
    Computes regime persistence + transition features from a decoded label sequence.

    Args:
        labels:     pd.Series of regime strings, same index as feature matrix
        hmm_model:  fitted GMMHMM — used to extract transition probabilities.
                    If None, transition_prob defaults to 0.5.
        state_map:  dict {int_state: regime_string} — inverse needed for prob lookup

    Returns:
        pd.DataFrame with 4 columns, same index as labels:
            candles_since_regime_start  — counter resets each time regime changes
            regime_duration_mean        — rolling mean of last 20 completed durations
            previous_regime_encoded     — integer encoding of previous regime
                                          (REVERSAL=0, BULL_TREND=1, BEAR_TREND=2,
                                           COMPRESSION=3, LOW_VOL_RANGE=4)
            regime_transition_prob      — P(current_state | previous_state) from HMM
                                          transmat_. Captures learned market flows:
                                          COMPRESSION→BULL_TREND has high prob,
                                          BULL_TREND→REVERSAL has high prob, etc.

    WHY TRANSITION FEATURES MATTER:
        Markets follow regime sequences, not random walks.
        COMPRESSION → BULL_TREND  : pre-breakout squeeze resolving upward
        BULL_TREND  → REVERSAL    : exhaustion after extended run
        REVERSAL    → LOW_VOL     : shock absorbed, market digesting
        XGBoost learns these sequential patterns when given prev_regime and
        transition_prob. Without them, it treats each candle as independent.
    """
    from labeler import ALL_REGIMES

    # Integer encoding for previous_regime_encoded
    regime_to_int = {r: i for i, r in enumerate(ALL_REGIMES)}

    # Build inverse state_map: regime_string → HMM state int (for transmat lookup)
    regime_to_state = {}
    if state_map is not None:
        for state_int, regime_str in state_map.items():
            if regime_str not in regime_to_state:
                regime_to_state[regime_str] = state_int

    # Extract HMM transition matrix if available
    transmat = None
    if hmm_model is not None and hasattr(hmm_model, 'transmat_'):
        transmat = hmm_model.transmat_   # shape: (n_states, n_states)

    n                    = len(labels)
    candles_since        = np.zeros(n, dtype=float)
    dur_mean_arr         = np.zeros(n, dtype=float)
    prev_regime_enc      = np.zeros(n, dtype=float)
    transition_prob_arr  = np.full(n, 0.5, dtype=float)   # default: uniform

    completed_durations  = []
    current_count        = 0
    current_regime       = None
    prev_regime          = None

    label_values = labels.values

    for i, regime in enumerate(label_values):
        if regime != current_regime:
            # Regime changed — record completed duration
            if current_regime is not None:
                completed_durations.append(current_count)
                if len(completed_durations) > 50:
                    completed_durations = completed_durations[-50:]
            prev_regime    = current_regime
            current_regime = regime
            current_count  = 1
        else:
            current_count += 1

        candles_since[i]    = float(current_count)
        dur_mean_arr[i]     = (float(np.mean(completed_durations[-20:]))
                               if completed_durations else float(current_count))
        prev_regime_enc[i]  = float(regime_to_int.get(prev_regime, 4)
                                    if prev_regime is not None else 4)

        # Transition probability: P(current_state | prev_state) from HMM transmat
        if transmat is not None and prev_regime is not None:
            prev_s = regime_to_state.get(prev_regime)
            curr_s = regime_to_state.get(regime)
            if prev_s is not None and curr_s is not None:
                transition_prob_arr[i] = float(transmat[prev_s, curr_s])

    return pd.DataFrame({
        'candles_since_regime_start': candles_since,
        'regime_duration_mean':       dur_mean_arr,
        'previous_regime_encoded':    prev_regime_enc,
        'regime_transition_prob':     transition_prob_arr,
    }, index=labels.index)


# ================================================================
# SAVE / LOAD HELPERS
# ================================================================
def save_features(features_df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    features_df.to_csv(path)
    print(f"[FeatureEngineer] Saved {len(features_df):,} rows × "
          f"{len(features_df.columns)} cols → {path}")


def load_features(path):
    if not os.path.exists(path):
        print(f"[FeatureEngineer] File not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[FeatureEngineer] Loaded {len(df):,} rows from {path}")
    return df
