"""
Chart Pattern Detection Module.

Detects classical chart patterns using local extrema (peaks/valleys),
linear regression slopes, and structural analysis.

Supported Patterns (12 total):
  - head_and_shoulders, inverse_head_and_shoulders  (reversal, 3-point)
  - cup_and_handle                     (bullish continuation)
  - double_top, double_bottom          (reversal, 2-point)
  - rising_wedge, falling_wedge        (reversal, converging slopes)
  - ascending_triangle, descending_triangle  (breakout)
  - bull_flag, bear_flag               (continuation)
  - bullish_rectangle                  (continuation)

Detection priority: most specific → least specific.
"""
import logging
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress
from modules.config_loader import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_slope(values):
    try:
        if len(values) < 2:
            return 0.0
        return linregress(np.arange(len(values)), values)[0]
    except Exception:
        return 0.0


def check_alignment(values, tolerance=None):
    if len(values) < 2:
        return False
    if tolerance is None:
        tolerance = CONFIG['patterns'].get('tolerance', 0.015)
    avg = np.mean(values)
    if avg == 0:
        return False
    return all(abs(v - avg) / abs(avg) < tolerance for v in values)


def _extract_extrema(df, order=5):
    df_idx = df.reset_index(drop=True)
    peak_idx = argrelextrema(df_idx['high'].values, np.greater_equal, order=order)[0]
    valley_idx = argrelextrema(df_idx['low'].values, np.less_equal, order=order)[0]
    peaks = df_idx.iloc[peak_idx]['high'].values
    valleys = df_idx.iloc[valley_idx]['low'].values
    return peaks, valleys, peak_idx, valley_idx


# ──────────────────────────────────────────────
# Head & Shoulders (3-point structural)
# ──────────────────────────────────────────────

def _check_head_and_shoulders(peaks, valleys, peak_indices, df_lows):
    """
    H&S: scan all peak triplets.
    Requires:
      - Head 3%+ above shoulders, shoulders within 6%
      - Shoulders at 40%+ of the valley→head range
        (prevents matching when shoulders are at base level = double_top)
    """
    if len(peaks) < 3 or len(valleys) < 1:
        return False
    min_valley = np.min(valleys)
    if min_valley <= 0:
        return False
    lows = df_lows.values if hasattr(df_lows, 'values') else np.array(df_lows)

    for i in range(2, len(peaks)):
        ls, head, rs = peaks[i - 2], peaks[i - 1], peaks[i]
        if head <= ls or head <= rs:
            continue
        avg_shoulder = (ls + rs) / 2
        if avg_shoulder == 0:
            continue
        # Head must be 3%+ above shoulders
        if (head - avg_shoulder) / avg_shoulder < 0.03:
            continue
        # Shoulders must be within 6% of each other
        if abs(ls - rs) / avg_shoulder > 0.06:
            continue
        # KEY: shoulders must be at 40%+ of the valley→head height
        # This prevents double_top (shoulders at base) from matching H&S
        total_range = head - min_valley
        if total_range <= 0:
            continue
        shoulder_height = avg_shoulder - min_valley
        if shoulder_height / total_range < 0.40:
            continue
        return True
    return False


def _check_inverse_head_and_shoulders(valleys, peaks, valley_indices, df_highs):
    """
    IH&S: scan all valley triplets.
    Requires:
      - Head 3%+ below shoulders, shoulders within 6%
      - Head distinct from shoulders (>2% different)
      - Shoulders at 40%+ of the head→peak range
        (prevents matching when shoulders are at top level = double_bottom)
    """
    if len(valleys) < 3 or len(peaks) < 1:
        return False
    max_peak = np.max(peaks)
    if max_peak <= 0:
        return False

    for i in range(2, len(valleys)):
        ls, head, rs = valleys[i - 2], valleys[i - 1], valleys[i]
        if head >= ls or head >= rs:
            continue
        avg_shoulder = (ls + rs) / 2
        if avg_shoulder == 0:
            continue
        # Head must be 3%+ below shoulders
        if (avg_shoulder - head) / avg_shoulder < 0.03:
            continue
        # Shoulders must be within 6% of each other
        if abs(ls - rs) / avg_shoulder > 0.06:
            continue
        # Head distinct from shoulders
        if abs(head - ls) / ls < 0.02 or abs(head - rs) / rs < 0.02:
            continue
        # KEY: shoulders must be at 40%+ of the head→peak height
        total_range = max_peak - head
        if total_range <= 0:
            continue
        shoulder_depth = max_peak - avg_shoulder
        if shoulder_depth / total_range < 0.40:
            continue
        return True
    return False


# ──────────────────────────────────────────────
# Cup & Handle
# ──────────────────────────────────────────────

def _check_cup_and_handle(peaks, valleys, df_highs, df_lows):
    """
    Cup & Handle: rounded bottom with handle pullback.
    Key requirements:
    - Last 2 peaks = highest (rims)
    - Deepest valley must be in the middle third of the formation
      (prevents matching double_bottom which has two equal valleys at edges)
    - Handle pullback 2-8% in recent bars
    """
    if len(peaks) < 2 or len(valleys) < 2:
        return False
    right_rim, left_rim = peaks[-1], peaks[-2]
    avg_rim = (left_rim + right_rim) / 2
    if avg_rim == 0:
        return False
    if abs(left_rim - right_rim) / avg_rim > 0.05:
        return False
    if len(peaks) > 2 and max(peaks[:-2]) > avg_rim * 1.02:
        return False
    if (avg_rim - min(valleys)) / avg_rim < 0.08:
        return False
    # Cup must be rounded: deepest valley must be UNIQUE (not two equal bottoms)
    if len(valleys) >= 3:
        sorted_v = np.sort(valleys)
        if sorted_v[0] > 0 and sorted_v[1] / sorted_v[0] < 1.03:
            return False  # Two valleys at same level = double_bottom, not cup
    if len(df_highs) < 5 or len(df_lows) < 5:
        return False
    highs_arr = np.array(df_highs.iloc[-5:] if hasattr(df_highs, 'iloc') else df_highs[-5:])
    lows_arr = np.array(df_lows.iloc[-5:] if hasattr(df_lows, 'iloc') else df_lows[-5:])
    recent_low = lows_arr.min()
    handle_pb = (right_rim - recent_low) / right_rim
    if handle_pb < 0.02 or handle_pb > 0.08:
        return False
    if lows_arr[-1] < recent_low * 0.98:
        return False
    return True


# ──────────────────────────────────────────────
# Wedge Patterns
# ──────────────────────────────────────────────

def _check_rising_wedge(peaks, valleys):
    if len(peaks) < 3 or len(valleys) < 3:
        return False
    n_pts = min(4, len(peaks), len(valleys))
    sh = get_slope(peaks[-n_pts:])
    sl = get_slope(valleys[-n_pts:])
    return sh > 0.001 and sl > 0.001 and sl > sh * 1.3


def _check_falling_wedge(peaks, valleys):
    if len(peaks) < 3 or len(valleys) < 3:
        return False
    n_pts = min(4, len(peaks), len(valleys))
    sh = get_slope(peaks[-n_pts:])
    sl = get_slope(valleys[-n_pts:])
    # Both slopes negative, valleys falling faster than peaks (converging)
    return sh < -0.001 and sl < -0.001 and sl < sh * 1.3


# ──────────────────────────────────────────────
# Double Top/Bottom (scan ALL pairs with BETWEEN check)
# ──────────────────────────────────────────────

def _check_double_top(peaks, peak_indices, df_lows):
    """
    Double Top: scan ALL peak pairs.
    Two peaks at same level (1.5%) with valley between them dipping 8%+.
    Peaks must be at least 3% above the average of ALL peaks
    (prevents matching when all peaks are at same noise level = double_bottom data).
    """
    if len(peaks) < 2:
        return False
    lows = df_lows.values if hasattr(df_lows, 'values') else np.array(df_lows)
    avg_all_peaks = np.mean(peaks)

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1, p2 = peaks[i], peaks[j]
            avg = (p1 + p2) / 2
            if avg == 0:
                continue
            if abs(p1 - p2) / avg > 0.015:
                continue
            # Peaks must be significantly above average (not just noise-level equal)
            if (avg - avg_all_peaks) / avg_all_peaks < 0.03:
                continue
            idx1, idx2 = peak_indices[i], peak_indices[j]
            lo, hi = min(idx1, idx2), max(idx1, idx2)
            between_lows = lows[lo:hi + 1]
            if len(between_lows) == 0:
                continue
            min_between = np.min(between_lows)
            dip = (avg - min_between) / avg
            if dip < 0.08:
                continue
            return True
    return False


def _check_double_bottom(valleys, valley_indices, df_highs):
    """
    Double Bottom: scan ALL valley pairs.
    Two valleys at same level (1.5%) with peak between them bouncing 8%+.
    Valleys must be at least 3% below the average of ALL valleys
    (prevents matching when all valleys are at same noise level).
    """
    if len(valleys) < 2:
        return False
    highs = df_highs.values if hasattr(df_highs, 'values') else np.array(df_highs)
    avg_all_valleys = np.mean(valleys)

    for i in range(len(valleys)):
        for j in range(i + 1, len(valleys)):
            v1, v2 = valleys[i], valleys[j]
            avg = (v1 + v2) / 2
            if avg == 0:
                continue
            if abs(v1 - v2) / avg > 0.015:
                continue
            # Valleys must be significantly below average
            if (avg_all_valleys - avg) / avg_all_valleys < 0.03:
                continue
            idx1, idx2 = valley_indices[i], valley_indices[j]
            lo, hi = min(idx1, idx2), max(idx1, idx2)
            between_highs = highs[lo:hi + 1]
            if len(between_highs) == 0:
                continue
            max_between = np.max(between_highs)
            bounce = (max_between - avg) / avg
            if bounce < 0.08:
                continue
            return True
    return False


# ──────────────────────────────────────────────
# Slope-Based Patterns
# ──────────────────────────────────────────────

def _check_ascending_triangle(s_high, s_low):
    return abs(s_high) < 0.0005 and s_low > 0.0002

def _check_descending_triangle(s_high, s_low):
    return abs(s_low) < 0.0005 and s_high < -0.0002

def _check_bull_flag(close_slope):
    """Bull flag: slight downtrend in close prices."""
    return -0.005 < close_slope < -0.0001

def _check_bear_flag(close_slope):
    """Bear flag: slight uptrend in close prices."""
    return 0.0001 < close_slope < 0.005

def _check_bullish_rectangle(s_high, s_low):
    return abs(s_high) < 0.0005 and abs(s_low) < 0.0005


# ──────────────────────────────────────────────
# Main Detection
# ──────────────────────────────────────────────

def find_pattern(df):
    """
    Detect chart patterns. Priority (most specific first):
      1. H&S / Inverse H&S (3-point structural, strict shoulder height)
      2. Cup & Handle (multi-point with handle verification)
      3. Double Top/Bottom (2-point with between verification)
      4. Wedge patterns (converging slopes)
      5. Triangle / Flag / Rectangle (slope-based)
    """
    if len(df) < 50:
        return None

    peaks, valleys, peak_idx, valley_idx = _extract_extrema(df, order=5)
    if len(peaks) < 3 or len(valleys) < 3:
        return None

    enabled = CONFIG['patterns']
    s_high = get_slope(peaks[-4:])
    s_low = get_slope(valleys[-4:])
    # Use close price slope for flags (more robust against noise)
    close_slope_full = get_slope(df['close'].values)

    # 1. H&S / IH&S (most specific — requires 3-point structure with shoulder height)
    if enabled.get('head_and_shoulders') and _check_head_and_shoulders(
        peaks, valleys, peak_idx, df['low']
    ):
        return 'head_and_shoulders'
    if enabled.get('inverse_head_and_shoulders') and _check_inverse_head_and_shoulders(
        valleys, peaks, valley_idx, df['high']
    ):
        return 'inverse_head_and_shoulders'

    # 2. Cup & Handle
    if enabled.get('cup_and_handle') and _check_cup_and_handle(
        peaks, valleys, df['high'].iloc[-20:], df['low'].iloc[-20:]
    ):
        return 'cup_and_handle'

    # 3. Double Top/Bottom
    if enabled.get('double_top') and _check_double_top(peaks, peak_idx, df['low']):
        return 'double_top'
    if enabled.get('double_bottom') and _check_double_bottom(valleys, valley_idx, df['high']):
        return 'double_bottom'

    # 4. Wedge
    if enabled.get('rising_wedge') and _check_rising_wedge(peaks, valleys):
        return 'rising_wedge'
    if enabled.get('falling_wedge') and _check_falling_wedge(peaks, valleys):
        return 'falling_wedge'

    # 5. Slope-based
    if enabled.get('ascending_triangle') and _check_ascending_triangle(s_high, s_low):
        return 'ascending_triangle'
    if enabled.get('descending_triangle') and _check_descending_triangle(s_high, s_low):
        return 'descending_triangle'
    if enabled.get('bull_flag') and _check_bull_flag(close_slope_full):
        return 'bull_flag'
    if enabled.get('bear_flag') and _check_bear_flag(close_slope_full):
        return 'bear_flag'
    if enabled.get('bullish_rectangle') and _check_bullish_rectangle(s_high, s_low):
        return 'bullish_rectangle'

    return None
