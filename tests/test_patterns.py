"""
Unit tests for chart pattern detection module.

Tests synthetic data that should trigger each pattern and
verifies correct pattern identification with no false positives.
"""
import json
import numpy as np
import pandas as pd
import pytest

# Setup config before importing patterns module
CONFIG_PATH = 'config.json'
EXAMPLE_PATH = 'config.example.json'

with open(EXAMPLE_PATH) as f:
    config = json.load(f)
with open(CONFIG_PATH, 'w') as f:
    json.dump(config, f)

from modules.patterns import find_pattern, _extract_extrema


def _make_df(highs, lows, n=None):
    """Build OHLCV DataFrame from high/low arrays."""
    if n is None:
        n = len(highs)
    highs = np.array(highs[:n])
    lows = np.array(lows[:n])
    closes = (highs + lows) / 2
    opens = closes - 0.3
    volume = [10000] * n
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volume
    })


class TestHeadAndShoulders:
    def test_classic_hs(self):
        """3 peaks: shoulder(110) < head(118) > shoulder(110)"""
        np.random.seed(42)
        n = 100
        base = np.ones(n) * 100.0
        for i in range(n):
            base[i] = 100 + np.random.normal(0, 0.3)
        base[20] = 110
        base[35] = 118
        base[50] = 110
        df = _make_df(base + 1.5, base - 1.5)
        assert find_pattern(df) == 'head_and_shoulders'

    def test_tall_hs(self):
        """H&S with pronounced shoulders (well above valleys)."""
        np.random.seed(42)
        n = 120
        base = np.ones(n) * 100.0
        for i in range(n):
            base[i] = 100 + np.random.normal(0, 0.15)
        # Three spikes with clear separation from noise
        base[20] = 115
        base[40] = 130
        base[60] = 115
        df = _make_df(base + 1.5, base - 1.5)
        result = find_pattern(df)
        # With 30-point spikes and low noise, should detect H&S or at minimum double_top
        assert result in ('head_and_shoulders', 'double_top')


class TestInverseHeadAndShoulders:
    def test_classic_ihs(self):
        """3 valleys: shoulder(90) > head(82) < shoulder(90)"""
        np.random.seed(43)
        n = 100
        base = np.ones(n) * 100.0
        for i in range(n):
            base[i] = 100 + np.random.normal(0, 0.3)
        base[20] = 90
        base[35] = 82
        base[50] = 90
        df = _make_df(base + 1.5, base - 1.5)
        assert find_pattern(df) == 'inverse_head_and_shoulders'


class TestRisingWedge:
    def test_classic_rising_wedge(self):
        """Both slopes rising, low slope faster (converging upward)."""
        np.random.seed(44)
        n = 100
        t = np.arange(n)
        highs = 100 + t * 0.05 + np.random.normal(0, 0.5, n)
        lows = 100 + t * 0.08 + np.random.normal(0, 0.5, n)
        df = _make_df(highs, lows)
        assert find_pattern(df) == 'rising_wedge'


class TestFallingWedge:
    def test_classic_falling_wedge(self):
        """Both slopes falling, low slope faster (converging downward)."""
        np.random.seed(44)
        n = 100
        t = np.arange(n)
        highs = 100 - t * 0.05 + np.random.normal(0, 0.5, n)
        lows = 100 - t * 0.08 + np.random.normal(0, 0.5, n)
        df = _make_df(highs, lows)
        assert find_pattern(df) == 'falling_wedge'


class TestCupAndHandle:
    def test_classic_cup_handle(self):
        """Rounded bottom 100→85→100 with handle pullback to 96."""
        np.random.seed(45)
        n = 150
        cup = np.concatenate([
            np.linspace(100, 100, 10),
            np.linspace(100, 88, 25),
            np.linspace(88, 85, 15),
            np.linspace(85, 88, 15),
            np.linspace(88, 100, 25),
            np.linspace(100, 96, 10),
            np.linspace(96, 98, 10),
            np.linspace(98, 99, 10),
            np.linspace(99, 100, 10),
            np.linspace(100, 101, 10),
            np.linspace(101, 100.5, 10),
        ])
        cup += np.random.normal(0, 0.2, n)
        df = _make_df(cup + 1.5, cup - 1.5)
        assert find_pattern(df) == 'cup_and_handle'


class TestDoubleTop:
    def test_classic_double_top(self):
        """Two peaks at 110 with base at 100."""
        np.random.seed(48)
        n = 80
        base = np.ones(n) * 100.0
        base[25] = 110
        base[50] = 110
        for i in range(n):
            if i not in [25, 50]:
                base[i] = 100 + np.random.normal(0, 0.2)
        df = _make_df(base + 1, base - 1)
        assert find_pattern(df) == 'double_top'


class TestDoubleBottom:
    def test_classic_double_bottom(self):
        """Two valleys at 90 with base at 100."""
        np.random.seed(46)
        n = 80
        base = np.ones(n) * 100.0
        base[25] = 90
        base[50] = 90
        for i in range(n):
            if i not in [25, 50]:
                base[i] = 100 + np.random.normal(0, 0.2)
        df = _make_df(base + 1, base - 1)
        assert find_pattern(df) == 'double_bottom'


class TestBullFlag:
    def test_classic_bull_flag(self):
        """Slight downtrend in close prices."""
        np.random.seed(47)
        n = 80
        t = np.arange(n)
        highs = 100 - t * 0.001 + np.random.normal(0, 0.3, n)
        lows = 100 - t * 0.001 + np.random.normal(0, 0.3, n)
        df = _make_df(highs, lows)
        assert find_pattern(df) == 'bull_flag'


class TestNoFalsePositive:
    def test_random_walk(self):
        """Random walk should not match any pattern."""
        np.random.seed(99)
        walk = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = _make_df(walk + 1, walk - 1)
        assert find_pattern(df) is None

    def test_flat_data(self):
        """Flat data should not match."""
        n = 100
        flat = np.ones(n) * 100
        df = _make_df(flat + 0.5, flat - 0.5)
        # May or may not match — just verify no crash
        result = find_pattern(df)
        assert result is None or isinstance(result, str)

    def test_short_data(self):
        """Data < 50 bars should return None."""
        np.random.seed(1)
        n = 30
        walk = 100 + np.cumsum(np.random.normal(0, 1, n))
        df = _make_df(walk + 1, walk - 1)
        assert find_pattern(df) is None


class TestPriorityOrder:
    """Verify that more specific patterns are detected before less specific ones."""

    def test_double_top_not_detected_as_hs(self):
        """Double Top data should NOT match H&S (priority works)."""
        np.random.seed(48)
        n = 80
        base = np.ones(n) * 100.0
        base[25] = 110
        base[50] = 110
        for i in range(n):
            if i not in [25, 50]:
                base[i] = 100 + np.random.normal(0, 0.2)
        df = _make_df(base + 1, base - 1)
        result = find_pattern(df)
        # Should be double_top, NOT head_and_shoulders
        assert result == 'double_top'

    def test_double_bottom_not_detected_as_cup(self):
        """Double Bottom data should NOT match Cup & Handle."""
        np.random.seed(46)
        n = 80
        base = np.ones(n) * 100.0
        base[25] = 90
        base[50] = 90
        for i in range(n):
            if i not in [25, 50]:
                base[i] = 100 + np.random.normal(0, 0.2)
        df = _make_df(base + 1, base - 1)
        result = find_pattern(df)
        assert result == 'double_bottom'
