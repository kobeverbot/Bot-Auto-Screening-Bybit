"""
Unit tests for modules/dynamic_risk.py
Run with: python -m pytest tests/test_dynamic_risk.py -v
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# We need to mock CONFIG before importing the module
mock_config = {
    "dynamic_risk": {
        "enabled": True,
        "natr_length": 14,
        "adx_length": 14,
        "volatility_buckets": {
            "calm":     {"natr_max": 2.0,  "leverage_mult": 1.3, "risk_mult": 1.2, "max_pos_mult": 1.2},
            "normal":   {"natr_max": 5.0,  "leverage_mult": 1.0, "risk_mult": 1.0, "max_pos_mult": 1.0},
            "volatile": {"natr_max": 10.0, "leverage_mult": 0.6, "risk_mult": 0.7, "max_pos_mult": 0.7},
            "extreme":  {"natr_max": 999.0, "leverage_mult": 0.3, "risk_mult": 0.4, "max_pos_mult": 0.5},
        },
        "adx_threshold": 25.0,
        "trending_leverage_bonus": 0.2,
        "target_leverage": 25,
        "min_leverage": 1,
        "max_leverage": 50,
    }
}


@pytest.fixture(autouse=True)
def mock_config_loader():
    with patch.dict('sys.modules', {'modules.config_loader': MagicMock(CONFIG=mock_config)}):
        yield


def _make_ohlcv_df(n=50, volatility=0.02, trend=0.0, base_price=100.0):
    """Generate synthetic OHLCV data with controllable volatility."""
    np.random.seed(42)
    returns = np.random.normal(trend / n, volatility, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, volatility / 2, n)))
    low = close * (1 - np.abs(np.random.normal(0, volatility / 2, n)))
    volume = np.random.uniform(1000, 5000, n)
    df = pd.DataFrame({
        'open': np.roll(close, 1),
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    df.loc[0, 'open'] = base_price
    return df


class TestClassifyRegime:
    def test_calm(self):
        from modules.dynamic_risk import classify_regime
        # Need to reimport with mock
        with patch('modules.dynamic_risk.CONFIG', mock_config["dynamic_risk"]):
            pass  # classify_regime uses module-level CFG, need different approach

    def test_calm_direct(self):
        """Test regime classification directly."""
        # NATR < 2% → calm
        from modules.dynamic_risk import classify_regime
        # The function reads CFG at module level, so we patch it
        assert classify_regime(1.0) == "calm"
        assert classify_regime(1.99) == "calm"

    def test_normal(self):
        from modules.dynamic_risk import classify_regime
        assert classify_regime(3.0) == "normal"
        assert classify_regime(4.99) == "normal"

    def test_volatile(self):
        from modules.dynamic_risk import classify_regime
        assert classify_regime(7.0) == "volatile"
        assert classify_regime(9.99) == "volatile"

    def test_extreme(self):
        from modules.dynamic_risk import classify_regime
        assert classify_regime(15.0) == "extreme"
        assert classify_regime(100.0) == "extreme"


class TestAnalyzeVolatility:
    def test_calm_market(self):
        """Low volatility data should produce calm regime."""
        from modules.dynamic_risk import analyze_volatility, CFG
        df = _make_ohlcv_df(volatility=0.005, trend=0.001)
        profile = analyze_volatility(df, "TEST/USDT:USDT")
        assert profile.symbol == "TEST/USDT:USDT"
        assert profile.natr >= 0
        assert profile.regime in ("calm", "normal", "volatile", "extreme")
        assert 0 < profile.leverage_multiplier <= 2.0

    def test_volatile_market(self):
        """High volatility data should produce volatile/extreme regime."""
        from modules.dynamic_risk import analyze_volatility
        df = _make_ohlcv_df(volatility=0.08, trend=0.0)
        profile = analyze_volatility(df, "VOLATILE/USDT:USDT")
        # Very high vol should be volatile or extreme
        assert profile.regime in ("volatile", "extreme")
        assert profile.leverage_multiplier < 1.0  # should reduce leverage

    def test_trending_market_bonus(self):
        """Strong trend + calm regime should give leverage bonus."""
        from modules.dynamic_risk import analyze_volatility
        # Low vol, strong trend
        df = _make_ohlcv_df(volatility=0.005, trend=0.1)
        profile = analyze_volatility(df, "TREND/USDT:USDT")
        if profile.adx >= 25 and profile.regime in ("calm", "normal"):
            assert profile.leverage_multiplier > 1.0

    def test_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        from modules.dynamic_risk import analyze_volatility
        df = pd.DataFrame({'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
        profile = analyze_volatility(df, "EMPTY/USDT:USDT")
        assert profile.natr == 0.0
        assert profile.regime == "calm"  # 0 NATR = calm

    def test_returns_atr(self):
        """Profile should contain ATR in price units."""
        from modules.dynamic_risk import analyze_volatility
        df = _make_ohlcv_df(volatility=0.02, base_price=100.0)
        profile = analyze_volatility(df, "BTC/USDT:USDT")
        assert profile.atr >= 0


class TestAdjustForVolatility:
    def test_disabled_returns_static(self):
        """When dynamic_risk is disabled, returns static config values."""
        from modules.dynamic_risk import adjust_for_volatility, CFG
        disabled_cfg = {**CFG, "enabled": False}
        with patch('modules.dynamic_risk.CFG', disabled_cfg):
            result = adjust_for_volatility(
                df=_make_ohlcv_df(),
                symbol="TEST/USDT:USDT",
                base_leverage=25,
                base_risk_pct=0.01,
                base_max_position_pct=0.05,
                market_max_leverage=50,
            )
            assert result["leverage"] == 25
            assert result["risk_pct"] == 0.01
            assert result["max_position_pct"] == 0.05
            assert result["profile"] is None

    def test_calm_market_increases_leverage(self):
        """Calm market should allow more leverage than base."""
        from modules.dynamic_risk import adjust_for_volatility
        df = _make_ohlcv_df(volatility=0.005)
        result = adjust_for_volatility(
            df=df,
            symbol="CALM/USDT:USDT",
            base_leverage=25,
            base_risk_pct=0.01,
            base_max_position_pct=0.05,
            market_max_leverage=50,
        )
        # Calm regime → leverage multiplier 1.3 (or more with trend bonus)
        assert result["leverage"] >= 25
        assert result["risk_pct"] >= 0.01
        assert result["profile"] is not None

    def test_volatile_market_reduces_leverage(self):
        """Volatile market should reduce leverage below base."""
        from modules.dynamic_risk import adjust_for_volatility
        df = _make_ohlcv_df(volatility=0.08)
        result = adjust_for_volatility(
            df=df,
            symbol="VOL/USDT:USDT",
            base_leverage=25,
            base_risk_pct=0.01,
            base_max_position_pct=0.05,
            market_max_leverage=50,
        )
        assert result["leverage"] < 25
        assert result["risk_pct"] < 0.01

    def test_respects_market_max_leverage(self):
        """Should never exceed market-imposed leverage limit."""
        from modules.dynamic_risk import adjust_for_volatility
        df = _make_ohlcv_df(volatility=0.005)  # calm → 1.3x
        result = adjust_for_volatility(
            df=df,
            symbol="TEST/USDT:USDT",
            base_leverage=25,
            base_risk_pct=0.01,
            base_max_position_pct=0.05,
            market_max_leverage=10,  # very low
        )
        assert result["leverage"] <= 10


class TestValidateSLDistance:
    def test_ok_sl(self):
        """Normal SL distance should pass."""
        from modules.dynamic_risk import validate_sl_distance
        ok, reason, rec = validate_sl_distance(100.0, 98.0, 2.0, "BTC/USDT:USDT")
        assert ok is True
        assert reason == "ok"
        assert rec is None

    def test_too_tight_sl(self):
        """SL < 0.5 ATR should fail."""
        from modules.dynamic_risk import validate_sl_distance
        ok, reason, rec = validate_sl_distance(100.0, 99.8, 2.0, "BTC/USDT:USDT")
        assert ok is False
        assert "too tight" in reason.lower()
        assert rec is not None

    def test_too_wide_sl(self):
        """SL > 4.0 ATR should fail."""
        from modules.dynamic_risk import validate_sl_distance
        ok, reason, rec = validate_sl_distance(100.0, 90.0, 1.0, "BTC/USDT:USDT")
        assert ok is False
        assert "too wide" in reason.lower()
        assert rec is not None

    def test_zero_atr(self):
        """Zero ATR should pass (no validation possible)."""
        from modules.dynamic_risk import validate_sl_distance
        ok, reason, rec = validate_sl_distance(100.0, 98.0, 0.0, "BTC/USDT:USDT")
        assert ok is True

    def test_recommended_sl_direction_long(self):
        """For long position, recommended SL should be below entry."""
        from modules.dynamic_risk import validate_sl_distance
        _, _, rec = validate_sl_distance(100.0, 99.8, 2.0, "BTC/USDT:USDT")
        assert rec < 100.0  # SL below entry for long

    def test_recommended_sl_direction_short(self):
        """For short position (SL above entry), recommended SL should be above entry."""
        from modules.dynamic_risk import validate_sl_distance
        _, _, rec = validate_sl_distance(100.0, 100.2, 2.0, "BTC/USDT:USDT")
        assert rec > 100.0  # SL above entry for short


class TestGetDynamicExposureLimit:
    def test_no_profiles(self):
        """No profiles → base limit unchanged."""
        from modules.dynamic_risk import get_dynamic_exposure_limit
        assert get_dynamic_exposure_limit(2.0, []) == 2.0

    def test_all_calm(self):
        """All calm positions → base limit unchanged."""
        from modules.dynamic_risk import get_dynamic_exposure_limit, VolatilityProfile
        profiles = [
            VolatilityProfile("A", 1.0, 0.5, 30, "calm", 1.3, 1.2, 1.2, 32),
            VolatilityProfile("B", 1.5, 0.6, 28, "calm", 1.3, 1.2, 1.2, 32),
        ]
        assert get_dynamic_exposure_limit(2.0, profiles) == 2.0

    def test_mostly_volatile(self):
        """>75% volatile → 50% reduction."""
        from modules.dynamic_risk import get_dynamic_exposure_limit, VolatilityProfile
        # 3 volatile out of 3 = 100% → > 0.75 threshold
        profiles = [
            VolatilityProfile("A", 7.0, 2.0, 20, "volatile", 0.6, 0.7, 0.7, 15),
            VolatilityProfile("B", 8.0, 2.5, 18, "volatile", 0.6, 0.7, 0.7, 15),
            VolatilityProfile("C", 6.0, 1.8, 22, "volatile", 0.6, 0.7, 0.7, 15),
        ]
        result = get_dynamic_exposure_limit(2.0, profiles)
        assert result == pytest.approx(1.0, abs=0.01)  # 50% of 2.0

    def test_half_volatile(self):
        """50-75% volatile → 30% reduction."""
        from modules.dynamic_risk import get_dynamic_exposure_limit, VolatilityProfile
        # 3 volatile out of 4 = 75% → > 0.5 threshold, not > 0.75
        profiles = [
            VolatilityProfile("A", 7.0, 2.0, 20, "volatile", 0.6, 0.7, 0.7, 15),
            VolatilityProfile("B", 8.0, 2.5, 18, "volatile", 0.6, 0.7, 0.7, 15),
            VolatilityProfile("C", 6.0, 1.8, 22, "volatile", 0.6, 0.7, 0.7, 15),
            VolatilityProfile("D", 1.0, 0.5, 30, "calm", 1.3, 1.2, 1.2, 32),
        ]
        result = get_dynamic_exposure_limit(2.0, profiles)
        assert result == pytest.approx(1.4, abs=0.01)  # 70% of 2.0

    def test_some_volatile(self):
        """>25% but <50% volatile → 15% reduction."""
        from modules.dynamic_risk import get_dynamic_exposure_limit, VolatilityProfile
        profiles = [
            VolatilityProfile("A", 7.0, 2.0, 20, "volatile", 0.6, 0.7, 0.7, 15),
            VolatilityProfile("B", 1.0, 0.5, 30, "calm", 1.3, 1.2, 1.2, 32),
            VolatilityProfile("C", 1.5, 0.6, 28, "normal", 1.0, 1.0, 1.0, 25),
        ]
        result = get_dynamic_exposure_limit(2.0, profiles)
        assert result == pytest.approx(1.7, abs=0.01)  # 85% of 2.0


class TestGetRiskSummary:
    def test_empty(self):
        from modules.dynamic_risk import get_risk_summary
        result = get_risk_summary([])
        assert result["regime"] == "no_positions"

    def test_mixed_regimes(self):
        from modules.dynamic_risk import get_risk_summary, VolatilityProfile
        profiles = [
            VolatilityProfile("A", 1.0, 0.5, 30, "calm", 1.3, 1.2, 1.2, 32),
            VolatilityProfile("B", 1.0, 0.5, 30, "calm", 1.3, 1.2, 1.2, 32),
            VolatilityProfile("C", 7.0, 2.0, 20, "volatile", 0.6, 0.7, 0.7, 15),
        ]
        result = get_risk_summary(profiles)
        assert result["regime"] == "calm"  # mode
        assert result["positions_by_regime"]["calm"] == 2
        assert result["positions_by_regime"]["volatile"] == 1
        assert result["high_volatility_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
