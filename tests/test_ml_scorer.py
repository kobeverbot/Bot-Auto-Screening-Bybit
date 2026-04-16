"""
Unit tests for modules/ml_scorer.py
Run with: PYTHONPATH=. python3.12 -m pytest tests/test_ml_scorer.py -v
"""
import json
import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# ── Mock config before import ──
mock_config = {
    "ml_scoring": {
        "enabled": True,
        "mode": "augment",
        "model_path": "",  # set per-test
        "min_training_samples": 10,
        "augment_score_range": [0, 5],
        "replace_score_weights": {
            "tech_score": 1.0, "smc_score": 1.2,
            "quant_score": 0.8, "deriv_score": 1.0,
            "ml_confidence": 2.0,
        },
        "heuristic_weights": {
            "tech_score": 1.0, "smc_score": 1.5,
            "quant_score": 0.8, "deriv_score": 1.2,
            "zeta_score": 0.03, "z_score": 0.1,
            "obi": 1.5, "rr_bonus": 0.5,
        },
    }
}


@pytest.fixture(autouse=True)
def mock_config_loader():
    """Patch CONFIG in the already-imported ml_scorer module."""
    import modules.ml_scorer as ml_mod
    import modules.config_loader as cfg_mod
    original_config = cfg_mod.CONFIG
    cfg_mod.CONFIG = mock_config
    ml_mod.CONFIG = mock_config  # Must also patch the ml_scorer reference
    ml_mod._CFG = None
    ml_mod._model = None
    ml_mod._metadata = None
    yield
    cfg_mod.CONFIG = original_config
    ml_mod.CONFIG = original_config
    ml_mod._CFG = None
    ml_mod._model = None
    ml_mod._metadata = None


def _sample_signal(tech=4, smc=3, quant=2, deriv=2, pattern="bull_flag",
                   side="long", bias="Bullish", rr=3.0, obi=0.6,
                   z_score=1.5, zeta=65, basis=0.5):
    return {
        "tech_score": tech, "smc_score": smc,
        "quant_score": quant, "deriv_score": deriv,
        "pattern": pattern, "side": side, "btc_bias": bias,
        "rr": rr, "obi": obi, "z_score": z_score,
        "zeta_score": zeta, "basis": basis,
    }


class TestBuildFeatureVector:
    def test_basic_dimensions(self):
        """Feature vector should have correct dimensions (9 num + 3 derived + patterns + side + bias)."""
        from modules.ml_scorer import build_feature_vector, KNOWN_PATTERNS
        vec = build_feature_vector(_sample_signal())
        expected = 9 + 3 + len(KNOWN_PATTERNS) + 1 + 2
        assert vec.shape == (expected,)

    def test_numeric_features_populated(self):
        """First 9 features should be numeric scores."""
        from modules.ml_scorer import build_feature_vector
        sig = _sample_signal(tech=4, smc=3, quant=2, deriv=2, z_score=1.5,
                             zeta=65, obi=0.6, basis=0.5, rr=3.0)
        vec = build_feature_vector(sig)
        assert vec[0] == 4.0   # tech_score
        assert vec[1] == 3.0   # smc_score
        assert vec[2] == 2.0   # quant_score
        assert vec[3] == 2.0   # deriv_score
        assert vec[4] == 1.5   # z_score
        assert vec[5] == 65.0  # zeta_score
        assert vec[6] == 0.6   # obi
        assert vec[7] == 0.5   # basis
        assert vec[8] == 3.0   # rr

    def test_derived_features(self):
        """total_score, balance, smc_tech_ratio."""
        from modules.ml_scorer import build_feature_vector
        sig = _sample_signal(tech=4, smc=3, quant=2, deriv=2)
        vec = build_feature_vector(sig)
        assert vec[9] == 11.0  # total = 4+3+2+2
        # balance = std([4,3,2,2])
        assert vec[10] > 0
        # smc_tech_ratio = 3 / max(4, 1) = 0.75
        assert abs(vec[11] - 0.75) < 0.01

    def test_pattern_one_hot(self):
        """Known pattern should set exactly one hot bit."""
        from modules.ml_scorer import build_feature_vector, KNOWN_PATTERNS
        sig = _sample_signal(pattern="head_shoulders")
        vec = build_feature_vector(sig)
        pattern_slice = vec[12:12 + len(KNOWN_PATTERNS)]
        assert pattern_slice.sum() == 1.0  # exactly one active

    def test_unknown_pattern_all_zeros(self):
        """Unknown pattern should have all zeros in pattern slice."""
        from modules.ml_scorer import build_feature_vector, KNOWN_PATTERNS
        sig = _sample_signal(pattern="nonexistent_pattern")
        vec = build_feature_vector(sig)
        pattern_slice = vec[12:12 + len(KNOWN_PATTERNS)]
        assert pattern_slice.sum() == 0.0

    def test_side_encoding(self):
        """Long → 1.0, Short → 0.0."""
        from modules.ml_scorer import build_feature_vector, KNOWN_PATTERNS
        long_vec = build_feature_vector(_sample_signal(side="long"))
        short_vec = build_feature_vector(_sample_signal(side="short"))
        side_idx = 12 + len(KNOWN_PATTERNS)
        assert long_vec[side_idx] == 1.0
        assert short_vec[side_idx] == 0.0

    def test_bias_encoding(self):
        """Bullish bias → bull=1, bear=0; Bearish → bull=0, bear=1."""
        from modules.ml_scorer import build_feature_vector, KNOWN_PATTERNS
        bull_vec = build_feature_vector(_sample_signal(bias="Bullish"))
        bear_vec = build_feature_vector(_sample_signal(bias="Bearish"))
        base = 13 + len(KNOWN_PATTERNS)
        assert bull_vec[base] == 1.0 and bull_vec[base + 1] == 0.0
        assert bear_vec[base] == 0.0 and bear_vec[base + 1] == 1.0

    def test_missing_fields_defaults(self):
        """Missing fields should default to 0."""
        from modules.ml_scorer import build_feature_vector
        vec = build_feature_vector({})
        assert vec[0] == 0.0  # tech_score default

    def test_deterministic(self):
        """Same input → same output."""
        from modules.ml_scorer import build_feature_vector
        sig = _sample_signal()
        v1 = build_feature_vector(sig)
        v2 = build_feature_vector(sig)
        assert np.array_equal(v1, v2)


class TestHeuristicScore:
    def test_all_zero(self):
        """All zero scores (explicit zeta=0) → 0."""
        from modules.ml_scorer import _heuristic_score
        assert _heuristic_score({"zeta_score": 0}) == 0.0

    def test_weighted_sum(self):
        """Verify weighted calculation with confluence bonus (4 active components = +15%)."""
        from modules.ml_scorer import _heuristic_score
        sig = _sample_signal(tech=4, smc=3, quant=2, deriv=2, rr=0, obi=0,
                             z_score=0, zeta=0, basis=0)
        # weights: tech=1.0, smc=1.5, quant=0.8, deriv=1.2
        raw = 4 * 1.0 + 3 * 1.5 + 2 * 0.8 + 2 * 1.2
        expected = raw * 1.15  # 15% confluence bonus (all 4 active)
        assert abs(_heuristic_score(sig) - expected) < 0.01

    def test_confluence_bonus_all_active(self):
        """4 active components → 15% bonus."""
        from modules.ml_scorer import _heuristic_score
        sig = _sample_signal(tech=1, smc=1, quant=1, deriv=1, rr=0, obi=0,
                             z_score=0, zeta=0, basis=0)
        raw = 1 * 1.0 + 1 * 1.5 + 1 * 0.8 + 1 * 1.2
        expected = raw * 1.15  # 15% confluence bonus
        assert abs(_heuristic_score(sig) - expected) < 0.01

    def test_confluence_bonus_3_active(self):
        """3 active components → 5% bonus."""
        from modules.ml_scorer import _heuristic_score
        sig = _sample_signal(tech=1, smc=1, quant=1, deriv=0, rr=0, obi=0,
                             z_score=0, zeta=0, basis=0)
        raw = 1 * 1.0 + 1 * 1.5 + 1 * 0.8 + 0 * 1.2
        expected = raw * 1.05  # 5% confluence bonus
        assert abs(_heuristic_score(sig) - expected) < 0.01


class TestGetMLScore:
    def test_disabled(self):
        """When disabled, returns 0 ml_score."""
        from modules.ml_scorer import get_ml_score
        mock_config["ml_scoring"]["enabled"] = False
        import modules.ml_scorer as ml_mod
        ml_mod._CFG = None
        result = get_ml_score(_sample_signal())
        assert result["ml_score"] == 0
        assert result["mode"] == "disabled"
        mock_config["ml_scoring"]["enabled"] = True
        ml_mod._CFG = None

    def test_augment_mode_range(self):
        """ML score in augment mode should be 0-5."""
        from modules.ml_scorer import get_ml_score
        mock_config["ml_scoring"]["mode"] = "augment"
        import modules.ml_scorer as ml_mod
        ml_mod._CFG = None
        result = get_ml_score(_sample_signal())
        assert 0 <= result["ml_score"] <= 5
        assert result["mode"] == "augment"

    def test_augment_high_score(self):
        """Strong signal should get bonus ML score."""
        from modules.ml_scorer import get_ml_score
        mock_config["ml_scoring"]["mode"] = "augment"
        import modules.ml_scorer as ml_mod
        ml_mod._CFG = None
        strong = _sample_signal(tech=6, smc=5, quant=4, deriv=4, obi=0.8, zeta=80)
        result = get_ml_score(strong)
        assert result["ml_score"] >= 0

    def test_model_used_false_without_model(self):
        """model_used should be False when no trained model loaded."""
        from modules.ml_scorer import get_ml_score
        import modules.ml_scorer as ml_mod
        ml_mod._model = None
        result = get_ml_score(_sample_signal())
        assert result["model_used"] is False


class TestPredictProbability:
    def test_returns_float_0_1(self):
        """Probability should be between 0 and 1."""
        from modules.ml_scorer import predict_probability
        prob = predict_probability(_sample_signal())
        assert 0.0 <= prob <= 1.0

    def test_strong_signal_higher_prob(self):
        """Strong signal should have higher prob than weak signal."""
        from modules.ml_scorer import predict_probability
        strong = _sample_signal(tech=6, smc=5, quant=4, deriv=4, obi=0.9, zeta=80)
        weak = _sample_signal(tech=1, smc=0, quant=0, deriv=0, obi=-0.5, zeta=30)
        p_strong = predict_probability(strong)
        p_weak = predict_probability(weak)
        assert p_strong >= p_weak


class TestTrainModel:
    def test_insufficient_data_raises(self):
        """Should raise if < min_training_samples."""
        from modules.ml_scorer import train_model
        import modules.ml_scorer as ml_mod
        mock_config["ml_scoring"]["min_training_samples"] = 50
        ml_mod._CFG = None
        X = np.random.rand(5, 28)
        y = np.array([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="Insufficient"):
            train_model(X, y)
        mock_config["ml_scoring"]["min_training_samples"] = 10
        ml_mod._CFG = None

    def test_train_gradient_boosting(self):
        """Should train a GBM and return model + metadata."""
        from modules.ml_scorer import train_model, build_feature_vector
        import modules.ml_scorer as ml_mod
        ml_mod._CFG = None
        np.random.seed(42)
        n_feat = len(build_feature_vector(_sample_signal()))
        X = np.random.rand(100, n_feat)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)  # label based on features
        model, meta = train_model(X, y, model_type="gradient_boosting")
        assert meta.n_samples == 100
        assert meta.model_type == "gradient_boosting"
        assert len(meta.feature_importance) > 0
        assert 0.0 <= meta.win_rate <= 1.0

    def test_train_random_forest(self):
        """Should train a RF and return model + metadata."""
        from modules.ml_scorer import train_model, build_feature_vector
        import modules.ml_scorer as ml_mod
        ml_mod._CFG = None
        np.random.seed(42)
        n_feat = len(build_feature_vector(_sample_signal()))
        X = np.random.rand(100, n_feat)
        y = (X[:, 0] > 0.5).astype(int)
        model, meta = train_model(X, y, model_type="random_forest")
        assert meta.model_type == "random_forest"

    def test_train_logistic(self):
        """Should train logistic regression."""
        from modules.ml_scorer import train_model, build_feature_vector
        import modules.ml_scorer as ml_mod
        ml_mod._CFG = None
        np.random.seed(42)
        n_feat = len(build_feature_vector(_sample_signal()))
        X = np.random.rand(100, n_feat)
        y = (X[:, 0] > 0.5).astype(int)
        model, meta = train_model(X, y, model_type="logistic")
        assert meta.model_type == "logistic"


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self):
        """Save model → load → predict should work."""
        from modules.ml_scorer import (
            train_model, save_model, load_model,
            predict_probability, ModelMetadata,
            build_feature_vector,
        )
        import modules.ml_scorer as ml_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.json")
            mock_config["ml_scoring"]["model_path"] = path
            ml_mod._CFG = None

            # Train
            np.random.seed(42)
            n_feat = len(build_feature_vector(_sample_signal()))
            X = np.random.rand(100, n_feat)
            y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
            model, meta = train_model(X, y)

            # Save
            assert save_model(model, meta) is True
            assert os.path.exists(path)

            # Load
            ml_mod._model = None
            ml_mod._metadata = None
            assert load_model() is True
            assert ml_mod._model is not None

            # Predict
            prob = predict_probability(_sample_signal())
            assert 0.0 <= prob <= 1.0

    def test_load_nonexistent_returns_false(self):
        """Loading from nonexistent path should return False."""
        from modules.ml_scorer import load_model
        import modules.ml_scorer as ml_mod
        mock_config["ml_scoring"]["model_path"] = "/nonexistent/model.json"
        ml_mod._CFG = None
        assert load_model() is False


class TestPrepareTrainingData:
    def test_matching_signals_trades(self):
        """Should correctly join signals to trade outcomes."""
        from modules.ml_scorer import prepare_training_data, build_feature_vector
        signals = [
            {"symbol": "BTC", "bar_ts": "100", "tech_score": 4, "smc_score": 3,
             "quant_score": 2, "deriv_score": 2, "z_score": 1, "zeta_score": 60,
             "obi": 0.5, "basis": 0, "rr": 3, "pattern": "bull_flag",
             "side": "long", "btc_bias": "Bullish"},
        ]
        trades = [
            {"symbol": "BTC", "bar_ts": "100", "is_win": True, "pnl_pct": 2.5},
        ]
        X, y, names = prepare_training_data(signals, trades)
        assert len(X) == 1
        assert y[0] == 1
        assert len(names) == len(build_feature_vector(signals[0]))

    def test_no_match_returns_empty(self):
        """Unmatched signals should produce empty arrays."""
        from modules.ml_scorer import prepare_training_data
        signals = [{"symbol": "ETH", "bar_ts": "200", "tech_score": 3}]
        trades = [{"symbol": "BTC", "bar_ts": "100", "is_win": True}]
        X, y, _ = prepare_training_data(signals, trades)
        assert len(X) == 0


class TestGetModelInfo:
    def test_no_model(self):
        """Without model, should report heuristic fallback."""
        from modules.ml_scorer import get_model_info
        info = get_model_info()
        assert info["loaded"] is False
        assert info["type"] == "heuristic_fallback"

    def test_with_model(self):
        """After training + loading, should report model info."""
        from modules.ml_scorer import (
            train_model, save_model, load_model, ModelMetadata,
            get_model_info, build_feature_vector,
        )
        import modules.ml_scorer as ml_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "info_model.json")
            mock_config["ml_scoring"]["model_path"] = path
            mock_config["ml_scoring"]["min_training_samples"] = 10
            ml_mod._CFG = None

            np.random.seed(42)
            n_feat = len(build_feature_vector(_sample_signal()))
            X = np.random.rand(50, n_feat)
            y = (X[:, 0] > 0.5).astype(int)
            model, meta = train_model(X, y)
            save_model(model, meta)

            ml_mod._model = None
            ml_mod._metadata = None
            load_model()

            info = get_model_info()
            assert info["loaded"] is True
            assert info["samples"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
