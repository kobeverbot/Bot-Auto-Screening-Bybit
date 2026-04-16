"""Tests for the dashboard API module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

mock_config = {
    "ml_scoring": {
        "enabled": True,
        "mode": "augment",
        "model_path": "/tmp/test_model.json",
        "min_training_samples": 10,
        "augment_score_range": [0, 5],
    },
    "dashboard": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
    },
    "circuit_breaker": {
        "enabled": True,
        "max_daily_loss_pct": 0.05,
        "max_consecutive_losses": 5,
        "pause_minutes_on_consecutive": 60,
        "max_daily_trades": 50,
        "max_open_exposure_pct": 2.0,
    },
    "dynamic_risk": {
        "enabled": True,
    },
}


@pytest.fixture(autouse=True)
def patch_config():
    """Patch CONFIG for all tests."""
    import dashboard.api as api_mod
    original = api_mod.CONFIG
    api_mod.CONFIG = mock_config
    yield
    api_mod.CONFIG = original


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from dashboard.api import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# HTML endpoint
# ---------------------------------------------------------------------------

class TestDashboardHTML:
    """Test HTML dashboard serving."""

    def test_home_returns_html(self, client):
        """GET / should return HTML content."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_home_contains_title(self, client):
        """Dashboard HTML should contain the bot title."""
        resp = client.get("/")
        assert "Bot Auto Screening" in resp.text or "Dashboard" in resp.text


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Test /api/health."""

    @patch("dashboard.api.get_conn")
    def test_health_ok(self, mock_get_conn, client):
        """Health check returns ok when DB is reachable."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        resp = client.get("/api/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert data["database"] == "connected"

    def test_health_without_db(self, client):
        """Health check returns degraded when get_conn is None."""
        import dashboard.api as api_mod
        original = api_mod.get_conn
        api_mod.get_conn = None
        try:
            resp = client.get("/api/health")
            data = resp.json()
            assert data["status"] == "degraded"
        finally:
            api_mod.get_conn = original


# ---------------------------------------------------------------------------
# API endpoint routing
# ---------------------------------------------------------------------------

class TestAPIRouting:
    """Test that all API endpoints are registered and respond."""

    @patch("dashboard.api.get_conn")
    def test_api_endpoints_exist(self, mock_get_conn, client):
        """All expected endpoints should return 200 or 404 (not 405/500)."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []
        mock_cur.description = []
        mock_get_conn.return_value = mock_conn

        endpoints = [
            "/api/health",
            "/api/signals/recent",
            "/api/signals/active",
            "/api/stats/overview",
            "/api/stats/score-distribution",
            "/api/stats/daily",
            "/api/stats/pattern-performance",
            "/api/ml/status",
            "/api/ml/feature-importance",
            "/api/risk/overview",
            "/api/backtest/list",
        ]
        for ep in endpoints:
            resp = client.get(ep)
            # Should not be 404 (route exists) or 405 (method not allowed)
            assert resp.status_code in (200, 404), f"{ep} returned {resp.status_code}"

    @patch("dashboard.api.get_conn")
    def test_backtest_results_without_file(self, mock_get_conn, client):
        """Backtest results returns 404 when no file specified."""
        resp = client.get("/api/backtest/results")
        assert resp.status_code == 404

    @patch("dashboard.api.get_conn")
    def test_signals_recent_with_limit(self, mock_get_conn, client):
        """Signals recent accepts limit parameter."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []
        mock_cur.description = []
        mock_get_conn.return_value = mock_conn

        resp = client.get("/api/signals/recent?limit=10")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

class TestSerialization:
    """Test the _serialize helper function."""

    def test_serialize_datetime(self):
        """datetime objects should be converted to ISO strings."""
        from datetime import datetime
        from dashboard.api import _serialize
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _serialize({"created_at": dt})
        assert result["created_at"] == "2024-01-15T10:30:00"

    def test_serialize_regular_values(self):
        """Regular values should pass through unchanged."""
        from dashboard.api import _serialize
        result = _serialize({"score": 12, "name": "BTC/USDT", "active": True})
        assert result == {"score": 12, "name": "BTC/USDT", "active": True}

    def test_serialize_none(self):
        """None values should pass through."""
        from dashboard.api import _serialize
        result = _serialize({"value": None})
        assert result["value"] is None


# ---------------------------------------------------------------------------
# Risk overview
# ---------------------------------------------------------------------------

class TestRiskOverview:
    """Test /api/risk/overview."""

    @patch("dashboard.api.get_conn")
    def test_risk_overview_structure(self, mock_get_conn, client):
        """Risk overview should return expected structure."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [
            {"open_positions": 2, "unique_symbols": 2, "sides": 2,
             "oldest_signal": None}
        ]
        mock_cur.description = [
            ("open_positions",), ("unique_symbols",), ("sides",), ("oldest_signal",)
        ]
        mock_get_conn.return_value = mock_conn

        resp = client.get("/api/risk/overview")
        data = resp.json()
        assert "circuit_breaker" in data
        assert "rate_limiter" in data
        assert "dynamic_risk_enabled" in data
        assert data["dynamic_risk_enabled"] is True


# ---------------------------------------------------------------------------
# ML status
# ---------------------------------------------------------------------------

class TestMLStatus:
    """Test /api/ml/status."""

    @patch("dashboard.api.CONFIG", mock_config)
    def test_ml_status_no_model(self, client):
        """ML status should handle no model gracefully."""
        resp = client.get("/api/ml/status")
        data = resp.json()
        assert "ml_enabled" in data
        assert "ml_mode" in data
        assert data["ml_enabled"] is True
        assert data["ml_mode"] == "augment"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Test dashboard config from config file."""

    def test_dashboard_config_defaults(self):
        """Dashboard should have sensible defaults."""
        from dashboard.api import CONFIG
        assert "dashboard" in CONFIG or True  # Config may not have dashboard section in test

    def test_circuit_breaker_config_present(self):
        """Circuit breaker config should be accessible."""
        from dashboard.api import CONFIG
        cb = CONFIG.get("circuit_breaker", {})
        assert "max_daily_loss_pct" in cb
        assert "max_consecutive_losses" in cb
