"""
Dashboard Web API — FastAPI backend for the trading bot dashboard.

Provides REST endpoints for signals, trades, backtest results,
ML model status, risk overview, and bot health.

Run:
    python -m dashboard.api
    # or
    uvicorn dashboard.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from modules.config_loader import CONFIG
    from modules.database import get_conn, release_conn
except ImportError:
    CONFIG = {}
    get_conn = release_conn = None

app = FastAPI(
    title="Bot Auto Screening — Dashboard",
    description="Web dashboard for the Bybit auto-screening trading bot",
    version="1.0.0",
)

# CORS — allow all origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (JS, CSS, images)
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a query and return rows as list of dicts."""
    if get_conn is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_conn(conn)


def _query_one(sql: str, params: tuple = ()) -> Optional[dict]:
    """Execute a query and return first row or None."""
    rows = _query(sql, params)
    return rows[0] if rows else None


def _serialize(row: dict) -> dict:
    """Convert non-JSON-serializable types."""
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, (timedelta,)):
            out[k] = str(v)
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# HTML Dashboard — serves the SPA
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the main dashboard HTML page."""
    html_path = Path(__file__).resolve().parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard template not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Basic health check."""
    db_ok = False
    try:
        _query("SELECT 1")
        db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if db_ok else "disconnected",
    }


@app.get("/api/signals/recent")
async def recent_signals(
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[str] = Query(default=None),
):
    """Get recent trading signals from the database."""
    sql = """
        SELECT id, symbol, side, timeframe, pattern, entry_price, sl_price,
               tp1, tp2, tp3, rr, status, reason,
               tech_score, smc_score, quant_score, deriv_score, ml_score,
               total_score, z_score, zeta_score, obi, basis, btc_bias,
               tech_reasons, quant_reasons, deriv_reasons, smc_reasons,
               created_at, entry_hit_at, closed_at, exit_price
        FROM trades
    """
    params: list = []
    if status:
        sql += " WHERE status = %s"
        params.append(status)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    rows = _query(sql, tuple(params))
    return [_serialize(r) for r in rows]


@app.get("/api/signals/active")
async def active_signals():
    """Get currently active (open) signals."""
    sql = """
        SELECT id, symbol, side, timeframe, pattern, entry_price, sl_price,
               tp1, tp2, tp3, rr, status,
               tech_score, smc_score, quant_score, deriv_score, ml_score,
               total_score, z_score, zeta_score, obi, basis, btc_bias,
               created_at, entry_hit_at
        FROM trades
        WHERE status NOT LIKE '%%Closed%%'
          AND status NOT LIKE '%%Cancelled%%'
          AND status NOT LIKE '%%Stop Loss%%'
        ORDER BY created_at DESC
    """
    rows = _query(sql)
    return [_serialize(r) for r in rows]


@app.get("/api/stats/overview")
async def stats_overview():
    """Aggregated trading statistics."""
    # Overall stats
    overall = _query_one("""
        SELECT
            COUNT(*) as total_trades,
            COUNT(*) FILTER (WHERE exit_price IS NOT NULL) as closed_trades,
            COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%') as wins,
            COUNT(*) FILTER (WHERE status LIKE '%%Stop Loss%%' OR status LIKE '%%Closed - SL%%') as losses,
            AVG(rr) FILTER (WHERE exit_price IS NOT NULL) as avg_rr,
            AVG(total_score) as avg_score,
            AVG(tech_score) as avg_tech,
            AVG(smc_score) as avg_smc,
            AVG(quant_score) as avg_quant,
            AVG(deriv_score) as avg_deriv,
            AVG(ml_score) as avg_ml
        FROM trades
    """)

    # Recent 24h stats
    last24 = _query_one("""
        SELECT
            COUNT(*) as trades_24h,
            COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%') as wins_24h,
            AVG(total_score) as avg_score_24h
        FROM trades
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """)

    # Per-side breakdown
    by_side = _query("""
        SELECT side,
               COUNT(*) as count,
               COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%') as wins,
               AVG(total_score) as avg_score,
               AVG(rr) as avg_rr
        FROM trades
        GROUP BY side
    """)

    # Per-pattern breakdown (top 10)
    by_pattern = _query("""
        SELECT pattern,
               COUNT(*) as count,
               COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%') as wins,
               AVG(total_score) as avg_score,
               AVG(rr) as avg_rr
        FROM trades
        WHERE pattern IS NOT NULL
        GROUP BY pattern
        ORDER BY count DESC
        LIMIT 10
    """)

    return {
        "overall": _serialize(overall) if overall else {},
        "last_24h": _serialize(last24) if last24 else {},
        "by_side": [_serialize(r) for r in by_side],
        "by_pattern": [_serialize(r) for r in by_pattern],
    }


@app.get("/api/stats/score-distribution")
async def score_distribution():
    """Score distribution for charting."""
    buckets = _query("""
        SELECT
            CASE
                WHEN total_score < 8 THEN '0-7'
                WHEN total_score < 11 THEN '8-10'
                WHEN total_score < 14 THEN '11-13'
                ELSE '14+'
            END as score_range,
            COUNT(*) as count,
            COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%') as wins,
            AVG(total_score) as avg_score
        FROM trades
        WHERE total_score IS NOT NULL
        GROUP BY score_range
        ORDER BY score_range
    """)
    return [_serialize(r) for r in buckets]


@app.get("/api/stats/daily")
async def daily_stats(days: int = Query(default=30, ge=1, le=365)):
    """Daily aggregated stats for time-series charts."""
    rows = _query("""
        SELECT
            DATE(created_at) as date,
            COUNT(*) as signals,
            COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%') as wins,
            AVG(total_score) as avg_score,
            AVG(rr) as avg_rr
        FROM trades
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(created_at)
        ORDER BY date
    """, (days,))
    return [_serialize(r) for r in rows]


@app.get("/api/stats/pattern-performance")
async def pattern_performance():
    """Pattern win rates and PnL analysis."""
    rows = _query("""
        SELECT
            pattern,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%') as wins,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE status LIKE '%%Closed - TP%%')
                / NULLIF(COUNT(*), 0), 1
            ) as win_rate,
            AVG(total_score) as avg_score,
            AVG(rr) as avg_rr,
            AVG(tech_score) as avg_tech,
            AVG(smc_score) as avg_smc,
            AVG(ml_score) as avg_ml
        FROM trades
        WHERE pattern IS NOT NULL AND exit_price IS NOT NULL
        GROUP BY pattern
        ORDER BY total DESC
    """)
    return [_serialize(r) for r in rows]


@app.get("/api/ml/status")
async def ml_status():
    """ML model status and info."""
    try:
        from modules.ml_scorer import get_model_info, predict_probability
        info = get_model_info()
        # Also check if model file exists
        model_path = CONFIG.get("ml_scoring", {}).get(
            "model_path", "ml_models/signal_model.json"
        )
        info["model_file_exists"] = Path(model_path).exists() if model_path else False
        info["ml_enabled"] = CONFIG.get("ml_scoring", {}).get("enabled", False)
        info["ml_mode"] = CONFIG.get("ml_scoring", {}).get("mode", "augment")
        return info
    except Exception as e:
        return {"error": str(e), "loaded": False}


@app.get("/api/ml/feature-importance")
async def ml_feature_importance():
    """ML model feature importance chart data."""
    try:
        from modules.ml_scorer import get_model_info
        info = get_model_info()
        importance = info.get("feature_importance", {})
        if not importance:
            return {"features": [], "values": []}
        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return {
            "features": [k for k, v in sorted_items],
            "values": [v for k, v in sorted_items],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/overview")
async def risk_overview():
    """Current risk state and configuration."""
    circuit_breaker = CONFIG.get("circuit_breaker", {})
    dynamic_risk = CONFIG.get("dynamic_risk", {})
    try:
        from modules.rate_limiter import get_rate_limiter
        rl = get_rate_limiter()
        rl_status = rl.status()
    except Exception:
        rl_status = {"tokens_available": 0, "max_calls_per_sec": 0, "total_waits": 0}

    # Current open exposure
    exposure = _query_one("""
        SELECT
            COUNT(*) as open_positions,
            COUNT(DISTINCT symbol) as unique_symbols,
            COUNT(DISTINCT side) as sides,
            MIN(created_at) as oldest_signal
        FROM trades
        WHERE status NOT LIKE '%%Closed%%'
          AND status NOT LIKE '%%Cancelled%%'
    """)

    return {
        "circuit_breaker": circuit_breaker,
        "dynamic_risk_enabled": dynamic_risk.get("enabled", False),
        "rate_limiter": rl_status,
        "open_positions": _serialize(exposure) if exposure else {},
    }


@app.get("/api/backtest/results")
async def backtest_results(filepath: Optional[str] = Query(default=None)):
    """Load and return backtest results from a JSON file."""
    # Default path
    if not filepath:
        bt_dir = PROJECT_ROOT / "backtest_results"
        if bt_dir.exists():
            files = sorted(bt_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                filepath = str(files[0])
    if not filepath or not Path(filepath).exists():
        raise HTTPException(status_code=404, detail="No backtest results found")

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/list")
async def backtest_list():
    """List available backtest result files."""
    bt_dir = PROJECT_ROOT / "backtest_results"
    if not bt_dir.exists():
        return []
    files = sorted(bt_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "path": str(f),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in files[:20]
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = CONFIG.get("dashboard", {}).get("host", "0.0.0.0")
    port = CONFIG.get("dashboard", {}).get("port", 8080)
    print(f"🚀 Dashboard starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
