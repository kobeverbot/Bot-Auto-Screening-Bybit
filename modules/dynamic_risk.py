"""
Dynamic Risk Management Module
Adjusts leverage, position size, and risk percentage based on asset volatility.

Core idea:
  - Low volatility  → can use more leverage, slightly larger risk
  - High volatility → reduce leverage, reduce risk per trade
  - Uses NATR (Normalized ATR) as the primary volatility metric
  - ADX determines market regime (trending vs ranging)

Integration points:
  - auto_trades.py: calls adjust_for_volatility() before placing orders
  - circuit_breaker.py: uses get_regime() for dynamic exposure limits
"""
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from modules.config_loader import CONFIG

logger = logging.getLogger("DynamicRisk")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VolatilityProfile:
    """Result of volatility analysis for a single symbol."""
    symbol: str
    natr: float              # Normalized ATR (percentage)
    atr: float               # ATR in price units
    adx: float               # ADX trend strength
    regime: str              # "calm", "normal", "volatile", "extreme"
    leverage_multiplier: float  # 1.0 = no change, <1 = reduce, >1 = increase
    risk_multiplier: float     # same idea for risk %
    max_position_multiplier: float  # same for max position size
    recommended_leverage: int  # final leverage suggestion


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {
    "enabled": True,
    "natr_length": 14,
    "adx_length": 14,
    "volatility_buckets": {
        "calm":     {"natr_max": 2.0,  "leverage_mult": 1.3, "risk_mult": 1.2, "max_pos_mult": 1.2},
        "normal":   {"natr_max": 5.0,  "leverage_mult": 1.0, "risk_mult": 1.0, "max_pos_mult": 1.0},
        "volatile": {"natr_max": 10.0, "leverage_mult": 0.6, "risk_mult": 0.7, "max_pos_mult": 0.7},
        "extreme":  {"natr_max": 999.0, "leverage_mult": 0.3, "risk_mult": 0.4, "max_pos_mult": 0.5},
    },
    "adx_threshold": 25.0,       # ADX above = trending
    "trending_leverage_bonus": 0.2,  # extra leverage when strong trend + calm/normal
    "target_leverage": 25,       # base leverage (from trading config)
    "min_leverage": 1,
    "max_leverage": 50,
}


def _load_config() -> dict:
    """Merge user config with defaults."""
    user_cfg = CONFIG.get("dynamic_risk", {})
    cfg = {**_DEFAULT_CFG, **user_cfg}
    # Deep merge volatility buckets (user may override individual buckets)
    default_buckets = _DEFAULT_CFG["volatility_buckets"]
    user_buckets = user_cfg.get("volatility_buckets", {})
    for regime, params in user_buckets.items():
        if regime in default_buckets:
            cfg["volatility_buckets"][regime] = {**default_buckets[regime], **params}
        else:
            cfg["volatility_buckets"][regime] = params
    return cfg


CFG = _load_config()


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def classify_regime(natr: float) -> str:
    """
    Classify market regime based on NATR value.
    Returns one of: 'calm', 'normal', 'volatile', 'extreme'.
    """
    buckets = CFG["volatility_buckets"]
    if natr <= buckets["calm"]["natr_max"]:
        return "calm"
    elif natr <= buckets["normal"]["natr_max"]:
        return "normal"
    elif natr <= buckets["volatile"]["natr_max"]:
        return "volatile"
    else:
        return "extreme"


def analyze_volatility(df: pd.DataFrame, symbol: str = "") -> VolatilityProfile:
    """
    Analyze OHLCV DataFrame and return a VolatilityProfile.
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
        symbol: Trading pair symbol (for logging).
    
    Returns:
        VolatilityProfile with all computed metrics and adjustment factors.
    """
    natr_val = 0.0
    atr_val = 0.0
    adx_val = 0.0

    try:
        # NATR (Normalized ATR as %)
        natr_series = ta.natr(df['high'], df['low'], df['close'], length=CFG["natr_length"])
        natr_val = float(natr_series.iloc[-1]) if not natr_series.empty else 0.0

        # ATR in price units (for SL distance validation)
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=CFG["natr_length"])
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        # ADX (trend strength)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=CFG["adx_length"])
        if adx_df is not None and not adx_df.empty:
            adx_val = float(adx_df['ADX_14'].iloc[-1])
    except Exception as e:
        logger.debug(f"Indicator calculation failed for {symbol}: {e}")

    # Classify regime
    regime = classify_regime(natr_val)
    bucket = CFG["volatility_buckets"].get(regime, CFG["volatility_buckets"]["normal"])

    lev_mult = bucket["leverage_mult"]
    risk_mult = bucket["risk_mult"]
    max_pos_mult = bucket["max_pos_mult"]

    # Trending bonus: if ADX is strong and we're in a calm/normal regime,
    # allow slightly more leverage (the trend is your friend)
    if adx_val >= CFG["adx_threshold"] and regime in ("calm", "normal"):
        lev_mult += CFG["trending_leverage_bonus"]

    # Calculate recommended leverage
    base_lev = CFG["target_leverage"]
    rec_lev = int(base_lev * lev_mult)
    rec_lev = max(CFG["min_leverage"], min(CFG["max_leverage"], rec_lev))

    return VolatilityProfile(
        symbol=symbol,
        natr=natr_val,
        atr=atr_val,
        adx=adx_val,
        regime=regime,
        leverage_multiplier=lev_mult,
        risk_multiplier=risk_mult,
        max_position_multiplier=max_pos_mult,
        recommended_leverage=rec_lev,
    )


# ---------------------------------------------------------------------------
# Adjustment functions (called from auto_trades.py)
# ---------------------------------------------------------------------------

def adjust_for_volatility(
    df: pd.DataFrame,
    symbol: str,
    base_leverage: int,
    base_risk_pct: float,
    base_max_position_pct: float,
    market_max_leverage: int,
) -> dict:
    """
    Main entry point for dynamic risk adjustment.
    
    Args:
        df: Recent OHLCV data for the symbol (at least 30 bars).
        symbol: Trading pair (e.g., 'BTC/USDT:USDT').
        base_leverage: Default leverage from config (e.g., 25).
        base_risk_pct: Default risk % from config (e.g., 0.01).
        base_max_position_pct: Default max position % from config (e.g., 0.05).
        market_max_leverage: Exchange-imposed max leverage for this symbol.
    
    Returns:
        dict with: leverage, risk_pct, max_position_pct, profile (VolatilityProfile)
    """
    if not CFG["enabled"]:
        return {
            "leverage": min(base_leverage, market_max_leverage),
            "risk_pct": base_risk_pct,
            "max_position_pct": base_max_position_pct,
            "profile": None,
        }

    profile = analyze_volatility(df, symbol)

    # Apply multipliers
    leverage = min(
        int(base_leverage * profile.leverage_multiplier),
        market_max_leverage,
        CFG["max_leverage"],
    )
    leverage = max(CFG["min_leverage"], leverage)

    risk_pct = base_risk_pct * profile.risk_multiplier
    max_position_pct = base_max_position_pct * profile.max_position_multiplier

    logger.info(
        f"🎯 {symbol}: regime={profile.regime} | NATR={profile.natr:.2f}% | "
        f"ADX={profile.adx:.1f} | Lev {base_leverage}→{leverage}x | "
        f"Risk {base_risk_pct*100:.2f}%→{risk_pct*100:.2f}%"
    )

    return {
        "leverage": leverage,
        "risk_pct": risk_pct,
        "max_position_pct": max_position_pct,
        "profile": profile,
    }


def get_dynamic_exposure_limit(base_exposure_pct: float, profiles: list) -> float:
    """
    Adjust the overall exposure limit based on current portfolio volatility.
    If many positions are in 'volatile' regime, reduce total exposure.
    
    Args:
        base_exposure_pct: Default max exposure from circuit_breaker config (e.g., 2.0 = 200%).
        profiles: List of VolatilityProfile for all open positions.
    
    Returns:
        Adjusted exposure limit as a multiplier of equity.
    """
    if not CFG["enabled"] or not profiles:
        return base_exposure_pct

    # Count regimes
    n_volatile = sum(1 for p in profiles if p.regime in ("volatile", "extreme"))
    n_total = len(profiles)

    if n_total == 0:
        return base_exposure_pct

    volatile_ratio = n_volatile / n_total

    # If >50% of positions are volatile, reduce exposure by 30%
    # If >75% are volatile, reduce by 50%
    if volatile_ratio > 0.75:
        return base_exposure_pct * 0.5
    elif volatile_ratio > 0.5:
        return base_exposure_pct * 0.7
    elif volatile_ratio > 0.25:
        return base_exposure_pct * 0.85

    return base_exposure_pct


def validate_sl_distance(entry: float, sl: float, atr: float, symbol: str = "") -> tuple:
    """
    Validate that stop-loss distance is reasonable relative to ATR.
    Prevents entering trades where SL is too tight (noise) or too wide (bad R:R).
    
    Returns:
        (ok: bool, reason: str, recommended_sl: float or None)
    """
    if atr <= 0 or entry <= 0:
        return True, "ok", None

    distance = abs(entry - sl)
    distance_atr = distance / atr

    # SL too tight: < 0.5 ATR → likely to get stopped by noise
    if distance_atr < 0.5:
        recommended = entry - 1.0 * atr if sl < entry else entry + 1.0 * atr
        reason = (
            f"SL too tight for {symbol}: {distance_atr:.2f} ATR "
            f"(distance=${distance:.4f}, ATR=${atr:.4f}). "
            f"Recommend {recommended:.4f} (1.0 ATR)"
        )
        return False, reason, recommended

    # SL too wide: > 4.0 ATR → bad risk/reward
    if distance_atr > 4.0:
        recommended = entry - 1.5 * atr if sl < entry else entry + 1.5 * atr
        reason = (
            f"SL too wide for {symbol}: {distance_atr:.2f} ATR. "
            f"Recommend {recommended:.4f} (1.5 ATR)"
        )
        return False, reason, recommended

    return True, "ok", None


def get_risk_summary(profiles: list) -> dict:
    """
    Generate a portfolio-level risk summary for reporting/Discord alerts.
    """
    if not profiles:
        return {"regime": "no_positions", "avg_natr": 0, "avg_adx": 0, "positions_by_regime": {}}

    regimes = [p.regime for p in profiles]
    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    return {
        "regime": max(set(regimes), key=regimes.count),  # mode
        "avg_natr": np.mean([p.natr for p in profiles]),
        "avg_adx": np.mean([p.adx for p in profiles]),
        "positions_by_regime": regime_counts,
        "high_volatility_count": sum(1 for r in regimes if r in ("volatile", "extreme")),
    }
