"""
ML Scoring Module — Composite Signal Quality Predictor

Learns from historical signals + outcomes (backtest or live) to produce an
ML-quality score that augments or replaces the simple additive scoring.

Two modes:
  - "augment" (default): adds ml_score (0-5 bonus) to existing total_score
  - "replace": replaces total_score with ML-weighted prediction

Model: Gradient Boosting (sklearn) — works well with small datasets, handles
       mixed feature types, no scaling needed. Falls back to heuristic scoring
       if no trained model exists.

Training data: Export from backtest via `python train_model.py --export training.json`
"""
import json
import logging
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from modules.config_loader import CONFIG

logger = logging.getLogger("MLScorer")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {
    "enabled": False,
    "mode": "augment",          # "augment" or "replace"
    "model_path": "ml_models/signal_model.json",
    "min_training_samples": 50,
    "augment_score_range": [0, 5],  # ml_score bonus range
    "replace_score_weights": {
        "tech_score": 1.0,
        "smc_score": 1.2,
        "quant_score": 0.8,
        "deriv_score": 1.0,
        "ml_confidence": 2.0,
    },
    "heuristic_weights": {
        "tech_score": 1.0,
        "smc_score": 1.5,
        "quant_score": 0.8,
        "deriv_score": 1.2,
        "zeta_score": 0.03,
        "z_score": 0.1,
        "obi": 1.5,
        "rr_bonus": 0.5,
    },
    "features": {
        "numeric": [
            "tech_score", "smc_score", "quant_score", "deriv_score",
            "z_score", "zeta_score", "obi", "basis", "rr",
        ],
        "categorical": ["pattern", "side", "btc_bias"],
        "derived": ["total_score", "score_balance", "smc_tech_ratio"],
    },
}

_CFG = None

def _get_cfg():
    global _CFG
    if _CFG is None:
        user_cfg = CONFIG.get("ml_scoring", {})
        _CFG = {**_DEFAULT_CFG, **user_cfg}
    return _CFG


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

# All known pattern names (for consistent one-hot encoding)
KNOWN_PATTERNS = sorted([
    "double_bottom", "double_top",
    "bull_flag", "bear_flag",
    "ascending_triangle", "descending_triangle",
    "bull_rectangle", "bear_rectangle",
    "ascending_wedge", "descending_wedge",
    "head_shoulders", "inverse_head_shoulders",
    "cup_handle",
])

PATTERN_TO_IDX = {p: i for i, p in enumerate(KNOWN_PATTERNS)}


def build_feature_vector(signal: dict) -> np.ndarray:
    """
    Convert a signal dict into a numeric feature vector for ML prediction.

    Features (22 dimensions):
      0-3:   tech_score, smc_score, quant_score, deriv_score
      4-8:   z_score, zeta_score, obi, basis, rr
      9:     total_score (sum of component scores)
      10:    score_balance (std dev of component scores — confluence measure)
      11:    smc_tech_ratio (smc_score / max(tech_score, 1))
      12-23: one-hot pattern encoding (12 patterns, padded)
      24:    side_long (1=Long, 0=Short)
      25:    btc_bias_bull (1=Bullish, 0=other)
      26:    btc_bias_bear (1=Bearish, 0=other)
    """
    cfg = _get_cfg()
    features = []

    # Numeric features (with safe defaults)
    tech = signal.get("tech_score", signal.get("Tech_Score", 0))
    smc = signal.get("smc_score", signal.get("SMC_Score", 0))
    quant = signal.get("quant_score", signal.get("Quant_Score", 0))
    deriv = signal.get("deriv_score", signal.get("Deriv_Score", 0))
    z_score = signal.get("z_score", signal.get("Z_Score", 0))
    zeta = signal.get("zeta_score", signal.get("Zeta_Score", 50))
    obi = signal.get("obi", signal.get("OBI", 0))
    basis = signal.get("basis", signal.get("Basis", 0))
    rr = signal.get("rr", signal.get("RR", 0))

    features.extend([
        float(tech), float(smc), float(quant), float(deriv),
        float(z_score), float(zeta), float(obi), float(basis), float(rr),
    ])

    # Derived features
    total = tech + smc + quant + deriv
    components = [tech, smc, quant, deriv]
    balance = float(np.std(components)) if components else 0.0
    smc_tech = float(smc) / max(float(tech), 1.0)

    features.extend([float(total), balance, smc_tech])

    # Pattern one-hot (fixed 12 dimensions)
    pattern = signal.get("pattern", signal.get("Pattern", "")).lower().replace(" ", "_")
    pattern_vec = [0.0] * len(KNOWN_PATTERNS)
    if pattern in PATTERN_TO_IDX:
        pattern_vec[PATTERN_TO_IDX[pattern]] = 1.0
    features.extend(pattern_vec)

    # Side encoding
    side = signal.get("side", signal.get("Side", "")).lower()
    features.append(1.0 if side == "long" else 0.0)

    # BTC bias encoding
    bias = signal.get("btc_bias", signal.get("BTC_Bias", "")).lower()
    features.append(1.0 if "bull" in bias else 0.0)
    features.append(1.0 if "bear" in bias else 0.0)

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """Metadata stored alongside the trained model."""
    n_samples: int
    win_rate: float
    avg_pnl_pct: float
    feature_importance: dict
    trained_at: str
    model_type: str
    version: str = "1.0"


_model = None
_metadata = None


def _model_path() -> Path:
    cfg = _get_cfg()
    return Path(cfg["model_path"])


def load_model() -> bool:
    """Load a trained model from disk. Returns True if successful."""
    global _model, _metadata
    path = _model_path()

    if not path.exists():
        logger.info(f"No ML model found at {path}. Using heuristic scoring.")
        return False

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct model from __getstate__ pickle
        model_type = data.get("model_type", "gradient_boosting")

        import pickle, base64
        if "model_bytes" not in data:
            logger.warning("No model_bytes found in model file")
            return False

        state = pickle.loads(base64.b64decode(data["model_bytes"]))

        # Create empty model instance and restore state
        if model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            _model = GradientBoostingClassifier()
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            _model = RandomForestClassifier()
        elif model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            _model = LogisticRegression()
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return False

        _model.__setstate__(state)

        meta = data.get("metadata", {})
        _metadata = ModelMetadata(
            n_samples=meta.get("n_samples", 0),
            win_rate=meta.get("win_rate", 0),
            avg_pnl_pct=meta.get("avg_pnl_pct", 0),
            feature_importance=meta.get("feature_importance", {}),
            trained_at=meta.get("trained_at", ""),
            model_type=model_type,
        )

        logger.info(
            f"✅ ML model loaded: {_metadata.model_type} | "
            f"{_metadata.n_samples} samples | Win rate: {_metadata.win_rate:.1%} | "
            f"Trained: {_metadata.trained_at}"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to load ML model: {e}. Using heuristic scoring.")
        _model = None
        return False


def save_model(model, metadata: ModelMetadata) -> bool:
    """Save a trained model to disk (joblib binary + JSON metadata)."""
    path = _model_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pickle, base64
        import numpy as np

        # Serialize via __getstate__, replacing unpicklable Cython objects
        state = model.__getstate__()
        state["_rng"] = np.random.RandomState(state.get("random_state", 42))

        if hasattr(model, "_loss"):
            # Reconstruct _loss from sklearn internal module
            loss_name = state.get("loss", "log_loss")
            try:
                from sklearn._loss import HalfBinomialLoss, HalfMultinomialLoss
                if state.get("n_classes_", 2) <= 2:
                    state["_loss"] = HalfBinomialLoss()
                else:
                    state["_loss"] = HalfMultinomialLoss()
            except ImportError:
                # Fallback: skip _loss entirely (will be rebuilt on predict)
                state.pop("_loss", None)
                state.pop("_rng", None)

        model_bytes = base64.b64encode(pickle.dumps(state)).decode('ascii')

        # Save metadata JSON with embedded model bytes
        model_path = path  # single JSON file
        data = {
            "model_type": metadata.model_type,
            "model_bytes": model_bytes,
            "metadata": {
                "n_samples": metadata.n_samples,
                "win_rate": metadata.win_rate,
                "avg_pnl_pct": metadata.avg_pnl_pct,
                "feature_importance": metadata.feature_importance,
                "trained_at": metadata.trained_at,
                "version": metadata.version,
            },
            "feature_names": _get_feature_names(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✅ ML model saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save ML model: {e}")
        return False

def _get_feature_names():
    """Return list of feature names for interpretation."""
    names = [
        "tech_score", "smc_score", "quant_score", "deriv_score",
        "z_score", "zeta_score", "obi", "basis", "rr",
        "total_score", "score_balance", "smc_tech_ratio",
    ]
    names.extend([f"pattern_{p}" for p in KNOWN_PATTERNS])
    names.extend(["side_long", "bias_bull", "bias_bear"])
    return names


# ---------------------------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------------------------

def _heuristic_score(signal: dict) -> float:
    """
    Heuristic scoring — used when no trained model exists.
    Applies learned weights to produce a quality-adjusted score.
    """
    cfg = _get_cfg()
    w = cfg["heuristic_weights"]

    tech = float(signal.get("tech_score", signal.get("Tech_Score", 0)))
    smc = float(signal.get("smc_score", signal.get("SMC_Score", 0)))
    quant = float(signal.get("quant_score", signal.get("Quant_Score", 0)))
    deriv = float(signal.get("deriv_score", signal.get("Deriv_Score", 0)))
    zeta = float(signal.get("zeta_score", signal.get("Zeta_Score", 50)))
    z = float(signal.get("z_score", signal.get("Z_Score", 0)))
    obi = float(signal.get("obi", signal.get("OBI", 0)))
    rr = float(signal.get("rr", signal.get("RR", 0)))

    # Weighted score
    score = (
        tech * w.get("tech_score", 1.0) +
        smc * w.get("smc_score", 1.5) +
        quant * w.get("quant_score", 0.8) +
        deriv * w.get("deriv_score", 1.2) +
        zeta * w.get("zeta_score", 0.03) +
        z * w.get("z_score", 0.1) +
        obi * w.get("obi", 1.5) +
        min(rr, 5.0) * w.get("rr_bonus", 0.5)
    )

    # Confluence bonus: if all 4 components are non-zero
    components = [tech, smc, quant, deriv]
    active = sum(1 for c in components if c > 0)
    if active >= 4:
        score *= 1.15  # 15% confluence bonus
    elif active >= 3:
        score *= 1.05

    return score


def predict_probability(signal: dict) -> float:
    """
    Predict win probability for a signal (0.0 to 1.0).
    Uses trained model if available, otherwise heuristic normalization.
    """
    if _model is not None:
        try:
            features = build_feature_vector(signal).reshape(1, -1)
            proba = _model.predict_proba(features)
            # predict_proba returns [[prob_class_0, prob_class_1]]
            if proba.shape[1] == 2:
                return float(proba[0, 1])  # probability of win
            return float(proba[0, 0])
        except Exception as e:
            logger.debug(f"ML prediction failed, using heuristic: {e}")

    # Fallback: normalize heuristic score to [0, 1]
    h_score = _heuristic_score(signal)
    # Heuristic typically ranges 5-20, normalize to 0-1
    return min(max(h_score / 20.0, 0.0), 1.0)


def get_ml_score(signal: dict) -> dict:
    """
    Main entry point — returns ML scoring result.

    Returns dict:
        - ml_score: int (0-5 bonus for augment mode, or full score for replace mode)
        - win_probability: float (0.0-1.0)
        - mode: str ("augment" or "replace")
        - model_used: bool (True if trained model, False if heuristic)
    """
    cfg = _get_cfg()

    if not cfg["enabled"]:
        return {
            "ml_score": 0,
            "win_probability": 0.0,
            "mode": "disabled",
            "model_used": False,
        }

    mode = cfg["mode"]
    win_prob = predict_probability(signal)

    if mode == "augment":
        # Convert win probability to bonus score (0-5)
        score_min, score_max = cfg["augment_score_range"]
        # Linear mapping: prob 0.5 → 0, prob 1.0 → score_max, prob 0.0 → score_min
        if win_prob >= 0.5:
            ml_score = score_min + (win_prob - 0.5) * 2.0 * (score_max - score_min)
        else:
            ml_score = score_min
        ml_score = max(score_min, min(score_max, int(round(ml_score))))

    elif mode == "replace":
        # Replace mode: ML confidence becomes the score modifier
        weights = cfg["replace_score_weights"]
        tech = float(signal.get("tech_score", signal.get("Tech_Score", 0)))
        smc = float(signal.get("smc_score", signal.get("SMC_Score", 0)))
        quant = float(signal.get("quant_score", signal.get("Quant_Score", 0)))
        deriv = float(signal.get("deriv_score", signal.get("Deriv_Score", 0)))

        weighted = (
            tech * weights.get("tech_score", 1.0) +
            smc * weights.get("smc_score", 1.2) +
            quant * weights.get("quant_score", 0.8) +
            deriv * weights.get("deriv_score", 1.0) +
            win_prob * 100 * weights.get("ml_confidence", 2.0)
        )
        ml_score = int(round(weighted))
    else:
        ml_score = 0

    return {
        "ml_score": ml_score,
        "win_probability": round(win_prob, 4),
        "mode": mode,
        "model_used": _model is not None,
    }


# ---------------------------------------------------------------------------
# Training Helpers (used by train_model.py)
# ---------------------------------------------------------------------------

def prepare_training_data(signals: list, trades: list) -> tuple:
    """
    Join signals with their trade outcomes to create ML training data.

    Args:
        signals: List of signal dicts from backtest (features).
        trades: List of trade dicts from backtest (outcomes).

    Returns:
        (X, y, feature_names) — numpy arrays ready for sklearn.
    """
    # Match signals to trades by symbol + timestamp
    trade_map = {}
    for t in trades:
        key = f"{t['symbol']}_{t.get('bar_ts', '')}"
        trade_map[key] = t

    X_list = []
    y_list = []

    for sig in signals:
        key = f"{sig.get('symbol', '')}_{sig.get('bar_ts', '')}"
        trade = trade_map.get(key)

        if trade is None:
            continue

        features = build_feature_vector(sig)
        is_win = 1 if trade.get("is_win", False) else 0

        X_list.append(features)
        y_list.append(is_win)

    if not X_list:
        return np.array([]), np.array([]), []

    X = np.array(X_list)
    y = np.array(y_list)
    feature_names = _get_feature_names()

    logger.info(f"📊 Training data prepared: {len(X)} samples, {len(feature_names)} features, "
                f"win rate: {y.mean():.1%}")

    return X, y, feature_names


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = "gradient_boosting") -> tuple:
    """
    Train an ML model on prepared features + labels.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels (0=loss, 1=win).
        model_type: One of "gradient_boosting", "random_forest", "logistic".

    Returns:
        (model, metadata) — trained sklearn model and ModelMetadata.
    """
    from sklearn.model_selection import cross_val_score
    from datetime import datetime

    cfg = _get_cfg()

    if len(X) < cfg["min_training_samples"]:
        raise ValueError(
            f"Insufficient training data: {len(X)} samples "
            f"(minimum: {cfg['min_training_samples']})"
        )

    # Create model
    if model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
        )
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.fit(X, y)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 10), scoring='accuracy')

    # Feature importance
    importance = {}
    feature_names = _get_feature_names()
    try:
        if hasattr(model, 'feature_importances_'):
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = round(float(imp), 4)
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['lr'], 'coef_'):
            for name, coef in zip(feature_names, model.named_steps['lr'].coef_[0]):
                importance[name] = round(float(coef), 4)
    except Exception:
        pass

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))

    metadata = ModelMetadata(
        n_samples=len(X),
        win_rate=float(y.mean()),
        avg_pnl_pct=0.0,  # filled by caller if available
        feature_importance=importance,
        trained_at=datetime.utcnow().isoformat(),
        model_type=model_type,
    )

    logger.info(
        f"🧠 Model trained: {model_type} | "
        f"CV accuracy: {cv_scores.mean():.1%} (±{cv_scores.std():.1%}) | "
        f"Top features: {list(importance.keys())[:5]}"
    )

    return model, metadata


# ---------------------------------------------------------------------------
# Auto-load on import
# ---------------------------------------------------------------------------

def is_model_loaded() -> bool:
    return _model is not None


def get_model_info() -> dict:
    """Return info about the current model state (for status/health checks)."""
    if _metadata:
        return {
            "loaded": True,
            "type": _metadata.model_type,
            "samples": _metadata.n_samples,
            "win_rate": _metadata.win_rate,
            "trained_at": _metadata.trained_at,
            "top_features": list(_metadata.feature_importance.keys())[:5],
        }
    return {"loaded": False, "type": "heuristic_fallback"}
