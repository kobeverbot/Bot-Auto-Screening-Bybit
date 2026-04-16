# 🤖 Bot Auto Screening - Bybit (Fork Mejorado)

Bot de screening automático para perpetuals en Bybit. Escanea cientos de pares USDT buscando setups de trading basados en análisis técnico, Smart Money Concepts (SMC), métricas cuantitativas y datos de derivados.

> **Fork del repo original de [@kurumichan987](https://x.com/kurumichan987)** con mejoras de robustez, seguridad y calidad de código.

---

## 📋 Funcionalidades

- 🔄 **Screening automatizado** de pares USDT perpetuals en Bybit
- 📊 **Análisis técnico**: Patrones chartistas (double top/bottom, flags, triangles, rectangles) + divergencias
- 🧠 **Smart Money Concepts (SMC)**: Order blocks, fair value gaps, liquidity sweeps
- 📈 **Métricas cuantitativas**: Basis, Z-Score, Zeta Score, OBI (Order Book Imbalance)
- 🔥 **Análisis de derivados**: Funding rate, open interest, liquidaciones
- 🐻 **BTC Bias filter**: Solo toma longs en tendencia alcista de BTC, shorts en bajista
- 🛡️ **Circuit breaker**: Protección contra rachas perdedoras y exposición excesiva
- 🚦 **Rate limiter**: Protección contra rate limits de la API de Bybit
- 📲 **Alertas Discord**: Webhook + dashboard en vivo + bot con comandos interactivos

---

## 🔧 Instalación

```bash
git clone https://github.com/tu-user/Bot-Auto-Screening-Bybit.git
cd Bot-Auto-Screening-Bybit

# Crear entorno virtual
python -m venv bot_env
source bot_env/bin/activate  # Linux/Mac
# bot_env\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Copiar y configurar
cp config.example.json config.json
# Editar config.json con tus API keys
```

## ⚙️ Configuración

Edita `config.json` con tus credenciales:

| Sección | Descripción |
|---------|-------------|
| `api` | Keys de Bybit + webhooks de Discord |
| `database` | PostgreSQL para persistencia de señales |
| `system` | Timeframes, threads, intervalo de escaneo |
| `setup` | Niveles Fibonacci para entry/SL/TP |
| `strategy` | Scores mínimos y R:R |
| `circuit_breaker` | Límites de pérdida y exposición |
| `dynamic_risk` | Gestión de riesgo dinámica por volatilidad (nuevo) |
| `rate_limiter` | Límites de llamadas API (nuevo) |

---

## 🚀 Uso

```bash
python main.py
```

El bot ejecutará un scan inmediato y luego repetirá según el intervalo configurado.

---

## 📝 Changelog del Fork

### vs. Original

| # | Mejora | Descripción |
|---|--------|-------------|
| 1 | **Rate Limiter** | Token bucket thread-safe para todas las llamadas API a Bybit. Previene bans por rate limiting durante scans masivos con ThreadPoolExecutor. |
| 2 | **.gitignore robusto** | .gitignore expandido: cubre Python caches, IDEs, OS files, logs, databases. Evita commitear archivos basura. |
| 3 | **config.example.json actualizado** | Documentación completa del nuevo campo `rate_limiter` en el config de ejemplo. |

### 🧠 Ideas de mejora pendientes (sugeridas por el autor original)

El autor menciona varias áreas donde se puede mejorar:

1. ~~**Backtesting**~~ — ✅ **Implementado!** Ver sección Backtesting abajo
2. ~~**Gestión de riesgo más granular**~~ — ✅ **Implementado!** Ver sección Dynamic Risk abajo
3. ~~**Más patrones chartistas**~~ — ✅ **Implementado!** Wedge, head & shoulders, cup & handle, inverse H&S, rising/falling wedge
4. ~~**Machine Learning**~~ — ✅ **Implementado!** Ver sección ML Scoring abajo
5. **Multi-exchange** — Soportar Binance, OKX además de Bybit
6. **Dashboard web** — Interfaz web además de Discord para monitoreo
7. **Notificaciones multi-canal** — Telegram, email además de Discord
8. **Paper trading mode** — Modo simulación para validar señales sin riesgo real

---

## 🧪 Backtesting

El módulo de backtesting permite validar la estrategia contra datos históricos antes de arriesgar capital real. Usa el **mismo pipeline** que el modo live: detección de patrones → technicals → SMC → quant → derivatives.

### Uso rápido

```bash
# Backtest con top 20 pares por volumen, 4h, últimos 90 días
python run_backtest.py

# Backtest con pares específicos
python run_backtest.py --pairs BTC/USDT ETH/USDT SOL/USDT

# Cambiar timeframe y período
python run_backtest.py --tf 1h --days 180

# Limitar pares y exportar resultados
python run_backtest.py --max-pairs 10 --export results.json

# Usando main.py directamente
python main.py --backtest --bt-pairs BTC/USDT --bt-tf 4h --bt-days 90 --bt-export bt_results.json
```

### Configuración (`config.json`)

```json
"backtest": {
    "lookback_days": 90,
    "timeframe": "4h",
    "max_pairs": 20,
    "cooldown_bars": 10,
    "max_bars_simulation": 100
}
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `lookback_days` | 90 | Días hacia atrás para descargar datos |
| `timeframe` | 4h | Timeframe de las velas |
| `max_pairs` | 20 | Máximo de pares a escanear (top por volumen) |
| `cooldown_bars` | 10 | Barras mínimas entre señales en el mismo símbolo |
| `max_bars_simulation` | 100 | Barras máximas para simular cada trade |

### Métricas del reporte

El backtest genera un reporte completo con:

- **Overall**: Win rate, profit factor, avg win/loss, total PnL, max drawdown, Sharpe ratio
- **Exit Reasons**: Distribución de tp2, tp3, stop_loss, timeout
- **Per Pattern**: Rendimiento por tipo de patrón (double_bottom, bull_flag, etc.)
- **Per Side**: Long vs Short
- **Score vs Performance**: Correlación entre total_score y resultados reales

### Exportar resultados

```bash
python run_backtest.py --export backtest_results.json
```

Genera un JSON con todas las señales y trades simulados para análisis posterior.

---

## 🛡️ Dynamic Risk Management

El módulo `dynamic_risk.py` ajusta automáticamente el leverage, tamaño de posición y riesgo por trade basándose en la **volatilidad del activo** (NATR) y la **fuerza de la tendencia** (ADX).

### ¿Cómo funciona?

```
OHLCV Data → NATR (volatilidad) + ADX (tendencia)
         → Clasificación de régimen
            ├─ Calm     (NATR < 2%)  → Leverage ×1.3, Risk ×1.2
            ├─ Normal   (NATR < 5%)  → Leverage ×1.0, Risk ×1.0
            ├─ Volatile (NATR < 10%) → Leverage ×0.6, Risk ×0.7
            └─ Extreme  (NATR > 10%) → Leverage ×0.3, Risk ×0.4
         → Si ADX > 25 (trending) + régimen calm/normal → bonus ×0.2 leverage
         → Validación de SL vs ATR (muy tight < 0.5 ATR, muy ancho > 4 ATR)
```

### Ejemplo práctico

| Activo | NATR | Régimen | Base Lev | Lev Final | Base Risk | Risk Final |
|--------|------|---------|----------|-----------|-----------|------------|
| BTC/USDT | 1.5% | Calm + Trending | 25x | 37x | 1.0% | 1.2% |
| ETH/USDT | 3.2% | Normal | 25x | 25x | 1.0% | 1.0% |
| DOGE/USDT | 8.5% | Volatile | 25x | 15x | 1.0% | 0.7% |
| PEPE/USDT | 15% | Extreme | 25x | 7x | 1.0% | 0.4% |

### Exposición dinámica del portfolio

El circuit breaker se beneficia del módulo de riesgo dinámico:
- Si >25% de posiciones están en régimen volatile → exposición total se reduce 15%
- Si >50% → reducción del 30%
- Si >75% → reducción del 50%

### Configuración (`config.json`)

```json
"dynamic_risk": {
    "enabled": true,
    "natr_length": 14,
    "adx_length": 14,
    "volatility_buckets": {
        "calm":     {"natr_max": 2.0,  "leverage_mult": 1.3, "risk_mult": 1.2, "max_pos_mult": 1.2},
        "normal":   {"natr_max": 5.0,  "leverage_mult": 1.0, "risk_mult": 1.0, "max_pos_mult": 1.0},
        "volatile": {"natr_max": 10.0, "leverage_mult": 0.6, "risk_mult": 0.7, "max_pos_mult": 0.7},
        "extreme":  {"natr_max": 999.0, "leverage_mult": 0.3, "risk_mult": 0.4, "max_pos_mult": 0.5}
    },
    "adx_threshold": 25.0,
    "trending_leverage_bonus": 0.2,
    "target_leverage": 25,
    "min_leverage": 1,
    "max_leverage": 50
}
```

---

## 🧠 ML Scoring

El módulo `ml_scorer.py` añade una capa de Machine Learning sobre el scoring heurístico existente. Usa un modelo entrenado con señales pasadas y sus resultados reales para mejorar la calidad de las señales.

### ¿Cómo funciona?

```
Señal (scores + pattern + side + bias)
         → Feature engineering (28 features)
            ├─ 9 numéricas: tech, smc, quant, deriv, z_score, zeta, obi, basis, rr
            ├─ 3 derivadas: total_score, score_balance, smc_tech_ratio
            ├─ 13 one-hot: patrones chartistas
            ├─ 1 binaria: side (long/short)
            └─ 2 binarias: btc_bias (bull/bear)
         → Modelo entrenado (GradientBoosting / RandomForest / LogisticRegression)
         → Probabilidad de win
         → ML Score (0-5 en modo augment, o reemplaza en modo replace)
```

### Dos modos de operación

| Modo | Comportamiento |
|------|---------------|
| **`augment`** (default) | Añade 0-5 puntos extra al total_score basado en la confianza del modelo |
| **`replace`** | Reemplaza el scoring heurístico con el scoring ML usando pesos configurables |

### Entrenar el modelo

```bash
# Desde resultados de backtest exportado
python train_model.py --source export --file backtest_results.json

# Backtest + entrenamiento en un solo paso
python train_model.py --source backtest --pairs BTC/USDT ETH/USDT --days 180

# Desde base de datos (requiere PostgreSQL configurado)
python train_model.py --source db

# Especificar tipo de modelo
python train_model.py --source export --file results.json --model random_forest
python train_model.py --source export --file results.json --model logistic
```

### Configuración (`config.json`)

```json
"ml_scoring": {
    "enabled": true,
    "mode": "augment",
    "model_path": "ml_models/signal_model.json",
    "min_training_samples": 50,
    "augment_score_range": [0, 5],
    "replace_score_weights": {
        "tech_score": 1.0, "smc_score": 1.2,
        "quant_score": 0.8, "deriv_score": 1.0,
        "ml_confidence": 2.0
    },
    "heuristic_weights": {
        "tech_score": 1.0, "smc_score": 1.5,
        "quant_score": 0.8, "deriv_score": 1.2,
        "zeta_score": 0.03, "z_score": 0.1,
        "obi": 1.5, "rr_bonus": 0.5
    }
}
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `enabled` | `true` | Activar/desactivar ML scoring |
| `mode` | `"augment"` | `"augment"` o `"replace"` |
| `model_path` | `"ml_models/signal_model.json"` | Path al modelo entrenado |
| `min_training_samples` | `50` | Mínimo de muestras para entrenar |
| `augment_score_range` | `[0, 5]` | Rango del bonus ML en modo augment |

### Fallback heurístico

Si no hay modelo entrenado, el sistema usa un **scoring heurístico ponderado** con confluence bonus:
- Cada componente (tech, smc, quant, deriv) tiene un peso configurable
- Si 4+ componentes están activos → +15% confluence bonus
- Si 3 componentes → +5% bonus

---

## 📊 Dashboard Web

Dashboard web integrado con FastAPI para monitorear señales, trades, rendimiento de patrones, estado del modelo ML y métricas de riesgo en tiempo real.

### ¿Qué incluye?

```
dashboard/
├── __init__.py
├── api.py              # Backend FastAPI (11 endpoints)
├── templates/
│   └── index.html      # Frontend single-page (Chart.js + dark theme)
└── static/             # Assets estáticos
```

### Tabs del dashboard

| Tab | Contenido |
|-----|-----------|
| **Overview** | KPIs generales, señales diarias, distribución de scores, últimas señales |
| **Signals** | Tabla completa de señales con filtros, scores detallados (tech/smc/quant/deriv/ML) |
| **Patterns** | Win rate por patrón, uso de patrones (doughnut), tabla de rendimiento |
| **ML Model** | Estado del modelo, feature importance, tipo de modelo, métricas de entrenamiento |
| **Risk** | Posiciones abiertas, circuit breaker, rate limiter, configuración de riesgo |
| **Backtest** | Resultados de backtests (equity curve, exit reasons, tabla de trades) |

### Ejecutar

```bash
# Opción 1: Junto con el bot (recomendado)
python main.py  # El dashboard arranca automáticamente si está enabled en config

# Opción 2: Solo el dashboard
cd dashboard && uvicorn api:app --host 0.0.0.0 --port 8080 --reload
```

### Configuración (`config.json`)

```json
"dashboard": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 8080
}
```

### API Endpoints

```
GET /                          → Dashboard HTML
GET /api/health                → Health check + DB status
GET /api/signals/recent        → Últimas señales (?limit=N&status=...)
GET /api/signals/active        → Señales activas (Waiting Entry / Active)
GET /api/stats/overview        → KPIs generales + últimas 24h
GET /api/stats/daily           → Señales diarias agrupadas (?days=N)
GET /api/stats/score-distribution → Distribución de scores
GET /api/stats/pattern-performance → Win rate y métricas por patrón
GET /api/ml/status             → Estado del modelo ML
GET /api/ml/feature-importance → Importancia de features
GET /api/risk/overview         → Posiciones, circuit breaker, rate limiter
GET /api/backtest/list         → Lista de archivos de backtest
GET /api/backtest/results      → Resultados de un backtest (?filepath=...)
```

El dashboard se auto-refresca cada 30 segundos.

---

## ⚠️ Disclaimer

Este bot es una herramienta de análisis y screening. **No es consejo financiero**. Trading de criptomonedas conlleva riesgo significativo. Usa siempre gestión de riesgo apropiada y nunca arriesgues lo que no puedes permitirte perder.

---

## 📜 Licencia

Ver [LICENSE](LICENSE) — Proyecto original de [@kurumichan987](https://x.com/kurumichan987).
