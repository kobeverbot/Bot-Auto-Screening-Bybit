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
2. **Gestión de riesgo más granular** — Ajuste dinámico de leverage según volatilidad del activo
3. **Más patrones chartistas** — Agregar wedge, head & shoulders, cup & handle
4. **Machine Learning** — Score compuesto con modelo entrenado en señales pasadas
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

## ⚠️ Disclaimer

Este bot es una herramienta de análisis y screening. **No es consejo financiero**. Trading de criptomonedas conlleva riesgo significativo. Usa siempre gestión de riesgo apropiada y nunca arriesgues lo que no puedes permitirte perder.

---

## 📜 Licencia

Ver [LICENSE](LICENSE) — Proyecto original de [@kurumichan987](https://x.com/kurumichan987).
