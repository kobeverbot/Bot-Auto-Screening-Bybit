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

1. **Gestión de riesgo más granular** — Ajuste dinámico de leverage según volatilidad del activo
2. **Backtesting** — Implementar validación histórica de los setups antes de alertar
3. **Más patrones chartistas** — Agregar wedge, head & shoulders, cup & handle
4. **Machine Learning** — Score compuesto con modelo entrenado en señales pasadas
5. **Multi-exchange** — Soportar Binance, OKX además de Bybit
6. **Dashboard web** — Interfaz web además de Discord para monitoreo
7. **Notificaciones multi-canal** — Telegram, email además de Discord
8. **Paper trading mode** — Modo simulación para validar señales sin riesgo real

---

## ⚠️ Disclaimer

Este bot es una herramienta de análisis y screening. **No es consejo financiero**. Trading de criptomonedas conlleva riesgo significativo. Usa siempre gestión de riesgo apropiada y nunca arriesgues lo que no puedes permitirte perder.

---

## 📜 Licencia

Ver [LICENSE](LICENSE) — Proyecto original de [@kurumichan987](https://x.com/kurumichan987).
