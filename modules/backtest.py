"""
Backtesting Engine for Bot-Auto-Screening-Bybit.

Replays historical data through the same screening pipeline used in live mode:
  Pattern Detection → Technicals → SMC → Quant → Derivatives → Setup Calculation

Then simulates trades with realistic entry/SL/TP hit logic and produces
comprehensive performance metrics.
"""
import json
import logging
import time
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta

from modules.config_loader import CONFIG
from modules.technicals import get_technicals, detect_divergence
from modules.quant import calculate_metrics, check_fakeout
from modules.derivatives import analyze_derivatives
from modules.smc import analyze_smc
from modules.patterns import find_pattern
from modules.ml_scorer import get_ml_score, load_model as load_ml_model

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────

def fetch_historical_ohlcv(exchange, symbol, timeframe, start_date, end_date):
    """
    Fetch full historical OHLCV data between two dates.
    Handles pagination via ccxt's since/limit mechanism.
    """
    all_bars = []
    since = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    limit = 1000  # Bybit max per request

    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    while since < end_ms:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not bars:
                break
            # Filter out bars beyond end date
            bars = [b for b in bars if b[0] <= end_ms]
            all_bars.extend(bars)
            # Move to next batch
            last_ts = bars[-1][0]
            if last_ts <= since:
                break
            since = last_ts + tf_ms
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limit
        except Exception as e:
            logger.warning(f"fetch_ohlcv error {symbol} @ {since}: {e}")
            time.sleep(2)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop_duplicates(subset='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_top_pairs(exchange, quote='USDT', limit=50):
    """Get top pairs by 24h volume for backtesting."""
    try:
        tickers = exchange.fetch_tickers()
        pairs = []
        STABLECOINS = ['USDC', 'USDT', 'DAI', 'FDUSD', 'USDD', 'USDE', 'TUSD', 'BUSD', 'PYUSD', 'USDS', 'EUR', 'USD']

        for symbol, ticker in tickers.items():
            market = exchange.market(symbol)
            if not market.get('swap'):
                continue
            if market.get('quote') != quote:
                continue
            if not market.get('active'):
                continue
            if market.get('base') in STABLECOINS:
                continue
            vol = ticker.get('quoteVolume', 0) or 0
            pairs.append((symbol, vol))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in pairs[:limit]]
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return []


# ──────────────────────────────────────────────
# Backtest Signal Detection (Offline)
# ──────────────────────────────────────────────

def analyze_bar_offline(df, idx, btc_bias, config):
    """
    Run the screening pipeline on a historical bar (index idx).
    Returns a signal dict or None — same logic as main.analyze_ticker
    but without live API calls (derivatives/funding approximated from df).
    """
    lookback = config.get('min_candles_analysis', 150)
    if idx < lookback:
        return None

    slice_df = df.iloc[idx - lookback:idx + 1].copy()
    if len(slice_df) < lookback:
        return None

    # ── Technicals ──
    slice_df = get_technicals(slice_df)
    if slice_df.empty:
        return None

    # ── Pattern ──
    pattern = find_pattern(slice_df)
    if not pattern:
        return None
    side = config.get('pattern_signals', {}).get(pattern)
    if not side:
        return None

    # ── BTC Bias filter ──
    if "Bearish" in btc_bias and side == "Long":
        return None
    if "Bullish" in btc_bias and side == "Short":
        return None

    # ── SMC ──
    valid_smc, smc_score, smc_reasons = analyze_smc(slice_df, side)
    min_smc = CONFIG['strategy'].get('min_smc_score', 0)
    if smc_score < min_smc:
        return None

    # ── Fakeout check (using volume from df) ──
    vol_sma = slice_df['volume'].rolling(20).mean().iloc[-1]
    rvol = slice_df['volume'].iloc[-1] / vol_sma if vol_sma > 0 else 0
    slice_df['Vol_SMA'] = slice_df['volume'].rolling(20).mean()
    slice_df['RVOL'] = slice_df['volume'] / slice_df['Vol_SMA']
    min_rvol = config.get('indicators', {}).get('min_rvol', 2.0)
    if rvol < min_rvol:
        return None

    # ── Quant (approximate — no live ticker) ──
    z_score_val = 0
    zeta_score_val = 50
    obi_val = 0
    basis_val = 0
    quant_score = 2
    quant_reasons = []

    if rvol > 5.0:
        quant_score += 1
        quant_reasons.append("Nuclear RVOL")
    elif rvol > 2.0:
        quant_reasons.append("Valid RVOL")

    # ── Derivatives (approximate — use CVD from df) ──
    deriv_score = 1
    deriv_reasons = []
    delta = np.where(slice_df['close'] > slice_df['open'], slice_df['volume'], -slice_df['volume'])
    cvd = pd.Series(delta).cumsum()
    p_slope = _get_slope(slice_df['close'].iloc[-10:])
    cvd_slope = _get_slope(cvd.iloc[-10:])

    if p_slope > 0 and cvd_slope < 0:
        if side == "Short":
            deriv_score += 2
            deriv_reasons.append("Bear CVD Div")
        elif side == "Long":
            deriv_score -= 2
    elif p_slope < 0 and cvd_slope > 0:
        if side == "Long":
            deriv_score += 2
            deriv_reasons.append("Bull CVD Div")
        elif side == "Short":
            deriv_score -= 2

    if deriv_score <= 0:
        return None

    # ── Divergence ──
    div_score, div_msg = detect_divergence(slice_df)
    tech_score = 3 + div_score
    tech_reasons = [f"Pattern: {pattern}", div_msg] + [r for r in smc_reasons if r]

    total_score = tech_score + smc_score + quant_score + deriv_score

    # ML Score Augmentation (same logic as main.py for consistency)
    ml_result = get_ml_score({
        "tech_score": tech_score, "smc_score": smc_score,
        "quant_score": quant_score, "deriv_score": deriv_score,
        "z_score": z_score, "zeta_score": zeta_score,
        "obi": obi, "basis": basis, "rr": 0,
        "pattern": pattern, "side": side, "btc_bias": "Sideways",
    })
    if ml_result["mode"] == "augment":
        total_score += ml_result["ml_score"]
    elif ml_result["mode"] == "replace":
        total_score = ml_result["ml_score"]

    min_tech = CONFIG['strategy'].get('min_tech_score', 5)
    if tech_score < min_tech:
        return None

    # ── Setup Calculation ──
    s = CONFIG['setup']
    window = min(50, len(slice_df))
    swing_high = slice_df['high'].iloc[-window:].max()
    swing_low = slice_df['low'].iloc[-window:].min()
    rng = swing_high - swing_low

    if rng <= 0:
        return None

    if side == 'Long':
        entry = (swing_high - (rng * s['fib_entry_start']) + swing_high - (rng * s['fib_entry_end'])) / 2
        sl = swing_low - (rng * s['fib_sl'])
        tp1 = swing_low + rng
        tp2 = swing_low + (rng * 1.618)
        tp3 = swing_low + (rng * 2.618)
    else:
        entry = (swing_low + (rng * s['fib_entry_start']) + swing_low + (rng * s['fib_entry_end'])) / 2
        sl = swing_high + (rng * s['fib_sl'])
        tp1 = swing_high - rng
        tp2 = swing_high - (rng * 1.618)
        tp3 = swing_high - (rng * 2.618)

    rr = _calculate_rr(entry, sl, tp3)
    min_rr = CONFIG['strategy'].get('risk_reward_min', 2.0)
    if rr < min_rr:
        return None

    return {
        "symbol": "",  # filled by caller
        "side": side,
        "timeframe": "",  # filled by caller
        "pattern": pattern,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "rr": float(rr),
        "tech_score": int(tech_score),
        "quant_score": int(quant_score),
        "deriv_score": int(deriv_score),
        "smc_score": int(smc_score),
        "total_score": int(total_score),
        "ml_score": int(ml_result.get("ml_score", 0)),
        "btc_bias": btc_bias,
        "bar_idx": idx,
        "bar_ts": str(df['timestamp'].iloc[idx]),
        "tech_reasons": ", ".join(tech_reasons),
        "quant_reasons": ", ".join(quant_reasons),
        "deriv_reasons": ", ".join(deriv_reasons),
        "smc_reasons": ", ".join([r for r in smc_reasons if r]),
    }


def _get_slope(values):
    """Simple linear regression slope."""
    try:
        from scipy.stats import linregress
        return linregress(np.arange(len(values)), np.array(values))[0]
    except Exception:
        return 0.0


def _calculate_rr(entry, sl, tp3):
    if entry <= 0 or sl <= 0 or tp3 <= 0:
        return 0.0
    risk = abs(entry - sl)
    return round(abs(tp3 - entry) / risk, 2) if risk > 0 else 0.0


# ──────────────────────────────────────────────
# Trade Simulation
# ──────────────────────────────────────────────

def simulate_trade(signal, future_df, max_bars=100):
    """
    Simulate a trade using future bars after signal.

    Logic:
      - Check if entry price is hit within max_bars
      - Once entry hit, check SL/TP hits bar-by-bar
      - Track partial TP hits (TP1, TP2, TP3)
      - Return trade result dict

    Args:
        signal: Signal dict with entry/sl/tp1/tp2/tp3/side
        future_df: DataFrame with bars AFTER the signal bar
        max_bars: Max bars to wait for entry

    Returns:
        Trade result dict or None if entry never hit
    """
    side = signal['side']
    entry = signal['entry']
    sl = signal['sl']
    tp1 = signal['tp1']
    tp2 = signal['tp2']
    tp3 = signal['tp3']

    entry_bar = None
    # Phase 1: Wait for entry
    for i in range(min(max_bars, len(future_df))):
        bar = future_df.iloc[i]
        if side == 'Long':
            if bar['low'] <= entry <= bar['high']:
                entry_bar = i
                break
        else:  # Short
            if bar['low'] <= entry <= bar['high']:
                entry_bar = i
                break

    if entry_bar is None:
        return None

    # Phase 2: Track SL/TP hits
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    sl_hit = False
    exit_price = entry  # default
    exit_reason = "timeout"
    exit_bar = len(future_df) - 1

    for i in range(entry_bar + 1, min(entry_bar + max_bars, len(future_df))):
        bar = future_df.iloc[i]
        high, low, close = bar['high'], bar['low'], bar['close']

        if side == 'Long':
            # Check SL first (conservative)
            if low <= sl:
                sl_hit = True
                exit_price = sl
                exit_reason = "stop_loss"
                exit_bar = i
                break
            # Check TPs in order
            if not tp1_hit and high >= tp1:
                tp1_hit = True
            if not tp2_hit and high >= tp2:
                tp2_hit = True
            if not tp3_hit and high >= tp3:
                tp3_hit = True
                exit_price = tp3
                exit_reason = "tp3"
                exit_bar = i
                break
            # If TP2 hit and we have a trailing mechanism, hold
            # For simplicity: close at TP2 if hit
            if tp2_hit and not tp3_hit:
                # Close at TP2 price (simplified)
                exit_price = tp2
                exit_reason = "tp2"
                exit_bar = i
                break
            # If only TP1 hit, keep going but track
            if tp1_hit:
                exit_price = close  # mark current

        else:  # Short
            if high >= sl:
                sl_hit = True
                exit_price = sl
                exit_reason = "stop_loss"
                exit_bar = i
                break
            if not tp1_hit and low <= tp1:
                tp1_hit = True
            if not tp2_hit and low <= tp2:
                tp2_hit = True
            if not tp3_hit and low <= tp3:
                tp3_hit = True
                exit_price = tp3
                exit_reason = "tp3"
                exit_bar = i
                break
            if tp2_hit and not tp3_hit:
                exit_price = tp2
                exit_reason = "tp2"
                exit_bar = i
                break
            if tp1_hit:
                exit_price = close

    # Calculate PnL
    if side == 'Long':
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    # Determine win
    is_win = pnl_pct > 0

    return {
        "symbol": signal.get('symbol', ''),
        "side": side,
        "pattern": signal['pattern'],
        "entry": entry,
        "sl": sl,
        "exit_price": float(exit_price),
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "tp3_hit": tp3_hit,
        "sl_hit": sl_hit,
        "pnl_pct": round(pnl_pct, 4),
        "is_win": is_win,
        "exit_reason": exit_reason,
        "bars_held": exit_bar - entry_bar,
        "total_score": signal.get('total_score', 0),
        "tech_score": signal.get('tech_score', 0),
        "bar_ts": signal.get('bar_ts', ''),
        "rr": signal['rr'],
    }


# ──────────────────────────────────────────────
# Backtest Runner
# ──────────────────────────────────────────────

class Backtester:
    """Orchestrates the full backtesting pipeline."""

    def __init__(self, exchange=None, config=None):
        self.config = config or CONFIG
        self.exchange = exchange or ccxt.bybit({
            'apiKey': self.config['api'].get('bybit_key', ''),
            'secret': self.config['api'].get('bybit_secret', ''),
            'options': {'defaultType': 'swap'}
        })
        self.trades = []
        self.signals = []

    def run(self, symbols=None, timeframe='4h', start_date=None, end_date=None,
            max_pairs=20, cooldown_bars=10):
        """
        Run the full backtest.

        Args:
            symbols: List of symbol strings (or None to auto-select top pairs)
            timeframe: Candle timeframe
            start_date: Start datetime
            end_date: End datetime
            max_pairs: Max pairs to scan if auto-selecting
            cooldown_bars: Min bars between signals on same symbol
        """
        bt_config = self.config.get('backtest', {})
        if start_date is None:
            days = bt_config.get('lookback_days', 90)
            start_date = datetime.utcnow() - timedelta(days=days)
        if end_date is None:
            end_date = datetime.utcnow()

        # Load ML model for scoring (if available)
        load_ml_model()

        # Get symbols
        if not symbols:
            symbols = get_top_pairs(self.exchange, limit=max_pairs)
            if not symbols:
                logger.error("No symbols found for backtesting")
                return self

        print(f"\n{'='*60}")
        print(f"📊 BACKTEST START")
        print(f"{'='*60}")
        print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        print(f"⏱️  Timeframe: {timeframe}")
        print(f"🔍 Symbols: {len(symbols)}")
        print(f"{'='*60}\n")

        total_signals = 0
        total_trades = 0

        for idx_sym, symbol in enumerate(symbols):
            print(f"[{idx_sym+1}/{len(symbols)}] 📥 Fetching {symbol}...", end=" ", flush=True)

            try:
                df = fetch_historical_ohlcv(self.exchange, symbol, timeframe, start_date, end_date)
            except Exception as e:
                print(f"❌ Fetch error: {e}")
                continue

            if len(df) < 200:
                print(f"⚠️ Too few bars ({len(df)})")
                continue

            print(f"{len(df)} bars", end=" ", flush=True)

            # BTC Bias: compute from BTC data if this IS BTC, else neutral
            btc_bias = "Sideways"
            if symbol == 'BTC/USDT':
                try:
                    ema13 = ta.ema(df['close'], length=13)
                    ema21 = ta.ema(df['close'], length=21)
                    btc_bias = "Bullish" if ema13.iloc[-1] > ema21.iloc[-1] else "Bearish"
                except Exception:
                    pass

            # Scan each bar
            min_candles = self.config.get('system', {}).get('min_candles_analysis', 150)
            last_signal_bar = -cooldown_bars  # enforce cooldown
            sym_signals = 0

            for i in range(min_candles, len(df) - 20):  # leave 20 bars for simulation
                if i - last_signal_bar < cooldown_bars:
                    continue

                signal = analyze_bar_offline(df, i, btc_bias, self.config)
                if signal is None:
                    continue

                signal['symbol'] = symbol
                signal['timeframe'] = timeframe
                self.signals.append(signal)
                sym_signals += 1
                last_signal_bar = i

                # Simulate trade
                future_df = df.iloc[i + 1:]
                result = simulate_trade(signal, future_df, max_bars=100)

                if result is not None:
                    result['symbol'] = symbol
                    result['timeframe'] = timeframe
                    self.trades.append(result)
                    total_trades += 1

            total_signals += sym_signals
            print(f"→ {sym_signals} signals ✅")

        print(f"\n{'='*60}")
        print(f"📊 BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"Total signals: {total_signals}")
        print(f"Total trades simulated: {total_trades}")

        return self

    def report(self):
        """Generate and print performance report."""
        if not self.trades:
            print("\n⚠️ No trades to report.")
            return self

        df = pd.DataFrame(self.trades)

        # ── Overall Metrics ──
        total = len(df)
        wins = df['is_win'].sum()
        losses = total - wins
        win_rate = wins / total * 100 if total > 0 else 0
        avg_pnl = df['pnl_pct'].mean()
        total_pnl = df['pnl_pct'].sum()

        # Profit Factor
        gross_profit = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(df[df['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average win/loss
        avg_win = df[df['is_win']]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = df[~df['is_win']]['pnl_pct'].mean() if losses > 0 else 0

        # Risk-Reward realized
        avg_rr = df['rr'].mean()

        # Max drawdown (cumulative)
        cum_pnl = df['pnl_pct'].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_drawdown = drawdown.min()

        # Sharpe-like ratio (simplified)
        sharpe = df['pnl_pct'].mean() / df['pnl_pct'].std() if df['pnl_pct'].std() > 0 else 0

        # ── Exit Reason Breakdown ──
        exit_counts = df['exit_reason'].value_counts().to_dict()

        # ── Per-Pattern Performance ──
        pattern_stats = df.groupby('pattern').agg(
            count=('pnl_pct', 'count'),
            win_rate=('is_win', 'mean'),
            avg_pnl=('pnl_pct', 'mean'),
            total_pnl=('pnl_pct', 'sum'),
        ).round(4)
        pattern_stats['win_rate'] = (pattern_stats['win_rate'] * 100).round(1)

        # ── Per-Side Performance ──
        side_stats = df.groupby('side').agg(
            count=('pnl_pct', 'count'),
            win_rate=('is_win', 'mean'),
            avg_pnl=('pnl_pct', 'mean'),
            total_pnl=('pnl_pct', 'sum'),
        ).round(4)
        side_stats['win_rate'] = (side_stats['win_rate'] * 100).round(1)

        # ── Score Analysis ──
        score_bins = [(0, 8), (8, 11), (11, 14), (14, 100)]
        score_analysis = []
        for lo, hi in score_bins:
            mask = (df['total_score'] >= lo) & (df['total_score'] < hi)
            subset = df[mask]
            if len(subset) > 0:
                score_analysis.append({
                    'range': f"{lo}-{hi}",
                    'count': len(subset),
                    'win_rate': round(subset['is_win'].mean() * 100, 1),
                    'avg_pnl': round(subset['pnl_pct'].mean(), 4),
                })

        # ── Print Report ──
        print(f"\n{'='*60}")
        print(f"📈 BACKTEST PERFORMANCE REPORT")
        print(f"{'='*60}")

        print(f"\n── Overall ──")
        print(f"  Total Trades:      {total}")
        print(f"  Wins / Losses:     {wins} / {losses}")
        print(f"  Win Rate:          {win_rate:.1f}%")
        print(f"  Profit Factor:     {profit_factor:.2f}")
        print(f"  Avg Win:           +{avg_win:.4f}%")
        print(f"  Avg Loss:          {avg_loss:.4f}%")
        print(f"  Avg Trade PnL:     {avg_pnl:+.4f}%")
        print(f"  Total PnL:         {total_pnl:+.2f}%")
        print(f"  Max Drawdown:      {max_drawdown:+.2f}%")
        print(f"  Sharpe Ratio:      {sharpe:.2f}")
        print(f"  Avg RR (target):   {avg_rr:.2f}")
        print(f"  Avg Bars Held:     {df['bars_held'].mean():.0f}")

        print(f"\n── Exit Reasons ──")
        for reason, count in exit_counts.items():
            pct = count / total * 100
            print(f"  {reason:15s}: {count:4d} ({pct:.1f}%)")

        print(f"\n── Per Pattern ──")
        print(f"  {'Pattern':<22s} {'Count':>6s} {'Win%':>6s} {'AvgPnL':>8s} {'TotPnL':>8s}")
        print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
        for pat, row in pattern_stats.iterrows():
            print(f"  {pat:<22s} {int(row['count']):>6d} {row['win_rate']:>5.1f}% {row['avg_pnl']:>+7.4f}% {row['total_pnl']:>+7.2f}%")

        print(f"\n── Per Side ──")
        print(f"  {'Side':<8s} {'Count':>6s} {'Win%':>6s} {'AvgPnL':>8s} {'TotPnL':>8s}")
        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
        for side, row in side_stats.iterrows():
            print(f"  {side:<8s} {int(row['count']):>6d} {row['win_rate']:>5.1f}% {row['avg_pnl']:>+7.4f}% {row['total_pnl']:>+7.2f}%")

        print(f"\n── Score vs Performance ──")
        print(f"  {'Score':>8s} {'Count':>6s} {'Win%':>6s} {'AvgPnL':>8s}")
        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*8}")
        for row in score_analysis:
            print(f"  {row['range']:>8s} {row['count']:>6d} {row['win_rate']:>5.1f}% {row['avg_pnl']:>+7.4f}%")

        print(f"\n{'='*60}\n")

        return self

    def export_results(self, filepath='backtest_results.json'):
        """Export trades and signals to JSON file."""
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "config_timeframe": self.config.get('backtest', {}).get('timeframe', '4h'),
            "total_trades": len(self.trades),
            "total_signals": len(self.signals),
            "trades": self.trades,
            "signals": [{k: v for k, v in s.items() if k != 'df'} for s in self.signals],
        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"📁 Results exported to {filepath}")
        return self


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Bot Auto Screening - Backtester')
    parser.add_argument('--pairs', nargs='+', default=None, help='Specific pairs (e.g. BTC/USDT ETH/USDT)')
    parser.add_argument('--tf', default='4h', help='Timeframe (default: 4h)')
    parser.add_argument('--days', type=int, default=90, help='Lookback days (default: 90)')
    parser.add_argument('--max-pairs', type=int, default=20, help='Max pairs to scan (default: 20)')
    parser.add_argument('--export', default=None, help='Export results to JSON file')
    parser.add_argument('--symbols-file', default=None, help='File with one symbol per line')
    args = parser.parse_args()

    # Resolve symbols
    symbols = args.pairs
    if args.symbols_file:
        with open(args.symbols_file) as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    bt_config = CONFIG.get('backtest', {})
    start_date = datetime.utcnow() - timedelta(days=args.days)
    end_date = datetime.utcnow()

    exchange = ccxt.bybit({
        'apiKey': CONFIG['api'].get('bybit_key', ''),
        'secret': CONFIG['api'].get('bybit_secret', ''),
        'options': {'defaultType': 'swap'}
    })

    backtester = Backtester(exchange=exchange)
    backtester.run(
        symbols=symbols,
        timeframe=args.tf,
        start_date=start_date,
        end_date=end_date,
        max_pairs=args.max_pairs,
        cooldown_bars=bt_config.get('cooldown_bars', 10),
    )
    backtester.report()

    if args.export:
        backtester.export_results(args.export)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    main()
