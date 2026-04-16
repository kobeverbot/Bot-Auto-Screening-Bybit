#!/usr/bin/env python3
"""
ML Model Training Script

Trains the ML scorer from backtest data. Two modes:

1. From backtest export:
   python train_model.py --from-export backtest_results.json

2. Run backtest + train:
   python train_model.py --pairs BTC/USDT ETH/USDT --days 180 --model gradient_boosting

3. From live database trades:
   python train_model.py --from-db
"""
import argparse
import json
import logging
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("TrainModel")


def train_from_export(export_path: str, model_type: str = "gradient_boosting",
                      output_path: str = None) -> bool:
    """Train from a backtest export JSON file."""
    logger.info(f"📂 Loading backtest export: {export_path}")

    if not os.path.exists(export_path):
        logger.error(f"Export file not found: {export_path}")
        return False

    with open(export_path, 'r') as f:
        data = json.load(f)

    signals = data.get("signals", [])
    trades = data.get("trades", [])

    if not signals or not trades:
        logger.error("Export file has no signals or trades. Run backtest with --export first.")
        return False

    logger.info(f"📊 Loaded {len(signals)} signals, {len(trades)} trades")

    return _train_and_save(signals, trades, model_type, output_path)


def train_from_backtest(pairs=None, days=180, timeframe='4h', max_pairs=30,
                        model_type="gradient_boosting", output_path=None) -> bool:
    """Run a backtest and train on the results."""
    import ccxt
    from modules.config_loader import CONFIG
    from modules.backtest import Backtester
    from datetime import datetime, timedelta

    logger.info(f"🔄 Running backtest: {days} days, {timeframe}, {max_pairs} pairs")

    exchange = ccxt.bybit({
        'apiKey': CONFIG['api']['bybit_key'],
        'secret': CONFIG['api']['bybit_secret'],
        'options': {'defaultType': 'swap'}
    })

    bt = Backtester(exchange=exchange)
    bt.run(
        symbols=pairs,
        timeframe=timeframe,
        start_date=datetime.utcnow() - timedelta(days=days),
        end_date=datetime.utcnow(),
        max_pairs=max_pairs,
    )
    bt.report()

    # Get signals and trades from backtester internals
    signals = bt.signals if hasattr(bt, 'signals') else []
    trades = bt.trades if hasattr(bt, 'trades') else []

    if not signals or not trades:
        logger.error("Backtest produced no signals or trades. Try longer period or more pairs.")
        return False

    logger.info(f"📊 Backtest produced {len(signals)} signals, {len(trades)} trades")

    # Export for future use
    export_path = output_path or "ml_models/latest_backtest.json"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    bt.export_results(export_path)
    logger.info(f"💾 Backtest results exported to {export_path}")

    return _train_and_save(signals, trades, model_type)


def train_from_db(model_type="gradient_boosting", output_path=None) -> bool:
    """Train from live database trades with outcomes."""
    from modules.database import get_conn, release_conn

    logger.info("📂 Loading trades from database...")

    conn = get_conn()
    try:
        cur = conn.cursor()
        # Get closed trades with all score data
        cur.execute("""
            SELECT t.symbol, t.side, t.pattern, t.tech_score, t.quant_score,
                   t.deriv_score, t.smc_score, t.z_score, t.zeta_score, t.obi,
                   t.basis, t.btc_bias, t.rr, t.exit_price, t.sl_price, t.tp3,
                   a.pnl, a.status
            FROM trades t
            JOIN active_trades a ON a.signal_id = t.id
            WHERE a.status = 'CLOSED' AND a.pnl IS NOT NULL
        """)
        rows = cur.fetchall()
    finally:
        release_conn(conn)

    if not rows:
        logger.error("No closed trades with PnL found in database.")
        return False

    # Convert to signals + trades format
    signals = []
    trades = []
    for i, row in enumerate(rows):
        (symbol, side, pattern, tech, quant, deriv, smc,
         z_score, zeta, obi, basis, btc_bias, rr,
         exit_price, sl_price, tp3, pnl, status) = row

        signals.append({
            "symbol": symbol, "side": side, "pattern": pattern,
            "tech_score": tech, "quant_score": quant,
            "deriv_score": deriv, "smc_score": smc,
            "z_score": z_score, "zeta_score": zeta,
            "obi": obi, "basis": basis, "btc_bias": btc_bias,
            "rr": rr, "bar_ts": str(i),
        })

        # Determine win from PnL
        is_win = pnl is not None and float(pnl) > 0
        trades.append({
            "symbol": symbol, "bar_ts": str(i),
            "is_win": is_win,
            "pnl_pct": float(pnl) if pnl else 0,
            "exit_reason": "closed",
        })

    logger.info(f"📊 Loaded {len(signals)} trades from database")
    return _train_and_save(signals, trades, model_type, output_path)


def _train_and_save(signals, trades, model_type="gradient_boosting",
                    output_path=None) -> bool:
    """Common training + save logic."""
    from modules.ml_scorer import (
        prepare_training_data, train_model, save_model, _get_cfg
    )

    # Prepare data
    X, y, feature_names = prepare_training_data(signals, trades)

    if len(X) == 0:
        logger.error("No matching signal-trade pairs found for training.")
        return False

    # Train
    try:
        model, metadata = train_model(X, y, model_type=model_type)
    except ValueError as e:
        logger.error(str(e))
        return False

    # Calculate avg PnL from trades
    pnl_values = [t.get("pnl_pct", 0) for t in trades if "pnl_pct" in t]
    if pnl_values:
        metadata.avg_pnl_pct = sum(pnl_values) / len(pnl_values)

    # Save
    cfg = _get_cfg()
    save_path = output_path or cfg["model_path"]
    success = save_model(model, metadata)

    if success:
        logger.info(f"✅ Model saved to {save_path}")
        logger.info(f"   Type: {metadata.model_type}")
        logger.info(f"   Samples: {metadata.n_samples}")
        logger.info(f"   Win rate: {metadata.win_rate:.1%}")
        logger.info(f"   Top features:")
        for feat, imp in list(metadata.feature_importance.items())[:10]:
            logger.info(f"     {feat}: {imp:.4f}")
    else:
        logger.error("Failed to save model.")

    return success


def main():
    parser = argparse.ArgumentParser(description='Train ML Scoring Model')
    parser.add_argument('--from-export', metavar='FILE',
                        help='Train from backtest export JSON file')
    parser.add_argument('--from-db', action='store_true',
                        help='Train from live database trades')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='Pairs for backtest-based training')
    parser.add_argument('--days', type=int, default=180,
                        help='Lookback days for backtest (default: 180)')
    parser.add_argument('--tf', default='4h',
                        help='Timeframe for backtest (default: 4h)')
    parser.add_argument('--max-pairs', type=int, default=30,
                        help='Max pairs for backtest (default: 30)')
    parser.add_argument('--model', default='gradient_boosting',
                        choices=['gradient_boosting', 'random_forest', 'logistic'],
                        help='Model type (default: gradient_boosting)')
    parser.add_argument('--output', default=None,
                        help='Output path for model file')

    args = parser.parse_args()

    if args.from_export:
        success = train_from_export(args.from_export, args.model, args.output)
    elif args.from_db:
        success = train_from_db(args.model, args.output)
    else:
        success = train_from_backtest(
            pairs=args.pairs, days=args.days, timeframe=args.tf,
            max_pairs=args.max_pairs, model_type=args.model,
            output_path=args.output,
        )

    if success:
        print("\n🎉 Training complete! Enable ml_scoring in config.json to use the model.")
    else:
        print("\n❌ Training failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
