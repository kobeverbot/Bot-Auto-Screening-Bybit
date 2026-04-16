#!/usr/bin/env python3
"""
Quick Backtest Runner.

Usage:
  python run_backtest.py                           # Top 20 pairs, 4h, 90 days
  python run_backtest.py --pairs BTC/USDT ETH/USDT # Specific pairs
  python run_backtest.py --tf 1h --days 180        # 1h timeframe, 6 months
  python run_backtest.py --max-pairs 5 --export results.json
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from modules.backtest import main

if __name__ == '__main__':
    main()
