import ccxt
import time
import schedule
import threading
import logging
from datetime import datetime
from pybit.unified_trading import WebSocket
from modules.config_loader import CONFIG
from modules.database import get_conn, release_conn
from modules.circuit_breaker import circuit_breaker

# --- ⚙️ CONFIGURATION ---
_trading_cfg = CONFIG.get('trading', {})
TARGET_LEVERAGE   = int(_trading_cfg.get('target_leverage', 25))
RISK_PERCENT      = float(_trading_cfg.get('risk_percent', 0.01))     # Risk 1% of Equity per trade
MAX_POSITIONS     = int(_trading_cfg.get('max_positions', 40))         # Max Concurrent OPEN positions
MAX_POSITION_PCT  = float(_trading_cfg.get('max_position_pct', 0.05)) # Max 5% of equity per position
TRAILING_STOP_PCT = float(_trading_cfg.get('trailing_stop_pct', 0.015))  # 1.5% trailing distance
TRAILING_ACT_TP   = int(_trading_cfg.get('trailing_activation_tp', 2))   # TP level that activates trailing (1=TP1, 2=TP2)
TP_SPLIT = [0.30, 0.30, 0.40] # 30% TP1, 30% TP2, 40% TP3

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AutoTrader")

# REST API Connection (CCXT)
exchange = ccxt.bybit({
    'apiKey': CONFIG['api']['bybit_key'],
    'secret': CONFIG['api']['bybit_secret'],
    'options': {'defaultType': 'swap', 'adjustForTimeDifference': True}
})

# ---------------------------------------------------------
# 🛠️ DATABASE INITIALIZATION (Self-Healing)
# ---------------------------------------------------------
def init_execution_db():
    """Creates execution tables if they don't exist."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        
        # 1. Active Trades Table (Isolated Execution)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS active_trades (
                id SERIAL PRIMARY KEY,
                signal_id INT,
                symbol VARCHAR(20),
                side VARCHAR(10),
                entry_price DECIMAL,
                sl_price DECIMAL,
                tp1 DECIMAL,
                tp2 DECIMAL,
                tp3 DECIMAL,
                quantity DECIMAL,
                leverage INT,
                order_id VARCHAR(50),
                status VARCHAR(20) DEFAULT 'PENDING',
                pnl DECIMAL DEFAULT 0,
                is_sl_moved BOOLEAN DEFAULT FALSE,
                trailing_activated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Daily Reports Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                report_date DATE PRIMARY KEY,
                total_pnl DECIMAL DEFAULT 0,
                win_rate DECIMAL DEFAULT 0,
                total_wins INT DEFAULT 0,
                total_losses INT DEFAULT 0,
                total_trades INT DEFAULT 0,
                best_trade_symbol VARCHAR(20),
                best_trade_pnl DECIMAL,
                worst_trade_symbol VARCHAR(20),
                worst_trade_pnl DECIMAL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logger.info("✅ Execution Database Tables Sync Complete.")
    except Exception as e:
        logger.error(f"❌ DB Init Error: {e}")
    finally:
        release_conn(conn)

# ---------------------------------------------------------
# ⚡ WEBSOCKET EVENT HANDLERS (Real-Time Brain)
# ---------------------------------------------------------
def place_split_tps(symbol, side, total_qty, tp1, tp2, tp3):
    """
    Called instantly via WebSocket when Entry is Filled.
    Places 3 Reduce-Only Limit orders.
    """
    try:
        # 1. NORMALIZE SIDE (Fixes the 110017 Error)
        # Entry could be 'Buy', 'Long', 'Sell', or 'Short'
        side_str = str(side).lower()
        
        # If Entry was Buy/Long -> We must SELL to close
        if side_str in ['buy', 'long']:
            tp_side = 'sell'
        # If Entry was Sell/Short -> We must BUY to close
        else:
            tp_side = 'buy'
        
        # 2. Calculate Splits
        q1 = float(exchange.amount_to_precision(symbol, total_qty * TP_SPLIT[0]))
        q2 = float(exchange.amount_to_precision(symbol, total_qty * TP_SPLIT[1]))
        q3 = float(exchange.amount_to_precision(symbol, total_qty * TP_SPLIT[2]))
        
        # Adjust remainder to ensure 100% close
        current_sum = q1 + q2 + q3
        if current_sum != total_qty:
            diff = total_qty - current_sum
            q3 += diff
            q3 = float(exchange.amount_to_precision(symbol, q3))

        params = {'reduceOnly': True}
        
        logger.info(f"⚡ Placing TPs for {symbol} ({tp_side.upper()}): {q1} | {q2} | {q3}")
        
        # Place Orders
        exchange.create_order(symbol, 'limit', tp_side, q1, float(tp1), params)
        exchange.create_order(symbol, 'limit', tp_side, q2, float(tp2), params)
        exchange.create_order(symbol, 'limit', tp_side, q3, float(tp3), params)
        
        return True
    except Exception as e:
        logger.error(f"⚠️ TP Placement Failed {symbol}: {e}")
        return False

def on_execution_update(message):
    """
    WebSocket Callback: Listens for Order Fills ('Trade').
    Triggers TP placement logic immediately.
    """
    try:
        data = message.get('data', [])
        for exec_item in data:
            symbol = exec_item['symbol']
            side = exec_item['side']
            exec_type = exec_item.get('execType') 
            
            # Filter: Only care about 'Trade' (Fills)
            if exec_type == 'Trade':
                conn = get_conn()
                try:
                    cur = conn.cursor()
                    
                    # Check if we have an OPEN trade waiting for TPs
                    cur.execute("""
                        SELECT id, tp1, tp2, tp3 
                        FROM active_trades 
                        WHERE symbol = %s AND status = 'OPEN'
                    """, (symbol,))
                    row = cur.fetchone()
                    
                    if row:
                        t_id, tp1, tp2, tp3 = row
                        logger.info(f"⚡ WS: Entry Filled for {symbol}! Placing TPs...")
                        
                        # Double check position size from REST to be accurate
                        pos = exchange.fetch_position(symbol)
                        current_size = float(pos['contracts'])
                        
                        if current_size > 0:
                            success = place_split_tps(symbol, side, current_size, tp1, tp2, tp3)
                            if success:
                                cur.execute("UPDATE active_trades SET status = 'OPEN_TPS_SET', updated_at = NOW() WHERE id = %s", (t_id,))
                                conn.commit()
                except Exception as e:
                    logger.error(f"WS Exec Logic Error: {e}")
                finally:
                    release_conn(conn)
    except Exception as e:
        logger.error(f"WS Payload Error: {e}")

def on_position_update(message):
    """
    WebSocket Callback: Listens for PnL/Price updates.
    Handles BEP moves and Close detection.
    """
    try:
        data = message.get('data', [])
        for pos in data:
            symbol = pos['symbol']
            size = float(pos['size'])
            mark_price = float(pos['markPrice'])
            side = pos['side'] # 'Buy' or 'Sell'
            
            conn = get_conn()
            try:
                cur = conn.cursor()
                
                # Fetch trade info
                cur.execute("""
                    SELECT id, entry_price, tp1, tp2, is_sl_moved, trailing_activated, status 
                    FROM active_trades 
                    WHERE symbol = %s AND status = 'OPEN_TPS_SET'
                """, (symbol,))
                row = cur.fetchone()
                
                if row:
                    t_id, entry, tp1, tp2, sl_moved, trailing_activated, status = row
                    
                    # 1. POSITION CLOSED CHECK (Size -> 0)
                    if size == 0:
                        logger.info(f"🏁 WS: {symbol} Position Closed. Fetching PnL...")
                        time.sleep(1) # Allow Bybit backend to settle PnL
                        try:
                            trades = exchange.fetch_my_trades(symbol, limit=1)
                            real_pnl = float(trades[0]['info'].get('closedPnl', 0)) if trades else 0
                            cur.execute("UPDATE active_trades SET status = 'CLOSED', pnl = %s, updated_at = NOW() WHERE id = %s", (real_pnl, t_id))
                        except Exception as trade_err:
                            logger.warning(f"WS: Could not fetch PnL for {symbol}: {trade_err}")
                            cur.execute("UPDATE active_trades SET status = 'CLOSED', updated_at = NOW() WHERE id = %s", (t_id,))
                        conn.commit()
                        return

                    # 2. BREAKEVEN LOGIC (Hit TP1 -> Move SL)
                    # For Long: Mark >= TP1. For Short: Mark <= TP1
                    hit_tp1 = (side == 'Buy' and mark_price >= float(tp1)) or \
                              (side == 'Sell' and mark_price <= float(tp1))
                    
                    if hit_tp1 and not sl_moved:
                        logger.info(f"♻️ WS: {symbol} hit TP1. Moving SL to Entry...")
                        try:
                            exchange.set_position_stop_loss(symbol, float(entry), side.lower())
                            cur.execute("UPDATE active_trades SET is_sl_moved = TRUE WHERE id = %s", (t_id,))
                            conn.commit()
                        except Exception as sl_err:
                            logger.error(f"⚠️ Failed to move SL for {symbol}: {sl_err}")

                    # 3. TRAILING STOP LOGIC
                    # Determine the activation TP level based on config
                    activation_price = float(tp1) if TRAILING_ACT_TP == 1 else float(tp2)
                    hit_activation_tp = (side == 'Buy' and mark_price >= activation_price) or \
                                        (side == 'Sell' and mark_price <= activation_price)

                    # Activate trailing once SL has been moved to breakeven and activation TP is hit
                    if sl_moved and not trailing_activated and hit_activation_tp:
                        logger.info(f"📐 WS: {symbol} hit TP{TRAILING_ACT_TP}. Activating trailing stop ({TRAILING_STOP_PCT*100:.1f}%).")
                        cur.execute("UPDATE active_trades SET trailing_activated = TRUE WHERE id = %s", (t_id,))
                        trailing_activated = True
                        conn.commit()

                    # Ratchet the trailing stop in the favorable direction
                    if trailing_activated:
                        # Fetch current SL from the position data
                        current_sl = float(pos.get('stopLoss', 0))
                        if current_sl == 0:
                            # Fallback: use entry price if SL not in WS data
                            current_sl = float(entry)

                        if side == 'Buy':
                            new_sl = mark_price * (1 - TRAILING_STOP_PCT)
                            if new_sl > current_sl:
                                logger.info(f"📈 WS: {symbol} Trailing SL moved up: {current_sl:.4f} → {new_sl:.4f} (mark={mark_price:.4f})")
                                try:
                                    exchange.set_position_stop_loss(symbol, new_sl, side.lower())
                                    cur.execute("UPDATE active_trades SET sl_price = %s WHERE id = %s", (new_sl, t_id))
                                    conn.commit()
                                except Exception as trail_err:
                                    logger.error(f"⚠️ Trailing SL update failed for {symbol}: {trail_err}")
                        elif side == 'Sell':
                            new_sl = mark_price * (1 + TRAILING_STOP_PCT)
                            if new_sl < current_sl:
                                logger.info(f"📉 WS: {symbol} Trailing SL moved down: {current_sl:.4f} → {new_sl:.4f} (mark={mark_price:.4f})")
                                try:
                                    exchange.set_position_stop_loss(symbol, new_sl, side.lower())
                                    cur.execute("UPDATE active_trades SET sl_price = %s WHERE id = %s", (new_sl, t_id))
                                    conn.commit()
                                except Exception as trail_err:
                                    logger.error(f"⚠️ Trailing SL update failed for {symbol}: {trail_err}")

            except Exception as e:
                # logger.error(f"WS Pos Error: {e}") # Silent fail on DB locks
                logger.debug(f"WS Pos Error (likely DB lock): {e}")
            finally:
                release_conn(conn)
    except Exception as e:
        logger.debug(f"WS Position payload error: {e}")

# ---------------------------------------------------------
# 📥 SIGNAL INGESTION (Loop)
# ---------------------------------------------------------
def ingest_fresh_signals():
    """Reads 'Waiting Entry' from scanner table, applies Risk/Lev, inserts to active_trades."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        
        # 1. Check Max Positions (OPEN only)
        cur.execute("SELECT COUNT(*) FROM active_trades WHERE status IN ('OPEN', 'OPEN_TPS_SET')")
        current_active = cur.fetchone()[0]
        
        if current_active >= MAX_POSITIONS:
            # logger.info(f"🚫 Max positions ({current_active}) reached.")
            return

        # 2. Fetch Data needed for calc
        try:
            balance = exchange.fetch_balance()
            total_equity = float(balance['total']['USDT'])
            markets = exchange.load_markets()
        except Exception as e:
            logger.error(f"API Fetch Error: {e}")
            return

        # 2b. Circuit Breaker — halt if safety limits breached
        allowed, reason = circuit_breaker.can_trade(total_equity)
        if not allowed:
            logger.warning(f"🛑 Circuit Breaker TRIPPED — skipping signals: {reason}")
            return

        # 3. Get New Signals
        query = """
            SELECT t.id, t.symbol, t.side, t.entry_price, t.sl_price, t.tp1, t.tp2, t.tp3
            FROM trades t
            LEFT JOIN active_trades a ON t.id = a.signal_id
            WHERE t.status = 'Waiting Entry'
            AND t.created_at >= NOW() - INTERVAL '12 hours'
            AND a.id IS NULL
        """
        cur.execute(query)
        signals = cur.fetchall()
        
        for sig in signals:
            if current_active >= MAX_POSITIONS: break
            
            sig_id, sym, side, entry, sl, tp1, tp2, tp3 = sig
            entry, sl = float(entry), float(sl)
            
            # A. Dynamic Leverage (Target 25x or Max)
            market = markets.get(sym)
            max_lev = 25
            if market and 'limits' in market:
                limit_lev = market['limits']['leverage']['max']
                if limit_lev: max_lev = float(limit_lev)
            
            final_leverage = min(TARGET_LEVERAGE, int(max_lev))
            
            # B. Risk-Based Position Sizing
            # -----------------------------------------------
            # risk_amount  = equity × risk%   (dollar amount we're willing to lose)
            # risk_distance = |entry − SL|    (stop-loss distance in price)
            # qty_coins    = risk_amount / risk_distance
            #   → if SL is hit, we lose exactly risk_amount (1% of equity)
            #
            # Leverage is set on the exchange but does NOT inflate position size;
            # it only determines the margin required to hold the position.
            # A max-position cap (5% of equity) prevents oversized positions
            # when the SL distance is very tight.
            # -----------------------------------------------
            risk_amount   = total_equity * RISK_PERCENT
            risk_distance = abs(entry - sl)

            if risk_distance == 0:
                logger.warning(f"⚠️ Signal {sym} skipped: SL equals entry (zero risk distance).")
                continue

            qty_coins = risk_amount / risk_distance

            # Cap position size to MAX_POSITION_PCT of equity (prevents huge sizes on tight SL)
            max_notional = total_equity * MAX_POSITION_PCT
            notional = qty_coins * entry
            if notional > max_notional:
                qty_coins = max_notional / entry
                notional = max_notional

            # Check Min Notional (Approx $6 for Bybit)
            if notional < 6.0:
                logger.warning(f"⚠️ Signal {sym} skipped: Notional ${notional:.2f} is below Bybit min ($6).")
                continue

            # C. Insert PENDING Trade
            cur.execute("""
                INSERT INTO active_trades (signal_id, symbol, side, entry_price, sl_price, tp1, tp2, tp3, quantity, leverage, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING')
            """, (sig_id, sym, side, entry, sl, tp1, tp2, tp3, qty_coins, final_leverage))
            
            logger.info(f"📥 Signal Ingested: {sym} | Lev: {final_leverage}x | Risk: ${risk_amount:.2f} | Qty: {qty_coins:.6f} | Notional: ${notional:.2f}")
            current_active += 1
            
        conn.commit()
    except Exception as e:
        logger.error(f"Ingest Error: {e}")
    finally:
        release_conn(conn)

# ---------------------------------------------------------
# 🚀 SMART EXECUTION (Limit or Market)
# ---------------------------------------------------------
def execute_pending_orders():
    """
    Checks PENDING orders.
    - If price is BETTER than entry -> MARKET Buy.
    - If price is WORSE than entry -> LIMIT Buy.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, symbol, side, entry_price, sl_price, quantity, leverage FROM active_trades WHERE status = 'PENDING'")
        orders = cur.fetchall()
        
        if not orders: return 

        for order in orders:
            oid, sym, side, entry, sl, qty, lev = order
            
            try:
                # 1. Set Leverage
                try: exchange.set_leverage(int(lev), sym)
                except Exception as lev_err:
                    logger.debug(f"set_leverage failed for {sym} (may already be set): {lev_err}")

                # 2. Check LIVE Price
                ticker = exchange.fetch_ticker(sym)
                current_price = float(ticker['last'])
                entry = float(entry)
                
                # Logic: Is the price better than our entry?
                is_better_price = (side == 'Long' and current_price <= entry) or \
                                  (side == 'Short' and current_price >= entry)

                type_side = 'buy' if side == 'Long' else 'sell'
                params = {'stopLoss': float(sl)}
                
                # Precision Handling
                qty = float(exchange.amount_to_precision(sym, qty))
                
                res = None
                
                # 3. Decision
                if is_better_price:
                    logger.info(f"⚡ {sym}: Price Better ({current_price} vs {entry}). Executing MARKET...")
                    res = exchange.create_order(sym, 'market', type_side, qty, None, params)
                else:
                    logger.info(f"⏳ {sym}: Waiting ({current_price} vs {entry}). Placing LIMIT...")
                    res = exchange.create_order(sym, 'limit', type_side, entry, qty, params)
                
                # 4. Update DB
                if res and 'id' in res:
                    new_status = 'OPEN' 
                    # If Market order, it fills instantly, so next loop will catch TPs via Safety Net or WS
                    cur.execute("UPDATE active_trades SET order_id = %s, status = %s WHERE id = %s", (res['id'], new_status, oid))
                    conn.commit()
                    logger.info(f"✅ Order Placed for {sym} (ID: {res['id']})")

            except Exception as e:
                logger.error(f"❌ Execution Failed {sym}: {e}")
                cur.execute("UPDATE active_trades SET status = 'FAILED' WHERE id = %s", (oid,))
                
        conn.commit()
    except Exception as e:
        logger.error(f"Exec Loop Error: {e}")
    finally:
        release_conn(conn)

# ---------------------------------------------------------
# 🛡️ SAFETY NET (Robust Version)
# ---------------------------------------------------------
def check_missed_tps():
    """
    Backup loop: Checks for trades that are marked 'OPEN' in DB
    but are already FILLED on the exchange.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, symbol, side, order_id, tp1, tp2, tp3 
            FROM active_trades 
            WHERE status = 'OPEN' AND order_id IS NOT NULL
        """)
        stuck_trades = cur.fetchall()
        
        for trade in stuck_trades:
            t_id, sym, side, oid, tp1, tp2, tp3 = trade
            
            try:
                order_status = None
                
                # 1. Try Fetch Order (with suppression param)
                try:
                    # 'acknowledged': True silences the "last 500 orders" warning
                    order = exchange.fetch_order(oid, sym, params={'acknowledged': True})
                    order_status = order['status']
                except Exception as e:
                    # If fetch_order fails (e.g. order too old), search Closed Orders manually
                    logger.debug(f"Fetch Order failed for {sym}, searching history... {e}")

                # 2. Fallback: Search Recent History if direct fetch failed
                if not order_status:
                    try:
                        # Look in last 50 closed orders
                        closed_orders = exchange.fetch_closed_orders(sym, limit=50)
                        for o in closed_orders:
                            if str(o['id']) == str(oid):
                                order_status = o['status']
                                break
                    except Exception as hist_err:
                        logger.debug(f"History search failed for {sym}: {hist_err}")
                
                # 3. Process Status
                if order_status == 'closed':
                    # It is FILLED! We missed the WebSocket event.
                    logger.warning(f"⚠️ Safety Net: Found filled entry for {sym} (ID: {oid}). Placing TPs...")
                    
                    pos = exchange.fetch_position(sym)
                    size = float(pos['contracts'])
                    
                    if size > 0:
                        success = place_split_tps(sym, side, size, tp1, tp2, tp3)
                        if success:
                            cur.execute("UPDATE active_trades SET status = 'OPEN_TPS_SET', updated_at = NOW() WHERE id = %s", (t_id,))
                            conn.commit()
                            logger.info(f"✅ Safety Net: TPs recovered for {sym}")
                            
                elif order_status == 'canceled':
                    cur.execute("UPDATE active_trades SET status = 'CANCELLED' WHERE id = %s", (t_id,))
                    conn.commit()
                    logger.info(f"🗑️ Safety Net: Marked {sym} as CANCELLED.")
                    
            except Exception as e:
                logger.error(f"Safety Check Error {sym}: {e}")
                
    except Exception as e:
        logger.error(f"Global Safety Loop Error: {e}")
    finally:
        release_conn(conn)

# ---------------------------------------------------------
# 📊 DAILY REPORTING
# ---------------------------------------------------------
def generate_daily_report():
    logger.info("📊 Generating Daily Report...")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*),
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END),
                SUM(pnl),
                MAX(pnl),
                MIN(pnl)
            FROM active_trades 
            WHERE status = 'CLOSED' 
            AND updated_at >= NOW() - INTERVAL '24 hours'
        """)
        row = cur.fetchone()
        
        if row and row[0] > 0:
            total, wins, losses, pnl, best, worst = row
            pnl = pnl if pnl else 0
            win_rate = (wins/total)*100
            
            # Fetch symbols
            cur.execute("SELECT symbol FROM active_trades WHERE pnl = %s LIMIT 1", (best,))
            b_sym = cur.fetchone(); best_sym = b_sym[0] if b_sym else "-"
            
            cur.execute("SELECT symbol FROM active_trades WHERE pnl = %s LIMIT 1", (worst,))
            w_sym = cur.fetchone(); worst_sym = w_sym[0] if w_sym else "-"
            
            cur.execute("""
                INSERT INTO daily_reports (report_date, total_pnl, win_rate, total_wins, total_losses, total_trades, best_trade_symbol, best_trade_pnl, worst_trade_symbol, worst_trade_pnl)
                VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (report_date) DO NOTHING
            """, (pnl, win_rate, wins, losses, total, best_sym, best, worst_sym, worst))
            conn.commit()
            logger.info(f"✅ Report Generated: ${pnl:.2f} ({wins}W/{losses}L)")
            
    except Exception as e:
        logger.error(f"Report Error: {e}")
    finally:
        release_conn(conn)

# ---------------------------------------------------------
# 🏁 MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    logger.info("🟢 Starting Auto-Trader (Hybrid Architecture)...")
    
    # 1. Init DB
    init_execution_db()
    
    # 2. Start WebSocket (Background Thread)
    ws = WebSocket(
        testnet=False,
        channel_type="private",
        api_key=CONFIG['api']['bybit_key'],
        api_secret=CONFIG['api']['bybit_secret'],
    )
    ws.execution_stream(callback=on_execution_update)
    ws.position_stream(callback=on_position_update)
    logger.info("🔌 WebSocket Connected.")
    
    # 3. Schedule Jobs (Foreground)
    schedule.every(1).minutes.do(ingest_fresh_signals)      # Check for new trades
    schedule.every(5).seconds.do(execute_pending_orders)    # Fast Execution
    schedule.every(10).seconds.do(check_missed_tps)         # Safety Net
    schedule.every().day.at("00:00").do(generate_daily_report)
    schedule.every().day.at("00:01").do(circuit_breaker.reset)
    
    logger.info(f"🚀 Bot is LIVE. Monitoring {MAX_POSITIONS} Max Positions.")
    
    while True:
        schedule.run_pending()
        time.sleep(1)