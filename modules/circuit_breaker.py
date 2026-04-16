"""
Circuit Breaker Module — Critical Safety Feature
Halts trading when risk limits are breached to protect capital.
"""

import logging
from datetime import datetime, timedelta
from modules.database import get_conn, release_conn
from modules.config_loader import CONFIG

logger = logging.getLogger("CircuitBreaker")


class CircuitBreaker:
    """
    Monitors daily P&L, consecutive losses, trade count, and open exposure.
    Returns a pass/fail verdict via can_trade() before any new position is opened.
    """

    def __init__(self):
        cb_cfg = CONFIG.get("circuit_breaker", {})
        self.enabled = cb_cfg.get("enabled", True)
        self.max_daily_loss_pct = float(cb_cfg.get("max_daily_loss_pct", 0.05))
        self.max_consecutive_losses = int(cb_cfg.get("max_consecutive_losses", 5))
        self.pause_minutes_on_consecutive = int(cb_cfg.get("pause_minutes_on_consecutive", 60))
        self.max_daily_trades = int(cb_cfg.get("max_daily_trades", 50))
        self.max_open_exposure_pct = float(cb_cfg.get("max_open_exposure_pct", 2.0))

        # Internal state
        self._consecutive_pause_until = None  # datetime when pause expires
        self._tripped_reason = None           # last trip reason (informational)

    # ------------------------------------------------------------------
    # Individual checks — each returns (bool_ok, str_reason)
    # ------------------------------------------------------------------

    def check_daily_pnl(self, equity: float):
        """
        Compare today's realized losses against max_daily_loss_pct of equity.
        Queries the daily_reports table for today's row.
        """
        if equity <= 0:
            return True, "equity_unavailable"

        max_loss = equity * self.max_daily_loss_pct

        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COALESCE(SUM(pnl), 0)
                FROM active_trades
                WHERE status = 'CLOSED'
                  AND updated_at >= CURRENT_DATE
                """
            )
            row = cur.fetchone()
            realized_pnl = float(row[0]) if row else 0.0

            # Only trip on *losses* (negative pnl)
            if realized_pnl < 0 and abs(realized_pnl) >= max_loss:
                reason = (
                    f"Daily loss ${abs(realized_pnl):.2f} exceeds "
                    f"limit ${max_loss:.2f} ({self.max_daily_loss_pct*100:.1f}% equity)"
                )
                return False, reason
            return True, "ok"

        except Exception as e:
            logger.error(f"check_daily_pnl DB error: {e}")
            return True, "db_error_pass"  # fail-open on DB error to avoid stuck bot
        finally:
            release_conn(conn)

    def check_consecutive_losses(self):
        """
        Count the most recent consecutive CLOSED losing trades.
        If >= threshold, pause for `pause_minutes_on_consecutive` minutes.
        """
        # If we're already in a timed pause, honour it
        if self._consecutive_pause_until:
            if datetime.utcnow() < self._consecutive_pause_until:
                remaining = (self._consecutive_pause_until - datetime.utcnow()).seconds // 60
                return False, f"Consecutive-loss pause active ({remaining} min left)"
            else:
                # Pause expired — reset
                self._consecutive_pause_until = None

        conn = get_conn()
        try:
            cur = conn.cursor()
            # Get recent closed trades ordered by time descending
            cur.execute(
                """
                SELECT pnl
                FROM active_trades
                WHERE status = 'CLOSED'
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (self.max_consecutive_losses + 5,),  # a little extra buffer
            )
            rows = cur.fetchall()

            consecutive = 0
            for row in rows:
                pnl = float(row[0]) if row[0] else 0.0
                if pnl < 0:
                    consecutive += 1
                else:
                    break  # a win resets the streak

            if consecutive >= self.max_consecutive_losses:
                self._consecutive_pause_until = (
                    datetime.utcnow() + timedelta(minutes=self.pause_minutes_on_consecutive)
                )
                reason = (
                    f"{consecutive} consecutive losses >= {self.max_consecutive_losses}. "
                    f"Pausing {self.pause_minutes_on_consecutive} min."
                )
                logger.warning(f"🔴 Circuit Breaker: {reason}")
                return False, reason

            return True, "ok"

        except Exception as e:
            logger.error(f"check_consecutive_losses DB error: {e}")
            return True, "db_error_pass"
        finally:
            release_conn(conn)

    def check_daily_trade_count(self):
        """Count trades opened today (any status) and compare to max_daily_trades."""
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*)
                FROM active_trades
                WHERE created_at >= CURRENT_DATE
                """
            )
            row = cur.fetchone()
            count = int(row[0]) if row else 0

            if count >= self.max_daily_trades:
                reason = f"Daily trade count {count} >= max {self.max_daily_trades}"
                return False, reason
            return True, "ok"

        except Exception as e:
            logger.error(f"check_daily_trade_count DB error: {e}")
            return True, "db_error_pass"
        finally:
            release_conn(conn)

    def check_exposure(self, total_equity: float):
        """
        Sum notional value of all OPEN + OPEN_TPS_SET positions and compare to
        max_open_exposure_pct × equity.
        """
        if total_equity <= 0:
            return True, "equity_unavailable"

        max_notional = total_equity * self.max_open_exposure_pct

        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COALESCE(SUM(quantity * entry_price), 0)
                FROM active_trades
                WHERE status IN ('OPEN', 'OPEN_TPS_SET', 'PENDING')
                """
            )
            row = cur.fetchone()
            open_notional = float(row[0]) if row else 0.0

            if open_notional >= max_notional:
                reason = (
                    f"Open exposure ${open_notional:.2f} >= "
                    f"limit ${max_notional:.2f} ({self.max_open_exposure_pct*100:.0f}% equity)"
                )
                return False, reason
            return True, "ok"

        except Exception as e:
            logger.error(f"check_exposure DB error: {e}")
            return True, "db_error_pass"
        finally:
            release_conn(conn)

    # ------------------------------------------------------------------
    # Master check
    # ------------------------------------------------------------------

    def can_trade(self, total_equity: float):
        """
        Run every safety check.  Returns (allowed: bool, reason: str).
        On the first failure, short-circuits and returns False.
        """
        if not self.enabled:
            return True, "circuit_breaker_disabled"

        # 1. Daily PnL limit
        ok, reason = self.check_daily_pnl(total_equity)
        if not ok:
            self._tripped_reason = reason
            return False, reason

        # 2. Consecutive losses
        ok, reason = self.check_consecutive_losses()
        if not ok:
            self._tripped_reason = reason
            return False, reason

        # 3. Daily trade count
        ok, reason = self.check_daily_trade_count()
        if not ok:
            self._tripped_reason = reason
            return False, reason

        # 4. Open exposure
        ok, reason = self.check_exposure(total_equity)
        if not ok:
            self._tripped_reason = reason
            return False, reason

        return True, "ok"

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all circuit-breaker state. Call at midnight or manually."""
        self._consecutive_pause_until = None
        self._tripped_reason = None
        logger.info("🔄 Circuit Breaker reset — all checks cleared.")

    def status(self) -> dict:
        """Return a dict summarising current circuit-breaker state."""
        paused_by_consecutive = (
            self._consecutive_pause_until is not None
            and datetime.utcnow() < self._consecutive_pause_until
        )
        return {
            "enabled": self.enabled,
            "tripped_reason": self._tripped_reason,
            "consecutive_pause_active": paused_by_consecutive,
            "consecutive_pause_until": (
                self._consecutive_pause_until.isoformat() if self._consecutive_pause_until else None
            ),
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "pause_minutes_on_consecutive": self.pause_minutes_on_consecutive,
            "max_daily_trades": self.max_daily_trades,
            "max_open_exposure_pct": self.max_open_exposure_pct,
        }


# Module-level singleton for easy import
circuit_breaker = CircuitBreaker()
