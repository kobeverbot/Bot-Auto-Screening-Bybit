"""
Rate Limiter for Bybit API calls.
Prevents hitting exchange rate limits during mass screening.
"""
import time
import logging
import threading

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Thread-safe for use with ThreadPoolExecutor.
    """
    def __init__(self, max_calls=20, per_seconds=1, burst=5):
        """
        Args:
            max_calls: Maximum API calls allowed in the window
            per_seconds: Time window in seconds
            burst: Max consecutive calls before forced wait
        """
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()
        self.total_waits = 0

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * (self.max_calls / self.per_seconds)
        if new_tokens > 0:
            self.tokens = min(self.burst, self.tokens + new_tokens)
            self.last_refill = now

    def acquire(self):
        """Block until a token is available."""
        with self.lock:
            self._refill()
            while self.tokens < 1:
                sleep_time = self.per_seconds / self.max_calls
                self.lock.release()
                time.sleep(sleep_time)
                self.lock.acquire()
                self._refill()
                self.total_waits += 1
            self.tokens -= 1

    def try_acquire(self, timeout=5.0):
        """Try to acquire a token, return True if successful within timeout."""
        deadline = time.monotonic() + timeout
        with self.lock:
            while time.monotonic() < deadline:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                self.lock.release()
                time.sleep(0.05)
                self.lock.acquire()
            return False

    def status(self):
        """Return current rate limiter status."""
        with self.lock:
            self._refill()
            return {
                "tokens_available": round(self.tokens, 2),
                "max_calls_per_sec": self.max_calls / self.per_seconds,
                "total_waits": self.total_waits
            }


# Module-level singleton — configurable via config.json
_limiter = None


def get_rate_limiter():
    """Get or create the global rate limiter instance."""
    global _limiter
    if _limiter is None:
        try:
            from modules.config_loader import CONFIG
            rl_config = CONFIG.get('rate_limiter', {})
            _limiter = RateLimiter(
                max_calls=rl_config.get('max_calls', 20),
                per_seconds=rl_config.get('per_seconds', 1),
                burst=rl_config.get('burst', 10)
            )
        except Exception:
            # Fallback to safe defaults
            _limiter = RateLimiter(max_calls=20, per_seconds=1, burst=10)
    return _limiter


def rate_limited(func):
    """Decorator to rate-limit any function that makes API calls."""
    def wrapper(*args, **kwargs):
        limiter = get_rate_limiter()
        limiter.acquire()
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
