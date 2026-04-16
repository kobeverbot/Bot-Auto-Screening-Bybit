import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Mapping: env var name -> (dict_path parts)
_ENV_OVERRIDES = {
    'BYBIT_API_KEY':              ('api', 'bybit_key'),
    'BYBIT_API_SECRET':           ('api', 'bybit_secret'),
    'DISCORD_WEBHOOK_URL':        ('api', 'discord_webhook'),
    'DISCORD_LIVE_WEBHOOK_URL':   ('api', 'discord_live_webhook'),
    'DISCORD_DASHBOARD_WEBHOOK_URL': ('api', 'discord_dashboard_webhook'),
    'DB_HOST':     ('database', 'host'),
    'DB_NAME':     ('database', 'database'),
    'DB_USER':     ('database', 'user'),
    'DB_PASSWORD': ('database', 'password'),
    'DB_PORT':     ('database', 'port'),
}


def _apply_env_overrides(config):
    """Override config values with matching environment variables when set."""
    for env_var, path in _ENV_OVERRIDES.items():
        value = os.getenv(env_var)
        if value is not None:
            section, key = path
            if section not in config:
                config[section] = {}
            config[section][key] = value
            logger.debug("Overrode config[%s][%s] from env %s", section, key, env_var)


def _ensure_nested_keys(config):
    """Ensure top-level sections exist so key assignments don't raise KeyError."""
    for _, (section, _key) in _ENV_OVERRIDES.items():
        if section not in config:
            config[section] = {}


def validate_config(config):
    """Validate the loaded configuration.

    Warns about placeholder or missing critical values.
    Returns True if all critical keys are present, False otherwise.
    """
    valid = True

    # --- API keys ---
    api = config.get('api', {})
    for key_name in ('bybit_key', 'bybit_secret'):
        value = api.get(key_name, '')
        if not value or 'YOUR_' in value:
            logger.warning("API key '%s' is missing or looks like a placeholder", key_name)
            valid = False

    # --- Database password ---
    db = config.get('database', {})
    if not db.get('password'):
        logger.warning("Database password is empty")
        valid = False

    # --- Database host ---
    if not db.get('host'):
        logger.warning("Database host is empty")
        valid = False

    return valid


def load_config():
    if not os.path.exists('config.json'):
        logger.warning("config.json not found; starting with empty config")
        config = {}
    else:
        with open('config.json', 'r') as f:
            config = json.load(f)

    _ensure_nested_keys(config)

    # Environment overrides (take precedence over config.json)
    _apply_env_overrides(config)

    if os.getenv('BOT_ENV') == 'testing':
        logger.warning("⚠️ RUNNING IN TEST MODE")
        config.setdefault('database', {})['database'] = 'bybit_bot_test'

    return config


CONFIG = load_config()
