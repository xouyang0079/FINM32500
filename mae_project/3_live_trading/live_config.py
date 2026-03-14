# -*- coding: utf-8 -*-
"""
Live trading configuration (hardcoded version for class project).
WARNING: Do NOT commit real keys to public GitHub.
"""

# =========================
# Alpaca API credentials
# =========================

ALPACA_API_KEY = "PKAQHQZWA2AHHIJPIXYBUG7GZB"
ALPACA_SECRET_KEY = "GPduQrJ8q27Md64bYfonqYeZkuLgc8i8swZqvQYSh5bS"

# ALPACA_API_KEY = "PKEYLJ6TNGJD3YBSDPBVASBSVX"
# ALPACA_SECRET_KEY = "6QaZXwaDqNXy31VWbXceioAVCdy2nSyV7UEocp3cHPzn"

# For Alpaca paper trading:
#   https://paper-api.alpaca.markets
# For live trading:
#   https://api.alpaca.markets
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# =========================
# Trading behavior
# =========================
USE_PAPER_TRADING = True
EXECUTE_AFTER_OPEN_MINUTES = 10  # your requirement
TIMEZONE_EXCHANGE = "America/New_York"

# Your strategy/account tag for idempotency
STRATEGY_NAME = "ml_return_to_weight_v1"

# Optional risk/trade knobs
MIN_TRADE_NOTIONAL = 50.0
ALLOW_FRACTIONAL = True