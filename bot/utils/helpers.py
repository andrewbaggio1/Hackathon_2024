import os
import alpaca_trade_api as tradeapi

# Set API keys and endpoint
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def get_account():
    """Fetch account information."""
    account = api.get_account()
    return account

def get_positions():
    """Fetch current positions."""
    positions = api.list_positions()
    return positions

def get_portfolio():
    """Fetch portfolio information including account and positions."""
    account_info = get_account()
    positions = get_positions()

    # Format and return portfolio information
    portfolio_summary = {
        "cash": account_info.cash,
        "equity": account_info.equity,
        "buying_power": account_info.buying_power,
        "positions": positions
    }
    return portfolio_summary

def place_order(symbol, qty, side, order_type='market', time_in_force='gtc'):
    """Place a new order."""
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force
    )
    return order

def get_order_history():
    """Fetch order history."""
    orders = api.list_orders()
    return orders

def get_asset(symbol):
    """Fetch asset information."""
    asset = api.get_asset(symbol)
    return asset

