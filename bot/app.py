from flask import Flask, render_template, request
from strategies.strategy_methods import (
    get_stock_data,
    get_options_data,
    generate_trade_recommendation,
    calculate_moving_average
)
from strategies.monte_carlo import monte_carlo_option_price
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'


# Initialize Alpaca Trading client
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
attributes = dir(trading_client)
# print("Attributes and Methods of trading_client:")
# for attribute in attributes:
#     print(attribute)

###########
# HELPERS #
###########

def get_account():
    """Fetch account information."""
    account = trading_client.get_account()
    return account

def get_positions():
    """Fetch current positions."""
    positions = trading_client.get_all_positions()
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
    order = trading_client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force
    )
    return order

def get_order_history():
    """Fetch order history."""
    orders = trading_client.list_orders()
    return orders

def get_asset(symbol):
    """Fetch asset information."""
    asset = trading_client.get_asset(symbol)
    return asset

def get_eqts_YF():
    """Fetch equities data from Yahoo Finance."""
    # Implementation to fetch options from Yahoo Finance
    pass

def get_eqts_Alp():
    """Fetch equities data from Alpaca."""
    # Implementation to fetch options from Alpaca
    pass

def get_opts_YF():
    """Fetch options data from Yahoo Finance."""
    # Implementation to fetch options from Yahoo Finance
    pass

def get_opts_Alp():
    """Fetch options data from Alpaca."""
    # Implementation to fetch options from Alpaca
    pass

def AI_goober():
    """Perform AI-based operations for recommendations."""
    # Implementation for AI recommendation logic
    pass

def visuals(**args):
    """Generate visual data for the app."""
    # Implementation for visualizations
    pass

def e_info():
    """Fetch extra information."""
    # Implementation for extra information logic
    pass

#############
# ROUTES    #
#############

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    """Render the portfolio page with current holdings."""
    port = get_portfolio()
    
    # Prepare data for the pie chart visualization
    portfolio_data = {
        "values": [float(pos.market_value) for pos in port['positions']],
        "labels": [pos.symbol for pos in port['positions']]
    }
    
    return render_template('portfolio.html', portfolio=port, portfolio_data=portfolio_data)


@app.route('/options_equities', methods=['GET', 'POST'])
def options_equities():
    """Render the options equities page."""
    optsYF = get_opts_YF()
    optsAlp = get_opts_Alp()
    return render_template('options_equities.html', optsYF=optsYF, optsAlp=optsAlp)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    """Render the recommendation page with AI-generated insights."""
    AI_goober()
    return render_template('recommendation.html')

@app.route('/visuals', methods=['GET', 'POST'])
def visualizations():
    """Render the visuals page with generated graphics."""
    visuals()
    return render_template('visuals.html')

@app.route('/extra_information', methods=['GET', 'POST'])
def extra_information():
    """Render the extra information page."""
    e_info()
    return render_template('extra_information.html')

if __name__ == '__main__':
    app.run(debug=True)
    # pass 