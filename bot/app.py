from flask import Flask, render_template, request, jsonify

import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os
import requests  # Import requests
import yfinance as yf  # Import yfinance
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

###########
# HELPERS #
###########

def get_stock_data(stock_symbol):
    """Fetch daily stock data from Alpha Vantage."""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={api_key}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            return None, None  # Handle API errors gracefully
        
        time_series = data['Time Series (Daily)']
        dates = list(time_series.keys())[:365]
        prices = [float(time_series[date]['4. close']) for date in dates]
        
        return dates, prices
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None, None

def get_options_data(stock_symbol):
    """Fetch options data using Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    expiration_dates = stock.options
    options_data = {}
    
    for date in expiration_dates:
        calls = stock.option_chain(date).calls
        puts = stock.option_chain(date).puts
        options_data[date] = {
            'calls': calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']],
            'puts': puts[['strike', 'lastPrice', 'bid', 'ask', 'volume']]
        }
    return options_data

# Alpaca Trading API Helpers
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

###########
# ROUTES  #
###########

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the index page."""
    return render_template('portfolio.html')

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    """Render the portfolio page with current holdings."""
    port = get_portfolio()
    
    # Convert relevant values to floats if they are in string format
    port['cash'] = float(port['cash']) if isinstance(port['cash'], str) else port['cash']
    port['equity'] = float(port['equity']) if isinstance(port['equity'], str) else port['equity']
    port['buying_power'] = float(port['buying_power']) if isinstance(port['buying_power'], str) else port['buying_power']

    for pos in port['positions']:
        pos.market_value = float(pos.market_value) if isinstance(pos.market_value, str) else pos.market_value
    
    # Prepare data for the pie chart visualization
    portfolio_data = {
        "values": [pos.market_value for pos in port['positions']],
        "labels": [pos.symbol for pos in port['positions']]
    }
    
    return render_template('portfolio.html', portfolio=port, portfolio_data=portfolio_data)

@app.route('/stock_data', methods=['GET'])
def stock_data():
    symbol = request.args.get('symbol')
    stock = yf.Ticker(symbol)
    
    # Get current price and name
    data = {
        'symbol': symbol,
        'name': stock.info['longName'],
        'price': stock.history(period='1d')['Close'].iloc[-1],
    }
    
    return jsonify(data)

@app.route('/options_data', methods=['GET'])
def options_data():
    symbol = request.args.get('symbol')
    stock = yf.Ticker(symbol)
    options_chain = stock.options
    options_data = []

    for expiration in options_chain:
        calls = stock.option_chain(expiration).calls
        puts = stock.option_chain(expiration).puts
        options_data.append({
            'expiration_date': expiration,
            'calls': calls.to_dict(orient='records'),
            'puts': puts.to_dict(orient='records')
        })

    return jsonify(options_data)

@app.route('/historical_data', methods=['GET'])
def historical_data():
    """Fetch historical stock data and calculate technical indicators."""
    symbol = request.args.get('symbol')
    period = request.args.get('period', default='1mo')  # Get period from query parameters with default value
    stock = yf.Ticker(symbol)
    
    # Get historical data based on the specified period
    historical_data = stock.history(period=period)  
    historical_data.reset_index(inplace=True)  # Reset index to have dates as a column

    # Calculate technical indicators (e.g., moving average, Bollinger Bands)
    historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()
    historical_data['Upper Band'] = historical_data['SMA_20'] + 2 * historical_data['Close'].rolling(window=20).std()
    historical_data['Lower Band'] = historical_data['SMA_20'] - 2 * historical_data['Close'].rolling(window=20).std()

    # Convert the DataFrame to a list of dictionaries and format dates
    historical_data_list = historical_data[['Date', 'Close', 'SMA_20', 'Upper Band', 'Lower Band']].to_dict(orient='records')

    # Adjust the date formatting to ISO format
    for entry in historical_data_list:
        entry['Date'] = entry['Date'].isoformat()  # Convert to ISO format string

    return jsonify(historical_data_list)


@app.route('/options_equities', methods=['GET', 'POST'])
def options_equities():
    """Render the options equities page."""
    options_data = {}
    
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol').strip().upper()  # Get the input stock symbol
        if stock_symbol:
            # Fetch options for the inputted stock symbol
            options_data = get_options_data(stock_symbol)
    
    return render_template('options_equities.html', options_data=options_data)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    """Render the recommendation page with AI-generated insights."""
    return render_template('recommendation.html')

@app.route('/visuals', methods=['GET', 'POST'])
def visualizations():
    """Render the visuals page with generated graphics."""
    return render_template('visuals.html')

@app.route('/extra_information', methods=['GET', 'POST'])
def extra_information():
    """Render the extra information page."""
    return render_template('extra_information.html')

if __name__ == '__main__':
    app.run(debug=True)
