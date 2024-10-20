from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import pandas as pd
import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
# from transformers import pipeline  # Import the transformers library
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus
import datetime
import pandas as pd

import math

load_dotenv()

app = Flask(__name__)

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Initialize Alpaca Trading client
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
# print(dir(trading_client))

# print(data_client)
HF_KEY = os.getenv('HF_Key')
###########
# HELPERS #
###########

def clean_nan_values(option_data):
    """Replace NaN values in the option data with None."""
    print("Cleaning NaN values from options data...")
    for option in option_data:
        # Iterate over calls and replace NaN values with None
        for call in option['calls']:
            call['lastPrice'] = None if pd.isna(call['lastPrice']) else call['lastPrice']
            call['bid'] = None if pd.isna(call['bid']) else call['bid']
            call['ask'] = None if pd.isna(call['ask']) else call['ask']
            call['volume'] = None if pd.isna(call['volume']) else call['volume']

        # Iterate over puts and replace NaN values with None
        for put in option['puts']:
            put['lastPrice'] = None if pd.isna(put['lastPrice']) else put['lastPrice']
            put['bid'] = None if pd.isna(put['bid']) else put['bid']
            put['ask'] = None if pd.isna(put['ask']) else put['ask']
            put['volume'] = None if pd.isna(put['volume']) else put['volume']
    
    print("NaN values cleaning completed.")
    return option_data



# Example to fetch recent bars for a symbol
def get_real_time_stock_data(symbol):
    start = datetime.datetime.now() - datetime.timedelta(days=1)
    end = datetime.datetime.now()

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end
    )

    # Fetch real-time stock data
    bars = data_client.get_stock_bars(request_params).df
    return bars

# Helper function to fetch stock data with optional indicators
def get_stock_data(stock_symbol, period='1mo'):
    """Fetch historical stock data using yfinance and clean NaN values."""
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return None, None, None, None, None, None

        # Calculate technical indicators: Bollinger Bands, Moving Averages
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['Upper Band'] = hist['SMA_20'] + (2 * hist['Close'].rolling(window=20).std())
        hist['Lower Band'] = hist['SMA_20'] - (2 * hist['Close'].rolling(window=20).std())
        hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()

        # Drop rows where any NaN values exist in the technical indicators
        hist = hist.dropna(subset=['SMA_20', 'Upper Band', 'Lower Band', 'EMA_20'])

        # Prepare data for JSON response
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        sma20 = hist['SMA_20'].tolist()
        upper_band = hist['Upper Band'].tolist()
        lower_band = hist['Lower Band'].tolist()
        ema20 = hist['EMA_20'].tolist()

        return dates, prices, sma20, upper_band, lower_band, ema20

    except Exception as e:
        print(f"Error fetching stock data for {stock_symbol}: {e}")
        return None, None, None, None, None, None

## Function to fetch options data using yfinance
def get_options_data(stock_symbol):
    """Fetch options data using yfinance."""
    try:
        stock = yf.Ticker(stock_symbol)
        expiration_dates = stock.options  # Get all available expiration dates
        options_data = []  # List to hold all options data for each expiration date

        # Loop over each expiration date and fetch calls and puts
        for date in expiration_dates:
            try:
                # Fetch the option chain for the specific expiration date
                option_chain = stock.option_chain(date)

                # Extract calls and puts
                calls = option_chain.calls
                puts = option_chain.puts

                # Convert calls and puts data to dictionaries
                calls_cleaned = calls.to_dict(orient='records')
                puts_cleaned = puts.to_dict(orient='records')

                # Append the expiration date and options data to the result
                options_data.append({
                    'expiration_date': date,
                    'calls': calls_cleaned,
                    'puts': puts_cleaned
                })

            except Exception as e:
                print(f"Error fetching options for expiration {date}: {e}")
                continue

        return options_data  # Return the complete options data with expiration dates
    except Exception as e:
        print(f"Error fetching options data for {stock_symbol}: {e}")
        return []

# act = trading_client.get_account
# print(act)

# Alpaca Trading API Helpers
def get_account():
    """Fetch account information."""
    account = trading_client.get_account
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
        "cash": float(account_info.cash),
        "equity": float(account_info.equity),
        "buying_power": float(account_info.buying_power),
        "positions": positions
    }
    return portfolio_summary

def place_order1(symbol, qty, side, order_type='market', time_in_force='gtc'):
    """Place a new order."""
    # Create an order request object
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
        time_in_force=TimeInForce(time_in_force.upper())
    )
    # Submit the order
    order = trading_client.submit_order(order_data)
    return order

def place_order(symbol, qty, side):
    # Create market order request
    market_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    # Submit order
    market_order = trading_client.submit_order(market_order_data)
    return market_order

def get_order_history():
    """Fetch order history."""
    # Create a filter for orders
    orders_request = GetOrdersRequest(
        status=OrderStatus.ALL,
        limit=100
    )
    orders = trading_client.get_orders(filter=orders_request)
    return orders

def get_asset(symbol):
    """Fetch asset information."""
    asset = trading_client.get_asset(symbol)
    return asset

def get_recent_trades():
    """Fetch recent trades/orders."""
    orders_request = GetOrdersRequest(
        status=OrderStatus.ALL,
        limit=50
    )
    orders = trading_client.get_orders(filter=orders_request)
    return orders

###########
# ROUTES  #
###########

@app.route('/', methods=['GET', 'POST'])
def index():
    # """Render the index page with LLM output and portfolio data."""
    # # List of tickers to summarize (top 50 S&P 500 stocks)
    # tickers = [
    #     'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'META', 'UNH',
    #     'XOM', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
    #     'KO', 'MRK', 'LLY', 'BAC', 'PEP', 'PFE', 'COST', 'TMO', 'DIS', 'CSCO',
    #     'MCD', 'ABT', 'DHR', 'ACN', 'WFC', 'AVGO', 'ADBE', 'VZ', 'CRM', 'TXN',
    #     'NEE', 'CMCSA', 'NFLX', 'INTC', 'LIN', 'NKE', 'AMD', 'MDT', 'UNP', 'QCOM'
    # ]

    # # Prepare data summary and generate market snapshot
    # data_summary = prepare_data_summary(tickers)
    # llm_output = generate_market_snapshot(data_summary)

    # # Get portfolio data
    # port = get_portfolio()

    # # Get recent trades
    # trades = get_recent_trades()

    # Pass data to the template
    return render_template(
        'portfolio.html',
        # llm_output=llm_output,
        # tickers=tickers,
        # portfolio=port,
        # trades=trades
    )

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

@app.route('/stock_data', methods=['GET'])
def stock_data():
    symbol = request.args.get('symbol')
    period = request.args.get('period', default='1mo')  # Get period from query parameters, default to '1mo'

    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    # Fetch stock data with Bollinger Bands
    dates, prices, sma20, upper_band, lower_band, ema20 = get_stock_data(symbol, period)

    if dates is None or prices is None:
        return jsonify({"error": f"Failed to fetch data for symbol: {symbol}"}), 400

    # Format the data for response
    data = {
        'symbol': symbol,
        'name': yf.Ticker(symbol).info.get('longName', symbol),  # Get the company name
        'dates': dates,
        'prices': prices,
        'sma20': sma20,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'ema20': ema20
    }

    print(f"Stock data with Bollinger Bands fetched for {symbol}")
    return jsonify(data)


@app.route('/options_data', methods=['GET'])
def options_data():
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    # Fetch options data using the enhanced get_options_data function
    options = get_options_data(symbol)

    if not options:
        return jsonify({"error": "Failed to fetch options data"}), 500

    return jsonify(options)

@app.route('/submit_trade', methods=['POST'])
def submit_trade():
    data = request.json
    symbol = data.get('symbol')
    qty = data.get('qty')
    side = data.get('side')

    if not symbol or not qty or not side:
        return jsonify({'error': 'Invalid trade data provided.'}), 400

    try:
        # Place the order with Alpaca API
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        order = trading_client.submit_order(order_data)
        return jsonify({'message': 'Trade executed successfully', 'order_id': order.id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trades_history', methods=['GET'])
def trades_history():
    """Return a list of all trades (executed and pending)."""
    trades = get_order_history()
    
    # Format trades for the frontend
    formatted_trades = []
    for trade in trades:
        formatted_trades.append({
            'id': trade.id,
            'symbol': trade.symbol,
            'qty': float(trade.qty),
            'side': trade.side,
            'type': trade.order_class,  # Use order_class or type depending on your needs
            'status': trade.status,
            'submitted_at': trade.submitted_at.strftime('%Y-%m-%d %H:%M:%S') if trade.submitted_at else 'N/A',
            'filled_at': trade.filled_at.strftime('%Y-%m-%d %H:%M:%S') if trade.filled_at else 'N/A',
            'filled_qty': float(trade.filled_qty),
        })
    
    return jsonify({'trades': formatted_trades})




@app.route('/cancel_trade', methods=['POST'])
def cancel_trade():
    """Cancel a pending or partially filled trade."""
    data = request.json
    trade_id = data.get('trade_id')

    if not trade_id:
        return jsonify({'error': 'Trade ID is required to cancel the trade.'}), 400

    try:
        # Attempt to cancel the trade
        trading_client.cancel_order_by_id(trade_id)
        return jsonify({'message': f'Trade {trade_id} canceled successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/port', methods=['GET'])
def port():
    """Render the portfolio page with current holdings."""
    port = get_portfolio()
    
    # Prepare data for the pie chart visualization
    portfolio_data = {
        "values": [float(pos.market_value) for pos in port['positions']],
        "labels": [pos.symbol for pos in port['positions']]
    }
    
    # Prepare portfolio details like buying power, cash, and equity
    portfolio_details = {
        "buying_power": float(port['buying_power']),
        "cash": float(port['cash']),
        "equity": float(port['equity']),
    }

    # Return data in JSON format to be consumed by the frontend
    return jsonify({
        "portfolio_data": portfolio_data,
        "portfolio_details": portfolio_details
    })

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

@app.route('/extra_information', methods=['GET', 'POST'])
def extra_information():
    """Render the extra information page."""
    return render_template('extra_information.html')

if __name__ == '__main__':
    app.run(debug=True)