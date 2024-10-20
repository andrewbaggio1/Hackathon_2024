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
import datetime
import pandas as pd

import math

# load_dotenv()

# Initialize Alpaca Data client
# data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Initialize Alpaca Trading client
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# HF_KEY = os.getenv('HF_Key')
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

# # Helper function to fetch stock data with optional indicators
# def get_stock_data(stock_symbol, period='1mo'):
#     """Fetch historical stock data using yfinance and clean NaN values."""
#     try:
#         stock = yf.Ticker(stock_symbol)
#         hist = stock.history(period=period)
#         if hist.empty:
#             return None, None, None, None, None, None

#         # Calculate technical indicators: Bollinger Bands, Moving Averages
#         hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
#         hist['Upper Band'] = hist['SMA_20'] + (2 * hist['Close'].rolling(window=20).std())
#         hist['Lower Band'] = hist['SMA_20'] - (2 * hist['Close'].rolling(window=20).std())
#         hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()

#         # Clean NaN values before preparing the data for JSON
#         hist = hist.fillna(value=None)

#         # Prepare data for JSON response
#         dates = hist.index.strftime('%Y-%m-%d').tolist()
#         prices = hist['Close'].tolist()
#         sma20 = hist['SMA_20'].tolist()
#         upper_band = hist['Upper Band'].tolist()
#         lower_band = hist['Lower Band'].tolist()
#         ema20 = hist['EMA_20'].tolist()

#         return dates, prices, sma20, upper_band, lower_band, ema20

#     except Exception as e:
#         print(f"Error fetching stock data for {stock_symbol}: {e}")
#         return None, None, None, None, None, None

# Helper function to fetch stock price data
def get_stock_data(stock_symbol, period='1mo'):
    """Fetch historical stock price data using yfinance."""
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period=period)
        
        # Check if the historical data is empty
        if hist.empty:
            return None, None
        
        # Prepare data for JSON response
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        return dates, prices

    except Exception as e:
        print(f"Error fetching stock data for {stock_symbol}: {e}")
        return None, None

# Function to fetch options data using yfinance
def get_options_data(stock_symbol):
    """Fetch options data using yfinance."""
    try:
        stock = yf.Ticker(stock_symbol)  # Corrected from yf.TTicker to yf.Ticker
        expiration_dates = stock.options
        options_data = []
    
        for date in expiration_dates:
            print(f"Fetching options for {stock_symbol} with expiration date: {date}")
            try:
                # Fetch calls and puts
                calls = stock.option_chain(date).calls
                puts = stock.option_chain(date).puts

                # Clean NaN values in the calls and puts DataFrames
                calls_cleaned = clean_nan_values(calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']])
                puts_cleaned = clean_nan_values(puts[['strike', 'lastPrice', 'bid', 'ask', 'volume']])

                options_data.append({
                    'expiration_date': date,
                    'calls': calls_cleaned,
                    'puts': puts_cleaned
                })
            except Exception as e:
                print(f"Error fetching options for expiration {date}: {e}")

        print(f"Options data fetched and cleaned for {stock_symbol}")
        return options_data
    except Exception as e:
        print(f"Error fetching options data for {stock_symbol}: {e}")
        return []



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
        "cash": float(account_info.cash),
        "equity": float(account_info.equity),
        "buying_power": float(account_info.buying_power),
        "positions": positions
    }
    return portfolio_summary

def place_order1(symbol, qty, side, order_type='market', time_in_force='gtc'):
    """Place a new order."""
    order = trading_client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force
    )
    return order

def place_order(symbol, qty, side):
    # Create market order request
    market_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    # Submit order
    market_order = trading_client.submit_order(market_order_data)
    return market_order

def get_order_history():
    """Fetch order history."""
    orders = trading_client.get_orders(status='all', limit=50)
    return orders

def get_asset(symbol):
    """Fetch asset information."""
    asset = trading_client.get_asset(symbol)
    return asset

def get_recent_trades():
    """Fetch recent trades/orders."""
    orders = trading_client.get_orders(status='all', limit=50)
    return orders

# # LLM Functions
# def preprocess_data(historical_data):
#     """Process historical stock data."""
#     historical_data['daily_return'] = historical_data['Close'].pct_change()
#     historical_data['volatility'] = historical_data['daily_return'].rolling(window=7).std()
#     historical_data['daily_range'] = historical_data['High'] - historical_data['Low']
#     historical_data['volume_change'] = historical_data['Volume'].pct_change()
#     return historical_data

# def prepare_data_summary(tickers):
#     """Prepare data summary from processed data."""
#     summaries = []
#     for ticker in tickers:
#         try:
#             # Fetch historical data using yfinance
#             stock = yf.Ticker(ticker)
#             historical_data = stock.history(period='1mo')  # Get 1 month of data
#             if historical_data.empty:
#                 continue
#             processed_data = preprocess_data(historical_data)

#             # Extract key insights
#             avg_return = processed_data['daily_return'].mean()
#             avg_volatility = processed_data['volatility'].mean()
#             avg_daily_range = processed_data['daily_range'].mean()
#             avg_volume_change = processed_data['volume_change'].mean()
#             summaries.append(
#                 f"{ticker}: Avg Return: {avg_return:.2%}, Avg Volatility: {avg_volatility:.2%}, "
#                 f"Avg Daily Range: {avg_daily_range:.2f}, Avg Volume Change: {avg_volume_change:.2%}"
#             )
#         except Exception as e:
#             print(f"Error processing data for {ticker}: {e}")
#             continue
#     return "\n".join(summaries)

# def generate_market_snapshot(data_summary):
#     """
#     Generate a market snapshot using an LLM.

#     :param data_summary: Summary of processed data to feed into the LLM.
#     :return: Market snapshot as plain text.
#     """
#     prompt = f"""
#     You are a financial market analyst. Here is this hour's processed market data:
#     {data_summary}

#     Provide a concise summary in plain language about the market's performance this hour.
#     Include insights on major trends, noteworthy stock movements, and any changes in options activity.
#     """
#     # Use transformers pipeline for text summarization
#     summarizer = pipeline(
#         "summarization", 
#         model="facebook/bart-large-cnn",
#         use_auth_token=HF_KEY
#         )
#     response = summarizer(prompt, max_length=150, min_length=50, do_sample=False)
#     return response[0]['summary_text']

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

# Flask route to handle stock data API requests
# @app.route('/stock_data', methods=['GET'])
# def stock_data():
#     symbol = request.args.get('symbol')
#     period = request.args.get('period', default='1mo')  # Get period from query parameters, default to '1mo'

#     # Fetch stock data using the helper function
#     dates, prices, sma20, upper_band, lower_band, ema20 = get_stock_data(symbol, period)

#     if dates is None or prices is None:
#         return jsonify({"error": f"Failed to fetch data for symbol: {symbol}"}), 400

#     # Format the data for response
#     data = {
#         'symbol': symbol,
#         'name': yf.Ticker(symbol).info.get('longName', symbol),  # Get the company name
#         'dates': dates,
#         'prices': prices,
#         'sma20': sma20,  # 20-day Simple Moving Average
#         'upper_band': upper_band,  # Upper Bollinger Band
#         'lower_band': lower_band,  # Lower Bollinger Band
#         'ema20': ema20,  # 20-day Exponential Moving Average
#     }

#     print(f"Stock data fetched for {symbol}: {data}")
#     return jsonify(data)

@app.route('/stock_data', methods=['GET'])
def stock_data():
    symbol = request.args.get('symbol')
    period = request.args.get('period', default='1mo')  # Get period from query parameters, default to '1mo'

    # Fetch stock data using the updated helper function
    dates, prices = get_stock_data(symbol, period)

    if dates is None or prices is None:
        return jsonify({"error": f"Failed to fetch data for symbol: {symbol}"}), 400

    # Format the data for response
    data = {
        'symbol': symbol,
        'name': yf.Ticker(symbol).info.get('longName', symbol),  # Get the company name
        'dates': dates,
        'prices': prices,
    }

    print(f"Stock data fetched for {symbol}: {data}")
    return jsonify(data)


@app.route('/options_data', methods=['GET'])
def options_data():
    symbol = request.args.get('symbol')
    stock = yf.Ticker(symbol)
    options_chain = stock.options
    options_data = []

    for expiration in options_chain:
        # print(f"Processing expiration: {expiration}")
        try:
            calls = stock.option_chain(expiration).calls
            puts = stock.option_chain(expiration).puts

            # Ensure to fill NaN values and convert to a list of dictionaries
            calls_cleaned = calls.where(pd.notnull(calls), None).to_dict(orient='records')
            puts_cleaned = puts.where(pd.notnull(puts), None).to_dict(orient='records')

            options_data.append({
                'expiration_date': expiration,
                'calls': calls_cleaned,
                'puts': puts_cleaned
            })
        except Exception as e:
            print(f"Error processing options for expiration {expiration}: {e}")

    print(f"Final cleaned options data for {symbol}")  # Final cleaned data output
    return jsonify(options_data)



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