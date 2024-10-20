# app.py
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import pandas as pd
import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce, OrderStatus, OrderClass
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    TrailingStopOrderRequest,
    TakeProfitRequest,
    StopLossRequest
)
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime
import math
import logging
import openai
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from llms.market_weather_report import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
OA_KEY = os.getenv('OAI_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'  # Ensure this is correct

# Initialize Alpaca clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

HF_KEY = os.getenv('HF_Key')  # If used elsewhere

###########
# HELPERS #
###########

# Configure logging to only display errors
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_market_summary(market_data_text):
    """
    Generates a summary of the market data using OpenAI API.

    Args:
    - market_data_text: A string containing market data and trends.

    Returns:
    - summary: A string containing the generated summary.
    """
    try:
        # Prepare the prompt with market data
        prompt = f"Provide a concise and insightful summary of the following market data and trends:\n\n{market_data_text}\n\nSummary:"
        
        # Call OpenAI's API
        response = openai.Completion.create(
            engine='text-davinci-003',  # Use the appropriate engine
            prompt=prompt,
            max_tokens=150,
            temperature=0.5,
            n=1,
            stop=None,
        )
        summary = response.choices[0].text.strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating market summary: {e}")
        return "Unable to generate market summary at this time."
    
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

    try:
        # Fetch real-time stock data
        bars = data_client.get_stock_bars(request_params).df
        logger.info(f"Fetched real-time data for {symbol}")
        return bars
    except Exception as e:
        logger.error(f"Error fetching real-time stock data for {symbol}: {e}")
        return pd.DataFrame()

# Helper function to fetch stock data with optional indicators
def get_stock_data(stock_symbol, period='1mo'):
    """Fetch historical stock data using yfinance and clean NaN values."""
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period=period)
        if hist.empty:
            logger.warning(f"No historical data found for {stock_symbol}")
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

        logger.info(f"Fetched and processed historical data for {stock_symbol}")
        return dates, prices, sma20, upper_band, lower_band, ema20

    except Exception as e:
        logger.error(f"Error fetching stock data for {stock_symbol}: {e}")
        return None, None, None, None, None, None

# Alpaca Trading API Helpers
def get_account():
    """Fetch account information."""
    try:
        account = trading_client.get_account()
        logger.info("Fetched account information successfully.")
        return account
    except Exception as e:
        logger.error(f"Error fetching account information: {e}")
        return None

def get_all_assets(asset_class=AssetClass.US_EQUITY):
    """Fetch all assets of a specific class."""
    try:
        search_params = GetAssetsRequest(asset_class=asset_class)
        assets = trading_client.get_all_assets(search_params)
        logger.info(f"Fetched {len(assets)} assets successfully.")
        return assets
    except Exception as e:
        logger.error(f"Error fetching assets: {e}")
        return []

def get_positions():
    """Fetch current open positions."""
    try:
        positions = trading_client.get_all_positions()
        logger.info(f"Fetched {len(positions)} open positions successfully.")
        return positions
    except Exception as e:
        logger.error(f"Error fetching open positions: {e}")
        return []

def get_portfolio():
    """Fetch portfolio information including account and positions."""
    account_info = get_account()
    positions = get_positions()

    if not account_info:
        logger.error("No account information available.")
        return {}

    # Access and convert necessary account attributes
    try:
        portfolio_summary = {
            "cash": float(account_info.cash),
            "equity": float(account_info.equity),
            "buying_power": float(account_info.buying_power),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value),
                    "current_price": float(pos.current_price),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "change_today": float(pos.change_today),
                }
                for pos in positions
            ]
        }
        logger.info("Portfolio summary created successfully.")
        return portfolio_summary
    except AttributeError as e:
        logger.error(f"Attribute error while creating portfolio summary: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while creating portfolio summary: {e}")
        return {}

# Define supported TIFs based on Asset Class
SUPPORTED_TIF = {
    'us_equity': ['day', 'gtc', 'opg', 'cls', 'ioc', 'fok'],
    'options': ['day'],
    'crypto': ['gtc', 'ioc']
}

def place_order(symbol, qty, side, order_type='market', time_in_force='day', limit_price=None):
    """Place a new order (market or limit) with proper TIF handling."""
    try:
        # Fetch asset information to determine the asset class
        asset = get_asset(symbol)
        if not asset:
            logger.error(f"Asset {symbol} not found.")
            return {'error': f"Asset {symbol} not found."}, 400  # Return status code 400 for bad request

        asset_class = asset.asset_class  # Corrected attribute name

        # Validate the provided TIF against the supported TIFs for the asset class
        if asset_class not in SUPPORTED_TIF:
            logger.error(f"Unsupported asset class: {asset_class} for symbol {symbol}.")
            return {'error': f"Unsupported asset class: {asset_class} for symbol {symbol}."}, 400

        if time_in_force.lower() not in SUPPORTED_TIF[asset_class]:
            logger.error(f"Invalid TimeInForce '{time_in_force}' for asset class '{asset_class}'.")
            return {'error': f"Invalid TimeInForce '{time_in_force}' for asset class '{asset_class}'."}, 400

        # Map the time_in_force string to the TimeInForce enum
        try:
            tif_enum = TimeInForce(time_in_force.lower())
        except ValueError:
            logger.error(f"TimeInForce '{time_in_force}' is not recognized.")
            return {'error': f"TimeInForce '{time_in_force}' is not recognized."}, 400

        # Create the appropriate order request based on order type
        if order_type.lower() == 'market':
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=tif_enum
            )
        elif order_type.lower() == 'limit' and limit_price is not None:
            order_data = LimitOrderRequest(
                symbol=symbol,
                limit_price=limit_price,
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=tif_enum
            )
        else:
            logger.error("Invalid order type or missing limit price.")
            return {'error': 'Invalid order type or missing limit price.'}, 400

        # Submit the order
        order = trading_client.submit_order(order_data)
        logger.info(f"Placed {order_type} order {order.id} for {qty} shares of {symbol} ({side}).")
        return {'message': 'Trade executed successfully', 'order_id': order.id}, 200

    except Exception as e:
        logger.error(f"Error placing order for {symbol}: {e}")
        return {'error': str(e)}, 500

def get_order_history(limit=100):
    """Fetch order history without filtering by status."""
    try:
        orders_request = GetOrdersRequest(
            limit=limit,
            nested=True  # Show nested multi-leg orders if applicable
        )
        orders = trading_client.get_orders(filter=orders_request)
        logger.info(f"Fetched {len(orders)} orders for history.")
        return orders
    except Exception as e:
        logger.error(f"Error fetching order history: {e}")
        return []

def get_asset(symbol):
    """Fetch asset information."""
    try:
        symbol_upper = symbol.upper()
        logger.info(f"Attempting to fetch asset information for {symbol_upper}.")
        asset = trading_client.get_asset(symbol_upper)  # Correct method
        if not asset.tradable:
            logger.error(f"Asset {symbol_upper} is not tradable.")
            return None
        if asset.status != 'active':
            logger.error(f"Asset {symbol_upper} is not active.")
            return None
        logger.info(f"Fetched asset information for {symbol_upper}: {asset}")
        return asset
    except Exception as e:
        logger.error(f"Error fetching asset information for {symbol_upper}: {e}")
        return None

def get_recent_trades():
    """Fetch recent trades/orders."""
    try:
        orders_request = GetOrdersRequest(
            status=[OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.PARTIALLY_FILLED],
            limit=50,
            nested=True
        )
        orders = trading_client.get_orders(filter=orders_request)
        logger.info(f"Fetched {len(orders)} recent trades.")
        return orders
    except Exception as e:
        logger.error(f"Error fetching recent trades: {e}")
        return []

###########
# ROUTES  #
###########

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('portfolio.html')

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio_route():
    """Render the portfolio page with current holdings and LLM-generated summary."""
    port = get_portfolio()

    if not port:
        return jsonify({"error": "Failed to retrieve portfolio data."}), 500

    # Prepare data for the pie chart visualization
    portfolio_data = {
        "values": [pos['market_value'] for pos in port.get('positions', [])],
        "labels": [pos['symbol'] for pos in port.get('positions', [])]
    }

    llm_output = main_green()

    # Render the template with the portfolio data and LLM summary
    return render_template('portfolio.html', portfolio=port, portfolio_data=portfolio_data, llm_output=llm_output)

@app.route('/stock_data', methods=['GET'])
def stock_data_route():
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

    logger.info(f"Stock data with Bollinger Bands fetched for {symbol}")
    return jsonify(data)

@app.route('/submit_trade', methods=['POST'])
def submit_trade_route():
    """Submit a trade order."""
    data = request.json
    symbol = data.get('symbol')
    qty = data.get('qty')
    side = data.get('side')
    order_type = data.get('order_type', 'market')  # Default to 'market'
    limit_price = data.get('limit_price', None)
    time_in_force = data.get('time_in_force', 'day')  # Default to 'day'

    if not symbol or not qty or not side:
        return jsonify({'error': 'Invalid trade data provided.'}), 400

    try:
        # Place the order with Alpaca API
        response, status_code = place_order(symbol, qty, side, order_type, time_in_force, limit_price)
        return jsonify(response), status_code
    except Exception as e:
        logger.error(f"Error submitting trade: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/trades_history', methods=['GET'])
def trades_history_route():
    """Return a list of all trades (executed and pending)."""
    trades = get_order_history()

    # Format trades for the frontend
    formatted_trades = []
    for trade in trades:
        formatted_trades.append({
            'id': trade.id,
            'symbol': trade.symbol,
            'qty': float(trade.qty),
            'side': trade.side.value,  # 'buy' or 'sell'
            'type': trade.order_class.value,  # e.g., 'simple', 'bracket'
            'status': trade.status.value,  # e.g., 'new', 'filled'
            'submitted_at': trade.submitted_at.strftime('%Y-%m-%d %H:%M:%S') if trade.submitted_at else 'N/A',
            'filled_at': trade.filled_at.strftime('%Y-%m-%d %H:%M:%S') if trade.filled_at else 'N/A',
            'filled_qty': float(trade.filled_qty),
        })

    return jsonify({'trades': formatted_trades})

@app.route('/cancel_trade', methods=['POST'])
def cancel_trade_route():
    """Cancel a pending or partially filled trade."""
    data = request.json
    trade_id = data.get('trade_id')

    if not trade_id:
        return jsonify({'error': 'Trade ID is required to cancel the trade.'}), 400

    try:
        # Attempt to cancel the trade
        trading_client.cancel_order(trade_id)
        logger.info(f"Trade {trade_id} canceled successfully.")
        return jsonify({'message': f'Trade {trade_id} canceled successfully'}), 200
    except Exception as e:
        logger.error(f"Error canceling trade {trade_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/port', methods=['GET'])
def port_route():
    """Return portfolio data in JSON format."""
    port = get_portfolio()

    if not port:
        return jsonify({"error": "Failed to retrieve portfolio data."}), 500

    # Prepare data for the pie chart visualization
    portfolio_data = {
        "values": [pos['market_value'] for pos in port.get('positions', [])],
        "labels": [pos['symbol'] for pos in port.get('positions', [])]
    }

    # Prepare portfolio details like buying power, cash, and equity
    portfolio_details = {
        "buying_power": port['buying_power'],
        "cash": port['cash'],
        "equity": port['equity'],
    }

    # Return data in JSON format to be consumed by the frontend
    return jsonify({
        "portfolio_data": portfolio_data,
        "portfolio_details": portfolio_details
    })

@app.route('/options_equities', methods=['GET', 'POST'])
def options_equities_route():
    return render_template('options_equities.html')

@app.route('/extra_information', methods=['GET', 'POST'])
def extra_information():
    """Render the extra information page."""
    return render_template('extra_information.html')

@app.route('/test_account', methods=['GET'])
def test_account_route():
    """Test route to fetch and display account information."""
    try:
        account = get_account()
        if account:
            account_info = {
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power)
            }
            return jsonify({"account": account_info}), 200
        else:
            return jsonify({"error": "Failed to fetch account information."}), 500
    except Exception as e:
        logger.error(f"Error accessing account: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/market_summary', methods=['GET'])
def market_summary_route():
    """
    Route to display the market summary and forecasting output.
    """
    try:
        # Get stock symbol and period from query parameters
        stock_symbol = request.args.get('symbol', 'AAPL')  # Default to AAPL
        period = request.args.get('period', '1mo')

        # Get stock data
        stock_data = yf.Ticker(stock_symbol).history(period=period)

        if stock_data.empty:
            return jsonify({"error": "No data available for the specified symbol and period."}), 400

        # Prepare market data for summary
        market_data_text = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string()

        # Generate market summary using OpenAI API
        summary = generate_market_summary(market_data_text)

        # Forecast using your model
        forecasted_prices, model, y_test, X_test, y_train = forecast_stock(stock_data)

        # Prepare data for display
        forecasted_days = [f'Day {i+1}' for i in range(len(forecasted_prices))]
        forecasted_values = forecasted_prices

        # Render the template with summary and forecast
        return render_template('market_summary.html', 
                               summary=summary, 
                               symbol=stock_symbol, 
                               forecasted_days=forecasted_days, 
                               forecasted_values=forecasted_values)
    except Exception as e:
        logger.error(f"Error in market_summary_route: {e}")
        return jsonify({"error": "An error occurred while generating the market summary."}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
