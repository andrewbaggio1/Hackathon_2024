from flask import Flask, render_template, request
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os

app = Flask(__name__)

# Fetch stock data
def get_stock_data(stock_symbol):
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

# Get options data
def get_options_data(stock_symbol):
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

# Moving averages strategy
def calculate_moving_average(prices, window):
    return prices.rolling(window=window).mean()

def generate_trade_recommendation(stock_symbol):
    dates, prices = get_stock_data(stock_symbol)
    if prices is None:
        return "Error fetching data."
    
    prices_series = pd.Series(prices)
    
    short_ma = calculate_moving_average(prices_series, window=20)
    long_ma = calculate_moving_average(prices_series, window=50)
    
    if short_ma.iloc[-1] > long_ma.iloc[-1]:
        return 'Buy'
    elif short_ma.iloc[-1] < long_ma.iloc[-1]:
        return 'Sell'
    else:
        return 'Hold'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()
    else:
        stock_symbol = 'AAPL'  # Default symbol
    
    dates, prices = get_stock_data(stock_symbol)
    if prices is None:
        return render_template('index.html', error="Could not fetch data.", stock_symbol=stock_symbol)

    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
    
    # Plot moving averages
    prices_series = pd.Series(prices)
    short_ma = calculate_moving_average(prices_series, window=20)
    long_ma = calculate_moving_average(prices_series, window=50)
    
    fig.add_trace(go.Scatter(x=dates[19:], y=short_ma[19:], mode='lines', name='20-Day MA'))
    fig.add_trace(go.Scatter(x=dates[49:], y=long_ma[49:], mode='lines', name='50-Day MA'))

    graphJSON = pio.to_json(fig)

    return render_template('index.html', stock_symbol=stock_symbol, graphJSON=graphJSON)

@app.route('/options', methods=['GET', 'POST'])
def options():
    stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()  # Use user input
    options_data = get_options_data(stock_symbol)
    return render_template('options.html', options_data=options_data, stock_symbol=stock_symbol)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()  # Use user input
    recommendation = generate_trade_recommendation(stock_symbol)
    return render_template('recommendation.html', recommendation=recommendation, stock_symbol=stock_symbol)

if __name__ == '__main__':
    app.run(debug=True)
