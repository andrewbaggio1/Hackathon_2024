# strategies/strategy_methods.py

import os
import requests
import pandas as pd
import yfinance as yf

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
