from bot import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Fetch stock data
def get_stock_data(stock_symbol, period='TIME_SERIES_DAILY', output_size='full'):
    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'
    url = f'https://www.alphavantage.co/query?function={period}&symbol={stock_symbol}&outputsize={output_size}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    time_series = data['Time Series (Daily)']
    
    dates = list(time_series.keys())[:365]
    prices = [float(time_series[date]['4. close']) for date in dates]
    
    return dates, prices

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
    stock_symbol = 'AAPL'  # Default symbol
    return render_template('index.html', stock_symbol=stock_symbol)

@app.route('/options', methods=['GET', 'POST'])
def options():
    stock_symbol = 'AAPL'  # Default symbol
    options_data = get_options_data(stock_symbol)
    return render_template('options.html', options_data=options_data, stock_symbol=stock_symbol)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    stock_symbol = 'AAPL'
    recommendation = generate_trade_recommendation(stock_symbol)
    return render_template('recommendation.html', recommendation=recommendation, stock_symbol=stock_symbol)

if __name__ == '__main__':
    app.run(debug=True)
