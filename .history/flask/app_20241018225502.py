from flask import Flask, render_template
import requests
import plotly.graph_objs as go
import plotly.io as pio
import json

app = Flask(__name__)

# Function to get stock data
def get_stock_data(stock_symbol):
    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your API key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    time_series = data['Time Series (Daily)']
    
    # Extracting recent prices (for example, last 7 days)
    dates = list(time_series.keys())[:7]
    prices = [float(time_series[date]['4. close']) for date in dates]
    
    return dates, prices

# Function to create a stock price plot
def create_stock_plot(dates, prices, stock_symbol):
    fig = go.Figure(data=go.Scatter(x=dates, y=prices, mode='lines+markers'))
    fig.update_layout(title=f'{stock_symbol} Stock Price',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)')
    return pio.to_html(fig, full_html=False)

@app.route('/')
def home():
    stock_symbol = 'AAPL'  # Example: Apple stock symbol
    dates, prices = get_stock_data(stock_symbol)
    stock_plot_html = create_stock_plot(dates, prices, stock_symbol)
    
    # Example of a simple portfolio summary (you can replace with real data)
    portfolio = {
        'AAPL': {'shares': 10, 'price': 150},
        'GOOGL': {'shares': 5, 'price': 2800}
    }
    
    total_value = sum([stock['shares'] * stock['price'] for stock in portfolio.values()])
    
    return render_template('index.html', stock_plot=stock_plot_html, portfolio