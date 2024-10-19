from flask import Flask, render_template, request
import requests
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np

app = Flask(__name__)

# Fetch stock data
def get_stock_data(stock_symbol, period='TIME_SERIES_DAILY', output_size='full'):
    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'
    url = f'https://www.alphavantage.co/query?function={period}&symbol={stock_symbol}&outputsize={output_size}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    time_series = data['Time Series (Daily)']
    
    # Extract prices for a larger historical period
    dates = list(time_series.keys())[:365]  # Example: last 365 days
    prices = [float(time_series[date]['4. close']) for date in dates]
    
    return dates, prices

# Create a stock price plot
def create_stock_plot(dates, prices, stock_symbol):
    fig = go.Figure(data=go.Scatter(x=dates, y=prices, mode='lines+markers'))
    fig.update_layout(title=f'{stock_symbol} Stock Price',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)')
    return pio.to_html(fig, full_html=False)

# Monte Carlo simulation function
def monte_carlo_simulation(stock_symbol, days=30, num_simulations=1000):
    dates, prices = get_stock_data(stock_symbol)
    returns = np.diff(prices) / prices[:-1]  # Calculate daily returns
    
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    last_price = prices[-1]
    
    simulations = np.zeros((days, num_simulations))
    for i in range(num_simulations):
        price_path = [last_price]
        for _ in range(days):
            next_price = price_path[-1] * (1 + np.random.normal(mean_return, std_dev))
            price_path.append(next_price)
        simulations[:, i] = price_path[1:]
    
    return simulations

# Create Monte Carlo plot
def create_monte_carlo_plot(simulations, stock_symbol):
    fig = go.Figure()
    for i in range(simulations.shape[1]):
        fig.add_trace(go.Scatter(x=list(range(1, simulations.shape[0] + 1)), 
                                 y=simulations[:, i], mode='lines', 
                                 line=dict(color='rgba(0,0,255,0.1)')))
    fig.update_layout(title=f'Monte Carlo Simulation for {stock_symbol} (Next 30 Days)',
                      xaxis_title='Days',
                      yaxis_title='Price (USD)')
    return pio.to_html(fig, full_html=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    stock_symbol = 'AAPL'  # Default stock symbol
    
    # Toggle visuals based on user input
    selected_visual = request.form.get('visual') if request.method == 'POST' else 'price_chart'
    
    if selected_visual == 'price_chart':
        dates, prices = get_stock_data(stock_symbol)
        stock_plot_html = create_stock_plot(dates, prices, stock_symbol)
    elif selected_visual == 'monte_carlo':
        simulations = monte_carlo_simulation(stock_symbol)
        stock_plot_html = create_monte_carlo_plot(simulations, stock_symbol)
    
    return render_template('index.html', stock_plot=stock_plot_html, selected_visual=selected_visual)

if __name__ == '__main__':
    app.run(debug=True)
