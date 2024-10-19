import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from strategies.monte_carlo import monte_carlo_option_price  # Assuming this is your Monte Carlo method
from strategies.options_pricing import cox_ross_rubinstein, black_scholes  # Ensure you have these functions

def plot_equity_recommendations(data):
    """
    Visualize equity price and recommendations.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot close price
    plt.plot(data['Close'], label='Close Price', color='blue')
    
    # Highlight Buy signals
    buy_signals = data[data['Recommendation'] == 'Buy']
    plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', s=100)

    # Highlight Sell signals
    sell_signals = data[data['Recommendation'] == 'Sell']
    plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', s=100)

    plt.title('Equity Price with Recommendations')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def fetch_option_data(stock_symbol, expiry_date):
    """
    Fetch option data using Yahoo Finance for a specific stock and expiry date.
    """
    options = yf.Ticker(stock_symbol).options
    option_data = {}
    
    if expiry_date in options:
        option_chain = yf.Ticker(stock_symbol).option_chain(expiry_date)
        option_data['calls'] = option_chain.calls
        option_data['puts'] = option_chain.puts
    
    return option_data

def compare_option_pricing(stock_symbol, strike_price, expiry_date, current_price, volatility, risk_free_rate):
    """
    Compare option pricing methods: CRR, BS, and Monte Carlo.
    """
    # Set parameters
    option_data = fetch_option_data(stock_symbol, expiry_date)
    if not option_data:
        print(f"No option data available for {stock_symbol} on {expiry_date}.")
        return

    # Parameters for option pricing
    time_to_expiry = (pd.to_datetime(expiry_date) - pd.to_datetime('today')).days / 365.0
    call_option_price_bs = black_scholes(current_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type='call')
    call_option_price_crr = cox_ross_rubinstein(current_price, strike_price, time_to_expiry, risk_free_rate, volatility, steps=100, option_type='call')
    call_option_price_mc = monte_carlo_option_price(current_price, strike_price, time_to_expiry, risk_free_rate, volatility, num_simulations=10000, option_type='call')

    # Visualize results
    methods = ['Black-Scholes', 'CRR', 'Monte Carlo']
    prices = [call_option_price_bs, call_option_price_crr, call_option_price_mc]

    plt.figure(figsize=(10, 5))
    plt.bar(methods, prices, color=['blue', 'orange', 'green'])
    plt.title(f'Option Pricing Comparison for {stock_symbol} - Call Option')
    plt.ylabel('Option Price')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # Example usage
    stock_symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Assume `data` is a DataFrame returned from your previous analysis script
    data = fetch_data(stock_symbol, start_date, end_date)  # Make sure this function is defined
    data = calculate_indicators(data)  # Make sure this function is defined
    data = generate_recommendations(data)  # Make sure this function is defined

    # Plot equity recommendations
    plot_equity_recommendations(data)

    # Option pricing comparison
    strike_price = 150  # Set a hypothetical strike price
    expiry_date = '2024-01-19'  # Set a hypothetical expiry date
    current_price = data['Close'].iloc[-1]
    volatility = 0.2  # Hypothetical volatility
    risk_free_rate = 0.01  # Hypothetical risk-free rate

    compare_option_pricing(stock_symbol, strike_price, expiry_date, current_price, volatility, risk_free_rate)
