import pandas as pd
import numpy as np
import yfinance as yf
import ta  # You may need to install this package using pip install ta

def fetch_data(stock_symbol, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

def calculate_indicators(data):
    """
    Calculate technical indicators.
    """
    # Calculate moving averages
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)

    # Calculate Bollinger Bands
    data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'], window=20, window_dev=2)
    data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'], window=20, window_dev=2)

    # Calculate RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

    # Calculate Stochastic Oscillator
    data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)

    # Calculate MACD
    data['MACD'] = ta.trend.macd(data['Close'])
    data['MACD_signal'] = ta.trend.macd_signal(data['Close'])

    # Calculate ATR for volatility
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

    return data

def generate_recommendations(data):
    """
    Generate buy/sell recommendations based on technical indicators.
    """
    recommendations = []

    for i in range(len(data)):
        if np.isnan(data['SMA_20'][i]) or np.isnan(data['SMA_50'][i]) or np.isnan(data['RSI'][i]):
            recommendations.append('Hold')
            continue

        # Buy signal based on moving averages
        if data['SMA_20'][i] > data['SMA_50'][i] and data['RSI'][i] < 30:  # RSI indicates oversold
            recommendations.append('Buy')
        # Sell signal based on moving averages
        elif data['SMA_20'][i] < data['SMA_50'][i] and data['RSI'][i] > 70:  # RSI indicates overbought
            recommendations.append('Sell')
        # Check Bollinger Bands
        elif data['Close'][i] < data['Bollinger_Low'][i]:
            recommendations.append('Buy')
        elif data['Close'][i] > data['Bollinger_High'][i]:
            recommendations.append('Sell')
        # Stochastic Oscillator signals
        elif data['Stoch_K'][i] < 20 and data['Stoch_D'][i] < 20:  # Oversold condition
            recommendations.append('Buy')
        elif data['Stoch_K'][i] > 80 and data['Stoch_D'][i] > 80:  # Overbought condition
            recommendations.append('Sell')
        else:
            recommendations.append('Hold')

    data['Recommendation'] = recommendations
    return data

def main(stock_symbol, start_date, end_date):
    # Fetch the stock data
    data = fetch_data(stock_symbol, start_date, end_date)
    
    # Calculate technical indicators
    data = calculate_indicators(data)
    
    # Generate recommendations
    data = generate_recommendations(data)
    
    # Display the last few rows of the dataframe
    print(data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'Stoch_K', 'Stoch_D', 'Recommendation']].tail(10))

if __name__ == "__main__":
    # Example usage
    stock_symbol = 'AAPL'  # Change to your desired stock symbol
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    main(stock_symbol, start_date, end_date)
