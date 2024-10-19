import numpy as np
import cvxpy as cp
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
def generate_mean_reversion_signals(stock_prices, trend):
    signals = np.zeros(len(stock_prices))  # Initialize an array of 0s for signals

    for i in range(1, len(stock_prices)):
        # Buy signal with threshold
        threshold=.01
        if stock_prices[i] < trend[i] and stock_prices[i-1] >= trend[i-1] and (trend[i] - stock_prices[i]) > threshold:
          signals[i] = 1  # Buy signal

# Sell signal with threshold
        elif stock_prices[i] > trend[i] and stock_prices[i-1] <= trend[i-1] and (stock_prices[i] - trend[i]) > threshold:
          signals[i] = -1  # Sell signal


    return signals


def backtest_mean_reversion(stock_prices, signals, initial_capital=10000, shares_per_trade=10):
    """
    Backtest the mean reversion strategy based on generated signals.
    
    :param stock_prices: Array of stock prices.
    :param signals: Array of trading signals (-1: Sell, 1: Buy, 0: Hold).
    :param initial_capital: Starting capital for the backtest.
    :param shares_per_trade: Number of shares to buy/sell per trade.
    :return: Time series of portfolio values.
    """
    capital = initial_capital
    num_shares = 0
    portfolio_values = []
    
    for i in range(len(signals)):
        # Buy signal
        if signals[i] == 1 and capital >= stock_prices[i] * shares_per_trade:
            # Buy shares
            num_shares += shares_per_trade
            capital -= stock_prices[i] * shares_per_trade
            print(f"Buy {shares_per_trade} shares at {stock_prices[i]} on day {i}")

        # Sell signal
        elif signals[i] == -1 and num_shares >= shares_per_trade:
            # Sell shares
            num_shares -= shares_per_trade
            capital += stock_prices[i] * shares_per_trade
            print(f"Sell {shares_per_trade} shares at {stock_prices[i]} on day {i}")
        
        # Calculate portfolio value (cash + value of shares)
        portfolio_value = capital + num_shares * stock_prices[i]
        portfolio_values.append(portfolio_value)
    
    return portfolio_values


# Implement Buy-and-Hold Strategy
def buy_and_hold(stock_prices, initial_capital=10000):
    """
    Simulate a buy-and-hold strategy where you buy at the start and hold until the end.
    
    :param stock_prices: Array of stock prices.
    :param initial_capital: Starting capital for the backtest.
    :return: Time series of portfolio values based on buy-and-hold.
    """
    shares_bought = initial_capital // stock_prices[0]  # Buy as many shares as possible with initial capital
    remaining_cash = initial_capital - (shares_bought * stock_prices[0])  # Leftover cash after buying shares

    portfolio_values = [remaining_cash + shares_bought * price for price in stock_prices]
    
    return portfolio_values
def moving_average_crossover(stock_prices, short_window, long_window):
    signals = np.zeros(len(stock_prices))

    short_ma = pd.Series(stock_prices).rolling(window=short_window).mean()
    long_ma = pd.Series(stock_prices).rolling(window=long_window).mean()

    for i in range(1, len(stock_prices)):
        if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
            signals[i] = 1  # Buy signal
        elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
            signals[i] = -1  # Sell signal

    return signals, short_ma, long_ma
def backtest_moving_average_crossover(stock_prices, signals, initial_capital=10000):
    position = 0  # 0 means no position, 1 means holding a position
    buy_price = 0
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = []

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy signal
            if position == 0:  # Only buy if not already in position
                position = cash / stock_prices[i]  # Buy as many shares as possible
                buy_price = stock_prices[i]
                cash = 0  # All cash is used to buy the stock
                positions.append(('Buy', i, stock_prices[i]))
        elif signals[i] == -1:  # Sell signal
            if position > 0:  # Only sell if holding a position
                cash = position * stock_prices[i]  # Sell all the shares
                position = 0  # No more position
                positions.append(('Sell', i, stock_prices[i]))
        
        # Update the portfolio value at each step
        portfolio_value.append(cash + position * stock_prices[i])

    # Final portfolio value
    final_value = cash + position * stock_prices[-1]
    return portfolio_value, positions, final_value

import ta  

def rsi_strategy(stock_prices, rsi_period=14, buy_threshold=30, sell_threshold=70):
    signals = np.zeros(len(stock_prices))

    rsi = ta.momentum.RSIIndicator(pd.Series(stock_prices), window=rsi_period).rsi()

    for i in range(1, len(stock_prices)):
        if rsi[i] < buy_threshold and rsi[i-1] >= buy_threshold:
            signals[i] = 1  # Buy signal
        elif rsi[i] > sell_threshold and rsi[i-1] <= sell_threshold:
            signals[i] = -1  # Sell signal

    return signals, rsi

# Backtesting function for RSI Strategy
def backtest_rsi_strategy(stock_prices, signals, initial_capital=10000):
    position = 0  # 0 means no position, 1 means holding a position
    buy_price = 0
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = []

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy signal
            if position == 0:  # Only buy if not already in position
                position = cash / stock_prices[i]  # Buy as many shares as possible
                buy_price = stock_prices[i]
                cash = 0  # All cash is used to buy the stock
                positions.append(('Buy', i, stock_prices[i]))
        elif signals[i] == -1:  # Sell signal
            if position > 0:  # Only sell if holding a position
                cash = position * stock_prices[i]  # Sell all the shares
                position = 0  # No more position
                positions.append(('Sell', i, stock_prices[i]))
        
        # Update the portfolio value at each step
        portfolio_value.append(cash + position * stock_prices[i])

    # Final portfolio value
    final_value = cash + position * stock_prices[-1]
    return portfolio_value, positions, final_value

def macd_strategy(stock_prices, short_window=12, long_window=26, signal_window=9):
    signals = np.zeros(len(stock_prices))

    short_ma = pd.Series(stock_prices).ewm(span=short_window, adjust=False).mean()
    long_ma = pd.Series(stock_prices).ewm(span=long_window, adjust=False).mean()
    macd = short_ma - long_ma
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()

    for i in range(1, len(stock_prices)):
        if macd[i] > signal_line[i] and macd[i-1] <= signal_line[i-1]:
            signals[i] = 1  # Buy signal
        elif macd[i] < signal_line[i] and macd[i-1] >= signal_line[i-1]:
            signals[i] = -1  # Sell signal

    return signals, macd, signal_line

# Backtesting function for MACD Strategy
def backtest_macd_strategy(stock_prices, signals, initial_capital=10000):
    position = 0  # 0 means no position, 1 means holding a position
    buy_price = 0
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = []

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy signal
            if position == 0:  # Only buy if not already in position
                position = cash / stock_prices[i]  # Buy as many shares as possible
                buy_price = stock_prices[i]
                cash = 0  # All cash is used to buy the stock
                positions.append(('Buy', i, stock_prices[i]))
        elif signals[i] == -1:  # Sell signal
            if position > 0:  # Only sell if holding a position
                cash = position * stock_prices[i]  # Sell all the shares
                position = 0  # No more position
                positions.append(('Sell', i, stock_prices[i]))
        
        # Update the portfolio value at each step
        portfolio_value.append(cash + position * stock_prices[i])

    # Final portfolio value
    final_value = cash + position * stock_prices[-1]
    return portfolio_value, positions, final_value

# Bollinger Bands Strategy Function
def bollinger_bands_strategy(stock_prices, window=20, num_std_dev=2):
    signals = np.zeros(len(stock_prices))

    # Calculate moving average and standard deviation
    rolling_mean = pd.Series(stock_prices).rolling(window=window).mean()
    rolling_std = pd.Series(stock_prices).rolling(window=window).std()

    # Calculate upper and lower Bollinger Bands
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    # Generate buy/sell signals based on Bollinger Band strategy
    for i in range(1, len(stock_prices)):
        if stock_prices[i] < lower_band[i] and stock_prices[i-1] >= lower_band[i-1]:
            signals[i] = 1  # Buy signal
        elif stock_prices[i] > upper_band[i] and stock_prices[i-1] <= upper_band[i-1]:
            signals[i] = -1  # Sell signal

    return signals, upper_band, lower_band, rolling_mean

# Backtesting function for Bollinger Bands Strategy
def backtest_bollinger_bands_strategy(stock_prices, signals, initial_capital=10000):
    position = 0  # 0 means no position, 1 means holding a position
    buy_price = 0
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = []

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy signal
            if position == 0:  # Only buy if not already in position
                position = cash / stock_prices[i]  # Buy as many shares as possible
                buy_price = stock_prices[i]
                cash = 0  # All cash is used to buy the stock
                positions.append(('Buy', i, stock_prices[i]))
        elif signals[i] == -1:  # Sell signal
            if position > 0:  # Only sell if holding a position
                cash = position * stock_prices[i]  # Sell all the shares
                position = 0  # No more position
                positions.append(('Sell', i, stock_prices[i]))
        
        # Update the portfolio value at each step
        portfolio_value.append(cash + position * stock_prices[i])

    # Final portfolio value
    final_value = cash + position * stock_prices[-1]
    return portfolio_value, positions, final_value


def bollinger_bands_with_trend_filter(stock_prices, trend, window, num_std_dev):
    signals = np.zeros(len(stock_prices))

    rolling_mean = pd.Series(stock_prices).rolling(window=window).mean()
    rolling_std = pd.Series(stock_prices).rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    for i in range(1, len(stock_prices)):
        # Buy when price falls below the lower band and is below the trend
        if stock_prices[i] < lower_band[i] and stock_prices[i] < trend[i]:
            signals[i] = 1  # Buy signal
        # Sell when price rises above the upper band and is above the trend
        elif stock_prices[i] > upper_band[i] and stock_prices[i] > trend[i]:
            signals[i] = -1  # Sell signal

    return signals, upper_band, lower_band, rolling_mean


# Backtesting function for Bollinger Bands with Trend Filter Strategy
def backtest_bollinger_bands_with_trend_filter(stock_prices, signals, initial_capital=10000):
    position = 0  # 0 means no position, 1 means holding a position
    buy_price = 0
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = []

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy signal
            if position == 0:  # Only buy if not already in position
                position = cash / stock_prices[i]  # Buy as many shares as possible
                buy_price = stock_prices[i]
                cash = 0  # All cash is used to buy the stock
                positions.append(('Buy', i, stock_prices[i]))
        elif signals[i] == -1:  # Sell signal
            if position > 0:  # Only sell if holding a position
                cash = position * stock_prices[i]  # Sell all the shares
                position = 0  # No more position
                positions.append(('Sell', i, stock_prices[i]))
        
        # Update the portfolio value at each step
        portfolio_value.append(cash + position * stock_prices[i])

    # Final portfolio value
    final_value = cash + position * stock_prices[-1]
    return portfolio_value, positions, final_value


# Buy and Hold Backtesting function
def buy_and_hold_backtesting(stock_prices, initial_capital=10000):
    # Buy at the start and hold until the end
    shares = initial_capital / stock_prices[0]  # Buy all shares with initial capital
    portfolio_value = shares * stock_prices  # Portfolio value changes with stock price
    final_value = shares * stock_prices[-1]  # Final portfolio value
    return portfolio_value, final_value

def rsi_with_trend_filter(stock_prices, trend, rsi_period=14, buy_threshold=30, sell_threshold=70):
    signals = np.zeros(len(stock_prices))

    rsi = ta.momentum.RSIIndicator(pd.Series(stock_prices), window=rsi_period).rsi()

    for i in range(1, len(stock_prices)):
        # Buy when RSI crosses below 30 and price is below the trend
        if rsi[i] < buy_threshold and stock_prices[i] < trend[i]:
            signals[i] = 1  # Buy signal
        # Sell when RSI crosses above 70 and price is above the trend
        elif rsi[i] > sell_threshold and stock_prices[i] > trend[i]:
            signals[i] = -1  # Sell signal

    return signals, rsi

# Backtesting function for RSI with Trend Filter Strategy
def backtest_rsi_with_trend_filter(stock_prices, signals, initial_capital=10000):
    position = 0  # 0 means no position, 1 means holding a position
    buy_price = 0
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = []

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy signal
            if position == 0:  # Only buy if not already in position
                position = cash / stock_prices[i]  # Buy as many shares as possible
                buy_price = stock_prices[i]
                cash = 0  # All cash is used to buy the stock
                positions.append(('Buy', i, stock_prices[i]))
        elif signals[i] == -1:  # Sell signal
            if position > 0:  # Only sell if holding a position
                cash = position * stock_prices[i]  # Sell all the shares
                position = 0  # No more position
                positions.append(('Sell', i, stock_prices[i]))
        
        # Update the portfolio value at each step
        portfolio_value.append(cash + position * stock_prices[i])

    # Final portfolio value
    final_value = cash + position * stock_prices[-1]
    return portfolio_value, positions, final_value


def macd_with_trend_filter(stock_prices, trend, short_window=12, long_window=26, signal_window=9):
    signals = np.zeros(len(stock_prices))

    short_ma = pd.Series(stock_prices).ewm(span=short_window, adjust=False).mean()
    long_ma = pd.Series(stock_prices).ewm(span=long_window, adjust=False).mean()
    macd = short_ma - long_ma
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()

    for i in range(1, len(stock_prices)):
        # Buy when MACD crosses above signal line and price is below trend
        if macd[i] > signal_line[i] and macd[i-1] <= signal_line[i-1] and stock_prices[i] < trend[i]:
            signals[i] = 1  # Buy signal
        # Sell when MACD crosses below signal line and price is above trend
        elif macd[i] < signal_line[i] and macd[i-1] >= signal_line[i-1] and stock_prices[i] > trend[i]:
            signals[i] = -1  # Sell signal

    return signals, macd, signal_line

# Backtesting function for MACD with Trend Filter Strategy
def backtest_macd_with_trend_filter(stock_prices, signals, initial_capital=10000):
    position = 0  # 0 means no position, 1 means holding a position
    buy_price = 0
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = []

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy signal
            if position == 0:  # Only buy if not already in position
                position = cash / stock_prices[i]  # Buy as many shares as possible
                buy_price = stock_prices[i]
                cash = 0  # All cash is used to buy the stock
                positions.append(('Buy', i, stock_prices[i]))
        elif signals[i] == -1:  # Sell signal
            if position > 0:  # Only sell if holding a position
                cash = position * stock_prices[i]  # Sell all the shares
                position = 0  # No more position
                positions.append(('Sell', i, stock_prices[i]))
        
        # Update the portfolio value at each step
        portfolio_value.append(cash + position * stock_prices[i])

    # Final portfolio value
    final_value = cash + position * stock_prices[-1]
    return portfolio_value, positions, final_value


def calculate_risk_metrics(portfolio_value, initial_capital=10000, risk_free_rate=0.01):
    """
    Calculate risk metrics for a given portfolio value time series.
    
    :param portfolio_value: List or array of portfolio values over time.
    :param initial_capital: Initial capital used in the strategy (default is 10,000).
    :param risk_free_rate: Annual risk-free rate (default is 1%).
    :return: A dictionary containing the risk metrics.
    """
    # Calculate returns
    returns = np.diff(portfolio_value) / portfolio_value[:-1]  # Daily returns
    
    # Calculate Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)  # Annualizing assuming 252 trading days
    
    # Calculate Maximum Drawdown
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calculate Sharpe Ratio
    excess_returns = returns - (risk_free_rate / 252)
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    # Calculate Sortino Ratio (focus on downside risk)
    downside_returns = returns[returns < 0]
    sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    return {
        "Volatility (Annualized)": volatility,
        "Maximum Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio
    }


def compare_strategies_risk_metrics(strategy_portfolios, initial_capital=10000):
    """
    Calculate and compare risk metrics across multiple strategies.

    :param strategy_portfolios: Dictionary of strategy names and their respective portfolio values.
    :param initial_capital: Initial capital used in the strategy (default is 10,000).
    :return: A dictionary containing the risk metrics for each strategy.
    """
    metrics = {}
    for strategy_name, portfolio_value in strategy_portfolios.items():
        metrics[strategy_name] = calculate_risk_metrics(portfolio_value, initial_capital)
    
    return metrics
