import numpy as np
import cvxpy as cp
import yfinance as yf

def l1_trend_filter(y, lambd):
    n = len(y)
    x = cp.Variable(n)
    
    # Define second differences
    D = np.diff(np.eye(n), n=2, axis=0)
    
    print("Shape of y:", y.shape)
    print("Shape of x:", x.shape)
    print("Shape of D:", D.shape)
    
    # Define the L1 trend filtering problem
    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lambd * cp.norm(D @ x, 1))
    problem = cp.Problem(objective)
    problem.solve()
    
    return x.value

# Usage example with stock data
historical_data = yf.Ticker('AAPL').history(start='2016-01-01', end='2024-10-01')
stock_prices = historical_data['Close'].values

print("Shape of stock_prices:", stock_prices.shape)

# Apply L1 trend filtering
trend = l1_trend_filter(stock_prices, lambd=20)

# Plotting stock prices and trend
import matplotlib.pyplot as plt
plt.plot(stock_prices, label='Stock Price')
plt.plot(trend, label='L1 Trend Filter', color='red')
plt.legend()
plt.show()

def second_order_trend_filter(y, lambd):
    """
    Second-order trend filtering based on the principles outlined in the trend filtering paper.
    
    :param y: The input time series (e.g., stock prices).
    :param lambd: The regularization parameter (controls smoothness).
    :return: The filtered signal (trend).
    """
    n = len(y)
    x = cp.Variable(n)

    # Second-order difference matrix
    D = np.diff(np.eye(n), n=2, axis=0)

    # Define the second-order L1 trend filtering problem
    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lambd * cp.norm(D @ x, 1))
    problem = cp.Problem(objective)
    problem.solve()

    return x.value



# Apply the second-order trend filter
trend_second_order = second_order_trend_filter(stock_prices, lambd=20)

# Plotting the results
plt.plot(stock_prices, label='Stock Price')
plt.plot(trend_second_order, label='Second-Order Trend Filter', color='red')
plt.legend()
plt.show()


# Threshold for buy/sell signals based on deviation from trend
threshold = 5  # Define a threshold value (e.g., 5 USD deviation)

# Buy and sell signals based on the stock price crossing the trend
buy_signals = (stock_prices < (trend_second_order - threshold))  # Price is below trend by threshold
sell_signals = (stock_prices > (trend_second_order + threshold))  # Price is above trend by threshold

# Generating signal dates
buy_dates = historical_data.index[buy_signals]
sell_dates = historical_data.index[sell_signals]

# Plotting buy and sell signals
plt.plot(stock_prices, label='Stock Price')
plt.plot(trend_second_order, label='Second-Order Trend Filter', color='red')
plt.scatter(buy_dates, stock_prices[buy_signals], label='Buy Signal', marker='^', color='green')
plt.scatter(sell_dates, stock_prices[sell_signals], label='Sell Signal', marker='v', color='red')
plt.legend()
plt.show()
