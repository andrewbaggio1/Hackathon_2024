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