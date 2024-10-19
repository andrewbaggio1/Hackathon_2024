import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import CubicSpline
import sympy as sp
import time
from scipy.interpolate import interp1d

def ZCB_return(R):
    return 1/R

def ZCB_func(T):
    pass

def expected_coupon(t, T):
    return ZCB_func(T)/ZCB_func(t)

def fwd_p(R, S_0):
    return R*S_0

def riskless_APR(T, r):
    return np.exp(r*T)

def riskless_air(R, T):
    return np.log(R)/T

def present_value_discrete(X, ZCBs):
    return np.dot(X, ZCBs)

def expected_value(S_T, P_S_T):
    # assert arrays
    return np.dot(S_T, P_S_T)

##############################
# IMPLEMENT VARIANCE IN S(T) #
##############################

def fwd_payout(S, K, long = False):
    if long:
        return S - K
    else:
        return K - S
    
def euro_opt_payout(S, K, call=True):
    """[S(T) - K]^+ for a call, [K - S(T)]^+ for a put."""
    if call:
        return max(0, S - K)
    else: 
        return max(0, K - S)

def calculate_mean(data):
    """Calculate the mean of a dataset."""
    return sum(data) / len(data)

def calculate_variance(data, sample=True):
    """Calculate the variance of a dataset."""
    mean = calculate_mean(data)
    n = len(data)
    squared_diffs = [(x - mean)**2 for x in data]
    if sample:
        return sum(squared_diffs) / (n - 1)
    else:
        return sum(squared_diffs) / n

def hedge_port(H, A):
    return np.dot(H, A)

def call_put_parity(C_0, P_0, S_0, K, R):
    return C_0 - P_0 == S_0 - K / R

#########################
# Black-Scholes Formula #
#########################

def black_scholes(S, K, T, r, sigma, call=True):
    """
    Black-Scholes formula for European call or put option.
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the stock
    call: True for call option, False for put option
    
    Returns:
    Price of the option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if call:
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
    
    return price

#######################
# Greeks Calculations #
#######################

def delta(S, K, T, r, sigma, call=True):
    """
    Calculate the Delta of an option.
    Delta measures the sensitivity of the option price to changes in the underlying asset's price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if call:
        return si.norm.cdf(d1)
    else:
        return si.norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    """
    Calculate the Gamma of an option.
    Gamma measures the rate of change of Delta with respect to changes in the underlying price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """
    Calculate the Vega of an option.
    Vega measures the sensitivity of the option price to changes in volatility.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * si.norm.pdf(d1) * np.sqrt(T)

def theta(S, K, T, r, sigma, call=True):
    """
    Calculate the Theta of an option.
    Theta measures the sensitivity of the option price to the passage of time.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if call:
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    else:
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2))
    
    return theta  # Return per day theta

def rho(S, K, T, r, sigma, call=True):
    """
    Calculate the Rho of an option.
    Rho measures the sensitivity of the option price to changes in the risk-free interest rate.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if call:
        return K * T * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * si.norm.cdf(-d2)

##############################
# Test example with values   #
##############################

# Parameters
# S = 100  # Current stock price
# K = 100  # Strike price
# T = 1    # Time to maturity (1 year)
# r = 0.05  # Risk-free rate (5%)
# sigma = 0.2  # Volatility (20%)

# Calculate option prices and Greeks
# call_price = black_scholes(S, K, T, r, sigma, call=True)
# put_price = black_scholes(S, K, T, r, sigma, call=False)

# print(f"Call Price: {call_price}")
# print(f"Put Price: {put_price}")
# print(f"Delta (Call): {delta(S, K, T, r, sigma, call=True)}")
# print(f"Gamma: {gamma(S, K, T, r, sigma)}")
# print(f"Vega: {vega(S, K, T, r, sigma)}")
# print(f"Theta (Call): {theta(S, K, T, r, sigma, call=True)}")
# print(f"Rho (Call): {rho(S, K, T, r, sigma, call=True)}")

######################
# Least Squares Reg. #
######################

def least_squares_fit(x_data, y_data):
    """
    Find the least-squares best fit coefficients for the model:
    f(x) = p1 + p2 * exp(x) + p3 * exp(-x)
    
    Parameters:
    x_data (array-like): The x-values of the data points
    y_data (array-like): The y-values of the data points
    
    Returns:
    tuple: Coefficients (p1, p2, p3) that minimize the least-squares error
    """
    # Set up the matrix A where each row corresponds to [1, exp(x), exp(-x)] for each x in the data
    A = np.column_stack((np.ones_like(x_data), np.exp(x_data), np.exp(-x_data)))

    # Solve the least-squares problem: A @ p = y_data
    p, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)

    # Return the coefficients p1, p2, p3
    return p[0], p[1], p[2]

def plot_least_squares_fit(x_data, y_data, p1, p2, p3, num_points=81):
    """
    Plot the least-squares best fit function f(x) = p1 + p2 * exp(x) + p3 * exp(-x)
    along with the data points with improved aesthetics.
    
    Parameters:
    x_data (array-like): The x-values of the data points
    y_data (array-like): The y-values of the data points
    p1 (float): Coefficient for the constant term
    p2 (float): Coefficient for exp(x)
    p3 (float): Coefficient for exp(-x)
    num_points (int): Number of equispaced points for plotting the fitted function
    """
    # Generate equispaced points for plotting the function
    x_plot = np.linspace(min(x_data), max(x_data), num_points)

    # Calculate the corresponding y values for the fitted function f(x)
    y_plot = p1 + p2 * np.exp(x_plot) + p3 * np.exp(-x_plot)

    # Create a figure with a larger size
    plt.figure(figsize=(8, 6))

    # Plot the original data points with larger red markers
    plt.scatter(x_data, y_data, color='darkred', label='Data Points', zorder=5, s=100, edgecolor='black')

    # Plot the fitted function with a thicker line
    plt.plot(x_plot, y_plot, label=r'$f(x) = p_1 + p_2 e^x + p_3 e^{-x}$', color='royalblue', lw=2.5)

    # Add gridlines with a light style
    plt.grid(True, which='both', linestyle='--', lw=0.7, color='gray', alpha=0.7)

    # Add x and y axis labels with larger fonts
    plt.xlabel('x', fontsize=14, labelpad=10)
    plt.ylabel('f(x)', fontsize=14, labelpad=10)

    # Set a title with a larger font size
    plt.title('Least-Squares Best Fit Function', fontsize=16, pad=20)

    # Customize ticks for a cleaner look
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a legend with a larger font size and edge
    plt.legend(fontsize=12, frameon=True, shadow=True)

    # Adjust the axis limits slightly for better spacing
    plt.xlim(min(x_data) - 0.5, max(x_data) + 0.5)
    plt.ylim(min(y_data) - 1, max(y_data) + 1)

    # Apply a function to format y-axis labels as integers
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{int(val)}'))

    # Save the plot to a file (optional)
    # plt.savefig("least_squares_fit.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()

######################
# PREMIUM PREDICTION #
######################

def unweighted_quadratic_regression(strike_prices, premiums, target_strike):
    # Fit a quadratic polynomial (degree 2)
    coeffs = np.polyfit(strike_prices, premiums, 2)
    # Generate the quadratic function
    quadratic_func = np.poly1d(coeffs)
    # Estimate the premium for the target strike price
    estimated_premium = quadratic_func(target_strike)
    return estimated_premium

# def weighted_quadratic_regression(strike_prices, premiums, open_interest, target_strike):
#     # Fit a quadratic polynomial (degree 2) with weights
#     coeffs = np.polyfit(strike_prices, premiums, 2, w=open_interest)
#     # Generate the quadratic function
#     quadratic_func = np.poly1d(coeffs)
#     # Estimate the premium for the target strike price
#     estimated_premium = quadratic_func(target_strike)
#     return estimated_premium

def weighted_quadratic_regression(x, y, w, S0):
    # Define the design matrix F for quadratic regression
    F = np.vstack((np.ones(len(x)), x, x**2)).T  # Shape (n, 3)

    # Create the weight matrix W as a diagonal matrix
    W = np.diag(w)

    # Calculate the regression coefficients p using the weighted least squares formula
    p = np.linalg.inv(F.T @ W @ F) @ (F.T @ W @ y)

    # Estimate the premium at the spot price S0
    estimate = np.array([1, S0, S0**2]) @ p  # F(S0)

    return estimate

def polynomial_interpolation(strike_prices, premiums, target_strike):
    # Fit a polynomial of degree len(strike_prices)-1 (degree 6 here)
    coeffs = np.polyfit(strike_prices, premiums, len(strike_prices)-1)
    # Generate the polynomial function
    poly_func = np.poly1d(coeffs)
    # Estimate the premium for the target strike price
    estimated_premium = poly_func(target_strike)
    return estimated_premium

def spline_interpolation(strike_prices, premiums, target_strike):
    # Create a cubic spline interpolation
    spline_func = CubicSpline(strike_prices, premiums)
    # Estimate the premium for the target strike price
    estimated_premium = spline_func(target_strike)
    return estimated_premium

def calculate_weights(X_T_up, X_T_down, R, S_T_up, S_T_down):
    # Coefficients matrix (A) for the linear system
    A = np.array([[R, S_T_up],
                  [R, S_T_down]])
    
    # Values matrix (b) for the linear system
    b = np.array([X_T_up, X_T_down])
    
    # Solve for h_0 and h_1 using numpy's linear algebra solver
    try:
        weights = np.linalg.solve(A, b)
        h_0, h_1 = weights
        return h_0, h_1
    except np.linalg.LinAlgError as e:
        return str(e)
    
def solve_portfolio_weights(X_T_up, X_T_down, R, S_T_up, S_T_down):
    # Define symbols for the weights
    h_0, h_1 = sp.symbols('h_0 h_1')
    
    # Set up the equations based on the portfolio values
    equation1 = sp.Eq(X_T_up, h_0 * R + h_1 * S_T_up)
    equation2 = sp.Eq(X_T_down, h_0 * R + h_1 * S_T_down)
    
    # Solve the system of equations
    solutions = sp.solve((equation1, equation2), (h_0, h_1))
    
    return solutions

def solve_portfolio_weights_symbolically():
    # Define symbolic variables
    h_0, h_1 = sp.symbols('h_0 h_1')
    X_T_up, X_T_down, R, S_T_up, S_T_down = sp.symbols('X_T_up X_T_down R S_T_up S_T_down')
    
    # Set up the equations based on the portfolio values
    equation1 = sp.Eq(X_T_up, h_0 * R + h_1 * S_T_up)
    equation2 = sp.Eq(X_T_down, h_0 * R + h_1 * S_T_down)
    
    # Solve the system of equations symbolically
    solutions = sp.solve((equation1, equation2), (h_0, h_1))
    
    return solutions

def crr_european_option(S0, K, T, r, sigma, N, option_type="call"):
    """
    Cox-Ross-Rubinstein (CRR) model for pricing European call and put options.

    :param S0: Initial stock price (Spot price)
    :param K: Strike price
    :param T: Time to maturity
    :param r: Risk-free interest rate
    :param sigma: Volatility of the underlying asset
    :param N: Number of time steps
    :param option_type: 'call' for a call option, 'put' for a put option
    :return: Option price at t=0 (C(0) or P(0))
    """
    # Time step
    dt = T / N
    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Step 1: Stock price tree
    stock_prices = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)
    
    # Step 2: Option value at maturity
    option_values = np.zeros((N + 1, N + 1))
    if option_type == "call":
        option_values[:, N] = np.maximum(stock_prices[:, N] - K, 0)
    elif option_type == "put":
        option_values[:, N] = np.maximum(K - stock_prices[:, N], 0)
    
    # Step 3: Backward induction to get the option price at t=0
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
    
    return option_values[0, 0]

def CRReurAD(S0, K, T, r, sigma, N, option_type="call"):
    """
    Cox-Ross-Rubinstein (CRR) model for pricing European call and put options using the Backward Pricing Formula.

    :param S0: Initial stock price (Spot price)
    :param K: Strike price
    :param T: Time to maturity
    :param r: Risk-free interest rate
    :param sigma: Volatility of the underlying asset
    :param N: Number of time steps
    :param option_type: 'call' for a call option, 'put' for a put option
    :return: Option price at t=0 (C(0) or P(0))
    """
    # Time step
    dt = T / N
    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Step 1: Stock price tree
    stock_prices = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)
    
    # Step 2: Option value at maturity
    option_values = np.zeros((N + 1, N + 1))
    if option_type == "call":
        option_values[:, N] = np.maximum(stock_prices[:, N] - K, 0)
    elif option_type == "put":
        option_values[:, N] = np.maximum(K - stock_prices[:, N], 0)
    
    # Step 3: Backward induction to get the option price at t=0
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
    
    return option_values[0, 0]

# def profile_CRR(N_values, S0, K, T, r, sigma, option_type="call"):
#     """
#     Profile the time taken to compute the call and put option prices for different N values.

#     Parameters:
#     N_values: list of time steps (e.g., [10, 100, 1000])
#     S0, K, T, r, sigma: CRR model parameters
#     option_type: 'call' or 'put'

#     Returns:
#     A dictionary of results: {N: (Option_price, time_taken)}
#     """
#     import time
#     results = {}

#     for N in N_values:
#         start_time = time.time()
#         option_price = crr_european_option(S0, K, T, r, sigma, N, option_type)
#         end_time = time.time()
#         time_taken = end_time - start_time
#         results[N] = (option_price, time_taken)

#     return results

# def CRReur(T, S0, K, r, sigma, N):
#     # Calculate parameters for the binomial model
#     dt = T / N  # Time step
#     u = np.exp(sigma * np.sqrt(dt))  # Up factor
#     d = 1 / u  # Down factor
#     p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

#     # Initialize call and put option value matrices
#     call_values = np.zeros((N + 1, N + 1))
#     put_values = np.zeros((N + 1, N + 1))

#     # Fill option values at maturity
#     for j in range(N + 1):
#         asset_price = S0 * (u ** (N - j)) * (d ** j)
#         call_values[j, N] = max(0, asset_price - K)
#         put_values[j, N] = max(0, K - asset_price)

#     # Backward induction to calculate option prices
#     for j in range(N - 1, -1, -1):
#         for i in range(j + 1):
#             call_values[i, j] = np.exp(-r * dt) * (p * call_values[i, j + 1] + (1 - p) * call_values[i + 1, j + 1])
#             put_values[i, j] = np.exp(-r * dt) * (p * put_values[i, j + 1] + (1 - p) * put_values[i + 1, j + 1])

#     return call_values[0, 0], put_values[0, 0]  # Return C(0) and P(0)

# def CRReur(T, S0, K, r, sigma, N):
#     """
#     Compute Call and Put premiums using the Cox-Ross-Rubinstein model.
    
#     Parameters:
#     T : float : Time to maturity
#     S0 : float : Initial stock price
#     K : float : Strike price
#     r : float : Risk-free interest rate
#     sigma : float : Volatility of the underlying asset
#     N : int : Number of time steps

#     Returns:
#     tuple : (call_values, put_values)
#     """
#     dt = T / N  # Time step
#     u = np.exp(sigma * np.sqrt(dt))  # Up factor
#     d = 1 / u  # Down factor
#     p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

#     # Initialize call and put option value matrices
#     call_values = np.zeros((N + 1, N + 1))
#     put_values = np.zeros((N + 1, N + 1))

#     # Fill option values at maturity
#     for j in range(N + 1):
#         asset_price = S0 * (u ** (N - j)) * (d ** j)
#         call_values[j, N] = max(0, asset_price - K)
#         put_values[j, N] = max(0, K - asset_price)

#     # Backward induction to calculate option prices
#     for j in range(N - 1, -1, -1):
#         for i in range(j + 1):
#             call_values[i, j] = np.exp(-r * dt) * (p * call_values[i, j + 1] + (1 - p) * call_values[i + 1, j + 1])
#             put_values[i, j] = np.exp(-r * dt) * (p * put_values[i, j + 1] + (1 - p) * put_values[i + 1, j + 1])

#     return call_values, put_values  # Return the matrices of call and put values

def CRReurT(S0, K, T, r, sigma, N):
    """
    Compute Call and Put premiums using the Cox-Ross-Rubinstein model.
    
    Parameters:
    T : float : Time to maturity
    S0 : float : Initial stock price
    K : float : Strike price
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying asset
    N : int : Number of time steps

    Returns:
    tuple : (call_values, put_values)
    """
    dt = T / N  # Time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize call and put option value matrices
    call_values = np.zeros((N + 1, N + 1))
    put_values = np.zeros((N + 1, N + 1))

    # Fill option values at maturity
    for j in range(N + 1):
        asset_price = S0 * (u ** (N - j)) * (d ** j)
        call_values[j, N] = max(0, asset_price - K)
        put_values[j, N] = max(0, K - asset_price)

    # Backward induction to calculate option prices
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            call_values[i, j] = np.exp(-r * dt) * (p * call_values[i, j + 1] + (1 - p) * call_values[i + 1, j + 1])
            put_values[i, j] = np.exp(-r * dt) * (p * put_values[i, j + 1] + (1 - p) * put_values[i + 1, j + 1])

    return call_values, put_values  # Return the matrices of call and put values

def profile_CRR(N_values, S0, K, T, r, sigma):
    """
    Profile the time taken to compute the call and put option prices for different N values.

    Parameters:
    N_values (list): List of time steps (e.g., [10, 100, 1000])
    S0 (float): Initial spot price of the underlying asset.
    K (float): Strike price of the options.
    T (float): Time to expiry of the options in years.
    r (float): Risk-free interest rate (annualized).
    sigma (float): Volatility of the underlying asset (annualized).

    Returns:
    dict: A dictionary of results: {N: (call_price, put_price, time_taken)}
    """
    results = {}

    for N in N_values:
        start_time = time.time()
        call_price, put_price = CRReur(S0, K, T, r, sigma, N)
        end_time = time.time()
        time_taken = end_time - start_time
        results[N] = (call_price, put_price, time_taken)

    return results

def CRR_Arrow_Debreu(S0, K, T, r, sigma, N, option_type='call'):
    """
    Computes the price of a European-style option using the Cox-Ross-Rubinstein
    (CRR) binomial tree model with Arrow-Debreu pricing.

    Parameters:
    S0 (float): Initial spot price of the underlying asset.
    K (float): Strike price of the option.
    T (float): Time to expiry of the option in years.
    r (float): Risk-free interest rate (annualized).
    sigma (float): Volatility of the underlying asset (annualized).
    N (int): Number of time steps in the binomial model.
    option_type (str): Type of the option ('call' for call options, 'put' for put options).
                       Default is 'call'.

    Returns:
    float: The present value of the option at time t=0.
    """
    
    # Calculate the parameters for the binomial tree
    dt = T / N  # Time step size
    u = np.exp(sigma * np.sqrt(dt))  # Up factor for price increase
    d = 1 / u  # Down factor for price decrease
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset prices at maturity
    asset_prices = np.zeros(N + 1)
    for i in range(N + 1):
        asset_prices[i] = S0 * (u ** (N - i)) * (d ** i)  # Calculate price at each node

    # Initialize option values at maturity based on the option type
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)  # Call option payoff
    else:
        option_values = np.maximum(0, K - asset_prices)  # Put option payoff

    # Backward induction to calculate option price at time t=0
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            # Calculate the present value of the option at node (j, i)
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])

    return option_values[0]  # Return the price of the option at time t=0

def CRReur(S0, K, T, r, sigma, N):
    """
    Computes the prices of European-style call and put options using 
    the Cox-Ross-Rubinstein (CRR) model with backward induction.

    Parameters:
    S0 (float): Initial spot price of the underlying asset.
    K (float): Strike price of the options.
    T (float): Time to expiry of the options in years.
    r (float): Risk-free interest rate (annualized).
    sigma (float): Volatility of the underlying asset (annualized).
    N (int): Number of time steps in the binomial model.

    Returns:
    tuple: A tuple containing the call price and put price at time t=0.
    """
    
    # Calculate parameters for the binomial tree
    dt = T / N  # Time step size
    u = np.exp(sigma * np.sqrt(dt))  # Up factor for price increase
    d = 1 / u  # Down factor for price decrease
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset prices and option values
    asset_prices = np.zeros((N + 1, N + 1))  # Asset prices at each node
    call_values = np.zeros((N + 1, N + 1))   # Call option values at each node
    put_values = np.zeros((N + 1, N + 1))    # Put option values at each node

    # Fill in the asset prices at maturity
    for i in range(N + 1):
        asset_prices[i, N] = S0 * (u ** (N - i)) * (d ** i)

    # Calculate option values at maturity
    for i in range(N + 1):
        call_values[i, N] = max(0, asset_prices[i, N] - K)  # Call option payoff
        put_values[i, N] = max(0, K - asset_prices[i, N])   # Put option payoff

    # Backward induction for call option values
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            call_values[i, j] = np.exp(-r * dt) * (p * call_values[i, j + 1] + (1 - p) * call_values[i + 1, j + 1])

    # Backward induction for put option values
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            put_values[i, j] = np.exp(-r * dt) * (p * put_values[i, j + 1] + (1 - p) * put_values[i + 1, j + 1])

    # The option price at time t=0 is found at the top of the trees
    call_price = call_values[0, 0]
    put_price = put_values[0, 0]

    return call_price, put_price  # Return the call and put prices at time t=0

def CRR_Arrow_Debreu_O(N, S0, K, T, r, sigma, option_type='call'):
    """
    Optimized Cox-Ross-Rubinstein (CRR) method using the Arrow-Debreu pricing model.
    
    Parameters:
    N (int): Number of time steps
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    option_type (str): 'call' for call option, 'put' for put option
    
    Returns:
    tuple: (option_price, computation_time)
    """
    start_time = time.time()  # Start timing

    # Calculate the parameters for the binomial tree
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize option values at maturity based on the option type
    option_values = np.zeros(N + 1)
    for i in range(N + 1):
        asset_price_at_maturity = S0 * (u ** (N - i)) * (d ** i)
        option_values[i] = np.maximum(0, asset_price_at_maturity - K) if option_type == 'call' else np.maximum(0, K - asset_price_at_maturity)

    # Backward induction to get option price at t=0
    for j in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])

    computation_time = time.time() - start_time  # End timing
    return option_values[0], computation_time  # Return option price and time taken

def plot_convergence(N_values, S0, K, T, r, sigma):
    """
    Plot the logarithm of the differences between successive Arrow-Debreu prices against log N.
    
    Parameters:
    N_values: List of time steps (e.g., [10, 100, 1000])
    S0, K, T, r, sigma: Parameters for option pricing
    """
    call_prices = []
    put_prices = []
    computation_times = []  # To store computation times if needed

    for N in N_values:
        call_price_ad, time_call = CRR_Arrow_Debreu_O(N, S0, K, T, r, sigma, option_type='call')
        put_price_ad, time_put = CRR_Arrow_Debreu_O(N, S0, K, T, r, sigma, option_type='put')
        call_prices.append(call_price_ad)
        put_prices.append(put_price_ad)
        computation_times.append((time_call, time_put))  # Store computation times

    # Compute the log differences and log N
    log_N = np.log(N_values)
    log_diff_call = np.log(np.abs(np.diff(call_prices)))  # Calculate log of the differences for calls
    log_diff_put = np.log(np.abs(np.diff(put_prices)))    # Calculate log of the differences for puts

    # Polynomial fitting
    p_call = np.polyfit(log_N[1:], log_diff_call, 1)  # Linear fit for call price differences
    p_put = np.polyfit(log_N[1:], log_diff_put, 1)    # Linear fit for put price differences

    # Create polynomial functions
    poly_call = np.poly1d(p_call)
    poly_put = np.poly1d(p_put)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot Call Price Differences
    plt.subplot(1, 2, 1)
    plt.plot(log_N[1:], log_diff_call, marker='o', label='Call Price Differences')
    plt.plot(log_N[1:], poly_call(log_N[1:]), linestyle='--', color='red', 
             label='Poly Fit: $y={:.2f} \cdot x + {:.2f}$'.format(p_call[0], p_call[1]))
    plt.title('Logarithm of Call Price Differences vs Log N')
    plt.xlabel('log(N)')
    plt.ylabel('log(|Call Price Difference|)')
    plt.grid()
    plt.legend()

    # Plot Put Price Differences
    plt.subplot(1, 2, 2)
    plt.plot(log_N[1:], log_diff_put, marker='o', color='orange', label='Put Price Differences')
    plt.plot(log_N[1:], poly_put(log_N[1:]), linestyle='--', color='red', 
             label='Poly Fit: $y={:.2f} \cdot x + {:.2f}$'.format(p_put[0], p_put[1]))
    plt.title('Logarithm of Put Price Differences vs Log N')
    plt.xlabel('log(N)')
    plt.ylabel('log(|Put Price Difference|)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def compute_call_prices(S0_range, K, T, r, sigma, N):
    """
    Compute call option prices for a range of spot prices S0.
    """
    call_prices = []
    for S0 in S0_range:
        call_price = CRR_Arrow_Debreu(S0, K, T, r, sigma, N, option_type='call')
        call_prices.append(call_price)
    return call_prices

# def CRR_AD_O(N, S0, K, T, r, sigma):
#     """
#     Optimized Cox-Ross-Rubinstein (CRR) method using the Arrow-Debreu pricing model.
    
#     Parameters:
#     N (int): Number of time steps
#     S0 (float): Initial stock price
#     K (float): Strike price
#     T (float): Time to maturity
#     r (float): Risk-free interest rate
#     sigma (float): Volatility of the underlying asset
    
#     Returns:
#     dict: Dictionary containing call and put option prices
#     """
#     start_time = time.time()  # Start timing

#     # Calculate the parameters for the binomial tree
#     dt = T / N
#     u = np.exp(sigma * np.sqrt(dt))  # Up factor
#     d = 1 / u  # Down factor
#     p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

#     # Initialize option values at maturity for call and put options
#     call_values = np.zeros(N + 1)
#     put_values = np.zeros(N + 1)
    
#     for i in range(N + 1):
#         asset_price_at_maturity = S0 * (u ** (N - i)) * (d ** i)
#         call_values[i] = np.maximum(0, asset_price_at_maturity - K)
#         put_values[i] = np.maximum(0, K - asset_price_at_maturity)

#     # Backward induction to get option prices at t=0
#     for j in range(N - 1, -1, -1):
#         call_values = np.exp(-r * dt) * (p * call_values[:-1] + (1 - p) * call_values[1:])
#         put_values = np.exp(-r * dt) * (p * put_values[:-1] + (1 - p) * put_values[1:])

#     # Compute the option prices at t=0
#     call_price = call_values[0]
#     put_price = put_values[0]

#     computation_time = time.time() - start_time  # End timing
    
#     return {
#         'call_price': call_price,
#         'put_price': put_price,
#         'computation_time': computation_time
#     }

# def compute_greeks(N, S0, K, T, r, sigma):
#     """
#     Compute Greeks for call and put options using the CRR_AD_O function.

#     Parameters:
#     N (int): Number of time steps
#     S0 (float): Initial stock price
#     K (float): Strike price
#     T (float): Time to maturity
#     r (float): Risk-free interest rate
#     sigma (float): Volatility of the underlying asset
    
#     Returns:
#     dict: Dictionary containing Greeks for call and put options
#     """
#     option_prices = CRR_AD_O(N, S0, K, T, r, sigma)
#     call_price = option_prices['call_price']
#     put_price = option_prices['put_price']
    
#     # Compute Greeks using interpolation method for Delta and Gamma
#     h0 = 2 * S0 * sigma * np.sqrt(T / N)  # critical h
#     u2 = np.exp(2 * sigma * np.sqrt(T / N))  # squared up factor
    
#     # Correctly create the shifted abscissas
#     x = [price - S0 for price in [S0 / u2, S0, S0 * u2]]  # shifted abscissas

#     # Prices for call options at shifted asset prices
#     call_u2 = CRR_AD_O(N, S0 * u2, K, T, r, sigma)['call_price']
#     call_d2 = CRR_AD_O(N, S0 / u2, K, T, r, sigma)['call_price']
    
#     # Fit polynomial and calculate Delta and Gamma for Call
#     yC = [call_d2, call_price, call_u2]  # Call ordinates
#     pC = np.polyfit(x, yC, 2)
#     delta_C = pC[2]
#     gamma_C = 2 * pC[1]

#     # Prices for put options at shifted asset prices
#     put_u2 = CRR_AD_O(N, S0 * u2, K, T, r, sigma)['put_price']
#     put_d2 = CRR_AD_O(N, S0 / u2, K, T, r, sigma)['put_price']
    
#     # Fit polynomial and calculate Delta and Gamma for Put
#     yP = [put_d2, put_price, put_u2]  # Put ordinates
#     pP = np.polyfit(x, yP, 2)
#     delta_P = pP[2]
#     gamma_P = 2 * pP[1]

#     # For other Greeks, use centered difference approximation
#     h = 0.10 * T  # for Theta
#     call_u = CRR_AD_O(N, S0, K, T + h, r, sigma)['call_price']
#     call_d = CRR_AD_O(N, S0, K, T - h, r, sigma)['call_price']
#     theta_C = -(call_u - call_d) / (2 * h)

#     put_u = CRR_AD_O(N, S0, K, T + h, r, sigma)['put_price']
#     put_d = CRR_AD_O(N, S0, K, T - h, r, sigma)['put_price']
#     theta_P = -(put_u - put_d) / (2 * h)

#     h = 0.10 * sigma  # for Vega
#     call_u = CRR_AD_O(N, S0, K, T, r, sigma + h)['call_price']
#     call_d = CRR_AD_O(N, S0, K, T, r, sigma - h)['call_price']
#     vega_C = (call_u - call_d) / (2 * h)

#     put_u = CRR_AD_O(N, S0, K, T, r, sigma + h)['put_price']
#     put_d = CRR_AD_O(N, S0, K, T, r, sigma - h)['put_price']
#     vega_P = (put_u - put_d) / (2 * h)

#     h = 0.10 * r  # for Rho
#     call_u = CRR_AD_O(N, S0, K, T, r + h, sigma)['call_price']
#     call_d = CRR_AD_O(N, S0, K, T, r - h, sigma)['call_price']
#     rho_C = (call_u - call_d) / (2 * h)

#     put_u = CRR_AD_O(N, S0, K, T, r + h, sigma)['put_price']
#     put_d = CRR_AD_O(N, S0, K, T, r - h, sigma)['put_price']
#     rho_P = (put_u - put_d) / (2 * h)

#     return {
#         'DeltaC': delta_C,
#         'GammaC': gamma_C,
#         'ThetaC': theta_C,
#         'VegaC': vega_C,
#         'RhoC': rho_C,
#         'DeltaP': delta_P,
#         'GammaP': gamma_P,
#         'ThetaP': theta_P,
#         'VegaP': vega_P,
#         'RhoP': rho_P
#     }

def CRR_AD_O(N, S0, K, T, r, sigma):
    """
    Optimized Cox-Ross-Rubinstein (CRR) method using the Arrow-Debreu pricing model.
    """
    start_time = time.time()  # Start timing

    # Calculate the parameters for the binomial tree
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize option values at maturity for call and put options
    call_values = np.zeros(N + 1)
    put_values = np.zeros(N + 1)
    
    for i in range(N + 1):
        asset_price_at_maturity = S0 * (u ** (N - i)) * (d ** i)
        call_values[i] = np.maximum(0, asset_price_at_maturity - K)
        put_values[i] = np.maximum(0, K - asset_price_at_maturity)

    # Backward induction to get option prices at t=0
    for j in range(N - 1, -1, -1):
        call_values = np.exp(-r * dt) * (p * call_values[:-1] + (1 - p) * call_values[1:])
        put_values = np.exp(-r * dt) * (p * put_values[:-1] + (1 - p) * put_values[1:])

    # Compute the option prices at t=0
    call_price = call_values[0]
    put_price = put_values[0]

    computation_time = time.time() - start_time  # End timing
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'computation_time': computation_time
    }

def compute_greeks(N, S0, K, T, r, sigma):
    """
    Compute Greeks for call and put options using the CRR_AD_O function.
    """
    # Calculate up factor and down factor
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor

    # Calculate shifted abscissas for interpolation
    S_points = [
        S0 / u**2,  # S0 / u^2
        S0,          # S0
        S0 * u**2    # S0 * u^2
    ]
    
    # Calculate option prices at shifted stock prices
    prices = {
        'call': [CRR_AD_O(N, S, K, T, r, sigma)['call_price'] for S in S_points],
        'put': [CRR_AD_O(N, S, K, T, r, sigma)['put_price'] for S in S_points]
    }

    # Unpack prices for call and put
    p0_C, p1_C, p2_C = prices['call']
    p0_P, p1_P, p2_P = prices['put']
    
    # Coefficients for call option using quadratic interpolation
    A_C = np.array([
        [S_points[0]**2, S_points[0], 1],
        [S_points[1]**2, S_points[1], 1],
        [S_points[2]**2, S_points[2], 1]
    ])
    B_C = np.array([p0_C, p1_C, p2_C])
    
    coeffs_C = np.linalg.solve(A_C, B_C)  # Solve for a, b, c
    a_C, b_C, _ = coeffs_C

    # Delta and Gamma for call
    delta_C = 2 * a_C * S0 + b_C  # Evaluate derivative at S0
    gamma_C = 2 * a_C  # Gamma is twice the quadratic term

    # Coefficients for put option using quadratic interpolation
    A_P = np.array([
        [S_points[0]**2, S_points[0], 1],
        [S_points[1]**2, S_points[1], 1],
        [S_points[2]**2, S_points[2], 1]
    ])
    B_P = np.array([p0_P, p1_P, p2_P])
    
    coeffs_P = np.linalg.solve(A_P, B_P)  # Solve for a, b, c
    a_P, b_P, _ = coeffs_P

    # Delta and Gamma for put
    delta_P = 2 * a_P * S0 + b_P  # Evaluate derivative at S0
    gamma_P = 2 * a_P  # Gamma is twice the quadratic term

    # Compute Theta using centered difference approximation
    h = 0.10 * T  # for Theta
    call_u = CRR_AD_O(N, S0, K, T + h, r, sigma)['call_price']
    call_d = CRR_AD_O(N, S0, K, T - h, r, sigma)['call_price']
    theta_C = -(call_u - call_d) / (2 * h)

    put_u = CRR_AD_O(N, S0, K, T + h, r, sigma)['put_price']
    put_d = CRR_AD_O(N, S0, K, T - h, r, sigma)['put_price']
    theta_P = -(put_u - put_d) / (2 * h)

    # Compute Vega using centered difference approximation
    h = 0.10 * sigma  # for Vega
    call_u = CRR_AD_O(N, S0, K, T, r, sigma + h)['call_price']
    call_d = CRR_AD_O(N, S0, K, T, r, sigma - h)['call_price']
    vega_C = (call_u - call_d) / (2 * h)

    put_u = CRR_AD_O(N, S0, K, T, r, sigma + h)['put_price']
    put_d = CRR_AD_O(N, S0, K, T, r, sigma - h)['put_price']
    vega_P = (put_u - put_d) / (2 * h)

    # Compute Rho using centered difference approximation
    h = 0.10 * r  # for Rho
    call_u = CRR_AD_O(N, S0, K, T, r + h, sigma)['call_price']
    call_d = CRR_AD_O(N, S0, K, T, r - h, sigma)['call_price']
    rho_C = (call_u - call_d) / (2 * h)

    put_u = CRR_AD_O(N, S0, K, T, r + h, sigma)['put_price']
    put_d = CRR_AD_O(N, S0, K, T, r - h, sigma)['put_price']
    rho_P = (put_u - put_d) / (2 * h)

    return {
        'DeltaC': delta_C,  # Call Delta
        'GammaC': gamma_C,  # Call Gamma
        'ThetaC': theta_C,  # Call Theta
        'VegaC': vega_C,    # Call Vega
        'RhoC': rho_C,      # Call Rho
        'DeltaP': delta_P,  # Put Delta
        'GammaP': gamma_P,  # Put Gamma
        'ThetaP': theta_P,  # Put Theta
        'VegaP': vega_P,    # Put Vega
        'RhoP': rho_P       # Put Rho
    }