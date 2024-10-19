import numpy as np
import time

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

def CRReur(S0, K, T, r, sigma, N):
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
    tuple : (call_price, put_price)
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

    return call_values[0, 0], put_values[0, 0]  # Return C(0) and P(0)

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

# Example usage
# S0 = 100  # Initial stock price
# K = 100   # Strike price
# T = 1     # Time to maturity in years
# r = 0.05  # Risk-free interest rate
# sigma = 0.2  # Volatility
# N_values = [10, 100, 1000]

# results = profile_CRR(N_values, S0, K, T, r, sigma)
# print(results)
