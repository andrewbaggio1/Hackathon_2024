# monte_carlo.py

import numpy as np

def monte_carlo_option_price(S0, K, T, r, sigma, num_paths=10000):
    """
    Calculate the European Call option price using the Monte Carlo method.

    Parameters:
    S0 : float
        Initial stock price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate (annual).
    sigma : float
        Volatility of the underlying stock (annual).
    num_paths : int
        Number of simulated paths.

    Returns:
    float
        Estimated price of the European Call option.
    """
    
    # Generate random numbers for the standard normal distribution
    Z = np.random.normal(0, 1, num_paths)
    
    # Simulate end stock price for each path
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate the payoff for each path
    payoffs = np.maximum(ST - K, 0)
    
    # Calculate the present value of expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price
