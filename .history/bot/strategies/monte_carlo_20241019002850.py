import numpy as np

def monte_carlo_option_price(S0, K, T, r, sigma, num_paths=10000, option_type='call'):
    """
    Calculate the European option price using the Monte Carlo method.

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
    option_type : str
        Type of option ('call' or 'put').

    Returns:
    float
        Estimated price of the European option.
    float
        Standard deviation of the estimated price.
    """
    if S0 <= 0 or K <= 0 or T <= 0 or sigma < 0:
        raise ValueError("Invalid input parameters. Prices and volatility must be positive.")

    # Generate random numbers for the standard normal distribution
    Z = np.random.normal(0, 1, num_paths)
    
    # Simulate end stock price for each path
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate the payoff for each path
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Calculate the present value of expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_dev = np.std(payoffs) * np.exp(-r * T)  # Standard deviation of the payoffs
    
    return option_price, std_dev

# Example usage
# if __name__ == '__main__':
#     S0 = 100  # Current stock price
#     K = 100   # Strike price
#     T = 1     # Time to maturity in years
#     r = 0.05  # Risk-free interest rate
#     sigma = 0.2  # Volatility

#     price, std_dev = monte_carlo_option_price(S0, K, T, r, sigma, num_paths=10000, option_type='call')
#     print(f"Estimated Call Option Price: {price:.2f}, Standard Deviation: {std_dev:.2f}")
