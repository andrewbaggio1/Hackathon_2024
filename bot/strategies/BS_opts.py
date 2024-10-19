import numpy as np
import scipy.stats as si

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize Black-Scholes model parameters.
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility of the stock
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)

    def price(self, call=True):
        """
        Calculate the price of the option.
        
        Parameters:
        call: True for call option, False for put option
        
        Returns:
        Price of the option
        """
        if call:
            return (self.S * si.norm.cdf(self.d1) - 
                    self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2))
        else:
            return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2) - 
                    self.S * si.norm.cdf(-self.d1))

    def delta(self, call=True):
        """Calculate the Delta of the option."""
        return si.norm.cdf(self.d1) if call else si.norm.cdf(self.d1) - 1

    def gamma(self):
        """Calculate the Gamma of the option."""
        return si.norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """Calculate the Vega of the option."""
        return self.S * si.norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self, call=True):
        """Calculate the Theta of the option."""
        theta_value = (-self.S * si.norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T)) -
                       self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2) if call else
                       -self.S * si.norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T)) +
                       self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2))
        return theta_value  # Return per day theta

    def rho(self, call=True):
        """Calculate the Rho of the option."""
        return (self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2) if call else
                -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2))

# Example usage:
# bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
# print("Call Price:", bs.price(call=True))
# print("Put Price:", bs.price(call=False))
# print("Delta:", bs.delta())
# print("Gamma:", bs.gamma())
# print("Vega:", bs.vega())
# print("Theta:", bs.theta())
# print("Rho:", bs.rho())
