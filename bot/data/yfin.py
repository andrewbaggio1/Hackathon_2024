# yfin.py

import yfinance as yf
import pandas as pd
import time
from datetime import datetime

class YahooFinance:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def get_historical_data(self, start_date, end_date):
        """
        Fetch historical stock data for the given ticker symbol.
        
        :param start_date: Start date for historical data (e.g., '2023-01-01').
        :param end_date: End date for historical data (e.g., '2023-10-01').
        :return: DataFrame containing historical stock data.
        """
        historical_data = self.stock.history(start=start_date, end=end_date)
        return historical_data

    def get_options_expiration_dates(self):
        """
        Fetch available options expiration dates for the given ticker symbol.
        
        :return: List of expiration dates.
        """
        return self.stock.options

    def get_options_chain(self, expiration_date):
        """
        Fetch options chain data for the given ticker symbol and expiration date.
        
        :param expiration_date: Expiration date for the options (e.g., '2023-10-20').
        :return: DataFrame containing options data.
        """
        options_chain = self.stock.option_chain(expiration_date)
        return options_chain

    def get_real_time_data(self):
        """
        Fetch real-time data for the given ticker symbol.
        
        :return: Dictionary containing real-time data.
        """
        real_time_data = self.stock.info
        return real_time_data