# market_data_streamer.py

import os
import time
from datetime import datetime
import pandas as pd
from yfin import YahooFinance  # Import the YahooFinance class from yfin.py

def fetch_market_data(tickers, start_date, end_date):
    """
    Fetch market data for a list of tickers.

    :param tickers: List of ticker symbols.
    :param start_date: Start date for historical data.
    :param end_date: End date for historical data.
    """
    for ticker_symbol in tickers:
        print(f"\nProcessing {ticker_symbol} at {datetime.now()}...")
        yf_data = YahooFinance(ticker_symbol)

        # Fetch historical data
        historical_data = yf_data.get_historical_data(start_date, end_date)
        historical_data.to_csv(f'data/{ticker_symbol}_historical_data.csv')
        print(f"Saved historical data for {ticker_symbol}")

        # Fetch options expiration dates
        expiration_dates = yf_data.get_options_expiration_dates()
        print(f"Options Expiration Dates for {ticker_symbol}: {expiration_dates}")

        # Fetch options chain data for each expiration date
        for expiration_date in expiration_dates:
            print(f"Fetching options data for {ticker_symbol} - Expiration Date: {expiration_date}")
            options_chain_data = yf_data.get_options_chain(expiration_date)
            # Save calls and puts to CSV files
            options_chain_data.calls.to_csv(f'data/{ticker_symbol}_options_calls_{expiration_date}.csv')
            options_chain_data.puts.to_csv(f'data/{ticker_symbol}_options_puts_{expiration_date}.csv')
            print(f"Saved options data for {ticker_symbol} - Expiration Date: {expiration_date}")

        # Fetch real-time data
        real_time_data = yf_data.get_real_time_data()
        # Convert real-time data dictionary to DataFrame and save
        real_time_df = pd.DataFrame([real_time_data])
        real_time_df.to_csv(f'data/{ticker_symbol}_real_time_data.csv', index=False)
        print(f"Saved real-time data for {ticker_symbol}")

        # Wait briefly to respect API rate limits
        time.sleep(1)

if __name__ == "__main__":
    # List of top 50 S&P 500 tickers (update this list as needed)
    top_50_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'META', 'UNH',
        'XOM', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
        'KO', 'MRK', 'LLY', 'BAC', 'PEP', 'PFE', 'COST', 'TMO', 'DIS', 'CSCO',
        'MCD', 'ABT', 'DHR', 'ACN', 'WFC', 'AVGO', 'ADBE', 'VZ', 'CRM', 'TXN',
        'NEE', 'CMCSA', 'NFLX', 'INTC', 'LIN', 'NKE', 'AMD', 'MDT', 'UNP', 'QCOM'
    ]

    # Define start and end dates for historical data
    start_date = '2016-01-01'
    end_date = '2024-10-01'

    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Fetch market data once
    try:
        fetch_market_data(top_50_tickers, start_date, end_date)
    except KeyboardInterrupt:
        print("\nFetching stopped by user.")