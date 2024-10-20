import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Define the list of tickers
tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'META', 'UNH',
    'XOM', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
    'KO', 'MRK', 'LLY', 'BAC', 'PEP', 'PFE', 'COST', 'TMO', 'DIS', 'CSCO',
    'MCD', 'ABT', 'DHR', 'ACN', 'WFC', 'AVGO', 'ADBE', 'VZ', 'CRM', 'TXN',
    'NEE', 'CMCSA', 'NFLX', 'INTC', 'LIN', 'NKE', 'AMD', 'MDT', 'UNP', 'QCOM'
]

# Replace 'BRK-B' with 'BRK-B' format acceptable by yfinance
# yfinance typically accepts 'BRK-B' or 'BRK.B'
tickers = ['BRK-B' if ticker == 'BRK-B' else ticker for ticker in tickers]

# Define the directory name
directory = "friday_data"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

# Define the target date
target_date = '2024-10-18'

# Convert the string to a datetime object
target_datetime = datetime.strptime(target_date, '%Y-%m-%d')

# Calculate the end date as the day after the target date
end_datetime = target_datetime + timedelta(days=1)
end_date = end_datetime.strftime('%Y-%m-%d')

# Initialize an empty list to store data
all_data = []

for ticker in tickers:
    try:
        # Fetch data for the specific date range
        data = yf.download(ticker, start=target_date, end=end_date, progress=False)
        
        if not data.empty:
            # Add the ticker symbol to the data
            data['Ticker'] = ticker
            
            # Reset the index to have the date as a column
            data.reset_index(inplace=True)
            
            # Append to the list
            all_data.append(data)
            print(f"Data fetched for {ticker}")
        else:
            print(f"No data found for {ticker} on {target_date}")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

if all_data:
    # Concatenate all dataframes
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Define the CSV file path
    csv_file_path = os.path.join(directory, f"market_data_{target_date}.csv")
    
    # Save to CSV
    combined_data.to_csv(csv_file_path, index=False)
    
    print(f"Market data saved to {csv_file_path}")
else:
    print("No data to save.")