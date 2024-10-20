import os
import pandas as pd
import csv

from strategies import (  # Import your strategies from strategies.py
    generate_mean_reversion_signals, backtest_mean_reversion,
    moving_average_crossover, backtest_moving_average_crossover,
    rsi_strategy, backtest_rsi_strategy,
    macd_strategy, backtest_macd_strategy,
    bollinger_bands_strategy, backtest_bollinger_bands_strategy,
    bollinger_bands_with_trend_filter, backtest_bollinger_bands_with_trend_filter,
    rsi_with_trend_filter, backtest_rsi_with_trend_filter,
    macd_with_trend_filter, backtest_macd_with_trend_filter,
    buy_and_hold_backtesting, calculate_risk_metrics, compare_strategies_risk_metrics
)

# Directory containing your CSV data
data_dir = '/Users/jbelmont/Downloads/WashU/Junior/Hackathon_2024/bot/data/data'
data_dir_out = '/Users/jbelmont/Downloads/WashU/Junior/Hackathon_2024/bot/data/data_out'


# List of top 50 tickers (you can adjust as needed)
top_50_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'META', 'UNH',
    'XOM', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
    'KO', 'MRK', 'LLY', 'BAC', 'PEP', 'PFE', 'COST', 'TMO', 'DIS', 'CSCO',
    'MCD', 'ABT', 'DHR', 'ACN', 'WFC', 'AVGO', 'ADBE', 'VZ', 'CRM', 'TXN',
    'NEE', 'CMCSA', 'NFLX', 'INTC', 'LIN', 'NKE', 'AMD', 'MDT', 'UNP', 'QCOM'
]

# Initialize a dictionary to store risk metrics for each stock
risk_metrics_by_stock = {}

# Loop over each stock ticker
for ticker in top_50_tickers:
    file_path = os.path.join(data_dir, f"{ticker}_historical_data.csv")
    
    if not os.path.exists(file_path):
        print(f"Data not found for {ticker}, skipping...")
        continue

    # Load historical data for the stock
    stock_data = pd.read_csv(file_path)
    stock_prices = stock_data['Close'].values
    
    # Apply strategies and backtest them
    strategy_portfolios = {}
    
    # 1. Mean Reversion
    trend = pd.Series(stock_prices).rolling(window=50).mean()  # Simple moving average trend
    mean_reversion_signals = generate_mean_reversion_signals(stock_prices, trend)
    mean_reversion_portfolio = backtest_mean_reversion(stock_prices, mean_reversion_signals)
    strategy_portfolios['Mean Reversion'] = mean_reversion_portfolio
    
    # 2. Moving Average Crossover
    ma_signals, _, _ = moving_average_crossover(stock_prices, short_window=50, long_window=200)
    ma_portfolio, _, _ = backtest_moving_average_crossover(stock_prices, ma_signals)
    strategy_portfolios['Moving Average Crossover'] = ma_portfolio
    
    # 3. RSI Strategy
    rsi_signals, _ = rsi_strategy(stock_prices, rsi_period=14)
    rsi_portfolio, _, _ = backtest_rsi_strategy(stock_prices, rsi_signals)
    strategy_portfolios['RSI'] = rsi_portfolio
    
    # 4. MACD Strategy
    macd_signals, _, _ = macd_strategy(stock_prices)
    macd_portfolio, _, _ = backtest_macd_strategy(stock_prices, macd_signals)
    strategy_portfolios['MACD'] = macd_portfolio
    
    # 5. Bollinger Bands
    bb_signals, _, _, _ = bollinger_bands_strategy(stock_prices)
    bb_portfolio, _, _ = backtest_bollinger_bands_strategy(stock_prices, bb_signals)
    strategy_portfolios['Bollinger Bands'] = bb_portfolio

    # 6. Buy and Hold Strategy
    bnh_portfolio, _ = buy_and_hold_backtesting(stock_prices)
    strategy_portfolios['Buy and Hold'] = bnh_portfolio
    
    # 7. RSI with Trend Filter
    rsi_with_trend_signals, _ = rsi_with_trend_filter(stock_prices, trend)
    rsi_with_trend_portfolio, _, _ = backtest_rsi_with_trend_filter(stock_prices, rsi_with_trend_signals)
    strategy_portfolios['RSI with Trend Filter'] = rsi_with_trend_portfolio
    
    # 8. MACD with Trend Filter
    macd_with_trend_signals, _, _ = macd_with_trend_filter(stock_prices, trend)
    macd_with_trend_portfolio, _, _ = backtest_macd_with_trend_filter(stock_prices, macd_with_trend_signals)
    strategy_portfolios['MACD with Trend Filter'] = macd_with_trend_portfolio
    
    # Calculate risk metrics for each strategy
    risk_metrics = compare_strategies_risk_metrics(strategy_portfolios)

    # Store the metrics for this stock
    risk_metrics_by_stock[ticker] = risk_metrics


# Define the output CSV file path
output_file = '/Users/jbelmont/Downloads/WashU/Junior/Hackathon_2024/bot/data/data_out/risk_metrics_by_stock.csv'

# Open the CSV file for writing
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Ticker', 'Strategy', 'Metric', 'Value'])
    
    # Loop through the risk metrics and write each row to the CSV
    for ticker, metrics in risk_metrics_by_stock.items():
        for strategy, metric_values in metrics.items():
            for metric_name, value in metric_values.items():
                writer.writerow([ticker, strategy, metric_name, f"{value:.4f}"])

print(f"Risk metrics successfully written to {output_file}")
