import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ta  # For technical indicators
from strategies import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown, calculate_deflated_sharpe_ratio, calculate_psr, calculate_omega_ratio
from strategies import generate_mean_reversion_signals, bollinger_bands_strategy
# Store results for each stock
results = []
# List of 50 stock symbols
top_50_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'META', 'UNH',
    'XOM', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
    'KO', 'MRK', 'LLY', 'BAC', 'PEP', 'PFE', 'COST', 'TMO', 'DIS', 'CSCO',
    'MCD', 'ABT', 'DHR', 'ACN', 'WFC', 'AVGO', 'ADBE', 'VZ', 'CRM', 'TXN',
    'NEE', 'CMCSA', 'NFLX', 'INTC', 'LIN', 'NKE', 'AMD', 'MDT', 'UNP', 'QCOM'
]
import os

# Directory where the historical data for all stocks is stored
data_dir = '/bot/data/data'

# Loop through each stock file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        # Process each stock file here
        print(f"Processing {filename}")

for stock in top_50_tickers:
    # Load stock data
    stock_file = os.path.join(data_dir, f'{stock}_historical_data.csv')
    if not os.path.exists(stock_file):
        print(f"Data for {stock} not found. Skipping...")
        continue

    df = pd.read_csv(stock_file)

    # Add RSI feature
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # Add MACD feature
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Add Moving Average Crossover feature
    df['short_ma'] = df['Close'].rolling(window=50).mean()
    df['long_ma'] = df['Close'].rolling(window=200).mean()
    df['ma_crossover_signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)

    # Add Mean Reversion Signal feature
    df['mean_reversion_signal'] = generate_mean_reversion_signals(df['Close'], df['short_ma'])

    # Add Bollinger Bands signals
    _, upper_band, lower_band, _ = bollinger_bands_strategy(df['Close'], window=20, num_std_dev=2)
    df['bollinger_signal'] = np.where(df['Close'] < lower_band, 1, np.where(df['Close'] > upper_band, -1, 0))

    # Calculate Future Returns (Target variable)
    df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1  # Predict 5-day future return

    # Drop NaN values
    df.dropna(inplace=True)

    # Prepare features and target variable
    X = df[['RSI', 'MACD', 'MACD_signal', 'ma_crossover_signal', 'mean_reversion_signal', 'bollinger_signal', 'Volume']]
    y = df['Future_Return']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for {stock}: {mse}')

    # Backtesting with the predicted future returns
    initial_capital = 10000
    position = 0
    cash = initial_capital
    portfolio_value_rf = [initial_capital]

    # Generate buy/sell signals from predicted returns
    threshold = 0.0  # Buy if predicted return is positive, sell if negative
    signals = np.where(y_pred > threshold, 1, np.where(y_pred < threshold, -1, 0))

    # Backtest the Random Forest strategy
    for i in range(len(signals)):
        if signals[i] == 1 and position == 0:  # Buy signal
            position = cash / df.iloc[i + len(X_train)]['Close']  # Buy as many shares as possible
            cash = 0
        elif signals[i] == -1 and position > 0:  # Sell signal
            cash = position * df.iloc[i + len(X_train)]['Close']  # Sell all shares
            position = 0

        # Update portfolio value
        portfolio_value_rf.append(cash + position * df.iloc[i + len(X_train)]['Close'])

    # Final portfolio value for Random Forest strategy
    final_value_rf = cash + position * df.iloc[-1]['Close']
    print(f'Final Random Forest Strategy Portfolio Value for {stock}: ${final_value_rf:.2f}')

    # Buy-and-Hold Strategy
    shares = initial_capital / df.iloc[len(X_train)]['Close']  # Buy all shares with initial capital at start of test period
    portfolio_value_bh = [initial_capital]

    # Buy-and-hold portfolio value: stock price * shares held
    for i in range(len(X_test)):
        portfolio_value_bh.append(shares * df.iloc[i + len(X_train)]['Close'])

    # Final portfolio value for Buy-and-Hold strategy
    final_value_bh = portfolio_value_bh[-1]
    print(f'Final Buy-and-Hold Portfolio Value for {stock}: ${final_value_bh:.2f}')

    # Calculate daily returns for risk metrics
    daily_returns_rf = np.diff(portfolio_value_rf) / portfolio_value_rf[:-1]
    daily_returns_bh = np.diff(portfolio_value_bh) / portfolio_value_bh[:-1]

    # Risk Metrics for Random Forest Strategy
    sharpe_ratio_rf = calculate_sharpe_ratio(daily_returns_rf)
    sortino_ratio_rf = calculate_sortino_ratio(daily_returns_rf)
    max_drawdown_rf = calculate_max_drawdown(portfolio_value_rf)
    dsr_rf = calculate_deflated_sharpe_ratio(daily_returns_rf, sharpe_ratio_rf, trials=len(daily_returns_rf))
    psr_rf = calculate_psr(daily_returns_rf, sharpe_ratio_rf, benchmark_sharpe=0, skewness=np.mean(daily_returns_rf),
                           kurtosis=3, T=len(daily_returns_rf))
    omega_ratio_rf = calculate_omega_ratio(daily_returns_rf)

    # Risk Metrics for Buy-and-Hold Strategy
    sharpe_ratio_bh = calculate_sharpe_ratio(daily_returns_bh)
    sortino_ratio_bh = calculate_sortino_ratio(daily_returns_bh)
    max_drawdown_bh = calculate_max_drawdown(portfolio_value_bh)
    dsr_bh = calculate_deflated_sharpe_ratio(daily_returns_bh, sharpe_ratio_bh, trials=len(daily_returns_bh))
    psr_bh = calculate_psr(daily_returns_bh, sharpe_ratio_bh, benchmark_sharpe=0, skewness=np.mean(daily_returns_bh),
                           kurtosis=3, T=len(daily_returns_bh))
    omega_ratio_bh = calculate_omega_ratio(daily_returns_bh)

    # Store the results for this stock with risk metrics and comparison
    results.append({
        'Stock': stock,
        'MSE': mse,
        'Final Portfolio Value (RF)': final_value_rf,
        'Final Portfolio Value (BH)': final_value_bh,
        'Sharpe Ratio (RF)': sharpe_ratio_rf,
        'Sharpe Ratio (BH)': sharpe_ratio_bh,
        'Sortino Ratio (RF)': sortino_ratio_rf,
        'Sortino Ratio (BH)': sortino_ratio_bh,
        'Max Drawdown (RF)': max_drawdown_rf,
        'Max Drawdown (BH)': max_drawdown_bh,
        'Deflated Sharpe Ratio (RF)': dsr_rf,
        'Deflated Sharpe Ratio (BH)': dsr_bh,
        'Probabilistic Sharpe Ratio (RF)': psr_rf,
        'Probabilistic Sharpe Ratio (BH)': psr_bh,
        'Omega Ratio (RF)': omega_ratio_rf,
        'Omega Ratio (BH)': omega_ratio_bh
    })

# Convert the results to a DataFrame and display it
results_df = pd.DataFrame(results)
print(results_df)
# Define the output path
output_path = '/Users/vpalava/PycharmProjects/Hackathon_2024/bot/data/data_out/random_forest_backtesting.csv'

# Save the DataFrame to CSV
results_df.to_csv(output_path, index=False)

# Confirm the file has been saved
print(f"Results saved to: {output_path}")

# Store results for each stock
# results = []
#
# # Add RSI feature (using existing RSI strategy)
# df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
#
# # Add MACD feature (using existing MACD strategy)
# macd = ta.trend.MACD(df['Close'])
# df['MACD'] = macd.macd()
# df['MACD_signal'] = macd.macd_signal()
#
# # Add Moving Average Crossover feature (using existing crossover strategy)
# df['short_ma'] = df['Close'].rolling(window=50).mean()
# df['long_ma'] = df['Close'].rolling(window=200).mean()
# df['ma_crossover_signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
#
# # Add Mean Reversion Signal feature (using existing mean reversion strategy)
# df['mean_reversion_signal'] = generate_mean_reversion_signals(df['Close'], df['short_ma'])
#
# # Add Bollinger Bands signals (using existing Bollinger Bands strategy)
# _, upper_band, lower_band, _ = bollinger_bands_strategy(df['Close'], window=20, num_std_dev=2)
# df['bollinger_signal'] = np.where(df['Close'] < lower_band, 1, np.where(df['Close'] > upper_band, -1, 0))
#
# # Calculate Future Returns (Target variable)
# df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1  # Predict 5-day future return
#
# # Drop NaN values
# df.dropna(inplace=True)
#
#
#
# # Prepare features and target variable
# X = df[['RSI', 'MACD', 'MACD_signal', 'ma_crossover_signal', 'mean_reversion_signal', 'bollinger_signal', 'Volume']]
# y = df['Future_Return']
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
# # Train a Random Forest Regressor
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Predict and evaluate on the test set
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
#
# # Backtesting with the predicted future returns
# initial_capital = 10000
# position = 0
# cash = initial_capital
# portfolio_value_rf = [initial_capital]
#
# # Generate buy/sell signals from predicted returns
# threshold = 0.0  # Buy if predicted return is positive, sell if negative
# signals = np.where(y_pred > threshold, 1, np.where(y_pred < threshold, -1, 0))
#
#
# # Backtest the Random Forest strategy
# for i in range(len(signals)):
#     if signals[i] == 1 and position == 0:  # Buy signal
#         position = cash / df.iloc[i + len(X_train)]['Close']  # Buy as many shares as possible
#         cash = 0
#     elif signals[i] == -1 and position > 0:  # Sell signal
#         cash = position * df.iloc[i + len(X_train)]['Close']  # Sell all shares
#         position = 0
#
#     # Update portfolio value
#     portfolio_value_rf.append(cash + position * df.iloc[i + len(X_train)]['Close'])
#
# # Final portfolio value for Random Forest strategy
# final_value_rf = cash + position * df.iloc[-1]['Close']
# print(f'Final Random Forest Strategy Portfolio Value: ${final_value_rf:.2f}')
#
# # Buy-and-Hold Strategy Calculation
# initial_capital = 10000  # Same starting capital as the backtest strategy
# shares = initial_capital / df.iloc[len(X_train)]['Close']  # Buy all shares with initial capital at start of test period
# buy_and_hold_portfolio_value = [initial_capital]
#
# # Buy-and-hold portfolio value: just the stock price times the number of shares held
# for i in range(len(X_test)):
#     buy_and_hold_portfolio_value.append(shares * df.iloc[i + len(X_train)]['Close'])
#
# # Final portfolio value for buy-and-hold strategy
# final_buy_and_hold_value = buy_and_hold_portfolio_value[-1]
# print(f'Final Buy-and-Hold Portfolio Value: ${final_buy_and_hold_value:.2f}')
#
# # Plot both portfolio values over timex
# plt.plot(portfolio_value_rf, label='Strategy Portfolio Value')
# plt.plot(buy_and_hold_portfolio_value, label='Buy and Hold Portfolio Value', linestyle='--')
# plt.title('Backtest: Strategy vs Buy-and-Hold')
# plt.xlabel('Days')
# plt.ylabel('Portfolio Value')
# plt.legend()
# plt.show()
#
# # Feature Importance
# importances = model.feature_importances_
# feature_names = X.columns
# feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# print(feature_importances.sort_values(by='Importance', ascending=False))
#
#
# # Final portfolio value
# final_value = cash + position * df.iloc[-1]['Close']
# print(f'Final Portfolio Value: ${final_value:.2f}')
#
# # Calculate daily returns for risk metrics
# daily_returns_rf = np.diff(portfolio_value_rf) / portfolio_value_rf[:-1]
# daily_returns_bh = np.diff(buy_and_hold_portfolio_value) / buy_and_hold_portfolio_value[:-1]
#
# # Risk Metrics Comparison
#
# # Sharpe Ratio
# sharpe_ratio_rf = calculate_sharpe_ratio(daily_returns_rf)
# sharpe_ratio_bh = calculate_sharpe_ratio(daily_returns_bh)
# print(f"Sharpe Ratio (Random Forest): {sharpe_ratio_rf}")
# print(f"Sharpe Ratio (Buy-and-Hold): {sharpe_ratio_bh}")
#
# # Sortino Ratio
# sortino_ratio_rf = calculate_sortino_ratio(daily_returns_rf)
# sortino_ratio_bh = calculate_sortino_ratio(daily_returns_bh)
# print(f"Sortino Ratio (Random Forest): {sortino_ratio_rf}")
# print(f"Sortino Ratio (Buy-and-Hold): {sortino_ratio_bh}")
#
# # Maximum Drawdown
# max_drawdown_rf = calculate_max_drawdown(portfolio_value_rf)
# max_drawdown_bh = calculate_max_drawdown(buy_and_hold_portfolio_value)
# print(f"Maximum Drawdown (Random Forest): {max_drawdown_rf}")
# print(f"Maximum Drawdown (Buy-and-Hold): {max_drawdown_bh}")
#
# # Deflated Sharpe Ratio
# dsr_rf = calculate_deflated_sharpe_ratio(daily_returns_rf, sharpe_ratio_rf, trials=len(daily_returns_rf))
# dsr_bh = calculate_deflated_sharpe_ratio(daily_returns_bh, sharpe_ratio_bh, trials=len(daily_returns_bh))
# print(f"Deflated Sharpe Ratio (Random Forest): {dsr_rf}")
# print(f"Deflated Sharpe Ratio (Buy-and-Hold): {dsr_bh}")
#
# # Probabilistic Sharpe Ratio
# psr_rf = calculate_psr(daily_returns_rf, sharpe_ratio_rf, benchmark_sharpe=0, skewness=np.mean(daily_returns_rf), kurtosis=3, T=len(daily_returns_rf))
# psr_bh = calculate_psr(daily_returns_bh, sharpe_ratio_bh, benchmark_sharpe=0, skewness=np.mean(daily_returns_bh), kurtosis=3, T=len(daily_returns_bh))
# print(f"Probabilistic Sharpe Ratio (Random Forest): {psr_rf}")
# print(f"Probabilistic Sharpe Ratio (Buy-and-Hold): {psr_bh}")
#
# # Omega Ratio
# omega_ratio_rf = calculate_omega_ratio(daily_returns_rf)
# omega_ratio_bh = calculate_omega_ratio(daily_returns_bh)
# print(f"Omega Ratio (Random Forest): {omega_ratio_rf}")
# print(f"Omega Ratio (Buy-and-Hold): {omega_ratio_bh}")
#
# # Plot both portfolio values over time
# plt.plot(portfolio_value_rf, label='Random Forest Portfolio Value')
# plt.plot(buy_and_hold_portfolio_value, label='Buy and Hold Portfolio Value', linestyle='--')
# plt.title('Backtest: Random Forest Strategy vs Buy-and-Hold')
# plt.xlabel('Days')
# plt.ylabel('Portfolio Value')
# plt.legend()
# plt.show()
#
# # Feature Importance
# importances = model.feature_importances_
# feature_names = X.columns
# feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# print(feature_importances.sort_values(by='Importance', ascending=False))