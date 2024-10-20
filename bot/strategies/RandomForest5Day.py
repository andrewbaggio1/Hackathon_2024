import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import ta  # For technical indicators
import numpy as np
from sklearn.metrics import mean_squared_error


def forecast_stock(stock_data, n_forecast_days=5):
    """
    Function to forecast stock returns or prices using Random Forest Regressor.

    Args:
    - stock_data: DataFrame containing the historical data for a stock (should have 'Date', 'Close', 'Volume').
    - n_forecast_days: Number of days to forecast (default: 5).

    Returns:
    - forecasted_values: The predicted returns or prices for the next n_forecast_days.
    - model: Trained Random Forest model (can be used to inspect feature importances).
    """
    df = stock_data.copy()

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
    df['mean_reversion_signal'] = np.where(df['Close'] < df['short_ma'], 1, 0)

    # Add Bollinger Bands signals
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Upper Band'] = df['SMA_20'] + (2 * df['Close'].rolling(window=20).std())
    df['Lower Band'] = df['SMA_20'] - (2 * df['Close'].rolling(window=20).std())
    df['bollinger_signal'] = np.where(df['Close'] < df['Lower Band'], 1,
                                      np.where(df['Close'] > df['Upper Band'], -1, 0))

    # Calculate Future Returns (Target variable)
    df['Future_Return'] = df['Close'].shift(-n_forecast_days) / df['Close'] - 1  # Predict future return

    # Drop NaN values
    df.dropna(inplace=True)

    # Prepare features and target variable
    X = df[['RSI', 'MACD', 'MACD_signal', 'ma_crossover_signal', 'mean_reversion_signal', 'bollinger_signal', 'Volume']]
    y = df['Future_Return']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Forecast the next n_forecast_days (since the last known data point)
    last_known_features = X.tail(n_forecast_days)
    forecasted_returns = model.predict(last_known_features)

    # Convert forecasted returns to forecasted prices
    last_known_price = df['Close'].iloc[-1]
    forecasted_prices = [last_known_price]

    for ret in forecasted_returns:
        forecasted_prices.append(forecasted_prices[-1] * (1 + ret))

    # Return forecasted prices and the model for future inspection
    return forecasted_prices[1:], model, y_test, X_test, y_train


def simulate_investing(stock_data, forecasted_prices, initial_capital=10000):
    """
    Simulates investing based on the forecasted prices.

    Args:
    - stock_data: The original stock data with actual prices.
    - forecasted_prices: The predicted prices from the model.
    - initial_capital: Starting capital for investment (default: $10,000).

    Returns:
    - final_portfolio_value: The final value of the portfolio after the investment simulation.
    """
    cash = initial_capital
    position = 0  # Number of shares
    portfolio_value = [initial_capital]  # Track portfolio value over time

    # Generate buy/sell signals based on forecasted prices
    actual_prices = stock_data['Close'].iloc[-len(forecasted_prices):].values

    for i in range(len(forecasted_prices)):
        # Buy signal: if forecasted price is higher than current actual price
        if forecasted_prices[i] > actual_prices[i] and cash > 0:
            # Buy as many shares as possible
            position = cash / actual_prices[i]
            cash = 0  # All cash is spent

        # Sell signal: if forecasted price is lower than current actual price
        elif forecasted_prices[i] < actual_prices[i] and position > 0:
            # Sell all shares
            cash = position * actual_prices[i]
            position = 0  # No more shares held

        # Update portfolio value
        portfolio_value.append(cash + position * actual_prices[i])

    # Final portfolio value after the simulation
    final_portfolio_value = portfolio_value[-1]
    return final_portfolio_value, portfolio_value


# List of top 50 stock tickers
top_50_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'META', 'UNH',
    'XOM', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
    'KO', 'MRK', 'LLY', 'BAC', 'PEP', 'PFE', 'COST', 'TMO', 'DIS', 'CSCO',
    'MCD', 'ABT', 'DHR', 'ACN', 'WFC', 'AVGO', 'ADBE', 'VZ', 'CRM', 'TXN',
    'NEE', 'CMCSA', 'NFLX', 'INTC', 'LIN', 'NKE', 'AMD', 'MDT', 'UNP', 'QCOM'
]

# Directory where the historical data for all stocks is stored
data_dir = '/bot/strategies/RandomForest5Day.py'  # Update this path

# Store forecast and investment results for each stock
forecast_results = []
investment_results = []

# Loop through each stock file
for stock in top_50_tickers:
    stock_file = os.path.join(data_dir, f'{stock}_historical_data.csv')
    if not os.path.exists(stock_file):
        print(f"Data for {stock} not found. Skipping...")
        continue

    # Load stock data
    stock_data = pd.read_csv(stock_file)

    # Get 5-day forecast
    forecasted_values, model, y_test, X_test, y_train = forecast_stock(stock_data, n_forecast_days=5)

    # Simulate investing based on forecasted prices
    final_value, portfolio_value = simulate_investing(stock_data, forecasted_values)


    # Store the forecast and investment results
    # Store the forecast results with each day's forecast in separate columns
    forecast_results.append({
        'Stock': stock,
        **{f'Day_{i + 1}': value for i, value in enumerate(forecasted_values)}
    })

    # Prepare investment results with portfolio value over time in separate columns
    investment_results.append({
        'Stock': stock,
        'Final Portfolio Value': final_value,
        **{f'Day_{i+1}': value for i, value in enumerate(portfolio_value)}
    })
# Convert the forecast results and investment results to DataFrames
forecast_df = pd.DataFrame(forecast_results)
investment_df = pd.DataFrame(investment_results)

# Display forecasted values and investment results
print(forecast_df)
print(investment_df)

# Save the forecasted values and investment results to CSV files
forecast_output_path = '/Users/vpalava/PycharmProjects/Hackathon_2024/bot/data/data_out/forecast_results.csv'  # Update this path
investment_output_path = '/Users/vpalava/PycharmProjects/Hackathon_2024/bot/data/data_out/investment_results.csv'  # Update this path

forecast_df.to_csv(forecast_output_path, index=False)
investment_df.to_csv(investment_output_path, index=False)
print(f"Forecast results saved to: {forecast_output_path}")
print(f"Investment results saved to: {investment_output_path}")
