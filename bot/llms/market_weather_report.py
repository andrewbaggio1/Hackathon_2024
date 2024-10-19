import os
import pandas as pd
import numpy as np
from transformers import pipeline  # import the transformers library

def preprocess_data(historical_data):
    # calculate daily returns and volatility using vectorized operations
    historical_data['daily_return'] = historical_data['Close'].pct_change()
    historical_data['volatility'] = historical_data['daily_return'].rolling(window=7).std()
    historical_data['daily_range'] = historical_data['High'] - historical_data['Low']
    historical_data['volume_change'] = historical_data['Volume'].pct_change()
    return historical_data

def prepare_data_summary(tickers):
    summaries = []
    for ticker in tickers:
        # load processed historical data with optimized csv reading
        historical_data = pd.read_csv(
            os.path.join('..', 'data', 'data', '{ticker}_historical_data.csv'),
            usecols=['Date', 'Close', 'High', 'Low', 'Volume'],
            dtype={'Date': 'str', 'Close': 'float', 'High': 'float', 'Low': 'float', 'Volume': 'int'}
        )
        processed_data = preprocess_data(historical_data)

        # extract key insights (e.g., average return, volatility, average daily range, volume change)
        avg_return = processed_data['daily_return'].mean()
        avg_volatility = processed_data['volatility'].mean()
        avg_daily_range = processed_data['daily_range'].mean()
        avg_volume_change = processed_data['volume_change'].mean()
        summaries.append(
            f"{ticker}: Avg Return: {avg_return:.2%}, Avg Volatility: {avg_volatility:.2%}, "
            f"Avg Daily Range: {avg_daily_range:.2f}, Avg Volume Change: {avg_volume_change:.2%}"
        )

    return "\n".join(summaries)

def generate_market_snapshot(data_summary):
    """
    Generate a market snapshot using an LLM.

    :param data_summary: Summary of processed data to feed into the LLM.
    :return: Market snapshot as plain text.
    """
    prompt = f"""
    You are a financial market analyst. Here is this hour's processed market data:
    {data_summary}

    Provide a concise summary in plain language about the market's performance this hour.
    Include insights on major trends, noteworthy stock movements, and any changes in options activity.
    """
    # use transformers pipeline for text summarization
    summarizer = pipeline("summarization", model="distilbert-base-uncased")
    response = summarizer(prompt, max_length=150, min_length=50, do_sample=False)
    return response[0]['summary_text']

def main():
    # list of tickers to summarize (top 50 s&p 500 stocks)
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'META', 'UNH',
        'XOM', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
        'KO', 'MRK', 'LLY', 'BAC', 'PEP', 'PFE', 'COST', 'TMO', 'DIS', 'CSCO',
        'MCD', 'ABT', 'DHR', 'ACN', 'WFC', 'AVGO', 'ADBE', 'VZ', 'CRM', 'TXN',
        'NEE', 'CMCSA', 'NFLX', 'INTC', 'LIN', 'NKE', 'AMD', 'MDT', 'UNP', 'QCOM'
    ]

    # prepare data summary from processed data
    data_summary = prepare_data_summary(tickers)

    # market snapshot
    snapshot = generate_market_snapshot(data_summary)
    return snapshot

if __name__ == "__main__":
    snapshot_text = main()
    print(snapshot_text)