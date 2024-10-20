import pandas as pd
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import os
import torch

# Configure logging to only display errors
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def load_market_data():
    """
    Load the market data from 'friday_data/market_data_2024-10-18.csv'.

    :return: DataFrame containing the market data or None if loading fails.
    """
    try:
        # Determine the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the path to the 'friday_data' folder
        data_folder = os.path.join(script_dir, '..', 'friday_data')
        
        # Specify the exact CSV file name
        csv_file = os.path.join(data_folder, 'market_data_2024-10-18.csv')
        
        # Check if the file exists
        if not os.path.isfile(csv_file):
            logging.error(f"CSV file not found at path: {csv_file}")
            return None
        
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_file, parse_dates=['Date'])
        return df
    except Exception as e:
        logging.error(f"Error loading market data: {e}")
        return None

def summarize_market_data(df):
    """
    Calculate key market statistics from the DataFrame.

    :param df: DataFrame containing market data.
    :return: Dictionary with calculated statistics or None if an error occurs.
    """
    try:
        # Calculate percentage change based on Adj Close
        df['pct_change'] = ((df['Close'] - df['Adj Close']) / df['Adj Close']) * 100

        # Calculate average percentage change
        average_pct_change = df['pct_change'].mean()

        # Calculate total volume
        total_volume = df['Volume'].sum()

        # Calculate number of gainers and losers
        num_gainers = df[df['pct_change'] > 0].shape[0]
        num_losers = df[df['pct_change'] < 0].shape[0]

        # Identify top 3 gainers and losers
        top_gainers = df.sort_values(by='pct_change', ascending=False).head(3)
        top_losers = df.sort_values(by='pct_change').head(3)

        # Extract the date (assuming all entries are from the same date)
        market_date = df['Date'].dt.strftime('%B %d, %Y').iloc[0]

        summary = {
            'Date': market_date,
            'Total Stocks Traded': df.shape[0],
            'Average Percentage Change': average_pct_change,
            'Total Volume': total_volume,
            'Number of Gainers': num_gainers,
            'Number of Losers': num_losers,
            'Top Gainers': top_gainers[['Ticker', 'pct_change']].to_dict('records'),
            'Top Losers': top_losers[['Ticker', 'pct_change']].to_dict('records')
        }

        return summary
    except Exception as e:
        logging.error(f"Error summarizing market data: {e}")
        return None

def generate_verbose_summary(summary):
    """
    Generate a verbose, newscaster-style market summary using a language model.

    :param summary: Dictionary containing market statistics.
    :return: String containing the detailed market summary.
    """
    try:
        # Determine the device: 0 for GPU, -1 for CPU
        device = 0 if torch.cuda.is_available() else -1

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

        # Initialize the summarization pipeline
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            framework="pt",  # Use PyTorch
            device=device,
            max_length=300,   # Increased max_length for more verbosity
            min_length=150,   # Increased min_length to ensure detail
            do_sample=True,   # Enable sampling for creativity
            top_k=50,         # Top-k sampling
            top_p=0.95,       # Nucleus sampling
            truncation=True
        )

        # Prepare the input text with clear instructions
        input_text = (
            f"Date: {summary['Date']}\n"
            f"Total Stocks Traded: {summary['Total Stocks Traded']}\n"
            f"Average Percentage Change: {summary['Average Percentage Change']:.2f}%\n"
            f"Total Volume: {summary['Total Volume']:,}\n"
            f"Number of Gainers: {summary['Number of Gainers']}\n"
            f"Number of Losers: {summary['Number of Losers']}\n"
            f"Top Gainers: {', '.join([g['Ticker'] for g in summary['Top Gainers']])}\n"
            f"Top Losers: {', '.join([l['Ticker'] for l in summary['Top Losers']])}\n\n"
            "Based on the above data, please provide a detailed newscaster-style summary highlighting major trends, overall market sentiment, and significant observations related to larger macroeconomic trends."
        )

        # Generate the summary
        response = summarizer(
            input_text,
            max_length=300,
            min_length=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            truncation=True
        )

        # Extract the generated summary
        generated_summary = response[0]['summary_text'].strip()

        # Check if the summary is sufficiently verbose
        if len(generated_summary.split()) < 50:
            # Fallback to manual summary if not verbose enough
            return manual_verbose_summary(summary)

        return generated_summary
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        # Fallback to manual summary in case of error
        return manual_verbose_summary(summary)

def manual_verbose_summary(summary):
    """
    Generate a verbose summary using a manual template.

    :param summary: Dictionary containing market statistics.
    :return: String containing the detailed market summary.
    """
    try:
        # Determine overall market sentiment
        if summary['Number of Gainers'] > summary['Number of Losers']:
            sentiment = "bullish"
        elif summary['Number of Gainers'] < summary['Number of Losers']:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # Format total volume with commas for readability
        formatted_volume = f"{summary['Total Volume']:,}"

        # Construct the verbose summary using a template
        verbose_summary = (
            f"Market Summary for {summary['Date']}:\n\n"
            f"Today's trading session was marked by the activity of {summary['Total Stocks Traded']} stocks, "
            f"with an average percentage change of {summary['Average Percentage Change']:.2f}%. "
            f"The market saw a substantial total volume of {formatted_volume} shares exchanged, "
            f"reflecting significant investor engagement.\n\n"
            f"In terms of performance, there were {summary['Number of Gainers']} gainers and {summary['Number of Losers']} losers. "
            f"This distribution indicates a {sentiment} sentiment prevailing in the trading environment.\n\n"
            f"Leading the charge among gainers were {', '.join([g['Ticker'] for g in summary['Top Gainers']])}, "
            f"each demonstrating notable upward movements. Conversely, the top performers among losers included {', '.join([l['Ticker'] for l in summary['Top Losers']])}, "
            f"showing downward trends.\n\n"
            f"These movements are reflective of broader macroeconomic trends, such as [Insert relevant macroeconomic trend here based on additional data]. "
            f"Overall, the market's performance today provides valuable insights into the current economic landscape and investor sentiment."
        )

        return verbose_summary
    except Exception as e:
        logging.error(f"Error generating manual summary: {e}")
        return "Error generating manual summary."

def main_green():
    """
    Main function to execute the data processing and generate the market summary.
    """
    # Load the market data
    df = load_market_data()

    if df is None or df.empty:
        logging.error("Market data is empty or could not be loaded.")
        return

    # Summarize the data
    summary = summarize_market_data(df)

    if summary is None:
        logging.error("Failed to summarize market data.")
        return

    # Generate the verbose summary
    verbose_summary = generate_verbose_summary(summary)

    # Output only the final summary text
    return verbose_summary

# if __name__ == "__main__":
#     main()
