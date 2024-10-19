from flask import Flask, render_template, request
from strategies.strategy_methods import (
    get_stock_data,
    get_options_data,
    generate_trade_recommendation,
    calculate_moving_average
)
from strategies.monte_carlo import monte_carlo_option_price
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def home():
    use_example_data = request.form.get('use_example_data', 'off') == 'on'  # Check if example data toggle is on
    stock_symbols = request.form.getlist('stock_symbols')  # Get list of selected stocks
    if not stock_symbols:  # Default stock if none selected
        stock_symbols = ['AAPL']
    
    stock_data = {}
    if use_example_data:
        # Using example data
        for stock_symbol in stock_symbols:
            # Create example data for stocks
            dates = pd.date_range(start='2023-01-01', periods=100).tolist()
            prices = [150 + i * 0.5 for i in range(100)]  # Example prices
            stock_data[stock_symbol] = {
                'dates': dates,
                'prices': prices,
                'short_ma': calculate_moving_average(pd.Series(prices), window=20),
                'long_ma': calculate_moving_average(pd.Series(prices), window=50)
            }
    else:
        for stock_symbol in stock_symbols:
            dates, prices = get_stock_data(stock_symbol)
            if prices is None:
                return render_template('index.html', error=f"Could not fetch data for {stock_symbol}.", stock_symbols=stock_symbols)
            
            stock_data[stock_symbol] = {
                'dates': dates,
                'prices': prices,
                'short_ma': calculate_moving_average(pd.Series(prices), window=20),
                'long_ma': calculate_moving_average(pd.Series(prices), window=50)
            }
    
    # Create Plotly figure
    fig = go.Figure()
    
    for stock_symbol, data in stock_data.items():
        fig.add_trace(go.Scatter(x=data['dates'], y=data['prices'], mode='lines', name=f'{stock_symbol} Price'))
        fig.add_trace(go.Scatter(x=data['dates'][19:], y=data['short_ma'][19:], mode='lines', name=f'{stock_symbol} 20-Day MA'))
        fig.add_trace(go.Scatter(x=data['dates'][49:], y=data['long_ma'][49:], mode='lines', name=f'{stock_symbol} 50-Day MA'))

    graphJSON = pio.to_json(fig)

    return render_template('index.html', stock_symbols=stock_symbols, graphJSON=graphJSON)

@app.route('/options', methods=['GET', 'POST'])

def options():
    use_example_data = request.form.get('use_example_data', 'off') == 'on'  # Get toggle status
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()
    else:
        stock_symbol = 'AAPL'  # Default symbol

    # Fetch options data
    options_data = get_options_data(stock_symbol) if not use_example_data else []  # Example: Use empty list for example data
    return render_template('options.html', options_data=options_data, stock_symbol=stock_symbol)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    use_example_data = request.form.get('use_example_data', 'off') == 'on'  # Get toggle status
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()
    else:
        stock_symbol = 'AAPL'  # Default symbol

    # Generate trade recommendation
    recommendation = generate_trade_recommendation(stock_symbol) if not use_example_data else "Example Recommendation"
    return render_template('recommendation.html', recommendation=recommendation, stock_symbol=stock_symbol)

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    use_example_data = request.form.get('use_example_data', 'off') == 'on'  # Get toggle status
    portfolio = []

    if use_example_data:
        # Using example portfolio data
        portfolio = [
            {
                'name': 'AAPL',
                'shares': 10,
                'avg_price': 150.00,
                'current_price': 160.00,
                'change': round(((160.00 - 150.00) / 150.00) * 100, 2),
                'value': 10 * 160.00  # current_price * shares
            },
            {
                'name': 'MSFT',
                'shares': 5,
                'avg_price': 250.00,
                'current_price': 240.00,
                'change': round(((240.00 - 250.00) / 250.00) * 100, 2),
                'value': 5 * 240.00  # current_price * shares
            },
        ]
    else:
        # Actual data retrieval logic should be implemented here
        # e.g., portfolio = get_actual_portfolio_data()
        pass

    total_investment = sum(stock['avg_price'] * stock['shares'] for stock in portfolio)
    total_value = sum(stock['value'] for stock in portfolio)
    overall_change = round(((total_value - total_investment) / total_investment) * 100, 2) if total_investment > 0 else 0

    # Create plotly figure for portfolio value visualization
    portfolio_data = {
        'values': [stock['value'] for stock in portfolio],
        'labels': [stock['name'] for stock in portfolio]
    }
    fig = go.Figure(data=[go.Pie(labels=portfolio_data['labels'], values=portfolio_data['values'])])
    graphJSON = pio.to_json(fig)

    return render_template('portfolio.html', 
                           portfolio=portfolio,
                           total_investment=total_investment,
                           total_value=total_value,
                           overall_change=overall_change,
                           graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
