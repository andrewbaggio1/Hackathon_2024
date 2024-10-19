# app.py
from flask import Flask, render_template, request
from strategies.strategy_methods import (
    get_stock_data,
    get_options_data,
    generate_trade_recommendation
)
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()
    else:
        stock_symbol = 'AAPL'  # Default symbol
    
    dates, prices = get_stock_data(stock_symbol)
    if prices is None:
        return render_template('index.html', error="Could not fetch data.", stock_symbol=stock_symbol)

    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
    
    # Plot moving averages
    prices_series = pd.Series(prices)
    short_ma = calculate_moving_average(prices_series, window=20)
    long_ma = calculate_moving_average(prices_series, window=50)
    
    fig.add_trace(go.Scatter(x=dates[19:], y=short_ma[19:], mode='lines', name='20-Day MA'))
    fig.add_trace(go.Scatter(x=dates[49:], y=long_ma[49:], mode='lines', name='50-Day MA'))

    graphJSON = pio.to_json(fig)

    return render_template('index.html', stock_symbol=stock_symbol, graphJSON=graphJSON)

@app.route('/options', methods=['GET', 'POST'])
def options():
    stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()  # Use user input
    options_data = get_options_data(stock_symbol)
    return render_template('options.html', options_data=options_data, stock_symbol=stock_symbol)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    stock_symbol = request.form.get('stock_symbol', 'AAPL').upper()  # Use user input
    recommendation = generate_trade_recommendation(stock_symbol)
    return render_template('recommendation.html', recommendation=recommendation, stock_symbol=stock_symbol)

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    # Example portfolio data (replace this with actual data)
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

    total_investment = sum(stock['avg_price'] * stock['shares'] for stock in portfolio)
    total_value = sum(stock['value'] for stock in portfolio)
    overall_change = round(((total_value - total_investment) / total_investment) * 100, 2)

    # Example portfolio data for visualization
    portfolio_data = {
        'values': [stock['value'] for stock in portfolio],
        'labels': [stock['name'] for stock in portfolio]
    }

    # Create plotly figure for portfolio value visualization
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
