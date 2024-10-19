import os
import alpaca_trade_api as tradeapi

# Set API keys and endpoint
API_KEY = os.getenv('APCA_API_KEY_ID')  # Replace with your API key
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')  # Replace with your secret key
BASE_URL = 'https://paper-api.alpaca.markets'  # Change to live API URL for live trading

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def get_account():
    """Fetch account information."""
    account = api.get_account()
    return account

def get_positions():
    """Fetch current positions."""
    positions = api.list_positions()
    return positions

def place_order(symbol, qty, side, order_type='market', time_in_force='gtc'):
    """Place a new order."""
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force
    )
    return order

def get_order_history():
    """Fetch order history."""
    orders = api.list_orders()
    return orders

def get_asset(symbol):
    """Fetch asset information."""
    asset = api.get_asset(symbol)
    return asset

def main():
    print("Welcome to Alpaca Trading API!")
    while True:
        print("\nSelect an option:")
        print("1. Get Account Information")
        print("2. Get Current Positions")
        print("3. Place an Order")
        print("4. Get Order History")
        print("5. Get Asset Information")
        print("6. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            account_info = get_account()
            print(account_info)
        elif choice == '2':
            positions = get_positions()
            print(positions)
        elif choice == '3':
            symbol = input("Enter the symbol: ")
            qty = int(input("Enter quantity: "))
            side = input("Enter side (buy/sell): ").lower()
            order_type = input("Enter order type (market/limit): ").lower()
            order = place_order(symbol, qty, side, order_type)
            print("Order placed:", order)
        elif choice == '4':
            orders = get_order_history()
            print(orders)
        elif choice == '5':
            symbol = input("Enter the symbol: ")
            asset = get_asset(symbol)
            print(asset)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            
