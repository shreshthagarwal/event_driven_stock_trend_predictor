"""
Main Orchestration Script for the Event-Driven Stock Trend Predictor

This script automates the entire daily workflow:
1.  Updates historical stock price data by adding only new entries.
2.  Updates macroeconomic data, adding only new entries for the past few days.
3.  Refreshes news articles by deleting old ones and fetching new, relevant headlines.
4.  Triggers the training process for all configured stock models via the API.

To run this script, ensure the FastAPI server is running in a separate terminal:
$ python src/backend/api_routes.py

Then, in another terminal, run this script from the project's root directory:
$ python main.py
"""

import yaml
import requests
import time
import sys
from datetime import datetime
import os

# Add the project's root directory to the Python path to allow for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data.market_collector import MarketDataCollector
    from src.data.macro_collector import MacroEconomicDataCollector
    from src.data.news_collector import IndianFinancialNewsCollector as NewsCollector
except ImportError as e:
    print(f"Error: Could not import necessary modules. Ensure you are running this script from the project's root directory.")
    print(f"Details: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
API_BASE_URL = "http://localhost:8000"
CONFIG_PATH = "config/stock_config.yaml"
# Number of days of recent news and macro data to fetch
DATA_FETCH_DAYS = 4 

def load_config():
    """Loads the stock configuration from the YAML file."""
    try:
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{CONFIG_PATH}'")
        sys.exit(1)

def check_api_status():
    """Checks if the backend API is running and operational."""
    try:
        response = requests.get(API_BASE_URL)
        if response.status_code == 200 and response.json().get("status") == "operational":
            print("✅ API is running and operational.")
            return True
        else:
            print(f"API is reachable but not fully operational. Status: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ API is not running. Please start the FastAPI server first in a separate terminal:")
        print("   python src/backend/api_routes.py")
        return False

def update_stock_data(collector, stocks):
    """Updates the stock price database for all configured stocks using the efficient update method."""
    print("\n--- STEP 1: Updating Stock Price Data ---")
    for stock_key, config in stocks.items():
        symbol = config['indian_symbol']
        print(f"Updating data for {stock_key} ({symbol})...")
        try:
            update_summary = collector.update_daily_prices(symbol)
            print(f"  -> {update_summary}")
        except Exception as e:
            print(f"  -> Error updating data for {symbol}: {e}")
    print("✅ Stock price data update complete.")

def update_macro_data(collector):
    """Updates the macroeconomic database with the latest data."""
    print("\n--- STEP 2: Updating Macro-Economic Data ---")
    try:
        update_summary = collector.update_all_macro_data(days=DATA_FETCH_DAYS)
        for line in update_summary.split('\n'):
            print(f"  -> {line}")
    except Exception as e:
        print(f"  -> Error updating macro data: {e}")
    print("✅ Macro-economic data update complete.")

def refresh_news_articles(collector, stocks):
    """Deletes old news and fetches new, relevant articles for each stock."""
    print("\n--- STEP 3: Refreshing News Articles ---")
    for stock_key, config in stocks.items():
        symbol = config['indian_symbol']
        keywords = config['news_keywords']
        print(f"Refreshing news for {stock_key} ({symbol})...")
        try:
            deleted_count, new_count = collector.refresh_stock_news(symbol, keywords, days=DATA_FETCH_DAYS)
            print(f"  -> Deleted {deleted_count} old articles, fetched and stored {new_count} new articles.")
        except Exception as e:
            print(f"  -> Error refreshing news for {symbol}: {e}")
    print("✅ News article refresh complete.")

def train_all_models(stocks):
    """Triggers the training process for all stock models via the API."""
    print("\n--- STEP 4: Triggering Model Training ---")
    for stock_key, config in stocks.items():
        symbol = config['indian_symbol']
        print(f"Sending training request for {stock_key} ({symbol})...")
        try:
            # Send a POST request to the training endpoint
            response = requests.post(f"{API_BASE_URL}/api/train/{symbol}?epochs=50", timeout=30)
            if response.status_code == 200:
                print(f"  -> {response.json().get('message')}")
            else:
                print(f"  -> Error training {symbol}. Status: {response.status_code}, Detail: {response.text}")
            time.sleep(5) # Small delay between requests
        except requests.RequestException as e:
            print(f"  -> Failed to send training request for {symbol}: {e}")
    print("\n✅ All training requests have been sent.")
    print("   Training will now proceed in the background on the server.")
    print("   You can monitor the progress in the FastAPI server's terminal.")

def main():
    """Main function to run the complete daily workflow."""
    print("="*60)
    print(f"Starting Daily Market Update Workflow: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if not check_api_status():
        return

    config = load_config()
    stocks = config.get('stocks', {})
    if not stocks:
        print("Error: No stocks found in the configuration file.")
        return

    # Initialize the data collectors
    market_collector = MarketDataCollector()
    macro_collector = MacroEconomicDataCollector()
    news_collector = NewsCollector()

    # Run the workflow steps in order
    update_stock_data(market_collector, stocks)
    update_macro_data(macro_collector)
    refresh_news_articles(news_collector, stocks)
    train_all_models(stocks)
    
    print("\n" + "="*60)
    print("Workflow complete. The system is now updated and models are training.")
    print("You can now launch the frontend to view the latest data and predictions.")
    print("="*60)

if __name__ == "__main__":
    main()

