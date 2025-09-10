"""
Market Data Collector for Event-Driven Stock Trend Predictor
Collects historical and real-time data for 5 target Indian stocks
Supports both NSE and ADR data collection
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import requests
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataCollector:
    """
    Collects market data for Indian stocks from multiple sources
    Handles both NSE (.NS) and ADR symbols
    """
    
    def __init__(self, config_path: str = "config/stock_config.yaml"):
        """Initialize the market data collector"""
        self.config = self._load_config(config_path)
        self.stocks = self.config['stocks']
        self.data_dir = Path("data/raw/market_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting for API calls
        self.last_api_call = {}
        self.min_delay_seconds = 1  # Minimum delay between API calls
        
    def _load_config(self, config_path: str) -> Dict:
        """Load stock configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def _rate_limit(self, source: str):
        """Implement rate limiting for API calls"""
        if source in self.last_api_call:
            elapsed = time.time() - self.last_api_call[source]
            if elapsed < self.min_delay_seconds:
                time.sleep(self.min_delay_seconds - elapsed)
        self.last_api_call[source] = time.time()
    
    def get_historical_data(self, 
                          stock_code: str, 
                          period: str = "5y",
                          source: str = "yfinance") -> pd.DataFrame:
        """
        Get historical data for a stock
        
        Args:
            stock_code: Stock symbol (e.g., 'HDFCBANK.NS' or 'HDB')
            period: Time period ('1y', '2y', '5y', '10y', 'max')
            source: Data source ('yfinance' or 'alpha_vantage')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._rate_limit(source)
            
            if source == "yfinance":
                return self._get_yfinance_data(stock_code, period)
            elif source == "alpha_vantage":
                return self._get_alpha_vantage_data(stock_code, period)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {stock_code}: {e}")
            return pd.DataFrame()
    
    def _get_yfinance_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add metadata
            data['symbol'] = symbol
            data['source'] = 'yfinance'
            data['timestamp'] = data.index
            
            logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_alpha_vantage_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get data from Alpha Vantage (placeholder - requires API key)"""
        # This would require Alpha Vantage API key
        # For now, return empty DataFrame
        logger.info("Alpha Vantage integration not implemented yet")
        return pd.DataFrame()

    def update_daily_prices(self, symbol: str) -> str:
        """
        Efficiently updates stock data by fetching only new records.
        It loads existing data, finds the last date, and appends new data since then.
        """
        stock_key = symbol.split('.')[0]
        file_path = self.data_dir / stock_key / f"{stock_key}_indian_historical.csv"
        
        start_date = None
        if file_path.exists():
            try:
                existing_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if not existing_df.empty:
                    last_date = existing_df.index.max()
                    start_date = last_date + timedelta(days=1)
            except Exception as e:
                return f"Error reading existing data for {symbol}: {e}"

        try:
            if start_date and start_date.date() >= datetime.today().date():
                return f"Data for {symbol} is already up to date."
        
            self._rate_limit('yfinance')
            ticker = yf.Ticker(symbol)
            # Fetch data from the day after the last recorded date up to the present
            new_data = ticker.history(start=start_date, period=None)
            
            if new_data.empty:
                return f"No new data found for {symbol} since {start_date.strftime('%Y-%m-%d') if start_date else 'the beginning'}."

            # Append new data to the existing dataframe
            if 'existing_df' in locals() and not existing_df.empty:
                updated_df = pd.concat([existing_df, new_data])
            else:
                updated_df = new_data
            
            # Standardize column names and save the updated file
            updated_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            updated_df['symbol'] = symbol
            self.save_data(updated_df, stock_key, 'indian')
            return f"Added {len(new_data)} new records for {symbol}."
        
        except Exception as e:
            return f"Error fetching new data for {symbol}: {e}"        
    
    def collect_dual_market_data(self, stock_name: str) -> Dict[str, pd.DataFrame]:
        """
        Collect data from both Indian market (NS) and ADR if available
        
        Args:
            stock_name: Stock name from config (e.g., 'HDFCBANK')
        
        Returns:
            Dictionary with 'indian' and 'adr' DataFrames
        """
        if stock_name not in self.stocks:
            raise ValueError(f"Stock {stock_name} not found in configuration")
        
        stock_config = self.stocks[stock_name]
        results = {}
        
        # Get Indian market data
        indian_symbol = stock_config['indian_symbol']
        logger.info(f"Collecting Indian market data for {indian_symbol}")
        
        indian_data = self.get_historical_data(
            indian_symbol, 
            stock_config['training_lookback']
        )
        
        if not indian_data.empty:
            results['indian'] = indian_data
            logger.info(f"Indian data: {len(indian_data)} records")
        
        # Get ADR data if available
        if stock_config.get('adr_symbol'):
            adr_symbol = stock_config['adr_symbol']
            logger.info(f"Collecting ADR data for {adr_symbol}")
            
            adr_data = self.get_historical_data(
                adr_symbol,
                stock_config['training_lookback']
            )
            
            if not adr_data.empty:
                results['adr'] = adr_data
                logger.info(f"ADR data: {len(adr_data)} records")
        else:
            logger.info(f"No ADR symbol for {stock_name}")
        
        return results
    
    def save_data(self, data: pd.DataFrame, stock_name: str, market_type: str):
        """
        Save data to CSV files
        
        Args:
            data: DataFrame to save
            stock_name: Stock name (e.g., 'HDFCBANK')
            market_type: 'indian' or 'adr'
        """
        stock_dir = self.data_dir / stock_name
        stock_dir.mkdir(exist_ok=True)
        
        filename = f"{stock_name}_{market_type}_historical.csv"
        filepath = stock_dir / filename
        
        try:
            data.to_csv(filepath, index=True)
            logger.info(f"Saved {len(data)} records to {filepath}")
            
            # Save metadata
            metadata = {
                'symbol': data['symbol'].iloc[0] if 'symbol' in data.columns else 'unknown',
                'records': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'last_updated': datetime.now().isoformat(),
                'market_type': market_type
            }
            
            metadata_file = stock_dir / f"{stock_name}_{market_type}_metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
                
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
    
    def collect_all_stocks(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect historical data for all configured stocks
        
        Returns:
            Nested dictionary: {stock_name: {market_type: DataFrame}}
        """
        all_data = {}
        
        for stock_name in self.stocks.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {stock_name}")
            logger.info(f"{'='*50}")
            
            try:
                stock_data = self.collect_dual_market_data(stock_name)
                all_data[stock_name] = stock_data
                
                # Save each market type
                for market_type, data in stock_data.items():
                    if not data.empty:
                        self.save_data(data, stock_name, market_type)
                
                # Small delay between stocks to be respectful to APIs
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing {stock_name}: {e}")
                continue
        
        return all_data
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """
        Get current/latest quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with current price, volume, etc.
        """
        try:
            self._rate_limit('yfinance_realtime')
            ticker = yf.Ticker(symbol)
            
            # Get latest data point
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return {}
            
            latest = data.iloc[-1]
            
            quote = {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'timestamp': latest.name.isoformat(),
                'source': 'yfinance'
            }
            
            return quote
            
        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {e}")
            return {}
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality and return metrics
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if data.empty:
            return {'status': 'empty', 'issues': ['No data']}
        
        issues = []
        warnings = []
        
        # Check for missing values
        missing_cols = data.isnull().sum()
        if missing_cols.any():
            issues.extend([f"Missing values in {col}: {count}" 
                          for col, count in missing_cols.items() if count > 0])
        
        # Check for duplicate dates
        if data.index.duplicated().any():
            issues.append("Duplicate timestamps found")
        
        # Check for unrealistic price movements (>50% in one day)
        if 'close' in data.columns:
            daily_returns = data['close'].pct_change()
            extreme_moves = abs(daily_returns) > 0.5
            if extreme_moves.any():
                warnings.append(f"Extreme price movements detected: {extreme_moves.sum()} days")
        
        # Check data recency
        if not data.empty:
            last_date = data.index.max()
            days_old = (datetime.now().date() - last_date.date()).days
            if days_old > 7:
                warnings.append(f"Data is {days_old} days old")
        
        quality_score = max(0, 100 - len(issues) * 10 - len(warnings) * 5)
        
        return {
            'status': 'good' if not issues else 'issues_found',
            'quality_score': quality_score,
            'record_count': len(data),
            'date_range': f"{data.index.min()} to {data.index.max()}" if not data.empty else "No data",
            'issues': issues,
            'warnings': warnings
        }

def main():
    """Main function to test the data collector"""
    try:
        # Initialize collector
        collector = MarketDataCollector()
        
        # Test single stock collection
        logger.info("Testing single stock data collection...")
        test_data = collector.collect_dual_market_data('HDFCBANK')
        
        for market_type, data in test_data.items():
            if not data.empty:
                quality = collector.validate_data_quality(data)
                logger.info(f"HDFCBANK {market_type} data quality: {quality}")
        
        # Uncomment to collect all stocks (takes 5-10 minutes)
        # logger.info("Collecting all stocks data...")
        # all_data = collector.collect_all_stocks()
        # logger.info("Data collection completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()