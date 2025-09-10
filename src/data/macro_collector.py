"""
Macro Economic Data Collector for Indian Stock Market
Collects RBI policy data, currency rates, commodity prices, and economic indicators
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import sqlite3
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MacroEconomicDataCollector:
    def __init__(self, config_path: str = "config/api_credentials.yaml"):
        """Initialize macro data collector for Indian market"""
        self.load_config(config_path)
        self.setup_database()
        self.setup_data_sources()
        
    def load_config(self, config_path: str):
        """Load API credentials"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.alpha_vantage_key = config['alpha_vantage']['api_key']
                print(f"Loaded Alpha Vantage key for macro data")
        except Exception as e:
            print(f"Error loading config: {e}")
            self.alpha_vantage_key = None
    
    def setup_database(self):
        """Setup database for macro economic data"""
        os.makedirs("data/macro", exist_ok=True)
        self.db_path = "data/macro/macro_data.db"
        
        with sqlite3.connect(self.db_path) as conn:
            # Currency exchange rates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS currency_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    pair TEXT,
                    rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, pair)
                )
            """)
            
            # RBI policy rates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rbi_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    repo_rate REAL,
                    reverse_repo_rate REAL,
                    crr REAL,
                    slr REAL,
                    bank_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            # Inflation data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inflation_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    cpi_inflation REAL,
                    wpi_inflation REAL,
                    food_inflation REAL,
                    core_inflation REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            # Commodity prices
            conn.execute("""
                CREATE TABLE IF NOT EXISTS commodity_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    commodity TEXT,
                    price REAL,
                    unit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, commodity)
                )
            """)
            
            # Global indices
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_indices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    index_name TEXT,
                    value REAL,
                    change_percent REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, index_name)
                )
            """)
            
            # Economic indicators
            conn.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    indicator TEXT,
                    value REAL,
                    unit TEXT,
                    frequency TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, indicator)
                )
            """)
        
        print(f"Macro database ready: {self.db_path}")
    
    def setup_data_sources(self):
        """Setup data source configurations"""
        self.data_sources = {
            'currency_pairs': {
                'USDINR': 'USD/INR',
                'EURINR': 'EUR/INR',
                'GBPINR': 'GBP/INR',
                'JPYINR': 'JPY/INR'
            },
            'commodities': {
                'BRENTOIL': 'Brent Crude Oil',
                'NATURALGAS': 'Natural Gas',
                'GOLD': 'Gold',
                'SILVER': 'Silver',
                'COPPER': 'Copper'
            },
            'global_indices': {
                'SPX': 'S&P 500',
                'IXIC': 'NASDAQ',
                'DJI': 'Dow Jones',
                'N225': 'Nikkei 225',
                'FTSE': 'FTSE 100'
            },
            'indian_indices': {
                'BSE': 'BSE Sensex',
                'NSE': 'Nifty 50'
            }
        }
        
        # RBI policy data (manual entry - RBI doesn't have public API)
        self.rbi_historical_rates = [
            ('2024-12-06', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2024-10-09', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2024-08-08', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2024-06-07', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2024-04-05', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2024-02-08', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2023-12-08', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2023-10-06', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2023-08-10', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2023-06-08', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2023-04-06', 6.50, 3.35, 4.50, 18.00, 7.00),
            ('2023-02-08', 6.25, 3.35, 4.50, 18.00, 6.75),
            ('2022-12-07', 6.25, 3.35, 4.50, 18.00, 6.75),
            ('2022-09-30', 5.90, 3.35, 4.50, 18.00, 6.40),
            ('2022-08-05', 5.40, 3.35, 4.50, 18.00, 5.90),
            ('2022-06-08', 4.90, 3.35, 4.50, 18.00, 5.40),
            ('2022-05-04', 4.40, 3.35, 4.50, 18.00, 4.90),
        ]
    
    def collect_currency_data(self, pair: str = 'USDINR', days: int = 365) -> pd.DataFrame:
        """Collect currency exchange rate data"""
        if not self.alpha_vantage_key:
            print(f"No Alpha Vantage key available for currency data")
            return pd.DataFrame()
        
        try:
            # Map to Alpha Vantage format
            from_currency = pair[:3]
            to_currency = pair[3:]
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series (FX)' not in data:
                print(f"Error in currency data response: {data}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            fx_data = data['Time Series (FX)']
            df = pd.DataFrame.from_dict(fx_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close']
            df = df.astype(float)
            
            # Filter recent data
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff_date]
            
            # Store in database
            stored_count = self._store_currency_data(pair, df)
            print(f"Collected {len(df)} {pair} rates, stored {stored_count} new records")
            
            return df
            
        except Exception as e:
            print(f"Error collecting {pair} data: {e}")
            return pd.DataFrame()
    
    def _store_currency_data(self, pair: str, df: pd.DataFrame) -> int:
        """Store currency data in database"""
        stored_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for date, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO currency_rates (date, pair, rate)
                        VALUES (?, ?, ?)
                    """, (date.strftime('%Y-%m-%d'), pair, row['close']))
                    
                    if conn.total_changes > 0:
                        stored_count += 1
                        
                except Exception as e:
                    continue
        
        return stored_count
    
    def collect_commodity_data(self, commodity: str = 'BRENTOIL', days: int = 365) -> pd.DataFrame:
        """Collect commodity price data"""
        if not self.alpha_vantage_key:
            print(f"No Alpha Vantage key available for commodity data")
            return pd.DataFrame()
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': commodity,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                print(f"Error in commodity data response for {commodity}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            commodity_data = data[time_series_key]
            df = pd.DataFrame.from_dict(commodity_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            
            # Filter recent data
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff_date]
            
            # Store in database
            stored_count = self._store_commodity_data(commodity, df)
            print(f"Collected {len(df)} {commodity} prices, stored {stored_count} new records")
            
            return df
            
        except Exception as e:
            print(f"Error collecting {commodity} data: {e}")
            return pd.DataFrame()
    
    def _store_commodity_data(self, commodity: str, df: pd.DataFrame) -> int:
        """Store commodity data in database"""
        stored_count = 0
        unit = 'USD' if commodity in ['BRENTOIL', 'NATURALGAS', 'GOLD', 'SILVER', 'COPPER'] else 'USD'
        
        with sqlite3.connect(self.db_path) as conn:
            for date, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO commodity_prices (date, commodity, price, unit)
                        VALUES (?, ?, ?, ?)
                    """, (date.strftime('%Y-%m-%d'), commodity, row['close'], unit))
                    
                    if conn.total_changes > 0:
                        stored_count += 1
                        
                except Exception as e:
                    continue
        
        return stored_count
    
    def store_rbi_rates(self):
        """Store historical RBI policy rates"""
        stored_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for date_str, repo, reverse_repo, crr, slr, bank_rate in self.rbi_historical_rates:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO rbi_rates 
                        (date, repo_rate, reverse_repo_rate, crr, slr, bank_rate)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (date_str, repo, reverse_repo, crr, slr, bank_rate))
                    
                    if conn.total_changes > 0:
                        stored_count += 1
                        
                except Exception as e:
                    continue
        
        print(f"Stored {stored_count} new RBI policy rate records")
        return stored_count
    
    def collect_global_index_data(self, symbol: str = 'SPX', days: int = 365) -> pd.DataFrame:
        """Collect global stock index data"""
        if not self.alpha_vantage_key:
            print(f"No Alpha Vantage key available for index data")
            return pd.DataFrame()
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                print(f"Error in index data response for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            index_data = data[time_series_key]
            df = pd.DataFrame.from_dict(index_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            
            # Calculate daily change percentage
            df['change_percent'] = df['close'].pct_change() * 100
            
            # Filter recent data
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff_date]
            
            # Store in database
            stored_count = self._store_index_data(symbol, df)
            print(f"Collected {len(df)} {symbol} index values, stored {stored_count} new records")
            
            return df
            
        except Exception as e:
            print(f"Error collecting {symbol} data: {e}")
            return pd.DataFrame()
    
    def _store_index_data(self, symbol: str, df: pd.DataFrame) -> int:
        """Store index data in database"""
        stored_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for date, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO global_indices (date, index_name, value, change_percent)
                        VALUES (?, ?, ?, ?)
                    """, (date.strftime('%Y-%m-%d'), symbol, row['close'], row['change_percent']))
                    
                    if conn.total_changes > 0:
                        stored_count += 1
                        
                except Exception as e:
                    continue
        
        return stored_count
    
    def collect_all_macro_data(self, days: int = 365) -> Dict:
        """Collect all macro economic data"""
        print("Collecting comprehensive macro economic data...")
        print("=" * 50)
        
        results = {
            'currency_data': {},
            'commodity_data': {},
            'index_data': {},
            'rbi_data': 0
        }
        
        # Collect currency data
        print("1. Collecting currency data...")
        for pair_code, pair_name in self.data_sources['currency_pairs'].items():
            if pair_code == 'USDINR':  # Primary focus for Indian stocks
                df = self.collect_currency_data(pair_code, days)
                results['currency_data'][pair_code] = len(df)
                time.sleep(12)  # Alpha Vantage rate limiting
        
        # Collect commodity data
        print(f"\n2. Collecting commodity data...")
        for commodity_code, commodity_name in self.data_sources['commodities'].items():
            if commodity_code in ['BRENTOIL', 'GOLD']:  # Key commodities for Indian market
                df = self.collect_commodity_data(commodity_code, days)
                results['commodity_data'][commodity_code] = len(df)
                time.sleep(12)  # Alpha Vantage rate limiting
        
        # Collect global index data
        print(f"\n3. Collecting global index data...")
        for index_code, index_name in self.data_sources['global_indices'].items():
            if index_code in ['SPX', 'IXIC']:  # Key global indices
                df = self.collect_global_index_data(index_code, days)
                results['index_data'][index_code] = len(df)
                time.sleep(12)  # Alpha Vantage rate limiting
        
        # Store RBI rates
        print(f"\n4. Storing RBI policy rates...")
        rbi_count = self.store_rbi_rates()
        results['rbi_data'] = rbi_count
        
        return results

    def update_all_macro_data(self, days: int = 7) -> str:
        """
        Efficiently updates all macro data sources for the last few days.
        Renamed from collect_all_macro_data for clarity in automation.
        """
        summary = []
        summary.append("Updating comprehensive macro economic data...")
        
        # Collect currency data for USDINR
        summary.append("-> Updating currency data...")
        self.collect_currency_data('USDINR', days)
        time.sleep(15) # Respect Alpha Vantage rate limit (5 calls per minute)
        
        # Collect key commodity data
        summary.append("-> Updating commodity data...")
        for commodity_code in ['BRENTOIL', 'WTI', 'GOLD']:
            self.collect_commodity_data(commodity_code, days)
            time.sleep(15)
        
        # Collect key global index data
        summary.append("-> Updating global index data...")
        for index_code in ['SPX', 'IXIC']: # S&P 500 and NASDAQ
            self.collect_global_index_data(index_code, days)
            time.sleep(15)
        
        # Store/update RBI rates (idempotent)
        summary.append("-> Storing RBI policy rates...")
        self.store_rbi_rates()
        
        return "\n".join(summary)
        
    
    def get_macro_features_for_stock(self, stock_symbol: str, date: str = None) -> Dict:
        """Get macro features for a specific stock and date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        features = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Get latest USD/INR rate
            cursor = conn.execute("""
                SELECT rate FROM currency_rates 
                WHERE pair = 'USDINR' AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, (date,))
            result = cursor.fetchone()
            features['usd_inr_rate'] = result[0] if result else 83.0  # Default fallback
            
            # Get latest RBI repo rate
            cursor = conn.execute("""
                SELECT repo_rate FROM rbi_rates 
                WHERE date <= ?
                ORDER BY date DESC LIMIT 1
            """, (date,))
            result = cursor.fetchone()
            features['repo_rate'] = result[0] if result else 6.5  # Default fallback
            
            # Get latest Brent oil price
            cursor = conn.execute("""
                SELECT price FROM commodity_prices 
                WHERE commodity = 'BRENTOIL' AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, (date,))
            result = cursor.fetchone()
            features['brent_oil_price'] = result[0] if result else 80.0  # Default fallback
            
            # Get latest S&P 500 level
            cursor = conn.execute("""
                SELECT value FROM global_indices 
                WHERE index_name = 'SPX' AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, (date,))
            result = cursor.fetchone()
            features['sp500_level'] = result[0] if result else 4500.0  # Default fallback
        
        # Add sector-specific features based on stock
        stock_key = stock_symbol.replace('.NS', '').replace('.', '').upper()
        
        if stock_key in ['HDFCBANK', 'ICICIBANK']:  # Banking sector
            features['interest_rate_environment'] = 1 if features['repo_rate'] > 6.0 else 0
            features['currency_stability'] = 1 if 82 <= features['usd_inr_rate'] <= 85 else 0
            
        elif stock_key == 'INFY':  # IT services
            features['usd_strength'] = features['usd_inr_rate'] / 83.0  # Normalized
            features['us_market_sentiment'] = features['sp500_level'] / 4500.0  # Normalized
            
        elif stock_key == 'TATAMOTORS':  # Automotive
            features['oil_price_impact'] = features['brent_oil_price'] / 80.0  # Normalized
            features['commodity_pressure'] = 1 if features['brent_oil_price'] > 90 else 0
            
        elif stock_key == 'RELIANCE':  # Conglomerate  
            features['oil_exposure'] = features['brent_oil_price'] / 80.0  # Normalized
            features['multi_sector_balance'] = (features['repo_rate'] + features['usd_inr_rate'] / 10) / 2
        
        return features
    
    def get_macro_summary(self) -> Dict:
        """Get summary of collected macro data"""
        with sqlite3.connect(self.db_path) as conn:
            summary = {}
            
            # Currency data summary
            cursor = conn.execute("""
                SELECT pair, COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest
                FROM currency_rates GROUP BY pair
            """)
            summary['currency_data'] = cursor.fetchall()
            
            # Commodity data summary  
            cursor = conn.execute("""
                SELECT commodity, COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest
                FROM commodity_prices GROUP BY commodity
            """)
            summary['commodity_data'] = cursor.fetchall()
            
            # Index data summary
            cursor = conn.execute("""
                SELECT index_name, COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest
                FROM global_indices GROUP BY index_name
            """)
            summary['index_data'] = cursor.fetchall()
            
            # RBI data summary
            cursor = conn.execute("""
                SELECT COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest
                FROM rbi_rates
            """)
            summary['rbi_data'] = cursor.fetchone()
        
        return summary

def test_macro_collector():
    """Test the macro economic data collector"""
    print("Testing Macro Economic Data Collector")
    print("=" * 50)
    
    collector = MacroEconomicDataCollector()
    
    # Test 1: Collect USD/INR data
    print("1. Testing USD/INR currency data...")
    usd_inr_data = collector.collect_currency_data('USDINR', days=30)
    if not usd_inr_data.empty:
        print(f"   Latest USD/INR rate: {usd_inr_data['close'].iloc[-1]:.2f}")
    
    time.sleep(12)  # Alpha Vantage rate limiting
    
    # Test 2: Collect WTI oil data
    print(f"\n2. Testing WTI oil commodity data...")
    oil_data = collector.collect_commodity_data('WTI', days=30)
    if not oil_data.empty:
        print(f"   Latest WTI oil price: ${oil_data['close'].iloc[-1]:.2f}")
    
    # Test 3: Store RBI rates
    print(f"\n3. Testing RBI rates storage...")
    rbi_count = collector.store_rbi_rates()
    print(f"   RBI rates stored: {rbi_count} records")
    
    # Test 4: Get macro features for HDFC Bank
    print(f"\n4. Testing macro features for HDFCBANK.NS...")
    features = collector.get_macro_features_for_stock('HDFCBANK.NS')
    print(f"   Generated features:")
    for feature, value in features.items():
        print(f"     {feature}: {value:.3f}")
    
    # Test 5: Get data summary
    print(f"\n5. Testing data summary...")
    summary = collector.get_macro_summary()
    print(f"   Macro data summary:")
    if summary.get('currency_data'):
        for pair, count, earliest, latest in summary['currency_data']:
            print(f"     {pair}: {count} records ({earliest} to {latest})")

if __name__ == "__main__":
    test_macro_collector()