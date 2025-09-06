"""
Technical Indicators Calculator for Stock Trend Prediction
Calculates 50+ technical indicators optimized for Indian banking sector
"""

import pandas as pd
import numpy as np
import talib
from sqlalchemy import create_engine
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicatorCalculator:
    def __init__(self, db_url: str = "postgresql://postgres:shreshth@localhost:5432/stock_predictor"):
        """Initialize with database connection"""
        self.engine = create_engine(db_url)
        
        # Banking sector specific parameters
        self.banking_params = {
            'rsi_periods': [14, 21, 30],  # Banking stocks respond well to longer RSI
            'ma_periods': [5, 10, 20, 50, 100, 200],  # Key moving averages
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'volume_sma': 20
        }
    
    def load_stock_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Load stock data from database"""
        query = """
        SELECT date, open, high, low, close, volume, adj_close
        FROM daily_prices 
        WHERE symbol = %(symbol)s 
        ORDER BY date DESC 
        LIMIT %(days)s
        """
        
        df = pd.read_sql(query, self.engine, params={'symbol': symbol, 'days': days})
        df = df.sort_values('date').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure we have required columns as float
        price_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        result = df.copy()
        
        # Moving Averages
        for period in self.banking_params['ma_periods']:
            result[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
            result[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'],
            fastperiod=self.banking_params['macd_fast'],
            slowperiod=self.banking_params['macd_slow'],
            signalperiod=self.banking_params['macd_signal']
        )
        result['MACD'] = macd
        result['MACD_Signal'] = macd_signal
        result['MACD_Histogram'] = macd_hist
        
        # Parabolic SAR
        result['PSAR'] = talib.SAR(df['high'], df['low'])
        
        # Trend strength
        result['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        return result
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum oscillators"""
        result = df.copy()
        
        # RSI for multiple periods (banking sector optimization)
        for period in self.banking_params['rsi_periods']:
            result[f'RSI_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        result['STOCH_K'] = slowk
        result['STOCH_D'] = slowd
        
        # Williams %R
        result['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Commodity Channel Index
        result['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Rate of Change
        result['ROC_10'] = talib.ROC(df['close'], timeperiod=10)
        result['ROC_20'] = talib.ROC(df['close'], timeperiod=20)
        
        return result
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and support/resistance indicators"""
        result = df.copy()
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['close'],
            timeperiod=self.banking_params['bollinger_period'],
            nbdevup=self.banking_params['bollinger_std'],
            nbdevdn=self.banking_params['bollinger_std']
        )
        result['BB_Upper'] = bb_upper
        result['BB_Middle'] = bb_middle
        result['BB_Lower'] = bb_lower
        result['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        result['BB_Position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range
        result['ATR'] = talib.ATR(df['high'], df['low'], df['close'], 
                                timeperiod=self.banking_params['atr_period'])
        
        # True Range
        result['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])
        
        # Normalized ATR (for cross-stock comparison)
        result['ATR_Normalized'] = result['ATR'] / df['close']
        
        return result
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        result = df.copy()
        
        # Volume moving averages
        result['Volume_SMA'] = talib.SMA(df['volume'], timeperiod=self.banking_params['volume_sma'])
        result['Volume_Ratio'] = df['volume'] / result['Volume_SMA']
        
        # On Balance Volume
        result['OBV'] = talib.OBV(df['close'], df['volume'])
        
        # Volume Price Trend
        result['VPT'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        result['VPT'] = result['VPT'].cumsum()
        
        # Chaikin Money Flow
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        result['CMF'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volume-Weighted Average Price (VWAP)
        result['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return result
    
    def calculate_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick patterns and price patterns"""
        result = df.copy()
        
        # Key candlestick patterns for banking stocks
        result['DOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        result['HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        result['ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        result['HARAMI'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        result['MORNING_STAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        result['EVENING_STAR'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Price patterns
        result['Higher_High'] = (df['high'] > df['high'].shift(1)).astype(int)
        result['Lower_Low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Gap analysis
        result['Gap_Up'] = ((df['low'] > df['high'].shift(1))).astype(int)
        result['Gap_Down'] = ((df['high'] < df['low'].shift(1))).astype(int)
        
        return result
    
    def calculate_banking_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate banking sector-specific features"""
        result = df.copy()
        
        # Banking stocks correlation with interest rate sensitivity
        # Price momentum in different timeframes
        result['Price_Change_1D'] = df['close'].pct_change(1)
        result['Price_Change_5D'] = df['close'].pct_change(5)
        result['Price_Change_20D'] = df['close'].pct_change(20)
        
        # Volatility clustering (important for banking stocks)
        result['Volatility_5D'] = result['Price_Change_1D'].rolling(5).std()
        result['Volatility_20D'] = result['Price_Change_1D'].rolling(20).std()
        
        # Support and resistance levels
        result['High_52W'] = df['high'].rolling(252).max()
        result['Low_52W'] = df['low'].rolling(252).min()
        result['Position_52W'] = (df['close'] - result['Low_52W']) / (result['High_52W'] - result['Low_52W'])
        
        # Banking sector momentum
        result['Price_vs_SMA50'] = (df['close'] / result['SMA_50']) - 1
        result['Price_vs_SMA200'] = (df['close'] / result['SMA_200']) - 1
        
        # Volume-price divergence (key for banking stock breakouts)
        price_trend = df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        volume_trend = df['volume'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        result['Volume_Price_Divergence'] = np.where(
            (price_trend > 0) & (volume_trend < 0), -1,  # Price up, volume down
            np.where((price_trend < 0) & (volume_trend > 0), 1, 0)  # Price down, volume up
        )
        
        return result
    
    def calculate_all_indicators(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Calculate all technical indicators for a stock"""
        try:
            # Load data
            df = self.load_stock_data(symbol, days)
            
            if len(df) < 50:
                raise ValueError(f"Insufficient data for {symbol}: only {len(df)} records")
            
            print(f"Loaded {len(df)} days of data for {symbol}")
            
            # Calculate all indicator categories
            df = self.calculate_trend_indicators(df)
            df = self.calculate_momentum_indicators(df)
            df = self.calculate_volatility_indicators(df)
            df = self.calculate_volume_indicators(df)
            df = self.calculate_pattern_indicators(df)
            df = self.calculate_banking_sector_features(df)
            
            # Clean data - remove rows with too many NaN values
            df = df.dropna(thresh=len(df.columns) * 0.7)  # Keep rows with at least 70% non-NaN values
            
            # Fill remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"Generated {len(df.columns)} features for {len(df)} trading days")
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators for {symbol}: {e}")
            raise
    
    def get_latest_signals(self, symbol: str) -> Dict:
        """Get latest trading signals"""
        try:
            df = self.calculate_all_indicators(symbol, days=100)
            latest = df.iloc[-1]
            
            signals = {
                'symbol': symbol,
                'date': latest['date'],
                'price': latest['close'],
                'signals': {
                    'rsi_14': 'overbought' if latest['RSI_14'] > 70 else 'oversold' if latest['RSI_14'] < 30 else 'neutral',
                    'macd': 'bullish' if latest['MACD'] > latest['MACD_Signal'] else 'bearish',
                    'bb_position': 'upper' if latest['BB_Position'] > 0.8 else 'lower' if latest['BB_Position'] < 0.2 else 'middle',
                    'trend_strength': 'strong' if latest['ADX'] > 25 else 'weak',
                    'volume': 'high' if latest['Volume_Ratio'] > 1.5 else 'low' if latest['Volume_Ratio'] < 0.8 else 'normal'
                },
                'key_levels': {
                    'support': latest['BB_Lower'],
                    'resistance': latest['BB_Upper'],
                    'sma_50': latest['SMA_50'],
                    'sma_200': latest['SMA_200']
                }
            }
            
            return signals
            
        except Exception as e:
            print(f"Error getting signals for {symbol}: {e}")
            return {}

def main():
    """Test the technical indicators calculator"""
    calc = TechnicalIndicatorCalculator()
    
    # Test with HDFC Bank
    try:
        features = calc.calculate_all_indicators('HDFCBANK.NS', days=100)
        print(f"\n✅ SUCCESS: Generated {len(features.columns)} features for {len(features)} days")
        print("\nSample features (first 10 columns):")
        print(features[features.columns[:10]].head())
        
        # Get latest signals
        signals = calc.get_latest_signals('HDFCBANK.NS')
        print(f"\n📊 Latest Signals for HDFCBANK.NS:")
        print(f"Price: ₹{signals.get('price', 0):.2f}")
        print(f"RSI Signal: {signals.get('signals', {}).get('rsi_14', 'N/A')}")
        print(f"MACD Signal: {signals.get('signals', {}).get('macd', 'N/A')}")
        print(f"Trend Strength: {signals.get('signals', {}).get('trend_strength', 'N/A')}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()