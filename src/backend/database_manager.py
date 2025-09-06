"""
Database Manager for Event-Driven Stock Trend Predictor
Handles PostgreSQL operations for stock data storage and retrieval
Optimized for time-series data with proper indexing
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import yaml
from pathlib import Path
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages PostgreSQL database operations for stock market data
    Includes caching with Redis for frequently accessed data
    """
    
    def __init__(self, config_path: str = "config/database_config.yaml"):
        """Initialize database manager"""
        self.config = self._load_config(config_path)
        self.db_config = self.config['postgresql']
        self.redis_config = self.config.get('redis', {})
        
        # Database connection
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis connection for caching
        self.redis_client = self._setup_redis()
        
        # Initialize database tables
        self._create_tables()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load database configuration"""
        # Default configuration if file doesn't exist
        default_config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'database': 'stock_predictor',
                'username': 'postgres',
                'password': 'password'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    # Merge with defaults
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                    return config
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return default_config
    
    def _create_engine(self):
        """Create SQLAlchemy engine"""
        db_url = (f"postgresql://{self.db_config['username']}:"
                 f"{self.db_config['password']}@{self.db_config['host']}:"
                 f"{self.db_config['port']}/{self.db_config['database']}")
        
        engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        
        # Test connection
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Please ensure PostgreSQL is running and credentials are correct")
        
        return engine
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """Setup Redis connection for caching"""
        try:
            client = redis.Redis(
                host=self.redis_config.get('host', 'localhost'),
                port=self.redis_config.get('port', 6379),
                db=self.redis_config.get('db', 0),
                decode_responses=True
            )
            client.ping()
            logger.info("Redis connection established")
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            return None
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        
        # Stock master table
        stock_master_sql = """
        CREATE TABLE IF NOT EXISTS stock_master (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            sector VARCHAR(50),
            exchange VARCHAR(10),
            currency VARCHAR(5),
            adr_symbol VARCHAR(20),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_stock_master_symbol ON stock_master(symbol);
        CREATE INDEX IF NOT EXISTS idx_stock_master_sector ON stock_master(sector);
        """
        
        # Daily price data table (time-series optimized)
        price_data_sql = """
        CREATE TABLE IF NOT EXISTS daily_prices (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(12,4) NOT NULL,
            high DECIMAL(12,4) NOT NULL,
            low DECIMAL(12,4) NOT NULL,
            close DECIMAL(12,4) NOT NULL,
            adj_close DECIMAL(12,4),
            volume BIGINT,
            market_type VARCHAR(10) DEFAULT 'indian',
            source VARCHAR(20) DEFAULT 'yfinance',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date, market_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol_date ON daily_prices(symbol, date DESC);
        CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date DESC);
        CREATE INDEX IF NOT EXISTS idx_daily_prices_market_type ON daily_prices(market_type);
        """
        
        # Technical indicators table
        technical_indicators_sql = """
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            sma_5 DECIMAL(12,4),
            sma_10 DECIMAL(12,4),
            sma_20 DECIMAL(12,4),
            sma_50 DECIMAL(12,4),
            sma_200 DECIMAL(12,4),
            ema_12 DECIMAL(12,4),
            ema_26 DECIMAL(12,4),
            rsi_14 DECIMAL(8,4),
            macd DECIMAL(12,6),
            macd_signal DECIMAL(12,6),
            macd_histogram DECIMAL(12,6),
            bollinger_upper DECIMAL(12,4),
            bollinger_lower DECIMAL(12,4),
            bollinger_middle DECIMAL(12,4),
            volume_sma_20 BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date ON technical_indicators(symbol, date DESC);
        """
        
        # News sentiment table
        news_sentiment_sql = """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            headline TEXT NOT NULL,
            content TEXT,
            sentiment_score DECIMAL(5,4),
            sentiment_label VARCHAR(20),
            source VARCHAR(50),
            url TEXT,
            published_at TIMESTAMP,
            relevance_score DECIMAL(5,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_date ON news_sentiment(symbol, date DESC);
        CREATE INDEX IF NOT EXISTS idx_news_sentiment_date ON news_sentiment(date DESC);
        CREATE INDEX IF NOT EXISTS idx_news_sentiment_score ON news_sentiment(sentiment_score);
        """
        
        # Model predictions table
        model_predictions_sql = """
        CREATE TABLE IF NOT EXISTS model_predictions (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            prediction_date DATE NOT NULL,
            target_date DATE NOT NULL,
            horizon VARCHAR(10) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            trend_prediction VARCHAR(20) NOT NULL,
            confidence_score DECIMAL(5,4) NOT NULL,
            price_prediction DECIMAL(12,4),
            actual_trend VARCHAR(20),
            actual_price DECIMAL(12,4),
            is_correct BOOLEAN,
            features_used JSONB,
            model_metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON model_predictions(symbol, prediction_date DESC);
        CREATE INDEX IF NOT EXISTS idx_predictions_target ON model_predictions(target_date);
        CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON model_predictions(horizon);
        """
        
        # Macro economic data table
        macro_data_sql = """
        CREATE TABLE IF NOT EXISTS macro_data (
            id BIGSERIAL PRIMARY KEY,
            indicator VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            value DECIMAL(15,6) NOT NULL,
            unit VARCHAR(20),
            source VARCHAR(50),
            frequency VARCHAR(20) DEFAULT 'daily',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(indicator, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_macro_data_indicator_date ON macro_data(indicator, date DESC);
        CREATE INDEX IF NOT EXISTS idx_macro_data_date ON macro_data(date DESC);
        """
        
        # Model training logs table
        training_logs_sql = """
        CREATE TABLE IF NOT EXISTS training_logs (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            training_start TIMESTAMP NOT NULL,
            training_end TIMESTAMP,
            status VARCHAR(20) DEFAULT 'running',
            accuracy DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall SCORE DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            loss DECIMAL(10,6),
            hyperparameters JSONB,
            training_data_size INTEGER,
            validation_data_size INTEGER,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_training_logs_symbol ON training_logs(symbol);
        CREATE INDEX IF NOT EXISTS idx_training_logs_status ON training_logs(status);
        CREATE INDEX IF NOT EXISTS idx_training_logs_date ON training_logs(created_at DESC);
        """
        
        # Execute all table creation queries
        try:
            with self.engine.connect() as conn:
                conn.execute(text(stock_master_sql))
                conn.execute(text(price_data_sql))
                conn.execute(text(technical_indicators_sql))
                conn.execute(text(news_sentiment_sql))
                conn.execute(text(model_predictions_sql))
                conn.execute(text(macro_data_sql))
                conn.execute(text(training_logs_sql))
                conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def insert_stock_master(self, stock_data: Dict):
        """Insert or update stock master data"""
        sql = """
        INSERT INTO stock_master (symbol, name, sector, exchange, currency, adr_symbol)
        VALUES (%(symbol)s, %(name)s, %(sector)s, %(exchange)s, %(currency)s, %(adr_symbol)s)
        ON CONFLICT (symbol) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            sector = EXCLUDED.sector,
            exchange = EXCLUDED.exchange,
            currency = EXCLUDED.currency,
            adr_symbol = EXCLUDED.adr_symbol,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(sql), stock_data)
                conn.commit()
            logger.info(f"Stock master updated for {stock_data['symbol']}")
        except Exception as e:
            logger.error(f"Error inserting stock master: {e}")
    
    def insert_price_data(self, df: pd.DataFrame, symbol: str, market_type: str = 'indian'):
        """
        Insert price data from DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            market_type: 'indian' or 'adr'
        """
        if df.empty:
            logger.warning(f"No data to insert for {symbol}")
            return
        
        # Prepare data
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        df_copy['market_type'] = market_type
        df_copy['date'] = df_copy.index.date
        
        # Select required columns
        columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'market_type']
        df_insert = df_copy[columns].copy()
        
        # Handle missing adj_close
        if 'adj_close' in df_copy.columns:
            df_insert['adj_close'] = df_copy['adj_close']
        else:
            df_insert['adj_close'] = df_copy['close']
        
        try:
            # Use pandas to_sql with upsert behavior
            df_insert.to_sql(
                'daily_prices', 
                self.engine, 
                if_exists='append', 
                index=False,
                method='multi'
            )
            logger.info(f"Inserted {len(df_insert)} price records for {symbol} ({market_type})")
            
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                logger.info(f"Price data already exists for {symbol} ({market_type}), updating...")
                self._update_price_data(df_insert, symbol, market_type)
            else:
                logger.error(f"Error inserting price data: {e}")
    
    def _update_price_data(self, df: pd.DataFrame, symbol: str, market_type: str):
        """Update existing price data"""
        sql = """
        INSERT INTO daily_prices (symbol, date, open, high, low, close, adj_close, volume, market_type)
        VALUES (%(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(adj_close)s, %(volume)s, %(market_type)s)
        ON CONFLICT (symbol, date, market_type)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            created_at = CURRENT_TIMESTAMP
        """
        
        try:
            records = df.to_dict('records')
            with self.engine.connect() as conn:
                conn.execute(text(sql), records)
                conn.commit()
            logger.info(f"Updated {len(records)} price records for {symbol}")
        except Exception as e:
            logger.error(f"Error updating price data: {e}")
    
    def get_price_data(self, 
                      symbol: str, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      market_type: str = 'indian') -> pd.DataFrame:
        """
        Retrieve price data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            market_type: 'indian', 'adr', or 'both'
        
        Returns:
            DataFrame with price data
        """
        # Check cache first
        cache_key = f"price_data:{symbol}:{market_type}:{start_date}:{end_date}"
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)
            except:
                pass
        
        # Build query
        sql = "SELECT * FROM daily_prices WHERE symbol = %(symbol)s"
        params = {'symbol': symbol}
        
        if market_type != 'both':
            sql += " AND market_type = %(market_type)s"
            params['market_type'] = market_type
        
        if start_date:
            sql += " AND date >= %(start_date)s"
            params['start_date'] = start_date
        
        if end_date:
            sql += " AND date <= %(end_date)s"
            params['end_date'] = end_date
        
        sql += " ORDER BY date ASC"
        
        try:
            df = pd.read_sql(sql, self.engine, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Cache the result
                if self.redis_client:
                    try:
                        self.redis_client.setex(cache_key, 900, df.to_json())  # 15 min cache
                    except:
                        pass
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving price data: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, symbol: str, market_type: str = 'indian') -> Optional[Dict]:
        """Get the latest price for a symbol"""
        sql = """
        SELECT * FROM daily_prices 
        WHERE symbol = %(symbol)s AND market_type = %(market_type)s 
        ORDER BY date DESC 
        LIMIT 1
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), {'symbol': symbol, 'market_type': market_type})
                row = result.fetchone()
                
                if row:
                    return {
                        'symbol': row.symbol,
                        'date': row.date,
                        'open': float(row.open),
                        'high': float(row.high),
                        'low': float(row.low),
                        'close': float(row.close),
                        'volume': row.volume,
                        'market_type': row.market_type
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None
    
    def insert_model_prediction(self, prediction_data: Dict):
        """Insert model prediction"""
        sql = """
        INSERT INTO model_predictions 
        (symbol, prediction_date, target_date, horizon, model_version, 
         trend_prediction, confidence_score, price_prediction, features_used, model_metadata)
        VALUES 
        (%(symbol)s, %(prediction_date)s, %(target_date)s, %(horizon)s, %(model_version)s,
         %(trend_prediction)s, %(confidence_score)s, %(price_prediction)s, %(features_used)s, %(model_metadata)s)
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(sql), prediction_data)
                conn.commit()
            logger.info(f"Inserted prediction for {prediction_data['symbol']}")
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
    
    def log_training_session(self, training_data: Dict):
        """Log model training session"""
        sql = """
        INSERT INTO training_logs 
        (symbol, model_type, model_version, training_start, training_end, status,
         accuracy, precision_score, recall_score, f1_score, loss, hyperparameters,
         training_data_size, validation_data_size, error_message)
        VALUES 
        (%(symbol)s, %(model_type)s, %(model_version)s, %(training_start)s, %(training_end)s, %(status)s,
         %(accuracy)s, %(precision_score)s, %(recall_score)s, %(f1_score)s, %(loss)s, %(hyperparameters)s,
         %(training_data_size)s, %(validation_data_size)s, %(error_message)s)
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(sql), training_data)
                conn.commit()
            logger.info(f"Logged training session for {training_data['symbol']}")
        except Exception as e:
            logger.error(f"Error logging training session: {e}")
    
    def get_data_summary(self) -> Dict:
        """Get summary of data in the database"""
        queries = {
            'stocks': "SELECT COUNT(*) FROM stock_master WHERE is_active = true",
            'price_records': "SELECT COUNT(*) FROM daily_prices",
            'predictions': "SELECT COUNT(*) FROM model_predictions",
            'training_sessions': "SELECT COUNT(*) FROM training_logs",
            'latest_data_date': "SELECT MAX(date) FROM daily_prices"
        }
        
        summary = {}
        try:
            with self.engine.connect() as conn:
                for key, sql in queries.items():
                    result = conn.execute(text(sql))
                    summary[key] = result.scalar()
            
            return summary
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old prediction and log data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleanup_queries = [
            f"DELETE FROM model_predictions WHERE created_at < '{cutoff_date}'",
            f"DELETE FROM training_logs WHERE created_at < '{cutoff_date}'"
        ]
        
        try:
            with self.engine.connect() as conn:
                for sql in cleanup_queries:
                    result = conn.execute(text(sql))
                    logger.info(f"Cleaned up {result.rowcount} old records")
                conn.commit()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Test the database manager"""
    try:
        # Initialize database manager
        db = DatabaseManager()
        
        # Test database connection
        summary = db.get_data_summary()
        logger.info(f"Database summary: {summary}")
        
        # Test stock master insertion
        test_stock = {
            'symbol': 'HDFCBANK.NS',
            'name': 'HDFC Bank Limited',
            'sector': 'banking',
            'exchange': 'NSE',
            'currency': 'INR',
            'adr_symbol': 'HDB'
        }
        
        db.insert_stock_master(test_stock)
        logger.info("Database manager test completed successfully!")
        
    except Exception as e:
        logger.error(f"Database manager test failed: {e}")

if __name__ == "__main__":
    main()