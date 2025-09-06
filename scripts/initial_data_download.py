"""
Initial Data Download Script for Event-Driven Stock Trend Predictor
Downloads historical data for all 5 target stocks and populates database
Run this once to set up your initial dataset
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import yaml

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.market_collector import MarketDataCollector
from backend.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/initial_data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InitialDataDownloader:
    """
    Orchestrates the initial data download and database setup process
    """
    
    def __init__(self):
        """Initialize the downloader"""
        self.setup_directories()
        self.collector = MarketDataCollector()
        self.db_manager = DatabaseManager()
        self.stock_config = self.collector.config['stocks']
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw/market_data',
            'data/processed',
            'models',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info("Directories created/verified")
    
    def populate_stock_master(self):
        """Populate stock master table with configuration data"""
        logger.info("Populating stock master table...")
        
        for stock_code, config in self.stock_config.items():
            stock_data = {
                'symbol': config['indian_symbol'],
                'name': config['name'],
                'sector': config['sector'],
                'exchange': config.get('exchange', 'NSE'),
                'currency': config.get('currency', 'INR'),
                'adr_symbol': config.get('adr_symbol')
            }
            
            self.db_manager.insert_stock_master(stock_data)
            
            # Also insert ADR if exists
            if config.get('adr_symbol'):
                adr_data = {
                    'symbol': config['adr_symbol'],
                    'name': f"{config['name']} ADR",
                    'sector': config['sector'],
                    'exchange': 'NYSE/NASDAQ',
                    'currency': 'USD',
                    'adr_symbol': None
                }
                self.db_manager.insert_stock_master(adr_data)
        
        logger.info("Stock master table populated")
    
    def download_and_store_data(self):
        """Download historical data for all stocks and store in database"""
        logger.info("Starting data download for all stocks...")
        
        total_records = 0
        success_count = 0
        
        for stock_name in self.stock_config.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {stock_name}")
            logger.info(f"{'='*60}")
            
            try:
                # Collect data from both markets
                stock_data = self.collector.collect_dual_market_data(stock_name)
                
                if not stock_data:
                    logger.warning(f"No data collected for {stock_name}")
                    continue
                
                # Process each market type
                for market_type, df in stock_data.items():
                    if df.empty:
                        logger.warning(f"Empty dataset for {stock_name} {market_type}")
                        continue
                    
                    # Determine symbol for database
                    if market_type == 'indian':
                        symbol = self.stock_config[stock_name]['indian_symbol']
                    else:  # adr
                        symbol = self.stock_config[stock_name]['adr_symbol']
                    
                    # Validate data quality
                    quality = self.collector.validate_data_quality(df)
                    logger.info(f"Data quality for {symbol}: {quality['quality_score']}/100")
                    
                    if quality['quality_score'] < 50:
                        logger.warning(f"Low quality data for {symbol}, skipping")
                        continue
                    
                    # Store in database
                    self.db_manager.insert_price_data(df, symbol, market_type)
                    
                    # Save to CSV files as backup
                    self.collector.save_data(df, stock_name, market_type)
                    
                    total_records += len(df)
                    logger.info(f"Stored {len(df)} records for {symbol} ({market_type})")
                
                success_count += 1
                logger.info(f"✅ {stock_name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing {stock_name}: {e}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA DOWNLOAD SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully processed: {success_count}/{len(self.stock_config)} stocks")
        logger.info(f"Total records downloaded: {total_records:,}")
        
        return success_count, total_records
    
    def verify_data_integrity(self):
        """Verify the downloaded data integrity"""
        logger.info("Verifying data integrity...")
        
        summary = self.db_manager.get_data_summary()
        logger.info(f"Database summary: {summary}")
        
        # Check each stock
        for stock_name, config in self.stock_config.items():
            indian_symbol = config['indian_symbol']
            
            # Check Indian market data
            indian_data = self.db_manager.get_price_data(indian_symbol, market_type='indian')
            
            if not indian_data.empty:
                logger.info(f"✅ {indian_symbol}: {len(indian_data)} records "
                          f"({indian_data.index.min().date()} to {indian_data.index.max().date()})")
                
                # Basic data quality checks
                if indian_data['close'].isnull().sum() > 0:
                    logger.warning(f"⚠️  {indian_symbol}: Has missing close prices")
                
                if (indian_data['close'] <= 0).sum() > 0:
                    logger.warning(f"⚠️  {indian_symbol}: Has zero/negative prices")
                
            else:
                logger.error(f"❌ {indian_symbol}: No data found in database")
            
            # Check ADR data if available
            if config.get('adr_symbol'):
                adr_symbol = config['adr_symbol']
                adr_data = self.db_manager.get_price_data(adr_symbol, market_type='adr')
                
                if not adr_data.empty:
                    logger.info(f"✅ {adr_symbol}: {len(adr_data)} records "
                              f"({adr_data.index.min().date()} to {adr_data.index.max().date()})")
                else:
                    logger.warning(f"⚠️  {adr_symbol}: No ADR data found")
    
    def generate_data_report(self):
        """Generate a comprehensive data report"""
        logger.info("Generating data report...")
        
        report = {
            'download_date': datetime.now().isoformat(),
            'stocks': {},
            'summary': self.db_manager.get_data_summary()
        }
        
        for stock_name, config in self.stock_config.items():
            stock_report = {
                'indian_symbol': config['indian_symbol'],
                'adr_symbol': config.get('adr_symbol'),
                'sector': config['sector'],
                'data': {}
            }
            
            # Indian market data stats
            indian_data = self.db_manager.get_price_data(
                config['indian_symbol'], 
                market_type='indian'
            )
            
            if not indian_data.empty:
                stock_report['data']['indian'] = {
                    'records': len(indian_data),
                    'date_range': f"{indian_data.index.min().date()} to {indian_data.index.max().date()}",
                    'missing_days': indian_data.isnull().sum().sum(),
                    'avg_volume': int(indian_data['volume'].mean()) if 'volume' in indian_data else 0,
                    'price_range': f"₹{indian_data['close'].min():.2f} - ₹{indian_data['close'].max():.2f}"
                }
            
            # ADR data stats
            if config.get('adr_symbol'):
                adr_data = self.db_manager.get_price_data(
                    config['adr_symbol'], 
                    market_type='adr'
                )
                
                if not adr_data.empty:
                    stock_report['data']['adr'] = {
                        'records': len(adr_data),
                        'date_range': f"{adr_data.index.min().date()} to {adr_data.index.max().date()}",
                        'missing_days': adr_data.isnull().sum().sum(),
                        'avg_volume': int(adr_data['volume'].mean()) if 'volume' in adr_data else 0,
                        'price_range': f"${adr_data['close'].min():.2f} - ${adr_data['close'].max():.2f}"
                    }
            
            report['stocks'][stock_name] = stock_report
        
        # Save report
        report_file = Path('data') / 'initial_data_report.yaml'
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False, indent=2)
        
        logger.info(f"Data report saved to {report_file}")
        return report
    
    def run_complete_setup(self):
        """Run the complete initial data setup process"""
        logger.info("🚀 Starting complete initial data setup...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Populate stock master
            self.populate_stock_master()
            
            # Step 2: Download and store all data
            success_count, total_records = self.download_and_store_data()
            
            # Step 3: Verify data integrity
            self.verify_data_integrity()
            
            # Step 4: Generate report
            report = self.generate_data_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"\n🎉 INITIAL DATA SETUP COMPLETED!")
            logger.info(f"⏱️  Duration: {duration}")
            logger.info(f"📊 Stocks processed: {success_count}/{len(self.stock_config)}")
            logger.info(f"📈 Total records: {total_records:,}")
            logger.info(f"📋 Report saved to: data/initial_data_report.yaml")
            
            if success_count == len(self.stock_config):
                logger.info("✅ All stocks processed successfully!")
                logger.info("🚀 Ready to start model training!")
            else:
                logger.warning("⚠️  Some stocks failed to process. Check logs for details.")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initial data setup failed: {e}")
            return False

def main():
    """Main function"""
    print("="*80)
    print("EVENT-DRIVEN STOCK TREND PREDICTOR - INITIAL DATA SETUP")
    print("="*80)
    print("This script will:")
    print("1. Set up database tables")
    print("2. Download historical data for 5 target stocks")
    print("3. Validate data quality")
    print("4. Generate data report")
    print("\nThis process may take 10-15 minutes...")
    print("="*80)
    
    # Confirm before proceeding
    response = input("Proceed with data download? (y/N): ").lower()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    try:
        downloader = InitialDataDownloader()
        success = downloader.run_complete_setup()
        
        if success:
            print("\n🎉 Setup completed successfully!")
            print("Next steps:")
            print("1. Review the data report in data/initial_data_report.yaml")
            print("2. Start the backend server: python main.py")
            print("3. Start the frontend: cd frontend && npm run dev")
        else:
            print("\n❌ Setup failed. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        print(f"\n❌ Setup failed: {e}")

if __name__ == "__main__":
    main()