"""
Model Training Controller
Orchestrates the complete training pipeline with real-time progress tracking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Callable, Any
import asyncio
import json
from pathlib import Path
import traceback

# Database and data components
from src.backend.database_manager import DatabaseManager
from src.features.technical_indicators import TechnicalIndicators, get_ml_ready_features
from src.models.sector_lstm import StockLSTMPredictor, create_hdfc_lstm_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingController:
    """
    Centralized controller for model training with progress tracking
    Handles the complete pipeline from data loading to model deployment
    """
    
    def __init__(self, db_config_path: str = "config/database_config.yaml"):
        self.db_manager = DatabaseManager(db_config_path)
        self.indicators_calculator = TechnicalIndicators()
        self.current_stock = None
        self.current_model = None
        self.training_progress = {
            'status': 'idle',
            'current_step': '',
            'progress_percentage': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'metrics': {},
            'error': None,
            'start_time': None,
            'estimated_completion': None
        }
        
        # Ensure models directory exists
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    async def get_stock_data_with_features(self, stock_symbol: str, 
                                         start_date: Optional[str] = None,
                                         end_date: Optional[str] = None) -> pd.DataFrame:
        """Load stock data and calculate technical indicators"""
        logger.info(f"Loading data for {stock_symbol}")
        
        # Get stock data from database
        stock_data = self.db_manager.get_stock_data(
            stock_symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        if stock_data.empty:
            raise ValueError(f"No data found for {stock_symbol}")
        
        logger.info(f"Loaded {len(stock_data)} records for {stock_symbol}")
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        df_with_indicators = self.indicators_calculator.calculate_all_indicators(stock_data)
        
        # Prepare ML features
        logger.info("Preparing ML features...")
        X, y = get_ml_ready_features(df_with_indicators)
        
        # Combine features with target
        df_final = df_with_indicators.copy()
        df_final['target'] = y
        
        # Remove rows with NaN targets
        df_final = df_final.dropna(subset=['target'])
        
        logger.info(f"Final dataset: {len(df_final)} records with {len(X.columns)} features")
        
        return df_final, list(X.columns)
    
    def update_progress(self, update: Dict[str, Any]):
        """Update training progress"""
        self.training_progress.update(update)
        
        # Calculate estimated completion time
        if (self.training_progress['current_epoch'] > 0 and 
            self.training_progress['total_epochs'] > 0 and
            self.training_progress['start_time']):
            
            elapsed = (datetime.now() - self.training_progress['start_time']).total_seconds()
            epochs_remaining = (self.training_progress['total_epochs'] - 
                              self.training_progress['current_epoch'])
            
            if self.training_progress['current_epoch'] > 0:
                time_per_epoch = elapsed / self.training_progress['current_epoch']
                estimated_remaining = time_per_epoch * epochs_remaining
                self.training_progress['estimated_completion'] = (
                    datetime.now() + timedelta(seconds=estimated_remaining)
                ).isoformat()
    
    def progress_callback(self, progress_info: Dict[str, Any]):
        """Progress callback for model training"""
        self.update_progress({
            'current_epoch': progress_info['current_epoch'],
            'total_epochs': progress_info['total_epochs'],
            'progress_percentage': progress_info['progress_percentage'],
            'metrics': {
                'train_loss': progress_info.get('train_loss', 0),
                'val_accuracy': progress_info.get('val_accuracy', 0)
            },
            'status': progress_info['status']
        })
        
        logger.info(f"Training progress: {progress_info['progress_percentage']:.1f}% - "
                   f"Epoch {progress_info['current_epoch']}/{progress_info['total_epochs']}")
    
    async def train_stock_model(self, stock_symbol: str, 
                               retrain: bool = False,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Train LSTM model for a specific stock
        Args:
            stock_symbol: Stock symbol to train (e.g., 'HDFCBANK')
            retrain: Whether to retrain even if model exists
            start_date: Start date for training data
            end_date: End date for training data
        """
        
        try:
            # Initialize progress tracking
            self.training_progress = {
                'status': 'initializing',
                'current_step': 'Loading data',
                'progress_percentage': 0,
                'current_epoch': 0,
                'total_epochs': 0,
                'metrics': {},
                'error': None,
                'start_time': datetime.now(),
                'estimated_completion': None,
                'stock_symbol': stock_symbol
            }
            
            # Check if model already exists
            model_path = self.models_dir / f"{stock_symbol.lower()}_lstm"
            if model_path.exists() and not retrain:
                logger.info(f"Model for {stock_symbol} already exists. Use retrain=True to retrain.")
                self.update_progress({
                    'status': 'completed',
                    'current_step': 'Model already exists',
                    'progress_percentage': 100
                })
                return {'status': 'skipped', 'message': 'Model already exists'}
            
            # Step 1: Load and prepare data (20% progress)
            self.update_progress({
                'status': 'loading_data',
                'current_step': f'Loading {stock_symbol} data and calculating indicators',
                'progress_percentage': 5
            })
            
            df_with_features, feature_columns = await self.get_stock_data_with_features(
                stock_symbol, start_date, end_date
            )
            
            self.update_progress({
                'current_step': 'Data loaded, initializing model',
                'progress_percentage': 20
            })
            
            # Step 2: Initialize model based on stock sector (25% progress)
            if stock_symbol.upper() in ['HDFCBANK', 'ICICIBANK']:
                config = create_hdfc_lstm_config()  # Banking-optimized config
                config.sector = "banking"
            elif stock_symbol.upper() == 'INFY':
                config = create_hdfc_lstm_config()
                config.sector = "it_services"
                config.macro_weight = 0.4  # IT less macro-sensitive
                config.technical_weight = 0.6
            elif stock_symbol.upper() == 'TATAMOTORS':
                config = create_hdfc_lstm_config()
                config.sector = "automotive"
                config.macro_weight = 0.8  # Auto very commodity-sensitive
                config.technical_weight = 0.2
            else:  # RELIANCE or others
                config = create_hdfc_lstm_config()
                config.sector = "conglomerate"
                config.macro_weight = 0.5
                config.technical_weight = 0.5
            
            predictor = StockLSTMPredictor(config, stock_symbol.upper())
            self.current_model = predictor
            self.current_stock = stock_symbol
            
            self.update_progress({
                'current_step': 'Preparing training data',
                'progress_percentage': 25
            })
            
            # Step 3: Prepare training data (30% progress)
            data_info = predictor.prepare_data(df_with_features, feature_columns)
            
            self.update_progress({
                'status': 'training',
                'current_step': 'Starting model training',
                'progress_percentage': 30,
                'total_epochs': config.num_epochs
            })
            
            # Step 4: Train model (30-85% progress)
            training_results = predictor.train_model(data_info, self.progress_callback)
            
            self.update_progress({
                'current_step': 'Evaluating model',
                'progress_percentage': 85
            })
            
            # Step 5: Evaluate model (90% progress)
            evaluation_results = predictor.evaluate_model(data_info)
            
            self.update_progress({
                'current_step': 'Analyzing features',
                'progress_percentage': 90
            })
            
            # Step 6: Feature importance (95% progress)
            try:
                feature_importance = predictor.get_feature_importance(data_info)
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {e}")
                feature_importance = {'feature_importance': {}, 'top_10_features': []}
            
            self.update_progress({
                'current_step': 'Saving model',
                'progress_percentage': 95
            })
            
            # Step 7: Save model and results (100% progress)
            predictor.save_model(str(model_path))
            
            # Save training metadata to database
            training_metadata = {
                'stock_symbol': stock_symbol,
                'training_date': datetime.now(),
                'model_type': 'LSTM',
                'sector': config.sector,
                'train_accuracy': training_results['final_train_accuracy'],
                'val_accuracy': training_results['final_val_accuracy'],
                'test_accuracy': evaluation_results['test_accuracy'],
                'epochs_trained': training_results['epochs_trained'],
                'features_count': len(feature_columns),
                'data_points': len(df_with_features)
            }
            
            # Store in database
            self.db_manager.store_training_log(training_metadata)
            
            # Combine all results
            complete_results = {
                'status': 'success',
                'stock_symbol': stock_symbol,
                'training': training_results,
                'evaluation': evaluation_results,
                'feature_importance': feature_importance,
                'model_metadata': training_metadata,
                'data_info': {
                    'total_records': len(df_with_features),
                    'features_count': len(feature_columns),
                    'train_size': data_info['train_size'],
                    'val_size': data_info['val_size'],
                    'test_size': data_info['test_size']
                }
            }
            
            self.update_progress({
                'status': 'completed',
                'current_step': 'Training completed successfully',
                'progress_percentage': 100,
                'final_results': {
                    'test_accuracy': evaluation_results['test_accuracy'],
                    'epochs_trained': training_results['epochs_trained']
                }
            })
            
            logger.info(f"Training completed for {stock_symbol} - "
                       f"Test Accuracy: {evaluation_results['test_accuracy']:.2f}%")
            
            return complete_results
            
        except Exception as e:
            error_msg = f"Training failed for {stock_symbol}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.update_progress({
                'status': 'error',
                'current_step': 'Training failed',
                'error': error_msg,
                'progress_percentage': 0
            })
            
            return {
                'status': 'error',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress"""
        return self.training_progress.copy()
    
    def load_trained_model(self, stock_symbol: str) -> Optional[StockLSTMPredictor]:
        """Load a previously trained model"""
        model_path = self.models_dir / f"{stock_symbol.lower()}_lstm"
        
        if not model_path.exists():
            logger.warning(f"No trained model found for {stock_symbol}")
            return None
        
        try:
            # Create predictor and load model
            config = create_hdfc_lstm_config()  # Default config, will be updated on load
            predictor = StockLSTMPredictor(config, stock_symbol.upper())
            predictor.load_model(str(model_path))
            
            logger.info(f"Loaded trained model for {stock_symbol}")
            return predictor
            
        except Exception as e:
            logger.error(f"Failed to load model for {stock_symbol}: {e}")
            return None
    
    async def predict_stock_trend(self, stock_symbol: str, 
                                days_back: int = 60) -> Dict[str, Any]:
        """
        Make trend prediction for a stock using trained model
        Args:
            stock_symbol: Stock symbol to predict
            days_back: Number of days of recent data to use
        """
        try:
            # Load trained model
            predictor = self.load_trained_model(stock_symbol)
            if predictor is None:
                return {
                    'status': 'error',
                    'error': f'No trained model found for {stock_symbol}. Train model first.'
                }
            
            # Get recent data with features
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back + 30)).strftime('%Y-%m-%d')
            
            df_with_features, feature_columns = await self.get_stock_data_with_features(
                stock_symbol, start_date, end_date
            )
            
            # Make prediction
            prediction = predictor.predict(df_with_features.tail(days_back))
            
            # Add additional context
            prediction['model_info'] = {
                'last_training_date': predictor.training_history[-1]['epoch'] if predictor.training_history else None,
                'data_freshness': df_with_features.index[-1].strftime('%Y-%m-%d'),
                'features_used': len(feature_columns)
            }
            
            return {
                'status': 'success',
                'prediction': prediction
            }
            
        except Exception as e:
            error_msg = f"Prediction failed for {stock_symbol}: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
    
    async def train_all_stocks(self, stock_symbols: List[str] = None, 
                              retrain: bool = False) -> Dict[str, Any]:
        """Train models for all configured stocks"""
        if stock_symbols is None:
            stock_symbols = ['HDFCBANK', 'ICICIBANK', 'INFY', 'TATAMOTORS', 'RELIANCE']
        
        results = {}
        total_stocks = len(stock_symbols)
        
        for i, stock_symbol in enumerate(stock_symbols):
            logger.info(f"Training model {i+1}/{total_stocks}: {stock_symbol}")
            
            self.update_progress({
                'status': 'training_multiple',
                'current_step': f'Training {stock_symbol} ({i+1}/{total_stocks})',
                'progress_percentage': (i / total_stocks) * 100,
                'stocks_completed': i,
                'total_stocks': total_stocks
            })
            
            result = await self.train_stock_model(stock_symbol, retrain)
            results[stock_symbol] = result
            
            # Brief pause between models
            await asyncio.sleep(1)
        
        self.update_progress({
            'status': 'completed',
            'current_step': 'All models trained successfully',
            'progress_percentage': 100,
            'stocks_completed': total_stocks,
            'total_stocks': total_stocks
        })
        
        return {
            'status': 'completed',
            'results': results,
            'summary': {
                'total_trained': len([r for r in results.values() if r['status'] == 'success']),
                'total_failed': len([r for r in results.values() if r['status'] == 'error']),
                'total_skipped': len([r for r in results.values() if r['status'] == 'skipped'])
            }
        }
    
    def get_model_performance_summary(self, stock_symbol: str = None) -> Dict[str, Any]:
        """Get performance summary for trained models"""
        try:
            if stock_symbol:
                # Get performance for specific stock
                training_logs = self.db_manager.get_training_logs(stock_symbol)
            else:
                # Get performance for all stocks
                training_logs = self.db_manager.get_training_logs()
            
            if not training_logs:
                return {'status': 'no_data', 'message': 'No training logs found'}
            
            # Process training logs
            performance_data = []
            for log in training_logs:
                performance_data.append({
                    'stock_symbol': log['stock_symbol'],
                    'training_date': log['training_date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'test_accuracy': log['test_accuracy'],
                    'val_accuracy': log['val_accuracy'],
                    'epochs_trained': log['epochs_trained'],
                    'sector': log['sector'],
                    'data_points': log['data_points']
                })
            
            # Calculate summary statistics
            accuracies = [p['test_accuracy'] for p in performance_data]
            summary = {
                'total_models': len(performance_data),
                'average_accuracy': np.mean(accuracies),
                'best_accuracy': np.max(accuracies),
                'worst_accuracy': np.min(accuracies),
                'models_above_70_percent': len([a for a in accuracies if a > 70])
            }
            
            return {
                'status': 'success',
                'performance_data': performance_data,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# Standalone training functions for easy use
async def train_hdfc_model_standalone(retrain: bool = False) -> Dict[str, Any]:
    """Standalone function to train HDFC Bank model"""
    controller = TrainingController()
    return await controller.train_stock_model('HDFCBANK', retrain=retrain)

async def train_all_models_standalone(retrain: bool = False) -> Dict[str, Any]:
    """Standalone function to train all stock models"""
    controller = TrainingController()
    return await controller.train_all_stocks(retrain=retrain)

def get_training_status() -> Dict[str, Any]:
    """Get current training status (for API endpoints)"""
    controller = TrainingController()
    return controller.get_training_progress()

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Model Training Controller')
    parser.add_argument('--stock', type=str, help='Stock symbol to train (e.g., HDFCBANK)')
    parser.add_argument('--all', action='store_true', help='Train all stocks')
    parser.add_argument('--retrain', action='store_true', help='Retrain even if model exists')
    parser.add_argument('--predict', type=str, help='Make prediction for stock symbol')
    parser.add_argument('--status', action='store_true', help='Show model performance status')
    
    args = parser.parse_args()
    
    async def main():
        controller = TrainingController()
        
        if args.status:
            # Show performance summary
            summary = controller.get_model_performance_summary()
            if summary['status'] == 'success':
                print("\n=== Model Performance Summary ===")
                print(f"Total Models: {summary['summary']['total_models']}")
                print(f"Average Accuracy: {summary['summary']['average_accuracy']:.2f}%")
                print(f"Best Accuracy: {summary['summary']['best_accuracy']:.2f}%")
                print(f"Models >70% Accuracy: {summary['summary']['models_above_70_percent']}")
                
                print("\n=== Individual Model Performance ===")
                for model in summary['performance_data']:
                    print(f"{model['stock_symbol']}: {model['test_accuracy']:.2f}% "
                          f"({model['training_date']}) - {model['sector']}")
            else:
                print("No training data found")
                
        elif args.predict:
            # Make prediction
            print(f"Making prediction for {args.predict}...")
            result = await controller.predict_stock_trend(args.predict.upper())
            
            if result['status'] == 'success':
                pred = result['prediction']
                print(f"\n=== Prediction for {args.predict.upper()} ===")
                print(f"Trend: {pred['predicted_trend']}")
                print(f"Confidence: {pred['confidence']:.1%}")
                print(f"Probabilities:")
                for trend, prob in pred['probabilities'].items():
                    print(f"  {trend.title()}: {prob:.1%}")
            else:
                print(f"Prediction failed: {result['error']}")
                
        elif args.all:
            # Train all stocks
            print("Training all stock models...")
            results = await controller.train_all_stocks(retrain=args.retrain)
            
            print(f"\n=== Training Summary ===")
            print(f"Successfully trained: {results['summary']['total_trained']}")
            print(f"Failed: {results['summary']['total_failed']}")
            print(f"Skipped: {results['summary']['total_skipped']}")
            
            for stock, result in results['results'].items():
                status = result['status']
                if status == 'success':
                    acc = result['evaluation']['test_accuracy']
                    print(f"  {stock}: ✅ {acc:.2f}% accuracy")
                elif status == 'error':
                    print(f"  {stock}: ❌ {result['error']}")
                else:
                    print(f"  {stock}: ⏭️  Skipped (model exists)")
                    
        elif args.stock:
            # Train specific stock
            stock_symbol = args.stock.upper()
            print(f"Training model for {stock_symbol}...")
            
            def progress_display(info):
                print(f"\rEpoch {info['current_epoch']}/{info['total_epochs']} "
                      f"({info['progress_percentage']:.1f}%) - "
                      f"Val Acc: {info['metrics'].get('val_accuracy', 0):.2f}%", end='')
            
            # Monitor progress
            import threading
            import time
            
            def monitor_progress():
                while True:
                    progress = controller.get_training_progress()
                    if progress['status'] == 'training':
                        print(f"\r{progress['current_step']} - "
                              f"{progress['progress_percentage']:.1f}%", end='')
                    elif progress['status'] in ['completed', 'error']:
                        break
                    time.sleep(2)
            
            monitor_thread = threading.Thread(target=monitor_progress)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            result = await controller.train_stock_model(stock_symbol, retrain=args.retrain)
            
            print()  # New line after progress
            
            if result['status'] == 'success':
                print(f"\n=== Training Completed for {stock_symbol} ===")
                print(f"Test Accuracy: {result['evaluation']['test_accuracy']:.2f}%")
                print(f"Epochs Trained: {result['training']['epochs_trained']}")
                print(f"Model saved successfully")
            elif result['status'] == 'skipped':
                print(f"Model for {stock_symbol} already exists. Use --retrain to retrain.")
            else:
                print(f"Training failed: {result['error']}")
                
        else:
            print("Please specify --stock, --all, --predict, or --status")
            parser.print_help()
    
    # Run the async main function
    asyncio.run(main())