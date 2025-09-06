"""
Test News Sentiment Integration with Existing LSTM Model
This script tests the integration of news sentiment with technical analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from src.features.technical_indicators import TechnicalIndicatorCalculator
from src.models.sector_lstm import SectorLSTM
from src.features.sentiment_processor import IndianFinancialSentimentProcessor

class NewsIntegratedLSTM:
    def __init__(self, sector: str = 'banking'):
        """Extended LSTM model with news sentiment integration"""
        self.sector = sector
        self.technical_calc = TechnicalIndicatorCalculator()
        self.sentiment_processor = IndianFinancialSentimentProcessor()
        self.base_lstm = SectorLSTM(sector)
        
    def prepare_enhanced_data(self, symbol: str, days: int = 1000) -> tuple:
        """Prepare data combining technical indicators + news sentiment"""
        print(f"Preparing enhanced dataset for {symbol}...")
        
        # 1. Get technical indicators (existing working system)
        print("1. Calculating technical indicators...")
        tech_df = self.technical_calc.calculate_all_indicators(symbol, days)
        print(f"   Generated {len(tech_df.columns)} technical features")
        
        # 2. Get news sentiment features
        print("2. Processing news sentiment...")
        sentiment_features = self.sentiment_processor.generate_sentiment_features(symbol, days=7)
        print(f"   Generated {len(sentiment_features)} sentiment features")
        
        # 3. Combine features
        print("3. Combining technical + sentiment features...")
        
        # Add sentiment features to technical dataframe (broadcast to all rows)
        for feature_name, feature_value in sentiment_features.items():
            tech_df[feature_name] = feature_value
        
        print(f"   Combined dataset shape: {tech_df.shape}")
        print(f"   Total features: {len(tech_df.columns)}")
        
        # 4. Create targets (same as original LSTM)
        future_returns = tech_df['close'].shift(-1) / tech_df['close'] - 1
        tech_df['target'] = np.where(future_returns > 0.02, 2,  # Up > 2%
                                   np.where(future_returns < -0.02, 0, 1))  # Down < -2%
        
        # Remove rows with NaN targets
        tech_df = tech_df.dropna(subset=['target'])
        
        # Prepare feature columns (exclude basic OHLCV + target)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'target']
        feature_cols = [col for col in tech_df.columns if col not in exclude_cols]
        
        X = tech_df[feature_cols].values
        y = tech_df['target'].values
        
        # Handle NaN values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        print(f"   Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Target distribution: {np.bincount(y.astype(int))}")
        
        return X, y, feature_cols, tech_df
    
    def train_enhanced_model(self, symbol: str, epochs: int = 20, validation_split: float = 0.2) -> dict:
        """Train LSTM model with technical + sentiment features"""
        print(f"Training enhanced LSTM model for {symbol}")
        print("=" * 50)
        
        # Prepare enhanced dataset
        X, y, feature_cols, df = self.prepare_enhanced_data(symbol)
        
        # Store feature columns in base LSTM
        self.base_lstm.feature_columns = feature_cols
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.base_lstm.scaler.fit_transform(X_train)
        X_val_scaled = self.base_lstm.scaler.transform(X_val)
        
        # Fit imputer
        from sklearn.impute import SimpleImputer
        self.base_lstm.imputer = SimpleImputer(strategy='mean')
        self.base_lstm.imputer.fit(X_train)
        
        # Train using existing LSTM infrastructure
        if self.base_lstm.pytorch_available:
            result = self.base_lstm._train_pytorch_model(
                X_train_scaled, y_train, X_val_scaled, y_val, epochs, verbose=True
            )
        else:
            result = self.base_lstm._train_sklearn_model(
                X_train_scaled, y_train, X_val_scaled, y_val, verbose=True
            )
        
        # Add enhancement info to result
        result.update({
            'total_features': len(feature_cols),
            'technical_features': len(feature_cols) - 8,  # 8 sentiment features
            'sentiment_features': 8,
            'enhancement': 'Technical + News Sentiment'
        })
        
        return result
    
    def predict_with_sentiment(self, symbol: str) -> dict:
        """Make prediction using technical + sentiment data"""
        print(f"Making enhanced prediction for {symbol}...")
        
        # Use the base LSTM predict method (it will handle the enhanced features)
        prediction = self.base_lstm.predict(symbol, days=300)
        
        # Add sentiment context to prediction
        if 'error' not in prediction:
            # Get current sentiment
            sentiment_data = self.sentiment_processor.process_stock_sentiment(symbol, days=3)
            
            prediction.update({
                'enhancement_type': 'Technical + News Sentiment',
                'news_sentiment': sentiment_data['sentiment_summary'],
                'news_confidence': sentiment_data['confidence'],
                'news_articles_analyzed': sentiment_data['articles_processed'],
                'sentiment_context': f"Recent news sentiment: {sentiment_data['sentiment_summary']} "
                                   f"(confidence: {sentiment_data['confidence']:.1%})"
            })
        
        return prediction
    
    def compare_performance(self, symbol: str, epochs: int = 10) -> dict:
        """Compare technical-only vs technical+sentiment performance"""
        print(f"Comparing model performance for {symbol}")
        print("=" * 60)
        
        results = {}
        
        # 1. Train technical-only model
        print("\n1. Training technical-only model...")
        tech_only_model = SectorLSTM(self.sector)
        tech_result = tech_only_model.train(symbol, epochs=epochs, verbose=True)
        results['technical_only'] = tech_result
        
        # 2. Train enhanced model
        print(f"\n2. Training technical + sentiment model...")
        enhanced_result = self.train_enhanced_model(symbol, epochs=epochs)
        results['technical_plus_sentiment'] = enhanced_result
        
        # 3. Compare predictions
        print(f"\n3. Comparing predictions...")
        tech_prediction = tech_only_model.predict(symbol)
        enhanced_prediction = self.predict_with_sentiment(symbol)
        
        results['predictions'] = {
            'technical_only': tech_prediction,
            'technical_plus_sentiment': enhanced_prediction
        }
        
        # 4. Summary
        print(f"\n" + "=" * 60)
        print(f"PERFORMANCE COMPARISON SUMMARY")
        print(f"=" * 60)
        
        if tech_result.get('success') and enhanced_result.get('success'):
            tech_acc = tech_result.get('val_accuracy', 0)
            enhanced_acc = enhanced_result.get('val_accuracy', 0)
            
            print(f"Technical Only Accuracy:    {tech_acc:.3f}")
            print(f"Technical + Sentiment:      {enhanced_acc:.3f}")
            print(f"Improvement:                {enhanced_acc - tech_acc:+.3f}")
            print(f"Relative Improvement:       {((enhanced_acc / tech_acc) - 1) * 100:+.1f}%")
            
            results['comparison'] = {
                'technical_accuracy': tech_acc,
                'enhanced_accuracy': enhanced_acc,
                'improvement': enhanced_acc - tech_acc,
                'relative_improvement': ((enhanced_acc / tech_acc) - 1) * 100
            }
        
        if 'error' not in tech_prediction and 'error' not in enhanced_prediction:
            print(f"\nPrediction Comparison:")
            print(f"Technical Only:             {tech_prediction['predicted_trend']} "
                  f"(conf: {tech_prediction['confidence']:.1%})")
            print(f"Technical + Sentiment:      {enhanced_prediction['predicted_trend']} "
                  f"(conf: {enhanced_prediction['confidence']:.1%})")
            print(f"News Context:               {enhanced_prediction.get('sentiment_context', 'N/A')}")
        
        return results

def test_news_integration():
    """Test the complete news sentiment integration"""
    print("Testing News Sentiment Integration with LSTM")
    print("=" * 60)
    
    # Initialize enhanced model
    enhanced_model = NewsIntegratedLSTM('banking')
    
    # Test with HDFC Bank
    symbol = 'HDFCBANK.NS'
    
    try:
        # Run comprehensive comparison
        results = enhanced_model.compare_performance(symbol, epochs=5)
        
        # Save results for analysis
        if results.get('comparison'):
            improvement = results['comparison']['improvement']
            if improvement > 0:
                print(f"\n🎉 SUCCESS: News sentiment improved accuracy by {improvement:.3f}!")
                print(f"Expected result: Technical analysis enhanced with market context")
            else:
                print(f"\n⚠️  No improvement detected. This may be due to:")
                print(f"   - Limited news data in test period")
                print(f"   - Short training epochs (use more for real training)")
                print(f"   - Market in neutral sentiment period")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_sentiment_test():
    """Quick test of sentiment processing alone"""
    print("Quick Sentiment Analysis Test")
    print("=" * 40)
    
    processor = IndianFinancialSentimentProcessor()
    
    # Test sentiment processing
    result = processor.process_stock_sentiment('HDFCBANK.NS', days=3)
    
    print(f"Stock: HDFCBANK.NS")
    print(f"Articles processed: {result['articles_processed']}")
    print(f"Overall sentiment: {result['sentiment_summary']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Bullish score: {result['bullish_score']:.2f}")
    print(f"Bearish score: {result['bearish_score']:.2f}")
    
    if result['articles_processed'] > 0:
        print(f"\nRecent headlines:")
        for article in result['articles_analyzed'][:3]:
            print(f"  • {article['title'][:80]}...")
            print(f"    Sentiment: {article['article_sentiment']} "
                  f"(confidence: {article['article_confidence']:.1%})")
    
    return result

if __name__ == "__main__":
    # Run quick test first
    print("1. Testing sentiment processing...")
    sentiment_result = quick_sentiment_test()
    
    # If sentiment works, run full integration
    if sentiment_result and sentiment_result['articles_processed'] > 0:
        print(f"\n2. Testing full integration...")
        integration_result = test_news_integration()
    else:
        print(f"\n⚠️  Limited news data found. Integration test may have limited improvement.")
        print(f"   Consider running during market hours or with different time periods.")