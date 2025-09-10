# Event-Driven Stock Trend Predictor

A sophisticated multi-modal stock trend prediction system that combines technical analysis, news sentiment, and macroeconomic factors to predict Indian stock movements using ensemble machine learning techniques.

## Overview

The Event-Driven Stock Trend Predictor is an advanced financial AI system designed to analyze Indian stock markets through multiple data sources and machine learning models. The system processes technical indicators, financial news sentiment, and macroeconomic factors to generate comprehensive buy/sell/hold signals with confidence scores.

## Architecture

### Multi-Modal Intelligence Pipeline

The system employs a three-pronged approach to stock prediction:

1. **Technical Analysis Engine**: PyTorch LSTM neural network analyzing 68+ technical indicators
2. **Macroeconomic Analysis**: XGBoost model processing 18+ economic indicators
3. **News Sentiment Analysis**: FinBERT-powered sentiment processing of financial headlines
4. **Ensemble Fusion**: Dynamic weighted combination of all three models

```
Technical Data (68 features) → PyTorch LSTM → 70-80% accuracy
News Headlines (80+ articles) → FinBERT Sentiment → Bullish/Bearish scores  
Macro Factors (18 features) → XGBoost → 60-70% accuracy
                ↓
Dynamic Ensemble Fusion → Final Prediction (Target: 85-95% accuracy)
```

## Target Assets

The system is designed to analyze five major Indian stocks across different sectors:

| Stock | Symbol | ADR Symbol | Sector | Market Cap |
|-------|--------|------------|---------|------------|
| **HDFC Bank** | HDFCBANK.NS | HDB | Banking | Large Cap |
| **ICICI Bank** | ICICIBANK.NS | IBN | Banking | Large Cap |
| **Infosys** | INFY.NS | INFY | IT Services | Large Cap |
| **Tata Motors** | TATAMOTORS.NS | TTM | Automotive | Large Cap |
| **Reliance Industries** | RELIANCE.NS | - | Conglomerate | Large Cap |

## Technical Analysis Model

### LSTM Neural Network Architecture

The technical analysis component uses a deep LSTM neural network optimized for financial time series:

**Model Specifications:**
- **Input Features**: 68 technical indicators from TA-Lib
- **Architecture**: 3-layer LSTM with attention mechanism
- **Hidden Size**: 128 neurons per layer
- **Dropout**: 30% for regularization
- **Sequence Length**: 60 days lookback window
- **Training**: Adam optimizer with learning rate scheduling

**Key Technical Indicators:**
- Price-based: RSI, MACD, Bollinger Bands, Moving Averages (SMA, EMA, WMA)
- Volume-based: On Balance Volume, Volume Rate of Change, Accumulation/Distribution
- Momentum: ADX, Commodity Channel Index, Williams %R, Stochastic Oscillator
- Volatility: Average True Range, Bollinger Band Width
- Sector-specific: Banking ratios, IT performance metrics

**Performance**: Achieves 70-80% validation accuracy on historical data

## Macroeconomic Analysis Model

### XGBoost Economic Factors Model

The macroeconomic component analyzes broader economic indicators affecting stock performance:

**Economic Indicators (18 features):**
- **RBI Policy**: Repo rate, reverse repo rate, CRR, SLR
- **Currency**: USD/INR exchange rate and volatility
- **Commodities**: WTI oil prices, gold prices
- **Global Markets**: S&P 500 levels, VIX volatility
- **Derived Indicators**: Real interest rates, currency strength index, economic stress index

**Sector-Specific Features:**
- **Banking Environment Score**: RBI policy impact, interest rate sensitivity
- **IT Favorability Index**: USD/INR correlation, global tech sentiment
- **Auto Commodity Impact**: Oil price sensitivity, raw material costs
- **Conglomerate Diversification**: Multi-sector exposure analysis

**Performance**: Achieves 60-70% test accuracy with repo rate as primary feature for banking stocks

## News Sentiment Analysis

### FinBERT Multi-Model Sentiment Processing

The sentiment analysis engine processes Indian financial news through multiple models:

**Sentiment Models:**
- **FinBERT**: Financial domain-specific BERT model for Indian markets
- **VADER**: Lexicon-based sentiment analyzer
- **Pattern Recognition**: Custom rules for Indian financial terminology
- **TextBlob**: Polarity and subjectivity analysis

**Data Sources:**
- Financial news headlines (80+ articles per stock)
- Indian business publications and financial websites
- Stock-specific keyword filtering and relevance scoring

**Processing Features:**
- Multi-language support for Hindi financial terms
- Context-aware sentiment scoring
- Temporal sentiment trends analysis
- Article relevance and credibility weighting

## Ensemble Model Weightings

### Dynamic Weight Allocation

The ensemble system uses sector-specific dynamic weighting based on model performance and confidence:

**Base Sector Weights:**

#### Banking Sector (HDFC Bank, ICICI Bank)
- **Technical Analysis**: 40% - Strong technical patterns in banking stocks
- **Macroeconomic**: 40% - High sensitivity to RBI policy and interest rates
- **News Sentiment**: 20% - Regulatory and policy news impact

#### IT Services Sector (Infosys)
- **Technical Analysis**: 50% - Strong technical trading patterns
- **Macroeconomic**: 30% - USD/INR correlation and global tech trends  
- **News Sentiment**: 20% - Earnings and client news sensitivity

#### Automotive Sector (Tata Motors)
- **Technical Analysis**: 35% - Cyclical technical patterns
- **Macroeconomic**: 45% - High commodity price and economic cycle sensitivity
- **News Sentiment**: 20% - Industry and policy news impact

#### Conglomerate Sector (Reliance)
- **Technical Analysis**: 45% - Diverse business technical patterns
- **Macroeconomic**: 35% - Multi-sector economic exposure
- **News Sentiment**: 20% - Business segment news variety

### Performance-Based Dynamic Adjustment

The system continuously adjusts weights based on:
- Individual model accuracy over recent predictions
- Market regime changes (trending vs sideways)
- Volatility environments (high vs low volatility)
- Model confidence levels and consensus agreement

## Prediction Output

### Comprehensive Prediction Results

Each prediction includes:

```json
{
  "stock_symbol": "HDFCBANK.NS",
  "ensemble_prediction": 0.7234,
  "ensemble_confidence": 0.8456,
  "signal": "BUY",
  "individual_predictions": {
    "technical": {
      "prediction": 0.7891,
      "confidence": 0.8234,
      "trend": "BULLISH"
    },
    "macro": {
      "prediction": 0.6789,
      "confidence": 0.7456,
      "trend": "BULLISH"
    },
    "sentiment": {
      "prediction": 0.6234,
      "confidence": 0.5678,
      "trend": "BULLISH"
    }
  },
  "model_weights": {
    "technical_weight": 0.42,
    "macro_weight": 0.38,
    "sentiment_weight": 0.20
  },
  "model_consensus": {
    "agreement_level": 1.0,
    "is_strong_consensus": true,
    "total_models": 3
  },
  "current_price": 1523.45,
  "prediction_date": "2025-09-10 14:30:00"
}
```

### Signal Classification

**Buy/Sell Signals:**
- **STRONG_BUY**: Ensemble prediction > 0.65 with confidence > 0.7
- **BUY**: Ensemble prediction > 0.55 with confidence > 0.4  
- **WEAK_BUY**: Ensemble prediction > 0.55 with confidence > 0.4
- **HOLD**: Ensemble prediction between 0.45-0.55 or low confidence
- **WEAK_SELL**: Ensemble prediction < 0.45 with confidence > 0.4
- **SELL**: Ensemble prediction < 0.45 with confidence > 0.4
- **STRONG_SELL**: Ensemble prediction < 0.35 with confidence > 0.7

## Technology Stack

### Backend Infrastructure
- **API Framework**: FastAPI with WebSocket support
- **Database**: PostgreSQL for stock data, SQLite for news/macro data
- **ML Frameworks**: PyTorch (LSTM), XGBoost, scikit-learn
- **NLP**: Transformers (FinBERT), NLTK, VADER sentiment

### Frontend Dashboard
- **Framework**: Next.js 13+ with TypeScript
- **UI Components**: Modern responsive design
- **Charts**: Interactive technical analysis visualizations
- **Real-time**: WebSocket connections for live updates

### Data Sources
- **Market Data**: Alpha Vantage API for NSE and ADR data
- **News Data**: NewsAPI for Indian financial headlines
- **Macro Data**: RBI, Federal Reserve, commodities APIs
- **Technical Indicators**: TA-Lib library with 68+ indicators

## Performance Metrics

### Model Accuracy Targets
- **Individual Models**: 60-80% accuracy per model
- **Ensemble System**: 85-95% target accuracy
- **Prediction Latency**: <2 seconds per stock
- **Data Processing**: <5 minutes for daily updates

### Risk Management Features
- **Confidence Scoring**: All predictions include confidence intervals
- **Model Consensus**: Agreement tracking across models  
- **Uncertainty Quantification**: Low confidence predictions flagged
- **Sector Adaptability**: Weights adjust based on market conditions

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- PostgreSQL 12+
- 8GB RAM minimum

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd event-driven-stock-trend-predictor

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies  
cd frontend && npm install

# Setup databases and configuration
python scripts/initial_data_download.py

# Start API server
python src/backend/api_routes.py

# Start frontend dashboard
cd frontend && npm run dev
```

### API Endpoints
- **GET** `/api/predict/{stock_symbol}` - Get ensemble prediction
- **GET** `/api/technical-analysis/{stock_symbol}` - Technical indicators
- **GET** `/api/sentiment/{stock_symbol}` - News sentiment analysis  
- **GET** `/api/macro-factors/{stock_symbol}` - Economic factors
- **POST** `/api/train/{stock_symbol}` - Train models
- **WebSocket** `/ws` - Real-time updates

## Limitations & Disclaimers

### Important Considerations
- **Market Risk**: All predictions are probabilistic and not guarantees
- **Data Dependency**: Predictions quality depends on data availability
- **Market Conditions**: Model performance may vary across market regimes
- **Regulatory Changes**: System may need updates for policy changes

### Not Financial Advice
This system is designed for educational and research purposes. All predictions should be considered as informational only and not as financial advice. Users should conduct their own research and consult with financial advisors before making investment decisions.

### Model Limitations
- Historical performance does not guarantee future results
- Models may have biases based on training data
- Black swan events are not predictable by the system
- System works best in normal market conditions

## Future Enhancements

### Planned Features
- **Social Media Integration**: Reddit and Twitter sentiment analysis
- **Options Flow Analysis**: Institutional options activity tracking
- **Sector Rotation Signals**: Inter-sector movement prediction
- **Risk-Adjusted Returns**: Sharpe ratio and volatility-adjusted signals
- **Portfolio Optimization**: Multi-stock portfolio recommendations

### Scalability
- Additional Indian stocks (mid-cap and small-cap)
- International market expansion
- Alternative data sources integration
- Real-time streaming prediction updates

## Contributing

The system is designed with modularity in mind, allowing for easy extension of new data sources, models, or prediction techniques. Each component (technical, macro, sentiment) can be enhanced independently while maintaining ensemble compatibility.

## License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service and local financial regulations.