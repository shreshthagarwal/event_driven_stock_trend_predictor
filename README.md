# Event-Driven Stock Trend Predictor

## 🎯 Project Overview

A comprehensive trend analysis platform that identifies and visualizes market trends across 5 data-rich Indian stocks using ensemble machine learning models and technical indicators. The system combines LSTM neural networks with traditional statistical methods to detect trend strength, direction, and potential reversals in real-time.

### What This Project Actually Does (Simple Terms)

**Main Purpose:** Tells you if a stock price is going UP, DOWN, or SIDEWAYS - and how confident it is about that prediction.

**Core Functions:**
1. **Trend Detection:** Analyzes stock prices and classifies trends as UPWARD/DOWNWARD/SIDEWAYS with strength indicators
2. **Smart Analysis:** Combines price patterns + news sentiment + social media buzz + macro economic factors
3. **Visual Dashboard:** Colorful charts with trend arrows, confidence scores, and sector heatmaps
4. **Alert System:** Notifications when trends are about to change or reverse

**Real-World Example:**
- **Input:** User selects HDFC Bank (HDFCBANK.NS)
- **Output:** "HDFCBANK is in a STRONG UPWARD trend (confidence: 85%)", "Trend likely to continue for next 5-7 days", Chart shows green arrows and positive sentiment score

## 🎯 Target Stocks (Data-Rich Indian Assets)

| Stock | Indian Symbol | ADR Symbol | Sector | Why Selected |
|-------|--------------|------------|---------|-------------|
| **HDFC Bank** | HDFCBANK.NS | HDB | Banking | India's top private bank, huge coverage, NYSE ADR |
| **ICICI Bank** | ICICIBANK.NS | IBN | Banking | Heavyweight private bank, robust disclosures |
| **Infosys** | INFY.NS | INFY | IT Services | Global revenue, strong ADR footprint, tech bellwether |
| **Tata Motors** | TATAMOTORS.NS | TTM | Automotive | Auto + global exposure (JLR), commodity sensitive |
| **Reliance** | RELIANCE.NS | - | Conglomerate | India's largest by market cap, multi-sector exposure |

### Why These 5 Stocks Are Perfect:
- **Diverse Sector Coverage:** Banking, IT, Auto, Conglomerate
- **Dual Listings:** NS + ADR = 2x more data points and news coverage  
- **High Liquidity:** Cleaner price action, less noise
- **Extensive Coverage:** More research reports and sentiment data
- **Regular Events:** Earnings, splits, policy announcements

## 🏗️ System Architecture

### Core System Logic

1. **Data Collection Process:** Background Python scripts continuously pull:
   - Stock price data from yfinance/Alpha Vantage (both NS and ADR)
   - Financial news from NewsAPI and Indian sources
   - Social media sentiment from Twitter/Reddit APIs
   - Macro data (RBI policy, INR/USD, commodity prices)

2. **Feature Engineering Pipeline:** 
   - Technical indicators (RSI, MACD, moving averages) via TA-Lib
   - News sentiment processing through FinBERT for Indian finance
   - Macro factor correlation analysis
   - Sector-specific feature extraction

3. **Machine Learning Prediction Engine:**
   - **LSTM Model:** Takes 60 days of price data + technical indicators + sentiment
   - **XGBoost Model:** Uses macro factors and sector-specific features
   - **Ensemble Approach:** Averages predictions with sector-aware weights
   - **Output:** Bullish/Bearish/Sideways classification + confidence percentage

4. **Event Detection System:** Monitors for earnings, RBI announcements, policy changes - triggers immediate recalculation (the "event-driven" part)

5. **Signal Generation:** Combines all predictions into final buy/sell/hold signals with risk-adjusted confidence scores

6. **Real-time Dashboard:** React.js frontend with WebSocket connections for live updates

## 🛠️ Tech Stack

### Backend
- **API Framework:** FastAPI
- **Database:** PostgreSQL (main data), Redis (caching)
- **ML Libraries:** PyTorch, Scikit-learn, XGBoost
- **Technical Analysis:** TA-Lib, pandas-ta
- **NLP:** NLTK, VADER sentiment, TextBlob, FinBERT

### Data Sources
- **Market Data:** yfinance, Alpha Vantage
- **News:** NewsAPI, Indian financial news sites
- **Social Media:** Twitter API, Reddit API
- **Macro Data:** RBI, Bloomberg APIs

### Frontend
- **Framework:** React.js
- **Charts:** Chart.js/Plotly.js
- **UI Components:** Material-UI
- **Real-time:** WebSocket connections

### Deployment
- **Containerization:** Docker
- **Prototyping:** Streamlit
- **CI/CD:** GitHub Actions
- **Monitoring:** Custom performance tracking

## 📁 Project Structure

```
event_driven_stock_trend_predictor/
├── src/
│   ├── data/                          # Data collection modules
│   │   ├── market_collector.py        # Dual market data (NS + ADR)
│   │   ├── macro_collector.py         # RBI, currency, commodity data
│   │   ├── news_collector.py          # Indian financial news scraping
│   │   ├── social_collector.py        # Twitter/Reddit sentiment
│   │   └── data_validator.py          # Data quality checks
│   ├── features/                      # Feature engineering
│   │   ├── technical_indicators.py    # TA-Lib indicators
│   │   ├── sector_features.py         # Sector-specific features
│   │   ├── macro_features.py          # RBI policy, INR/USD impact
│   │   ├── sentiment_processor.py     # FinBERT for Indian finance
│   │   └── feature_pipeline.py        # Combined feature engineering
│   ├── models/                        # Machine learning models
│   │   ├── sector_lstm.py             # Sector-aware LSTM models
│   │   ├── macro_xgboost.py           # Macro factor models
│   │   ├── cross_asset_gnn.py         # Graph networks for correlations
│   │   ├── ensemble_predictor.py      # Multi-model combination
│   │   └── model_trainer.py           # Training pipeline with progress
│   ├── signals/                       # Signal generation
│   │   ├── trend_classifier.py        # Multi-timeframe trend detection
│   │   ├── sector_rotation.py         # Cross-sector momentum
│   │   ├── event_detector.py          # RBI, earnings, policy events
│   │   ├── risk_calculator.py         # Sector-specific risk metrics
│   │   └── signal_fusion.py           # Final buy/sell/hold signals
│   ├── backend/                       # API and server
│   │   ├── api_routes.py              # FastAPI endpoints
│   │   ├── database_manager.py        # PostgreSQL operations
│   │   ├── model_manager.py           # Model loading/caching
│   │   ├── websocket_server.py        # Real-time updates
│   │   └── training_controller.py     # Progress tracking
│   └── utils/                         # Utilities
│       ├── config.py                  # Stock configurations
│       ├── indian_market_utils.py     # NSE/BSE specific functions
│       ├── notification_handler.py    # Alert system
│       └── performance_tracker.py     # Model accuracy tracking
├── frontend/                          # React.js dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── StockSelector.js       # 5-stock dropdown with search
│   │   │   ├── SectorDashboard.js     # Banking, IT, Auto, Conglomerate
│   │   │   ├── DualMarketChart.js     # NS + ADR overlay charts
│   │   │   ├── MacroOverlay.js        # RBI policy, INR/USD trends
│   │   │   ├── TrendIndicators.js     # Multi-timeframe signals
│   │   │   ├── TrainingProgress.js    # Real-time training updates
│   │   │   ├── SectorRotation.js      # Cross-sector comparison
│   │   │   └── AlertCenter.js         # Notification management
│   │   ├── services/
│   │   │   ├── api_client.js          # Backend communication
│   │   │   ├── websocket_client.js    # Real-time data stream
│   │   │   └── chart_utils.js         # Chart configuration helpers
│   │   └── hooks/                     # Custom React hooks
│   └── package.json
├── data/                              # Data storage
│   ├── raw/market_data/               # Individual stock folders
│   ├── processed/features/            # Engineered features by stock
│   └── models/                        # Stock-specific trained models
├── config/                            # Configuration files
│   ├── stock_config.yaml              # 5-stock configurations
│   ├── api_credentials.yaml           # API keys and secrets
│   ├── database_config.yaml           # PostgreSQL settings
│   └── model_hyperparams.yaml         # ML model parameters
├── scripts/                           # Automation scripts
└── tests/                             # Test cases
```

## 🎯 Key Design Decisions

### 1. Single-Stock Focus with Dynamic Retraining
- **Approach:** One specialized model per stock, retrain when switching
- **Rationale:** Each stock has unique patterns, macro sensitivities
- **Implementation:** Progress bar during retraining process
- **Benefits:** Higher accuracy, sector-specific optimization

### 2. Dual-Market Data Strategy
- **NS Data:** Real-time Indian market sentiment and volumes
- **ADR Data:** Extended hours trading, US market sentiment
- **Arbitrage Opportunities:** Price differences between markets
- **Enhanced Coverage:** 24-hour news cycle capture

### 3. Sector-Aware Ensemble Models
```yaml
sector_models:
  banking: {lstm_weight: 0.4, macro_weight: 0.6}    # Macro-sensitive
  it_services: {lstm_weight: 0.6, currency_weight: 0.4}  # USD/INR focus
  automotive: {lstm_weight: 0.3, commodity_weight: 0.7}  # Commodity-driven
  conglomerate: {lstm_weight: 0.5, multi_factor_weight: 0.5}  # Balanced
```

### 4. Macro Factor Integration
Each stock tracks sector-specific macro indicators:
- **Banking:** RBI repo rate, CPI, credit growth, INR/USD
- **IT Services:** USD/INR, NASDAQ, US GDP, IT spending, H1B policy
- **Automotive:** Steel prices, crude oil, GBP/USD (JLR), EV policy
- **Conglomerate:** Brent crude, telecom policy, retail consumption

## 📊 Stock Configuration Details

### HDFC Bank Configuration
```yaml
HDFCBANK:
  indian_symbol: "HDFCBANK.NS"
  adr_symbol: "HDB"
  sector: "banking"
  macro_factors: [repo_rate, cpi_inflation, inr_usd_rate, credit_growth]
  news_keywords: ["HDFC Bank", "private banking", "RBI policy"]
  training_lookback: "5y"
  prediction_horizons: ["1d", "3d", "1w", "1m"]
```

### Historical Data Strategy
- **Source:** yfinance + Alpha Vantage (10+ years available)
- **No separate datasets needed:** APIs provide extensive history
- **Indian Market Focus:** Alpha Vantage better for NSE/BSE data
- **Example:** `yf.download('HDFCBANK.NS', period='10y')` gets decade of data

## 🧠 Machine Learning Pipeline

### Feature Engineering Process
1. **Technical Indicators:** RSI, MACD, Bollinger Bands, Moving Averages
2. **Macro Integration:** RBI policy sentiment, currency correlations
3. **Sentiment Scores:** Financial news + social media processing
4. **Sector Features:** Banking cycle timing, IT contract patterns
5. **Cross-Asset Signals:** Sector rotation indicators

### Model Architecture
1. **LSTM Network:** 
   - Input: 60-day price + technical + sentiment data
   - Architecture: 3 layers, dropout regularization
   - Output: Trend probability distribution

2. **XGBoost Classifier:**
   - Input: Macro factors + fundamental ratios
   - Features: 50+ engineered variables
   - Output: Trend classification confidence

3. **Ensemble Fusion:**
   - Weighted combination based on recent performance
   - Dynamic weight adjustment based on market regime
   - Final output: Bullish/Bearish/Sideways + confidence score

### Training Strategy
- **Individual Stock Models:** Specialized for each of the 5 stocks
- **Cross-Validation:** Time-series aware splits
- **Hyperparameter Optimization:** Grid search with sector constraints
- **Performance Tracking:** Accuracy, Sharpe ratio, max drawdown metrics

## 🚨 Current Technical Challenges

### 1. ✅ SOLVED: Historical Data
- **Solution:** yfinance/Alpha Vantage provide 10+ years of data
- **Implementation:** Direct API calls, no separate datasets needed

### 2. ⚠️ NEEDS SOLUTION: Financial News Sentiment
- **Challenge:** Indian financial headlines like "1:1 Split Soon: Tata's Auto Stock Is Rs 6.45 Away From Rs 700 Mark, Jumps 11.50% In Six Months; Time To Buy?"
- **Requirements:** 
  - Understand stock splits, bonus issues
  - Parse price targets and analyst recommendations
  - Handle Indian market terminology
  - Detect bullish/bearish sentiment in complex headlines
- **Proposed Solution:** Fine-tuned FinBERT model for Indian finance

### 3. 🔄 IN PROGRESS: Real-time Model Training
- **Challenge:** Dashboard progress bar during model retraining
- **Requirements:**
  - WebSocket progress updates
  - Non-blocking training process
  - Model validation during training
  - Rollback capability if training fails

## 🎛️ Dashboard Features

### Stock Selection Interface
- **Dropdown:** 5-stock selector with search functionality
- **Current Status:** Shows active model and last training time
- **Retraining Trigger:** Automatic model retraining when switching stocks
- **Progress Tracking:** Real-time training progress with ETA

### Multi-Panel Dashboard
1. **Main Chart:** Dual-market price overlay (NS + ADR)
2. **Trend Indicators:** Multi-timeframe signals (1D, 1W, 1M)
3. **Macro Overlay:** Sector-specific macro factor trends
4. **Sentiment Panel:** News sentiment + social media buzz
5. **Sector Rotation:** Cross-sector momentum comparison
6. **Alert Center:** Recent alerts and notifications

### Real-time Features
- **WebSocket Updates:** Live price and prediction updates
- **Alert Notifications:** Browser notifications for trend changes
- **Performance Metrics:** Model accuracy tracking
- **Backtesting Results:** Historical performance validation

## 🏃‍♂️ 10-Day Implementation Plan

### Days 1-2: Data Foundation
- [ ] Set up yfinance/Alpha Vantage data collection for 5 stocks
- [ ] Implement dual-market data synchronization (NS + ADR)
- [ ] Create PostgreSQL schema for time-series data
- [ ] Build macro data collection (RBI, currency, commodities)

### Days 3-4: Feature Engineering + ML Models
- [ ] Technical indicator calculation pipeline
- [ ] LSTM model training for first stock (HDFC Bank)
- [ ] XGBoost macro factor model development
- [ ] Model validation and performance metrics

### Days 5-6: Sentiment Analysis + Signal Generation
- [ ] Financial news scraping and sentiment processing
- [ ] Social media sentiment integration
- [ ] Event detection system (RBI, earnings announcements)
- [ ] Signal fusion and trend classification

### Days 7-8: React Dashboard
- [ ] Stock selector with retraining progress bar
- [ ] Real-time chart components with dual-market data
- [ ] WebSocket integration for live updates
- [ ] Alert system and notification center

### Days 9-10: Integration + Deployment
- [ ] Model training pipeline with progress tracking
- [ ] Cross-stock model deployment
- [ ] Backtesting module with performance visualization
- [ ] Docker containerization and deployment

## 🔄 Current Development Status

### ✅ Completed
- Project architecture design
- 5-stock selection and rationale
- Technology stack decisions
- Configuration structure (stock_config.yaml)
- File structure and module organization

### 🔄 In Progress
- Financial news sentiment analysis strategy
- Model training pipeline design
- Dashboard component specifications

### ⏳ Next Steps
1. **Immediate Priority:** Solve financial news sentiment analysis challenge
2. **Technical Implementation:** Begin data collection pipeline
3. **Model Development:** Start with HDFC Bank LSTM model
4. **Dashboard Development:** Build React components with progress tracking

## 🔑 API Keys Required

```yaml
# api_credentials.yaml (template)
alpha_vantage:
  api_key: "YOUR_ALPHA_VANTAGE_KEY"
  calls_per_minute: 5

news_api:
  api_key: "YOUR_NEWS_API_KEY"
  calls_per_hour: 100

twitter_api:
  bearer_token: "YOUR_TWITTER_BEARER_TOKEN"
  consumer_key: "YOUR_CONSUMER_KEY"
  consumer_secret: "YOUR_CONSUMER_SECRET"

database:
  postgresql_url: "postgresql://user:pass@localhost:5432/stock_predictor"
  redis_url: "redis://localhost:6379"
```

## 📈 Success Metrics

### Model Performance
- **Accuracy:** >70% trend direction prediction
- **Sharpe Ratio:** >1.5 for signal-based strategies  
- **Max Drawdown:** <15% in backtesting
- **Latency:** <2 seconds for prediction updates

### System Performance
- **Data Freshness:** <5 minute delay for market data
- **Dashboard Load Time:** <3 seconds initial load
- **Real-time Updates:** <1 second WebSocket latency
- **Model Training:** <10 minutes per stock retraining

## 🎯 Target Users

### Primary Users
- **Day Traders:** Quick trend signals for entry/exit decisions
- **Swing Traders:** Multi-timeframe trend analysis
- **Retail Investors:** Simplified trend identification

### Use Cases
- **Trend Following:** Identify momentum breakouts
- **Sector Rotation:** Track relative sector performance
- **Risk Management:** Early trend reversal warnings
- **Educational:** Learn correlation between macro factors and stock trends

## 📚 Learning Objectives

Through this project, developers will learn:
- **Multi-modal ML:** Combining price, sentiment, and macro data
- **Real-time Systems:** WebSocket implementation and live dashboards
- **Indian Market Dynamics:** RBI policy impact, sector correlations
- **Financial NLP:** Processing Indian financial news and terminology
- **Time Series Forecasting:** LSTM networks for financial prediction
- **System Architecture:** Scalable ML pipelines with progress tracking

---

## 🚀 Quick Start Commands

### Initial Setup
```bash
# Clone and setup
git clone <repository-url>
cd event_driven_stock_trend_predictor

# Install Python dependencies  
pip install -r requirements.txt

# Create React frontend (using Vite - modern alternative to CRA)
npm create vite@latest frontend -- --template react
cd frontend && npm install

# Install additional frontend dependencies
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
npm install chart.js react-chartjs-2
npm install plotly.js-dist-min react-plotly.js
npm install axios socket.io-client
npm install @fontsource/roboto

# Configure API keys
cp config/api_credentials.yaml.template config/api_credentials.yaml
# Edit with your API keys

# Download initial data
python scripts/initial_data_download.py
```

### Development Commands
```bash
# Start backend
python main.py

# Start frontend (new terminal)
cd frontend && npm run dev

# Build for production
cd frontend && npm run build
```

### Alternative: Using Next.js (Recommended for Production)
```bash
# Create Next.js app instead of Vite (better for SEO & SSR)
npx create-next-app@latest frontend --typescript --tailwind --eslint --app
cd frontend && npm install

# Install additional dependencies for Next.js
npm install @mui/material @emotion/react @emotion/styled
npm install recharts lucide-react
npm install socket.io-client axios
```

## 📞 Next Development Phase

**Ready to continue development on:**
1. Financial news sentiment analyzer implementation
2. Model training pipeline with progress tracking
3. React dashboard component development
4. Data collection module implementation
5. Real-time WebSocket integration

**Current focus:** Solving the financial news sentiment analysis challenge for Indian market headlines with specialized terminology and complex sentence structures.