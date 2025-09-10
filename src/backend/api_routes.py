"""
FastAPI Backend Routes for Stock Trend Predictor Dashboard
Serves ML model predictions and data to Next.js frontend
"""

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml

# Import your ML models
from src.models.ensemble_predictor import EnsemblePredictor
from src.models.sector_lstm import SectorLSTM
from src.models.macro_xgboost import MacroEconomicXGBoostModel
from src.features.sentiment_processor import IndianFinancialSentimentProcessor
from src.features.technical_indicators import TechnicalIndicatorCalculator
from src.data.macro_collector import MacroEconomicDataCollector

app = FastAPI(
    title="Stock Trend Predictor API",
    description="Multi-modal stock prediction API with technical, macro, and sentiment analysis",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (initialize on startup)
ensemble_models = {}
active_connections: List[WebSocket] = []

# Load stock configuration
with open("config/stock_config.yaml", 'r') as file:
    stock_config = yaml.safe_load(file)
    AVAILABLE_STOCKS = list(stock_config['stocks'].keys())

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    print("Initializing ML models...")
    
    # Initialize ensemble models for each stock
    for stock_key in AVAILABLE_STOCKS:
        try:
            stock_symbol = stock_config['stocks'][stock_key]['indian_symbol']
            sector = stock_config['stocks'][stock_key]['sector']
            
            ensemble = EnsemblePredictor(sector, stock_symbol)
            ensemble_models[stock_key] = ensemble
            
            print(f"Initialized {stock_key} model")
        except Exception as e:
            print(f"Failed to initialize {stock_key}: {e}")
    
    print(f"API ready with {len(ensemble_models)} stock models")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Stock Trend Predictor API",
        "status": "operational",
        "models_loaded": len(ensemble_models),
        "available_stocks": AVAILABLE_STOCKS,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stocks")
async def get_available_stocks():
    """Get list of available stocks with configurations"""
    stocks_info = {}
    
    for stock_key, config in stock_config['stocks'].items():
        stocks_info[stock_key] = {
            "name": config['name'],
            "symbol": config['indian_symbol'],
            "adr_symbol": config.get('adr_symbol'),
            "sector": config['sector'],
            "market_cap_rank": config.get('market_cap_rank'),
            "model_loaded": stock_key in ensemble_models
        }
    
    return stocks_info

@app.get("/api/predict/{stock_symbol}")
async def get_prediction(stock_symbol: str):
    """Get ensemble prediction for a stock"""
    # Convert symbol to stock key
    stock_key = stock_symbol.replace('.NS', '').replace('.', '').upper()
    
    if stock_key not in ensemble_models:
        raise HTTPException(status_code=404, detail=f"Model not found for {stock_symbol}")
    
    try:
        ensemble = ensemble_models[stock_key]
        
        # Get ensemble prediction
        prediction = ensemble.predict_ensemble_with_fallback(stock_symbol)
        
        if 'error' in prediction:
            raise HTTPException(status_code=500, detail=prediction['error'])
        
        return {
            "success": True,
            "stock_symbol": stock_symbol,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/technical-analysis/{stock_symbol}")
async def get_technical_analysis(stock_symbol: str, days: int = 100):
    """Get detailed technical analysis for a stock"""
    try:
        calc = TechnicalIndicatorCalculator()
        df = calc.calculate_all_indicators(stock_symbol, days)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {stock_symbol}")
        
        # Get latest technical indicators
        latest = df.iloc[-1]
        
        technical_data = {
            "stock_symbol": stock_symbol,
            "current_price": float(latest['close']),
            "indicators": {
                "rsi_14": float(latest.get('RSI_14', 0)),
                "macd": float(latest.get('MACD', 0)),
                "macd_signal": float(latest.get('MACD_Signal', 0)),
                "bb_upper": float(latest.get('BB_Upper', 0)),
                "bb_lower": float(latest.get('BB_Lower', 0)),
                "sma_20": float(latest.get('SMA_20', 0)),
                "sma_50": float(latest.get('SMA_50', 0)),
                "volume": float(latest.get('volume', 0)),
                "adx": float(latest.get('ADX', 0))
            },
            "chart_data": {
                "dates": df['date'].dt.strftime('%Y-%m-%d').tolist()[-50:],
                "prices": df['close'].tolist()[-50:],
                "volume": df['volume'].tolist()[-50:],
                "sma_20": df.get('SMA_20', [0]*len(df)).tolist()[-50:],
                "sma_50": df.get('SMA_50', [0]*len(df)).tolist()[-50:],
                "bb_upper": df.get('BB_Upper', [0]*len(df)).tolist()[-50:],
                "bb_lower": df.get('BB_Lower', [0]*len(df)).tolist()[-50:]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return technical_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment/{stock_symbol}")
async def get_sentiment_analysis(stock_symbol: str, days: int = 7):
    """Get news sentiment analysis for a stock"""
    try:
        processor = IndianFinancialSentimentProcessor()
        sentiment_data = processor.process_stock_sentiment(stock_symbol, days)
        
        return {
            "success": True,
            "stock_symbol": stock_symbol,
            "sentiment": sentiment_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/macro-factors/{stock_symbol}")
async def get_macro_factors(stock_symbol: str):
    """Get macro economic factors affecting a stock"""
    try:
        collector = MacroEconomicDataCollector()
        macro_features = collector.get_macro_features_for_stock(stock_symbol)
        
        # Get additional macro context
        macro_summary = collector.get_macro_summary()
        
        return {
            "success": True,
            "stock_symbol": stock_symbol,
            "macro_features": macro_features,
            "macro_summary": macro_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/{stock_symbol}")
async def train_model(stock_symbol: str, background_tasks: BackgroundTasks, epochs: int = 50):
    """Trigger model training for a stock"""
    stock_key = stock_symbol.replace('.NS', '').replace('.', '').upper()
    
    if stock_key not in ensemble_models:
        raise HTTPException(status_code=404, detail=f"Model not found for {stock_symbol}")
    
    # Add training task to background with epochs parameter
    background_tasks.add_task(train_model_background, stock_symbol, stock_key, epochs)
    
    return {
        "success": True,
        "message": f"Training started for {stock_symbol} with {epochs} epochs",
        "stock_symbol": stock_symbol,
        "epochs": epochs,
        "timestamp": datetime.now().isoformat()
    }

# Also update the background training function (around line 195):
async def train_model_background(stock_symbol: str, stock_key: str, epochs: int = 50):
    """Background task to train model"""
    try:
        ensemble = ensemble_models[stock_key]
        
        # Send training start notification
        await broadcast_message({
            "type": "training_started",
            "stock_symbol": stock_symbol,
            "epochs": epochs,
            "message": f"Training started for {stock_symbol} with {epochs} epochs"
        })
        
        # Train all models with custom epochs and verbose output
        results = ensemble.train_all_models(stock_symbol, epochs=epochs, verbose=True)
        
        # Send training complete notification
        await broadcast_message({
            "type": "training_completed",
            "stock_symbol": stock_symbol,
            "results": results,
            "message": f"Training completed for {stock_symbol}"
        })
        
    except Exception as e:
        await broadcast_message({
            "type": "training_error",
            "stock_symbol": stock_symbol,
            "error": str(e),
            "message": f"Training failed for {stock_symbol}"
        })


@app.get("/api/model-performance/{stock_symbol}")
async def get_model_performance(stock_symbol: str):
    """Get model performance metrics"""
    stock_key = stock_symbol.replace('.NS', '').replace('.', '').upper()
    
    if stock_key not in ensemble_models:
        raise HTTPException(status_code=404, detail=f"Model not found for {stock_symbol}")
    
    try:
        ensemble = ensemble_models[stock_key]
        
        # Get model performance data
        performance_data = {
            "stock_symbol": stock_symbol,
            "model_performances": ensemble.model_performances,
            "current_weights": ensemble.weights,
            "sector": ensemble.sector,
            "timestamp": datetime.now().isoformat()
        }
        
        return performance_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle different message types
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                stock_symbol = message.get("stock_symbol")
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "stock_symbol": stock_symbol,
                    "message": f"Subscribed to {stock_symbol} updates"
                }))
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if active_connections:
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove dead connections
                active_connections.remove(connection)

@app.get("/api/market-status")
async def get_market_status():
    """Get current market status and system health"""
    return {
        "market_open": True,  # You can implement actual market hours check
        "system_status": "operational",
        "models_loaded": len(ensemble_models),
        "active_connections": len(active_connections),
        "last_updated": datetime.now().isoformat(),
        "available_stocks": AVAILABLE_STOCKS
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.backend.api_routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )