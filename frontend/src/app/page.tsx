'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { TrendingUp, TrendingDown, Minus, Activity, Brain, DollarSign, Newspaper, AlertTriangle } from 'lucide-react'

// FIX 1: Updated the StockData interface for robustness
interface StockData {
  symbol: string
  name: string
  sector: string
  current_price: number
  prediction: {
    predicted_trend: string
    confidence: number
    warning?: string // Added warning field
    ensemble_probabilities: {
      bearish: number
      sideways: number  
      bullish: number
    }
    // Made technical and macro predictions optional
    individual_predictions: {
      technical?: { trend: string; confidence: number; weight_used: number }
      macro?: { trend: string; confidence: number; weight_used: number }
      sentiment: { trend: string; confidence: number; weight_used: number }
    }
    model_consensus: {
      agreement_level: number
      is_strong_consensus: boolean
    }
  }
  technical_data: {
    indicators: {
      rsi_14: number
      macd: number
      sma_20: number
      sma_50: number
      volume: number
    }
    chart_data: {
      dates: string[]
      prices: number[]
      sma_20: number[]
      sma_50: number[]
    }
  }
  sentiment_data: {
    sentiment_summary: string
    confidence: number
    articles_processed: number
    bullish_score: number
    bearish_score: number
  }
  macro_data: {
    usd_inr_rate: number
    repo_rate: number
    brent_oil_price: number // Corrected from wti_oil_price to brent_oil_price
    sp500_level: number
  }
}

const AVAILABLE_STOCKS = [
  { key: 'HDFCBANK', name: 'HDFC Bank', symbol: 'HDFCBANK.NS', sector: 'Banking' },
  { key: 'ICICIBANK', name: 'ICICI Bank', symbol: 'ICICIBANK.NS', sector: 'Banking' },
  { key: 'INFY', name: 'Infosys', symbol: 'INFY.NS', sector: 'IT Services' },
  { key: 'TATAMOTORS', name: 'Tata Motors', symbol: 'TATAMOTORS.NS', sector: 'Automotive' },
  { key: 'RELIANCE', name: 'Reliance', symbol: 'RELIANCE.NS', sector: 'Conglomerate' }
]

export default function StockDashboard() {
  const [selectedStock, setSelectedStock] = useState('HDFCBANK')
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isTraining, setIsTraining] = useState(false)

  const fetchStockData = async (stockKey: string) => {
    setLoading(true)
    setError(null)
    
    try {
      const stockInfo = AVAILABLE_STOCKS.find(s => s.key === stockKey)
      if (!stockInfo) return

      const [predictionRes, technicalRes, sentimentRes, macroRes] = await Promise.all([
        fetch(`http://localhost:8000/api/predict/${stockInfo.symbol}`),
        fetch(`http://localhost:8000/api/technical-analysis/${stockInfo.symbol}`),
        fetch(`http://localhost:8000/api/sentiment/${stockInfo.symbol}`),
        fetch(`http://localhost:8000/api/macro-factors/${stockInfo.symbol}`)
      ])

      const [prediction, technical, sentiment, macro] = await Promise.all([
        predictionRes.json(),
        technicalRes.json(),
        sentimentRes.json(),
        macroRes.json()
      ])

      setStockData({
        symbol: stockInfo.symbol,
        name: stockInfo.name,
        sector: stockInfo.sector,
        current_price: technical.current_price,
        prediction: prediction.prediction,
        technical_data: technical,
        sentiment_data: sentiment.sentiment,
        macro_data: macro.macro_features
      })

    } catch (err) {
      setError('Failed to fetch stock data. Ensure the backend is running and reachable.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const trainModel = async () => {
    if (!stockData) return
    
    setIsTraining(true)
    try {
      await fetch(`http://localhost:8000/api/train/${stockData.symbol}`, {
        method: 'POST'
      })
      setTimeout(() => fetchStockData(selectedStock), 5000)
    } catch (err) {
      console.error('Training failed:', err)
    } finally {
      setIsTraining(false)
    }
  }

  useEffect(() => {
    fetchStockData(selectedStock)
  }, [selectedStock])

  const getTrendIcon = (trend: string) => {
    switch (trend?.toLowerCase()) {
      case 'bullish': return <TrendingUp className="h-4 w-4 text-green-600" />
      case 'bearish': return <TrendingDown className="h-4 w-4 text-red-600" />
      default: return <Minus className="h-4 w-4 text-yellow-600" />
    }
  }

  const getTrendColor = (trend: string) => {
    switch (trend?.toLowerCase()) {
      case 'bullish': return 'bg-green-100 text-green-800'
      case 'bearish': return 'bg-red-100 text-red-800'
      case 'neutral': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto text-center py-12">
          <Activity className="h-12 w-12 animate-spin mx-auto text-blue-600" />
          <p className="mt-4 text-lg text-gray-600">Loading stock analysis...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto text-center py-12">
          <p className="text-red-600 text-lg">{error}</p>
          <Button onClick={() => fetchStockData(selectedStock)} className="mt-4">
            Retry
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Multi-Modal Stock Predictor</h1>
            <p className="text-gray-600">AI-powered predictions using Technical, Sentiment, and Macroeconomic data.</p>
          </div>
          <div className="flex gap-4 items-center">
            <Select value={selectedStock} onValueChange={setSelectedStock}>
              <SelectTrigger className="w-48"><SelectValue placeholder="Select stock" /></SelectTrigger>
              <SelectContent>
                {AVAILABLE_STOCKS.map(stock => (
                  <SelectItem key={stock.key} value={stock.key}>{stock.name} ({stock.sector})</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button onClick={trainModel} disabled={isTraining} className="bg-blue-600 hover:bg-blue-700">
              {isTraining ? 'Training...' : 'Retrain Model'}
            </Button>
          </div>
        </div>

        {stockData && (
          <>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div>
                    <span className="text-2xl">{stockData.name}</span>
                    <Badge className="ml-2" variant="secondary">{stockData.sector}</Badge>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold">₹{stockData.current_price.toFixed(2)}</div>
                    <div className="text-sm text-gray-500">{stockData.symbol}</div>
                  </div>
                </CardTitle>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Brain className="h-5 w-5" />Ensemble Prediction</CardTitle>
              </CardHeader>
              <CardContent>
                {/* FIX 2: Display warning if present in API response */}
                {stockData.prediction.warning && (
                    <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-center gap-2 text-sm text-yellow-800">
                        <AlertTriangle className="h-4 w-4" />
                        {stockData.prediction.warning}
                    </div>
                )}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      {getTrendIcon(stockData.prediction.predicted_trend)}
                      <Badge className={`text-lg px-4 py-2 ${getTrendColor(stockData.prediction.predicted_trend)}`}>
                        {stockData.prediction.predicted_trend}
                      </Badge>
                    </div>
                    <div className="text-2xl font-bold mb-1">{((stockData.prediction.confidence || 0) * 100).toFixed(1)}%</div>
                    <div className="text-sm text-gray-500">Confidence</div>
                    {stockData.prediction.model_consensus?.is_strong_consensus && (
                      <div className="mt-2"><Badge variant="outline" className="text-green-700 border-green-300">Strong Consensus</Badge></div>
                    )}
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3">Probability Distribution</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-red-600">Bearish</span>
                        <div className="flex-1 mx-3 bg-gray-200 rounded-full h-2"><div className="bg-red-500 h-2 rounded-full" style={{ width: `${(stockData.prediction.ensemble_probabilities?.bearish || 0) * 100}%` }}/></div>
                        <span className="text-sm">{((stockData.prediction.ensemble_probabilities?.bearish || 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-yellow-600">Sideways</span>
                        <div className="flex-1 mx-3 bg-gray-200 rounded-full h-2"><div className="bg-yellow-500 h-2 rounded-full" style={{ width: `${(stockData.prediction.ensemble_probabilities?.sideways || 0) * 100}%` }}/></div>
                        <span className="text-sm">{((stockData.prediction.ensemble_probabilities?.sideways || 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-green-600">Bullish</span>
                        <div className="flex-1 mx-3 bg-gray-200 rounded-full h-2"><div className="bg-green-500 h-2 rounded-full" style={{ width: `${(stockData.prediction.ensemble_probabilities?.bullish || 0) * 100}%` }}/></div>
                        <span className="text-sm">{((stockData.prediction.ensemble_probabilities?.bullish || 0) * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3">Model Contributions</h4>
                    <div className="space-y-2">
                      {/* FIX 3: Used optional chaining (?.) to prevent crashes if a model is missing */}
                      <div className="flex items-center justify-between p-2 bg-blue-50 rounded">
                        <div className="flex items-center gap-2"><Activity className="h-4 w-4 text-blue-600" /><span className="text-sm">Technical</span></div>
                        <div className="text-right">
                          <div className="text-sm font-medium">{stockData.prediction.individual_predictions?.technical?.trend || 'N/A'}</div>
                          <div className="text-xs text-gray-500">{((stockData.prediction.individual_predictions?.technical?.weight_used || 0) * 100).toFixed(0)}% weight</div>
                        </div>
                      </div>
                      <div className="flex items-center justify-between p-2 bg-green-50 rounded">
                        <div className="flex items-center gap-2"><DollarSign className="h-4 w-4 text-green-600" /><span className="text-sm">Macro</span></div>
                        <div className="text-right">
                          <div className="text-sm font-medium">{stockData.prediction.individual_predictions?.macro?.trend || 'N/A'}</div>
                          <div className="text-xs text-gray-500">{((stockData.prediction.individual_predictions?.macro?.weight_used || 0) * 100).toFixed(0)}% weight</div>
                        </div>
                      </div>
                      <div className="flex items-center justify-between p-2 bg-purple-50 rounded">
                        <div className="flex items-center gap-2"><Newspaper className="h-4 w-4 text-purple-600" /><span className="text-sm">Sentiment</span></div>
                        <div className="text-right">
                          <div className="text-sm font-medium">{stockData.prediction.individual_predictions.sentiment.trend}</div>
                          <div className="text-xs text-gray-500">{(stockData.prediction.individual_predictions.sentiment.weight_used * 100).toFixed(0)}% weight</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader><CardTitle>Price Chart with Technical Indicators</CardTitle></CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={stockData.technical_data.chart_data.dates.map((date, i) => ({
                        date: new Date(date).toLocaleDateString(),
                        price: stockData.technical_data.chart_data.prices[i],
                        sma20: stockData.technical_data.chart_data.sma_20[i],
                        sma50: stockData.technical_data.chart_data.sma_50[i]
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={['dataMin - 10', 'dataMax + 10']} />
                        <Tooltip />
                        <Line type="monotone" dataKey="price" stroke="#2563eb" strokeWidth={2} name="Price" dot={false} />
                        <Line type="monotone" dataKey="sma20" stroke="#dc2626" strokeWidth={1} name="SMA 20" dot={false} />
                        <Line type="monotone" dataKey="sma50" stroke="#16a34a" strokeWidth={1} name="SMA 50" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Key Technical Indicators</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{stockData.technical_data.indicators.rsi_14.toFixed(1)}</div>
                      <div className="text-sm text-gray-600">RSI (14)</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{stockData.technical_data.indicators.macd.toFixed(2)}</div>
                      <div className="text-sm text-gray-600">MACD</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">₹{stockData.technical_data.indicators.sma_20.toFixed(2)}</div>
                      <div className="text-sm text-gray-600">SMA 20</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{(stockData.technical_data.indicators.volume / 1000000).toFixed(1)}M</div>
                      <div className="text-sm text-gray-600">Volume</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader><CardTitle className="flex items-center gap-2"><Newspaper className="h-5 w-5" />News Sentiment Analysis</CardTitle></CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="text-center">
                      <Badge className={`text-lg px-4 py-2 ${getTrendColor(stockData.sentiment_data.sentiment_summary)}`}>
                        {stockData.sentiment_data.sentiment_summary.toUpperCase()}
                      </Badge>
                      <div className="mt-2">
                        <span className="text-2xl font-bold">{(stockData.sentiment_data.confidence * 100).toFixed(1)}%</span>
                        <div className="text-sm text-gray-500">Confidence</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="text-center">
                        <div className="text-lg font-semibold text-green-600">{(stockData.sentiment_data.bullish_score * 100).toFixed(1)}%</div>
                        <div>Bullish Score</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold text-red-600">{(stockData.sentiment_data.bearish_score * 100).toFixed(1)}%</div>
                        <div>Bearish Score</div>
                      </div>
                    </div>
                    <div className="text-center text-sm text-gray-600">
                      Based on {stockData.sentiment_data.articles_processed} articles analyzed
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle className="flex items-center gap-2"><DollarSign className="h-5 w-5" />Macro Economic Factors</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{stockData.macro_data.usd_inr_rate.toFixed(2)}</div>
                      <div className="text-sm text-gray-600">USD/INR Rate</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{stockData.macro_data.repo_rate.toFixed(2)}%</div>
                      <div className="text-sm text-gray-600">RBI Repo Rate</div>
                    </div>
                    {/* FIX 4: Corrected key and label for oil price */}
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">${stockData.macro_data.brent_oil_price.toFixed(0)}</div>
                      <div className="text-sm text-gray-600">Brent Oil Price</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{(stockData.macro_data.sp500_level / 1000).toFixed(1)}K</div>
                      <div className="text-sm text-gray-600">S&P 500 Level</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </>
        )}
      </div>
    </div>
  )
}