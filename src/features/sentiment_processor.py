"""
Financial Sentiment Processor for Indian Stock Market
Handles complex Indian financial headlines and extracts sentiment signals
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced NLP libraries
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from textblob import TextBlob
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    print("Advanced NLP libraries not found. Using basic sentiment analysis.")
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = False

class IndianFinancialSentimentProcessor:
    def __init__(self, config_path: str = "config/api_credentials.yaml"):
        """Initialize sentiment processor with Indian financial market focus"""
        self.load_config(config_path)
        self.load_stock_config()
        self.setup_sentiment_models()
        self.setup_indian_financial_patterns()
        
    def load_config(self, config_path: str):
        """Load API credentials"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.news_api_key = config['news_api']['api_key']
                print(f"Loaded NewsAPI key: {self.news_api_key[:8]}...")
        except Exception as e:
            print(f"Error loading config: {e}")
            self.news_api_key = None
    
    def load_stock_config(self):
        """Load stock-specific configurations"""
        try:
            with open("config/stock_config.yaml", 'r') as file:
                config = yaml.safe_load(file)
                self.stock_configs = config['stocks']
                print(f"Loaded configurations for {len(self.stock_configs)} stocks")
        except Exception as e:
            print(f"Error loading stock config: {e}")
            self.stock_configs = {}
    
    def setup_sentiment_models(self):
        """Setup sentiment analysis models"""
        self.sentiment_models = {}
        
        # Basic TextBlob sentiment (always available)
        self.sentiment_models['textblob'] = TextBlob
        
        # VADER sentiment for financial text
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.sentiment_models['vader'] = SentimentIntensityAnalyzer()
                print("VADER sentiment analyzer loaded")
            except:
                print("VADER failed to load")
        
        # FinBERT for financial sentiment (if available)
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.sentiment_models['finbert'] = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
                print("FinBERT model loaded successfully")
            except Exception as e:
                print(f"FinBERT not available: {e}")
                self.sentiment_models['finbert'] = None
    
    def setup_indian_financial_patterns(self):
        """Setup patterns for Indian financial terminology"""
        self.bullish_patterns = [
            # Positive price movements
            r'(?:jumps?|surge[sd]?|gain[s]?|rise[s]?|up|high(?:er)?)\s+(?:\d+(?:\.\d+)?%?)',
            r'(?:target|price target)\s+(?:raised?|increased?|upgrade[d]?)',
            r'(?:buy|strong buy|outperform|overweight)',
            
            # Corporate actions (positive)
            r'(?:bonus|dividend|split)(?:\s+(?:declared?|announced?))?',
            r'(?:merger|acquisition|deal)(?:\s+(?:approved?|completed?))?',
            
            # Business performance
            r'(?:profit|earnings|revenue|sales)\s+(?:growth|increase|beat|above)',
            r'(?:expansion|new\s+(?:plant|facility|project))',
            
            # Sector/policy positive
            r'(?:policy|regulation)\s+(?:support|favorable|positive)',
            r'(?:rbi|government)\s+(?:support|stimulus|boost)',
        ]
        
        self.bearish_patterns = [
            # Negative price movements
            r'(?:falls?|drops?|decline[s]?|down|crash|plunge[s]?)\s+(?:\d+(?:\.\d+)?%?)',
            r'(?:target|price target)\s+(?:cut|reduced?|downgrade[d]?)',
            r'(?:sell|strong sell|underperform|underweight)',
            
            # Corporate issues
            r'(?:loss|losses|deficit|debt|liability)',
            r'(?:investigation|probe|scandal|fraud)',
            r'(?:layoff|job cut|restructur)',
            
            # Regulatory issues
            r'(?:penalty|fine|violation|ban)',
            r'(?:rbi|sebi)\s+(?:action|warning|restriction)',
        ]
        
        # Indian financial terminology
        self.indian_terms = {
            'crore': 10000000,  # 1 crore = 10 million
            'lakh': 100000,     # 1 lakh = 100 thousand
            'npa': 'non-performing assets',
            'casa': 'current account savings account',
            'rbi': 'reserve bank of india',
            'sebi': 'securities exchange board of india',
            'nse': 'national stock exchange',
            'bse': 'bombay stock exchange',
        }
        
        # Compile patterns for efficiency
        self.bullish_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.bullish_patterns]
        self.bearish_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.bearish_patterns]
    
    def collect_stock_news(self, stock_symbol: str, days: int = 7) -> List[Dict]:
        """Collect news for specific stock using configured keywords"""
        if not self.news_api_key:
            print("No NewsAPI key available")
            return []
        
        # Get stock configuration
        stock_key = stock_symbol.replace('.NS', '').replace('.', '').upper()
        if stock_key not in self.stock_configs:
            print(f"No configuration found for {stock_symbol}")
            return []
        
        stock_config = self.stock_configs[stock_key]
        keywords = stock_config.get('news_keywords', [])
        
        all_articles = []
        
        # Search for each keyword
        for keyword in keywords:
            try:
                articles = self._fetch_news_for_keyword(keyword, days)
                all_articles.extend(articles)
                print(f"Found {len(articles)} articles for '{keyword}'")
            except Exception as e:
                print(f"Error fetching news for '{keyword}': {e}")
        
        # Remove duplicates and sort by date
        unique_articles = self._deduplicate_articles(all_articles)
        print(f"Total unique articles: {len(unique_articles)}")
        
        return unique_articles
    
    def _fetch_news_for_keyword(self, keyword: str, days: int) -> List[Dict]:
        """Fetch news from NewsAPI for a specific keyword"""
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': keyword,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.news_api_key,
            'domains': 'economictimes.indiatimes.com,business-standard.com,livemint.com,moneycontrol.com,reuters.com,bloomberg.com'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get('articles', [])
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles and clean data"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article.get('title', '').strip().lower()
            if title and title not in seen_titles and len(title) > 20:
                seen_titles.add(title)
                unique_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', '')
                })
        
        return sorted(unique_articles, key=lambda x: x['publishedAt'], reverse=True)
    
    def analyze_headline_sentiment(self, headline: str) -> Dict:
        """Analyze sentiment of a financial headline with Indian market context"""
        sentiment_scores = {}
        
        # Clean and preprocess headline
        cleaned_headline = self._preprocess_indian_text(headline)
        
        # Pattern-based analysis (Indian financial patterns)
        pattern_score = self._analyze_with_patterns(cleaned_headline)
        sentiment_scores['pattern_based'] = pattern_score
        
        # TextBlob sentiment
        try:
            blob = TextBlob(cleaned_headline)
            sentiment_scores['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            sentiment_scores['textblob'] = {'polarity': 0, 'subjectivity': 0.5}
        
        # VADER sentiment (if available)
        if 'vader' in self.sentiment_models and self.sentiment_models['vader']:
            try:
                vader_scores = self.sentiment_models['vader'].polarity_scores(cleaned_headline)
                sentiment_scores['vader'] = vader_scores
            except:
                sentiment_scores['vader'] = {'compound': 0}
        
        # FinBERT sentiment (if available)
        if self.sentiment_models.get('finbert'):
            try:
                finbert_result = self.sentiment_models['finbert'](cleaned_headline)[0]
                sentiment_scores['finbert'] = {
                    'label': finbert_result['label'],
                    'score': finbert_result['score']
                }
            except:
                sentiment_scores['finbert'] = {'label': 'neutral', 'score': 0.5}
        
        # Combine scores into final sentiment
        final_sentiment = self._combine_sentiment_scores(sentiment_scores)
        
        return {
            'headline': headline,
            'cleaned_headline': cleaned_headline,
            'individual_scores': sentiment_scores,
            'final_sentiment': final_sentiment['sentiment'],
            'confidence': final_sentiment['confidence'],
            'bullish_score': final_sentiment['bullish_score'],
            'bearish_score': final_sentiment['bearish_score']
        }
    
    def _preprocess_indian_text(self, text: str) -> str:
        """Preprocess text with Indian financial terminology"""
        text = text.lower().strip()
        
        # Replace Indian financial terms
        for term, expansion in self.indian_terms.items():
            if isinstance(expansion, str):
                text = re.sub(rf'\b{term}\b', expansion, text)
        
        # Handle Indian number formatting
        text = re.sub(r'rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)', r'inr \1', text)
        text = re.sub(r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', r'inr \1', text)
        
        return text
    
    def _analyze_with_patterns(self, text: str) -> Dict:
        """Analyze sentiment using Indian financial patterns"""
        bullish_matches = sum(1 for pattern in self.bullish_regex if pattern.search(text))
        bearish_matches = sum(1 for pattern in self.bearish_regex if pattern.search(text))
        
        total_matches = bullish_matches + bearish_matches
        
        if total_matches == 0:
            return {'sentiment': 'neutral', 'score': 0.0, 'matches': 0}
        
        bullish_ratio = bullish_matches / total_matches
        bearish_ratio = bearish_matches / total_matches
        
        if bullish_ratio > bearish_ratio:
            sentiment = 'bullish'
            score = bullish_ratio
        elif bearish_ratio > bullish_ratio:
            sentiment = 'bearish'
            score = -bearish_ratio
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'bullish_matches': bullish_matches,
            'bearish_matches': bearish_matches,
            'total_matches': total_matches
        }
    
    def _combine_sentiment_scores(self, scores: Dict) -> Dict:
        """Combine multiple sentiment scores into final sentiment"""
        # Extract individual scores
        pattern_score = scores.get('pattern_based', {}).get('score', 0)
        textblob_score = scores.get('textblob', {}).get('polarity', 0)
        vader_score = scores.get('vader', {}).get('compound', 0)
        
        # FinBERT score processing
        finbert_score = 0
        if 'finbert' in scores:
            finbert_data = scores['finbert']
            if finbert_data['label'].lower() == 'positive':
                finbert_score = finbert_data['score']
            elif finbert_data['label'].lower() == 'negative':
                finbert_score = -finbert_data['score']
        
        # Weighted combination (Indian market focus)
        weights = {
            'pattern': 0.4,    # High weight for Indian patterns
            'finbert': 0.3,    # Financial BERT
            'vader': 0.2,      # VADER for financial text
            'textblob': 0.1    # Basic sentiment
        }
        
        combined_score = (
            pattern_score * weights['pattern'] +
            finbert_score * weights['finbert'] +
            vader_score * weights['vader'] +
            textblob_score * weights['textblob']
        )
        
        # Determine sentiment and confidence
        if combined_score > 0.1:
            sentiment = 'bullish'
            confidence = min(combined_score, 1.0)
        elif combined_score < -0.1:
            sentiment = 'bearish'  
            confidence = min(abs(combined_score), 1.0)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'combined_score': combined_score,
            'bullish_score': max(0, combined_score),
            'bearish_score': max(0, -combined_score)
        }
    
    def process_stock_sentiment(self, stock_symbol: str, days: int = 7) -> Dict:
        """Process complete sentiment analysis for a stock"""
        print(f"Processing sentiment for {stock_symbol}...")
        
        # Collect news
        articles = self.collect_stock_news(stock_symbol, days)
        
        if not articles:
            return {
                'stock_symbol': stock_symbol,
                'sentiment_summary': 'neutral',
                'confidence': 0.5,
                'article_count': 0,
                'bullish_score': 0.5,
                'bearish_score': 0.5,
                'articles_analyzed': []
            }
        
        # Analyze each article
        analyzed_articles = []
        sentiment_scores = []
        
        for article in articles[:20]:  # Limit to recent 20 articles
            title_analysis = self.analyze_headline_sentiment(article['title'])
            
            # Also analyze description if available
            desc_analysis = None
            if article.get('description'):
                desc_analysis = self.analyze_headline_sentiment(article['description'])
            
            article_sentiment = {
                'title': article['title'],
                'publishedAt': article['publishedAt'],
                'source': article['source'],
                'title_sentiment': title_analysis,
                'description_sentiment': desc_analysis,
                # Combined article sentiment (title weighted higher)
                'article_sentiment': title_analysis['final_sentiment'],
                'article_confidence': title_analysis['confidence']
            }
            
            analyzed_articles.append(article_sentiment)
            sentiment_scores.append({
                'sentiment': title_analysis['final_sentiment'],
                'bullish_score': title_analysis['bullish_score'],
                'bearish_score': title_analysis['bearish_score'],
                'confidence': title_analysis['confidence']
            })
        
        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(sentiment_scores)
        
        return {
            'stock_symbol': stock_symbol,
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'days_analyzed': days,
            'article_count': len(articles),
            'articles_processed': len(analyzed_articles),
            'sentiment_summary': overall_sentiment['sentiment'],
            'confidence': overall_sentiment['confidence'],
            'bullish_score': overall_sentiment['bullish_score'],
            'bearish_score': overall_sentiment['bearish_score'],
            'sentiment_distribution': overall_sentiment['distribution'],
            'articles_analyzed': analyzed_articles[:10]  # Return top 10 for review
        }
    
    def _calculate_overall_sentiment(self, sentiment_scores: List[Dict]) -> Dict:
        """Calculate overall sentiment from individual article sentiments"""
        if not sentiment_scores:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'bullish_score': 0.5,
                'bearish_score': 0.5,
                'distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0}
            }
        
        # Count sentiment distribution
        sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        total_bullish_score = 0
        total_bearish_score = 0
        total_confidence = 0
        
        for score in sentiment_scores:
            sentiment_counts[score['sentiment']] += 1
            total_bullish_score += score['bullish_score']
            total_bearish_score += score['bearish_score']
            total_confidence += score['confidence']
        
        total_articles = len(sentiment_scores)
        
        # Calculate averages
        avg_bullish = total_bullish_score / total_articles
        avg_bearish = total_bearish_score / total_articles
        avg_confidence = total_confidence / total_articles
        
        # Determine overall sentiment
        bullish_ratio = sentiment_counts['bullish'] / total_articles
        bearish_ratio = sentiment_counts['bearish'] / total_articles
        
        if bullish_ratio > 0.6 or (bullish_ratio > bearish_ratio and avg_bullish > 0.6):
            overall_sentiment = 'bullish'
        elif bearish_ratio > 0.6 or (bearish_ratio > bullish_ratio and avg_bearish > 0.6):
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'bullish_score': avg_bullish,
            'bearish_score': avg_bearish,
            'distribution': {
                'bullish': bullish_ratio,
                'bearish': bearish_ratio, 
                'neutral': sentiment_counts['neutral'] / total_articles
            }
        }
    
    def generate_sentiment_features(self, stock_symbol: str, days: int = 7) -> Dict:
        """Generate sentiment features for ML model integration"""
        sentiment_data = self.process_stock_sentiment(stock_symbol, days)
        
        # Convert to ML features
        features = {
            'news_sentiment_score': sentiment_data['bullish_score'] - sentiment_data['bearish_score'],
            'news_confidence': sentiment_data['confidence'],
            'news_article_count': sentiment_data['article_count'],
            'news_bullish_ratio': sentiment_data['sentiment_distribution']['bullish'],
            'news_bearish_ratio': sentiment_data['sentiment_distribution']['bearish'],
            'news_neutral_ratio': sentiment_data['sentiment_distribution']['neutral'],
            'news_sentiment_strength': abs(sentiment_data['bullish_score'] - sentiment_data['bearish_score']),
            'has_recent_news': 1 if sentiment_data['article_count'] > 0 else 0
        }
        
        return features

def test_sentiment_processor():
    """Test the sentiment processor with HDFC Bank"""
    print("Testing Financial Sentiment Processor")
    print("=" * 50)
    
    processor = IndianFinancialSentimentProcessor()
    
    # Test 1: Single headline analysis
    print("\n1. Testing headline analysis...")
    test_headlines = [
        "HDFC Bank Q3 Results: Net Profit Jumps 33% to Rs 16,373 Crore, Beats Estimates",
        "RBI Policy: Interest Rates Cut by 25 bps, Banking Stocks Rally",
        "HDFC Bank Faces NPA Concerns as Economic Slowdown Hits Credit Growth",
        "1:1 Bonus Issue: HDFC Bank Board Approves Stock Split, Shares Up 5%"
    ]
    
    for headline in test_headlines:
        result = processor.analyze_headline_sentiment(headline)
        print(f"\nHeadline: {headline}")
        print(f"Sentiment: {result['final_sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Bullish Score: {result['bullish_score']:.2f}")
    
    # Test 2: Stock sentiment analysis
    print(f"\n2. Testing stock sentiment analysis for HDFCBANK.NS...")
    sentiment_result = processor.process_stock_sentiment('HDFCBANK.NS', days=3)
    
    print(f"Articles analyzed: {sentiment_result['articles_processed']}")
    print(f"Overall sentiment: {sentiment_result['sentiment_summary']}")
    print(f"Confidence: {sentiment_result['confidence']:.2f}")
    print(f"Bullish score: {sentiment_result['bullish_score']:.2f}")
    print(f"Bearish score: {sentiment_result['bearish_score']:.2f}")
    
    # Test 3: ML features generation
    print(f"\n3. Testing ML features generation...")
    features = processor.generate_sentiment_features('HDFCBANK.NS', days=3)
    
    print("Generated features for ML model:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.3f}")

if __name__ == "__main__":
    test_sentiment_processor()