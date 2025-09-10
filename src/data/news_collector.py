"""
Indian Financial News Collector
Specialized news collection for Indian stock market
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
from typing import Dict, List, Optional, Tuple
import sqlite3
import os

class IndianFinancialNewsCollector:
    def __init__(self, config_path: str = "config/api_credentials.yaml"):
        """Initialize news collector with Indian financial sources"""
        self.load_config(config_path)
        self.setup_database()
        self.setup_indian_sources()
        
    def load_config(self, config_path: str):
        """Load API credentials and settings"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.news_api_key = config['news_api']['api_key']
                self.calls_per_hour = config['news_api'].get('calls_per_hour', 100)
                print(f"Loaded NewsAPI configuration")
        except Exception as e:
            print(f"Error loading config: {e}")
            self.news_api_key = None
    
    def setup_database(self):
        """Setup local database for news storage"""
        os.makedirs("data/news", exist_ok=True)
        self.db_path = "data/news/financial_news.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_symbol TEXT,
                    title TEXT,
                    description TEXT,
                    content TEXT,
                    published_at TEXT,
                    source TEXT,
                    url TEXT UNIQUE,
                    keywords TEXT,
                    sentiment_processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_date 
                ON news_articles(stock_symbol, published_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment 
                ON news_articles(sentiment_processed)
            """)
        
        print(f"News database ready: {self.db_path}")
    
    def setup_indian_sources(self):
        """Setup Indian financial news sources"""
        self.indian_sources = {
            'primary': [
                'economictimes.indiatimes.com',
                'business-standard.com',
                'livemint.com',
                'moneycontrol.com'
            ],
            'secondary': [
                'reuters.com',
                'bloomberg.com',
                'cnbc.com',
                'marketwatch.com'
            ],
            'rss_feeds': [
                'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
                'https://www.business-standard.com/rss/markets-106.rss',
                'https://www.livemint.com/rss/markets'
            ]
        }
        
        # Indian financial terminology for better search
        self.indian_financial_terms = [
            'rbi policy', 'repo rate', 'reverse repo',
            'npa', 'bad loans', 'asset quality',
            'sebi', 'nse', 'bse', 'sensex', 'nifty',
            'fii', 'dii', 'mutual funds',
            'rupee', 'inr', 'currency',
            'gdp growth', 'inflation', 'cpi', 'wpi'
        ]
    
    def collect_stock_news(self, stock_symbol: str, keywords: List[str], days: int = 7) -> List[Dict]:
        """Collect news for specific stock using multiple sources"""
        all_articles = []
        
        # Collect from NewsAPI
        newsapi_articles = self._collect_from_newsapi(keywords, days)
        all_articles.extend(newsapi_articles)
        
        # Store in database
        stored_count = self._store_articles(stock_symbol, all_articles, keywords)
        
        print(f"Collected {len(all_articles)} articles, stored {stored_count} new articles")
        return all_articles
    
    def _collect_from_newsapi(self, keywords: List[str], days: int) -> List[Dict]:
        """Collect from NewsAPI with Indian financial sources"""
        if not self.news_api_key:
            print("No NewsAPI key available")
            return []
        
        all_articles = []
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Combine primary and secondary sources
        domains = ','.join(self.indian_sources['primary'] + self.indian_sources['secondary'])
        
        for keyword in keywords:
            try:
                # Rate limiting
                time.sleep(1)
                
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{keyword}"',  # Exact phrase search
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'domains': domains,
                    'apiKey': self.news_api_key,
                    'pageSize': 50  # Max articles per keyword
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get('articles', [])
                
                print(f"Found {len(articles)} articles for '{keyword}'")
                all_articles.extend(articles)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching news for '{keyword}': {e}")
            except Exception as e:
                print(f"Unexpected error for '{keyword}': {e}")
        
        # Deduplicate articles
        unique_articles = self._deduplicate_articles(all_articles)
        print(f"Total unique articles: {len(unique_articles)}")
        
        return unique_articles
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on URL and title"""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').strip().lower()
            
            # Skip if URL or title already seen
            if url in seen_urls or title in seen_titles:
                continue
            
            # Skip if title is too short or empty
            if not title or len(title) < 20:
                continue
            
            # Skip if article is removed or not accessible
            if 'removed' in title.lower() or not article.get('description'):
                continue
            
            seen_urls.add(url)
            seen_titles.add(title)
            unique_articles.append(article)
        
        return sorted(unique_articles, key=lambda x: x.get('publishedAt', ''), reverse=True)

    def refresh_stock_news(self, stock_symbol: str, keywords: List[str], days: int = 4) -> Tuple[int, int]:
        """
        Deletes old news for a stock and fetches fresh articles for the past few days.
        This is the recommended method for daily updates.
        """
        # 1. Delete old articles (e.g., older than 14 days) to keep DB clean
        cutoff_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        deleted_count = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM news_articles 
                WHERE stock_symbol = ? AND date(published_at) < date(?)
            """, (stock_symbol, cutoff_date))
            deleted_count = cursor.rowcount

        # 2. Collect new articles for the recent period
        new_articles = self._collect_from_newsapi(keywords, days)
        stored_count = self._store_articles(stock_symbol, new_articles, keywords)
        
        return deleted_count, stored_count
    
    def _store_articles(self, stock_symbol: str, articles: List[Dict], keywords: List[str]) -> int:
        """Store articles in database, avoiding duplicates"""
        if not articles:
            return 0
        
        stored_count = 0
        keywords_str = ','.join(keywords)
        
        with sqlite3.connect(self.db_path) as conn:
            for article in articles:
                try:
                    # Clean and prepare data
                    title = article.get('title', '').strip()
                    description = article.get('description', '').strip()
                    content = article.get('content', '').strip()
                    published_at = article.get('publishedAt', '')
                    source = article.get('source', {}).get('name', '')
                    url = article.get('url', '')
                    
                    # Skip if essential fields are missing
                    if not title or not url:
                        continue
                    
                    # Insert article (URL is unique, so duplicates will be ignored)
                    conn.execute("""
                        INSERT OR IGNORE INTO news_articles 
                        (stock_symbol, title, description, content, published_at, source, url, keywords)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (stock_symbol, title, description, content, published_at, source, url, keywords_str))
                    
                    if conn.total_changes > 0:
                        stored_count += 1
                        
                except Exception as e:
                    print(f"Error storing article: {e}")
                    continue
        
        return stored_count
    
    def get_stored_news(self, stock_symbol: str, days: int = 7, limit: int = 50) -> List[Dict]:
        """Retrieve stored news for a stock"""
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM news_articles 
                WHERE stock_symbol = ? AND date(published_at) >= date(?)
                ORDER BY published_at DESC
                LIMIT ?
            """, (stock_symbol, from_date, limit))
            
            articles = [dict(row) for row in cursor.fetchall()]
        
        return articles
    
    def collect_market_wide_news(self, days: int = 3) -> Dict:
        """Collect market-wide Indian financial news"""
        market_keywords = [
            'RBI policy', 'repo rate', 'interest rates',
            'Sensex', 'Nifty', 'BSE', 'NSE',
            'FII', 'DII', 'foreign investment',
            'India GDP', 'inflation', 'rupee'
        ]
        
        market_articles = self._collect_from_newsapi(market_keywords, days)
        stored_count = self._store_articles('MARKET_WIDE', market_articles, market_keywords)
        
        return {
            'total_articles': len(market_articles),
            'stored_articles': stored_count,
            'articles': market_articles[:10]  # Return top 10
        }
    
    def get_news_summary(self, stock_symbol: str = None, days: int = 7) -> Dict:
        """Get summary of collected news"""
        with sqlite3.connect(self.db_path) as conn:
            if stock_symbol:
                cursor = conn.execute("""
                    SELECT COUNT(*) as total,
                           COUNT(CASE WHEN date(published_at) >= date('now', '-{} days') THEN 1 END) as recent,
                           MIN(published_at) as earliest,
                           MAX(published_at) as latest
                    FROM news_articles 
                    WHERE stock_symbol = ?
                """.format(days), (stock_symbol,))
            else:
                cursor = conn.execute("""
                    SELECT COUNT(*) as total,
                           COUNT(CASE WHEN date(published_at) >= date('now', '-{} days') THEN 1 END) as recent,
                           MIN(published_at) as earliest,
                           MAX(published_at) as latest
                    FROM news_articles
                """.format(days))
            
            result = cursor.fetchone()
            
            # Get top sources
            cursor = conn.execute("""
                SELECT source, COUNT(*) as count
                FROM news_articles 
                WHERE stock_symbol = ? OR ? IS NULL
                GROUP BY source
                ORDER BY count DESC
                LIMIT 5
            """, (stock_symbol, stock_symbol))
            
            top_sources = cursor.fetchall()
        
        return {
            'stock_symbol': stock_symbol or 'ALL',
            'total_articles': result[0],
            'recent_articles': result[1],
            'earliest_date': result[2],
            'latest_date': result[3],
            'top_sources': [{'source': s[0], 'count': s[1]} for s in top_sources]
        }

def test_news_collector():
    """Test the news collector"""
    print("Testing Indian Financial News Collector")
    print("=" * 50)
    
    collector = IndianFinancialNewsCollector()
    
    # Test 1: Collect HDFC Bank news
    print("\n1. Testing HDFC Bank news collection...")
    hdfc_keywords = ["HDFC Bank", "private banking", "RBI policy"]
    articles = collector.collect_stock_news('HDFCBANK.NS', hdfc_keywords, days=3)
    
    print(f"Collected {len(articles)} articles for HDFC Bank")
    if articles:
        print(f"Latest article: {articles[0].get('title', 'N/A')}")
        print(f"Source: {articles[0].get('source', {}).get('name', 'N/A')}")
    
    # Test 2: Get stored news
    print(f"\n2. Testing stored news retrieval...")
    stored_articles = collector.get_stored_news('HDFCBANK.NS', days=7)
    print(f"Found {len(stored_articles)} stored articles")
    
    # Test 3: Market-wide news
    print(f"\n3. Testing market-wide news collection...")
    market_news = collector.collect_market_wide_news(days=2)
    print(f"Market news: {market_news['total_articles']} total, {market_news['stored_articles']} new")
    
    # Test 4: News summary
    print(f"\n4. Testing news summary...")
    summary = collector.get_news_summary('HDFCBANK.NS')
    print(f"HDFC Bank news summary:")
    print(f"  Total articles: {summary['total_articles']}")
    print(f"  Recent articles: {summary['recent_articles']}")
    print(f"  Top sources: {[s['source'] for s in summary['top_sources'][:3]]}")

if __name__ == "__main__":
    test_news_collector()