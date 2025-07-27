import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json
import concurrent.futures
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Configuration
POSITIONS_FILE = "stock_positions.json"
CHARTS_FILE = "charts.json"
RECOMMENDATIONS_FILE = "stock_recommendations.txt"

class TechnicalAnalyzer:
    """Advanced technical analysis class"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential technical indicators for performance"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Essential Moving Averages (most important)
            df['MA20'] = ta.sma(df['Close'], 20)
            df['MA50'] = ta.sma(df['Close'], 50)
            df['MA200'] = ta.sma(df['Close'], 200)
            
            # Essential EMAs
            df['EMA12'] = ta.ema(df['Close'], 12)
            df['EMA26'] = ta.ema(df['Close'], 26)
            df['EMA50'] = ta.ema(df['Close'], 50)
            
            # RSI (most important oscillator)
            df['RSI'] = ta.rsi(df['Close'], 14)
            
            # MACD (most important trend indicator)
            macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd_data['MACD_12_26_9']
            df['MACD_Signal'] = macd_data['MACDs_12_26_9']
            
            # Bollinger Bands (important for volatility)
            bb_data = ta.bbands(df['Close'], length=20, std=2)
            df['BB_Upper'] = bb_data['BBU_20_2.0']
            df['BB_Lower'] = bb_data['BBL_20_2.0']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Volume indicators (important for confirmation)
            df['Volume_MA'] = ta.sma(df['Volume'], 20)
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # ATR for volatility
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            
            # Support and Resistance levels
            df['Resistance'] = df['High'].rolling(window=20).max()
            df['Support'] = df['Low'].rolling(window=20).min()
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
        
        return df
    
    @staticmethod
    def get_trend_strength(df: pd.DataFrame) -> Dict[str, float]:
        """Analyze trend strength across different timeframes"""
        if df.empty or len(df) < 200:
            return {"short": 0, "medium": 0, "long": 0, "overall": 0}
        
        latest = df.iloc[-1]
        
        # Short-term trend (20 vs 50 day MA)
        short_trend = 0
        if not pd.isna(latest['MA20']) and not pd.isna(latest['MA50']):
            if latest['Close'] > latest['MA20'] > latest['MA50']:
                short_trend = 1
            elif latest['Close'] < latest['MA20'] < latest['MA50']:
                short_trend = -1
        
        # Medium-term trend (50 vs 200 day MA)
        medium_trend = 0
        if not pd.isna(latest['MA50']) and not pd.isna(latest['MA200']):
            if latest['MA50'] > latest['MA200']:
                medium_trend = 1
            elif latest['MA50'] < latest['MA200']:
                medium_trend = -1
        
        # Long-term trend (Price vs 200 day MA)
        long_trend = 0
        if not pd.isna(latest['MA200']):
            if latest['Close'] > latest['MA200']:
                long_trend = 1
            elif latest['Close'] < latest['MA200']:
                long_trend = -1
        
        # Overall trend strength
        overall = (short_trend + medium_trend + long_trend) / 3
        
        return {
            "short": short_trend,
            "medium": medium_trend,
            "long": long_trend,
            "overall": overall
        }

class AdvancedRecommendationEngine:
    """Advanced recommendation engine with multiple factors"""
    
    def __init__(self):
        self.weights = {
            'technical': 0.4,
            'momentum': 0.25,
            'volume': 0.15,
            'sentiment': 0.1,
            'fundamental': 0.1
        }
    
    def generate_recommendation(self, ticker: str, stock_data: pd.DataFrame, 
                             sentiment: float, macro_data: Dict, 
                             micro_data: Dict) -> Tuple[str, List[str], float]:
        """Generate comprehensive recommendation"""
        
        if stock_data.empty:
            return "Hold", ["No data available"], 0
        
        # Calculate component scores
        technical_score = self._calculate_technical_score(stock_data)
        momentum_score = self._calculate_momentum_score(stock_data)
        volume_score = self._calculate_volume_score(stock_data)
        sentiment_score = self._normalize_sentiment(sentiment)
        fundamental_score = self._calculate_fundamental_score(micro_data)
        
        # Weighted final score
        final_score = (
            technical_score * self.weights['technical'] +
            momentum_score * self.weights['momentum'] +
            volume_score * self.weights['volume'] +
            sentiment_score * self.weights['sentiment'] +
            fundamental_score * self.weights['fundamental']
        )
        
        # Generate reasons
        reasons = self._generate_reasons(stock_data, sentiment, macro_data, micro_data)
        
        # Determine recommendation
        if final_score > 0.6:
            recommendation = "Strong Buy"
        elif final_score > 0.2:
            recommendation = "Buy"
        elif final_score > -0.2:
            recommendation = "Hold"
        elif final_score > -0.6:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
        
        return recommendation, reasons, final_score
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculate technical analysis score"""
        if df.empty or len(df) < 50:
            return 0
        
        latest = df.iloc[-1]
        score = 0
        
        # Trend analysis
        trend_strength = TechnicalAnalyzer.get_trend_strength(df)
        score += trend_strength['overall'] * 0.4
        
        # RSI analysis
        if not pd.isna(latest['RSI']):
            if 30 <= latest['RSI'] <= 45:  # Oversold but recovering
                score += 0.3
            elif 55 <= latest['RSI'] <= 70:  # Strong but not overbought
                score += 0.2
            elif latest['RSI'] > 80:  # Overbought
                score -= 0.3
            elif latest['RSI'] < 20:  # Extremely oversold
                score -= 0.2
        
        # MACD analysis
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                score += 0.2
            else:
                score -= 0.2
        
        # Bollinger Bands analysis
        if not pd.isna(latest['BB_Position']):
            if 0.2 <= latest['BB_Position'] <= 0.8:  # In healthy range
                score += 0.1
            elif latest['BB_Position'] < 0.1:  # Near lower band
                score += 0.2  # Potential bounce
            elif latest['BB_Position'] > 0.9:  # Near upper band
                score -= 0.2  # Potential pullback
        
        return np.clip(score, -1, 1)
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score using available indicators"""
        if df.empty or len(df) < 20:
            return 0
        
        latest = df.iloc[-1]
        score = 0
        
        # Price momentum using EMA crossovers
        if not pd.isna(latest['EMA12']) and not pd.isna(latest['EMA26']):
            if latest['EMA12'] > latest['EMA26']:
                score += 0.3  # Bullish crossover
            else:
                score -= 0.3  # Bearish crossover
        
        # RSI momentum
        if not pd.isna(latest['RSI']):
            if latest['RSI'] < 30:
                score += 0.2  # Oversold - potential bounce
            elif latest['RSI'] > 70:
                score -= 0.2  # Overbought - potential reversal
            elif 40 <= latest['RSI'] <= 60:
                score += 0.1  # Neutral zone
        
        # MACD momentum
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                score += 0.2  # Bullish MACD
            else:
                score -= 0.2  # Bearish MACD
        
        # Bollinger Bands momentum
        if not pd.isna(latest['BB_Position']):
            if latest['BB_Position'] < 0.2:
                score += 0.1  # Near lower band - potential bounce
            elif latest['BB_Position'] > 0.8:
                score -= 0.1  # Near upper band - potential reversal
        
        return np.clip(score, -1, 1)
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume-based score using available indicators"""
        if df.empty or len(df) < 20:
            return 0
        
        latest = df.iloc[-1]
        score = 0
        
        # Volume ratio analysis
        if not pd.isna(latest['Volume_Ratio']):
            if latest['Volume_Ratio'] > 1.5:  # High volume
                score += 0.4
            elif latest['Volume_Ratio'] > 1.0:  # Above average volume
                score += 0.2
            elif latest['Volume_Ratio'] < 0.5:  # Low volume
                score -= 0.2
        
        # Price-volume relationship
        if not pd.isna(latest['Volume_Ratio']) and not pd.isna(latest['Close']):
            # Check if price is up with high volume (bullish)
            if len(df) >= 2:
                price_change = (latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']
                if price_change > 0 and latest['Volume_Ratio'] > 1.2:
                    score += 0.3
                elif price_change < 0 and latest['Volume_Ratio'] > 1.2:
                    score -= 0.3
        
        return np.clip(score, -1, 1)
    
    def _normalize_sentiment(self, sentiment: float) -> float:
        """Normalize sentiment score"""
        return np.clip(sentiment * 2, -1, 1)  # Scale sentiment
    
    def _calculate_fundamental_score(self, micro_data: Dict) -> float:
        """Calculate fundamental score"""
        score = 0
        
        # P/E ratio analysis
        pe_ratio = micro_data.get('pe_ratio')
        if pe_ratio:
            if 10 <= pe_ratio <= 20:
                score += 0.3
            elif 20 < pe_ratio <= 30:
                score += 0.1
            elif pe_ratio > 40:
                score -= 0.3
        
        # Dividend yield
        div_yield = micro_data.get('dividend_yield')
        if div_yield and div_yield >= 0.02:
            score += 0.2
        
        return np.clip(score, -1, 1)
    
    def _generate_reasons(self, df: pd.DataFrame, sentiment: float, 
                         macro_data: Dict, micro_data: Dict) -> List[str]:
        """Generate detailed reasons for recommendation"""
        reasons = []
        
        if df.empty:
            return ["No data available"]
        
        latest = df.iloc[-1]
        trend_strength = TechnicalAnalyzer.get_trend_strength(df)
        
        # Trend analysis
        if trend_strength['overall'] > 0.5:
            reasons.append("Strong bullish trend across multiple timeframes")
        elif trend_strength['overall'] < -0.5:
            reasons.append("Strong bearish trend across multiple timeframes")
        elif trend_strength['short'] > 0:
            reasons.append("Short-term bullish trend")
        elif trend_strength['short'] < 0:
            reasons.append("Short-term bearish trend")
        
        # RSI analysis
        if not pd.isna(latest['RSI']):
            if latest['RSI'] < 30:
                reasons.append(f"Oversold (RSI: {latest['RSI']:.1f})")
            elif latest['RSI'] > 70:
                reasons.append(f"Overbought (RSI: {latest['RSI']:.1f})")
            elif 45 <= latest['RSI'] <= 55:
                reasons.append("RSI in neutral zone")
        
        # MACD analysis
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                reasons.append("MACD bullish crossover")
            else:
                reasons.append("MACD bearish crossover")
        
        # Volume analysis
        if not pd.isna(latest['Volume_Ratio']):
            if latest['Volume_Ratio'] > 1.5:
                reasons.append("High volume confirmation")
            elif latest['Volume_Ratio'] < 0.7:
                reasons.append("Low volume concern")
        
        # Bollinger Bands
        if not pd.isna(latest['BB_Position']):
            if latest['BB_Position'] > 0.8:
                reasons.append("Price near upper Bollinger Band")
            elif latest['BB_Position'] < 0.2:
                reasons.append("Price near lower Bollinger Band")
        
        # Sentiment
        if sentiment > 0.1:
            reasons.append("Positive market sentiment")
        elif sentiment < -0.1:
            reasons.append("Negative market sentiment")
        
        # Fundamental factors
        pe_ratio = micro_data.get('pe_ratio')
        if pe_ratio:
            if pe_ratio > 30:
                reasons.append(f"High P/E ratio ({pe_ratio:.1f})")
            elif pe_ratio < 15:
                reasons.append(f"Attractive P/E ratio ({pe_ratio:.1f})")
        
        div_yield = micro_data.get('dividend_yield')
        if div_yield and div_yield >= 0.02:
            reasons.append(f"Attractive dividend yield ({div_yield*100:.1f}%)")
        
        # Macro factors
        if macro_data.get("treasury_yield", 0) > 4.0:
            reasons.append("High Treasury yield environment")
        if macro_data.get("cpi", 0) > 300:
            reasons.append("High inflation environment")
        
        return reasons

def get_stock_data_optimized(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Optimized stock data fetching with error handling"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"No data available for {ticker}")
            return pd.DataFrame()
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        
        # Calculate essential technical indicators
        df = TechnicalAnalyzer.calculate_indicators(df)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_news_sentiment_optimized(query: str = "stock market", days: int = 7) -> Tuple[float, str]:
    """Optimized news sentiment analysis with faster processing"""
    if not NEWS_API_KEY:
        return 0, "No NewsAPI key provided"
    
    try:
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 10  # Reduced from 20
        }
        
        response = requests.get(url, params=params, timeout=5)  # Reduced from 10s
        response.raise_for_status()
        
        articles = response.json().get('articles', [])
        
        if not articles:
            return 0, "No articles found"
        
        sentiment_scores = []
        for article in articles[:8]:  # Reduced from 15
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            
            if text.strip():
                analysis = TextBlob(text)
                sentiment_scores.append(analysis.sentiment.polarity)
        
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            return avg_sentiment, f"Analyzed {len(sentiment_scores)} articles"
        else:
            return 0, "No valid articles for sentiment analysis"
            
    except requests.exceptions.RequestException as e:
        return 0, f"Network error: {str(e)}"
    except Exception as e:
        return 0, f"Error fetching news: {str(e)}"

def get_enhanced_micro_data(ticker: str) -> Tuple[Dict, str]:
    """Optimized fundamental data fetching - only essential data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Only fetch essential fundamental data for performance
        micro_data = {
            "pe_ratio": info.get('trailingPE'),
            "forward_pe": info.get('forwardPE'),
            "dividend_yield": info.get('dividendYield'),
            "market_cap": info.get('marketCap'),
            "beta": info.get('beta'),
            "fifty_two_week_high": info.get('fiftyTwoWeekHigh'),
            "fifty_two_week_low": info.get('fiftyTwoWeekLow'),
            "analyst_target_price": info.get('targetMeanPrice'),
            "recommendation": info.get('recommendationMean')
        }
        
        return micro_data, f"Essential fundamental data fetched for {ticker}"
        
    except Exception as e:
        return {
            "pe_ratio": None,
            "dividend_yield": None,
            "market_cap": None
        }, f"Error fetching data for {ticker}: {str(e)}"

def parallel_data_fetch(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch stock data in parallel for better performance"""
    stock_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(get_stock_data_optimized, ticker): ticker 
            for ticker in tickers
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if not data.empty:
                    stock_data[ticker] = data
                    print(f"✓ Data loaded for {ticker}")
                else:
                    print(f"✗ No data for {ticker}")
            except Exception as e:
                print(f"✗ Error loading {ticker}: {str(e)}")
    
    return stock_data

def generate_enhanced_charts(ticker: str, stock_data: pd.DataFrame) -> Dict:
    """Generate optimized chart data with essential indicators"""
    if stock_data.empty:
        return {"type": "line", "data": {"labels": [], "datasets": []}, "options": {}}
    
    # Get last 20 days of data for faster processing
    recent_data = stock_data.tail(20)
    dates = recent_data.index.strftime("%m-%d").tolist()  # Shorter date format
    
    chart = {
        "type": "line",
        "data": {
            "labels": dates,
            "datasets": [
                {
                    "label": f"{ticker} Close",
                    "data": recent_data['Close'].tolist(),
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "label": "MA20",
                    "data": recent_data['MA20'].fillna(0).tolist(),
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "fill": False,
                    "borderWidth": 2
                },
                {
                    "label": "MA50",
                    "data": recent_data['MA50'].fillna(0).tolist(),
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "fill": False,
                    "borderWidth": 2
                },
                {
                    "label": "BB Upper",
                    "data": recent_data['BB_Upper'].fillna(0).tolist(),
                    "borderColor": "rgba(153, 102, 255, 0.5)",
                    "fill": False,
                    "borderDash": [5, 5]
                },
                {
                    "label": "BB Lower",
                    "data": recent_data['BB_Lower'].fillna(0).tolist(),
                    "borderColor": "rgba(153, 102, 255, 0.5)",
                    "fill": False,
                    "borderDash": [5, 5]
                }
            ]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"{ticker} - Advanced Technical Analysis"
                },
                "legend": {
                    "display": True,
                    "position": "top"
                }
            },
            "scales": {
                "y": {
                    "beginAtZero": False,
                    "title": {
                        "display": True,
                        "text": "Price"
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": "Date"
                    }
                }
            },
            "interaction": {
                "intersect": False,
                "mode": "index"
            }
        }
    }
    
    return chart

def load_positions() -> Dict:
    """Load positions with better error handling"""
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, "r") as f:
                positions = json.load(f)
                return positions
        return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load positions - {str(e)}")
        return {}

def save_position(ticker: str, buy_price: float, buy_date: str) -> None:
    """Save position with error handling"""
    try:
        positions = load_positions()
        positions[ticker] = {
            "buy_price": float(buy_price),
            "buy_date": buy_date,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=4)
        print(f"✓ Position saved for {ticker}")
    except Exception as e:
        print(f"✗ Error saving position for {ticker}: {str(e)}")

def get_user_input_enhanced() -> List[str]:
    """Enhanced user input with better validation and suggestions"""
    print("\n" + "="*60)
    print("🚀 ADVANCED STOCK ANALYSIS SYSTEM")
    print("="*60)
    
    print("\n📊 Available Analysis Options:")
    print("1. Enter custom tickers")
    print("2. Indian Large Cap stocks")
    print("3. Indian Mid Cap stocks") 
    print("4. US Tech stocks")
    print("5. US Blue Chip stocks")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    predefined_lists = {
        "2": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
              "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS"],
        "3": ["DIXON.NS", "COFORGE.NS", "MPHASIS.NS", "PERSISTENT.NS", "MINDTREE.NS",
              "LTTS.NS", "LTIM.NS", "TECHM.NS", "WIPRO.NS", "HCLTECH.NS"],
        "4": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM"],
        "5": ["AAPL", "MSFT", "JNJ", "V", "WMT", "PG", "JPM", "UNH", "MA", "HD"]
    }
    
    if choice in predefined_lists:
        tickers = predefined_lists[choice]
        print(f"\n✓ Selected: {', '.join(tickers)}")
        return tickers
    else:
        print("\n📝 Enter stock tickers:")
        print("• For Indian stocks: Add .NS (e.g., RELIANCE.NS)")
        print("• For US stocks: Use symbol only (e.g., AAPL)")
        print("• Separate multiple tickers with commas")
        
        user_input = input("\nTickers: ").strip()
        if not user_input:
            print("No input provided. Using default Indian stocks...")
            return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        
        tickers = [ticker.strip().upper() for ticker in user_input.split(',')]
        return [t for t in tickers if t]

def main():
    """Enhanced main function with better organization"""
    print("🔄 Initializing Advanced Stock Analysis...")
    
    # Get user input
    tickers = get_user_input_enhanced()
    
    if not tickers:
        print("❌ No valid tickers provided. Exiting...")
        return
    
    print(f"\n🎯 Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
    
    # Initialize components
    recommendation_engine = AdvancedRecommendationEngine()
    
    # Parallel data fetching
    print("\n📡 Fetching stock data in parallel...")
    stock_data = parallel_data_fetch(tickers)
    
    if not stock_data:
        print("❌ No stock data could be fetched. Exiting...")
        return
    
    # Get market sentiment
    print("\n📰 Analyzing market sentiment...")
    sentiment, news_message = get_news_sentiment_optimized()
    print(f"Sentiment: {sentiment:.3f} ({news_message})")
    
    # Macro data (placeholder - can be enhanced with real APIs)
    macro_data = {"treasury_yield": 4.42, "cpi": 321.465}
    
    # Load existing positions
    positions = load_positions()
    charts = {}
    
    # Analysis results storage
    analysis_results = []
    
    print(f"\n{'='*80}")
    print("📈 DETAILED STOCK ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze each stock
    for ticker in stock_data.keys():
        print(f"\n🔍 Analyzing {ticker}...")
        print("-" * 50)
        
        df = stock_data[ticker]
        
        # Get enhanced fundamental data
        micro_data, micro_message = get_enhanced_micro_data(ticker)
        
        # Generate recommendation
        recommendation, reasons, score = recommendation_engine.generate_recommendation(
            ticker, df, sentiment, macro_data, micro_data
        )
        
        # Current price and technical levels
        current_price = df['Close'].iloc[-1]
        latest = df.iloc[-1]
        
        # Display current status
        print(f"Current Price: ₹{current_price:.2f}")
        print(f"RSI: {latest['RSI']:.1f}" if not pd.isna(latest['RSI']) else "RSI: N/A")
        print(f"MACD: {latest['MACD']:.2f}" if not pd.isna(latest['MACD']) else "MACD: N/A")
        print(f"Volume Ratio: {latest['Volume_Ratio']:.2f}" if not pd.isna(latest['Volume_Ratio']) else "Volume Ratio: N/A")
        
        # Recommendation
        print(f"\n🎯 RECOMMENDATION: {recommendation} (Score: {score:.2f})")
        print("📋 Reasons:")
        for reason in reasons:
            print(f"   • {reason}")
        
        # Position management
        if ticker in positions:
            pos = positions[ticker]
            buy_price = pos["buy_price"]
            profit_loss = (current_price - buy_price) / buy_price * 100
            
            print(f"\n💼 POSITION STATUS:")
            print(f"   Bought at: ₹{buy_price:.2f}")
            print(f"   Current: ₹{current_price:.2f}")
            print(f"   P&L: {profit_loss:+.1f}%")
            
            # Position alerts
            if profit_loss > 15:
                print(f"   🎉 ALERT: Consider taking profits!")
            elif profit_loss < -10:
                print(f"   ⚠️  ALERT: Consider stop loss!")
        
        # Buy signal handling
        if recommendation in ["Buy", "Strong Buy"] and ticker not in positions:
            save_position(ticker, current_price, datetime.now().strftime('%Y-%m-%d'))
            print(f"🛒 NEW POSITION: Added {ticker} at ₹{current_price:.2f}")
        
        # Generate enhanced chart
        charts[ticker] = generate_enhanced_charts(ticker, df)
        
        # Store results
        analysis_results.append({
            'ticker': ticker,
            'recommendation': recommendation,
            'score': score,
            'current_price': current_price,
            'reasons': reasons
        })
    
    # Save charts
    try:
        with open(CHARTS_FILE, "w") as f:
            json.dump(charts, f, indent=4)
        print(f"\n💾 Charts saved to {CHARTS_FILE}")
    except Exception as e:
        print(f"❌ Error saving charts: {str(e)}")
    
    # Save recommendations
    try:
        with open(RECOMMENDATIONS_FILE, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"ANALYSIS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Market Sentiment: {sentiment:.3f}\n")
            f.write(f"Stocks Analyzed: {len(analysis_results)}\n\n")
            
            for result in analysis_results:
                f.write(f"{result['ticker']}: {result['recommendation']} (Score: {result['score']:.2f})\n")
                f.write(f"Price: ₹{result['current_price']:.2f}\n")
                for reason in result['reasons']:
                    f.write(f"  • {reason}\n")
                f.write("\n")
    except Exception as e:
        print(f"❌ Error saving recommendations: {str(e)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Sort by score
    analysis_results.sort(key=lambda x: x['score'], reverse=True)
    
    buy_recommendations = [r for r in analysis_results if 'Buy' in r['recommendation']]
    sell_recommendations = [r for r in analysis_results if 'Sell' in r['recommendation']]
    
    print(f"🟢 BUY/STRONG BUY: {len(buy_recommendations)} stocks")
    for result in buy_recommendations:
        print(f"   • {result['ticker']}: {result['recommendation']} (Score: {result['score']:.2f})")
    
    print(f"\n🔴 SELL/STRONG SELL: {len(sell_recommendations)} stocks") 
    for result in sell_recommendations:
        print(f"   • {result['ticker']}: {result['recommendation']} (Score: {result['score']:.2f})")
    
    print(f"\n💡 Market Sentiment: {sentiment:.3f} ({'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'})")
    print(f"📁 Charts available in: {CHARTS_FILE}")
    print(f"📄 Full report saved to: {RECOMMENDATIONS_FILE}")
    
    print(f"\n✅ Analysis Complete!")

if __name__ == "__main__":
    main() 