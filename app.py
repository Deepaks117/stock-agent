"""
🚀 ENHANCED STOCK ANALYSIS WEB APPLICATION
==========================================

DEVELOPMENT STATUS (July 27, 2025):
✅ COMPLETED: Flask web application with modern UI
✅ COMPLETED: Smart caching system (80-90% performance improvement)
✅ COMPLETED: Technical analysis engine with 10 essential indicators
✅ COMPLETED: Position management system
✅ COMPLETED: Error handling and user feedback

CURRENT PERFORMANCE:
- First analysis: 1.0-2.2 seconds (was 10-18 seconds)
- Cached analysis: 0.3-0.8 seconds
- Stock data: 0.3-0.9s (cached: 0.1s)
- Sentiment: 0.5-1.0s (cached: 0.1s)
- Micro data: 0.2-0.8s (cached: 0.1s)

KNOWN ISSUES:
1. Loading time still ~2 seconds for first analysis
2. Need async processing for parallel API calls
3. Consider Redis for better caching

NEXT STEPS:
1. Implement async/parallel processing (Priority #1)
2. Add Redis caching (Priority #2)
3. User authentication (Priority #3)

HOW TO CONTINUE:
1. Run: python app.py
2. Open: http://localhost:5000
3. Analyze stocks and monitor performance
4. Check console logs for timing breakdown

PROBLEMS SOLVED TODAY:
- KeyError 'ROC' - Fixed by updating recommendation engine
- ImportError TechnicalAnalyzer - Fixed by cleaning up file structure
- JavaScript null errors - Fixed with null checks and try-catch
- Performance bottlenecks - Fixed with caching and optimization
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json
import warnings
import time
from functools import lru_cache
warnings.filterwarnings('ignore')

# Import enhanced functions from stock_agent
from stock_agent import (
    TechnicalAnalyzer, 
    AdvancedRecommendationEngine,
    get_stock_data_optimized,
    get_news_sentiment_optimized,
    get_enhanced_micro_data,
    generate_enhanced_charts,
    save_position,
    load_positions
)

app = Flask(__name__)
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize the enhanced recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

# Cache for stock data (2 minutes)
stock_cache = {}
CACHE_DURATION = 120  # 2 minutes

# Cache for sentiment (15 minutes)
sentiment_cache = {}
SENTIMENT_CACHE_DURATION = 900  # 15 minutes

def get_cached_stock_data(ticker: str, period: str = "6mo"):
    """Get stock data with caching"""
    current_time = time.time()
    
    # Check if we have cached data that's still valid
    if ticker in stock_cache:
        cached_data, timestamp = stock_cache[ticker]
        if current_time - timestamp < CACHE_DURATION:
            return cached_data
    
    # Fetch new data
    data = get_stock_data_optimized(ticker, period)
    stock_cache[ticker] = (data, current_time)
    return data

def get_cached_sentiment(query: str = "stock market", days: int = 7):
    """Get sentiment with caching"""
    current_time = time.time()
    cache_key = f"{query}_{days}"
    
    # Check if we have cached sentiment that's still valid
    if cache_key in sentiment_cache:
        cached_data, timestamp = sentiment_cache[cache_key]
        if current_time - timestamp < SENTIMENT_CACHE_DURATION:
            return cached_data
    
    # Fetch new sentiment
    sentiment, message = get_news_sentiment_optimized(query, days)
    sentiment_cache[cache_key] = ((sentiment, message), current_time)
    return sentiment, message

# --- Enhanced Data Fetching & Analysis Functions ---
def get_stock_data(ticker, period="3mo"):
    """Enhanced stock data fetching using the optimized function with caching"""
    return get_cached_stock_data(ticker, period)

def get_news_sentiment(query="global economy", days=7):
    """Enhanced news sentiment analysis with caching"""
    return get_cached_sentiment(query, days)

def get_micro_data(ticker):
    """Enhanced fundamental data fetching with basic caching"""
    # Simple cache for micro data (10 minutes)
    cache_key = f"micro_{ticker}"
    current_time = time.time()
    
    if hasattr(get_micro_data, 'cache') and cache_key in get_micro_data.cache:
        cached_data, timestamp = get_micro_data.cache[cache_key]
        if current_time - timestamp < 300:  # 5 minutes
            return cached_data
    
    # Fetch new data
    micro_data, message = get_enhanced_micro_data(ticker)
    
    # Initialize cache if it doesn't exist
    if not hasattr(get_micro_data, 'cache'):
        get_micro_data.cache = {}
    
    get_micro_data.cache[cache_key] = ((micro_data, message), current_time)
    return micro_data, message

def make_recommendation(stock_data, sentiment, micro_data):
    """Enhanced recommendation using the advanced engine"""
    if stock_data.empty:
        return "Hold", ["No data available"], 0
    
    # Macro data (placeholder - can be enhanced with real APIs)
    macro_data = {"treasury_yield": 4.42, "cpi": 321.465}
    
    # Generate comprehensive recommendation
    recommendation, reasons, score = recommendation_engine.generate_recommendation(
        ticker="", stock_data=stock_data, sentiment=sentiment, 
        macro_data=macro_data, micro_data=micro_data
    )
    
    return recommendation, reasons, score

def get_chart_data(stock_data, ticker):
    """Enhanced chart generation"""
    if stock_data.empty:
        return None
    
    # Use the enhanced chart generation function
    chart = generate_enhanced_charts(ticker, stock_data)
    return chart

def get_technical_summary(stock_data):
    """Get technical analysis summary"""
    if stock_data.empty or len(stock_data) < 50:
        return {}
    
    latest = stock_data.iloc[-1]
    trend_strength = TechnicalAnalyzer.get_trend_strength(stock_data)
    
    summary = {
        "current_price": latest['Close'],
        "trend_strength": trend_strength,
        "rsi": latest.get('RSI', None),
        "macd": latest.get('MACD', None),
        "macd_signal": latest.get('MACD_Signal', None),
        "volume_ratio": latest.get('Volume_Ratio', None),
        "bb_position": latest.get('BB_Position', None),
        "atr": latest.get('ATR', None),
        "support": latest.get('Support', None),
        "resistance": latest.get('Resistance', None),
        "ema12": latest.get('EMA12', None),
        "ema26": latest.get('EMA26', None),
        "ma20": latest.get('MA20', None),
        "ma50": latest.get('MA50', None)
    }
    
    return summary

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = time.time()
    
    data = request.json
    ticker = data.get("ticker")
    buy_price = data.get("buy_price")
    
    try:
        buy_price = float(buy_price) if buy_price else None
    except:
        buy_price = None
    
    print(f"Starting analysis for {ticker}...")
    
    # Enhanced data fetching with timing
    print("Fetching stock data...")
    stock_start = time.time()
    stock_data = get_stock_data(ticker)
    print(f"Stock data fetched in {time.time() - stock_start:.2f}s")
    
    print("Fetching sentiment...")
    sentiment_start = time.time()
    sentiment, sentiment_message = get_news_sentiment(ticker)
    print(f"Sentiment fetched in {time.time() - sentiment_start:.2f}s")
    
    print("Fetching micro data...")
    micro_start = time.time()
    micro_data, micro_message = get_micro_data(ticker)
    print(f"Micro data fetched in {time.time() - micro_start:.2f}s")
    
    # Enhanced recommendation
    print("Generating recommendation...")
    rec_start = time.time()
    recommendation, reasons, score = make_recommendation(stock_data, sentiment, micro_data)
    print(f"Recommendation generated in {time.time() - rec_start:.2f}s")
    
    # Enhanced chart
    print("Generating chart...")
    chart_start = time.time()
    chart = get_chart_data(stock_data, ticker)
    print(f"Chart generated in {time.time() - chart_start:.2f}s")
    
    # Technical summary
    print("Generating technical summary...")
    tech_start = time.time()
    technical_summary = get_technical_summary(stock_data)
    print(f"Technical summary generated in {time.time() - tech_start:.2f}s")
    
    # Position management
    positions = load_positions()
    position_info = None
    if ticker in positions:
        pos = positions[ticker]
        current_price = technical_summary.get("current_price", 0)
        if current_price > 0:
            buy_price_pos = pos["buy_price"]
            profit_loss = (current_price - buy_price_pos) / buy_price_pos * 100
            position_info = {
                "buy_price": buy_price_pos,
                "buy_date": pos["buy_date"],
                "current_price": current_price,
                "profit_loss": profit_loss
            }
    
    # Save new position if buy signal and not already tracked
    if recommendation in ["Buy", "Strong Buy"] and ticker not in positions and buy_price:
        save_position(ticker, buy_price, datetime.now().strftime('%Y-%m-%d'))
    
    total_time = time.time() - start_time
    print(f"Total analysis time: {total_time:.2f}s")
    
    return jsonify({
        "recommendation": recommendation,
        "reasons": reasons,
        "score": score,
        "chart": chart,
        "technical_summary": technical_summary,
        "position_info": position_info,
        "sentiment": sentiment,
        "sentiment_message": sentiment_message,
        "micro_data": micro_data,
        "micro_message": micro_message,
        "analysis_time": round(total_time, 2)
    })

@app.route("/positions")
def get_positions():
    """Get all tracked positions"""
    positions = load_positions()
    position_details = []
    
    for ticker, pos in positions.items():
        try:
            stock_data = get_stock_data(ticker)
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                profit_loss = (current_price - pos["buy_price"]) / pos["buy_price"] * 100
                
                position_details.append({
                    "ticker": ticker,
                    "buy_price": pos["buy_price"],
                    "buy_date": pos["buy_date"],
                    "current_price": current_price,
                    "profit_loss": profit_loss,
                    "status": "profit" if profit_loss > 0 else "loss"
                })
        except Exception as e:
            print(f"Error getting position details for {ticker}: {str(e)}")
    
    return jsonify(position_details)

@app.route("/cache/clear")
def clear_cache():
    """Clear all caches (for debugging)"""
    global stock_cache, sentiment_cache
    stock_cache.clear()
    sentiment_cache.clear()
    if hasattr(get_micro_data, 'cache'):
        get_micro_data.cache.clear()
    return jsonify({"message": "All caches cleared"})

if __name__ == "__main__":
    app.run(debug=True) 