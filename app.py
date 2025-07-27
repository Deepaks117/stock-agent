from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json

app = Flask(__name__)
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# Configuration
POSITIONS_FILE = "stock_positions.json"

def get_stock_data(ticker, period="6mo"):
    """Fetch stock data with technical indicators using pandas-ta"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return df
        
        # Calculate technical indicators using pandas-ta
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=100, append=True)
        df.ta.rsi(length=14, append=True)
        
        return df.dropna()  # Drop rows with NaN values
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_news_sentiment(query="global economy", days=7):
    """Fetch and analyze news sentiment"""
    if not NEWS_API_KEY:
        return 0, "Error: No NewsAPI key provided."
    
    url = f"https://newsapi.org/v2/everything?q={query}&from={(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        sentiment_score = 0
        count = 0
        
        for article in articles[:10]:
            text = article.get('title', '') + ' ' + article.get('description', '')
            analysis = TextBlob(text)
            sentiment_score += analysis.sentiment.polarity
            count += 1
        
        return sentiment_score / count if count > 0 else 0, f"Analyzed {count} articles."
    except Exception as e:
        return 0, f"Error fetching news: {str(e)}"

def get_micro_data(ticker):
    """Fetch fundamental data for the stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe_ratio = info.get('trailingPE', None)
        dividend_yield = info.get('dividendYield', None)
        trailing_eps = info.get('trailingEps', None)
        
        return {
            "pe_ratio": pe_ratio,
            "latest_eps": trailing_eps,
            "dividend_yield": dividend_yield
        }, f"Micro data fetched for {ticker}."
    except Exception as e:
        return {
            "pe_ratio": None,
            "latest_eps": None,
            "dividend_yield": None
        }, f"Error fetching micro data for {ticker}: {str(e)}"

def load_positions():
    """Load existing positions from JSON file"""
    try:
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_position(ticker, buy_price, buy_date):
    """Save a new position to JSON file"""
    positions = load_positions()
    positions[ticker] = {"buy_price": float(buy_price), "buy_date": buy_date}
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving position for {ticker}: {str(e)}")
        return False

def make_recommendation(ticker, stock_data, sentiment, micro_data, buy_price=None):
    """Event-driven recommendation system with triple EMA crossover logic"""
    score = 0
    reasons = []
    
    if stock_data.empty or len(stock_data) < 2:
        return "Hold", ["Not enough data available"], score, None

    # Get the latest two days of data for crossover detection
    latest = stock_data.iloc[-1]
    previous = stock_data.iloc[-2]
    current_price = latest['Close']

    # --- EMA Crossover Logic (Primary Signal) ---
    # Check for a "Golden Cross" event between EMA20 and EMA50
    ema20_crossed_above_ema50 = latest['EMA_20'] > latest['EMA_50'] and previous['EMA_20'] < previous['EMA_50']
    
    # For a strong signal, EMA20 should also be above EMA100
    if ema20_crossed_above_ema50 and latest['EMA_20'] > latest['EMA_100']:
        score += 2  # Strong bullish signal
        reasons.append("⭐ Strong Bullish Signal: EMA20 crossed above EMA50 and is above EMA100")
    
    # Check for a "Death Cross" event
    ema20_crossed_below_ema50 = latest['EMA_20'] < latest['EMA_50'] and previous['EMA_20'] > previous['EMA_50']
    
    if ema20_crossed_below_ema50 and latest['EMA_20'] < latest['EMA_100']:
        score -= 2  # Strong bearish signal
        reasons.append("⭐ Strong Bearish Signal: EMA20 crossed below EMA50 and is below EMA100")

    # --- RSI Logic (Secondary Confirmation) ---
    if latest['RSI_14'] < 30:
        score += 1
        reasons.append(f"Oversold (RSI: {latest['RSI_14']:.1f})")
    elif latest['RSI_14'] > 70:
        score -= 1
        reasons.append(f"Overbought (RSI: {latest['RSI_14']:.1f})")

    # --- Sentiment Analysis (Secondary Factor) ---
    if sentiment > 0.1:
        score += 1
        reasons.append(f"Positive news sentiment ({sentiment:.2f})")
    elif sentiment < -0.1:
        score -= 1
        reasons.append(f"Negative news sentiment ({sentiment:.2f})")

    # --- Fundamental Analysis (Secondary Factor) ---
    if micro_data["pe_ratio"] and micro_data["pe_ratio"] < 25:
        score += 0.5
        reasons.append(f"Low P/E ratio ({micro_data['pe_ratio']:.1f})")
    elif micro_data["pe_ratio"] and micro_data["pe_ratio"] > 50:
        score -= 0.5
        reasons.append(f"High P/E ratio ({micro_data['pe_ratio']:.1f})")
    
    if micro_data["dividend_yield"] and micro_data["dividend_yield"] >= 0.02:
        score += 1
        reasons.append(f"Attractive dividend yield ({micro_data['dividend_yield']:.2%})")

    # --- Position Management (if user owns the stock) ---
    if buy_price is not None:
        profit_loss = (current_price - buy_price) / buy_price * 100
        reasons.append(f"Current P/L: {profit_loss:.2f}%")
        
        if profit_loss > 15:
            score += 1
            reasons.append("Profit > 15%: Consider taking profits")
        elif profit_loss < -10:
            score -= 1
            reasons.append("Loss > 10%: Consider stop-loss")

    # --- Final Decision Based on Total Score ---
    if score > 1.5:
        recommendation = "Buy"
    elif score < -1.5:
        recommendation = "Sell/Short"
    else:
        recommendation = "Hold"

    return recommendation, reasons, score, current_price

def generate_price_chart(ticker, stock_data):
    """Generate chart data with EMAs for visualization"""
    if stock_data.empty:
        return None
    
    # Use last 30 days for better visualization
    chart_data = stock_data.tail(30).copy()
    dates = chart_data.index.strftime("%Y-%m-%d").tolist()
    
    chart = {
        "type": "line",
        "data": {
            "labels": dates,
            "datasets": [
                {
                    "label": f"{ticker} Close",
                    "data": chart_data['Close'].tolist(),
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": False,
                    "borderWidth": 2
                },
                {
                    "label": "EMA20",
                    "data": chart_data['EMA_20'].tolist(),
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "fill": False,
                    "borderWidth": 1
                },
                {
                    "label": "EMA50",
                    "data": chart_data['EMA_50'].tolist(),
                    "borderColor": "rgba(255, 205, 86, 1)",
                    "fill": False,
                    "borderWidth": 1
                },
                {
                    "label": "EMA100",
                    "data": chart_data['EMA_100'].tolist(),
                    "borderColor": "rgba(75, 192, 192, 0.5)",
                    "fill": False,
                    "borderWidth": 1
                }
            ]
        },
        "options": {
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"{ticker} Price & EMA Trends"
                }
            },
            "scales": {
                "y": {
                    "beginAtZero": False
                }
            },
            "responsive": True,
            "maintainAspectRatio": False
        }
    }
    return chart

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint"""
    try:
        data = request.json
        ticker = data.get("ticker", "").strip().upper()
        buy_price = data.get("buy_price")
        
        # Convert buy_price to float if provided
        try:
            buy_price = float(buy_price) if buy_price else None
        except (ValueError, TypeError):
            buy_price = None
        
        if not ticker:
            return jsonify({"error": "Please provide a valid stock ticker"}), 400
        
        # Fetch data
        stock_data = get_stock_data(ticker)
        if stock_data.empty:
            return jsonify({"error": f"No data available for {ticker}"}), 400
        
        # Get sentiment and fundamental data
        sentiment, sentiment_message = get_news_sentiment()
        micro_data, micro_message = get_micro_data(ticker)
        
        # Generate recommendation
        recommendation, reasons, score, current_price = make_recommendation(
            ticker, stock_data, sentiment, micro_data, buy_price
        )
        
        # Generate chart
        chart = generate_price_chart(ticker, stock_data)
        
        # Position management (if it's a buy recommendation)
        position_info = None
        if recommendation == "Buy" and buy_price is None:
            # Suggest buying at current price
            position_info = {
                "suggested_buy_price": current_price,
                "message": f"Consider buying {ticker} at ₹{current_price:.2f}"
            }
        
        return jsonify({
            "ticker": ticker,
            "recommendation": recommendation,
            "confidence_score": score,
            "reasons": reasons,
            "current_price": current_price,
            "sentiment_score": sentiment,
            "sentiment_message": sentiment_message,
            "micro_data": micro_data,
            "chart": chart,
            "position_info": position_info
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True) 