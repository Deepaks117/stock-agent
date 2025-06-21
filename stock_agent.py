import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Configuration
POSITIONS_FILE = "stock_positions.json"
CHARTS_FILE = "charts.json"

def get_stock_data(ticker, period="3mo"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = compute_rsi(df['Close'], 14)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss.where(loss != 0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(rs != np.inf, 100)
    return rsi

def get_news_sentiment(query="global economy", days=7):
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

def get_macro_data():
    return {"treasury_yield": 4.42, "cpi": 321.465}, "Used fallback macro data (no free API available)."

def get_micro_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe_ratio = info.get('trailingPE', None)
        dividend_yield = info.get('dividendYield', None)
        return {
            "pe_ratio": pe_ratio,
            "latest_eps": None,
            "dividend_yield": dividend_yield
        }, f"Micro data fetched for {ticker}."
    except Exception as e:
        return {
            "pe_ratio": None,
            "latest_eps": None,
            "dividend_yield": None
        }, f"Error fetching micro data for {ticker}: {str(e)}"

def load_positions():
    try:
        with open(POSITIONS_FILE, "r") as f:
            positions = json.load(f)
            print(f"Loaded positions: {positions}")
            return positions
    except FileNotFoundError:
        print(f"Warning: {POSITIONS_FILE} not found, initializing empty positions.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: {POSITIONS_FILE} is corrupted, initializing empty positions.")
        return {}

def save_position(ticker, buy_price, buy_date):
    positions = load_positions()
    positions[ticker] = {"buy_price": float(buy_price), "buy_date": buy_date}
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=4)
        print(f"Saved position for {ticker}: {buy_price} on {buy_date}")
    except Exception as e:
        print(f"Error saving position for {ticker}: {str(e)}")

def make_recommendation(ticker, stock_data, sentiment, macro_data, micro_data):
    score = 0
    reasons = []
    if stock_data.empty:
        return "Hold", ["No data available"], score

    latest = stock_data.iloc[-1]
    is_bullish_short = latest['Close'] > latest['MA20'] if not pd.isna(latest['MA20']) else False
    is_bullish_long = latest['Close'] > latest['MA50'] if not pd.isna(latest['MA50']) else False
    is_oversold = latest['RSI'] < 30 if not pd.isna(latest['RSI']) else False
    is_overbought = latest['RSI'] > 70 if not pd.isna(latest['RSI']) else False

    if is_bullish_short:
        score += 1
        reasons.append("Short-term bullish trend (Close > MA20)")
    else:
        score -= 1
        reasons.append("Short-term bearish trend (Close < MA20)")
    if is_bullish_long:
        score += 0.5
        reasons.append("Long-term bullish trend (Close > MA50)")
    else:
        score -= 0.5
        reasons.append("Long-term bearish trend (Close < MA50)")
    if is_oversold:
        score += 1
        reasons.append("Oversold (RSI < 30)")
    if is_overbought:
        score -= 1
        reasons.append("Overbought (RSI > 70)")

    if sentiment > 0.1:
        score += 1
        reasons.append("Positive news sentiment")
    elif sentiment < -0.1:
        score -= 1
        reasons.append("Negative news sentiment")

    if macro_data["treasury_yield"] and macro_data["treasury_yield"] > 4.0:
        score -= 1
        reasons.append("High Treasury yield (>4%)")
    if macro_data["cpi"] and macro_data["cpi"] > 300:
        score -= 0.5
        reasons.append("High inflation (CPI > 300)")

    pe_threshold = 30
    if micro_data["pe_ratio"] and micro_data["pe_ratio"] > pe_threshold:
        score -= 1
        reasons.append(f"High P/E ratio (>{pe_threshold})")
    if micro_data["dividend_yield"] and micro_data["dividend_yield"] >= 0.02:
        score += 1
        reasons.append("Attractive dividend yield (>=2%)")

    if score > 1:
        return "Buy", reasons, score
    elif score < -1:
        return "Short", reasons, score
    else:
        return "Hold", reasons, score

def generate_price_chart(ticker, stock_data):
    if stock_data.empty:
        return {"type": "line", "data": {"labels": [], "datasets": []}, "options": {}}
    dates = stock_data.index.strftime("%Y-%m-%d").tolist()[-10:]
    closes = stock_data['Close'].tail(10).tolist()
    ma20 = stock_data['MA20'].tail(10).tolist()
    
    chart = {
        "type": "line",
        "data": {
            "labels": dates,
            "datasets": [
                {
                    "label": f"{ticker} Close",
                    "data": closes,
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": False
                },
                {
                    "label": "MA20",
                    "data": ma20,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "fill": False
                }
            ]
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": f"{ticker} Price Trend"}
            },
            "scales": {
                "y": {"beginAtZero": False}
            }
        }
    }
    return chart

def main():
    # Get user input for tickers
    print("Enter stock tickers (e.g., DIXON.NS, HDFCBANK.NS), separated by commas, or press Enter for default [DIXON.NS]:")
    user_input = input().strip()
    STOCK_TICKERS = [ticker.strip().upper() for ticker in user_input.split(',')] if user_input else ["DIXON.NS"]
    
    sentiment, news_message = get_news_sentiment()
    macro_data, macro_message = get_macro_data()
    positions = load_positions()
    charts = {}

    with open("stock_recommendations.txt", "a") as f:
        f.write(f"\n--- Recommendations on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        for ticker in STOCK_TICKERS:
            stock_data = get_stock_data(ticker)
            if stock_data.empty:
                print(f"\nNo stock data for {ticker}")
                continue

            print(f"\nStock Data for {ticker}")
            print(stock_data[['Close', 'MA20', 'MA50', 'RSI']].tail(5))

            print(f"\nPrice Chart for {ticker} (Last 10 Days)")
            chart = generate_price_chart(ticker, stock_data)
            charts[ticker] = chart
            print(f"Dates: {chart['data']['labels']}")
            print(f"Close: {chart['data']['datasets'][0]['data']}")
            print(f"MA20: {chart['data']['datasets'][1]['data']}")

            print(f"\nMicro Data for {ticker}")
            micro_data, micro_message = get_micro_data(ticker)
            print(f"P/E Ratio: {micro_data['pe_ratio']}")
            print(f"Latest Reported EPS: {micro_data['latest_eps']}")
            print(f"Dividend Yield: {micro_data['dividend_yield']}")
            print(micro_message)

            print(f"\nRecommendation for {ticker}")
            recommendation, reasons, score = make_recommendation(ticker, stock_data, sentiment, macro_data, micro_data)
            print(f"Action: {recommendation} (Score: {score})")
            print("Reasons:")
            for reason in reasons:
                print(f"- {reason}")

            if recommendation == "Buy":
                buy_price = stock_data['Close'].iloc[-1]
                buy_date = datetime.now().strftime('%Y-%m-%d')
                save_position(ticker, buy_price, buy_date)
                print(f"ALERT: Buy {ticker} at {buy_price:.2f} on {buy_date}")

            if ticker in positions:
                buy_price = float(positions[ticker]["buy_price"])
                current_price = stock_data['Close'].iloc[-1]
                profit_loss = (current_price - buy_price) / buy_price * 100
                print(f"Position: Bought at {buy_price:.2f}, Current: {current_price:.2f}, Profit/Loss: {profit_loss:.1f}%")
                if profit_loss > 15 or profit_loss < -10 or score < -1:
                    print(f"ALERT: Sell {ticker} at {current_price:.2f} (Profit/Loss: {profit_loss:.1f}%)")
                elif score >= -1:
                    print(f"ALERT: Hold {ticker} at {current_price:.2f} (Profit/Loss: {profit_loss:.1f}%)")

            f.write(f"{ticker}: {recommendation} (Score: {score})\n")
            f.write("Reasons:\n")
            for reason in reasons:
                f.write(f"- {reason}\n")
            if ticker in positions:
                f.write(f"Position: Bought at {buy_price:.2f}, Current: {current_price:.2f}, Profit/Loss: {profit_loss:.1f}%\n")

        # Save charts to JSON
        try:
            with open(CHARTS_FILE, "w") as f:
                json.dump(charts, f, indent=4)
            print(f"\nSaved chart data to {CHARTS_FILE}")
        except Exception as e:
            print(f"Error saving charts to {CHARTS_FILE}: {str(e)}")

        print("\nNews Sentiment Analysis")
        print(f"Sentiment Score: {sentiment:.3f} ({'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'})")
        print(news_message)

        print("\nMacroeconomic Data")
        print(f"10-Year Treasury Yield: {macro_data['treasury_yield']}%")
        print(f"CPI (Inflation): {macro_data['cpi']}")
        print(macro_message)

if __name__ == "__main__":
    main()