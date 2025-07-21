# Stock Analysis Agent

A Python-based stock analysis tool that provides technical analysis, sentiment analysis, and investment recommendations for Indian stocks.

## Features

- **Technical Analysis**: Calculates Moving Averages (MA20, MA50), RSI indicators, MACD, OBV, ATR
- **News Sentiment Analysis**: Analyzes market sentiment using NewsAPI
- **Macroeconomic Data**: Fetches Treasury yields and CPI data from Alpha Vantage
- **Stock Fundamentals**: Retrieves P/E ratios and earnings data
- **Interactive Charts**: Visualizes stock price trends using Chart.js
- **Investment Recommendations**: Provides Buy/Sell/Hold recommendations based on multiple factors
- **Position Tracking**: Monitors existing positions and profit/loss
- **Backtesting**: Robust event-driven backtesting with flexible filter tuning and multi-ticker support

## Backtesting Workflow (New!)

- The `backtest_ema.py` script now supports running backtests for a list of tickers (e.g., RELIANCE.NS, TCS.NS, INFY.NS).
- The strategy uses a combination of EMA crossovers, MACD crossovers, ATR-based stop-loss, and can be easily tuned for additional filters.
- Results for each ticker are printed with clear labels for easy comparison.
- You can further tune the filters and parameters in the script to experiment with different strategies.

### How to Use the Backtest Script

1. **Edit the ticker list** at the top of `backtest_ema.py`:
   ```python
   TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
   ```
2. **Run the script:**
   ```bash
   python backtest_ema.py
   ```
3. **Interpret the results:**
   - For each ticker, you'll see a summary of performance stats (return, drawdown, win rate, etc.).
   - Compare the results to buy-and-hold to evaluate your strategy's effectiveness.
   - Adjust filters and parameters as needed to improve performance.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-agent.git
   cd stock-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `backtest_ema.py` to modify:
- Stock tickers to analyze (`TICKERS` list)
- Analysis period (default: 5 years)
- Technical indicators and filter logic

## Project Structure

```
stock-agent/
├── stock_agent.py          # Main analysis script
├── backtest_ema.py         # Event-driven backtesting script (multi-ticker)
├── chart.html              # Interactive charts interface
├── charts.json             # Generated chart data
├── stock_recommendations.txt # Analysis history
├── stock_positions.json    # Position tracking
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .env                   # API keys (not in repo)
```

## Dependencies

- yfinance
- pandas
- numpy
- pandas-ta
- backtesting
- requests
- textblob
- python-dotenv

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions. 