# Stock Analysis Agent

A Python-based stock analysis tool that provides technical analysis, sentiment analysis, and investment recommendations for Indian stocks.

## Features

- **Technical Analysis**: Calculates Moving Averages (MA20, MA50), RSI indicators
- **News Sentiment Analysis**: Analyzes market sentiment using NewsAPI
- **Macroeconomic Data**: Fetches Treasury yields and CPI data from Alpha Vantage
- **Stock Fundamentals**: Retrieves P/E ratios and earnings data
- **Interactive Charts**: Visualizes stock price trends using Chart.js
- **Investment Recommendations**: Provides Buy/Sell/Hold recommendations based on multiple factors
- **Position Tracking**: Monitors existing positions and profit/loss

## Screenshots

- Stock price charts with technical indicators
- Investment recommendations with detailed reasoning
- News sentiment analysis results

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
   pip install yfinance pandas numpy requests textblob python-dotenv
   ```

5. **Set up environment variables**
   Create a `.env` file in the project root with your API keys:
   ```
   NEWS_API_KEY=your_news_api_key_here
   ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
   ```

## Usage

1. **Run the stock analysis**
   ```bash
   python stock_agent.py
   ```

2. **View interactive charts**
   ```bash
   python -m http.server 8000
   ```
   Then open `http://localhost:8000/chart.html` in your browser.

## Configuration

Edit `stock_agent.py` to modify:
- Stock tickers to analyze (`STOCK_TICKERS` list)
- Analysis period (default: 3 months)
- Technical indicators parameters

## API Keys Required

- **NewsAPI**: Get free key from [newsapi.org](https://newsapi.org)
- **Alpha Vantage**: Get free key from [alphavantage.co](https://alphavantage.co)

## Project Structure

```
stock-agent/
├── stock_agent.py          # Main analysis script
├── chart.html              # Interactive charts interface
├── charts.json             # Generated chart data
├── stock_recommendations.txt # Analysis history
├── stock_positions.json    # Position tracking
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .env                   # API keys (not in repo)
```

## Features in Detail

### Technical Analysis
- 20-day and 50-day moving averages
- 14-day RSI (Relative Strength Index)
- Price trend analysis

### Sentiment Analysis
- Analyzes recent news articles
- Calculates sentiment scores
- Considers market sentiment in recommendations

### Investment Logic
- Combines technical, fundamental, and sentiment analysis
- Provides weighted scoring system
- Generates actionable recommendations

### Risk Management
- Tracks existing positions
- Monitors profit/loss percentages
- Provides sell signals based on performance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.

## Support

If you encounter any issues or have questions, please open an issue on GitHub. 