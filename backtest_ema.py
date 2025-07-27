import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
warnings.filterwarnings('ignore')

class DynamicEmaStrategy(Strategy):
    """
    Dynamic EMA crossover strategy with adaptive parameters
    """
    # Strategy parameters - can be optimized
    ema_short = 12
    ema_medium = 26
    ema_long = 50
    rsi_oversold = 30
    rsi_overbought = 70
    atr_multiplier = 2.0
    volume_threshold = 1.2  # Volume must be 20% above average
    
    def init(self):
        """Initialize indicators"""
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # EMA indicators
        self.ema_short_line = self.I(ta.ema, close, self.ema_short)
        self.ema_medium_line = self.I(ta.ema, close, self.ema_medium)
        self.ema_long_line = self.I(ta.ema, close, self.ema_long)
        
        # MACD
        self.macd_line = self.I(lambda x: ta.macd(x, fast=12, slow=26, signal=9)['MACD_12_26_9'], close)
        self.macd_signal = self.I(lambda x: ta.macd(x, fast=12, slow=26, signal=9)['MACDs_12_26_9'], close)
        self.macd_histogram = self.I(lambda x: ta.macd(x, fast=12, slow=26, signal=9)['MACDh_12_26_9'], close)
        
        # RSI
        self.rsi = self.I(ta.rsi, close, 14)
        
        # ATR for stop loss
        self.atr = self.I(ta.atr, high, low, close, 14)
        
        # Volume indicators
        self.volume_sma = self.I(ta.sma, volume, 20)
        
        # Bollinger Bands
        self.bb_upper = self.I(lambda x: ta.bbands(x, length=20, std=2)['BBU_20_2.0'], close)
        self.bb_lower = self.I(lambda x: ta.bbands(x, length=20, std=2)['BBL_20_2.0'], close)
        self.bb_middle = self.I(lambda x: ta.bbands(x, length=20, std=2)['BBM_20_2.0'], close)
        
    def next(self):
        """Execute trading logic"""
        if len(self.data) < max(self.ema_long, 50):  # Need enough data
            return
            
        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Entry conditions
        entry_conditions = self._check_entry_conditions(current_price, current_volume)
        exit_conditions = self._check_exit_conditions(current_price)
        
        # Position management
        if not self.position and entry_conditions:
            self._enter_position(current_price)
        elif self.position and exit_conditions:
            self.position.close()
    
    def _check_entry_conditions(self, price, volume):
        """Check if all entry conditions are met"""
        # Trend conditions
        ema_bullish = (self.ema_short_line[-1] > self.ema_medium_line[-1] > self.ema_long_line[-1])
        ema_crossover = (self.ema_short_line[-1] > self.ema_medium_line[-1] and 
                        self.ema_short_line[-2] <= self.ema_medium_line[-2])
        
        # MACD conditions
        macd_bullish = (self.macd_line[-1] > self.macd_signal[-1] and 
                       self.macd_histogram[-1] > self.macd_histogram[-2])
        
        # RSI conditions (not overbought)
        rsi_ok = self.rsi[-1] < self.rsi_overbought and self.rsi[-1] > self.rsi_oversold
        
        # Volume confirmation
        volume_ok = volume > (self.volume_sma[-1] * self.volume_threshold)
        
        # Bollinger Bands - price near middle or lower band (not at upper)
        bb_ok = price < self.bb_upper[-1]
        
        # Price momentum
        price_momentum = price > self.data.Close[-5]  # Price higher than 5 days ago
        
        return (ema_bullish and ema_crossover and macd_bullish and 
                rsi_ok and volume_ok and bb_ok and price_momentum)
    
    def _check_exit_conditions(self, price):
        """Check if exit conditions are met"""
        # Bearish EMA cross
        ema_bearish = (self.ema_short_line[-1] < self.ema_medium_line[-1])
        
        # MACD bearish
        macd_bearish = (self.macd_line[-1] < self.macd_signal[-1] and 
                       self.macd_histogram[-1] < 0)
        
        # RSI overbought
        rsi_exit = self.rsi[-1] > self.rsi_overbought
        
        # Price near upper Bollinger Band
        bb_exit = price > self.bb_upper[-1]
        
        return ema_bearish or macd_bearish or rsi_exit or bb_exit
    
    def _enter_position(self, price):
        """Enter position with dynamic stop loss"""
        # Dynamic stop loss based on ATR and recent volatility
        atr_stop = self.atr[-1] * self.atr_multiplier
        recent_low = min(self.data.Low[-10:])  # 10-day low
        
        # Use the more conservative stop loss
        stop_loss = max(price - atr_stop, recent_low * 0.95)
        
        # Take profit at 2:1 risk-reward ratio
        take_profit = price + (price - stop_loss) * 2
        
        self.buy(sl=stop_loss, tp=take_profit)

def get_stock_data(ticker, period="2y"):
    """
    Fetch and prepare stock data with error handling
    """
    try:
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            print(f"No data available for {ticker}")
            return None
            
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns for {ticker}")
            return None
            
        # Remove rows with missing data
        df = df[required_cols].dropna()
        
        if len(df) < 100:  # Need minimum data for meaningful backtest
            print(f"Insufficient data for {ticker} (only {len(df)} rows)")
            return None
            
        print(f"Successfully loaded {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def optimize_strategy_parameters(df, ticker):
    """
    Optimize strategy parameters for specific stock
    """
    print(f"Optimizing parameters for {ticker}...")
    
    # Quick optimization on key parameters
    try:
        bt = Backtest(df, DynamicEmaStrategy, cash=100000, commission=0.002)
        
        # Define parameter ranges for optimization
        optimization_params = {
            'ema_short': range(8, 16, 2),
            'ema_medium': range(20, 30, 2),
            'atr_multiplier': [1.5, 2.0, 2.5],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75]
        }
        
        # Run optimization (limited to prevent overfitting)
        stats = bt.optimize(**optimization_params, maximize='Return [%]', max_tries=50)
        
        return stats, stats._strategy if hasattr(stats, '_strategy') else None
        
    except Exception as e:
        print(f"Optimization failed for {ticker}: {str(e)}")
        print("Using default parameters...")
        bt = Backtest(df, DynamicEmaStrategy, cash=100000, commission=0.002)
        stats = bt.run()
        return stats, None

def run_backtest(ticker):
    """
    Run backtest for a single ticker with optimization
    """
    print(f"\n{'='*50}")
    print(f"BACKTESTING: {ticker}")
    print(f"{'='*50}")
    
    # Get data
    df = get_stock_data(ticker)
    if df is None:
        return None
        
    # Run optimization and backtest
    try:
        stats, optimized_params = optimize_strategy_parameters(df, ticker)
        
        # Display results
        print(f"\nResults for {ticker}:")
        print("-" * 30)
        
        # Key metrics
        key_metrics = [
            'Return [%]', 'Buy & Hold Return [%]', 'Max Drawdown [%]',
            'Win Rate [%]', 'Profit Factor', 'Sharpe Ratio',
            '# Trades', 'Avg Trade [%]'
        ]
        
        for metric in key_metrics:
            if metric in stats:
                value = stats[metric]
                if isinstance(value, float):
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value}")
        
        # Performance summary
        returns = stats['Return [%]']
        buy_hold = stats['Buy & Hold Return [%]']
        excess_return = returns - buy_hold
        
        print(f"\nPerformance Summary:")
        print(f"Strategy Return: {returns:.2f}%")
        print(f"Buy & Hold Return: {buy_hold:.2f}%")
        print(f"Excess Return: {excess_return:.2f}%")
        
        if optimized_params:
            print(f"\nOptimized Parameters:")
            # Get the optimized parameters from the strategy instance
            strategy_instance = optimized_params
            for param in ['ema_short', 'ema_medium', 'ema_long', 'rsi_oversold', 'rsi_overbought', 'atr_multiplier', 'volume_threshold']:
                if hasattr(strategy_instance, param):
                    value = getattr(strategy_instance, param)
                    print(f"  {param}: {value}")
        
        # Generate HTML chart
        generate_html_chart(df, ticker, stats)
        
        return {
            'ticker': ticker,
            'stats': stats,
            'success': True
        }
        
    except Exception as e:
        print(f"Backtest failed for {ticker}: {str(e)}")
        return {
            'ticker': ticker,
            'error': str(e),
            'success': False
        }

def get_user_input():
    """
    Get tickers from user input with validation
    """
    print("Dynamic Stock Backtesting System")
    print("=" * 40)
    print("Enter stock tickers to analyze (NSE stocks should end with .NS)")
    print("Examples: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS")
    print("For US stocks: AAPL, MSFT, GOOGL")
    print("\nOptions:")
    print("1. Enter tickers manually")
    print("2. Use predefined Indian stock list")
    print("3. Use predefined US stock list")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "2":
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
                "BHARTIARTL.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS"]
    elif choice == "3":
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    else:
        tickers_input = input("Enter tickers separated by commas: ").strip()
        if not tickers_input:
            print("No tickers entered. Using default...")
            return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        return [ticker for ticker in tickers if ticker]  # Remove empty strings

def main():
    """
    Main execution function
    """
    # Get tickers from user
    tickers = get_user_input()
    
    if not tickers:
        print("No valid tickers provided. Exiting...")
        return
    
    print(f"\nStarting backtests for {len(tickers)} tickers...")
    print(f"Tickers: {', '.join(tickers)}")
    
    results = []
    successful_tests = 0
    
    # Run backtests
    for ticker in tickers:
        result = run_backtest(ticker)
        if result:
            results.append(result)
            if result['success']:
                successful_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tickers analyzed: {len(tickers)}")
    print(f"Successful backtests: {successful_tests}")
    print(f"Failed backtests: {len(tickers) - successful_tests}")
    
    # Sort successful results by return
    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_results.sort(key=lambda x: x['stats']['Return [%]'], reverse=True)
        
        print(f"\nTop Performers:")
        print("-" * 40)
        for i, result in enumerate(successful_results[:5], 1):
            ticker = result['ticker']
            returns = result['stats']['Return [%]']
            buy_hold = result['stats']['Buy & Hold Return [%]']
            trades = result['stats']['# Trades']
            win_rate = result['stats']['Win Rate [%]']
            
            print(f"{i}. {ticker}")
            print(f"   Strategy Return: {returns:.2f}%")
            print(f"   Buy & Hold: {buy_hold:.2f}%")
            print(f"   Trades: {trades}, Win Rate: {win_rate:.1f}%")
            print()
    
    # Failed tickers
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("Failed Analysis:")
        print("-" * 40)
        for result in failed_results:
            print(f"• {result['ticker']}: {result.get('error', 'Unknown error')}")

def generate_html_chart(df, ticker, stats):
    """
    Generate HTML chart with backtest results
    """
    try:
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Backtest Results for {ticker}', fontsize=16, fontweight='bold')
        
        # Plot 1: Price and EMAs
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1, alpha=0.8)
        ax1.plot(df.index, df['Close'].rolling(window=20).mean(), label='SMA 20', linewidth=1, alpha=0.8)
        ax1.plot(df.index, df['Close'].rolling(window=50).mean(), label='SMA 50', linewidth=1, alpha=0.8)
        ax1.plot(df.index, df['Close'].rolling(window=100).mean(), label='SMA 100', linewidth=1, alpha=0.8)
        ax1.set_title('Price and Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        rsi = ta.rsi(df['Close'], length=14)
        ax2.plot(df.index, rsi, label='RSI', color='purple', linewidth=1)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax2.set_title('RSI (14)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volume
        ax3.bar(df.index, df['Volume'], alpha=0.6, color='blue', label='Volume')
        ax3.set_title('Volume')
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance summary text
        summary_text = f"""
        Strategy Return: {stats['Return [%]']:.2f}%
        Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%
        Max Drawdown: {stats['Max Drawdown [%]']:.2f}%
        Win Rate: {stats['Win Rate [%]']:.1f}%
        Number of Trades: {stats['# Trades']}
        """
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save as HTML
        html_filename = f"{ticker.replace('.', '_')}_backtest.html"
        plt.savefig(f"{ticker.replace('.', '_')}_chart.png", dpi=300, bbox_inches='tight')
        
        # Create HTML file
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Results - {ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .chart {{ text-align: center; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric h3 {{ margin: 0; color: #2c3e50; }}
                .metric p {{ margin: 5px 0; font-size: 18px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Backtest Results for {ticker}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Performance Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Strategy Return</h3>
                        <p style="color: {'green' if stats['Return [%]'] > 0 else 'red'}">{stats['Return [%]']:.2f}%</p>
                    </div>
                    <div class="metric">
                        <h3>Buy & Hold Return</h3>
                        <p style="color: {'green' if stats['Buy & Hold Return [%]'] > 0 else 'red'}">{stats['Buy & Hold Return [%]']:.2f}%</p>
                    </div>
                    <div class="metric">
                        <h3>Max Drawdown</h3>
                        <p style="color: red">{stats['Max Drawdown [%]']:.2f}%</p>
                    </div>
                    <div class="metric">
                        <h3>Win Rate</h3>
                        <p style="color: {'green' if stats['Win Rate [%]'] > 50 else 'orange'}">{stats['Win Rate [%]']:.1f}%</p>
                    </div>
                    <div class="metric">
                        <h3>Number of Trades</h3>
                        <p>{stats['# Trades']}</p>
                    </div>
                    <div class="metric">
                        <h3>Sharpe Ratio</h3>
                        <p style="color: {'green' if stats['Sharpe Ratio'] > 1 else 'orange'}">{stats['Sharpe Ratio']:.2f}</p>
                    </div>
                </div>
            </div>
            
            <div class="chart">
                <h2>Technical Analysis Chart</h2>
                <img src="{ticker.replace('.', '_')}_chart.png" alt="Backtest Chart" style="max-width: 100%; height: auto;">
            </div>
        </body>
        </html>
        """
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML chart generated: {html_filename}")
        print(f"📊 Chart image saved: {ticker.replace('.', '_')}_chart.png")
        
    except Exception as e:
        print(f"❌ Error generating HTML chart: {str(e)}")

if __name__ == "__main__":
    main()