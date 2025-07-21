import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

for TICKER in TICKERS:
    print(f"\n{'='*30}\nRunning backtest for {TICKER}\n{'='*30}")
    df = yf.download(TICKER, period="5y")
    print("Downloaded columns:", df.columns)
    # If columns are MultiIndex, select by first level (price type)
    if isinstance(df.columns, pd.MultiIndex):
        df = df[[col for col in df.columns if col[0] in ['Open', 'High', 'Low', 'Close', 'Volume']]]
        df.columns = [col[0] for col in df.columns]
    else:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    # Calculate OBV and OBV_SMA10 in the DataFrame
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_SMA10'] = ta.sma(df['OBV'], 10)
    df = df.dropna()

    class EmaCrossStrategy(Strategy):
        def init(self):
            close = pd.Series(self.data.Close)
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)
            volume = pd.Series(self.data.Volume)
            self.ema20 = self.I(ta.ema, close, 20)
            self.ema50 = self.I(ta.ema, close, 50)
            self.ema100 = self.I(ta.ema, close, 100)
            self.macd_line = self.I(lambda x: ta.macd(x, fast=12, slow=26, signal=9)['MACD_12_26_9'], close)
            self.macd_signal = self.I(lambda x: ta.macd(x, fast=12, slow=26, signal=9)['MACDs_12_26_9'], close)
            self.atr = self.I(ta.atr, high, low, close, 14)

        def next(self):
            price = self.data.Close[-1]
            # Entry conditions
            ema_crossed = self.ema20[-1] > self.ema50[-1] and self.ema20[-2] < self.ema50[-2]
            strong_trend = self.ema20[-1] > self.ema100[-1]
            # Loosened MACD filter: allow crossover in last 3 bars
            macd_crossed_up = any(
                self.macd_line[-i] > self.macd_signal[-i] and self.macd_line[-i-1] < self.macd_signal[-i-1]
                for i in range(1, 4)
            )
            if ema_crossed and strong_trend and macd_crossed_up and not self.position:
                stop_loss = price - (self.atr[-1] * 2)
                self.buy(sl=stop_loss)
            elif self.ema20[-1] < self.ema50[-1]:
                self.position.close()

    bt = Backtest(df, EmaCrossStrategy, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)
    # Optionally, comment out the next line if you don't want to see the plot for every ticker
    # bt.plot() 