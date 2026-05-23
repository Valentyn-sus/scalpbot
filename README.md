# Crypto Market Scalping & Backtesting Script

An algorithmic trading and market analysis script written in Python that fetches live market data from the Binance API, tracks market structures, and simulates high-frequency scalping strategies.

##  Key Features
- **Exchange API Integration:** Utilizes the `ccxt` library to connect to Binance Futures, retrieving heavy historical OHLCV data structures.
- **Market Structure Tracking:** Implemented logic to detect Swing Highs/Lows and identify Break of Structure (BOS) or Change of Character (CHoCH) patterns.
- **Technical Indicators Engine:** Hand-coded indicators calculation using `pandas` and `numpy`, including ATR, EMA (20/50 crossovers), RSI, MACD, and Volume Spikes.
- **Backtesting & Simulation:** Simulates full trading loops complete with trailing Stop Loss adjustments, fixed risk per trade (10%), and comprehensive profit metrics generation.
- **Signal Automation:** Connects seamlessly with the Telegram Bot API to broadcast high-probability alerts including entries, SL, TP, and Risk/Reward parameters.

##  Tech Stack
- **Language:** Python
- **Libraries:** ccxt, pandas, numpy, requests, pyTelegramBotAPI
