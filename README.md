# Trade Bot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automated cryptocurrency trading bot with ML-based predictions.

## 📖 Description

This project implements an automated trading system that uses machine learning models to predict price movements and execute trades on cryptocurrency exchanges. The bot supports multiple exchanges through the CCXT library and includes risk management features.

### Key Features

- 🤖 **ML-Powered Predictions**: LSTM/GRU models for price prediction
- 📊 **Technical Analysis**: Built-in indicators (RSI, MACD, Bollinger Bands, etc.)
- 💼 **Risk Management**: Stop-loss, take-profit, position sizing
- 🔄 **Multi-Exchange Support**: Binance, Kraken, and more via CCXT
- 📝 **Comprehensive Logging**: Track all trades and system events
- 🧪 **Backtesting**: Test strategies on historical data
- 🛡️ **Paper Trading**: Safe testing mode without real money

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/trade_bot.git
cd trade_bot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. **Update configuration**
```bash
# Edit configs/config.yaml with your trading preferences
```

### Usage

```bash
# Run the trading bot
python src/main.py

# Run in dry-run mode (paper trading)
python src/main.py --dry-run

# Run backtest
python src/main.py --backtest

# Train model
python src/main.py --train
```

## 📁 Project Structure

```
trade_bot/
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # Entry point
│   ├── trading/           # Trading logic
│   │   ├── __init__.py
│   │   ├── executor.py    # Order execution
│   │   └── strategy.py    # Trading strategies
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   ├── fetcher.py     # Data fetching
│   │   └── processor.py   # Data processing
│   ├── models/            # ML models
│   │   ├── __init__.py
│   │   ├── predictor.py   # Prediction engine
│   │   └── trainer.py     # Model training
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── logger.py      # Logging setup
│       └── config.py      # Configuration loader
├── data/                   # Data storage
│   ├── raw/               # Raw market data
│   └── processed/         # Processed data
├── models/                 # Saved ML models
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit and integration tests
├── configs/                # Configuration files
│   └── config.yaml        # Main configuration
├── logs/                   # Log files
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Trading**: Exchange, symbols, timeframe
- **Risk Management**: Position size, stop-loss, take-profit
- **Model**: Type, hyperparameters, training settings
- **Data**: Features, cache settings
- **Logging**: Level, output format

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_trading.py
```

## 📊 Example Strategy

The default strategy uses:
1. Fetch historical OHLCV data
2. Calculate technical indicators
3. Generate ML predictions
4. Execute trades based on signals and risk parameters

## 📈 Performance Monitoring

Track your bot's performance:
- Check `logs/trade_bot.log` for detailed logs
- Review trade history in the database
- Monitor P&L and win rate

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **isort** for import sorting

```bash
black src/
flake8 src/
mypy src/
isort src/
```

## ⚠️ Disclaimer

**Trading cryptocurrencies involves significant risk and can result in the loss of your capital.** 

- This software is for educational purposes only
- Do not trade with money you cannot afford to lose
- Past performance does not guarantee future results
- Always test strategies in paper trading mode first
- The authors are not responsible for any financial losses

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- Create an issue for bugs or feature requests
- Join our community discussions

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) - Crypto exchange trading library
- [Pandas](https://pandas.pydata.org/) - Data analysis library
- [PyTorch](https://pytorch.org/) - Machine learning framework
