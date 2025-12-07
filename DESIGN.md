# Equity Factor Model - Design Document

## Introduction

This project implements a comprehensive algorithmic trading system designed to model and predict equity returns based on market regime detection, fundamental analysis, and technical indicators. The system is built on the foundation of quantitative factor investing, enhanced with machine learning techniques to identify and exploit patterns in historical market data.

### Overview

The equity factor model operates on the principle that stock returns can be predicted through a combination of:
1. **Market Regime Analysis**: Understanding the current broader market environment
2. **Fundamental Factors**: Evaluating company financials and valuations
3. **Technical Indicators**: Analyzing price action, momentum, and volatility patterns

The system is designed to backtest trading strategies using historical data, allowing for rigorous evaluation before deployment. This approach ensures that all trading decisions are based on empirical evidence rather than speculation.

### Algorithmic Trading Workflow

The model follows a structured machine learning workflow with three distinct phases:

#### 1. Training Set (Historical Period)
The training set uses the earliest portion of historical data to learn relationships between features and future returns. During this phase:
- Market regime states are identified using a Hidden Markov Model (HMM)
- Fundamental ratios (earnings-to-price, book-to-price, debt-to-assets, sales-to-price) are calculated
- Technical indicators (RSI, ATR, momentum, drawdown, volume scores) are computed
- Feature-return relationships are learned through supervised learning

The training period establishes the baseline patterns that the model will use to make predictions.

#### 2. Development Set (Validation Period)
The development set serves as an out-of-sample validation period to:
- Tune hyperparameters without overfitting to the training data
- Evaluate different feature combinations and their predictive power
- Calibrate position sizing and risk management rules
- Detect potential data leakage or overfitting issues

This intermediate period is crucial for model refinement and helps ensure that the model generalizes beyond the training data.

#### 3. Testing Set (Final Evaluation)
The testing set represents completely unseen data that simulates real-world deployment:
- No model parameters are adjusted based on test set performance
- Final performance metrics are calculated (Sharpe ratio, maximum drawdown, total return)
- Comparison to benchmark indices (S&P 500, market-neutral strategies)
- Stress testing under different market conditions (bull markets, bear markets, crashes)

The strict separation between these sets is essential for honest performance evaluation and prevents look-ahead bias that would inflate backtested returns.

### Data Pipeline

The system ingests multiple data sources to create a comprehensive feature set:

**Market Data (via `markets/`):**
- S&P 500 index returns and volatility
- Treasury yield curves (2-year, 10-year spreads)
- Futures term structures (VIX, commodity futures)
- Trading volume and liquidity metrics

**Fundamental Data (via `equities/fundamentals/`):**
- Quarterly balance sheets (assets, liabilities, equity)
- Income statements (revenue, net income, operating income)
- Share counts and market capitalization
- Calculated ratios: E/P, B/P, S/P, Debt/Assets

**Technical Data (via `equities/technicals/`):**
- Price data: open, high, low, close, volume
- Momentum indicators: rate of change, RSI
- Volatility measures: ATR, historical volatility
- Drawdown metrics: maximum drawdown over multiple timeframes
- Volume-based scores: normalized volume patterns

### Hidden Markov Model for Market Regimes

A key innovation in this model is the use of a Hidden Markov Model to detect latent market regimes. Markets exhibit different behavioral patterns during:
- **Bull regimes**: Low volatility, positive returns, low yield spreads
- **Bear regimes**: High volatility, negative returns, widening credit spreads
- **Transitional regimes**: Mixed signals, increased uncertainty

The HMM learns these states from observable market features and provides probabilistic predictions of the current regime. This regime information serves as an additional input feature, allowing the model to adapt its predictions based on market conditions.

The HMM is trained using the Baum-Welch algorithm (Expectation-Maximization) on market features including:
- 20-day log returns of market indices
- 20-day rolling standard deviation (volatility)
- Treasury yield curve slopes (2Y-10Y)
- Futures term structure differentials

Each trading day, the model computes the probability distribution over hidden states, providing a quantitative measure of market uncertainty that enhances predictive accuracy.

### Feature Engineering Philosophy

Features are designed with the following principles:
1. **No look-ahead bias**: All features use only information available at the time of prediction
2. **Multiple timeframes**: Indicators calculated over 5, 20, 60, 120, 250, 500, and 1000 trading days
3. **Log transformations**: Applied to price and volume data to normalize distributions
4. **Normalized scores**: Features are standardized to comparable scales for machine learning
5. **Fundamental-technical fusion**: Combines value investing principles with momentum strategies

### Neural Network Architecture

The system employs neural networks for return prediction, leveraging their ability to capture complex non-linear relationships between features and future returns. The neural network architecture is designed to process the diverse feature set (market regimes, fundamentals, technicals) and generate accurate return forecasts.

Key architectural components include:
- **Input layer**: Accepts normalized feature vectors combining market regime probabilities, fundamental ratios, and technical indicators
- **Hidden layers**: Multiple dense layers with non-linear activation functions (ReLU, tanh) to learn complex feature interactions
- **Output layer**: Single neuron with linear activation for continuous return prediction

The network predicts future returns (typically 20-day forward returns) based on the current feature set. Portfolio construction then translates these predictions into position weights, subject to risk constraints.

Training incorporates:
- **Loss function**: Mean squared error or Huber loss for robust optimization
- **Optimizer**: Adam or RMSprop with learning rate scheduling
- **Batch normalization**: To stabilize training and improve convergence
- **Early stopping**: Based on validation set performance to prevent overfitting
- **Regularization**: L2 weight decay and dropout to improve generalization

### Risk Management

The system incorporates multiple layers of risk control:
- Position size limits based on volatility
- Portfolio-level constraints on sector exposure
- Stop-loss rules based on maximum drawdown thresholds
- Leverage limits to control tail risk
- Regime-dependent position sizing (more conservative in volatile regimes)

## Project Structure

### Directory Organization

```
equity-factor-model/
│
├── data/
│   ├── loader.py                 # Data loading utilities
│   └── external/
│       └── simfin/              # SimFin fundamental data
│           ├── us-balance-quarterly.csv
│           ├── us-income-quarterly.csv
│           └── us-shareprices-daily.csv
│
├── equities/
│   ├── data.py                  # Main equity data aggregator
│   ├── fundamentals/
│   │   ├── fundamentals.py      # Fundamental data management
│   │   ├── balances.py          # Balance sheet processing
│   │   ├── earnings.py          # Income statement processing
│   │   └── shares.py            # Share count processing
│   └── technicals/
│       ├── technicals.py        # Technical indicator management
│       ├── atr.py               # Average True Range
│       ├── drawdown.py          # Maximum drawdown calculations
│       ├── returns.py           # Return calculations
│       ├── rsi.py               # Relative Strength Index
│       ├── scores.py            # Normalized price scores
│       └── volumes.py           # Volume-based indicators
│
├── markets/
│   ├── data.py                  # Market benchmark data aggregator
│   ├── model.py                 # Market prediction models
│   ├── hidden_markov_model.py   # HMM implementation for regime detection
│   ├── features/
│   │   ├── futures.py           # Futures term structure features
│   │   ├── index_returns.py     # Market index return features
│   │   ├── index_volatility.py  # Market volatility features
│   │   └── treasuries.py        # Treasury yield curve features
│   └── returns/
│       ├── index_returns.py     # Forward return calculations
│       └── index_volatility.py  # Forward volatility calculations
│
├── other/
│   ├── dates.py                 # Date handling utilities
│   ├── keys.py                  # API key management
│   └── tickers.py               # Ticker utilities and price lookups
│
├── main.py                      # Main entry point and testing
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview

```

### Module Descriptions

#### `data/`
Handles all data loading and caching operations. The `loader.py` module provides unified interfaces for loading fundamental data from SimFin, ensuring efficient data access across the application.

#### `equities/`
Contains all equity-specific data processing:
- **`data.py`**: Aggregates fundamental and technical data for individual stocks
- **`fundamentals/`**: Processes quarterly financial statements and calculates valuation ratios
- **`technicals/`**: Computes technical indicators from price and volume data

#### `markets/`
Manages market-level features and regime detection:
- **`data.py`**: Aggregates market features (returns, volatility, yield curves)
- **`hidden_markov_model.py`**: Implements HMM for regime detection using Baum-Welch training
- **`features/`**: Calculates current market state features
- **`returns/`**: Calculates forward-looking target variables

#### `other/`
Utility modules for common operations:
- **`dates.py`**: Trading calendar, timezone handling, date utilities
- **`keys.py`**: Secure API key management
- **`tickers.py`**: Ticker symbol validation and price lookup functions

### Data Flow

1. **Raw Data Ingestion**: Historical data is loaded from SimFin (fundamentals) and Yahoo Finance (technicals)
2. **Feature Calculation**: Each module calculates its respective features with appropriate lookback windows
3. **Feature Alignment**: All features are aligned to common trading dates, handling missing data appropriately
4. **Regime Detection**: HMM processes market features to identify current regime probabilities
5. **Model Input Construction**: Features are combined into a unified dataset for model training/prediction
6. **Prediction Generation**: Trained models generate return predictions for each stock
7. **Portfolio Construction**: Predictions are translated into position weights with risk constraints
8. **Performance Evaluation**: Backtest results are analyzed across multiple metrics

### Key Design Decisions

1. **Separation of Concerns**: Market data, equity fundamentals, and technicals are processed independently
2. **Lazy Loading**: Data is cached locally to minimize API calls and improve performance
3. **Timezone Awareness**: All datetime objects use timezone-aware representations to prevent subtle bugs
4. **Vectorized Operations**: Pandas and NumPy are used extensively for efficient computation
5. **Modular Architecture**: Each component can be tested and validated independently
6. **Neural Network Training**: PyTorch or TensorFlow framework for flexible model development and GPU acceleration

### Future Enhancements

Potential improvements to the system include:
- **Alternative data sources**: Sentiment analysis, satellite imagery, web scraping
- **Real-time execution**: Integration with brokerage APIs for live trading
- **Transaction cost modeling**: Bid-ask spreads, market impact, commissions
- **Portfolio optimization**: Mean-variance optimization, risk parity
- **Model monitoring**: Drift detection, performance degradation alerts
- **Attention mechanisms**: Transformer architectures for better feature weighting
- **Advanced neural architectures**: LSTMs for temporal dependencies, CNNs for pattern recognition, Graph Neural Networks for sector relationships

### Performance Metrics

The system evaluates strategies using:
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: Return/maximum drawdown
- **Win Rate**: Percentage of profitable trades
- **Alpha/Beta**: Performance vs. market benchmark
- **Turnover**: Portfolio rebalancing frequency

These metrics provide a comprehensive view of strategy performance across risk, return, and efficiency dimensions.
