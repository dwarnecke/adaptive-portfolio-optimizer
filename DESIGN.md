# Transformer-Based Portfolio System - Design Document

## Introduction

This project implements an end-to-end quantitative portfolio management system that combines transformer-based machine learning with mean-variance optimization to construct market-neutral equity portfolios. The system predicts both returns and volatility for individual stocks, then uses these predictions to build optimal portfolios with explicit risk control.

### Overview

The system operates through four integrated components:

1. **Feature Engineering**: Constructs 41 technical, fundamental, and market features for each stock-date observation
2. **Market Regime Detection**: Hidden Markov Model identifies latent market states (bull/bear/transitional)
3. **Neural Forward Model**: Transformer neural network predicts 20-day forward returns and daily volatility
4. **Portfolio Optimization**: Mean-variance optimization constructs market-neutral portfolios with transaction cost modeling

The architecture achieves an Information Coefficient (IC) of ~0.12 on out-of-sample data, demonstrating predictive power while maintaining low correlation with market beta through strict market neutrality constraints.

### Data Pipeline and Train/Eval/Test Split

The system uses a strict temporal train/validation/test split to prevent look-ahead bias:

#### Training Set (2010-01-01 to 2021-01-01)
- **Purpose**: Learn relationships between features and forward returns/volatility
- **Samples**: ~1.29M stock-date observations
- **Usage**: Train transformer model weights, fit normalization parameters
- **Expected IC**: 0.15-0.17 (with some overfitting expected)

#### Evaluation Set (2022-01-01 to 2023-07-01)  
- **Purpose**: Unbiased model selection and hyperparameter tuning
- **Samples**: ~238K observations
- **Usage**: Select best model, tune portfolio parameters (position sizing, rebalancing frequency)
- **Target IC**: 0.10-0.13 (out-of-sample generalization)

#### Test Set (2024-07-01 to 2025-07-01)
- **Purpose**: Final unbiased performance evaluation
- **Samples**: ~172K observations  
- **Usage**: Simulate live deployment, no parameter changes allowed
- **Constraint**: No model or portfolio parameters adjusted based on test performance

The temporal structure ensures that all predictions use only information available at the prediction time, preventing subtle forms of data leakage that would inflate backtested performance.

### Feature Engineering

The system constructs a comprehensive feature set combining technical indicators, fundamental ratios, and market observations. All features are calculated using only point-in-time data to prevent look-ahead bias.

#### Technical Features (via `equities/technicals/`)

**Price Momentum (Multiple Timeframes)**
- Rate of Change (ROC): 5, 60, 120, 250-day periods
- Captures both short-term reversals and long-term trends
- Normalized to handle different price scales

**Relative Performance**  
- Relative Return vs. Market: 5, 60, 250-day periods
- Measures stock-specific performance independent of market moves
- Critical for market-neutral strategies

**Beta Estimation**
- Rolling Beta vs. S&P 500: 60, 250-day windows
- Quantifies systematic risk exposure
- Used for position sizing and risk management

**Volatility Indicators**
- ATR (Average True Range): 14-day period
- RSI (Relative Strength Index): 14-day period  
- Identifies overbought/oversold conditions

**Drawdown Metrics**
- Maximum Drawdown: 5, 60, 120, 250-day lookbacks
- Measures downside risk and recovery patterns
- Early warning signal for deteriorating stocks

**Volume Analysis**
- Log Volume Normal Scores: 5, 60, 120, 250-day windows
- Detects abnormal trading activity
- Leading indicator for price moves

**Price Distribution Scores**
- Log Close Normal Scores: 5, 60, 120, 250-day windows
- Standardized position within recent price range
- Mean-reversion signal

#### Fundamental Features (via `equities/fundamentals/`)

**Valuation Ratios**
- E/P (Earnings-to-Price): Value investing signal
- S/P (Sales-to-Price): Revenue-based valuation  
- CF/P (Cash Flow-to-Price): Quality of earnings
- B/P (Book-to-Price): Asset-based valuation

**Financial Health**
- D/A (Debt-to-Assets): Leverage and solvency
- Operating Margin: Profitability efficiency
- ROE (Return on Equity): Shareholder value creation

**Growth Metrics**
- Revenue Growth: YoY quarterly growth rate
- Market Cap: Size factor exposure

**Data Quality**
- FUND_NA, TECH_NA: Binary indicators for missing data
- Allows model to learn missingness patterns

All fundamental data uses quarterly filings with proper alignment to announcement dates (not report period end dates) to prevent look-ahead bias.

#### Market Features (via `markets/`)

**Market Observations**
- Term Structure Difference: VIX futures term structure slope
- 20-Day Log Return: Recent market momentum  
- 20-Day Log Return Std: Recent market volatility
- Yield Slope: 10Y-2Y Treasury spread (recession indicator)

**Regime Probabilities (from HMM)**
- regime_0, regime_1, regime_2: Probability distribution over 3 hidden states
- Captures bull/bear/transitional market environments
- Allows model to adapt predictions to market conditions

**Total Feature Count**: 41 features per stock-date observation
- 24 technical indicators (momentum, volatility, volume, drawdown)
- 10 fundamental ratios (valuation, profitability, growth)
- 4 market observations (returns, volatility, term structure, yield curve)
- 3 regime probabilities (HMM-based market states)

Features are stored in a 60-day temporal window (lookback period), allowing the transformer model to capture time-dependent patterns and feature interactions across multiple timeframes.

### Hidden Markov Model for Market Regimes

The Hidden Markov Model (HMM) identifies latent market states that influence all stocks simultaneously. Markets exhibit distinct behavioral regimes:

- **Regime 0 (Low Volatility/Bull)**: Low volatility, positive returns, narrow yield spreads
- **Regime 1 (Transitional)**: Mixed signals, moderate volatility, uncertain direction
- **Regime 2 (High Volatility/Bear)**: High volatility, negative returns, widening spreads

**Training Process**:
1. **Input Features**: Market observations (20-day returns, volatility, yield slope, term structure)
2. **Algorithm**: Baum-Welch (Expectation-Maximization) with 50 iterations
3. **States**: 3 hidden states with learned transition probabilities
4. **Output**: Probability distribution over states for each trading day

**Implementation** (`markets/regime/`):
- `model.py`: Orchestrates regime detection pipeline
- `hmm.py`: Core HMM implementation with forward-backward algorithm

The regime probabilities serve as additional features to the forward model, allowing predictions to adapt based on market environment. This captures the empirical observation that factor performance varies across regimes (e.g., momentum stronger in bull markets, value stronger in bear markets).

### Transformer Forward Model

The forward model is a transformer-based neural network that predicts both 20-day forward returns and daily volatility for each stock. This joint prediction enables sophisticated portfolio construction with explicit uncertainty quantification.

**Architecture** (`models/model.py`):
- **Input**: 60-day window × 41 features = temporal sequence
- **Input Layer**: Linear projection from 41 features to hidden dimensions
- **Attention Mechanism**: 
  * 1 transformer encoder layer
  * 2 attention heads
  * Learns which historical time steps and features are most predictive
  * Captures non-linear feature interactions across time
- **Hidden Layers**: 64 hidden units with layer normalization
- **Feed-Forward**: 64 → 256 → 64 dimensions within transformer layer
- **Output Layer**: Linear projection from 64 hidden to 2 targets
- **Total Parameters**: ~52,800 trainable parameters
- **Output**: 2 values per prediction
  * μ̂: Predicted 20-day forward return
  * σ̂: Predicted daily volatility over the forward period

**Training Process** (`scripts/train.py`):

**Loss Function**: Negative Log-Likelihood (NLL)
```
L = λ_μ × [(y - μ̂)² / (2σ̂²)] + log(σ̂)
```
Where:
- λ_μ: Weight on return prediction (typically 4-8)
- First term: Penalizes return prediction errors, scaled by uncertainty
- Second term: Penalizes excessive (low σ̂) predictions

This loss encourages the model to:
1. Predict returns accurately
2. Estimate uncertainty honestly (avoids predicting zero volatility)
3. Be more confident when predictions are accurate

**Hyperparameters**:
- Learning rate: 2^-8 to 2^-6
- Batch size: 1024-2048 (balances stability and training speed)
- Epochs: 16-64 with early stopping based on eval IC
- Regularization: L2 weight decay 2^-10, dropout 0-0.15
- Optimizer: SGD with weight decay

**Training Metrics**:
- **Train IC**: Spearman correlation between predictions and realized returns (target: 0.15-0.17)
- **Eval IC**: Out-of-sample IC on evaluation set (target: 0.10-0.13)
- **Loss**: NLL loss value (decreases during training)

**Prediction Quality**:
- Information Coefficient: 0.10-0.13 on out-of-sample data
- Volatility IC: ~0.65 (excellent uncertainty calibration)
- Quintile Spread: Top quintile outperforms bottom by 2-3% over 20 days

The joint return-volatility prediction is critical for portfolio optimization, as it enables both expected return maximization and explicit risk control.

### Dataset Construction

**Universe**: Russell 1000 stocks (~600-700 tickers depending on period)

**Data Sources**:
- **Fundamentals**: SimFin quarterly financial statements
- **Prices**: Daily OHLCV data from Yahoo Finance
- **Market Data**: S&P 500, Treasury yields, VIX futures

**Dataset Generation** (`features/dataset.py`):
1. **Feature Calculation**: Compute all technical, fundamental, and market features
2. **Temporal Alignment**: Align features to common trading dates
3. **Window Creation**: Create 60-day lookback windows for each prediction point
4. **Target Calculation**: Compute 20-day forward returns and realized volatility
5. **Validation**: Remove samples with insufficient history or missing targets

**Data Structure**:
- **Features (X)**: (samples, 60 days, 41 features) tensor
- **Targets (y)**: (samples, 2) tensor [forward return, forward volatility]
- **Metadata**: Ticker symbols, dates, market caps for each sample

**Sample Counts**:
- Training: ~1.29M samples (2016-2021)
- Evaluation: ~238K samples (2022-mid 2023)
- Testing: ~172K samples (mid 2024-mid 2025)

**Storage**: Pickled FeaturesDataset objects (~4-5 GB each) for fast loading

### Portfolio Optimization

The portfolio construction module (`portfolio/`) translates model predictions into optimal position weights using mean-variance optimization with explicit constraints.

**Mean-Variance Optimization** (`portfolio/model.py`):

**Objective**: Maximize risk-adjusted returns
```
max w^T μ - λ w^T Σ w
```
Where:
- w: Portfolio weights (position sizes)
- μ: Predicted returns from forward model
- Σ: Covariance matrix (mixed forward + historical)
- λ: Risk aversion parameter (implicitly determined by constraints)

**Market Neutral Constraint**:
```
1^T w = 0  (sum of weights equals zero)
```
This ensures the portfolio is dollar-neutral (equal long and short exposure), eliminating market beta and focusing returns on stock selection skill.

**Solution** (closed-form):
```
λ = (1^T Σ^-1 μ) / (1^T Σ^-1 1)
w = Σ^-1 (μ - λ1)
```
This computes optimal weights that maximize the Sharpe ratio subject to market neutrality.

**Covariance Estimation** (`portfolio/covariance.py`):

**Mixed Covariance Approach**:
```
Σ = α × Σ_forward + (1-α) × Σ_historical
```
Where:
- α = 0.3-0.5: Mixing parameter
- Σ_forward: Covariance from model-predicted volatilities (diagonal)
- Σ_historical: Sample covariance with Ledoit-Wolf shrinkage

**Historical Covariance**:
- Window: 250 trading days (~1 year)
- Shrinkage: 0.5 (50% toward diagonal)
- Handles 600+ stocks with robust estimation

**Forward Covariance**:
- Uses predicted daily volatilities (σ̂) from forward model
- Diagonal matrix: assumes zero correlation in residual component
- Captures predicted changes in individual stock volatilities

This mixed approach combines:
- Forward model's ability to predict changing volatilities
- Historical correlation structure for diversification
- Shrinkage to prevent overfitting with high-dimensional covariance

**Position Sizing Constraints** (`config/hyperparameters.py`):
- **min_scalar**: Minimum position size (e.g., 0.25% = 2^-2)
  * Prevents dust positions with high turnover
- **max_scalar**: Maximum position size (e.g., 8 = 2^3 → 1.2% max)
  * Prevents over-concentration in single stocks
- **max_leverage**: Total leverage limit (e.g., 2.0)
  * Sum of absolute weights ≤ 2.0
  * Allows 100% long + 100% short = 200% gross exposure

**Rebalancing**:
- Frequency: Every 16 trading days (~3 weeks)
- Rationale: Balances alpha capture vs. transaction costs
- Universe: 100-400 positions depending on scalars

**Transaction Costs** (`portfolio/portfolio.py`):
- **Slippage**: 5 basis points (0.05%)
  * Applied via price multiplier at entry
- **Commission**: 2 basis points (0.02%)  
  * Applied on dollar value of trades
- **Total**: ~7 bps one-way, ~12-14 bps round-trip
- **Annual Impact**: ~0.6-1.5% drag depending on turnover

**Portfolio Simulation**:
1. Generate predictions for all stocks on rebalancing date
2. Estimate covariance matrix (mixed approach)
3. Solve for optimal weights with market neutral constraint
4. Apply position size limits and leverage constraint
5. Execute trades with transaction costs
6. Track P&L, drawdown, Sharpe ratio daily
7. Repeat on next rebalancing date

### Performance Metrics and Results

**Information Coefficient (IC)**: 0.10-0.13 on evaluation set
- Measures rank correlation between predictions and realized returns
- IC > 0.10 is considered strong in equity markets
- Volatility predictions achieve IC ~0.65 (exceptional)

**Portfolio Performance** (Evaluation Set, 2022-mid 2023):
- Total Return: 1.5-3.0%
- Sharpe Ratio: 0.15-0.35
- Maximum Drawdown: 3-4%
- Positions: 100-400 stocks depending on constraints

**Key Insights**:
- Model demonstrates genuine predictive power (IC 0.12)
- Market neutrality eliminates beta exposure effectively
- Risk control is excellent (low drawdowns)
- Sharpe ratio limited by over-diversification and transaction costs
- Further optimization of position sizing and rebalancing frequency could improve Sharpe

**Observed Trade-offs**:
- Larger max positions → Higher conviction but more concentration risk
- More frequent rebalancing → Better alpha capture but higher costs
- Larger batch sizes → Smoother training but fewer gradient updates
- Higher lambda_mu → Better return prediction but worse volatility estimation

## Project Structure

```
transformer-portfolio/
│
├── config/
│   ├── hyperparameters.py       # All model and portfolio parameters
│   └── paths.py                 # Data paths configuration
│
├── data/
│   ├── raw/
│   │   ├── loader.py            # SimFin data loading
│   │   └── simfin/              # Fundamental data CSVs
│   └── datasets/                # Generated FeaturesDataset pickles
│
├── features/
│   ├── dataset.py               # FeaturesDataset class and generation
│   └── features.py              # Feature orchestration
│
├── equities/
│   ├── data.py                  # Equity data aggregator
│   ├── features.py              # Equity feature orchestration
│   ├── fundamentals/
│   │   ├── fundamentals.py      # Fundamental data management
│   │   ├── balances.py          # Balance sheet processing
│   │   ├── earnings.py          # Income statement processing
│   │   └── shares.py            # Share count and market cap
│   └── technicals/
│       ├── technicals.py        # Technical indicator management
│       ├── trend.py             # Momentum and ROC
│       ├── relation.py          # Relative returns and beta
│       ├── volatility.py        # ATR and volatility
│       ├── score.py             # Normalized price/volume scores
│       └── ...
│
├── markets/
│   ├── data.py                  # Market data aggregator
│   ├── features.py              # Market feature orchestration
│   ├── observations.py          # Observable market features
│   ├── regime/
│   │   ├── model.py             # Regime detection pipeline
│   │   └── hmm.py               # HMM implementation
│   └── macros/
│       ├── index_momentum.py    # Market return features
│       ├── index_volatility.py  # Market volatility features
│       ├── treasury_slope.py    # Yield curve features
│       └── vix_term.py          # VIX futures features
│
├── models/
│   ├── model.py                 # ForwardModel transformer architecture
│   └── models/                  # Saved trained models
│
├── portfolio/
│   ├── portfolio.py             # Portfolio simulation
│   ├── model.py                 # MVO optimization
│   ├── covariance.py            # Covariance estimation
│   ├── universe.py              # Universe selection
│   └── positions/
│       ├── position.py          # Position tracking
│       └── entry.py             # Entry price with slippage
│
├── scripts/
│   ├── train.py                 # Model training script
│   └── ...                      # Various analysis scripts
│
├── main.py                      # Main entry point
├── analyze_model.py             # Model evaluation script
└── playground.py                # Experimentation notebook
```

### Key Design Principles

1. **Modularity**: Each component (features, HMM, forward model, portfolio) is independent
2. **No Look-Ahead Bias**: Strict temporal ordering ensures all features use only past data
3. **Efficient Caching**: Datasets pre-computed and stored for fast iteration
4. **Configurable**: All hyperparameters centralized in config/hyperparameters.py
5. **Extensible**: New features, models, or portfolio strategies easily added
6. **Type Safety**: Type hints throughout for code clarity
7. **GPU Support**: Automatic CUDA utilization for model training

### Data Flow Summary

```
Raw Data (SimFin, Yahoo Finance)
         ↓
Feature Engineering (Technical + Fundamental + Market)
         ↓
HMM Regime Detection → Regime Probabilities
         ↓
FeaturesDataset (60-day windows with targets)
         ↓
Forward Model Training (Transformer + NLL Loss)
         ↓
Predictions (Returns + Volatilities)
         ↓
Covariance Estimation (Mixed Forward + Historical)
         ↓
Mean-Variance Optimization (Market Neutral)
         ↓
Portfolio Simulation (With Transaction Costs)
         ↓
Performance Metrics (IC, Sharpe, Drawdown)
```

## Conclusion

This system demonstrates a complete quantitative portfolio management pipeline, from raw data to backtested performance. Key achievements:

- **Strong Predictive Power**: IC 0.12 validates model signal quality
- **Sophisticated Architecture**: Transformer attention mechanism with joint return-volatility prediction
- **Robust Risk Management**: Market neutral with explicit covariance estimation
- **Production-Ready**: Transaction costs, rebalancing, position limits all modeled
- **Extensible Framework**: Modular design supports rapid experimentation

The project showcases expertise in:
- Machine learning for finance (transformers, attention mechanisms, NLL loss)
- Portfolio theory (mean-variance optimization, market neutrality, covariance estimation)
- Feature engineering (technical, fundamental, regime-based)
- Production ML considerations (data leakage prevention, transaction costs, backtesting)

Future enhancements could include: ensemble models, alternative universes (Russell 2000, international), sector neutrality constraints, alternative optimization methods (risk parity, CVaR), and real-time execution capabilities.
