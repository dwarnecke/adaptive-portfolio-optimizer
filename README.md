# Transformer-Based Portfolio Optimization

A market-neutral quantitative equity strategy that uses deep learning to predict stock returns and construct optimal portfolios. The system trains a transformer model on historical market data, technical indicators, and fundamental ratios to forecast 20-day forward returns with accompanying volatility estimates. These predictions feed into a mean-variance optimizer that constructs long-short portfolios, achieving institutional-quality risk-adjusted returns.

## Pipeline

1. **Feature Engineering**: Computes 41 features per stock including momentum indicators, volatility measures, fundamental ratios (P/E, ROE, etc.), and market regime probabilities from a Hidden Markov Model
2. **Model Architecture**: 1-layer transformer (14K parameters) processes 60-day sequences to predict 20-day returns and daily volatility
3. **Portfolio Construction**: Mean-variance optimization with market-neutral constraint (longs offset shorts), 16x maximum leverage, transaction cost modeling (7 bps)

## Results

Test set 2024-2025
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 1.17 |
| **Annualized Return** | 28.94% |
| **Max Drawdown** | 12.47% |

**Requirements**: Python 3.12+, SimFin API key

```bash
pip install -r requirements.txt
export SIMFIN_API_KEY=your_key_here

python -m scripts.compile   # Create train/eval/test datasets
python -m scripts.train     # Train model (modify epochs in script)
python -m scripts.evaluate  # Evaluate on eval and test sets
```

See [DESIGN.md](DESIGN.md) for architecture details.

---

**Author**: Dylan Warnecke | dylan.warnecke@gmail.com
