__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from config.hyperparameters import HYPERPARAMETERS
from features.dataset import FeaturesDataset
from portfolio.portfolio import Portfolio

if __name__ == "__main__":
    print("=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    # Load latest model and test dataset
    model_dir = Path("outputs/models/forward_20260122_091741")
    dataset_dir = Path("outputs/datasets/dataset_20260120_223016")

    print(f"\nModel: {model_dir.name}")
    print(f"Dataset: {dataset_dir.name}")

    print(f"\nLoading model...")
    model = torch.load(
        model_dir / "forward_model.pth",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = FeaturesDataset.load(dataset_dir, "russell1000_test.pkl")
    print(
        f"Loaded {len(test_dataset):,} test samples ({len(test_dataset.index)} tickers)"
    )

    # Portfolio simulation
    print(f"\n{'='*80}")
    print("PORTFOLIO SIMULATION")
    print(f"{'='*80}")

    print(f"\nPortfolio parameters:")
    for key, value in HYPERPARAMETERS["portfolio"].items():
        print(f"  {key}: {value}")

    print(f"\nInitializing portfolio...")
    print(f"  Period: 2024-07-01 to 2025-07-31")
    print(f"  Initial capital: $100,000,000")

    capital = 100_000_000
    portfolio_test = Portfolio(
        dataset=test_dataset,
        forward_model=model,
        capital=capital,
        parameters=HYPERPARAMETERS["portfolio"],
    )

    # Simulate test period
    start_date = datetime(2024, 7, 15)
    end_date = datetime(2025, 7, 31)

    print(f"\nRunning simulation...")
    stats_test = portfolio_test.simulate(start_date, end_date)

    # Extract final metrics
    final_value_test = list(portfolio_test.history.values())[-1]
    final_return_test = list(stats_test["return"].values())[-1]
    max_drawdown_test = max(stats_test["drawdown"].values())
    final_sharpe_test = list(stats_test["sharpe"].values())[-1]

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\nPerformance Metrics:")
    print(f"  Final value:     ${final_value_test:,.0f}")
    print(f"  Total return:    {final_return_test*100:+.2f}%")
    print(f"  Max drawdown:    {max_drawdown_test*100:.2f}%")
    print(f"  Sharpe ratio:    {final_sharpe_test:.2f}")

    # Portfolio statistics
    print(f"\n{'='*80}")
    print("PORTFOLIO STATISTICS")
    print(f"{'='*80}")

    leverages = [
        sum(abs(w) for w in weights.values())
        for weights in portfolio_test.weight_history.values()
    ]
    num_securities = [
        len(weights) for weights in portfolio_test.weight_history.values()
    ]

    print(f"\nLeverage:")
    print(f"  Average: {np.mean(leverages):.2f}x")
    print(f"  Min:     {min(leverages):.2f}x")
    print(f"  Max:     {max(leverages):.2f}x")

    print(f"\nSecurities per rebalance:")
    print(f"  Average: {np.mean(num_securities):.0f}")
    print(f"  Min:     {min(num_securities)}")
    print(f"  Max:     {max(num_securities)}")

    print(f"\nRebalancing:")
    rebal_freq = HYPERPARAMETERS["portfolio"].get(
        "rebalance_frequency", HYPERPARAMETERS["portfolio"]["length"]
    )
    print(f"  Frequency:        Every {rebal_freq} days")
    total_cost_bps = (
        HYPERPARAMETERS["portfolio"]["slippage_bps"]
        + HYPERPARAMETERS["portfolio"]["commission_bps"]
    )
    print(f"  Cost per trade:   {total_cost_bps} bps")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nModel: {model_dir.name}")
    print(f"Parameters: {total_params:,}")
    print(f"Test period: 2024-07-01 to 2025-07-31 (out-of-sample)")
    print(
        f"Portfolio: {final_return_test*100:+.2f}% return, {final_sharpe_test:.2f} Sharpe, {max_drawdown_test*100:.2f}% max drawdown."
    )
    print(f"\n{'='*80}")
