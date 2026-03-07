__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
from pathlib import Path
from datetime import datetime

from features.dataset import FeaturesDataset
from portfolio.portfolio import Portfolio

if __name__ == "__main__":
    print("=" * 80)
    print("PORTFOLIO SIMULATION ON TRAINING SET")
    print("=" * 80)

    # Load the trained model
    model_dir = Path("outputs/models/forward_20260211_033719")
    model_path = model_dir / "forward_model.pth"

    print(f"\nLoading model from {model_dir.name}...")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    print("  Model loaded succesAsfully")

    # Load the training dataset
    dataset_dir = Path("outputs/datasets/dataset_20260205_230400")
    print("\nLoading pre-computed training dataset...")
    train_dataset = FeaturesDataset.load(dataset_dir, "russell1000_train.pkl")

    # Training period dates
    train_start = datetime(2010, 1, 1)
    train_end = datetime(2021, 1, 1)

    # Create portfolio
    print("\nInitializing portfolio...")
    portfolio = Portfolio(
        dataset=train_dataset,
        forward_model=model,
        capital=100_000_000,  # $100M starting capital
    )
    print("  Portfolio initialized")

    # Simulate portfolio performance on training period
    print("\nSimulating portfolio performance...")
    stats = portfolio.simulate(train_start, train_end)

    # Report results
    print("\n" + "=" * 80)
    print("PORTFOLIO PERFORMANCE RESULTS")
    print("=" * 80)

    final_return = stats["return"][max(stats["return"].keys())]
    max_drawdown = max(stats["drawdown"].values())
    final_sharpe = stats["sharpe"][max(stats["sharpe"].keys())]

    print(f"  Period:              {train_start.date()} to {train_end.date()}")
    print(f"  Total Return:        {final_return * 100:.2f}%")
    print(f"  Max Drawdown:        {max_drawdown * 100:.2f}%")
    print(f"  Sharpe Ratio:        {final_sharpe:.3f}")
    print("=" * 80)
