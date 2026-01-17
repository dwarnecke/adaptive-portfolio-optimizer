__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
from pathlib import Path
from datetime import datetime

from features.dataset import FeaturesDataset
from portfolio.portfolio import Portfolio

if __name__ == "__main__":
    print("=" * 80)
    print("PORTFOLIO SIMULATION ON EVALUATION SET")
    print("=" * 80)

    # Load the trained model
    model_dir = Path("outputs/models/forward_20260112_022203-evalIC_117")
    model_path = model_dir / "forward_model.pth"

    print(f"\nLoading model from {model_dir.name}...")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    print("  Model loaded successfully")

    # Dataset directory with regime model
    dataset_dir = Path("outputs/datasets/dataset_20260112_221236")
    regime_path = dataset_dir / "regimes" / "regime_model.pkl"

    # Load pre-computed dataset
    print("\nLoading pre-computed evaluation dataset...")
    eval_dataset = FeaturesDataset.load(dataset_dir, "russell1000_eval.pkl")

    # Evaluation period dates
    eval_start = datetime(2022, 1, 1)
    eval_end = datetime(2023, 7, 1)

    # Create portfolio
    print("\nInitializing portfolio...")
    portfolio = Portfolio(
        dataset=eval_dataset,
        forward_model=model,
        capital=100_000_000,  # $100M starting capital
    )
    print("  Portfolio initialized")

    # Simulate portfolio performance on evaluation period
    print("\nSimulating portfolio performance...")
    print(f"  Period: {eval_start.date()} to {eval_end.date()}")

    stats = portfolio.simulate(eval_start, eval_end)

    # Report results
    print("\n" + "=" * 80)
    print("PORTFOLIO PERFORMANCE RESULTS")
    print("=" * 80)

    final_return = stats["return"][max(stats["return"].keys())]
    max_drawdown = max(stats["drawdown"].values())
    final_sharpe = stats["sharpe"][max(stats["sharpe"].keys())]

    print(f"  Period:              {eval_start.date()} to {eval_end.date()}")
    print(f"  Total Return:        {final_return * 100:.2f}%")
    print(f"  Max Drawdown:        {max_drawdown * 100:.2f}%")
    print(f"  Sharpe Ratio:        {final_sharpe:.3f}")
    print("=" * 80)
