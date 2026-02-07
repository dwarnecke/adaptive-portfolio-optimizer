__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from config.hyperparameters import HYPERPARAMETERS
from config.paths import MODELS_DIR, DATASETS_DIR
from features.dataset import FeaturesDataset
from portfolio.portfolio import Portfolio


def evaluate(
    model_path: Path | str,
    dataset_path: Path | str,
    dataset_name: str,
    start_date: datetime,
    end_date: datetime,
    capital: float = 100_000_000,
    parameters: dict = HYPERPARAMETERS["portfolio"],
) -> dict:
    """
    Evaluate a trained model on a dataset via portfolio simulation.
    :param model_path: Path to saved model file
    :param dataset_path: Path to dataset directory
    :param dataset_name: Name of dataset file
    :param start_date: Start date for simulation
    :param end_date: End date for simulation
    :param capital: Initial portfolio capital
    :param parameters: Portfolio parameters dictionary
    :return: Dictionary with performance metrics
    """
    print(f"\nLoading model from {model_path}...")
    model = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.eval()

    print(f"Loading dataset {dataset_name}...")
    dataset = FeaturesDataset.load(dataset_path, dataset_name)
    print(f"Loaded {len(dataset):,} samples ({len(dataset.index)} tickers)")

    print(f"\nInitializing portfolio...")
    portfolio = Portfolio(
        dataset=dataset,
        forward_model=model,
        capital=capital,
        parameters=parameters,
    )

    print(f"Running simulation from {start_date.date()} to {end_date.date()}...")
    stats = portfolio.simulate(start_date, end_date)

    final_value = list(portfolio.history.values())[-1]
    final_return = list(stats["return"].values())[-1]
    max_drawdown = max(stats["drawdown"].values())
    final_sharpe = list(stats["sharpe"].values())[-1]

    leverages = [
        sum(abs(w) for w in weights.values())
        for weights in portfolio.weight_history.values()
    ]

    results = {
        "final_value": final_value,
        "total_return": final_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": final_sharpe,
        "avg_leverage": np.mean(leverages),
        "num_rebalances": len(portfolio.weight_history),
    }

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\nFinal value:     ${final_value:,.0f}")
    print(f"Total return:    {final_return*100:+.2f}%")
    print(f"Sharpe ratio:    {final_sharpe:.2f}")
    print(f"Max drawdown:    {max_drawdown*100:.2f}%")
    print(f"Avg leverage:    {np.mean(leverages):.2f}x")
    print(f"Rebalances:      {len(portfolio.weight_history)}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    
    # Find most recent model and dataset for simplicity
    models = sorted(MODELS_DIR.glob("forward_*"))
    datasets = sorted(DATASETS_DIR.glob("dataset_*"))
    
    if not models:
        print("No models found. Run train.py first.")
        exit(1)
    if not datasets:
        print("No datasets found. Run compile.py first.")
        exit(1)

    model_dir = models[-1]
    dataset_path = datasets[-1]
    model_path = model_dir / "forward_model.pth"
    
    # Find dataset files in the directory
    eval_files = list(dataset_path.glob("*_eval.pkl"))
    test_files = list(dataset_path.glob("*_test.pkl"))
    if not eval_files or not test_files:
        print("Eval or test dataset not found.")
        exit(1)
    
    print(f"Using model: {model_dir.name}")
    print(f"Using dataset: {dataset_path.name}")
    
    # Evaluate the model on the evaluation set for trained performance
    print("\n" + "="*80)
    print("EVAL SET EVALUATION")
    print("="*80)
    evaluate(
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_name=eval_files[0].name,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 7, 1),
    )
    
    # Evaluate the model on the test set for out of sample performance
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    evaluate(
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_name=test_files[0].name,
        start_date=datetime(2024, 7, 1),
        end_date=datetime(2025, 7, 1),
    )
