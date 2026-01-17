__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json
from pathlib import Path
from datetime import datetime

from config.hyperparameters import HYPERPARAMETERS
from scripts.compile import compile

if __name__ == "__main__":
    print("=" * 80)
    print("COMPILING RUSSELL 1000 DATASET")
    print("=" * 80)

    # Load Russell 1000 tickers
    with open("russell1000_tickers.json", "r") as f:
        tickers = json.load(f)

    print(f"\nLoaded {len(tickers)} Russell 1000 tickers")

    # Define date ranges for train/eval/test splits
    splits = {
        "train": (datetime(2010, 1, 1), datetime(2021, 1, 1)),
        "eval": (datetime(2022, 1, 1), datetime(2023, 7, 1)),
        "test": (datetime(2024, 7, 1), datetime(2025, 7, 1)),
    }

    print(f"\nDate ranges:")
    print(f"  Train: {splits['train'][0].date()} to {splits['train'][1].date()}")
    print(f"  Eval:  {splits['eval'][0].date()} to {splits['eval'][1].date()}")
    print(f"  Test:  {splits['test'][0].date()} to {splits['test'][1].date()}")
    print("=" * 80)
    print()

    # Compile the dataset
    dataset_dir = compile(
        tickers=tickers,
        dataset_name="russell1000",
        splits=splits,
        parameters=HYPERPARAMETERS["features"],
        directory="outputs/datasets",
    )

    print("\n" + "=" * 80)
    print("COMPILATION COMPLETE")
    print(f"Dataset saved to: {dataset_dir}")
    print("=" * 80)
