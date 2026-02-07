__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json
from datetime import datetime

from config.paths import PROJECT_ROOT
from scripts.compile import compile


if __name__ == "__main__":
    print("=" * 80)
    print("DATASET COMPILATION")
    print("=" * 80)

    # Load tickers from tickers.json
    tickers_path = PROJECT_ROOT / "tickers.json"
    print(f"\nLoading tickers from {tickers_path}...")
    with open(tickers_path) as f:
        tickers = json.load(f)
    print(f"Loaded {len(tickers)} tickers")

    # Define dataset splits with new date ranges
    splits = {
        "train": (datetime(2010, 1, 1), datetime(2021, 1, 1)),
        "eval": (datetime(2022, 1, 1), datetime(2023, 7, 7)),
        "test": (datetime(2024, 7, 1), datetime(2026, 1, 1)),
    }

    print(f"\nDataset splits:")
    for split_name, (start, end) in splits.items():
        print(f"  {split_name:5s}: {start.date()} to {end.date()}")

    # Create timestamped dataset name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"russell1000"

    print(f"\nDataset name: {dataset_name}")
    print(f"\n{'='*80}")

    # Compile datasets using the compile function
    compile(tickers, dataset_name, splits)

    print(f"\n{'='*80}")
    print("COMPILATION COMPLETE")
    print(f"{'='*80}")
