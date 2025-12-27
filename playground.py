__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json
from pathlib import Path
from datetime import datetime
from compile import compile

if __name__ == "__main__":
    print("=" * 60)
    print("COMPILING DOW 30 DATASETS")
    print("=" * 60)

    # Load the Dow 30 tickers
    dow30_path = Path("data/raw/dow30_tickers.json")
    with open(dow30_path, "r") as f:
        tickers = json.load(f)

    print(f"\nTotal tickers: {len(tickers)}")

    # Define date ranges for train/dev/test splits
    train_dates = (datetime(2010, 1, 1), datetime(2021, 1, 1))
    dev_dates = (datetime(2022, 1, 1), datetime(2023, 7, 1))
    test_dates = (datetime(2024, 7, 1), datetime(2025, 7, 1))

    print(f"\nDate ranges:")
    print(f"  Train: {train_dates[0].date()} to {train_dates[1].date()}")
    print(f"  Dev:   {dev_dates[0].date()} to {dev_dates[1].date()}")
    print(f"  Test:  {test_dates[0].date()} to {test_dates[1].date()}")

    # Compile the datasets
    compile(
        tickers=tickers,
        dataset_name="dow30",
        train_dates=train_dates,
        dev_dates=dev_dates,
        test_dates=test_dates,
        regimes=3,
        length=60,
        model_dir="models/checkpoints",
        output_dir="data/processed",
    )

    print("\n" + "=" * 60)
    print("COMPILATION COMPLETED")
    print("=" * 60)
