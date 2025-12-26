__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json
from datetime import datetime

from models.universe import Universe
from features.dataset import FeaturesDataset


def compile(
    tickers: list[str],
    train_range: tuple[datetime, datetime],
    dev_range: tuple[datetime, datetime],
    test_range: tuple[datetime, datetime],
    length: int = 60,
    regime_path: str = "models/checkpoints/regime_model.pkl",
    output_dir: str = "data/processed",
):
    """
    Compile and save feature datasets from tickers for train/dev/test splits.
    :param tickers: List of ticker symbols to include in the universe
    :param train_range: Tuple of training set date range (start_inclusive, end_exclusive)
    :param dev_range: Tuple of development set date range (start_inclusive, end_exclusive)
    :param test_range: Tuple of test set date range (start_inclusive, end_exclusive)
    :param length: Length of feature windows, default 60
    :param regime_path: Path to regime model file
    :param output_dir: Directory to save dataset files
    """
    print(f"Compiling datasets for {len(tickers)} tickers...")

    universe = Universe(tickers=tickers, length=length)
    dataset = FeaturesDataset(universe=universe, path=regime_path)

    dataset.save(
        start_date=train_range[0],
        end_date=train_range[1],
        directory=output_dir,
        filename="train_dataset.pkl",
    )

    dataset.save(
        start_date=dev_range[0],
        end_date=dev_range[1],
        directory=output_dir,
        filename="val_dataset.pkl",
    )

    dataset.save(
        start_date=test_range[0],
        end_date=test_range[1],
        directory=output_dir,
        filename="test_dataset.pkl",
    )

    print(f"Datasets saved to {output_dir}/")


if __name__ == "__main__":
    
    # Example usage with S&P 500 tickers
    with open("data/raw/sp500_tickers.json", "r") as f:
        sp500_tickers = json.load(f)
    
    compile(
        tickers=sp500_tickers,
        train_range=(datetime(2010, 1, 1), datetime(2020, 1, 1)),
        dev_range=(datetime(2021, 1, 1), datetime(2023, 1, 1)),
        test_range=(datetime(2024, 1, 1), datetime.now()),
    )
