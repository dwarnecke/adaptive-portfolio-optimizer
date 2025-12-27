__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime

from models.universe import Universe
from features.dataset import FeaturesDataset
from features.markets.observations import ObservationsData
from features.markets.regime_model import RegimeModel


def compile(
    tickers: list[str],
    dataset_name: str,
    train_dates: tuple[datetime, datetime],
    dev_dates: tuple[datetime, datetime],
    test_dates: tuple[datetime, datetime],
    regimes: int = 3,
    length: int = 60,
    model_dir: str = "models/checkpoints",
    output_dir: str = "data/processed",
):
    """
    Compile and save feature datasets from tickers for train/dev/test splits.
    :param tickers: List of ticker symbols to include in the universe
    :param dataset_name: Name of the dataset files to save
    :param train_dates: Tuple of training set date range (inclusive, exclusive)
    :param dev_dates: Tuple of development set date range (inclusive, exclusive)
    :param test_dates: Tuple of test set date range (inclusive, exclusive)
    :param regimes: Number of market regimes for HMM, default 3
    :param length: Length of feature windows, default 60
    :param model_dir: Directory to save regime model, default "models/checkpoints"
    :param output_dir: Directory to save dataset files, default "data/processed"
    """
    train_start, train_end = train_dates
    dev_start, dev_end = dev_dates
    test_start, test_end = test_dates

    # Calculate overall date range
    min_date = min(train_start, dev_start, test_start)
    max_date = max(train_end, dev_end, test_end)

    # Train regime model for the market features
    print(f"Training regime model on {train_start.date()} to {train_end.date()}...")
    observations = ObservationsData(train_start, train_end)
    regime_model = RegimeModel(states=regimes, data=observations.inputs)
    regime_model.train(observations.inputs)
    regime_path = model_dir + "/regime_model.pkl"
    regime_model.save(regime_path)
    print(f"Regime model saved to {regime_path}")

    # Compile datasets for performance splits
    print(f"\nCompiling datasets for {len(tickers)} tickers...")
    universe = Universe(tickers=tickers, length=length)
    dataset = FeaturesDataset(universe, str(regime_path), min_date, max_date)
    dataset.save(train_start, train_end, output_dir, f"{dataset_name}_train.pkl")
    dataset.save(dev_start, dev_end, output_dir, f"{dataset_name}_dev.pkl")
    dataset.save(test_start, test_end, output_dir, f"{dataset_name}_test.pkl")
    print(f"Datasets saved to {output_dir}")
