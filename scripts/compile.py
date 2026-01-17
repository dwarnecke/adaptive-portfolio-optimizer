__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime
from pathlib import Path

from config.hyperparameters import HYPERPARAMETERS
from config.paths import DATASETS_DIR
from features.dataset import FeaturesDataset
from features.markets.observations import ObservationsData
from features.markets.regime.model import RegimeModel


def compile(
    tickers: list[str],
    dataset_name: str,
    splits: dict[str, tuple[datetime, datetime]],
    parameters: dict = HYPERPARAMETERS["features"],
    directory: Path | str = DATASETS_DIR,
):
    """
    Compile and save feature datasets from tickers for multiple splits.
    :param tickers: List of ticker symbols to include in the universe
    :param dataset_name: Name of the dataset files to save
    :param splits: Dictionary mapping split names to (start_date, end_date) tuples
    :param parameters: Dictionary with hyperparameters, default config
    :param directory: Directory to save dataset files, default config
    """
    # Determine min/max dates across all splits
    all_starts = [start for start, _ in splits.values()]
    all_ends = [end for _, end in splits.values()]
    min_date = min(all_starts)
    max_date = max(all_ends)
    train_start, train_end = splits["train"]

    # Train regime model for the market features
    print(f"Training regime model on {train_start.date()} to {train_end.date()}...")
    observations = ObservationsData(train_start, train_end)
    regime_model = RegimeModel(observations.data, parameters=parameters)
    regimes_dir = regime_model.save()

    # Load datasets for performance splits
    print(f"\nCompiling datasets for {len(tickers)} tickers...")
    regime_path = regimes_dir / "regime_model.pkl"
    length = parameters["features"]["length"]
    dataset = FeaturesDataset(tickers, regime_path, min_date, max_date, length)
    dataset.save(splits, dataset_name, directory)