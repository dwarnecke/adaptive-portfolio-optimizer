__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pickle
import json
import torch
from datetime import datetime
from pathlib import Path

from features.data import FeaturesData
from features.markets.data import MarketData
from features.markets.observations import ObservationsData
from features.equities.data import EquityData
from data.loader import load_fundamentals


class FeaturesDataset:
    """
    Class to hold calculated features for a universe of equities.
    """

    def __init__(
        self,
        tickers: list[str],
        filepath: Path | str,
        start_date: datetime,
        end_date: datetime,
        length: int = 120,
    ):
        """
        Initialize the features dataset for a list of tickers.
        :param tickers: List of ticker symbols to include
        :param filepath: Path to regime model file
        :param start_date: Start date for dataset features, inclusive
        :param end_date: End date for dataset features, exclusive
        :param length: Window length for rolling features
        """
        print(f"\nLoading dataset for {start_date.date()} to {end_date.date()}...")
        self._model_path = filepath
        self._path = None
        fund_data = load_fundamentals()

        print(f"Generating market data from {start_date.date()}...")
        observations_data = ObservationsData(start_date, end_date)
        market_data = MarketData(observations_data, filepath)

        # Skip saving any features for tickers without data
        self.data = {}
        index = 0
        print(f"Processing {len(tickers)} tickers for features...")
        for ticker in tickers:
            print(f"Processing ticker {ticker}...", end="\r")
            equity_data = EquityData(ticker, fund_data, max_date=end_date)
            features_data = FeaturesData(equity_data, market_data, length)
            if len(features_data) == 0:
                continue
            self.data[index] = features_data
            index += 1
        print(f"\nLoaded {index} tickers with features")

    def _to_dict(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Convert dataset to dictionary with date filtering.
        :param start_date: Start date for masking data, inclusive
        :param end_date: End date for masking data, exclusive
        :return: Dictionary of feature data with new indices
        """
        data_dict = {}
        new_index = 0
        for data in self.data.values():
            # Mask the data to the requested date range
            data = data.get_subset(start_date, end_date)
            if len(data) == 0:
                continue
            data_dict[new_index] = data.to_dict()
            new_index += 1
        return data_dict

    def save(
        self,
        splits: dict[str, tuple[datetime, datetime]],
        name: str,
        directory: Path | str = "outputs/datasets",
    ) -> Path:
        """
        Save the preprocessed dataset splits in a single timestamped directory.
        :param splits: Dictionary mapping split names to (start_date, end_date) tuples
        :param name: Base name for the dataset files
        :param directory: Base directory for datasets, default "outputs/datasets"
        :returns: Path to the timestamped directory
        """
        # Timestamp directory to prevent overwriting existing datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(directory) / f"dataset_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving dataset to {save_dir.absolute()}...")

        for split_name, (start_date, end_date) in splits.items():
            filename = f"{name}_{split_name}.pkl"
            filepath = save_dir / filename
            print(f"\nProcessing {split_name} split...")

            # Save the split dataset to the timestamp directory
            data_dict = self._to_dict(start_date, end_date)
            self._path = filepath
            data_bundle = {"data": data_dict, "path": str(filepath)}
            with open(filepath, "wb") as f:
                pickle.dump(data_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save manifest for this split for metadata
            manifest_path = save_dir / f"{name}_{split_name}_manifest.json"
            manifest = self._to_manifest(filename, start_date, end_date, data_dict)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Manifest: {manifest_path}")

            file_size = filepath.stat().st_size / (1024 * 1024)
            num_ticks = len(data_dict)
            num_samps = sum(len(dict["dates"]) for dict in data_dict.values())
            print(f"Saved {num_ticks} tickers, {num_samps} samples, {file_size:.2f} MB")

        return save_dir

    def _to_manifest(
        self,
        filename: str,
        start_date: datetime,
        end_date: datetime,
        data_dict: dict,
    ) -> dict:
        """
        Create manifest dictionary with dataset metadata.
        :param filename: Name of the dataset file
        :param start_date: Start date of the dataset
        :param end_date: End date of the dataset
        :param data_dict: Dictionary of saved feature data
        :return: Manifest dictionary
        """
        tickers = [data_dict[idx]["ticker"] for idx in sorted(data_dict.keys())]
        num_samples = sum(len(data_dict[idx]["dates"]) for idx in data_dict.keys())
        return {
            "dataset_file": filename,
            "created_at": datetime.now().isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "num_tickers": len(data_dict),
            "num_samples": num_samples,
            "regime_model": str(self._model_path) if self._model_path else None,
            "tickers": tickers,
        }

    @classmethod
    def load(
        cls, directory: Path | str = "outputs/datasets", filename: str = "dataset.pkl"
    ):
        """
        Load a preprocessed dataset from a single file.
        :param directory: Directory containing preprocessed data
        :param filename: Name of the dataset file, default "dataset.pkl"
        :return: FeaturesDataset instance with loaded data
        """
        filepath = Path(directory) / filename
        print(f"\nLoading dataset from {filepath}...")
        with open(filepath, "rb") as f:
            dataset_bundle = pickle.load(f)

        # Initialize instance without calling __init__ to avoid reprocessing
        instance = cls.__new__(cls)
        instance.data = {
            index: FeaturesData.from_dict(data_dict)
            for index, data_dict in dataset_bundle["data"].items()
        }
        instance._path = Path(dataset_bundle.get("path", filepath))
        instance._model_path = None  # Not needed when loading

        file_size = filepath.stat().st_size / (1024 * 1024)
        num_ticks = len(instance.data)
        num_samps = sum(len(data.dates) for data in instance.data.values())
        print(f"Loaded {num_ticks} tickers, {num_samps} samples, {file_size:.2f} MB")
        return instance

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        """
        Get the feature tensor and target for the given index.
        :param index: Index of the desired feature sample
        :return: Tuple of (feature tensor, target value)
        """
        total = 0
        for data in self.data.values():
            length = len(data)
            if total <= index < total + length:
                local_index = index - total
                return data[local_index]
            total += length
        raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

    def __len__(self) -> int:
        """
        Get the number of feature samples in the dataset.
        :return: Number of feature samples
        """
        return sum(len(data) for data in self.data.values())

    @property
    def index(self) -> list[int]:
        """
        Get the list of indices for the dataset.
        :return: List of indices
        """
        return list(self.data.keys())

    @property
    def path(self) -> Path | None:
        """
        Get the path where this dataset was saved.
        :return: Path to the dataset file, or None if not yet saved
        """
        return self._path
