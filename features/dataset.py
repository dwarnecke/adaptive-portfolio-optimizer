__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pickle
import torch
from datetime import datetime
from pathlib import Path

from features.features import FeaturesData
from features.markets.data import MarketData
from features.markets.features import MarketFeatures
from features.markets.observations import ObservationsData
from models.universe import Universe


class FeaturesDataset:
    """
    Class to hold calculated features for a universe of equities.
    """

    def __init__(
        self,
        universe: Universe,
        path: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Initialize the features dataset for a given FeaturesData object.
        :param universe: Universe object containing equity data
        :param path: Path to regime model file
        :param start_date: Start date for dataset features
        :param end_date: End date for dataset features
        """
        print("\nLoading features dataset for universe...")
        self.tickers = {}
        self.data = {}
        self.lengths = {}

        print(f"Generating index data from {start_date.date()} to {end_date.date()}...")
        market_data = MarketData(ObservationsData(start_date, end_date), path)
        market_features = MarketFeatures(market_data)

        # Generate feature data using the universe of tickers
        print(f"Generating feature data for {len(universe.data)} tickers...")
        index = 0
        for equity_features in universe.features.values():
            data = FeaturesData(equity_features, market_features)
            if len(data) == 0:
                continue
            self.tickers[index] = equity_features.TICKER
            self.data[index] = data
            self.lengths[index] = len(data)
            index += 1

    def save(
        self, start_date: datetime, end_date: datetime, directory: str, filename: str
    ):
        """
        Save the preprocessed dataset to disk as a single compressed file.
        :param start_date: Start date for masking data, inclusive
        :param end_date: End date for masking data, exclusive
        :param directory: Directory to save the dataset file
        :param filename: Name of the dataset file to save
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename
        print(f"\nSaving dataset to {filepath.absolute()}...")

        # Collect all ticker data into a single dictionary
        data_dict, tickers, lengths = {}, {}, {}
        total_samples = 0
        new_index = 0
        for index, data in self.data.items():
            data = data.mask_dates(start_date, end_date)
            if len(data) == 0:
                continue
            data_dict[new_index] = {"x": data.x, "y": data.y, "dates": data.dates}
            tickers[new_index] = self.tickers[index]
            lengths[new_index] = len(data)
            total_samples += len(data)
            new_index += 1
            print(f"  Packed {self.tickers[index]}: {len(data)} samples")

        # Save the complete dataset structure into a file
        dataset_bundle = {"tickers": tickers, "lengths": lengths, "data": data_dict}
        with open(filepath, "wb") as f:
            pickle.dump(dataset_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        num_tickers = len(data_dict)
        print(f"\nSaved {num_tickers} tickers, {total_samples} total samples")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Location: {filepath}")

    @classmethod
    def load(cls, directory: str = "data/processed", filename: str = "dataset.pkl"):
        """
        Load a preprocessed dataset from a single file.
        :param directory: Directory containing preprocessed data
        :param filename: Name of the dataset file (default: dataset.pkl)
        :return: FeaturesDataset instance with loaded data
        """
        print(f"\nLoading dataset from {directory}/{filename}...")
        load_dir = Path(directory)
        filepath = load_dir / filename
        with open(filepath, "rb") as f:
            dataset_bundle = pickle.load(f)

        # Initialize instance without calling __init__
        instance = cls.__new__(cls)
        instance.tickers = dataset_bundle["tickers"]
        instance.lengths = dataset_bundle["lengths"]
        instance.data = {}

        # Unpack data for each ticker
        data_dict = dataset_bundle["data"]
        for index in instance.tickers.keys():
            data = data_dict[index]
            data_obj = FeaturesData.__new__(FeaturesData)
            data_obj.x = data["x"]
            data_obj.y = data["y"]
            data_obj.dates = data["dates"]
            instance.data[index] = data_obj

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        num_tickers = len(instance.tickers)
        num_samples = len(instance)
        print(f"Loaded {num_tickers} tickers, {num_samples} total samples")
        print(f"File size: {file_size_mb:.2f} MB")
        
        return instance

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        """
        Get the feature tensor and target for the given index.
        :param index: Index of the desired feature sample
        :return: Tuple of (feature tensor, target value)
        """
        total = 0
        for ticker_index, length in self.lengths.items():
            if total <= index < total + length:
                local_index = index - total
                data = self.data[ticker_index]
                return data[local_index]
            total += length
        raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

    def __len__(self) -> int:
        """
        Get the number of feature samples in the dataset.
        :return: Number of feature samples
        """
        return sum(self.lengths.values())
