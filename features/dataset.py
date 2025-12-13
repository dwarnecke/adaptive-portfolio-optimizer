__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pickle
import torch
from datetime import datetime
from pathlib import Path

from data.loader import load_fundamentals
from features.equities.equity import EquityData
from features.features import FeaturesData
from features.markets.market import MarketData
from features.markets.observations import ObservationsData


class FeaturesDataset:
    """
    Class to hold calculated features for a universe of equities.
    """

    def __init__(
        self,
        path: str,
        tickers: list[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ):
        """
        Initialize the features dataset for a given FeaturesData object.
        :param path: Path to regime model file
        :param tickers: Ticker symbols in the dataset, default None
        :param start_date: Start datetime for data retrieval, default None
        :param end_date: End datetime for data retrieval, default None
        """
        print("Loading features dataset for universe...")

        # Load fundamental data to determine available tickers
        fund_data = load_fundamentals()
        available_tickers = set(fund_data["shares"]["Ticker"].unique())
        if tickers is None:
            tickers = sorted([str(tick) for tick in available_tickers])
        else:
            tickers = sorted([tick for tick in tickers if tick in available_tickers])

        self._tickers = {i: ticker for i, ticker in enumerate(tickers)}
        self._data = {}
        self._lengths = {}

        # Generate feature data using the universe of tickers
        print(f"Generating market data from {start_date} to {end_date}...")
        market_data = MarketData(ObservationsData(start_date, end_date), path)
        print(f"Generating feature data for {len(tickers)} tickers...")
        index = 0
        for ticker in self._tickers.values():
            print(f"  Processing {ticker}...")
            data = EquityData(ticker, fund_data)
            if data.empty:
                print(f"    [INFO] No data available for {ticker}, skipping...")
                continue
            data = FeaturesData(data, market_data)
            self._data[index] = data
            self._lengths[index] = len(data)
            index += 1

    def save(self, directory: str = "features/data", filename: str = "dataset.pkl"):
        """
        Save the preprocessed dataset to disk as a single compressed file.
        :param directory: Directory to save preprocessed data
        :param filename: Name of the output file (default: dataset.pkl)
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename
        print(f"Saving preprocessed dataset to {filepath.absolute()}...")

        # Collect all ticker data into a single dictionary
        data_dict = {}
        total_samples = 0
        for index, data in self._data.items():
            ticker = self._tickers[index]
            data_dict[ticker] = {
                "x": data.x,
                "y": data.y,
                "dates": data.dates,
                "length": self._lengths[index],
            }
            total_samples += self._lengths[index]
            print(f"  Packed {ticker}: {self._lengths[index]} samples")

        # Create the complete dataset structure
        dataset_bundle = {
            "version": "1.0",
            "tickers": list(self._tickers.values()),
            "ticker_to_index": self._tickers,
            "lengths": self._lengths,
            "total_samples": total_samples,
            "data": data_dict,
        }

        # Save as single compressed pickle file
        with open(filepath, "wb") as f:
            pickle.dump(dataset_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        num_tickers = len(self._tickers)
        print(f"\nSaved {num_tickers} tickers, {total_samples} total samples")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Location: {filepath}")

    @classmethod
    def load(cls, directory: str = "features/data", filename: str = "dataset.pkl"):
        """
        Load a preprocessed dataset from a single file.
        :param directory: Directory containing preprocessed data
        :param filename: Name of the dataset file (default: dataset.pkl)
        :return: FeaturesDataset instance with loaded data
        """
        load_dir = Path(directory)
        filepath = load_dir / filename
        print(f"Loading preprocessed dataset from {filepath.absolute()}...")

        # Load the complete bundle
        with open(filepath, "rb") as f:
            dataset_bundle = pickle.load(f)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance._tickers = dataset_bundle["ticker_to_index"]
        instance._lengths = dataset_bundle["lengths"]
        instance._data = {}

        # Unpack ticker data
        ticker_data_dict = dataset_bundle["data"]
        for index, ticker in instance._tickers.items():
            ticker_data = ticker_data_dict[ticker]
            # Reconstruct FeaturesData-like object
            data_obj = FeaturesData.__new__(FeaturesData)
            data_obj.x = ticker_data["x"]
            data_obj.y = ticker_data["y"]
            data_obj.dates = ticker_data["dates"]
            instance._data[index] = data_obj
            print(f"  Loaded {ticker}: {ticker_data['length']} samples")

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        num_tickers = len(instance._tickers)
        num_samples = len(instance)
        print(f"\nLoaded {num_tickers} tickers, {num_samples} total samples")
        print(f"File size: {file_size_mb:.2f} MB")
        return instance

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        """
        Get the feature tensor and target for the given index.
        :param index: Index of the desired feature sample
        :return: Tuple of (feature tensor, target value)
        """
        total = 0
        for ticker_index, length in self._lengths.items():
            if total <= index < total + length:
                local_index = index - total
                data = self._data[ticker_index]
                x, y = data[local_index]
                return x, y

    def __len__(self) -> int:
        """
        Get the number of feature samples in the dataset.
        :return: Number of feature samples
        """
        return sum(self._lengths.values())
