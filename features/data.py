__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from datetime import datetime
from pandas import DataFrame
from torch import Tensor

from features.equities.data import EquityData
from features.markets.data import MarketData


class FeaturesData:
    """
    Class to aggregate data and create windowed views.
    """

    def __init__(
        self,
        equity_data: EquityData,
        market_data: MarketData,
        length: int,
    ):
        """
        Initialize combined features with raw data and windowed views.
        :param equity_data: EquityData object with raw equity data
        :param market_data: MarketData object with raw market data
        :param length: Length of rolling window for features
        """
        self.equity_data = equity_data
        self.market_data = market_data

        # Join equity and market data into single DataFrame
        equity_frame = equity_data.data
        market_frame = market_data.data
        self.data = equity_frame.join(market_frame, how="inner")
        targets = equity_data.targets
        self.targets = targets.loc[self.data.index]

        # Aggregate rolled windows from joined data for transformer sequences
        windows, targets, self.dates = self.get_windows(length)
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)

        if len(windows) == 0:
            self.windows.reshape(0, 0, 0)
            self.targets.reshape(0, 0)

    def get_subset(self, start_date: datetime, end_date: datetime) -> "FeaturesData":
        """
        Subset data to only that within a date range.
        :param start_date: Start date to mask, inclusive
        :param end_date: End date to mask, exclusive
        :return: Filtered FeaturesData object
        """
        if len(self) == 0:
            return self
        
        # Convert dates to numpy array for comparison
        dates_array = np.array(self.dates)
        mask = (start_date <= dates_array) & (dates_array < end_date)
        
        filtered = FeaturesData.__new__(FeaturesData)

        filtered.equity_data = self.equity_data
        filtered.market_data = self.market_data
        filtered.data = self.data.loc[dates_array[mask]]
        filtered.windows = self.windows[mask]
        filtered.targets = self.targets[mask]
        filtered.dates = dates_array[mask].tolist()
        return filtered

    def get_windows(self, length: int) -> tuple[list, list, list]:
        """
        Window the joined data into windows of specified length.
        :param length: Length of rolling window
        :return: Tuple of (windows, targets, dates) tensor data
        """
        # Convert dataframe to numpy for faster rolling
        data_values = self.data.values
        target_values = self.targets.values
        data_dates = self.data.index

        windows = []
        targets = []
        dates = []
        for i in range(length, len(data_values)):
            if np.any(np.isnan(target_values[i])):
                continue
            window = data_values[i - length : i]
            if np.any(np.isnan(window)):
                continue
            windows.append(window)
            targets.append(target_values[i])
            dates.append(data_dates[i])
        return windows, targets, dates

    def get_data(self, field: str, date: datetime) -> float:
        """
        Get a data field value for a specific date.
        :param field: Column name to retrieve
        :param date: Date to retrieve data for
        :return: Value of the field on the given date
        """
        return self.data.loc[date, field]

    def get_targets(self, date: datetime) -> tuple[float, float]:
        """
        Get the forward return and volatility targets for a given date.
        :param date: Date to get the targets for
        :return: Tuple of (mu, sigma) on the given date
        """
        data = self.equity_data.targets.loc[date]
        return data["mu"], data["sigma"]

    def get_series(self, field: str):
        """
        Get the full time series for a given field from the joined features DataFrame.
        :param field: Column name to retrieve (equity, market, or regime features)
        :return: Pandas Series for the requested field aligned to the joined index
        """
        return self.data[field]

    @property
    def ticker(self) -> str:
        """
        Get the ticker symbol for this equity.
        :return: Ticker symbol
        """
        return self.equity_data.TICKER

    def to_dict(self) -> dict:
        """
        Serialize the FeaturesData object to a dictionary for saving.
        :return: Dictionary containing all necessary data for reconstruction
        """
        return {
            "ticker": self.equity_data.TICKER,
            "windows": self.windows,
            "targets": self.targets,
            "dates": self.dates,
            "data": self.data,
            "equity_data": self.equity_data,
            "market_data": self.market_data,
        }

    @classmethod
    def from_dict(cls, data_dict: dict) -> "FeaturesData":
        """
        Reconstruct a FeaturesData object from a dictionary.
        :param data_dict: Dictionary containing serialized FeaturesData
        :return: Reconstructed FeaturesData instance
        """
        instance = cls.__new__(cls)
        instance.windows = data_dict["windows"]
        instance.targets = data_dict["targets"]
        instance.dates = data_dict["dates"]
        instance.data = data_dict["data"]
        instance.equity_data = data_dict["equity_data"]
        instance.market_data = data_dict["market_data"]
        return instance

    def __getitem__(self, index: int | datetime) -> tuple[Tensor, Tensor]:
        """
        Index the feature data by integer position.
        :param index: Integer position to get features for
        :return: Tuple (x, y) where y is [return, volatility]
        """
        if isinstance(index, datetime):
            return self.__getitem__(self.dates.index(index))
        return self.windows[index], self.targets[index]

    def __len__(self) -> int:
        """
        Get the number of feature windows.
        :return: Number of feature windows
        """
        return len(self.windows)
