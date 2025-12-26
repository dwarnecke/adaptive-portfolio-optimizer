__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from datetime import datetime, timedelta

from features.markets.data import MarketData


class MarketFeatures:
    """
    Class to hold computed feature tensors for market data.
    """

    def __init__(self, market: MarketData, length: int = 60):
        """
        Initialize market features with rolling windows.
        :param market: MarketData object containing market data
        :param length: Length of feature windows to aggregate, default 60
        """
        # Roll data into windows and associated dates
        windows, dates = self._roll(market, length)
        if not windows:
            self.x = torch.tensor([], dtype=torch.float32).reshape(0, 0, 0)
        else:
            self.x = torch.tensor(np.array(windows), dtype=torch.float32)
        self.dates = np.array(dates)

    def _roll(self, data: MarketData, length: int) -> tuple[list, list]:
        """
        Roll data into windows of given length.
        :param data: MarketData object containing market data
        :param length: Length of windows to create
        :return: Tuple of (windows, dates) of data
        """
        index = data.index
        date1, date2 = index[length], index[-1] + timedelta(days=1)
        index_dates = index[(date1 <= index) & (index < date2)]

        # Feature windows collect a rolling length of data for each date
        windows, dates = [], []
        for date in index_dates:
            date_data = data.data[data.index <= date]
            window = date_data.tail(length).values

            # Do not use incomplete windows as features
            if len(window) != length:
                continue
            windows.append(window.astype(float))
            dates.append(date)

        return windows, dates

    def __getitem__(self, date: datetime) -> torch.Tensor | None:
        """
        Index the feature window at a specific date.
        :param date: Date to get features for
        :return: Feature tensor of shape (length, num_features) or None if unavailable
        """
        if len(self) == 0 or date not in self.dates:
            return None
        index = np.where(self.dates == date)[0][0]
        return self.x[index]

    def __len__(self) -> int:
        """
        Get the number of feature windows.
        :return: Number of feature windows
        """
        return len(self.x)
