__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from datetime import datetime, timedelta
from torch import Tensor

from features.equities.data import EquityData


class EquityFeatures:
    """
    Class to hold computed feature tensors for a single equity (no market data).
    """

    def __init__(self, data: EquityData, length: int = 60):
        """
        Initialize equity features with rolling windows and targets.
        :param data: EquityData object containing equity data
        :param length: Length of feature windows to aggregate, default 60
        """
        self.TICKER = data.TICKER

        # Roll data into windows and associated targets
        windows, targets, dates = self._roll_data(data, length)
        if not windows:
            self.x = torch.tensor([], dtype=torch.float32).reshape(0, 0, 0)
        else:
            self.x = torch.tensor(np.array(windows), dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.dates = np.array(dates)

    def _roll_data(self, data: EquityData, length: int) -> tuple[list, list, list]:
        """
        Roll data into windows and associated targets of given length.
        :param data: EquityData object containing equity data
        :param length: Length of windows to create
        :return: Tuple of (windows, targets, dates) of data
        """
        windows, targets, dates = [], [], []
        index = data.index
        if len(index) < length:
            return windows, targets, dates

        # Feature windows collect a rolling data length for each date
        date1, date2 = index[length], index[-1] + timedelta(days=1)
        index_dates = index[(date1 <= index) & (index < date2)]
        for date in index_dates:
            date_data = data.data[data.index <= date]
            window = date_data.tail(length).values
            target = data.get_return(date)
            if len(window) != length or date not in index:
                continue
            windows.append(window.astype(float))
            targets.append(target)
            dates.append(date)

        return windows, targets, dates

    def __getitem__(self, date: datetime) -> tuple[Tensor, float] | None:
        """
        Index the feature window at a specific date.
        :param date: Date to get features for
        :return: Tuple of feature tensor (length, num_features) and target
        """
        if len(self) == 0 or date not in self.dates:
            return None
        index = np.where(self.dates == date)[0][0]
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        """
        Get the number of feature windows.
        :return: Number of feature windows
        """
        return len(self.x)
