__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from datetime import timedelta

from features.equities.equity import EquityData
from features.markets.market import MarketData


class FeaturesData:
    """
    Class to hold calculated features for a particular equity.
    """

    def __init__(self, equity: EquityData, market: MarketData, length: int = 60):
        """
        Initialize the features data for a given equity and market.
        :param equity: EquityData object containing equity data
        :param market: MarketData object containing market data
        :param length: Length of feature windows to aggregate, default 60
        """
        # Only use dates that have both features and targets
        data = equity.data.join(market.data, how="inner")
        dates = data.index.intersection(equity.targets.index)
        data = data.loc[dates]
        index = data.index

        # Feature windows collect a rolling length of data for each date
        windows, returns = [], []
        if len(index) <= length:
            # Not enough data, create empty tensors
            x = torch.tensor([], dtype=torch.float32)
            self.x = x.reshape(0, length, data.shape[1])
            self.y = torch.tensor([], dtype=torch.float32)
            return

        # Calculate windows and corresponding targets
        date1, date2 = index[length], index[-1] + timedelta(days=1)
        dates = index[(date1 <= index) & (index < date2)]
        for date in dates:
            if date in equity.targets.index:
                date_data = data[data.index <= date]
                window = date_data.tail(length).values
                target = equity.targets.loc[date]
                # Do not use incomplete windows as features
                if len(window) != length:
                    continue
                windows.append(window.astype(float))
                returns.append(target)

        # Create empty tensors if no windows were created
        if len(windows) == 0:
            x = torch.tensor([], dtype=torch.float32)
            self.x = x.reshape(0, length, data.shape[1])
            self.y = torch.tensor([], dtype=torch.float32)
            return

        self.x = torch.tensor(np.array(windows), dtype=torch.float32)
        self.y = torch.tensor(np.array(returns), dtype=torch.float32)
        self.dates = dates

    def __len__(self) -> int:
        """
        Get the number of feature windows.
        :return: Number of feature windows
        """
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        """
        Get the feature window tensor and target at the given index.
        :param index: Index of the desired feature window
        :return: Tuple of (feature window tensor, target value)
        """
        return (self.x[index], self.y[index])
