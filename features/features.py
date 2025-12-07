__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
from datetime import datetime

from features.equities.equity import EquityData
from features.markets.market import MarketData


class FeaturesData:
    """
    Class to hold calculated features for a particular equity.
    """

    def __init__(self, equity: EquityData, market: MarketData):
        """
        Initialize the features data for a given equity and market.
        :param equity: EquityData object containing equity data
        :param market: MarketData object containing market data
        """
        self._equity = equity
        self._market = market
        self._features = equity.features.join(market.features, how="inner")

    def slice_windows(
        self, start_date: datetime, end_date: datetime, length: int = 60
    ) -> torch.Tensor:
        """
        Slice windows of given length from the features for the given date range.
        :param start_date: Start date for the window, inclusive
        :param end_date: End date for the window, exclusive
        :param length: Number of most recent rows to return, default 60
        :return: Torch tensor of shape (dates, length, features)
        """
        windows = []
        index = self._features.index
        dates = index[(start_date <= index) & (index < end_date)]

        # Slice windows for each date in the range
        for date in dates:
            date_data = self._features[self._features.index <= date]
            window = date_data.tail(length).values
            windows.append(window)

        # Convert list of arrays to tensor with shape (dates, length, features)
        return torch.tensor(windows, dtype=torch.float32)
