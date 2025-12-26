__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from datetime import datetime

from features.equities.features import EquityFeatures
from features.markets.features import MarketFeatures


class FeaturesData:
    """
    Class to combine equity and market features with targets for training.
    """

    def __init__(
        self,
        equity: EquityFeatures,
        market: MarketFeatures,
    ):
        """
        Initialize combined features data with targets.
        :param equity: EquityFeatures object with equity features and targets
        :param market: MarketFeatures object with market features
        """
        # Find dates that exist in both feature sets
        equity_dates = set(equity.dates)
        market_dates = set(market.dates)
        index_dates = equity_dates & market_dates
        index_dates = sorted(index_dates)  # Keep chronological order

        # Build tensors only for common dates
        windows, targets, dates = [], [], []
        for date in index_dates:
            equity_window = equity[date]
            market_window = market[date]
            if equity_window is None or market_window is None:
                continue
            combined = torch.cat([equity_window, market_window], dim=-1)
            index = np.where(equity.dates == date)[0][0]
            target = equity.y[index]
            windows.append(combined)
            targets.append(target)
            dates.append(date)

        if not windows:
            self.x = torch.tensor([], dtype=torch.float32).reshape(0, 0, 0)
            self.y = torch.tensor([], dtype=torch.float32)
            self.dates = np.array([])

        self.x = torch.stack(windows)
        self.y = torch.stack(targets)
        self.dates = np.array(dates)

    def mask(self, start_date, end_date):
        """
        Mask features to only include data within a date range.
        :param start_date: Start date to mask, inclusive
        :param end_date: End date to mask, exclusive
        :return: Filtered FeaturesData object
        """
        if len(self) == 0:
            return self
        mask = (start_date <= self.dates) & (self.dates < end_date)
        filtered = FeaturesData.__new__(FeaturesData)
        filtered.x = self.x[mask]
        filtered.y = self.y[mask]
        filtered.dates = self.dates[mask]
        return filtered

    def __len__(self) -> int:
        """
        Get the number of feature windows.
        :return: Number of feature windows
        """
        return len(self.x)

    def __getitem__(self, date: datetime) -> torch.Tensor | None:
        """
        Index the feature window for prediction at a specific date.
        :param date: Date to get features for
        :return: Feature tensor of shape (length, features) or None if unavailable
        """
        if len(self) == 0 or date not in self.dates:
            return None
        index = np.where(self.dates == date)[0][0]
        return self.x[index]
