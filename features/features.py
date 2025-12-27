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
        # Join equity and market features on common dates
        windows, targets, dates = self._join_features(equity, market)
        if not windows:
            self.x = torch.tensor([], dtype=torch.float32).reshape(0, 0, 0)
        else:
            self.x = torch.tensor(np.array(windows), dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.dates = np.array(dates)

    def mask_dates(self, start_date, end_date):
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

    def _join_features(
        self, equity: EquityFeatures, market: MarketFeatures
    ) -> tuple[list, list, list]:
        """
        Join equity and market features on common dates.
        :param equity: EquityFeatures object with equity features and targets
        :param market: MarketFeatures object with market features
        :return: Tuple of (windows, targets, dates) of combined features
        """
        equity_dates = set(equity.dates)
        market_dates = set(market.dates)
        index_dates = equity_dates & market_dates
        index_dates = sorted(index_dates)  # Keep chronological order

        # Combine features and targets for common dates
        windows, targets, dates = [], [], []
        for date in index_dates:
            equity_result = equity[date]
            market_window = market[date]
            if equity_result is None or market_window is None:
                continue
            equity_window, target = equity_result
            combined = torch.cat([equity_window, market_window], dim=-1)
            windows.append(combined)
            targets.append(target)
            dates.append(date)

        return windows, targets, dates

    def __len__(self) -> int:
        """
        Get the number of feature windows.
        :return: Number of feature windows
        """
        return len(self.x)

    def __getitem__(self, index: int | datetime) -> tuple[torch.Tensor, float] | torch.Tensor | None:
        """
        Index the feature data by integer position or by date.
        :param index: Integer position or datetime to get features for
        :return: For int: (x, y) tuple. For datetime: x tensor or None if unavailable
        """
        if isinstance(index, int):
            # Integer indexing: return (x, y) tuple
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range for length {len(self)}")
            return self.x[index], self.y[index].item()
        elif isinstance(index, datetime):
            # Datetime indexing: return x tensor
            if len(self) == 0 or index not in self.dates:
                return None
            date_index = np.where(self.dates == index)[0][0]
            return self.x[date_index]
        else:
            raise TypeError(f"Index must be int or datetime, not {type(index).__name__}")
    