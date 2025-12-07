__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
from datetime import datetime

from markets.features.vix_term import calc_term_structures
from markets.features.index_momentum import calc_log_returns_20d
from markets.features.index_volatility import calc_log_return_stds_20d
from markets.features.treasury_slope import calc_2y10y_slopes
from markets.targets.forward_returns import calc_forward_log_returns_20d
from markets.targets.forward_volatility import calc_forward_log_return_stds_20d
from other.dates import list_trading_dates


class ObservationsData:
    """
    Class for managing regime data for a market.
    """

    def __init__(self, start: datetime, end: datetime):
        """
        Initialize the Data object by loading market features.
        :param start: Start datetime for the data retrieval, inclusive
        :param end: End datetime for the data retrieval, exclusive
        """
        self.start = start
        self.end = end
        self._dates = list_trading_dates(start, end)

        self._inputs = pd.DataFrame(
            {
                "term_difference": calc_term_structures(start, end),
                "log_return_20d": calc_log_returns_20d(start, end),
                "log_return_std_20d": calc_log_return_stds_20d(start, end),
                "yield_slope": calc_2y10y_slopes(start, end),
            }
        ).set_index(pd.Index(self._dates), drop=True)

        self._outputs = pd.DataFrame(
            {
                "log_return_20d": calc_forward_log_returns_20d(start, end),
                "log_return_std_20d": calc_forward_log_return_stds_20d(start, end),
            }
        ).set_index(pd.Index(self._dates), drop=True)

    @property
    def dates(self) -> list[datetime]:
        """
        Get the dates for the market data.
        :returns: List of dates
        """
        return self._dates

    @property
    def inputs(self) -> pd.DataFrame:
        """
        Get the input features for the market data.
        :returns: DataFrame of input features
        """
        return self._inputs

    @property
    def outputs(self) -> pd.DataFrame:
        """
        Get the output targets for the market data.
        :returns: DataFrame of output targets
        """
        return self._outputs
