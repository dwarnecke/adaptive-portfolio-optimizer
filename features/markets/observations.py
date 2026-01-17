__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
from datetime import datetime, timedelta

from features.markets.macros.vix_term import calc_term_structures
from features.markets.macros.index_momentum import calc_log_returns_20d
from features.markets.macros.index_volatility import calc_log_return_stds_20d
from features.markets.macros.treasury_slope import calc_2y10y_slopes
from utils import list_dates


class ObservationsData:
    """
    Class for managing regime data for a market.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        """
        Initialize the Data object by loading market features.
        :param start_date: Start datetime for the data retrieval, inclusive
        :param end_date: End datetime for the data retrieval, exclusive
        """
        # Subtract 6 months of data to allow for feature calculations
        start_date = start_date - timedelta(days=180)
        self._start_date = start_date
        self._end_date = end_date
        self.dates = list_dates(start_date, end_date)

        data = {
            "term_difference": calc_term_structures(start_date, end_date),
            "log_ret_20d": calc_log_returns_20d(start_date, end_date),
            "log_ret_std_20d": calc_log_return_stds_20d(start_date, end_date),
            "yield_slope": calc_2y10y_slopes(start_date, end_date),
        }
        self.data = pd.DataFrame(data).set_index(pd.Index(self.dates), drop=True)
