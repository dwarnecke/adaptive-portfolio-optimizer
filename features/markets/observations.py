__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
from datetime import datetime

from features.markets.indicators.vix_term import calc_term_structures
from features.markets.indicators.index_momentum import calc_log_returns_20d
from features.markets.indicators.index_volatility import calc_log_return_stds_20d
from features.markets.indicators.treasury_slope import calc_2y10y_slopes
from features.markets.forwards.forward_returns import calc_forward_log_returns_20d
from features.markets.forwards.forward_volatility import calc_forward_log_return_stds_20d
from utils.dates import list_dates


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
        self._start_date = start_date
        self._end_date = end_date
        self.dates = list_dates(start_date, end_date)
        
        x = {
            "term_difference": calc_term_structures(start_date, end_date),
            "log_ret_20d": calc_log_returns_20d(start_date, end_date),
            "log_ret_std_20d": calc_log_return_stds_20d(start_date, end_date),
            "yield_slope": calc_2y10y_slopes(start_date, end_date),
        }
        y = {
            "log_ret_20d": calc_forward_log_returns_20d(start_date, end_date),
            "log_ret_std_20d": calc_forward_log_return_stds_20d(start_date, end_date),
        }
        self.inputs = pd.DataFrame(x).set_index(pd.Index(self.dates), drop=True)
        self.outputs = pd.DataFrame(y).set_index(pd.Index(self.dates), drop=True)
