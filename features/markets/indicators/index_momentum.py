__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from utils.dates import list_dates

SPY = yf.Ticker("^GSPC")


def calc_log_returns(
    start_date: datetime, end_date: datetime, length: int
) -> pd.Series:
    """
    Calculate log returns for the S&P 500 for each period of given length between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the range, exclusive
    :param length: Number of trading days for each log return
    :returns: Series of log returns indexed by end-of-period date
    """
    dates = pd.to_datetime(list_dates(start_date, end_date))
    start = pd.to_datetime(start_date) - pd.Timedelta(days=length * 2)
    prices = SPY.history(start=start, end=end_date)["Close"].astype(float)
    # Remove timezone to make dates timezone-naive
    prices.index = prices.index.tz_localize(None)
    log_prices = prices.apply(np.log)
    log_returns = log_prices.diff(periods=length)
    log_returns_aligned = log_returns.reindex(dates)
    return log_returns_aligned


def calc_log_returns_5d(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate log returns for SPX over 5 trading days for each period between start_date (inclusive) and end_date (exclusive).
    :param start_date: Start date (inclusive) as a datetime object
    :param end_date: End date (exclusive) as a datetime object
    :returns: Series of 5-day log returns indexed by end-of-period date
    """
    return calc_log_returns(start_date, end_date, length=5)


def calc_log_returns_20d(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate log returns for SPX over 20 trading days for each period between start_date (inclusive) and end_date (exclusive).
    :param start_date: Start date (inclusive) as a datetime object
    :param end_date: End date (exclusive) as a datetime object
    :returns: Series of 20-day log returns indexed by end-of-period date
    """
    return calc_log_returns(start_date, end_date, length=20)


def calc_log_returns_60d(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate log returns for SPX over 60 trading days for each period between start_date (inclusive) and end_date (exclusive).
    :param start_date: Start date (inclusive) as a datetime object
    :param end_date: End date (exclusive) as a datetime object
    :returns: Series of 60-day log returns indexed by end-of-period date
    """
    return calc_log_returns(start_date, end_date, length=60)
