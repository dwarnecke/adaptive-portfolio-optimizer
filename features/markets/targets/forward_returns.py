__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
import yfinance as yf

SPY = yf.Ticker("^GSPC")


def calc_forward_log_returns(start_date, end_date, length: int) -> pd.Series:
    """
    Calculate forward log returns for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :param forward_length: Number of trading days to look ahead for each return
    :returns: Series of forward log returns indexed by start-of-period date
    """
    prices = SPY.history(start=start_date, end=end_date)["Close"].astype(float)
    # Remove timezone to make dates timezone-naive
    prices.index = prices.index.tz_localize(None)
    log_prices = prices.apply(np.log)
    log_returns = log_prices.shift(-length) - log_prices
    start_index = pd.to_datetime(start_date)
    end_index = pd.to_datetime(end_date)
    indices = (start_index <= prices.index) & (prices.index < end_index)
    return log_returns[indices]


def calc_forward_log_returns_5d(start_date, end_date) -> pd.Series:
    """
    Calculate 5-day forward log returns for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :returns: Series of 5-day forward log returns indexed by start-of-period date
    """
    return calc_forward_log_returns(start_date, end_date, length=5)


def calc_forward_log_returns_20d(start_date, end_date) -> pd.Series:
    """
    Calculate 20-day forward log returns for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :returns: Series of 20-day forward log returns indexed by start-of-period date
    """
    return calc_forward_log_returns(start_date, end_date, length=20)


def calc_forward_log_returns_60d(start_date, end_date) -> pd.Series:
    """
    Calculate 60-day forward log returns for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :returns: Series of 60-day forward log returns indexed by start-of-period date
    """
    return calc_forward_log_returns(start_date, end_date, length=60)
