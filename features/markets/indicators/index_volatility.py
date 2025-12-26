__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from utils.dates import list_dates

SPY = yf.Ticker("^GSPC")


def calc_log_return_stds(
    start_date: datetime, end_date: datetime, length: int
) -> pd.Series:
    """
    Calculate annualized log return standard deviation for the S&P 500 for lengths
    between dates.
    :param start_date: Start datetime of the date range, inclusive
    :param end_date: End datetime of the date range, exclusive
    :param length: Number of periods (days) for each volatility calculation
    :returns: Series of annualized volatilities indexed by end-of-period date
    """
    dates = pd.to_datetime(list_dates(start_date, end_date))
    start = start_date - pd.Timedelta(days=length * 2)
    prices = SPY.history(start=start, end=end_date)["Close"].astype(float)
    # Remove timezone to make dates timezone-naive
    prices.index = prices.index.tz_localize(None)
    log_prices = prices.apply(np.log)
    log_returns = log_prices.diff()
    volatilities = log_returns.rolling(window=length).std(ddof=1) * np.sqrt(252)
    volatilities_aligned = volatilities.reindex(dates)
    return volatilities_aligned


def calc_log_return_std(date: datetime, length: int = 20) -> float:
    """
    Calculate the annualized log return volatility for the S&P 500 for a specific date.
    :param date: End date for the period to calculate volatility.
    :param length: Number of trading days to look back from the end date (default is 20).
    :returns: Annualized volatility as a float.
    """
    end_date = date
    start_date = end_date - timedelta(days=length * 2)
    data = SPY.history(start=start_date, end=end_date)
    # Remove timezone to make dates timezone-naive
    data.index = data.index.tz_localize(None)
    close_prices = data["Close"].astype(float).tail(length + 1)
    log_returns = (close_prices / close_prices.shift(1)).dropna().apply(math.log)
    volatility = math.sqrt(252) * log_returns.std(ddof=1)
    return volatility


def calc_log_return_stds_5d(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate the annualized log return standard deviation for the S&P 500 over 5
    trading days for each period between dates.
    :param start_date: Start date (inclusive) as datetime
    :param end_date: End date (exclusive) as datetime
    :returns: Series of 5-day annualized volatilities indexed by end-of-period date
    """
    return calc_log_return_stds(start_date, end_date, length=5)


def calc_log_return_stds_20d(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate the annualized log return standard deviation for the S&P 500 over 20
    trading days for each period between dates.
    :param start_date: Start date (inclusive) as datetime
    :param end_date: End date (exclusive) as datetime
    :returns: Series of 20-day annualized volatilities indexed by end-of-period date
    """
    return calc_log_return_stds(start_date, end_date, length=20)


def calc_log_return_stds_60d(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate the annualized log return standard deviation for the S&P 500 over 60
    trading days for each period between dates.
    :param start_date: Start date (inclusive) as datetime
    :param end_date: End date (exclusive) as datetime
    :returns: Series of 60-day annualized volatilities indexed by end-of-period date
    """
    return calc_log_return_stds(start_date, end_date, length=60)


def calc_log_return_std_5d(date: datetime) -> float:
    """
    Calculate the annualized volatility for SPX over 5 trading days ending on a specific datetime.
    :param date: End date for the period to calculate volatility.
    :returns: Annualized volatility as a float.
    """
    return calc_log_return_std(date, length=5)


def calc_log_return_std_20d(date: datetime) -> float:
    """
    Calculate the annualized volatility for SPX over 20 trading days ending on a specific datetime.
    :param date: End date for the period to calculate volatility.
    :returns: Annualized volatility as a float.
    """
    return calc_log_return_std(date, length=20)


def calc_log_return_std_60d(date: datetime) -> float:
    """
    Calculate the annualized volatility for SPX over 60 trading days ending on a specific datetime.
    :param date: End date for the period to calculate volatility.
    :returns: Annualized volatility as a float.
    """
    return calc_log_return_std(date, length=60)
