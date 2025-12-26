__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
import yfinance as yf
from utils.dates import list_dates

SPY = yf.Ticker("^GSPC")


def calc_forward_log_return_stds(start_date, end_date, length: int) -> pd.Series:
    """
    Calculate forward log return standard deviations for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :param length: Number of trading days to look ahead for each volatility calculation
    :returns: Series of forward annualized volatilities indexed by start date
    """
    dates = pd.to_datetime(list_dates(start_date, end_date))

    # Extend end date to get enough forward data
    extended_end = end_date + pd.Timedelta(days=length * 2)
    prices = SPY.history(start=start_date, end=extended_end)["Close"].astype(float)
    prices.index = prices.index.tz_localize(None)
    log_prices = prices.apply(np.log)
    log_returns = log_prices.diff()

    # Calculate forward-looking volatilities using rolling window shifted backward
    volatilities = log_returns.rolling(window=length).std(ddof=1) * np.sqrt(252)
    forward_volatilities = volatilities.shift(-length)
    
    # Reindex to trading dates
    forward_volatilities_aligned = forward_volatilities.reindex(dates)
    return forward_volatilities_aligned


def calc_forward_log_return_stds_5d(start_date, end_date) -> pd.Series:
    """
    Calculate 5-day forward log return standard deviations for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :returns: Series of 5-day forward annualized volatilities indexed by start date
    """
    return calc_forward_log_return_stds(start_date, end_date, length=5)


def calc_forward_log_return_stds_20d(start_date, end_date) -> pd.Series:
    """
    Calculate 20-day forward log return standard deviations for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :returns: Series of 20-day forward annualized volatilities indexed by start date
    """
    return calc_forward_log_return_stds(start_date, end_date, length=20)


def calc_forward_log_return_stds_60d(start_date, end_date) -> pd.Series:
    """
    Calculate 60-day forward log return standard deviations for the S&P 500 between dates.
    :param start_date: Start date for the date range, inclusive
    :param end_date: End date for the date range, exclusive
    :returns: Series of 60-day forward annualized volatilities indexed by start date
    """
    return calc_forward_log_return_stds(start_date, end_date, length=60)
