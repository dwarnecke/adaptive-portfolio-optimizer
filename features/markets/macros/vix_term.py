__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
import yfinance as yf
from datetime import datetime

from utils import get_next_date, list_dates

VIX = yf.Ticker("^VIX")
VIX3M = yf.Ticker("^VIX3M")


def calc_term_structures(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate the term structure between VIX and VIX3M for dates.
    :param start_date: Start date for the range, inclusive
    :param end_date: End date for the range, exclusive
    :returns: Series of term differences (VIX3M - VIX) indexed by date
    """
    dates = pd.to_datetime(list_dates(start_date, end_date))
    prices = VIX.history(start=start_date, end=end_date)["Close"].astype(float)
    prices3m = VIX3M.history(start=start_date, end=end_date)["Close"].astype(float)
    # Remove timezone to make dates timezone-naive
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    if isinstance(prices3m.index, pd.DatetimeIndex) and prices3m.index.tz is not None:
        prices3m.index = prices3m.index.tz_localize(None)
    # Reindex to trading dates and forward fill
    prices_aligned = prices.reindex(dates).ffill()
    prices3m_aligned = prices3m.reindex(dates).ffill()
    term_differences = prices3m_aligned - prices_aligned
    return term_differences


def calc_term_difference(date: datetime) -> float:
    """
    Calculate the term difference between VIX and VIX3M for a date.
    :param date: Date for which to calculate the term difference
    :returns: Term difference (VIX3M - VIX) as a float
    """
    vix_close = get_vix_close(date)
    vix3m_close = get_vix3m_close(date)
    term_diff = vix3m_close - vix_close
    return term_diff


def get_vix_close(date: datetime) -> float:
    """
    Get the VIX closing value for a date.
    :param date: Date for which to retrieve the VIX closing value.
    :returns: VIX closing value.
    """
    next_date = get_next_date(date)
    data = VIX.history(start=date, end=next_date)
    # Remove timezone to make dates timezone-naive
    data.index = data.index.tz_localize(None)
    close = data["Close"].iloc[0]
    return float(close)


def get_vix3m_close(date: datetime) -> float:
    """
    Get the VIX3M (3-month implied volatility) closing value for a date.
    :param date: Date for which to retrieve the VIX3M closing value.
    :returns: VIX3M closing value as a float.
    """
    next_date = get_next_date(date)
    data = VIX3M.history(start=date, end=next_date)
    # Remove timezone to make dates timezone-naive
    data.index = data.index.tz_localize(None)
    close = data["Close"].iloc[0]
    return float(close)
