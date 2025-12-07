__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
import yfinance as yf
from datetime import datetime

from zother.dates import get_next_trading_date

VIX = yf.Ticker("^VIX")
VIX3M = yf.Ticker("^VIX3M")


def calc_term_structures(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate the term structure between VIX and VIX3M for dates.
    :param start_date: Start date for the range, inclusive
    :param end_date: End date for the range, exclusive
    :returns: Series of term differences (VIX3M - VIX) indexed by date
    """
    prices = VIX.history(start=start_date, end=end_date)["Close"].astype(float)
    prices3m = VIX3M.history(start=start_date, end=end_date)["Close"].astype(float)

    # Remove timezone to make dates timezone-naive
    prices.index = prices.index.tz_localize(None)
    prices3m.index = prices3m.index.tz_localize(None)

    # Align indices to the same timezones and times
    prices_daily = prices.groupby(prices.index.date).last()
    prices3m_daily = prices3m.groupby(prices3m.index.date).last()
    prices_daily.index = pd.to_datetime(prices_daily.index)
    prices3m_daily.index = pd.to_datetime(prices3m_daily.index)
    prices_daily, prices3m_daily = prices_daily.align(prices3m_daily, join="inner")
    term_differences = prices3m_daily - prices_daily

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
    next_date = get_next_trading_date(date)
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
    next_date = get_next_trading_date(date)
    data = VIX3M.history(start=date, end=next_date)
    # Remove timezone to make dates timezone-naive
    data.index = data.index.tz_localize(None)
    close = data["Close"].iloc[0]
    return float(close)
