__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from pandas import DataFrame

from utils.dates import get_last_date

# Suppress yfinance error messages
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def get_close_price(ticker: str, date: datetime) -> float:
    """
    Get the last closing price of a stock on a date.
    :param ticker: Stock ticker symbol
    :param date: Date for which to get the closing price
    :return: Closing price as a float
    """
    date = get_last_date(date)
    start = date.strftime("%Y-%m-%d")
    end = date + timedelta(days=1)
    history = yf.download(ticker, start=start, end=end, progress=False)
    close = history["Close"].iloc[0]
    return close.item()


def download_data(ticker: str) -> DataFrame:
    """
    Download historical stock data for a ticker.
    :param ticker: Stock ticker symbol to download
    :return: DataFrame containing historical stock data
    """
    data = yf.download(ticker, period="max", progress=False, auto_adjust=True)

    # Skip processing if no data was downloaded
    if data.empty:
        return pd.DataFrame

    # Reindex multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = data.index.tz_localize(None)
    return data
