__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import yfinance as yf
from datetime import datetime, timedelta

from utils.dates import get_last_trading_date


def get_close_price(ticker: str, date: datetime) -> float:
    """
    Get the last closing price of a stock on a date.
    :param ticker: Stock ticker symbol
    :param date: Date for which to get the closing price
    :return: Closing price as a float
    """
    date = get_last_trading_date(date)
    start = date.strftime("%Y-%m-%d")
    end = date + timedelta(days=1)
    history = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = history["Close"].iloc[0]
    return close.item()
