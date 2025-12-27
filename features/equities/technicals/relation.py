__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import yfinance as yf

# Download SPY data once at module level
SPY = yf.Ticker("SPY")
SPY_CLOSE = SPY.history(start="2007-01-01", end="2030-01-01")["Close"]
SPY_CLOSE.index = pd.DatetimeIndex(SPY_CLOSE.index).tz_localize(None)
SPY_LOG_CLOSE = np.log(SPY_CLOSE + 1e-8)


def calc_relative_return(data: DataFrame, length: int) -> Series:
    """
    Calculate the relative return against the market over a given length.
    :param data: DataFrame containing stock log close data
    :param length: Length to calculate the relative return over
    :return: Series containing log relative returns
    """
    data = data.sort_index()

    # Calculate stock returns as the difference in log prices
    log_close = data["Log Close"]
    returns = log_close - log_close.shift(length)
    market_returns = SPY_LOG_CLOSE - SPY_LOG_CLOSE.shift(length)
    market_returns = market_returns.reindex(returns.index, method="ffill")
    relative_return = returns - market_returns

    return relative_return


def calc_rolling_beta(data: DataFrame, length: int) -> Series:
    """
    Calculate the rolling beta (sensitivity to market movements) over a given length.
    Beta measures how much the stock moves relative to the market (SPY).
    :param data: DataFrame containing stock log close data
    :param length: Length of the rolling window for beta calculation
    :return: Series containing rolling beta values
    """
    data = data.sort_index()

    # Calculate rolling beta over the specified length
    returns = data["Log Close"].diff()
    market_returns = SPY_LOG_CLOSE.diff()
    market_returns = market_returns.reindex(returns.index, method="ffill")
    rolling_cov = returns.rolling(window=length).cov(market_returns, ddof=1)
    rolling_var = market_returns.rolling(window=length).var(ddof=1)
    beta = rolling_cov / rolling_var

    return beta
