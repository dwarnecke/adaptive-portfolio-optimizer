__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import yfinance as yf

# Download SPY data once at module level
SPY = yf.Ticker("SPY")


def calc_relative_return(data: DataFrame, length: int) -> Series:
    """
    Calculate the relative return vs SPY (market) over a given length.
    This measures outperformance or underperformance relative to the market.
    :param data: DataFrame containing stock log close data
    :param length: Length to calculate the relative return over
    :return: Series containing relative returns (stock return - SPY return)
    """
    data = data.sort_index()

    # Download market data covering the same date range
    start_date = data.index.min()
    end_date = data.index.max()
    market_closes = SPY.history(start=start_date, end=end_date)["Close"]
    # Remove timezone to make dates timezone-naive
    market_closes.index = market_closes.index.tz_localize(None)
    market_log_close = np.log(market_closes + 1e-8)

    # Calculate stock returns as the difference in log prices
    log_close = data["Log Close"]
    returns = log_close - log_close.shift(length)
    market_returns = market_log_close - market_log_close.shift(length)
    aligned_market_returns = market_returns.reindex(returns.index, method="ffill")
    relative_return = returns - aligned_market_returns

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

    # Download market benchmark data covering the same date range
    start_date = data.index.min()
    end_date = data.index.max()
    market_closes = SPY.history(start=start_date, end=end_date)["Close"]
    market_closes.index = market_closes.index.tz_localize(None)
    market_log_close = np.log(market_closes + 1e-8)

    # Calculate rolling beta over the specified length
    returns = data["Log Close"].diff()
    market_returns = market_log_close.diff()
    aligned_market_returns = market_returns.reindex(returns.index, method="ffill")
    beta = pd.Series(index=returns.index, dtype=float)
    for i in range(length, len(returns)):
        stock_window = returns.iloc[i - length : i]
        market_window = aligned_market_returns.iloc[i - length : i]
        beta.iloc[i] = _calc_beta(stock_window, market_window)

    return beta


def _calc_beta(returns: Series, market_returns: Series) -> float:
    """
    Calculate beta (covariance / variance) for two return series.
    :param returns: Series of stock returns over a window
    :param market_returns: Series of market returns over the same window
    :return: Beta value or NaN if insufficient data
    """
    # Process only if there are enough valid data points
    if len(returns) < 2 or returns.isna().all() or market_returns.isna().all():
        return np.nan
    valid_mask = ~(returns.isna() | market_returns.isna())
    if valid_mask.sum() < 2:
        return np.nan

    # Calculate beta as covariance(stock, market) / variance(market)
    stock_clean = returns[valid_mask]
    market_clean = market_returns[valid_mask]
    covariance = np.cov(stock_clean, market_clean)[0, 1]
    market_variance = np.var(market_clean, ddof=1)
    if market_variance == 0 or not np.isfinite(market_variance):
        return np.nan

    return covariance / market_variance
