__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
from pandas import DataFrame, Series


def calc_average_true_range(data: DataFrame, length: int = 14) -> Series:
    """
    Calculate the average true range (ATR) using log price differences.
    ATR measures volatility as percentage moves rather than absolute dollars.
    :param data: DataFrame containing high, low, and close data (in log format)
    :param length: Length to calculate the ATR over (default 14)
    :return: Series containing ATR values (percentage volatility)
    """
    data = data.sort_index()

    # Calculate true range using log prices
    log_high = data["Log High"]
    log_low = data["Log Low"]
    log_close = data["Log Close"]
    prev_log_close = log_close.shift(1)

    # Three components of true range
    tr1 = log_high - log_low
    tr2 = np.abs(log_high - prev_log_close)
    tr3 = np.abs(log_low - prev_log_close)

    # True range is the maximum
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Average true range
    atr = true_range.rolling(window=length).mean()

    return atr


def calc_max_drawdown(data: DataFrame, length: int) -> Series:
    """
    Calculate the maximum drawdown over a given length.
    Drawdown measures the largest peak-to-trough decline in the rolling window.
    :param data: DataFrame containing close data
    :param length: Length to calculate the maximum drawdown over
    :return: Series containing maximum drawdowns (always ≤ 0 in log space)
    """
    data = data.sort_index()
    log_close = data["Log Close"]
    rolling_max = log_close.rolling(window=length).max()
    drawdown = log_close - rolling_max
    max_drawdown = drawdown.rolling(window=length).min()
    return max_drawdown
