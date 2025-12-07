__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from pandas import DataFrame, Series


def calc_rate_of_change(data: DataFrame, length: int) -> Series:
    """
    Calculate the rate of change (ROC) for a given data set and length.
    ROC measures momentum by comparing current price to past price.
    :param data: DataFrame containing close data
    :param length: Length to calculate the rate of change over
    :return: Series containing rate of change (log returns)
    """
    data = data.sort_index()
    log_close = data["Log Close"]
    lagged_close = log_close.shift(length)
    rate_of_change = log_close - lagged_close
    return rate_of_change


def calc_relative_strength_index(data: DataFrame, length: int = 14) -> Series:
    """
    Calculate the relative strength index (RSI) for a given data set and length.
    RSI measures momentum on a 0-100 scale, with >70 overbought and <30 oversold.
    :param data: DataFrame containing close data
    :param length: Length to calculate the RSI over (default 14)
    :return: Series containing RSI values (0-100)
    """
    data = data.sort_index()
    log_close = data["Log Close"]
    price_changes = log_close.diff()

    # Separate gains and losses
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=length).mean()
    avg_losses = losses.rolling(window=length).mean()

    # Calculate RSI
    rs = avg_gains / (avg_losses + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi
