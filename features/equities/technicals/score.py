__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from pandas import DataFrame, Series


def calc_zscore(data: DataFrame, length: int, column: str = "Log Close") -> Series:
    """
    Calculate the z-score (standard score) for a given column and length.
    :param data: DataFrame containing the data
    :param length: Length to calculate the rolling mean and std over
    :param column: Column name to calculate the z-score for
    :return: Series containing z-scores
    """
    data = data.sort_index()
    values = data[column]

    rolling_mean = values.rolling(window=length).mean()
    rolling_std = values.rolling(window=length).std()

    zscore = (values - rolling_mean) / (rolling_std + 1e-4)

    return zscore
