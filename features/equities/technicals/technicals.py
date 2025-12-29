__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd

from features.equities.technicals.trend import (
    calc_rate_of_change,
    calc_relative_strength_index,
)
from features.equities.technicals.volatility import (
    calc_average_true_range,
    calc_max_drawdown,
)
from features.equities.technicals.score import calc_zscore
from features.equities.technicals.relation import (
    calc_relative_return,
    calc_rolling_beta,
)

from utils.tickers import download_data


class TechnicalsData:
    """
    Class to hold and manage technical data for a company.
    """

    def __init__(self, ticker: str):
        """
        Initialize the Technicals object with ticker symbol.
        :param ticker: Stock ticker symbol of the company
        """
        self._TICKER = ticker
        self._data = None
        self._download_data()

        self._features = pd.DataFrame()
        self._calc_indicators()

    def _download_data(self):
        """
        Download historical stock data.
        """
        data = download_data(self._TICKER)

        # Skip processing if no data was downloaded
        if data.empty:
            self._data = pd.DataFrame()
            return

        # Log transformations standardize movements on larger price scales
        data["Log Close"] = np.log(data["Close"] + 1e-8)
        data["Log High"] = np.log(data["High"] + 1e-8)
        data["Log Low"] = np.log(data["Low"] + 1e-8)
        data["Log Open"] = np.log(data["Open"] + 1e-8)
        data["Log Volume"] = np.log(data["Volume"] + 1e-8)
        self._data = data

    def _calc_indicators(self):
        """
        Calculate all technical indicators and add them to the data DataFrame.
        """
        if len(self._data) == 0:
            self._features = pd.DataFrame()
            return
        self._features = pd.DataFrame(index=self._data.index)

        lengths = [5, 60, 120, 250]
        for length in lengths:
            self._calc_normal_scores(length)
            self._calc_drawdown(length)
            self._calc_momentum(length)

        lengths = [5, 60, 250]
        for length in lengths:
            self._calc_relative_return(length)
        
        lengths = [60, 250]
        for length in lengths:
            self._calc_beta(length)
        
        self._calc_strength()
        self._calc_volatility()
        
        # Add NA indicator for insufficient technical data and fill NaNs as 0
        tech_cols = [col for col in self._features.columns]
        indicator = self._features[tech_cols].isna().any(axis=1).astype(int)
        self._features["TECH_NA"] = indicator
        self._features = self._features.fillna(0)

    def _calc_normal_scores(self, length: int):
        """
        Calculate the moving average price and volume normal scores over a given length.
        :param length: Length to calculate the moving average over
        """
        columns = ["Log Close", "Log Volume"]
        for column in columns:
            scores = calc_zscore(self._data, length, column=column)
            self._features[f"{column} Normal Score {length}"] = scores

    def _calc_drawdown(self, length: int):
        """
        Calculate the maximum drawdown and drawup for the stock.
        :param length: Length to calculate the drawdowns over
        """
        max_drawdown = calc_max_drawdown(self._data, length)
        self._features[f"Max Drawdown {length}"] = max_drawdown

    def _calc_momentum(self, length: int = 14):
        """
        Calculate the period returns for the stock.
        :param length: Length to calculate the change over, default is 14
        """
        returns = calc_rate_of_change(self._data, length)
        self._features[f"ROC {length}"] = returns

    def _calc_strength(self, length: int = 14):
        """
        Calculate the Relative Strength Index (RSI) for the stock.
        :param length: Length to calculate the RSI over, default is 14
        """
        rsi = calc_relative_strength_index(self._data, length)
        self._features[f"RSI {length}"] = rsi

    def _calc_volatility(self, length: int = 14):
        """
        Calculate the Average True Range (ATR) for the stock.
        :param length: Length to calculate the ATR over, default is 14
        """
        atr = calc_average_true_range(self._data, length)
        self._features[f"ATR {length}"] = atr

    def _calc_relative_return(self, length: int):
        """
        Calculate the relative return vs SPY for the stock.
        :param length: Length to calculate the relative return over
        """
        relative_return = calc_relative_return(self._data, length)
        self._features[f"Relative Return {length}"] = relative_return

    def _calc_beta(self, length: int):
        """
        Calculate the rolling beta vs SPY for the stock.
        :param length: Length to calculate the rolling beta over
        """
        beta = calc_rolling_beta(self._data, length)
        self._features[f"Beta {length}"] = beta

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the loaded technical data for the ticker.
        :return: DataFrame containing loaded technical data
        """
        return self._data.copy()
    
    @property
    def features(self) -> pd.DataFrame:
        """
        Get the calculated features DataFrame.
        :return: DataFrame of calculated technical features
        """
        return self._features.copy()

    def __len__(self) -> bool:
        """
        Get the number of data points available.
        :return: Number of index dates in the data DataFrame
        """
        return len(self._features)
