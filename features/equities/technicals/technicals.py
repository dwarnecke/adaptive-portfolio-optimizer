__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
import yfinance as yf

from features.equities.technicals.trend import calc_rate_of_change, calc_relative_strength_index
from features.equities.technicals.volatility import calc_average_true_range, calc_max_drawdown
from features.equities.technicals.position import calc_zscore
from features.equities.technicals.relation import calc_relative_return, calc_rolling_beta


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

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the technical data DataFrame.
        :return: DataFrame copy containing technical data
        """
        return self._data.copy()
    
    @property
    def features(self) -> pd.DataFrame:
        """
        Get the technical features DataFrame.
        :return: DataFrame copy containing technical features
        """
        return self._features.copy()

    def _download_data(self):
        """
        Download historical stock data.
        """
        # Download maximum available historical data
        data = yf.download(self._TICKER, period="max", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Remove timezone to make dates timezone-naive
        data.index = data.index.tz_localize(None)

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
        self._features = pd.DataFrame(index=self._data.index)

        # Core indicator measure different aspects of stock performance
        lengths = [5, 60, 250, 1000]
        for length in lengths:
            self._calc_normal_scores(length)
            self._calc_drawdown(length)
            self._calc_momentum(length)

        # Market relative indicators measure performance against the market
        relative_return_lengths = [5, 60, 250]
        for length in relative_return_lengths:
            self._calc_relative_return(length)
        beta_lengths = [60, 250]
        for length in beta_lengths:
            self._calc_beta(length)

        # Technical indicators are widely used measures of stock behavior
        self._calc_strength()
        self._calc_volatility()

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
