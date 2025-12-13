__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
from pandas import DataFrame

from features.equities.fundamentals.fundamentals import FundamentalsData
from features.equities.technicals.technicals import TechnicalsData


class EquityData:
    """
    Class to hold and manage ticker technical and fundamental data for an equity.
    """

    def __init__(self, ticker: str, data: dict):
        """
        Initialize the Equity object by loading ticker data.
        :param ticker: Stock ticker symbol
        :param data: Dictionary containing loaded fundamentals data
        """
        self._TICKER = ticker
        self._fundamentals = FundamentalsData(ticker, data)
        self._technicals = TechnicalsData(ticker)

        # Skip processing if either technical or fundamental data is empty
        if self._technicals.empty or self._fundamentals.empty:
            self.data = pd.DataFrame()
            self.targets = pd.DataFrame()
            return

        data, targets = self._calculate()
        self.data = data
        self.targets = targets

    def _calculate(self) -> tuple[DataFrame, DataFrame]:
        """
        Calculates input data and output targets from technical and fundamental data.
        :returns: Tuple of (features, targets) DataFrames
        """
        data = self._calc_data()
        targets = self._calc_targets()

        # Only keep dates present in both data and targets
        index = data.index.intersection(targets.index)
        data = data.loc[index]
        targets = targets.loc[index]
        return data, targets

    def _calc_data(self) -> DataFrame:
        """
        Calculates input data from technical and fundamental data.
        :returns: DataFrame of combined features
        """
        tech_features = self._technicals.features
        fund_features = self._fundamentals.features
        data = tech_features.join(fund_features, how="inner")
        return data

    def _calc_targets(self) -> DataFrame:
        """
        Calculates 20-day log returns as targets from technical data.
        :returns: DataFrame of target variables
        """
        data = self._technicals.data
        if data.empty:
            return pd.DataFrame()
        closes = data["Close"]
        targets = np.log(closes.shift(-20) / closes)
        targets = targets.dropna()
        return targets

    @property
    def empty(self) -> bool:
        """
        Check if the equity data is empty.
        :return: True if no data is available, False otherwise
        """
        return self.data.empty
