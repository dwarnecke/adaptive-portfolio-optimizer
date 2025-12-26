__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex

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
        self.TICKER = ticker
        self.fundamentals = FundamentalsData(ticker, data)
        self.technicals = TechnicalsData(ticker)
        
        if len(self.technicals) == 0 or len(self.fundamentals) == 0:
            self.data = pd.DataFrame()
            self.targets = pd.DataFrame()
            return

        data, targets = self._calculate()
        self.data = data
        self.targets = targets

    def get_open(self, date: datetime) -> float:
        """
        Get the opening price for the equity on a given date.
        :param date: Date to get the opening price as of the open
        :return: Opening price on the given date
        """
        return self.data["Open"].loc[date]
    
    def get_close(self, date: datetime) -> float:
        """
        Get the closing price for the equity on a given date.
        :param date: Date to get the closing price as of the close
        :return: Closing price on the given date
        """
        return self.data["Close"].loc[date]
    
    def get_return(self, date: datetime) -> float:
        """
        Get the return for the equity on a given date.
        :param date: Date to get the return for
        :return: Return on the given date
        """
        return self.targets.loc[date]

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
        tech_features = self.technicals._features
        fund_features = self.fundamentals.features
        data = tech_features.join(fund_features, how="inner")
        return data

    def _calc_targets(self) -> DataFrame:
        """
        Calculates 20-day log returns as targets from technical data.
        :returns: DataFrame of target variables
        """
        data = self.technicals._data
        if len(data) == 0:
            return pd.DataFrame()
        closes = data["Close"]
        targets = np.log(closes.shift(-20) / closes)
        targets = targets.dropna()
        return targets

    def __len__(self) -> bool:
        """
        Get the number of data points available.
        :return: Number of index dates in the data DataFrame
        """
        return len(self.data)
    
    @property
    def index(self) -> DatetimeIndex:
        """
        Get the index dates of the equity data.
        :return: DatetimeIndex of the data DataFrame
        """
        return self.data.index
