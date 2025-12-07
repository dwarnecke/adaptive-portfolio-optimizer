__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from pandas import DataFrame

from features.equities.fundamentals.fundamentals import FundamentalsData
from features.equities.technicals.technicals import TechnicalsData


class Equity:
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

    @property
    def ticker(self) -> str:
        """
        Get the ticker symbol.
        :returns: Ticker symbol as a string
        """
        return self._TICKER

    @property
    def features(self) -> DataFrame:
        """
        Get the features data for the equity.
        :returns: Features object
        """
        fundamentals_features = self._fundamentals.features
        technicals_features = self._technicals.features
        combined_features = fundamentals_features.join(technicals_features, how="inner")
        return combined_features
