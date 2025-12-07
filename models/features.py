__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime

from features.equities.equity import EquityData
from features.markets.observations import ObservationsData


class FeaturesData:
    """
    Class to hold calculated features for particular equity.
    """

    def __init__(self, equity: EquityData, market: ObservationsData):
        """
        Initialize the features data for a given equity and market.
        :param equity: EquityData object containing equity data
        :param market: MarketData object containing market data
        """
        self._equity = equity
        self._market = market
        self._features = equity.features.join(market.features, how="inner")

    def split(
        self, train_end: datetime, dev_end: datetime, test_end: datetime
    ) -> tuple:
        """
        Split the features data into training, development, and testing sets.
        :param train_end: End date for the training set, exclusive
        :param dev_end: End date for the development set, exclusive
        :param test_end: End date for the testing set, exclusive
        :return: Tuple containing (train_data, dev_data, test_data) DataFrames
        """
        train_data = self._features[self._features.index < train_end].copy()
        dev_data = self._features[
            (self._features.index > train_end) & (self._features.index < dev_end)
        ].copy()
        test_data = self._features[
            (self._features.index > dev_end) & (self._features.index < test_end)
        ].copy()
        return train_data, dev_data, test_data
