__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
from pandas import DatetimeIndex

from features.markets.observations import ObservationsData
from features.markets.regime.model import RegimeModel


class MarketData:
    """
    Class to hold and manage market regime and observation data.
    """

    def __init__(self, observations: ObservationsData, filepath: str):
        """
        Initialize the MarketData object by loading market data.
        :param observations: ObservationsData object containing market observations
        :param filepath: Path to regime model file
        """
        self.observations = observations
        self.regime = RegimeModel.load(filepath)
        self.proba = self.regime.calc_regime_proba(observations.data)
        self.data = pd.concat([observations.data, self.proba], axis=1)

    @property
    def index(self) -> DatetimeIndex:
        """
        Get the index dates of the market data.
        :return: DatetimeIndex of the data DataFrame
        """
        return self.data.index
