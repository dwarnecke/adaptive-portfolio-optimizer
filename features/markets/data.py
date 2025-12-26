__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex

from features.markets.observations import ObservationsData
from features.markets.regime_model import RegimeModel


class MarketData:
    """
    Class to hold and manage market regime and observation data.
    """

    def __init__(self, observations: ObservationsData, model_path: str):
        """
        Initialize the MarketData object by loading market data.
        :param observations: ObservationsData object containing market observations
        :param model_path: Path to regime model file
        """
        self._observations = observations
        self._regime = RegimeModel.load(model_path)
        self.data = self._aggregate()

    def _aggregate(self) -> DataFrame:
        """
        Aggregate observation features with regime state probabilities.
        :returns: DataFrame with all features including regime probabilities
        """
        observation_inputs = self._observations.inputs.copy()
        state_probs = self._regime.calc_regime_proba(observation_inputs)
        combined = pd.concat([observation_inputs, state_probs], axis=1)
        return combined
    
    @property
    def index(self) -> DatetimeIndex:
        """
        Get the index dates of the market data.
        :return: DatetimeIndex of the data DataFrame
        """
        return self.data.index
