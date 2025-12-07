__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd

from features.markets.observations import ObservationsData
from features.markets.regime import RegimeModel


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
        self._features = self._combine_features()

    @property
    def observations(self) -> ObservationsData:
        """
        Get the observations data.
        :returns: ObservationsData object
        """
        return self._observations

    @property
    def regime(self) -> RegimeModel:
        """
        Get the regime model.
        :returns: RegimeModel object
        """
        return self._regime

    @property
    def features(self) -> pd.DataFrame:
        """
        Get the combined features DataFrame with observations and regime probabilities.
        :returns: DataFrame with observation features + regime state probabilities
        """
        return self._features.copy()

    @property
    def targets(self) -> pd.DataFrame:
        """
        Get the target variables for prediction.
        :returns: DataFrame with forward returns and volatility
        """
        return self._observations.outputs.copy()

    def _combine_features(self) -> pd.DataFrame:
        """
        Combine observation features with regime state probabilities.
        :returns: DataFrame with all features including regime probabilities
        """
        observation_inputs = self._observations.inputs.copy()

        # Calculate state probabilities for each date
        state_probs = []
        dates = observation_inputs.index.tolist()
        for i, date in enumerate(dates):
            # Use only data up to current date for state estimation
            data_to_date = observation_inputs.iloc[: i + 1]
            normalized_data = self._regime._normalize_data(data_to_date)
            probs = self._regime.hmm.predict_transition_proba(normalized_data)
            state_probs.append(probs)

        # Create DataFrame with state probabilities
        prob_columns = [f"regime_{i}_prob" for i in range(self._regime.num_states)]
        state_prob_df = pd.DataFrame(state_probs, index=dates, columns=prob_columns)

        # Combine observations and regime probabilities
        combined = pd.concat([observation_inputs, state_prob_df], axis=1)

        return combined
