__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd

from markets.observations import ObservationsData
from markets.regime import RegimeModel


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
        obs_features = self._observations.inputs.copy()

        # Calculate state probabilities for each date
        state_probs = []
        dates = obs_features.index.tolist()

        for i, date in enumerate(dates):
            # Use only data up to current date for state estimation
            data_up_to_date = obs_features.iloc[:i+1]
            
            # Normalize using regime model's scaling
            normalized_data = self._regime._normalize_data(data_up_to_date)
            
            # Get state probabilities at this date
            probs = self._regime.hmm.predict_transition_proba(normalized_data)
            state_probs.append(probs)

        # Create DataFrame with state probabilities
        state_prob_columns = [f"regime_{i}_prob" for i in range(self._regime._num_states)]
        state_prob_df = pd.DataFrame(
            state_probs,
            index=dates,
            columns=state_prob_columns
        )

        # Combine observations and regime probabilities
        combined = pd.concat([obs_features, state_prob_df], axis=1)

        return combined

