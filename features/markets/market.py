__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
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

        # Normalize all data once instead of per-iteration
        normalized_data = self._regime._normalize_data(observation_inputs)

        # Get state probabilities for all dates at once using vectorized forward algorithm
        # This is much faster than calling predict_transition_proba in a loop
        logB = self._regime.hmm._compute_log_emission_probs(normalized_data)
        log_alpha, log_scale = self._regime.hmm._log_forward(logB)
        log_beta = self._regime.hmm._log_backward(logB, log_scale)

        # Compute gamma (state probabilities) for all timesteps
        log_gamma = log_alpha + log_beta
        log_gamma -= log_gamma.max(axis=1, keepdims=True)  # Numerical stability
        gamma = pd.DataFrame(
            np.exp(log_gamma),
            index=observation_inputs.index,
            columns=[f"regime_{i}_prob" for i in range(self._regime.num_states)],
        )

        # Normalize probabilities to sum to 1
        gamma = gamma.div(gamma.sum(axis=1), axis=0)

        # Combine observations and regime probabilities
        combined = pd.concat([observation_inputs, gamma], axis=1)

        return combined
