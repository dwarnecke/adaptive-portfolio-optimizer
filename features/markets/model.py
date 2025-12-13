__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"


import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path

from features.markets.hmm import HiddenMarkovModel
from utils.dates import list_dates


class RegimeModel:
    """
    Regime model that manages features and delegates HMM operations.
    :param data: DataFrame containing the features for regime modeling.
    :param states: Number of hidden states in the HMM.
    """

    def __init__(self, states: int, data: pd.DataFrame = None) -> None:
        """
        Initialize the regime model with feature scaling and HMM instance.
        :param data: DataFrame containing the features for regime modeling
        :param states: Number of hidden states in the HMM
        """
        self._num_states = states
        self._num_features = None
        self._num_observations = None

        self._scaling_means = None
        self._scaling_stds = None
        self._data = None

        # Initialize scaling parameters if data is provided
        if data is not None:
            self._update_parameters(data)

    def train(self, data: pd.DataFrame, max_iter: int = 50, tol: float = 1e-4) -> None:
        """
        Train the hidden Markov model to the data using the Baum-Welch algorithm.
        :param data: DataFrame containing the features for training the HMM
        :param max_iter: Maximum number of EM iterations
        :param tol: Convergence tolerance for log-likelihood improvement
        """
        self._update_parameters(data)
        self.hmm.train(self._data, max_iter=max_iter, tol=tol)

    def forward(self, date: datetime) -> np.ndarray:
        """
        Estimate the probabilities of each hidden state at the given date.
        :param date: Target date (pd.Timestamp or datetime) to autoregress to
        :returns: Probabilities for each state at the next time, shape (n_states,)
        """
        last_date = pd.to_datetime(self._data.index[-1])
        end_date = pd.to_datetime(date)
        trading_days = list_dates(last_date, end_date)

        # Skip the first date if it is the last date in the data
        if trading_days and trading_days[0] == last_date:
            trading_days = trading_days[1:]

        # Extrapolate state probabilities over the number of unseen trading days
        state_proba = self.hmm.predict_transition_proba(self._data)
        for _ in trading_days:
            state_proba = state_proba @ self.hmm.A

        return state_proba

    def calc_regime_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime probabilities for all dates in the given data.
        :param data: DataFrame with observation features to compute probabilities for
        :returns: DataFrame with state probabilities for each date
        """
        normalized_data = self._normalize_data(data)
        state_probs = self.hmm.calc_state_proba(normalized_data)
        prob_columns = [f"regime_{i}_prob" for i in range(self.num_states)]
        state_prob_df = pd.DataFrame(
            state_probs, index=data.index, columns=prob_columns
        )
        return state_prob_df

    def extend(self, new_data: pd.DataFrame) -> None:
        """
        Extend the data with new data from a DataFrame, excluding that present.
        Normalize new data using stored scaling parameters
        :param new_data: DataFrame with new data to add
        """
        # Initialize the data with the new normalized data if none exists
        if self._data is None:
            self._update_parameters(new_data)
            return

        # Extend the existing data to new dates with the new data
        norm_new = self._normalize_data(new_data)
        self._data = pd.concat([self._data, norm_new])
        self._data = self._data[~self._data.index.duplicated(keep="first")]
        self._data = self._data.sort_index()
        self._num_observations = self._data.shape[0]

    def save(self, filepath: str | Path) -> None:
        """
        Save the regime model to a file, including scaling parameters and HMM model.
        :param filepath: Path where to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save the HMM model to a separate file with _hmm suffix
        hmm_filepath = filepath.parent / f"{filepath.stem}_hmm.pkl"
        self.hmm.save(hmm_filepath)

        # Save the regime model metadata and scaling parameters
        regime_data = {
            "num_states": self.num_states,
            "num_observations": self._num_observations,
            "num_features": self._num_features,
            "scaling_means": self._scaling_means,
            "scaling_stds": self._scaling_stds,
            "data": self._data,
            "hmm_filepath": str(hmm_filepath),
        }

        with open(filepath, "wb") as f:
            pickle.dump(regime_data, f)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeModel":
        """
        Create a new RegimeModel instance by loading from a file.
        :param path: Path to the saved model file
        :returns: New RegimeModel instance with loaded parameters
        """
        path = Path(path)
        with open(path, "rb") as f:
            regime_data = pickle.load(f)

        # Create a new instance and load all the saved attributes
        model = cls(regime_data["num_states"], None)
        model._num_observations = regime_data["num_observations"]
        model._scaling_means = regime_data["scaling_means"]
        model._scaling_stds = regime_data["scaling_stds"]
        model._data = regime_data["data"]

        # Load the HMM model
        hmm_filepath = Path(regime_data["hmm_filepath"])
        model.hmm = HiddenMarkovModel.load(hmm_filepath)

        return model

    @property
    def num_states(self) -> int:
        """
        Get the number of hidden states in the HMM.
        :returns: Number of hidden states
        """
        return self._num_states

    def _update_parameters(self, data: pd.DataFrame) -> None:
        """
        Initialize scaling parameters from the provided data.
        :param data: DataFrame to use for initializing parameters
        """
        # Scale the data to the new calculated parameters
        self._scaling_means = data.mean().values.copy()
        self._scaling_stds = data.std(ddof=0).replace(0, 1).values.copy()
        self._data = self._normalize_data(data)

        # Initialize the HMM model with the updated parameters
        self._num_observations = data.shape[0]
        self._num_features = data.shape[1]
        self.hmm = HiddenMarkovModel(self.num_states, self._num_features)

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data using stored means and stds.
        :param data: DataFrame to normalize
        :returns: Normalized DataFrame
        """
        arr = (data.values - self._scaling_means) / self._scaling_stds
        normal_data = pd.DataFrame(arr, index=data.index, columns=data.columns)
        return normal_data.sort_index()
