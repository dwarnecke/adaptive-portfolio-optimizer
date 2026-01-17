__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"


import json
import pandas as pd
import pickle
from datetime import datetime
from pandas import DataFrame
from pathlib import Path

from config.hyperparameters import HYPERPARAMETERS
from config.paths import REGIMES_DIR
from features.markets.regime.hmm import HiddenMarkovModel


class RegimeModel:
    """
    Regime model that manages features and delegates HMM operations.
    :param data: DataFrame containing the features for regime modeling.
    :param states: Number of hidden states in the HMM.
    """

    def __init__(
        self,
        data: DataFrame,
        parameters: dict = HYPERPARAMETERS["features"],
    ) -> None:
        """
        Initialize the regime model with feature scaling and HMM instance.
        :param data: DataFrame containing the features for regime modeling
        :param parameters: Dictionary of regime model parameters
        """
        self._hyperparameters = parameters
        self._num_states = parameters["num_states"]
        self._num_observations = data.shape[0]
        self._num_features = data.shape[1]
        max_iter = parameters["max_iter"]
        tol = parameters["tol"]

        # Normalization is important for HMM EM training
        self._scaling_means = data.mean().values.copy()
        self._scaling_stds = data.std(ddof=0).replace(0, 1).values.copy()
        self._data = self._normalize_data(data)

        self._hmm = HiddenMarkovModel(self._num_states, self._num_features)
        self._hmm.train(self._data, max_iter=max_iter, tol=tol)

    def calc_regime_proba(self, data: DataFrame) -> DataFrame:
        """
        Calculate regime probabilities for all dates in the given data.
        :param data: DataFrame with observation features to compute probabilities for
        :returns: DataFrame with state probabilities for each date
        """
        normalized_data = self._normalize_data(data)
        probas = self._hmm.calc_state_proba(normalized_data)
        columns = [f"regime_{i}_prob" for i in range(self._num_states)]
        state_prob_df = pd.DataFrame(probas, index=data.index, columns=columns)
        return state_prob_df

    def extend(self, new_data: DataFrame):
        """
        Extend the data with new data from a DataFrame, excluding that present.
        :param new_data: DataFrame with new data to add
        """
        norm_new = self._normalize_data(new_data)
        self._data = pd.concat([self._data, norm_new])
        self._data = self._data[~self._data.index.duplicated(keep="first")]
        self._data = self._data.sort_index()
        self._num_observations = self._data.shape[0]

    def save(self, directory: Path | str = REGIMES_DIR) -> Path:
        """
        Save the regime model to a timestamped directory with manifest.
        :param directory: Base directory for regime models, default config
        :returns: Path to the saved model directory
        """
        # Timestamp directory to prevent overwriting existing models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = Path(directory) / f"regime_{timestamp}"
        directory.mkdir(parents=True, exist_ok=True)

        hmm_path = directory / "regime_model_hmm.pkl"
        self._hmm.save(hmm_path)

        regime_path = directory / "regime_model.pkl"
        regime_data = self._to_dict(hmm_path)
        with open(regime_path, "wb") as f:
            pickle.dump(regime_data, f)
        
        manifest_path = directory / "regime_manifest.json"
        manifest = self._to_manifest()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Regime model saved to {directory}")
        print(f"Manifest: {manifest_path}")
        return directory

    def _to_dict(self, hmm_path: Path) -> dict:
        """
        Create dictionary with regime model data for serialization.
        :param hmm_path: Path to the saved HMM file
        :return: Dictionary containing regime model data
        """
        return {
            "num_states": self._num_states,
            "num_observations": self._num_observations,
            "num_features": self._num_features,
            "scaling_means": self._scaling_means,
            "scaling_stds": self._scaling_stds,
            "data": self._data,
            "hmm_filepath": str(hmm_path),
        }

    def _to_manifest(self) -> dict:
        """
        Create manifest dictionary with regime model metadata.
        :return: Manifest dictionary
        """
        return {
            "regime_file": "regime_model.pkl",
            "created_at": datetime.now().isoformat(),
            "start_date": self._data.index.min().isoformat(),
            "end_date": self._data.index.max().isoformat(),
            "num_states": self._num_states,
            "num_observations": self._num_observations,
            "num_features": self._num_features,
            "features": list(self._data.columns),
            "hyperparameters": self._hyperparameters,
        }

    @classmethod
    def load(cls, filepath: Path | str) -> "RegimeModel":
        """
        Create a new RegimeModel instance by loading from a file.
        :param filepath: Path to the saved model file
        :returns: New RegimeModel instance with loaded parameters
        """
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            regime_data = pickle.load(f)

        # Load the regime model without __init__ to avoid stochastic retraining
        model = cls.__new__(cls)
        model._num_states = regime_data["num_states"]
        model._num_features = regime_data["num_features"]
        model._num_observations = regime_data["num_observations"]
        model._scaling_means = regime_data["scaling_means"]
        model._scaling_stds = regime_data["scaling_stds"]
        model._data = regime_data["data"]

        hmm_path = Path(regime_data["hmm_filepath"])
        model._hmm = HiddenMarkovModel.load(hmm_path)

        return model

    def _normalize_data(self, data: DataFrame) -> DataFrame:
        """
        Normalize data using stored means and stds.
        :param data: DataFrame to normalize
        :returns: Normalized DataFrame
        """
        arr = (data.values - self._scaling_means) / self._scaling_stds
        normal_data = pd.DataFrame(arr, index=data.index, columns=data.columns)
        return normal_data.sort_index()
