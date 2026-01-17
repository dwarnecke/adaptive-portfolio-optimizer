__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import torch
from datetime import datetime

from config.hyperparameters import HYPERPARAMETERS
from features.dataset import FeaturesDataset
from models.model import ForwardModel
from portfolio.covariance import Covariance


class PortfolioModel:
    """
    Portfolio optimization model that estimates performance and optimizes weights.
    Uses a transformer model for predictions and covariance-based optimization.
    """

    def __init__(
        self,
        dataset: FeaturesDataset,
        forward_model: ForwardModel,
        parameters: dict = HYPERPARAMETERS["portfolio"],
    ):
        """
        Initialize the portfolio optimization model.
        :param dataset: FeaturesDataset with feature windows and equity data
        :param forward_model: Trained ForwardModel for return and volatility prediction
        :param parameters: Dictionary of portfolio parameters, default config
        """
        self._dataset = dataset
        self._forward_model = forward_model
        self._parameters = parameters

        self._covariance = Covariance(dataset, parameters)

    def predict(self, date: datetime, dataset) -> dict:
        """
        Predict expected returns and volatilities for each equity in the universe.
        Uses batched inference for efficiency.
        :param date: Date to make predictions for
        :param dataset: FeaturesDataset with feature windows
        :return: Tuple of dictionaries (mus, sigmas) mapping index to estimates
        """
        # Collect all equity windows into a batch
        batch_windows = []
        indices = []
        for index in self._dataset.index:
            features_data = dataset.data.get(index)
            if features_data is None:
                continue
            result = features_data.get_item_from_date(date)
            if result is None:
                continue
            window, _ = result  # Window already includes both equity and market features
            batch_windows.append(window)
            indices.append(index)
        
        if len(batch_windows) == 0:
            return {}, {}

        # Batch all windows and make single forward pass to parallelize
        batch_tensor = torch.stack(batch_windows)
        preds = self._forward_model(batch_tensor)
        mus = {index: preds[i, 0].item() for i, index in enumerate(indices)}
        sigmas = {index: preds[i, 1].item() for i, index in enumerate(indices)}
        return mus, sigmas

    def weigh(self, date: datetime, mus: dict, sigmas: dict) -> dict:
        """
        Weigh portfolio positions using mean-variance optimization.
        :param date: Date to calculate weights for
        :param mus: Dictionary mapping equity index to expected returns
        :param sigmas: Dictionary mapping equity index to expected volatilities
        :param parameters: Dictionary of portfolio parameters
        :return: Dictionary mapping equity index to optimal equity weights
        """
        indices = list(mus.keys())
        if len(indices) == 0:
            return {}

        covariance = self._covariance.calc_matrix(date, sigmas)
        mu_vector = np.array([mus[index] for index in indices])

        # Risk free asset ensures non-singularity of the weights
        mu_vector = np.append(mu_vector, 0.0)
        covariance = np.pad(covariance, ((0, 1), (0, 1)), constant_values=0.0)
        covariance[-1, -1] = 1e-8

        # Mean-variance optimization minimizes return-variance difference
        inv_covariance = np.linalg.pinv(covariance)
        ones = np.ones(len(mu_vector))
        weights = (inv_covariance @ mu_vector) / (ones.T @ inv_covariance @ mu_vector)
        weights = weights[:-1]  # Remove risk free asset weight

        # Filter out dust positions from the weights
        threshold = self._parameters["dust_threshold"]
        total0 = sum(weight for weight in weights)
        weights = [weight if abs(weight) > threshold else 0.0 for weight in weights]
        total1 = sum(weight for weight in weights)
        scale = total0 / total1 if total1 != 0 else 0.0
        weights = [weight * scale for weight in weights]

        return {index: weights[i] for i, index in enumerate(indices)}
