__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import torch
from datetime import datetime
from numpy import ndarray

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
        self._max_leverage = parameters["max_leverage"]
        self._min_scalar = parameters["min_scalar"]
        self._max_scalar = parameters["max_scalar"]

        self._covariance = Covariance(dataset, parameters)

    def predict(self, date: datetime, dataset: FeaturesDataset) -> dict:
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
            if date not in features_data.dates:
                continue
            window, _ = features_data[date]  # Get window and target using datetime
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
        covariance = self._covariance.calc_matrix(date, sigmas)
        indices = covariance.index.tolist()
        covariance = covariance.values
        mu = np.array([mus[idx] for idx in indices])
        mu = mu - np.mean(mu) # Debias expected returns

        # Mean-variance optimization minimizes return-variance difference
        inv_covariance = np.linalg.pinv(covariance)
        ones = np.ones(len(mu))
        lambda_ = (ones.T @ inv_covariance @ mu) / (ones.T @ inv_covariance @ ones)
        weights = inv_covariance @ (mu - lambda_ * ones)

        # Limit position sizes and remove dust positions for risk control
        min_weight = self._min_scalar / len(indices)
        max_weight = self._max_scalar / len(indices)
        weights = self._limit_leverage(weights)
        weights = self._limit_positions(weights, max_weight)
        weights = self._zero_positions(weights, min_weight)
        return {index: weights[i] for i, index in enumerate(indices)}

    def _limit_leverage(self, weights: ndarray) -> np.ndarray:
        """
        Limit total portfolio leverage to a maximum value.
        :param weights: Array of portfolio weights
        :param max_leverage: Maximum allowed leverage
        :return: Weights array with limited leverage
        """
        weights = np.array(weights)
        weight_sum = np.sum(np.abs(weights))
        if weight_sum > self._max_leverage:
            weights = self._max_leverage * weights / weight_sum
        return weights

    def _limit_positions(self, weights: ndarray, max_weight: float) -> np.ndarray:
        """
        Apply position size limits by clipping to constraints.
        :param weights: Array of portfolio weights
        :param max_weight: Maximum absolute position size
        :return: Constrained weights array with max leverage and position size
        """
        weights = np.array(weights)
        weight_sum = np.sum(np.abs(weights))

        # Iteratively rescale to enforce max weight constraint
        for _ in range(4):
            weight_delta = np.sum(np.maximum(np.abs(weights) - max_weight, 0))
            weight_sum_hat = weight_sum - weight_delta
            max_weight_hat = max_weight * weight_sum_hat / weight_sum
            weights = np.clip(weights, -max_weight_hat, max_weight_hat)
            weights = weights * weight_sum / weight_sum_hat
        return weights

    def _zero_positions(self, weights: ndarray, min_weight: float) -> np.ndarray:
        """
        Zero out small positions below a certain threshold.
        :param weights: Array of portfolio weights
        :param min_weight: Minimum absolute weight to keep
        :return: Weights array with dust positions set to zero
        """
        weights = np.array(weights)
        weight_sum0 = np.sum(np.abs(weights))
        weights[np.abs(weights) < min_weight] = 0.0

        # Rescale weights to maintain original leverage
        weight_sum1 = np.sum(np.abs(weights))
        if weight_sum1 > 0:
            weights = weights * (weight_sum0 / weight_sum1)
        return weights
