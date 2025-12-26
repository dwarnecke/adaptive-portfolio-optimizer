__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import torch
from datetime import datetime
from torch import Tensor

from features.markets import data
from features.markets.regime_model import RegimeModel
from features.markets.features import MarketFeatures
from models.universe import Universe
from models.forward_model import ForwardModel
from models.covariance import Covariance


class PortfolioModel:
    """
    Portfolio optimization model that estimates performance and optimizes weights.
    Uses a transformer model for predictions and covariance-based optimization.
    """

    def __init__(
        self,
        universe: Universe,
        market: MarketFeatures,
        forward_model: ForwardModel,
        regime_model: RegimeModel,
        covariance: Covariance,
    ):
        """
        Initialize the portfolio optimization model.
        :param universe: Universe object containing EquityFeatures for each equity
        :param market: MarketFeatures object with market-level features
        :param forward_model: Trained ForwardModel for return and volatility prediction
        :param regime_model: RegimeModel for market regime analysis
        :param covariance: Covariance object for computing covariance matrices
        """
        self.universe = universe
        self.market = market
        self.forward_model = forward_model
        self.regime_model = regime_model
        self.covariance = covariance

    def estimate(self, date: datetime) -> dict:
        """
        Estimate expected returns and volatilities for each equity in the universe.
        :param date: Date to make predictions for
        :return: Tuple of (mus, sigmas) dictionaries mapping index to estimates
        """
        predictions = {}
        market_window = self.market[date]

        # Estimate forward returns and volatilities with the forward model
        for index in self.universe.index:
            prediction = self._estimate_index(date, index, market_window)
            if prediction is None:
                continue
            predictions[index] = prediction

        mus = {index: pred[0] for index, pred in predictions.items()}
        sigmas = {index: pred[1] for index, pred in predictions.items()}
        return mus, sigmas

    def weight(self, date: datetime, mus: dict, sigmas: dict) -> dict:
        """
        Compute optimal portfolio weights using mean-variance optimization.
        :param mus: Dictionary mapping equity index to expected returns
        :param sigmas: Dictionary mapping equity index to expected volatilities
        :return: Dictionary mapping equity index to optimal portfolio weights
        """
        weights = {}
        indices = list(mus.keys())

        # Compute weights using mean-variance optimization formula
        covariance = self.covariance.calc_matrix(date, sigmas)
        inv_covariance = np.linalg.pinv(covariance)
        mu_vector = np.array([mus[index] for index in indices])
        ones = np.ones(len(mu_vector))
        weights = (inv_covariance @ mu_vector) / (ones.T @ inv_covariance @ mu_vector)

        return {index: weights[i] for i, index in enumerate(indices)}
    
    def _estimate_index(
        self, date: datetime, index: int, market_window: Tensor
    ) -> tuple[float, float]:
        """
        Estimate expected return and volatility for a single equity index.
        :param date: Date to make prediction for
        :param index: Equity index in the universe
        :return: Tuple of (return, volatility) estimates
        """
        equity_window = self.universe.features[index][date]
        if equity_window is None:
            return None

        window = torch.cat([equity_window, market_window], dim=-1)
        window = window.unsqueeze(0)  # Add batch dimension
        prediction = self.forward_model(window)
        return (prediction[0, 0].item(), prediction[0, 1].item())
