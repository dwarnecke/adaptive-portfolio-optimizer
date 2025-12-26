__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame

from models.universe import Universe


class Covariance:
    """
    Class to hold and manage covariance data between an equity universe.
    """

    def __init__(
        self,
        universe: Universe,
        alpha: float = 2**-1,
        length: int = 120,
    ):
        """
        Initialize the Covariance object with an equity universe and shrinkage factor.
        :param universe: List of TechnicalsData objects representing the equity universe
        :param alpha: Shrinkage factor to minimize non-diagonal elements, default 0.5
        :param length: Length to calculate rolling covariance over, default 120
        """
        # Alpha must be between 0 and 1
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")

        self.universe = universe
        self._alpha = alpha
        self._length = length
        self._log_returns = self._calc_log_daily_returns()

    def calc_matrix(self, date: datetime, sigmas: dict) -> DataFrame:
        """
        Calculate the covariance matrix for the equity universe as of a specific date.
        :param date: Date to calculate covariance matrix for
        :param sigmas: Dictionary mapping index to predicted volatility
        :return: DataFrame containing covariance matrix
        """
        returns = self._log_returns[self._log_returns.index <= date]
        returns = returns.tail(self._length)
        returns = returns.dropna(axis=1)
        covariance = returns.cov()

        # Shrink the non-diagonal elements for noise reduction
        covariance = self._shrink_covariance(covariance)

        # Convert the covariance matrix to correlation for scaling
        correlation = self._convert_correlation(covariance)

        # Scale the correlation matrix with forward volatilities for future covariance
        covariance = self._scale_correlations(correlation, sigmas).values

        return covariance

    def _calc_log_daily_returns(self) -> DataFrame:
        """
        Calculate the log daily returns for the universe of tickers.
        :return: DataFrame containing log daily returns for each ticker
        """
        returns = {}
        for index, equity in self.universe.data.items():
            log_returns = equity.technicals.data["Log Close"].diff()
            returns[index] = log_returns
        return pd.DataFrame(returns).sort_index(ascending=True)

    def _shrink_covariance(self, covariance: DataFrame) -> DataFrame:
        """
        Shrink the non-diagonal elements of the covariance matrix to improve stability.
        :param covariance: DataFrame containing the covariance matrix
        :return: Shrunk covariance matrix as a DataFrame
        """
        diagonal = pd.DataFrame(
            np.diag(np.diag(covariance.values)),
            index=covariance.index,
            columns=covariance.columns,
        )
        covariance = self._alpha * diagonal + (1 - self._alpha) * covariance
        return covariance

    def _convert_correlation(self, covariance: DataFrame) -> DataFrame:
        """
        Convert the covariance matrix to a correlation matrix.
        :param covariance: DataFrame containing the covariance matrix
        :return: Correlation matrix as a DataFrame
        """
        deviations = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(deviations, deviations)
        correlation[~np.isfinite(correlation)] = 0.0
        return pd.DataFrame(
            correlation, index=covariance.index, columns=covariance.columns
        )

    def _scale_correlations(self, correlations: DataFrame, sigmas: dict) -> DataFrame:
        """
        Scale a correlation matrix by predicted volatilities to obtain covariance.
        :param correlations: DataFrame containing the correlation matrix
        :param sigmas: Dictionary mapping index to predicted, forward volatility
        :return: Covariance matrix as a DataFrame
        """
        deviations = pd.DataFrame(
            np.diag([sigmas[index] for index in correlations.index]),
            index=correlations.index,
            columns=correlations.columns,
        )
        covariance = deviations @ correlations @ deviations
        return covariance
