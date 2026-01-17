__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
from datetime import datetime, timedelta

from config.hyperparameters import HYPERPARAMETERS
from features.dataset import FeaturesDataset
from models.model import ForwardModel
from portfolio.model import PortfolioModel
from portfolio.positions.position import Position
from utils import list_dates, get_last_date


class Portfolio:
    """
    Class to manage portfolio weights.
    """

    def __init__(
        self,
        dataset: FeaturesDataset,
        forward_model: ForwardModel,
        capital: float = 100_000_000,
        parameters: dict = HYPERPARAMETERS["portfolio"],
    ):
        """
        Initialize the Portfolio object with given weights.
        :param dataset: FeaturesDataset with feature windows and equity data
        :param forward_model: ForwardModel object for estimating returns and volatilities
        :param capital: Total capital for the portfolio, default 100,000,000
        :param parameters: Dictionary of portfolio parameters, default config
        """
        self._dataset = dataset
        self._model = PortfolioModel(dataset, forward_model, parameters)

        self._parameters = parameters
        self._TRANSACTION_BPS = (
            parameters["slippage_bps"] + parameters["commission_bps"]
        )

        self._init_capital = capital
        self.capital = {}
        self.history = {}
        self.statistics = {}

        # Initialize positions for each equity in the dataset
        self._positions = {}
        for index in dataset.index:
            self._positions[index] = Position(dataset.data[index])

    def simulate(self, start_date: datetime, end_date: datetime):
        """
        Simulate portfolio performance over a date range.
        :param start_date: Start date for simulation, inclusive
        :param end_date: End date for simulation, exclusive
        :return: Dictionary of portfolio statistics
        """
        dates = list_dates(start_date, end_date)
        prev_date = get_last_date(start_date - timedelta(days=1))
        self.capital[prev_date] = self._init_capital
        self.history[prev_date] = self._init_capital

        weights = {i: 0.0 for i in self._dataset.index}
        for date in dates:
            capital = self.capital[prev_date]
            capital += self._rebalance(weights, date, capital)
            self.capital[date] = capital
            self.history[date] = self._value(date, capital)

            # Estimate weights for trading on the next date using the close
            mus, sigmas = self._model.predict(date)
            weights = self._model.weigh(date, mus, sigmas)
            prev_date = date

        # Calculate portfolio statistics after simulation
        self.statistics["drawdown"] = self._calc_drawdown()
        self.statistics["sharpe"] = self._calc_sharpe()
        self.statistics["return"] = self._calc_return()

        return self.statistics

    def _rebalance(self, weights: dict, date: datetime, capital: float) -> float:
        """
        Rebalance the portfolio to target weights.
        :param weights: Dictionary of target weights for each equity index
        :param date: Date to rebalance the portfolio on the open
        :param capital: Capital available for rebalancing
        :return: Capital change from rebalancing
        """
        delta = 0
        weights0 = self._weigh(date, capital, close=False)
        total_value = self._value(date, capital, close=False)

        for index, position in self._positions.items():
            weight0 = weights0[index]
            weight = weights[index] if index in weights else 0.0

            # Exit positions with no target weight
            if weight == 0:
                delta += self._trade(position, 0, date)
                continue

            # Rebalance only if the change greatly exceeds transaction costs
            weight_delta = abs(weight - weight0)
            trade_value = weight_delta * total_value
            transaction_cost = self._TRANSACTION_BPS / 10_000 * trade_value
            threshold = 4 * transaction_cost / total_value
            if weight_delta < threshold:
                continue

            target = weight * total_value
            delta += self._trade(position, target, date)

        return delta

    def _weigh(self, date: datetime, capital: float, close: bool = True) -> dict:
        """
        Weight the current positions based on their values.
        :param date: Date to calculate position weights
        :param capital: Capital to include in the valuation
        :param close: Whether to value at close price, default True
        :return: Dictionary of position weights by equity index
        """
        total = self._value(date, capital, close=close)
        weights = {}
        for index, position in self._positions.items():
            value = position.value(date, close=close)
            weights[index] = value / total if total > 0 else 0.0
        return weights

    def _value(self, date: datetime, capital: float, close: bool = True) -> float:
        """
        Value the total of the portfolio positions on a given date.
        :param date: Date of the portfolio valuation
        :param capital: Capital to include in the valuation
        :param close: Whether to value at close price, default True
        :return: Total portfolio value as of the close of the date
        """
        value = 0
        for position in self._positions.values():
            value += position.value(date, close=close)
        return value + capital

    def _trade(self, position: Position, target: float, date: datetime) -> float:
        """
        Trade a position to reach a target value.
        :param position: Position object to trade
        :param target: Target value for the position
        :param date: Date to execute the trade on the open
        :return: Change in capital from the trade
        """
        SLIPPAGE_BPS = self._parameters["slippage_bps"]
        COMMISSION_BPS = self._parameters["commission_bps"]
        delta = position.rebalance(target, date, SLIPPAGE_BPS)
        commission = COMMISSION_BPS / 10_000 * abs(delta)
        return delta - commission

    def _calc_drawdown(self) -> dict:
        """
        Calculate the percentage drawdown for the portfolio over time.
        :return: Dictionary mapping date to drawdown percentage
        """
        drawdown = {}
        max_value = float("-inf")
        for date, value in self.history.items():
            if value > max_value:
                max_value = value
            drawdown[date] = (max_value - value) / max_value
        return drawdown

    def _calc_sharpe(self) -> dict:
        """
        Calculate the Sharpe ratio for the portfolio over time.
        :return: Dictionary mapping date to Sharpe ratio
        """
        sharpe = {}
        returns = []
        free_mu = self._parameters["rate0"] / 252
        values = list(self.history.values())
        return_sigma = 0
        for i, date in enumerate(self.history.keys()):
            if i > 0:
                returns.append((values[i] - values[i - 1]) / values[i - 1])
                return_mu = np.mean(returns)
                return_sigma = np.std(returns)
            if return_sigma > 0:
                sharpe[date] = (return_mu - free_mu) / return_sigma * np.sqrt(252)
        return sharpe

    def _calc_return(self) -> dict:
        """
        Calculate the cumulative return for the portfolio over time.
        :return: Dictionary mapping date to cumulative return
        """
        returns = {}
        initial_value = None
        for date, value in self.history.items():
            if initial_value is None:
                initial_value = value
            returns[date] = (value - initial_value) / initial_value
        return returns
