__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime, timedelta

from models.portfolio_model import PortfolioModel
from models.universe import Universe
from positions.position import Position
from utils.dates import list_dates, get_last_date

TRANSACTION_BPS = 15


class Portfolio:
    """
    Class to manage portfolio weights.
    """

    def __init__(
        self,
        universe: Universe,
        model: PortfolioModel,
        capital: float = 10_000_000,
    ):
        """
        Initialize the Portfolio object with given weights.
        :param universe: Universe object containing equity data
        :param model: PortfolioModel object for weighting
        :param capital: Total capital for the portfolio, default 10,000,000
        """
        self.universe = universe
        self.model = model

        self._initial_capital = capital
        self.capital = {}
        self.history = {}

        # Initialize positions for each equity in the universe
        self.positions = {}
        for index in universe.data.keys():
            self.positions[index] = Position(universe.data[index])

    def simulate(self, start_date: datetime, end_date: datetime):
        """
        Simulate portfolio performance over a date range.
        :param start_date: Start date for simulation, inclusive
        :param end_date: End date for simulation, exclusive
        """
        dates = list_dates(start_date, end_date)
        prev_date = get_last_date(start_date - timedelta(days=1))
        self.capital[prev_date] = self._initial_capital
        self.history[prev_date] = self._initial_capital

        weights = {i: 0.0 for i in self.universe.index}
        for date in dates:
            capital = self.capital[prev_date]
            capital += self._rebalance(weights, date, capital)
            self.capital[date] = capital
            self.history[date] = self._value(date, capital)

            # Estimate weights for trading on the next date using the close
            mus, sigmas = self.model.estimate(date)
            weights = self.model.weight(mus, sigmas)
            prev_date = date

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
        total_equities = self.universe.size(date)

        for index, position in self.positions.items():
            weight0 = weights0[index]
            weight = weights[index] if index in weights else 0.0

            # Exit positions with no target weight
            if weight == 0:
                delta += position.rebalance(0, date)
                continue

            # Rebalance only if the change greatly exceeds transaction costs
            weigh_delta = abs(weight - weight0)
            trade_value = weigh_delta * total_value
            transaction_cost = TRANSACTION_BPS / 10_000 * trade_value
            threshold = max(4 * transaction_cost / total_value, 1 / total_equities / 10)
            if weigh_delta < threshold:
                continue

            target = weight * total_value
            delta += position.rebalance(target, date)

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
        for index, position in self.positions.items():
            value = position.value(date, close=close)
            weights[index] = value / total if total > 0 else 0.0
        return weights

    def _value(self, date: datetime, capital: float, close: bool = True) -> float:
        """
        Value the total of the portfolio positions on a given date.
        :param date: Date of the portfolio valuation
        :param capital: Captial to include in the valuation
        :param close: Whether to value at close price, default True
        :return: Total portfolio value as of the close of the date
        """
        value = 0
        for position in self.positions.values():
            value += position.value(date, close=close)
        return value + capital
