__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime
from features.data import FeaturesData


class Entry:
    """
    Class to hold and manage entry data for a position.
    """

    def __init__(self, data: FeaturesData, shares: int, date: datetime, slippage: float):
        """
        Initialize the Entry object with entry details.
        :param data: FeaturesData object containing equity data
        :param shares: Number of shares entered
        :param date: Date of the entry
        :param slippage: Slippage in basis points to apply to cost basis
        """
        self.data = data
        self.shares = shares
        self.date = date
        price = data.get_data("Open", date)

        # Slippage adjustment accounts for trade influence
        slippage_rate = slippage / 10000
        slippage_multiplier = 1 + (slippage_rate if shares > 0 else -slippage_rate)
        self._cost = price * slippage_multiplier

    def close(self, shares: int, date: datetime) -> tuple[int, float]:
        """
        Close a portion of the entry at a given price.
        :param date: Date to close the entry as of the open
        :param shares: Number of shares to close
        :return: Capital change of the closed position, shares remaining
        """
        # Close only up to the number of shares held
        if abs(shares) > abs(self.shares):
            close_shares = self.shares
        else:
            close_shares = -shares

        value = self.value(date, close_shares, close=False)
        self.shares -= close_shares
        shares += close_shares
        return value, shares

    def value(self, date: datetime, shares: int = None, close: bool = True) -> float:
        """
        Calculate the value of the entry at a given price.
        :param date: Date to value the entry as of the close
        :param shares: Number of shares to value, default all shares
        :param close: Whether to use close price, default True
        :return: Capital change of the opened position
        """
        if shares is None:
            shares = self.shares
        field = "Close" if close else "Open"
        price = self.data.get_data(field, date)
        return shares * price

    @property
    def long(self) -> bool:
        """
        Check if the entry is a long position.
        :return: True if long position, False if short
        """
        return self.shares > 0

    @property
    def short(self) -> bool:
        """
        Check if the entry is a short position.
        :return: True if short position, False if long
        """
        return self.shares < 0

    @property
    def open(self) -> bool:
        """
        Check if the entry is still open.
        :return: True if entry has shares remaining, False otherwise
        """
        return self.shares != 0

    @property
    def cost(self) -> float:
        """
        Get the total cost basis of the entry.
        :return: Total cost of the entry
        """
        return self._cost * self.shares
