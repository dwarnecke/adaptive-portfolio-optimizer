__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime

from features.equities.data import EquityData
from positions.entry import Entry


class Position:
    """
    Class to hold and manage portfolio position data.
    """

    def __init__(self, data: EquityData):
        """
        Initialize the Position object with the equity data.
        :param data: EquityData object containing equity data
        """
        self.data = data
        self.entries = {}

    def rebalance(self, value: float, date: datetime) -> float:
        """
        Rebalance the position to a target weight in the open.
        :param value: Target value for the position
        :param date: Date to rebalance the position
        :return: Value of the trade executed to rebalance
        """
        value0 = self.value(date, close=False)
        price = self.data.get_open(date)
        shares = round((value - value0) / price)
        return self._trade(date, shares)

    def _trade(self, date: datetime, shares: int) -> float:
        """
        Execute a trade for the position in the open.
        :param date: Date to execute the trade
        :param shares: Number of shares to trade, positive for buy, negative for sell
        :return: Capital change of the trade
        """
        delta = 0

        # Close opposite positions before opening new ones
        long = shares > 0
        short = shares < 0
        if (short and self.long) or (long and self.short):
            # Exit positions in first in, last out order for taxes
            dates = sorted(list(self.entries.keys()), reverse=True)
            for entry_date in dates:
                entry = self.entries[entry_date]
                value, shares = self._close(entry, shares, date)
                delta += value
                if shares == 0:
                    break

        # Open new position if shares remain
        if shares != 0:
            delta += self._open(date, shares)

        return delta

    def value(self, date: datetime, close: bool = True) -> float:
        """
        Value the position on a given date.
        :param date: Date to value the position as of the close
        :param close: Whether to value at close price, default True
        :return: Total position value on the given date
        """
        value = sum(entry.value(date, close=close) for entry in self.entries.values())
        return value

    def _open(self, date: datetime, shares: int):
        """
        Open a new position entry for a specific date.
        :param date: Date to open the position as of the open
        :param shares: Number of shares to open, positive for buy, negative for sell
        :return: Capital change of the opened position
        """
        entry = Entry(self.data, shares, date)
        self.entries[date] = entry
        return -entry.cost

    def _close(self, entry: Entry, shares: int, date: datetime) -> tuple[int, float]:
        """
        Close a portion of an entry at a given price.
        :param date: Date to close the entry as of the open
        :param shares: Number of shares to close
        :return: Capital change of the closed position, shares remaining
        """
        value, shares = entry.close(shares, date)
        if not entry.open:
            del self.entries[entry.date]
        return value, shares

    @property
    def long(self) -> bool:
        """
        Check if the position is a long position.
        :return: True if long position, False if short
        """
        if not self.entries:
            return False
        min_shares = min(entry.shares for entry in self.entries.values())
        return min_shares > 0

    @property
    def short(self) -> bool:
        """
        Check if the position is a short position.
        :return: True if short position, False if long
        """
        if not self.entries:
            return False
        max_shares = max(entry.shares for entry in self.entries.values())
        return max_shares < 0
