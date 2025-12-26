__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime

from data.loader import load_fundamentals
from features.equities.data import EquityData
from features.equities.features import EquityFeatures


class Universe:
    """
    Class to manage a collection of tickers representing a universe.
    """

    def __init__(
        self,
        tickers: list[str] = None,
        data: dict = None,
        length: int = 60,
    ):
        """
        Initialize the Universe with equities and their features.
        :param tickers: List of ticker symbols, default None loads all available
        :param data: Dictionary containing loaded fundamentals data, default None
        :param length: Length of feature windows to aggregate, default 60
        """
        self.data = {}
        self.features = {}

        # Load fundamentals if not provided
        if data is None:
            print("Loading fundamentals data for universe...")
            data = load_fundamentals()

        # Filter tickers to the available list
        avail_tickers = set(data["shares"]["Ticker"].unique().tolist())
        if tickers is None:
            tickers = list(avail_tickers)
        else:
            tickers = [ticker for ticker in tickers if ticker in avail_tickers]

        # Load equity data and features for the tickers
        self.data = self._load_data(tickers, data)
        self.features = self._load_features(length)

    def _load_data(self, tickers: list[str], fund_data: dict) -> None:
        """
        Load equity data for each ticker in the list.
        :param tickers: List of ticker symbols to load
        :param fund_data: Fundamentals data dictionary
        :return: Data dictionary mapping index to EquityData objects
        """
        print("Loading universe of equities...")
        index = 0
        data = {}
        for ticker in tickers:
            print(f"Processing ticker {ticker} data...", end="\r")
            equity_data = EquityData(ticker, fund_data)
            if len(equity_data) == 0:
                continue
            data[index] = equity_data
            index += 1
        print(end="\n")
        print(f"Loaded {len(data)} equities into the universe.")
        return data

    def _load_features(self, length: int) -> dict:
        """
        Load equity features for all equities in the universe.
        :param length: Length of feature windows to aggregate
        :return: Features dictionary mapping index to EquityFeatures objects
        """
        print("Loading equity features for universe...")
        features = {}
        for index, equity in self.data.items():
            features[index] = EquityFeatures(equity, length)
        print(f"Loaded features for {len(self.features)} equities.")
        return features

    def size(self, date: datetime) -> int:
        """
        Size the number of equities in the universe on a given date.
        :param date: Date to size the universe as of the open
        :return: Number of equities in the universe on the given date
        """
        size = 0
        for equity in self.data.values():
            if date in equity.index:
                size += 1
        return size

    def __len__(self):
        """
        Get the number of tickers in the universe.
        :return: Number of tickers
        """
        return len(self.data)

    @property
    def index(self) -> list[int]:
        """
        Get the index for the equities in the universe.
        :return: List of equity indices
        """
        return list(self.data.keys())
