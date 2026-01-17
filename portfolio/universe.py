__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime

from config.hyperparameters import HYPERPARAMETERS
from data.loader import load_fundamentals
from features.equities.data import EquityData


class Universe:
    """
    Class to manage a collection of tickers representing a universe.
    """

    def __init__(
        self,
        tickers: list[str] = None,
        fund_data: dict = None,
        parameters=HYPERPARAMETERS["features"],
        max_date: datetime = None,
    ):
        """
        Initialize the Universe with equities and their features.
        :param tickers: List of ticker symbols, default None loads all available
        :param data: Dictionary containing loaded fundamentals data, default None
        :param parameters: Dictionary of feature parameters, defaults to config
        :param max_date: Maximum date for data, default None
        """
        if fund_data is None:
            print("Loading fundamentals data...")
            fund_data = load_fundamentals()

        # Filter tickers to the available list as not all tickers have fundamentals
        avail_tickers = set(fund_data["shares"]["Ticker"].unique().tolist())
        if tickers is None:
            tickers = list(avail_tickers)
        else:
            tickers = [ticker for ticker in tickers if ticker in avail_tickers]

        self.data = self._load_data(tickers, fund_data, max_date)

    @classmethod
    def from_dataset(cls, dataset):
        """
        Create a Universe from a FeaturesDataset.
        :param dataset: FeaturesDataset instance with equity_data
        :return: Universe instance
        """
        print(f"\nCreating Universe from dataset with {len(dataset)} tickers...")
        instance = cls.__new__(cls)
        instance.data = dataset.equity_data
        instance.features = None  # Features are now part of FeaturesData, not separate
        print(f"  Universe created with {len(instance.data)} equities")
        return instance

    def _load_data(
        self, tickers: list[str], fund_data: dict, max_date: datetime = None
    ) -> None:
        """
        Load equity data for each ticker in the list.
        :param tickers: List of ticker symbols to load
        :param fund_data: Fundamentals data dictionary
        :param max_date: Maximum date for data
        :return: Data dictionary mapping index to EquityData objects
        """
        print("Loading universe of equities...")
        index = 0
        data = {}
        for ticker in tickers:
            print(f"Processing ticker {ticker} data...", end="\r")
            equity_data = EquityData(ticker, fund_data, max_date=max_date)
            if len(equity_data) == 0:
                continue
            data[index] = equity_data
            index += 1
        print(end="\n")
        print(f"Loaded data for {len(data)} equities")
        return data

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
