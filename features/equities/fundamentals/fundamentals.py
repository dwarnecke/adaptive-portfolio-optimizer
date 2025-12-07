__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
import simfin as sf
import yfinance as yf
from datetime import datetime, timedelta

from features.equities.fundamentals.balances import get_balance
from features.equities.fundamentals.earnings import get_trailing_earnings
from features.equities.fundamentals.shares import get_shares_outstanding
from utils.dates import cast_str_date, convert_datetimeindex, list_trading_dates
from utils.keys import get_api_key

# Set SimFin configuration
sf.set_api_key(get_api_key("simfin"))
sf.set_data_dir("data/simfin/")


class FundamentalsData:
    """
    Class to hold and manage fundamental data for a specific ticker.

    Date data is what is known up to and including that date at close.
    """

    def __init__(self, ticker: str, data: dict):
        """
        Initialize a fundamental dataset for a specific ticker.
        :param ticker: String ticker to download
        :param data: Dictionary containing loaded fundamentals data
        """
        self._TICKER = ticker
        self._dates = []
        self._min_date = None

        # Load data and calculate fundamental features for the ticker
        self._data = {}
        self._load_data(data)
        self._features = pd.DataFrame()
        self._calc_features()

    @property
    def ticker(self) -> str:
        """
        Get the ticker symbol for this data.
        :return: Ticker symbol as a string
        """
        return self._TICKER

    @property
    def data(self) -> dict:
        """
        Get the loaded fundamentals data dictionary.
        :return: Dictionary containing fundamentals data DataFrames
        """
        return self._data.copy()

    @property
    def features(self) -> pd.DataFrame:
        """
        Get the calculated features DataFrame.
        :return: DataFrame containing calculated features with NA indicators
        """
        # Drop market capitalization as it provides no predictive power
        features = self._features.copy()
        features = features.drop(columns=["MC"])
        return features

    def _load_data(self, data: dict):
        """
        Load the loaded fundamental data for the ticker.
        :param data: Dictionary containing loaded fundamentals data
        """
        for concept, concept_data in data.items():
            statement_data = concept_data[concept_data["Ticker"] == self._TICKER]
            self._data[concept] = statement_data
            dates_name = "Report Date" if concept != "shares" else "Date"
            dates = statement_data[dates_name].apply(cast_str_date)
            # Use the maximum of all minimum dates to ensure all data sources have data
            if self._min_date is None:
                self._min_date = min(dates)
            else:
                self._min_date = max(min(dates), self._min_date)
            self._dates = sorted(set(self._dates).union(set(dates)))

    def _calc_features(self, dates: list[datetime] = None):
        """
        Calculate features for the ticker for faster access.
        :param dates: Dates to calculate features for, default None calculates all
        """
        # Calculate to the end of available reports if none provided
        if dates is None:
            max_date = max(self._dates)
            dates = list_trading_dates(self._min_date, max_date)
        dates = [date for date in dates if date >= self._min_date]

        # Calculate fundamental features for the dates as of the end of the day
        if self._features.empty:
            self._features = pd.DataFrame(index=dates)
        self._calc_market_caps(dates)
        self._calc_debt_assets_ratio(dates)
        self._calc_earnings_price_ratio(dates)
        self._calc_sales_price_ratio(dates)
        self._calc_cash_flow_price_ratio(dates)
        self._calc_book_price_ratio(dates)
        self._calc_operating_margin(dates)
        self._calc_return_on_equity(dates)
        self._calc_revenue_growth(dates)
        self._calc_earnings_growth(dates)

        # Add indicator features for features that can be NA
        self._features.replace([np.inf, -np.inf, np.nan], None, inplace=True)
        na_features = ["OM", "ROE", "RG", "EG"]
        for feature in na_features:
            self._features[f"{feature}_NA"] = self._features[feature].isna().astype(int)
            # Zero out NA values in the original feature
            self._features[feature] = self._features[feature].fillna(0)

    def _calc_market_caps(self, dates: list[datetime]):
        """
        Calculate market capitalizations for all dates for faster access.
        :param dates: Dates to calculate market caps for
        """
        ticker = yf.Ticker(self._TICKER)
        prices = ticker.history(period="max")
        prices.index = convert_datetimeindex(prices.index)
        shares = get_shares_outstanding(self._data["shares"], dates)
        self._features["MC"] = shares * prices["Close"]

    def _calc_debt_assets_ratio(self, dates: list[datetime]):
        """
        Calculate the debt-to-assets ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        valid_dates = [date for date in dates if self._is_date_valid(date)]
        debt = get_balance("Total Liabilities", self._data["balances"], valid_dates)
        assets = get_balance("Total Assets", self._data["balances"], valid_dates)
        self._features.loc[valid_dates, "D/A"] = debt / assets

    def _calc_earnings_price_ratio(self, dates: list[datetime]):
        """
        Calculate the earnings-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        valid_dates = [date for date in dates if self._is_date_valid(date)]
        earnings = get_trailing_earnings(
            "Net Income", self._data["incomes"], valid_dates
        )
        price = self._features.loc[valid_dates, "MC"]
        self._features.loc[valid_dates, "E/P"] = earnings / price

    def _calc_sales_price_ratio(self, dates: list[datetime]):
        """
        Calculate the sales-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        valid_dates = [date for date in dates if self._is_date_valid(date)]
        sales = get_trailing_earnings("Revenue", self._data["incomes"], valid_dates)
        price = self._features.loc[valid_dates, "MC"]
        self._features.loc[valid_dates, "S/P"] = sales / price

    def _calc_cash_flow_price_ratio(self, dates: list[datetime]):
        """
        Calculate the free cash flow-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        valid_dates = [date for date in dates if self._is_date_valid(date)]
        cash_flow = get_trailing_earnings(
            "Net Change in Cash", self._data["cashflows"], valid_dates
        )
        price = self._features.loc[valid_dates, "MC"]
        self._features.loc[valid_dates, "CF/P"] = cash_flow / price

    def _calc_book_price_ratio(self, dates: list[datetime]):
        """
        Calculate the book-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        valid_dates = [date for date in dates if self._is_date_valid(date)]
        equity = get_balance("Total Equity", self._data["balances"], valid_dates)
        price = self._features.loc[valid_dates, "MC"]
        self._features.loc[valid_dates, "B/P"] = equity / price

    def _calc_operating_margin(self, dates: list[datetime]):
        """
        Calculate the operating margin for the given dates.
        :param dates: Dates to calculate the margin for
        """
        valid_dates = [date for date in dates if self._is_date_valid(date)]
        earnings = get_trailing_earnings(
            "Operating Income (Loss)",
            self._data["incomes"],
            valid_dates,
        )
        revenue = get_trailing_earnings("Revenue", self._data["incomes"], valid_dates)
        self._features.loc[valid_dates, "OM"] = earnings / revenue

    def _calc_return_on_equity(self, dates: list[datetime]):
        """
        Calculate the return on equity (ROE) for the given dates.
        :param dates: Dates to calculate ROE for
        """
        valid_dates = [date for date in dates if self._is_date_valid(date)]
        earnings = get_trailing_earnings(
            "Net Income", self._data["incomes"], valid_dates
        )
        equity = get_balance("Total Equity", self._data["balances"], valid_dates)
        self._features.loc[valid_dates, "ROE"] = earnings / equity

    def _calc_revenue_growth(self, dates: list[datetime]):
        """
        Calculate the year-over-year revenue growth for the given dates.
        :param dates: Dates to calculate revenue growth for
        """
        dates0 = [date - timedelta(days=365) for date in dates]
        valid_pairs = [
            (date, date0)
            for date, date0 in zip(dates, dates0)
            if self._is_date_valid(date0)
        ]
        if not valid_pairs:
            return
        valid_dates = [pair[0] for pair in valid_pairs]
        valid_dates0 = [pair[1] for pair in valid_pairs]

        revenue = get_trailing_earnings("Revenue", self._data["incomes"], valid_dates)
        revenue0 = get_trailing_earnings("Revenue", self._data["incomes"], valid_dates0)
        self._features.loc[valid_dates, "RG"] = (revenue - revenue0) / revenue0

    def _calc_earnings_growth(self, dates: list[datetime]):
        """
        Calculate the year-over-year earnings growth for the given dates.
        :param dates: Dates to calculate earnings growth for
        """
        dates0 = [date - timedelta(days=365) for date in dates]
        valid_pairs = [
            (date, date0)
            for date, date0 in zip(dates, dates0)
            if self._is_date_valid(date0)
        ]
        if not valid_pairs:
            return
        valid_dates = [pair[0] for pair in valid_pairs]
        valid_dates0 = [pair[1] for pair in valid_pairs]

        earnings = get_trailing_earnings(
            "Net Income", self._data["incomes"], valid_dates
        )
        earnings0 = get_trailing_earnings(
            "Net Income", self._data["incomes"], valid_dates0
        )
        self._features.loc[valid_dates, "EG"] = (earnings - earnings0) / earnings0

    def _is_date_valid(self, date: datetime) -> bool:
        """
        Check if a date is valid for calculations (after min date).
        :param date: Date to check
        :return: True if valid, False otherwise
        """
        if date < self._min_date:
            return False
        if date not in self._features.index:
            return False
        if "MC" in self._features.columns:
            return not pd.isna(self._features.loc[date, "MC"])
        return True

    def _get_market_cap(self, date: datetime) -> float:
        """
        Get the market capitalization for a specific date.
        :param date: Date to calculate market cap for
        :return: Market capitalization as a float
        """
        return self._features.loc[date, "MC"]
