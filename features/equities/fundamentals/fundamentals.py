__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from features.equities.fundamentals.balances import get_balance
from features.equities.fundamentals.earnings import get_trailing_earnings
from features.equities.fundamentals.shares import get_shares_outstanding
from utils import (
    cast_str_date,
    convert_datetimeindex,
    get_last_date,
    list_dates,
    get_max_trading_date,
)


class FundamentalsData:
    """
    Class to hold and manage fundamental data for a specific ticker.
    """

    def __init__(self, ticker: str, fund_data: dict, max_date: datetime = None):
        """
        Initialize a fundamental dataset for a specific ticker.
        :param ticker: String ticker to download
        :param fund_data: Dictionary containing loaded fundamentals data
        :param max_date: Maximum date for fundamental data
        """
        self._TICKER = ticker
        self._dates = []
        self._min_date = datetime(1900, 1, 1)
        self._max_date = max_date

        self.data = self._load_data(fund_data)

        # Return if no data is available for the ticker
        if len(self._dates) == 0 or self._min_date >= datetime(2100, 1, 1):
            self._features = pd.DataFrame()
            return

        self._features = self._calc_features()

    def _load_data(self, fund_data: dict) -> dict:
        """
        Load the loaded fundamental data for the ticker.
        :param data: Dictionary containing loaded fundamentals data
        :return: Dictionary containing loaded fundamental data for this ticker
        """
        data = {}

        for concept, concept_data in fund_data.items():
            concept_ticker_data = concept_data[concept_data["Ticker"] == self._TICKER]
            data[concept] = concept_ticker_data

            # Use the publish date to avoid lookahead bias of released earnings
            dates_name = "Publish Date" if concept != "shares" else "Date"
            dates = concept_ticker_data[dates_name].apply(cast_str_date).tolist()
            
            # Filter dates by the max date if specified
            if self._max_date is not None:
                dates = [d for d in dates if d <= self._max_date]
            
            self._dates = sorted(set(self._dates).union(set(dates)))

            # Use the maximum of all minimum dates to ensure all sources have data
            concept_min_date = min(dates) if dates else datetime(2100, 1, 1)
            self._min_date = max(concept_min_date, self._min_date)

        return data

    def _calc_features(self):
        """
        Calculate features for the ticker for faster access.
        :return: DataFrame containing calculated features
        """
        max_date = max(self._dates) + timedelta(days=1)
        # Cap max_date to the last available trading date to prevent KeyError
        max_available = get_max_trading_date()
        max_date = min(max_date, max_available)
        dates = list_dates(self._min_date, max_date)
        dates = [date for date in dates if date >= self._min_date]

        # Calculate market capitalizations for use in other features
        features = pd.DataFrame(index=dates)
        features["MC"] = self._calc_market_caps(dates)
        self._features = features

        # Calculate all features and store in a DataFrame
        features["D/A"] = self._calc_debt_assets_ratio(dates)
        features["E/P"] = self._calc_earnings_price_ratio(dates)
        features["S/P"] = self._calc_sales_price_ratio(dates)
        features["CF/P"] = self._calc_cash_flow_price_ratio(dates)
        features["B/P"] = self._calc_book_price_ratio(dates)
        features["OM"] = self._calc_operating_margin(dates)
        features["ROE"] = self._calc_return_on_equity(dates)
        features["RG"] = self._calc_revenue_growth(dates)
        features["EG"] = self._calc_earnings_growth(dates)

        # Replace infinities and add a single NA indicator for trailing features
        features.replace([np.inf, -np.inf], None, inplace=True)
        trailing_cols = ["D/A", "E/P", "S/P", "CF/P", "B/P", "OM", "ROE", "RG", "EG"]
        features["FUND_NA"] = features[trailing_cols].isna().any(axis=1).astype(int)
        for feat in trailing_cols:
            features[feat] = pd.to_numeric(features[feat], errors="coerce").fillna(0)

        return features

    def _calc_market_caps(self, dates: list[datetime]):
        """
        Calculate market capitalizations for all dates for faster access.
        :param dates: Dates to calculate market caps for
        """
        ticker = yf.Ticker(self._TICKER)
        prices = ticker.history(period="max")
        prices.index = convert_datetimeindex(prices.index)
        shares = get_shares_outstanding(self.data["shares"], dates)
        return shares * prices["Close"]

    def _calc_debt_assets_ratio(self, dates: list[datetime]):
        """
        Calculate the debt-to-assets ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        debt = get_balance("Total Liabilities", self.data["balances"], dates)
        assets = get_balance("Total Assets", self.data["balances"], dates)
        return debt / assets

    def _calc_earnings_price_ratio(self, dates: list[datetime]):
        """
        Calculate the earnings-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        earnings = get_trailing_earnings("Net Income", self.data["incomes"], dates)
        price = self._features.loc[dates, "MC"]
        return earnings / price

    def _calc_sales_price_ratio(self, dates: list[datetime]):
        """
        Calculate the sales-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        sales = get_trailing_earnings("Revenue", self.data["incomes"], dates)
        price = self._features.loc[dates, "MC"]
        return sales / price

    def _calc_cash_flow_price_ratio(self, dates: list[datetime]):
        """
        Calculate the free cash flow-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        cash_flow = get_trailing_earnings(
            "Net Change in Cash", self.data["cashflows"], dates
        )
        price = self._features.loc[dates, "MC"]
        return cash_flow / price

    def _calc_book_price_ratio(self, dates: list[datetime]):
        """
        Calculate the book-to-price ratio for the given dates.
        :param dates: Dates to calculate the ratio for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        equity = get_balance("Total Equity", self.data["balances"], dates)
        price = self._features.loc[dates, "MC"]
        return equity / price

    def _calc_operating_margin(self, dates: list[datetime]):
        """
        Calculate the operating margin for the given dates.
        :param dates: Dates to calculate the margin for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        earnings = get_trailing_earnings(
            "Operating Income (Loss)", self.data["incomes"], dates
        )
        revenue = get_trailing_earnings("Revenue", self.data["incomes"], dates)
        return earnings / revenue

    def _calc_return_on_equity(self, dates: list[datetime]):
        """
        Calculate the return on equity (ROE) for the given dates.
        :param dates: Dates to calculate ROE for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        earnings = get_trailing_earnings("Net Income", self.data["incomes"], dates)
        equity = get_balance("Total Equity", self.data["balances"], dates)
        return earnings / equity

    def _calc_revenue_growth(self, dates: list[datetime]):
        """
        Calculate the year-over-year revenue growth for the given dates.
        :param dates: Dates to calculate revenue growth for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        dates0 = [date - timedelta(days=365) for date in dates]
        dates0 = [get_last_date(date) for date in dates0]
        revenue = get_trailing_earnings("Revenue", self.data["incomes"], dates)
        revenue0 = get_trailing_earnings("Revenue", self.data["incomes"], dates0)

        # Reindex revenue0 to align with current dates for vectorized calculation
        revenue0_aligned = pd.Series(revenue0.values, index=dates)
        return (revenue - revenue0_aligned) / revenue0_aligned

    def _calc_earnings_growth(self, dates: list[datetime]):
        """
        Calculate the year-over-year earnings growth for the given dates.
        :param dates: Dates to calculate earnings growth for
        """
        dates = [date for date in dates if self._is_date_valid(date)]
        dates0 = [date - timedelta(days=365) for date in dates]
        dates0 = [get_last_date(date) for date in dates0]
        earnings = get_trailing_earnings("Net Income", self.data["incomes"], dates)
        earnings0 = get_trailing_earnings("Net Income", self.data["incomes"], dates0)

        # Reindex earnings0 to align with current dates for vectorized calculation
        earnings0_aligned = pd.Series(earnings0.values, index=dates)
        return (earnings - earnings0_aligned) / earnings0_aligned

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
        return not pd.isna(self._features.loc[date, "MC"])

    def _get_market_cap(self, date: datetime) -> float:
        """
        Get the market capitalization for a specific date.
        :param date: Date to calculate market cap for
        :return: Market capitalization as a float
        """
        return self._features.loc[date, "MC"]

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

    def __len__(self) -> int:
        """
        Get the number of available feature dates.
        :return: Number of index dates in the features DataFrame
        """
        return len(self._features)
