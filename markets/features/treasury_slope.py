__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
from datetime import datetime
from utils.dates import list_trading_dates

# FRED CSV URLs for 2Y and 10Y Treasury yields (daily, no API key required)
FRED_2Y_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"
FRED_10Y_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"


def calc_2y10y_slopes(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Calculate the 2-year and 10-year yield slopes for dates.
    :param start_date: Start datetime for the date range, inclusive
    :param end_date: End datetime for the date range, exclusive
    :returns: Series of 2-year and 10-year slopes indexed by date
    """
    yields_2y, yields_10y = load_treasury_yields(start_date, end_date)
    yields = yields_2y.join(yields_10y, lsuffix="_2y", rsuffix="_10y", how="inner")
    yields["spread"] = yields["DGS10"] - yields["DGS2"]
    slopes = yields["spread"]
    return slopes


def load_treasury_yields(start_date: datetime, end_date: datetime):
    """
    Load the 2-year and 10-year treasury yield rates for dates.
    :param start_date: Start datetime for the range, inclusive
    :param end_date: End datetime for the range, exclusive
    :returns: 2-year and 10-year yield rates
    """
    yields_2y = pd.read_csv(FRED_2Y_URL, parse_dates=["observation_date"])
    yields_10y = pd.read_csv(FRED_10Y_URL, parse_dates=["observation_date"])

    # Reindex to trading dates and forward-fill missing values
    yields_2y.index = pd.DatetimeIndex(yields_2y["observation_date"])
    yields_10y.index = pd.DatetimeIndex(yields_10y["observation_date"])
    dates = list_trading_dates(start_date, end_date)
    dates = pd.to_datetime(dates)
    yields_2y = yields_2y.reindex(dates).ffill()
    yields_10y = yields_10y.reindex(dates).ffill()

    return yields_2y, yields_10y
