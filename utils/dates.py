__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import yfinance as yf
from datetime import datetime, timedelta
from pandas import Series, DatetimeIndex
from zoneinfo import ZoneInfo

HEADERS = {"User-Agent": "Dylan Warnecke <dylan.warnecke@gmail.com>"}

SPY = yf.Ticker("^GSPC")


def get_next_trading_date(date: datetime) -> datetime:
    """
    Get the next trading date after the given date.
    :param date: Datetime to find the next trading date after
    :returns: Next trading datetime
    """
    start_date = date + timedelta(days=1)
    end_date = date + timedelta(days=30)
    dates = list_trading_dates(start_date, end_date)
    return dates[0]


def get_last_trading_date(date: datetime) -> datetime:
    """
    Get the last trading date at or before the given date.
    :param date: Datetime to find the last trading date before
    :returns: Last trading datetime
    """
    start_date = date - timedelta(days=30)
    end_date = date + timedelta(days=1)
    dates = list_trading_dates(start_date, end_date)
    return dates[-1]


def list_trading_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """
    List all trading dates between start and end date.
    :param start_date: Start datetime for the range, inclusive
    :param end_date: End datetime for the range, exclusive
    :returns: List of timezone-naive datetime trading dates
    """
    # Ensure datetimes are naive for yfinance
    start_date = remove_timezone(start_date)
    end_date = remove_timezone(end_date)
    data = SPY.history(start=start_date, end=end_date)
    dates = [remove_timezone(date.to_pydatetime()) for date in data.index]
    return dates


def cast_str_dates(date_strs: Series) -> Series:
    """
    Convert a list of date strings in 'YYYY-MM-DD' format to datetime objects.
    :param date_strs: List of date strings in 'YYYY-MM-DD' format
    :returns: List of Datetime objects localized to default timezone
    """
    dates = date_strs.apply(lambda date_str: cast_str_date(date_str))
    return dates


def cast_str_date(date_str: str) -> datetime:
    """
    Convert a date string in 'YYYY-MM-DD' format to a datetime object.
    :param date_str: Date string in 'YYYY-MM-DD' format
    :returns: Datetime object localized to default timezone
    """
    return datetime.strptime(date_str, "%Y-%m-%d")


def remove_timezones(dates: Series) -> Series:
    """
    Remove timezone information from a series of datetimes, making them naive.
    :param dates: Series of timezone-aware datetimes
    :returns: Series of naive datetimes
    """
    return dates.apply(lambda date: remove_timezone(date))


def remove_timezone(date: datetime) -> datetime:
    """
    Remove timezone information from a datetime, making it naive.
    :param date: Timezone-aware datetime
    :returns: Naive datetime
    """
    return date.replace(tzinfo=None)


def convert_datetimeindex(index: DatetimeIndex) -> DatetimeIndex:
    """
    Convert a DatetimeIndex to datetime timezone-naive DatetimeIndex.
    :param index: DatetimeIndex to convert
    :returns: DatetimeIndex with timezone-naive datetime objects
    """
    datetimes = [remove_timezone(date.to_pydatetime()) for date in index]
    return DatetimeIndex(datetimes)
