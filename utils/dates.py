__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import yfinance as yf
from datetime import datetime, timedelta
from pandas import Series, DatetimeIndex
from zoneinfo import ZoneInfo

HEADERS = {"User-Agent": "Dylan Warnecke <dylan.warnecke@gmail.com>"}

SPY = yf.Ticker("^GSPC")

# Load all historical trading dates once
data = SPY.history(period="max")
open_dates = [date.to_pydatetime().replace(tzinfo=None) for date in data.index]

# Build constant dictionaries for O(1) lookups
_NEXT_TRADING_DATE = {}
_LAST_TRADING_DATE = {}

# Populate next trading date mapping
for i in range(len(open_dates) - 1):
    current_date = open_dates[i]
    next_date = open_dates[i + 1]
    # Map all dates from current to next to the next trading date
    check_date = current_date
    while check_date < next_date:
        _NEXT_TRADING_DATE[check_date] = next_date
        check_date += timedelta(days=1)

# Populate last trading date mapping
for i in range(1, len(open_dates)):
    current_date = open_dates[i]
    prev_date = open_dates[i - 1]
    # Map all dates from previous to current to current trading date
    check_date = prev_date + timedelta(days=1)
    while check_date <= current_date:
        _LAST_TRADING_DATE[check_date] = current_date
        check_date += timedelta(days=1)

# Add mapping for the first trading date (map to itself)
if open_dates:
    first_date = open_dates[0]
    _LAST_TRADING_DATE[first_date] = first_date


def list_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """
    List all trading dates between start and end date.
    :param start_date: Start datetime for the range, inclusive
    :param end_date: End datetime for the range, exclusive
    :returns: List of timezone-naive datetime trading dates
    """
    dates = []
    current_date = get_last_date(start_date)
    if current_date == start_date:
        dates.append(current_date)
    current_date = get_next_date(current_date + timedelta(days=1))
    while current_date < end_date:
        dates.append(current_date)
        current_date = get_next_date(current_date + timedelta(days=1))
    return dates


def get_next_date(date: datetime) -> datetime:
    """
    Get the next trading date another date, exclusive.
    :param date: Datetime to find the next trading date after
    :returns: Next trading datetime, exclusive
    """
    return _NEXT_TRADING_DATE[date]


def get_last_date(date: datetime) -> datetime:
    """
    Get the last trading date at or before the given date.
    :param date: Datetime to find the last trading date before
    :returns: Last trading datetime, inclusive
    """
    return _LAST_TRADING_DATE[date]


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
