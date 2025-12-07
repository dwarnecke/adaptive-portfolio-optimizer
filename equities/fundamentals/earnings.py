__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime
from pandas import DataFrame, Series
import pandas as pd

from utils.dates import cast_str_date


def get_trailing_earnings(
    concept: str, data: DataFrame, dates: list[datetime]
) -> Series:
    """
    Get the trailing twelve months earnings for ticker data for multiple dates efficiently.
    :param concept: Concept ('Net Income', 'Revenue') to calculate earnings for
    :param data: DataFrame of the income or cashflow statement data
    :param dates: List of dates to calculate earnings for
    :return: Series with dates as index and TTM earnings values
    """
    # Copy and prepare the data once
    data = data.copy()
    data["Local Publish Date"] = data["Publish Date"].apply(cast_str_date)
    data = data.sort_values(by="Local Publish Date", ascending=True)

    # Calculate rolling 4-quarter earnings sum
    data["TTM"] = data[concept].rolling(window=4, min_periods=4).sum()

    # Merge the dates and the data for efficient lookup
    dates_df = DataFrame({"query_date": pd.to_datetime(dates)})
    dates_df = dates_df.sort_values("query_date")
    merged = pd.merge_asof(
        dates_df,
        data[["Local Publish Date", "TTM"]],
        left_on="query_date",
        right_on="Local Publish Date",
        direction="backward",
    )

    # Map results back to original dates
    earnings = Series(index=dates, dtype=float)
    for i, date in enumerate(dates):
        ttm_value = merged.iloc[i]["TTM"]
        if pd.notna(ttm_value):
            earnings[date] = ttm_value

    return earnings
