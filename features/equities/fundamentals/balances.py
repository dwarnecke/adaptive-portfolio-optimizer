__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime
from pandas import DataFrame, Series
import pandas as pd

from utils import cast_str_date


def get_balance(concept: str, data: DataFrame, dates: list[datetime]) -> Series:
    """
    Calculate the balance for ticker data for multiple dates efficiently.
    :param concept: Concept ('Total Assets', 'Total Liabilities') to calculate balance for
    :param data: DataFrame of the balance sheet data
    :param dates: List of dates to calculate balance for
    :return: Series with dates as index and balance values
    """
    data = data.copy()
    data["Local Publish Date"] = pd.to_datetime(
        data["Publish Date"].apply(cast_str_date)
    )
    data = data.sort_values(by="Local Publish Date", ascending=True)

    # Merge the dates and the data for efficient lookup
    dates_df = DataFrame({"query_date": pd.to_datetime(dates)})
    dates_df = dates_df.sort_values("query_date")
    merged = pd.merge_asof(
        dates_df,
        data[["Local Publish Date", concept]],
        left_on="query_date",
        right_on="Local Publish Date",
        direction="backward",
    )

    # Map results back to original dates
    balances = Series(index=dates, dtype=float)
    for i, date in enumerate(dates):
        balance_value = merged.iloc[i][concept]
        if pd.notna(balance_value):
            balances[date] = balance_value

    return balances
