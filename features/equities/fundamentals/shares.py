__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import pandas as pd
from datetime import datetime

from utils.dates import cast_str_date


def get_shares_outstanding(data: pd.DataFrame, dates: list[datetime]) -> pd.Series:
    """
    Get the number of shares outstanding for multiple dates efficiently.
    :param data: DataFrame containing shares outstanding data
    :param dates: List of dates to get shares outstanding for
    :return: Series with dates as index and shares outstanding as values
    """
    data = data.copy()
    
    # Convert the Date column to datetime objects
    data["Date"] = pd.to_datetime(data["Date"].apply(cast_str_date))
    data = data.sort_values(by="Date", ascending=True)

    # Merge the dates and the data for efficient lookup
    dates_df = pd.DataFrame({"query_date": pd.to_datetime(dates)})
    dates_df = dates_df.sort_values("query_date")
    merged = pd.merge_asof(
        dates_df,
        data[["Date", "Shares Outstanding"]],
        left_on="query_date",
        right_on="Date",
        direction="backward",
    )

    # Map results back to original dates
    shares = pd.Series(index=dates, dtype=float)
    for i, date in enumerate(dates):
        shares_value = merged.iloc[i]["Shares Outstanding"]
        if pd.notna(shares_value):
            shares[date] = shares_value

    return shares
