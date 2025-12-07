# __author__ = "Dylan Warnecke"
# __email__ = "dylan.warnecke@gmail.com"

# import pandas as pd
# from datetime import datetime

# from equities.technicals.technicals import TechnicalsData


# class Covariance:
#     """
#     Class to hold and manage covariance data between an equity universe.
#     """

#     def __init__(self, universe: list[TechnicalsData], alpha: float, length: int = 250):
#         """
#         Initialize the Covariance object with an equity universe and shrinkage factor.
#         :param universe: List of TechnicalsData objects representing the equity universe
#         :param alpha: Shrinkage factor to minimize non-diagonal elements
#         :param length: Length to calculate rolling covariance over
#         """
#         # Alpha must be between 0 and 1
#         if not (0 <= alpha <= 1):
#             raise ValueError("Alpha must be between 0 and 1.")

#         self.universe = universe
#         self._alpha = alpha
#         self._length = length

#     def calc_covariance_matrix(self, date: datetime, alpha: float) -> pd.DataFrame:
#         """
#         Calculate the covariance matrix for the equity universe as of a specific date.
#         :param date: Date to calculate covariance matrix for
#         :return: DataFrame containing covariance matrix
#         """
#         # Calculate the log returns for each equity in the universe

#         # Calculate the covariance of the 

#     def calc_log_daily_returns(self, date: datetime, length: int) -> pd.Series:
#         """
#         Calculate the log daily returns for a specific date over a given length.
#         :param date: Datetime to calculate log daily returns for, exclusive
#         :param length: Length to calculate log daily returns over
#         :return: Series containing log daily returns
#         """
#         end_date = date
#         start_date = end_date - pd.Timedelta(days=length * 2)  # Buffer for non-trading days

#         returns = pd.Series()
#         for equity in self.universe:
#             log_returns = equity.get_log_daily_returns(start_date, end_date, length)
#             returns[equity.ticker] = log_returns.iloc[-1]
#         data_slice = self._data.loc[start_date:end_date]
#         data_slice = data_slice.sort_index(ascending=True)

#         log_returns = data_slice["Log Close"].diff().dropna()

#         return log_returns[-length:]