__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import logging
import simfin as sf

from utils.keys import get_api_key

# Suppress SimFin logging messages unless they are critical
logging.getLogger("simfin").setLevel(logging.CRITICAL)


def load_fundamentals():
    """
    Load all SimFin fundamentals data for immediate use.
    """
    sf.set_api_key(get_api_key("simfin"))
    sf.set_data_dir("data/other/simfin/")

    # Load the fundamentals data from SimFin
    incomes = sf.load("income", market="us", variant="quarterly")
    balances = sf.load("balance", market="us", variant="quarterly")
    cashflows = sf.load("cashflow", market="us", variant="quarterly")
    shares = sf.load("shareprices", market="us", variant="daily")

    # Filter the shares data to only include relevant columns
    shares = shares[["Ticker", "Date", "Shares Outstanding"]]
    shares = shares[shares["Shares Outstanding"].notna()]

    return {"incomes": incomes, "balances": balances, "cashflows": cashflows, "shares": shares}


if __name__ == "__main__":
    load_fundamentals()
