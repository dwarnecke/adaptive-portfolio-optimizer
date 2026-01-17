__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import logging
import simfin as sf
import warnings
import sys
import os

# Suppress SimFin logging messages unless they are critical
logging.getLogger("simfin").setLevel(logging.CRITICAL)

# Suppress FutureWarning about date_parser deprecation in simfin
warnings.filterwarnings("ignore", category=FutureWarning, module="simfin")


def load_fundamentals():
    """
    Load all SimFin fundamentals data for immediate use.
    """
    sf.set_api_key(os.environ["SIMFIN_KEY"])
    sf.set_data_dir("data/simfin/")

    # Suppress all simfin print statements to avoid the unhandled warnings
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        incomes = sf.load("income", market="us", variant="quarterly")
        balances = sf.load("balance", market="us", variant="quarterly")
        cashflows = sf.load("cashflow", market="us", variant="quarterly")
        shares = sf.load("shareprices", market="us", variant="daily")
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    # Filter the shares data to only include relevant columns
    shares = shares[["Ticker", "Date", "Shares Outstanding"]]
    shares = shares[shares["Shares Outstanding"].notna()]
    return {
        "incomes": incomes,
        "balances": balances,
        "cashflows": cashflows,
        "shares": shares,
    }


if __name__ == "__main__":
    load_fundamentals()
