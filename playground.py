__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json
import pandas as pd
from pathlib import Path

# ========== GET S&P 500 TICKERS ==========
print("=" * 60)
print("GETTING S&P 500 TICKERS")
print("=" * 60)

print("\nFetching S&P 500 ticker list from Wikipedia...")
try:
    # Get S&P 500 list from Wikipedia with proper headers
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    tables = pd.read_html(
        url, header=0, storage_options={"User-Agent": headers["User-Agent"]}
    )
    sp500_table = tables[0]

    # Extract tickers and clean them
    tickers = sp500_table["Symbol"].tolist()
    # Replace periods with hyphens (e.g., BRK.B -> BRK-B for yfinance)
    tickers = [ticker.replace(".", "-") for ticker in tickers]

    print(f"[OK] Found {len(tickers)} tickers")

    # Save to JSON
    output_path = Path("data/raw/sp500_tickers.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tickers, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    print(f"\nFirst 10 tickers: {tickers[:10]}")
    print(f"Last 10 tickers: {tickers[-10:]}")

    print("\n" + "=" * 60)
    print("S&P 500 TICKERS SAVED")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Failed to fetch S&P 500 tickers: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
print("DATASET LOADED")
print("=" * 60)
