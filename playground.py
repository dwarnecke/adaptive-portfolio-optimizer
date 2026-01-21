__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from config.hyperparameters import HYPERPARAMETERS
from features.dataset import FeaturesDataset
from portfolio.portfolio import Portfolio

if __name__ == "__main__":
    print("=" * 80)
    print("PORTFOLIO SIMULATION - DOW 30")
    print("=" * 80)

    # Load the train dataset (for overfitting test)
    dataset_dir = Path("outputs/datasets/dataset_20260117_121735")
    model_dir = Path(
        "outputs/models/forward_20260117_181318"
    )  # Latest trained model (IC 0.14)

    print(f"\nLoading train dataset...")
    train_dataset = FeaturesDataset.load(dataset_dir, "dow30_train.pkl")
    print(f"  Loaded {len(train_dataset)} train samples")
    print(f"  Tickers: {len(train_dataset.index)}")

    # Load the forward model
    print(f"\nLoading forward model...")
    forward_model = torch.load(model_dir / "forward_model.pth", weights_only=False)
    forward_model.eval()
    print(f"  Model loaded from {model_dir}")

    # Initialize portfolio
    print(f"\nInitializing portfolio...")
    capital = 100_000_000  # $100M
    portfolio = Portfolio(
        dataset=train_dataset,
        forward_model=forward_model,
        capital=capital,
        parameters=HYPERPARAMETERS["portfolio"],
    )
    print(f"  Initial capital: ${capital:,.0f}")

    # Run simulation on training period starting from 2011
    # Portfolio will automatically handle tickers that don't have data yet
    start_date = datetime(2011, 2, 15)  # ~30th trading day of 2011
    end_date = datetime(2020, 12, 31)  # End of training period

    print(f"\nSimulating portfolio from {start_date.date()} to {end_date.date()}...")
    print("=" * 80)
    print()

    statistics = portfolio.simulate(start_date, end_date)

    # Display results
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)

    final_value = list(portfolio.history.values())[-1]
    final_return = list(statistics["return"].values())[-1]
    max_drawdown = max(statistics["drawdown"].values())
    final_sharpe = list(statistics["sharpe"].values())[-1]

    print(f"\nFinal portfolio value: ${final_value:,.0f}")
    print(f"Total return: {final_return*100:.2f}%")
    print(f"Maximum drawdown: {max_drawdown*100:.2f}%")
    print(f"Sharpe ratio: {final_sharpe:.2f}")

    # Check model predictions
    print("\n" + "-" * 80)
    print("MODEL PREDICTIONS (first 5 days)")
    print("-" * 80)
    dates_to_check = list(portfolio.weight_history.keys())[:5]
    for date in dates_to_check:
        # Get predictions from the portfolio model
        mus, sigmas = portfolio._model.predict(date, train_dataset)

        if not mus:
            continue

        # Analyze the predicted returns
        predictions = list(mus.values())
        num_positive = sum(1 for p in predictions if p > 0)
        num_negative = sum(1 for p in predictions if p < 0)
        num_zero = sum(1 for p in predictions if p == 0)
        avg_prediction = np.mean(predictions)
        min_pred = min(predictions)
        max_pred = max(predictions)

        print(
            f"  {date.date()}: pos={num_positive}, neg={num_negative}, zero={num_zero}, avg={avg_prediction:.6f}, range=[{min_pred:.6f}, {max_pred:.6f}]"
        )

    # Diagnostic info
    print("\n" + "-" * 80)
    print("DIAGNOSTICS")
    print("-" * 80)
    print(
        f"Transaction costs: {HYPERPARAMETERS['portfolio']['slippage_bps'] + HYPERPARAMETERS['portfolio']['commission_bps']} bps"
    )

    # Check leverage distribution over time
    print(f"\nLeverage statistics:")
    leverages = [
        sum(abs(w) for w in weights.values())
        for weights in portfolio.weight_history.values()
    ]
    print(f"  Average leverage: {sum(leverages)/len(leverages):.2f}x")
    print(f"  Min leverage: {min(leverages):.2f}x")
    print(f"  Max leverage: {max(leverages):.2f}x")

    # Check long/short balance
    print(f"\nLong/Short balance (first 10 days):")
    for i, (date, weights) in enumerate(list(portfolio.weight_history.items())[:10]):
        long_weight = sum(w for w in weights.values() if w > 0)
        short_weight = sum(w for w in weights.values() if w < 0)
        net_weight = long_weight + short_weight
        num_securities = len(weights)
        max_allowed = 4.0 / num_securities if num_securities > 0 else 0
        print(
            f"  {date.date()}: Long={long_weight:.2f}, Short={short_weight:.2f}, Net={net_weight:.2f}, Securities={num_securities}, MaxPos={max_allowed:.2%}"
        )

    # Check securities count over time
    print(f"\nSecurities availability over time:")
    securities_counts = [len(weights) for weights in portfolio.weight_history.values()]
    print(f"  Min securities: {min(securities_counts)}")
    print(f"  Max securities: {max(securities_counts)}")
    print(f"  Average securities: {sum(securities_counts)/len(securities_counts):.1f}")

    # Show days with very few securities
    print(f"\nDays with fewer than 10 securities:")
    count = 0
    for date, weights in portfolio.weight_history.items():
        if len(weights) < 10:
            print(f"  {date.date()}: {len(weights)} securities")
            count += 1
            if count >= 20:  # Limit output
                print(f"  ... and more")
                break

    # Compare to SPY benchmark
    print(f"\nBenchmark comparison:")

    # Calculate actual SPY return
    try:
        import yfinance as yf

        spy = yf.Ticker("SPY")
        hist = spy.history(start=start_date, end=end_date)
        if len(hist) > 0:
            spy_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
            print(
                f"  SPY ({start_date.date()} to {end_date.date()}): +{spy_return:.2f}%"
            )
            print(f"  Portfolio: {final_return*100:.2f}%")
            print(f"  Alpha: {(final_return*100 - spy_return):.2f}%")
        else:
            print(f"  SPY: Unable to fetch data")
            print(f"  Portfolio: {final_return*100:.2f}%")
    except Exception as e:
        print(f"  SPY: Unable to fetch data ({e})")
        print(f"  Portfolio: {final_return*100:.2f}%")

    # Check position sizing
    print(f"\nPosition sizing:")
    print(f"  Max allowed position: {4.0/22:.2%}")
    print(f"  Number of securities: 22")

    # Show weight history
    print(f"\nWeight history (showing dates with extreme weights):")
    for date, weights in portfolio.weight_history.items():
        total_weight = sum(abs(w) for w in weights.values())
        max_weight = max(weights.values()) if weights else 0
        min_weight = min(weights.values()) if weights else 0

        # Show if extreme leverage or extreme positions
        if total_weight > 2.0 or max_weight > 0.5 or min_weight < -0.5:
            print(
                f"\n  {date.date()}: Total leverage: {total_weight:.2f}, Max: {max_weight:.2f}, Min: {min_weight:.2f}"
            )
            # Show top 3 largest absolute weights
            sorted_weights = sorted(
                weights.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            for idx, weight in sorted_weights:
                print(f"    Index {idx}: {weight:.3f}")

    # Full value history
    print(f"\nFull value history ({len(portfolio.history)} days):")
    for date, value in portfolio.history.items():
        marker = " (NEGATIVE)" if value < 0 else ""
        print(f"  {date.date()}: ${value:,.0f}{marker}")

    # Check for negative values
    negative_days = [(d, v) for d, v in portfolio.history.items() if v < 0]
    if negative_days:
        print(f"\n*** WARNING: Portfolio went negative on {len(negative_days)} days!")
        print(
            f"  First negative: {negative_days[0][0].date()} = ${negative_days[0][1]:,.0f}"
        )
        print(
            f"  Most negative: {min(negative_days, key=lambda x: x[1])[0].date()} = ${min(negative_days, key=lambda x: x[1])[1]:,.0f}"
        )

    print("\n" + "=" * 80)
