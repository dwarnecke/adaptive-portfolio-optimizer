__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime
from pathlib import Path
from features.markets.observations import ObservationsData
from features.markets.model import RegimeModel
from models.trainer import main as train_forward_model
from features.equities.equity import EquityData
from data.loader import load_fundamentals

# ========== TEST EQUITY DATA ==========
print("=" * 60)
print("TESTING EQUITY DATA")
print("=" * 60)

print("\nLoading fundamentals data...")
fund_data = load_fundamentals()
print("[OK] Fundamentals loaded!")

# Test with AAPL
test_ticker = "AAPL"
print(f"\nTesting EquityData for {test_ticker}...")

try:
    equity_data = EquityData(test_ticker, fund_data)
    print(f"[OK] EquityData created for {test_ticker}")

    print(f"\nData shape: {equity_data.data.shape}")
    print(f"Targets shape: {equity_data.targets.shape}")

    print(f"\nData columns ({len(equity_data.data.columns)}):")
    print(f"  {list(equity_data.data.columns)}")

    print(f"\nData date range:")
    print(f"  First: {equity_data.data.index[0]}")
    print(f"  Last: {equity_data.data.index[-1]}")

    print(f"\nSample data (first 5 rows):")
    print(equity_data.data.head())

    print(f"\nSample targets (first 5 rows):")
    print(equity_data.targets.head())

    print(f"\nData statistics:")
    print(equity_data.data.describe())

    # Check for NaN values
    nan_counts = equity_data.data.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n[WARNING] NaN values found in features:")
        print(nan_counts[nan_counts > 0])
    else:
        print(f"\n[OK] No NaN values in features")

    target_nans = equity_data.targets.isna().sum()
    if target_nans > 0:
        print(f"[WARNING] {target_nans} NaN values in targets")
    else:
        print(f"[OK] No NaN values in targets")

    print("\n" + "=" * 60)
    print("EQUITY DATA TEST COMPLETE")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Failed to create EquityData: {e}")
    import traceback

    traceback.print_exc()
    print("\n" + "=" * 60)
    print("EQUITY DATA TEST FAILED")
    print("=" * 60)
    exit(1)

# ========== TRAIN AND SAVE REGIME MODEL ==========
print("\n" + "=" * 60)
print("STEP 1: TRAINING REGIME MODEL")
print("=" * 60)

# Use full historical data for training regime model (2010-2020)
train_start = datetime(2010, 1, 1)
train_end = datetime(2020, 12, 31)
print(f"\nTraining period: {train_start.date()} to {train_end.date()}")

# Check if regime model already exists
model_dir = Path("models/checkpoints")
model_path = model_dir / "regime_model_3states.pkl"

if model_path.exists():
    print("\nRegime model already exists, skipping training...")
    print(f"Loading from: {model_path}")
    model = RegimeModel.load(model_path)
    print("[OK] Model loaded!")
else:
    print("\nCreating ObservationsData for training...")
    observations = ObservationsData(train_start, train_end)
    print(f"[OK] Observations shape: {observations.inputs.shape}")
    print(f"  Features: {list(observations.inputs.columns)}")

    print("\nInitializing RegimeModel with 3 states...")
    model = RegimeModel(states=3, data=observations.inputs)

    print("\nTraining HMM with Baum-Welch algorithm...")
    print("  max_iter=100, tol=1e-6 for production-quality model")
    model.train(observations.inputs, max_iter=100, tol=1e-6)
    print("[OK] Model trained successfully!")

    # Save to models/checkpoints directory
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving regime model to: {model_path}")
    model.save(model_path)
    print("[OK] Model saved!")

print("\n" + "=" * 60)
print("REGIME MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"Model saved to: {model_path.absolute()}")

# ========== TRAIN FORWARD MODEL ==========
print("\n" + "=" * 60)
print("STEP 2: TRAINING FORWARD MODEL")
print("=" * 60)
print("\nStarting forward model training...")

train_forward_model()

print("\n" + "=" * 60)
print("ALL TRAINING COMPLETE")
print("=" * 60)
