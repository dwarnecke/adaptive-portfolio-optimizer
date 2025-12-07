__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime
from features.markets.observations import ObservationsData
from features.markets.regime import RegimeModel
from features.markets.market import MarketData
from features.equities.equity import EquityData
from features.features import FeaturesData
from data.loader import load_fundamentals

# Test ObservationsData
print("Testing ObservationsData...")
start = datetime(2020, 1, 1)
end = datetime(2024, 1, 1)
print(f"Creating ObservationsData from {start.date()} to {end.date()}...")
observations = ObservationsData(start, end)
print(f"ObservationsData created successfully!\n")

# Test RegimeModel
print("Testing RegimeModel...")
print("Creating and training RegimeModel with 3 states...")
model = RegimeModel(states=3, data=observations.inputs)
model.train(observations.inputs, max_iter=10, tol=1e-4)
print("Model trained successfully!\n")

# Save the model
print("Saving regime model...")
model.save("test_regime_model.pkl")
print("Model saved!\n")

# Test MarketData
print("Testing MarketData...")
print("Creating MarketData with observations and regime model...")
market = MarketData(observations, "test_regime_model.pkl")
print("MarketData created successfully!\n")

# Display market info
print(f"Market Features shape: {market.features.shape}")
print(f"Market Feature columns: {list(market.features.columns)}")
print()

print(f"Market Targets shape: {market.targets.shape}")
print(f"Market Target columns: {list(market.targets.columns)}")
print()

# Test EquityData
print("Testing EquityData...")
ticker = "AAPL"
print(f"Loading fundamental data...")
fund_data = load_fundamentals()
print(f"Creating EquityData for {ticker}...")
equity = EquityData(ticker, fund_data)
print(f"EquityData created successfully!\n")

print(f"Equity Features shape: {equity.features.shape}")
print(f"Equity Feature columns: {list(equity.features.columns)}")
print()

# Test FeaturesData
print("Testing FeaturesData...")
print(f"Combining equity and market features...")
features = FeaturesData(equity, market)
print(f"FeaturesData created successfully!\n")

print(f"Combined Features shape: {features._features.shape}")
print(f"Combined Feature columns: {list(features._features.columns)}")
print()

# Test NA indicators
print("Testing NA indicator features...")
equity_features = equity.features
indicator_cols = [col for col in equity_features.columns if col.endswith('_NA')]
print(f"Found {len(indicator_cols)} NA indicator columns: {indicator_cols}")
print()

# Check if non-indicator features have any NAs
non_indicator_cols = [col for col in equity_features.columns if not col.endswith('_NA')]
print(f"Checking {len(non_indicator_cols)} non-indicator features for NA values...")
na_counts = equity_features[non_indicator_cols].isna().sum()
features_with_na = na_counts[na_counts > 0]
if len(features_with_na) > 0:
    print(f"WARNING: Found NA values in non-indicator features:")
    print(features_with_na)
else:
    print("✓ All non-indicator features have no NA values!")
print()

# Verify indicator features work correctly
print("Verifying NA indicators...")
for indicator_col in indicator_cols:
    base_col = indicator_col.replace('_NA', '')
    if base_col in equity_features.columns:
        # Check that when indicator is 1, original was 0 (filled)
        indicator_is_one = equity_features[indicator_col] == 1
        base_is_zero = equity_features[base_col] == 0
        if (indicator_is_one & ~base_is_zero).any():
            print(f"WARNING: {base_col} has non-zero values when {indicator_col}=1")
        else:
            print(f"✓ {base_col} correctly zeroed when NA (indicator: {indicator_col})")
print()

# Test slice_windows
print("Testing slice_windows...")
window_start = datetime(2023, 1, 1)
window_end = datetime(2023, 6, 1)
window_length = 60
print(f"Creating windows from {window_start.date()} to {window_end.date()} with length {window_length}...")
windows = features.slice_windows(window_start, window_end, window_length)
print(f"Windows shape: {windows.shape}")
print(f"Expected shape: (dates, {window_length}, {features._features.shape[1]})")
print()

# Verify tensor properties
print(f"Tensor dtype: {windows.dtype}")
print(f"Tensor has NaN: {windows.isnan().any().item()}")
print(f"Tensor has Inf: {windows.isinf().any().item()}")
print()

print("All tests completed successfully!")
