__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime
from markets.observations import ObservationsData
from markets.regime import RegimeModel
from markets.market import MarketData

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

# Display info
print(f"Features shape: {market.features.shape}")
print(f"Feature columns: {list(market.features.columns)}")
print()

print(f"Targets shape: {market.targets.shape}")
print(f"Target columns: {list(market.targets.columns)}")
print()

print("Features summary (first 5 rows):")
print(market.features.head())
print()

print("Features summary (last 5 rows):")
print(market.features.tail())
print()

print("Regime probability statistics:")
regime_cols = [col for col in market.features.columns if "regime" in col]
print(market.features[regime_cols].describe())
print()

print("All tests completed successfully!")
