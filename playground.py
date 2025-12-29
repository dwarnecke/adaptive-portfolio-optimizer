__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
import json
from pathlib import Path
from data.raw.loader import load_fundamentals
from features.equities.data import EquityData
from features.equities.technicals.technicals import TechnicalsData
from features.equities.fundamentals.fundamentals import FundamentalsData


def get_feature_names():
    """Get the feature names by creating sample equity features."""
    # Load fundamental data
    fund_data = load_fundamentals()
    
    # Try to find a valid ticker with data
    ticker = None
    for test_ticker in ["AAPL", "MSFT", "JPM", "JNJ", "PG"]:
        try:
            equity_data = EquityData(test_ticker, fund_data)
            if len(equity_data.data) > 0:
                ticker = test_ticker
                break
        except:
            continue
    
    if ticker is None:
        # Fallback to hardcoded feature names if no ticker works
        return _get_hardcoded_feature_names()
    
    # Get feature column names
    feature_names = list(equity_data.data.columns)
    
    # Add market regime features
    feature_names.extend(["regime_0_prob", "regime_1_prob", "regime_2_prob"])
    
    return feature_names


def _get_hardcoded_feature_names():
    """Fallback hardcoded feature names based on code structure."""
    names = []
    
    # Technical features
    for length in [5, 60, 120, 250]:
        names.append(f"Log Close Normal Score {length}")
        names.append(f"Log Volume Normal Score {length}")
    for length in [5, 60, 120, 250]:
        names.append(f"Max Drawdown {length}")
    for length in [5, 60, 120, 250]:
        names.append(f"ROC {length}")
    names.append("RSI 5")
    names.append("ATR 5")
    for length in [5, 60, 250]:
        names.append(f"Relative Return {length}")
    for length in [60, 250]:
        names.append(f"Beta {length}")
    names.append("TECH_NA")
    
    # Fundamental features (approximate)
    fund_features = ["PE", "PB", "PS", "ROE", "ROA", "Debt/Equity", "Current Ratio",
                     "Earnings Growth", "Revenue Growth", "FUND_NA"]
    names.extend(fund_features)
    
    # Market regime features
    names.extend(["regime_0_prob", "regime_1_prob", "regime_2_prob"])
    
    return names

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Get feature names
    print("\nLoading feature names...")
    feature_names = get_feature_names()
    print(f"Loaded {len(feature_names)} feature names")

    # Load the trained model checkpoint
    model_path = Path("models/checkpoints/forward_model.pt")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Get input layer weights (shape: units_hidden x units_in)
    input_weights = state_dict["input_layer.weight"]
    print(f"\nInput layer weight shape: {input_weights.shape}")
    print(f"Number of features: {input_weights.shape[1]}")

    # Calculate feature importance as sum of absolute weights
    # Each feature connects to all hidden units, sum |weight| across all connections
    feature_importance = torch.sum(torch.abs(input_weights), dim=0).numpy()

    # Create list of (feature_index, feature_name, importance) and sort by importance
    feature_ranking = [
        (i, feature_names[i] if i < len(feature_names) else f"Feature_{i}", importance) 
        for i, importance in enumerate(feature_importance)
    ]
    feature_ranking.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE RANKING (Top 20)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Feature Name':<40} {'Importance':<15} {'% Total':<10}")
    print("-" * 80)

    total_importance = feature_importance.sum()
    for rank, (feat_idx, feat_name, importance) in enumerate(feature_ranking[:20], 1):
        pct = 100 * importance / total_importance
        # Truncate long names
        display_name = feat_name[:37] + "..." if len(feat_name) > 40 else feat_name
        print(f"{rank:<6} {display_name:<40} {importance:<15.4f} {pct:<10.2f}%")

    print("\n" + "=" * 80)
    print("LEAST IMPORTANT FEATURES (Bottom 10)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Feature Name':<40} {'Importance':<15} {'% Total':<10}")
    print("-" * 80)

    for rank, (feat_idx, feat_name, importance) in enumerate(
        feature_ranking[-10:], len(feature_ranking) - 9
    ):
        pct = 100 * importance / total_importance
        display_name = feat_name[:37] + "..." if len(feat_name) > 40 else feat_name
        print(f"{rank:<6} {display_name:<40} {importance:<15.4f} {pct:<10.2f}%")

    # Statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total importance (sum of all |weights|): {total_importance:.4f}")
    print(f"Mean importance per feature: {feature_importance.mean():.4f}")
    print(f"Std dev of importance: {feature_importance.std():.4f}")
    print(
        f"Max importance: {feature_importance.max():.4f} ({feature_ranking[0][1]})"
    )
    print(
        f"Min importance: {feature_importance.min():.4f} ({feature_ranking[-1][1]})"
    )

    # Top 5 features contribution
    top5_importance = sum([imp for _, _, imp in feature_ranking[:5]])
    print(
        f"\nTop 5 features account for: {100 * top5_importance / total_importance:.2f}% of total importance"
    )

    # Top 10 features contribution
    top10_importance = sum([imp for _, _, imp in feature_ranking[:10]])
    print(
        f"Top 10 features account for: {100 * top10_importance / total_importance:.2f}% of total importance"
    )
