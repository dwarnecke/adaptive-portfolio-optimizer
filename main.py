__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from features.dataset import FeaturesDataset

if __name__ == "__main__":
    print("\nLoading training dataset...")
    dataset = FeaturesDataset.load(
        directory="data/processed", filename="train_dataset.pkl"
    )

    print(f"\nDataset loaded successfully!")
    print(f"Number of tickers: {len(dataset.tickers)}")
    print(f"Total samples: {len(dataset)}")
    print(f"\nFirst 10 tickers: {list(dataset.tickers.values())[:10]}")

    # Show sample information
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"\nSample shape: {x.shape}")
        print(f"Sample target: {y}")
