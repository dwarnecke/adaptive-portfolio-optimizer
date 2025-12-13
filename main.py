__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from datetime import datetime

from features.dataset import FeaturesDataset


if __name__ == "__main__":
    dataset = FeaturesDataset(
        path="models/checkpoints/regime_model_3states.pkl",
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2020, 1, 1),
    )
    dataset.save(directory="features/data", filename="train_dataset.pkl")
    print(f"Dataset saved successfully with {len(dataset)} samples")
