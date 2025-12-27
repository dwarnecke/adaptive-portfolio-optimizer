__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from train import train
from features.dataset import FeaturesDataset

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING FORWARD MODEL ON DOW 30 DATASET")
    print("=" * 60)

    # Load the training dataset
    print("\nLoading training dataset...")
    train_dataset = FeaturesDataset.load(
        directory="data/processed", filename="dow30_train.pkl"
    )

    # Train the model
    print("\nStarting training...")
    model = train(dataset=train_dataset, epochs=2, batch_size=64, learning_rate=2**-13)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
