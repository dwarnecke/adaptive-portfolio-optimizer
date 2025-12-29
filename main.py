__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

from train import train
from features.dataset import FeaturesDataset

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING FORWARD MODEL ON RUSSELL 1000 DATASET")
    print("=" * 60)

    # Load the training dataset
    print("\nLoading training dataset...")
    train_dataset = FeaturesDataset.load(
        directory="data/processed",
        filename="russell1000_train.pkl",
    )

    # Load the evaluation dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = FeaturesDataset.load(
        directory="data/processed",
        filename="russell1000_eval.pkl",
    )

    # Train the model
    print("\nStarting training...")
    model = train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=4,
        batch_size=64,
        learning_rate=2**-13,
        l1_lambda=2**-14,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
