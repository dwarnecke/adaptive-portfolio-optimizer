__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

from models.model import ForwardModel
from features.dataset import FeaturesDataset
from utils.tickers import get_universe


def main(use_preprocessed: bool = False):
    """
    Main function for training the transformer forward model.
    :param use_preprocessed: Whether to use preprocessed datasets from disk
    """
    print("\nInitializing training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get universe of tickers (S&P 100)
    tickers = get_universe()[:10]  # Use only first 10 for testing
    print(f"\nUniverse: {len(tickers)} tickers (testing with subset)")

    # Regime model path
    regime_model_path = "models/checkpoints/regime_model_3states.pkl"

    # Create or load datasets
    if use_preprocessed:
        print("\n[INFO] Loading preprocessed datasets...")

        print("\nLoading training dataset...")
        train_dataset = FeaturesDataset.load("features/data", "train_dataset.pkl")

        print("\nLoading dev dataset...")
        dev_dataset = FeaturesDataset.load("features/data", "dev_dataset.pkl")

    else:
        print("\n[INFO] Generating features on-the-fly (slow)...")

        print("\nCreating training dataset (2010-01-01 to 2020-01-01)...")
        train_dataset = FeaturesDataset(
            tickers=tickers,
            path=regime_model_path,
            start_date=datetime(2010, 1, 1),
            end_date=datetime(2020, 1, 1),
        )
        print(f"[OK] Training samples: {len(train_dataset)}")

        print("\nCreating dev dataset (2021-01-01 to 2023-01-01)...")
        dev_dataset = FeaturesDataset(
            tickers=tickers,
            path=regime_model_path,
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2023, 1, 1),
        )
        print(f"[OK] Dev samples: {len(dev_dataset)}")

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Get a sample batch to determine input size
    print("\nDetermining input dimensions...")
    sample_x, _ = train_dataset[0]
    num_features = sample_x.shape[-1]
    print(f"Input features: {num_features}")

    # Initialize model
    print("\nInitializing ForwardModel...")
    model = ForwardModel(
        units_in=num_features,
        hidden_layers=2,
        units_hidden=64,
        num_heads=4,
        dropout=0.1,
    )
    model.to(device)

    # Initialize normalization with training data
    print("Computing normalization statistics...")
    all_train_x = []
    for i in range(min(1000, len(train_dataset))):
        x, _ = train_dataset[i]
        all_train_x.append(x)
    all_train_x = torch.stack(all_train_x).to(device)
    model.initialize(all_train_x)
    print("[OK] Model initialized")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop (minimal epochs for testing)
    num_epochs = 2
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            mu_hat, sigma_hat = output[:, 0], output[:, 1]

            loss = _calc_loss(model, mu_hat, sigma_hat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        dev_loss = 0.0
        dev_batches = 0

        with torch.no_grad():
            for x, y in dev_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                mu_hat, sigma_hat = output[:, 0], output[:, 1]
                loss = _calc_loss(model, mu_hat, sigma_hat, y)
                dev_loss += loss.item()
                dev_batches += 1

        avg_dev_loss = dev_loss / dev_batches

        print(
            f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}"
        )

    # Save model
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "forward_model.pt"

    print(f"\nSaving model to {model_path}...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": num_features,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_path,
    )
    print("[OK] Model saved!")

    return model


def _calc_loss(model, mu_hat, sigma_hat, r_true, l1_lambda=2**-20):
    """
    Calculate the model loss using Gaussian negative log-likelihood and L1 regularization.
    :param model: Transformer model to calculate loss for
    :param mu_hat: Predicted mean returns, shape (batch_size,)
    :param sigma_hat: Predicted standard deviations, shape (batch_size,)
    :param r_true: True returns, shape (batch_size,)
    :param l1_lambda: L1 regularization lambda, default 2**-20
    """
    nll_loss = _calc_gaus_nll(mu_hat, sigma_hat, r_true)
    # L1 regularization encourages sparsity in model parameters
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    total_loss = nll_loss + l1_lambda * l1_loss
    return total_loss


def _calc_gaus_nll(mu_hat, sigma_hat, r_true):
    """
    Calculate the model loss using Gaussian negative log-likelihood.
    :param mu_hat: Predicted mean returns, shape (batch_size,)
    :param sigma_hat: Predicted standard deviations, shape (batch_size,)
    :param r_true: True returns, shape (batch_size,)
    :returns: Mean negative log-likelihood loss
    """
    # Optimizes the gaussian negative log-likelihood
    nll = 0.5 * (((r_true - mu_hat) / sigma_hat) ** 2 + 2 * torch.log(sigma_hat))
    return nll.mean()


if __name__ == "__main__":
    main()
