__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.forward_model import ForwardModel
from features.dataset import FeaturesDataset


def train(
    dataset: FeaturesDataset,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 2**-13,
) -> ForwardModel:
    """
    Main function for training the transformer forward model.
    :param dataset: Training dataset for the model
    :param epochs: Number of training epochs
    :param batch: Batch size for training
    :param learning_rate: Learning rate for optimizer, default 1e-4
    :return: Trained ForwardModel instance
    """
    print("\nInitializing training... ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = dataset[0][0].shape[-1]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Initializing model... ", end="")
    model = ForwardModel(units_in=features, hidden_layers=2, units_hidden=64)
    model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{params} parameters initialized", end="\n")

    print("Computing normalization statistics... ", end="")
    # Sample a subset for initialization to avoid memory issues
    sample_size = min(1000, len(dataset))
    stack = torch.stack(
        [torch.nan_to_num(dataset[i][0], nan=0.0) for i in range(sample_size)]
    )
    stack = stack.to(device)
    model.initialize(stack)
    print("calculated", end="\n")

    # Train over the number of epochs specified
    print(f"\nTraining for {epochs} epochs with learning rate {learning_rate:.6f}...")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Check for NaN/Inf in input
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: NaN/Inf in input x")
                continue
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"WARNING: NaN/Inf in target y")
                continue

            optimizer.zero_grad()
            output = model(x)

            # Check for NaN/Inf in model output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"WARNING: NaN/Inf in model output")
                print(
                    f"  x stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}"
                )
                print(f"  output: {output[:3]}")
                continue

            mu_hat, sigma_hat = output[:, 0], output[:, 1]
            # Clamp sigma to prevent numerical instability
            sigma_hat = torch.clamp(sigma_hat, min=2**-12, max=2**3)

            # Check sigma_hat
            if torch.isnan(sigma_hat).any() or (sigma_hat <= 0).any():
                print(f"WARNING: Invalid sigma_hat")
                print(f"  sigma_hat: {sigma_hat[:10]}")
                continue

            loss = _calc_loss(model, mu_hat, sigma_hat, y)

            # Check loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss")
                print(f"  mu_hat: {mu_hat[:5]}")
                print(f"  sigma_hat: {sigma_hat[:5]}")
                print(f"  y: {y[:5]}")
                continue

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train_loss = (
            train_loss / train_batches if train_batches > 0 else float("nan")
        )
        print(f"\nEpoch {epoch+1}/{epochs}, Loss {avg_train_loss:.4f}")

    # Save model to the models directory
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "forward_model.pt"
    print(f"\nSaving model to {model_path}...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": features,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_path,
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model saved with {params} parameters")

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
    train()
