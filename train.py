__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import os
import random
import torch
from datetime import datetime
from torch import device, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.forward_model import ForwardModel
from features.dataset import FeaturesDataset


def train(
    train_dataset: FeaturesDataset,
    eval_dataset: FeaturesDataset = None,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 2**-10,
    l1_lambda: float = 2**-20,
) -> ForwardModel:
    """
    Main function for training the transformer forward model.
    :param train_dataset: Training dataset for the model
    :param eval_dataset: Evaluation dataset for the model, default None
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :param learning_rate: Learning rate for optimizer, default 2**-10
    :param l1_lambda: L1 regularization lambda, default 2**-20
    :return: Trained ForwardModel instance
    """
    print("\nInitializing training... ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if eval_dataset is not None:
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    model = _initialize_model(train_dataset, device)

    # Train over the number of epochs specified
    print(f"\nTraining for {epochs} epochs")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        report = f"Epoch {epoch+1}/{epochs}"
        loss = 0

        model.train()
        for batch, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            loss += _step_update(x, y, model, optimizer, l1_lambda)
            print(f"{report}, batch {batch}/{len(train_loader)}", end="\r")
        loss = loss / len(train_loader)
        report += f", train loss {loss:.4f}"

        # Evaluate on the evaluation dataset if provided for monitoring
        if eval_dataset is not None:
            model.eval()
            with torch.no_grad():
                loss = _calc_dataset_loss(eval_loader, model, device, l1_lambda)
            report += f", eval loss {loss:.4f}"

        print(report, end="\n")

    _save_model(model, optimizer, path="models/checkpoints/forward_model.pth")
    return model


def _initialize_model(dataset: FeaturesDataset, device: device) -> ForwardModel:
    """
    Initialize the transformer forward model on the specified device.
    :param device: Device to initialize the model on
    :param features: Number of input features
    :return: Initialized ForwardModel instance
    """
    units_in = dataset[0][0].shape[-1]
    model = ForwardModel(units_in=units_in, hidden_layers=2, units_hidden=64)
    model.to(device)

    # Sample a subset for normalization calculations to avoid memory issues
    sample_size = min(10000, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)
    sample = torch.stack([dataset[i][0] for i in sample_indices])
    sample = sample.to(device)
    model.initialize(sample)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {params} parameters", end="\n")
    return model


def _step_update(
    x: Tensor, y: Tensor, model: ForwardModel, optimizer: Optimizer, l1_lambda: float
) -> float:
    """
    Perform a single training step update.
    :param x: Input feature tensor
    :param y: True target tensor
    :param model: Transformer model to train
    :param optimizer: Optimizer for updating model parameters
    :param l1_lambda: L1 regularization lambda
    :return: Calculated loss for the step
    """
    optimizer.zero_grad()
    loss = _calc_loss(x, y, model, l1_lambda=l1_lambda)
    loss.backward()
    optimizer.step()
    return loss.item()


def _calc_dataset_loss(
    data_loader: DataLoader,
    model: ForwardModel,
    device: device,
    l1_lambda: float = 2**-20,
) -> float:
    """
    Calculate the loss over an entire dataset.
    :param data_loader: DataLoader for the dataset
    :param model: Transformer model to calculate loss for
    :param l1_lambda: L1 regularization lambda
    """
    loss = 0.0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        loss += _calc_loss(x, y, model, l1_lambda).item()
    return loss / len(data_loader)


def _save_model(model: ForwardModel, optimizer: Optimizer, path: str):
    """
    Save the trained model to disk in a timestamped directory.
    :param model: Trained ForwardModel instance
    :param path: Base path for saving the model
    """
    # Create timestamped directory to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(path)
    model_dir = os.path.join(base_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model in the new directory
    model_path = os.path.join(model_dir, "forward_model.pth")
    model_dict = {
        "model_state_dict": model.state_dict(),
        "num_features": model._mean.shape[-1],
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(model_dict, model_path)
    print(f"Model saved to {model_path}")


def _calc_loss(x: Tensor, y: Tensor, model: ForwardModel, l1_lambda: float = 2**-20):
    """
    Calculate the loss using Gaussian negative log-likelihood and L1 regularization.
    :param x: Input feature tensor
    :param y: True target tensor
    :param model: Transformer model to calculate loss for
    :param l1_lambda: L1 regularization lambda, default 2**-20
    """
    y_hat = model(x)
    mu_hat, sigma_hat = y_hat[:, 0], y_hat[:, 1]
    sigma_hat = torch.clamp(sigma_hat, min=2**-12, max=2**3)

    # Negative log-likelihood loss optimizes fit to predicted Gaussian distribution
    loss = (y - mu_hat) ** 2 / (2 * sigma_hat**2) + torch.log(sigma_hat)
    loss = loss.mean()

    # L1 regularization encourages sparsity in model parameters
    for param in model.parameters():
        loss += l1_lambda * torch.sum(torch.abs(param))

    return loss


if __name__ == "__main__":
    train()
