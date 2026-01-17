__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json
import random
import torch
from datetime import datetime
from pathlib import Path
from torch import device, Tensor
from pathlib import Path
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from config.hyperparameters import HYPERPARAMETERS
from models.model import ForwardModel
from features.dataset import FeaturesDataset


def train(
    train_data: FeaturesDataset,
    eval_data: FeaturesDataset,
    directory: Path | str = "outputs/models",
    parameters: dict = HYPERPARAMETERS["forward"],
) -> ForwardModel:
    """
    Main function for training the transformer forward model.
    :param train_data: Training dataset for the model
    :param eval_data: Evaluation dataset for the model
    :param directory: Directory to save trained models
    :param parameters: Dictionary of training parameters, defaults to config
    :return: Trained ForwardModel, dictionary of training losses
    """

    batch_size = parameters["batch_size"]
    epochs = parameters["num_epochs"]
    alpha = parameters["alpha"]
    lambda_l2 = parameters["lambda_l2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data loaders, model, and optimizer
    print("\nInitializing training... ")
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_eval = DataLoader(eval_data, batch_size=batch_size)
    model = _init_model(train_data, parameters, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=lambda_l2)

    # Train and evaluate the model over many epochs
    print(f"\nTraining for {epochs} epochs on device {device}...")
    losses = {"train": [], "eval": []}
    ics = {"train": [], "eval": []}
    for epoch in range(epochs):
        report = f"Epoch {epoch+1}/{epochs}"

        train_loss, train_ic = _train_epoch(model, loader_train, optimizer, parameters)
        losses["train"].append(train_loss)
        ics["train"].append(train_ic)
        report += f", train loss {train_loss:.4f}, train IC {train_ic:.4f}"

        eval_loss, eval_ic = _eval_epoch(model, loader_eval, parameters)
        losses["eval"].append(eval_loss)
        ics["eval"].append(eval_ic)
        report += f", eval loss {eval_loss:.4f}, eval IC {eval_ic:.4f}"

        print(report, end="\n")

    # Save the trained model and parameters to disk
    statistics = {"losses": losses, "ics": ics}
    _save_model(model, parameters, directory, statistics, train_data, eval_data)
    return model, losses


def _init_model(
    dataset: FeaturesDataset, parameters: dict, device: device
) -> ForwardModel:
    """
    Initialize the transformer forward model on the specified device.
    :param dataset: Dataset to initialize the model with
    :param parameters: Dictionary of model parameters
    :param device: Device to initialize the model on
    :return: Initialized ForwardModel instance
    """
    units_in = dataset[0][0].shape[-1]
    model = ForwardModel(
        units_in=units_in,
        units_hidden=parameters["units_hidden"],
        num_layers=parameters["num_layers"],
        num_heads=parameters["num_heads"],
    )
    model.to(device)

    # Sampling data for normalization parameters avoids memory issues
    sample_size = min(2**16, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)
    sample = torch.stack([dataset[i][0] for i in sample_indices])
    sample = sample.to(device)
    model.initialize(sample)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {params} parameters", end="\n")
    return model


def _train_epoch(
    model: ForwardModel,
    loader: DataLoader,
    optimizer: Optimizer,
    parameters: dict,
) -> tuple[float, float]:
    """
    Train the model for one epoch over the provided DataLoader.
    :param model: Transformer model to train
    :param loader: DataLoader for training data
    :param optimizer: Optimizer for updating model parameters
    :param parameters: Dictionary of training parameters
    :return: Average loss and IC over the epoch
    """
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    y_hats = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss, y_hat = _calc_loss(model, x, y, parameters)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_hats.extend(y_hat[:, 0].detach().cpu().numpy())
        ys.extend(y[:, 0].cpu().numpy())  # Extract mu values for IC
    mean_loss = total_loss / len(loader)
    ic, _ = spearmanr(y_hats, ys)
    return mean_loss, ic


def _eval_epoch(
    model: ForwardModel, loader: DataLoader, parameters: dict
) -> tuple[float, float]:
    """
    Evaluate the model for one epoch over the provided DataLoader.
    :param model: Transformer model to evaluate
    :param loader: DataLoader for evaluation data
    :param parameters: Dictionary of training parameters
    :return: Average loss and IC over the epoch
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_actuals = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss, y_hat = _calc_loss(model, x, y, parameters)
            total_loss += loss.item()
            all_preds.extend(y_hat[:, 0].cpu().numpy())
            all_actuals.extend(y[:, 0].cpu().numpy())  # Extract mu values for IC
    average_loss = total_loss / len(loader)
    ic, _ = spearmanr(all_preds, all_actuals)
    return average_loss, ic


def _calc_loss(
    model: ForwardModel, x: Tensor, y: Tensor, parameters: dict
) -> tuple[Tensor, Tensor]:
    """
    Calculate the loss based on specified loss type.
    :param model: Transformer model to calculate loss
    :param x: Input feature tensor
    :param y: True target tensor of shape (batch, 2) with [return, volatility]
    :param parameters: Dictionary of training parameters
    :return: Loss tensor and model outputs
    """
    y_hat = model(x)
    mu_hat, sigma_hat = y_hat[:, 0], y_hat[:, 1]
    sigma_hat = torch.clamp(sigma_hat, min=2**-12, max=2**3)
    mu = y[:, 0]
    sigma = y[:, 1]
    
    loss_type = parameters["loss_type"]
    lambda_mu = parameters["lambda_mu"]

    # Mean squared error loss optimizes direct fit to targets
    if loss_type == "mse":
        mu_loss = ((mu - mu_hat) ** 2).mean()
        sigma_loss = ((sigma_hat - sigma) ** 2).mean()
        loss = lambda_mu * mu_loss + sigma_loss
    # Negative log-likelihood loss optimizes fit to Gaussian distribution
    else:
        score_loss = (mu - mu_hat) ** 2 / (2 * sigma_hat**2)
        sigma_loss = torch.log(sigma_hat)
        loss = lambda_mu * score_loss.mean() + sigma_loss.mean()

    return loss, y_hat


def _save_model(
    model: ForwardModel,
    parameters: dict,
    directory: Path | str,
    statistics: dict,
    train_data: FeaturesDataset,
    eval_data: FeaturesDataset,
):
    """
    Save the trained model to disk in a timestamped directory.
    :param model: Trained ForwardModel instance
    :param parameters: Dictionary of model parameters
    :param directory: Base directory for saving the model
    :param stats: Dictionary with training statistics
    :param train_data: Training dataset
    :param eval_data: Evaluation dataset
    """
    # Timestamp directory to prevent overwriting existing models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(directory) / f"forward_{timestamp}"
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / "forward_model.pth"
    torch.save(model, model_path)

    params_path = directory / "forward_hyperparameters.json"
    with open(params_path, "w") as f:
        json.dump(parameters, f)

    manifest_path = directory / "manifest.json"
    manifest = _to_manifest(model, parameters, statistics, train_data, eval_data)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Model and parameters saved to {directory}")
    print(f"Manifest: {manifest_path}")


def _to_manifest(
    model: ForwardModel,
    parameters: dict,
    statistics: dict,
    train_data: FeaturesDataset,
    eval_data: FeaturesDataset,
) -> dict:
    """
    Create manifest dictionary with model training metadata.
    :param model: Trained ForwardModel instance
    :param parameters: Dictionary of model parameters
    :param statistics: Dictionary with training statistics
    :param train_data: Training dataset
    :param eval_data: Evaluation dataset
    :return: Manifest dictionary
    """
    return {
        "model_file": "forward_model.pth",
        "created_at": datetime.now().isoformat(),
        "train_dataset": str(train_data._path) if train_data._path else None,
        "eval_dataset": str(eval_data._path) if eval_data._path else None,
        "hyperparameters": parameters,
        "performance": statistics,
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
