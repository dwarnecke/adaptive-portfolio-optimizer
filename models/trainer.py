__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import random
import torch
from torch import device, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from models.model import ForwardModel
from features.dataset import FeaturesDataset


class Trainer:
    """Handles training and evaluation of ForwardModel."""

    def __init__(
        self,
        model: ForwardModel,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer: Optimizer,
        parameters: dict,
        device: device,
    ):
        """
        Initialize Trainer with model, data loaders, optimizer, and parameters.
        :param model: ForwardModel to train
        :param train_loader: DataLoader for training data
        :param eval_loader: DataLoader for evaluation data
        :param optimizer: Optimizer for model training
        :param parameters: Training parameters
        :param device: Device to run training on
        """
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.parameters = parameters
        self.device = device

    def train(self, num_epochs: int) -> tuple[dict, dict]:
        """
        Train model for specified number of epochs.
        :param num_epochs: Number of training epochs
        :return: Tuple of losses and best model state
        """
        losses = {"train": [], "eval": []}
        best_eval_loss = float("inf")
        best_model_state = None

        print(f"\nTraining for {num_epochs} epochs on device {self.device}...")
        for epoch in range(num_epochs):
            report = f"Epoch {epoch+1}/{num_epochs}"

            train_loss = self._train_epoch()
            losses["train"].append(train_loss)
            report += f", train loss {train_loss:.4f}"

            eval_loss = self._eval_epoch()
            losses["eval"].append(eval_loss)
            report += f", eval loss {eval_loss:.4f}"

            # Save best model based on eval loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model_state = self.model.state_dict().items()
                best_model_state = {k: v.cpu().clone() for k, v in model_state}
                report += "*"

            print(report)

        # Restore best model after training for optimization
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nRestored best model with eval loss {best_eval_loss:.4f}")
        return losses, best_model_state

    def _train_epoch(self) -> float:
        """
        Train for one epoch over the training set.
        :return: Average loss over all training batches
        """
        self.model.train()
        total_loss = 0.0

        # Train over many batches to update multiple times per epoch
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            loss, _ = self._calc_loss(x, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _eval_epoch(self) -> float:
        """
        Evaluate for one epoch over the evaluation set.
        :return: Average loss over all evaluation batches
        """
        self.model.eval()
        total_loss = 0.0

        # Evaluate without gradient tracking to save computation
        with torch.no_grad():
            for x, y in self.eval_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                loss, _ = self._calc_loss(x, y)
                total_loss += loss.item()
        return total_loss / len(self.eval_loader)

    def _calc_loss(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Calculate loss based on specified loss type.
        :param x: Input features
        :param y: True targets [return, volatility]
        :return: Loss tensor and predictions
        """
        y_hat = self.model(x)
        mu_hat, sigma_hat = y_hat[:, 0], y_hat[:, 1]

        # Clamp predicted volatility for numerical stability
        sigma_hat = torch.clamp(sigma_hat, min=2**-12, max=2**3)
        mu, sigma = y[:, 0], y[:, 1]

        loss_type = self.parameters["loss_type"]
        lambda_mu = self.parameters["lambda_mu"]

        if loss_type == "mse":
            # MSE optimizes direct squared error on mean and volatility
            mu_loss = ((mu - mu_hat) ** 2).mean()
            sigma_loss = ((sigma_hat - sigma) ** 2).mean()
            loss = lambda_mu * mu_loss + sigma_loss
        else:
            # NLL optimizes Gaussian likelihood of targets
            score_loss = (mu - mu_hat) ** 2 / (2 * sigma_hat**2)
            sigma_loss = torch.log(sigma_hat)
            loss = lambda_mu * score_loss.mean() + sigma_loss.mean()

        return loss, y_hat


def initialize_model(
    dataset: FeaturesDataset,
    parameters: dict,
    device: device,
) -> ForwardModel:
    """
    Initialize ForwardModel with normalization.
    :param dataset: Dataset to sample for normalization
    :param parameters: Model parameters
    :param device: Device to initialize on
    :return: Initialized ForwardModel
    """
    units_in = dataset[0][0].shape[-1]
    model = ForwardModel(
        units_in=units_in,
        units_hidden=parameters["units_hidden"],
        num_layers=parameters["num_layers"],
        num_heads=parameters["num_heads"],
        dropout=parameters["dropout"],
    )
    model.to(device)

    # Compute normalization statistics from sample
    sample_size = min(2**16, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)
    sample = torch.stack([dataset[i][0] for i in sample_indices])
    sample = sample.to(device)
    model.initialize(sample)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {params:,} parameters")
    return model
