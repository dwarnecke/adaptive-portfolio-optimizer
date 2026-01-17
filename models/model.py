__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import torch
from torch import nn


class ForwardModel(nn.Module):
    """
    Transformer model to predict 20-day forward returns and volatility.
    """

    def __init__(
        self,
        units_in: int,
        units_hidden: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize the forward return and volatility transformer model.
        :param units_in: Number of input features
        :param units_hidden: Number of hidden units in transformer layers
        :param num_layers: Number of hidden transformer layers
        :param num_heads: Number of attention heads in transformer layers
        :param dropout: Dropout probability for regularization
        """
        super().__init__()

        # Register the normalization parameters as buffers
        self.register_buffer("_mean", torch.zeros(1, 1, units_in))
        self.register_buffer("_std", torch.ones(1, 1, units_in))
        self.register_buffer("_initialized", torch.tensor(False))
        
        layer = nn.TransformerEncoderLayer(
            d_model=units_hidden,
            nhead=num_heads,
            dim_feedforward=4 * units_hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.input_layer = nn.Linear(units_in, units_hidden)
        self.output_layer = nn.Linear(units_hidden, 2)

    def initialize(self, x):
        """
        Initialize normalization parameters based on input data.
        :param x: Input tensor of shape (num_samples, sequence_len, units_in)
        """
        means = x.mean(dim=(0, 1), keepdim=True)
        stds = x.std(dim=(0, 1), keepdim=True)
        stds = torch.clamp(stds, min=0.0001)

        # Copy the statistics into the registered buffers
        self._mean.copy_(means)
        self._std.copy_(stds)
        self._initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        :param x: Input tensor of shape (batch_size, sequence_len, units_in)
        :return: Output tensor of shape (batch_size, 2) with return and volatility
        """
        x_norm = self._normalize(x)
        y = self.input_layer(x_norm)

        # Causal attention mask prevents attention to future time steps
        _, length, _ = y.shape
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=y.device),
            diagonal=1,
        )
        y = self.encoder(y, mask=mask)

        # Softplus volatility output to ensure positivity
        y = y[:, -1]
        y = self.output_layer(y)
        mu = y[:, 0]
        sigma = nn.functional.softplus(y[:, 1])

        return torch.stack((mu, sigma), dim=1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor on the training means and deviations.
        :param x: Input tensor to normalize
        :return: Normalized tensor
        """
        if not self._initialized:
            raise ValueError("Normalization parameters not initialized.")
        return (x - self._mean) / self._std
