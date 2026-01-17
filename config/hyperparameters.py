__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

"""
Configuration file for model hyperparameters.
"""

HYPERPARAMETERS = {
    "features": {
        "length": 60,  # feature window length
        "num_states": 3,  # number of market regimes
        "max_iter": 50,  # maximum EM iterations
        "tol": 1e-4,  # convergence tolerance
    },
    "forward": {
        "alpha": 2**-10,  # learning rate
        "batch_size": 1024,
        "num_epochs": 8,
        "num_layers": 1,
        "num_heads": 2,
        "units_hidden": 32,
        "lambda_l2": 2**-16,  # L2 regularization
        "lambda_mu": 2**2,  # mean return loss coefficient
        "loss_type": "mse",  # loss function type: "nll" or "mse"
    },
    "portfolio": {
        "covariance_shrinkage": 2**-1,  # covariance shrinkage factor
        "length": 120,  # rolling covariance window length
        "rate0": 0.02,  # risk-free rate
        "slippage_bps": 5,  # slippage in basis points
        "commission_bps": 2,  # commission in basis points
        "dust_threshold": 0.001,  # minimum weight threshold
    },
}
