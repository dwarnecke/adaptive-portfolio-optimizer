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
        "alpha": 2**-6,
        "batch_size": 2**11,
        "num_epochs": 16,
        "num_layers": 1,
        "num_heads": 1,
        "units_hidden": 8,
        "dropout": 0,
        "lambda_l2": 2**-20,
        "lambda_mu": 2**2,  # return loss coefficient
        "loss_type": "mse",  # "nll" or "mse"
    },
    "portfolio": {
        "covariance_shrinkage": 2**-2,  # covariance shrinkage factor
        "covariance_mix": 2**-1,  # mixing weight for covariance, 0=hist, 1=pred
        "length": 250,  # rolling covariance window length
        "risk_free_rate": 0.02,  # risk-free rate
        "slippage_bps": 5,  # slippage in basis points
        "commission_bps": 2,  # commission in basis points
        "min_scalar": 2**0,  # minimum weight threshold
        "max_scalar": 2**6,  # maximum weight threshold
        "max_leverage": 2**3,  # maximum portfolio leverage
        "rebalance_frequency": 2**3,  # trading days between rebalances
    },
}
