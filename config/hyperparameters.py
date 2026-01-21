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
        "alpha": 2**-8,
        "batch_size": 1024,
        "num_epochs": 32,
        "num_layers": 1,
        "num_heads": 2,
        "units_hidden": 32,
        "dropout": 0,
        "lambda_l2": 2**-6,
        "lambda_mu": 2**0,  # mean return loss coefficient
        "loss_type": "nll",  # loss function type: "nll" or "mse"
        "ic_cutoff": 0.135,  # early stopping threshold for eval IC (None = no early stopping)
    },
    "portfolio": {
        "covariance_shrinkage": 2**-1,  # covariance shrinkage factor
        "covariance_mix": 2**-1,  # mixing weight for forward vs historical covariance 
        "length": 250,  # rolling covariance window length
        "risk_free_rate": 0,  # risk-free rate 
        "slippage_bps": 5,  # slippage in basis points
        "commission_bps": 2,  # commission in basis points
        "min_scalar": 2**-2,  # minimum weight threshold
        "max_scalar": 2**3,  # maximum weight threshold
        "max_leverage": 2**1,  # maximum portfolio leverage
    },
}
