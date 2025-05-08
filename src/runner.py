# runner.py
import torch
import numpy as np

from src.optimizers import (
    gradient_descent,
    gradient_descent_momentum,
    nesterov_accelerated_gradient
)

from src.optimizers_torch import (
    gradient_descent_torch,
    gradient_descent_momentum_torch,
    nesterov_accelerated_gradient_torch
)

from src.config import LAMBDA, ETA, MAX_ITERS

def run_optimizer(Z, method="gd", lambd=LAMBDA, eta=ETA, beta=0.9, max_iters=MAX_ITERS):
    """
    Run specified optimizer on Z(x, t).
    Parameters:
        Z: ndarray or tensor (2D: [X, T])
        method: 'gd', 'momentum', 'nag', 'gd_torch', 'momentum_torch', 'nag_torch'
    Returns:
        gamma: estimated trajectory (1D)
        loss: list of loss values
    """
    if method == "gd":
        return gradient_descent(Z, lambd, eta, max_iters)
    elif method == "momentum":
        return gradient_descent_momentum(Z, lambd, eta, beta, max_iters)
    elif method == "nag":
        return nesterov_accelerated_gradient(Z, lambd, eta, beta, max_iters)

    elif method == "gd_torch":
        return gradient_descent_torch(torch.tensor(Z), lambd, eta, max_iters)
    elif method == "momentum_torch":
        return gradient_descent_momentum_torch(torch.tensor(Z), lambd, eta, beta, max_iters)
    elif method == "nag_torch":
        return nesterov_accelerated_gradient_torch(torch.tensor(Z), lambd, eta, beta, max_iters)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "'gd', 'momentum', 'nag', 'gd_torch', 'momentum_torch', 'nag_torch'"
        )
