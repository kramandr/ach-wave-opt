import numpy as np

def compute_loss(gamma, Z, lambd):
    signal_term = -np.array([np.interp(gamma[t], np.arange(Z.shape[0]), Z[:, t]) for t in range(len(gamma))])
    smoothness_term = lambd * np.sum(np.diff(gamma)**2)
    return np.sum(signal_term) + smoothness_term

def compute_gradient(gamma, Z, lambd, eps=1e-5):
    grad = np.zeros_like(gamma)
    for t in range(len(gamma)):
        gamma_eps = gamma.copy()
        gamma_eps[t] += eps
        grad[t] = (compute_loss(gamma_eps, Z, lambd) - compute_loss(gamma, Z, lambd)) / eps
    return grad

def gradient_descent(Z, lambd=0.1, eta=0.1, max_iters=100):
    T = Z.shape[1]
    gamma = np.argmax(Z, axis=0).astype(float)
    losses = []
    for _ in range(max_iters):
        loss = compute_loss(gamma, Z, lambd)
        grad = compute_gradient(gamma, Z, lambd)
        gamma -= eta * grad
        losses.append(loss)
    return gamma, losses

def gradient_descent_momentum(Z, lambd=0.1, eta=0.1, beta=0.9, max_iters=100):
    T = Z.shape[1]
    gamma = np.argmax(Z, axis=0).astype(float)
    v = np.zeros_like(gamma)
    losses = []
    for _ in range(max_iters):
        grad = compute_gradient(gamma, Z, lambd)
        v = beta * v + eta * grad
        gamma -= v
        losses.append(compute_loss(gamma, Z, lambd))
    return gamma, losses

def nesterov_accelerated_gradient(Z, lambd=0.1, eta=0.1, beta=0.9, max_iters=100):
    T = Z.shape[1]
    gamma = np.argmax(Z, axis=0).astype(float)
    v = np.zeros_like(gamma)
    losses = []
    for _ in range(max_iters):
        lookahead = gamma - beta * v
        grad = compute_gradient(lookahead, Z, lambd)
        v = beta * v + eta * grad
        gamma -= v
        losses.append(compute_loss(gamma, Z, lambd))
    return gamma, losses

def rmse(gamma_est, gamma_true):
    return np.sqrt(np.mean((gamma_est - gamma_true) ** 2))

def relative_rmse(gamma_est, gamma_true):
    error = rmse(gamma_est, gamma_true)
    range_ = np.max(gamma_true) - np.min(gamma_true)
    return error / range_ if range_ != 0 else np.nan