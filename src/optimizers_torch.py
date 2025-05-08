import torch

def compute_loss_torch(gamma, Z, lambd):
    """
    gamma: (T,) - 1D tensor, trajectory
    Z: (X, T) - 2D tensor, spatial signal
    lambd: regularization weight
    """
    T = gamma.shape[0]
    X = Z.shape[0]

    # Interpolate Z at fractional gamma positions
    gamma_floor = torch.clamp(gamma.long(), 0, X - 2)
    gamma_frac = gamma - gamma_floor.float()

    z0 = Z[gamma_floor, torch.arange(T)]
    z1 = Z[gamma_floor + 1, torch.arange(T)]
    z_interp = (1 - gamma_frac) * z0 + gamma_frac * z1

    signal_term = -z_interp.sum()
    smoothness_term = lambd * torch.sum((gamma[1:] - gamma[:-1]) ** 2)
    loss = signal_term + smoothness_term
    return loss

def gradient_descent_torch(Z, lambd=0.1, eta=0.1, max_iters=100):
    """
    Z: (X, T) - PyTorch tensor
    Returns: optimized gamma and loss curve
    """
    T = Z.shape[1]
    gamma = torch.argmax(Z, dim=0).float().clone().detach()
    gamma.requires_grad_(True)
    optimizer = torch.optim.SGD([gamma], lr=eta)

    losses = []
    for _ in range(max_iters):
        optimizer.zero_grad()
        loss = compute_loss_torch(gamma, Z, lambd)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return gamma.detach().numpy(), losses

# ---- Momentum Gradient Descent ----
def gradient_descent_momentum_torch(Z, lambd=0.1, eta=0.1, beta=0.9, max_iters=100):
    T = Z.shape[1]
    gamma = torch.argmax(Z, dim=0).float().clone().detach().requires_grad_(True)
    v = torch.zeros_like(gamma)
    losses = []

    for _ in range(max_iters):
        gamma.grad = None
        loss = compute_loss_torch(gamma, Z, lambd)
        loss.backward()
        v = beta * v + eta * gamma.grad
        gamma.data -= v
        losses.append(loss.item())

    return gamma.detach().numpy(), losses

# ---- Nesterov Accelerated Gradient ----
def nesterov_accelerated_gradient_torch(Z, lambd=0.1, eta=0.1, beta=0.9, max_iters=100):
    T = Z.shape[1]
    gamma = torch.argmax(Z, dim=0).float().clone().detach()
    v = torch.zeros_like(gamma)
    gamma.requires_grad = True

    losses = []
    for _ in range(max_iters):
        lookahead = (gamma - beta * v).detach().requires_grad_(True)
        loss = compute_loss_torch(lookahead, Z, lambd)
        loss.backward()
        v = beta * v + eta * lookahead.grad
        gamma.data -= v
        losses.append(loss.item())

    return gamma.detach().numpy(), losses
