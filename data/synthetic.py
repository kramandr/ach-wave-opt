import numpy as np

def generate_synthetic_data(T=100, X=50, sigma=3.0, noise_std=0.0, seed=None):
    """
    Create synthetic Z(x,t) with a moving Gaussian peak.
    Returns:
        Z: np.ndarray of shape (X, T)
        gamma_true: np.ndarray of shape (T,)
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.arange(X)
    Z = np.zeros((X, T))
    gamma_true = np.linspace(10, 40, T)
    for t in range(T):
        Z[:, t] = np.exp(-(x - gamma_true[t])**2 / (2 * sigma**2))
    if noise_std > 0.0:
        Z += np.random.normal(scale=noise_std, size=Z.shape)
    return Z, gamma_true

def generate_realistic_synthetic_data(T=300, X=100, sigma=3.0, noise_std=0.05, seed=None):
    """
    Generates a more realistic Z(x,t) with bursts, jitter, and baseline fluctuations.
    Returns:
        Z: np.ndarray of shape (X, T)
        gamma_true: np.ndarray of shape (T,)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.arange(X)
    Z = np.zeros((X, T))

    # Create jittery, bursty γ(t)
    gamma_true = np.cumsum(np.random.normal(loc=0.1, scale=1.0, size=T)).clip(10, X-10)
    gamma_true = np.round(gamma_true).astype(int)

    for t in range(T):
        peak_intensity = 1.0 + 0.5 * np.sin(0.05 * t) + 0.3 * np.random.randn()  # bursty
        center = gamma_true[t]
        Z[:, t] += peak_intensity * np.exp(-(x - center)**2 / (2 * sigma**2))

    # Add a second "false" path with weaker intensity
    gamma_false = np.cumsum(np.random.normal(loc=0.0, scale=1.0, size=T)).clip(0, X-1).astype(int)
    for t in range(T):
        Z[:, t] += 0.3 * np.exp(-(x - gamma_false[t])**2 / (2 * (sigma * 1.5)**2))

    # Add baseline fluctuations
    baseline = 0.1 + 0.05 * np.sin(0.03 * np.outer(np.linspace(0, 1, X), np.linspace(0, T, T)))
    Z += baseline

    # Add Gaussian noise
    Z += np.random.normal(scale=noise_std, size=Z.shape)

    return Z, gamma_true

