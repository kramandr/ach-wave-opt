import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import (
    decompose_tensor_to_Z_tucker,
    decompose_tensor_to_Z_cp,
    collapse_with_pca
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Simulate synthetic 5D tensor: (N, H, W, L, 1)
N, H, W, L = 30, 32, 32, 5
x_tensor = torch.rand(N, H, W, L, 1, dtype=torch.float32)

# Run decompositions
print("Running Tucker decomposition...")
Z_tucker = decompose_tensor_to_Z_tucker(x_tensor, rank=(20, 20, 20, 5))

print("Running CP decomposition...")
Z_cp = decompose_tensor_to_Z_cp(x_tensor, rank=10)

print("Running PCA-based collapse...")
Z_pca = collapse_with_pca(x_tensor, n_components=1)

# Visualize
def show_Z(Z, title):
    if isinstance(Z, torch.Tensor):
        Z = Z.numpy()
    print(f"{title} → shape: {Z.shape}, min: {Z.min():.3f}, max: {Z.max():.3f}")
    plt.figure(figsize=(10, 4))
    plt.imshow(Z, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

show_Z(Z_tucker, "Z from Tucker Decomposition")
show_Z(Z_cp,     "Z from CP Decomposition")
show_Z(Z_pca,    "Z from PCA-based Collapse")
