import numpy as np
import matplotlib.pyplot as plt
from src.runner import run_optimizer
from src.visualization import plot_Z_and_gamma
from data.synthetic import generate_synthetic_data  # assuming synthetic data for now

Z, gamma_true = generate_synthetic_data()

lambdas = [0.01, 0.1, 1.0]
etas = [0.01, 0.1, 0.5]

results = {}

for lambd in lambdas:
    for eta in etas:
        gamma_est, loss = run_optimizer(Z, method="gd", lambd=lambd, eta=eta)
        rmse = np.sqrt(np.mean((gamma_est - gamma_true) ** 2))
        results[(lambd, eta)] = (gamma_est, loss, rmse)
        print(f"λ={lambd:.2f}, η={eta:.2f} → RMSE={rmse:.4f}")

        # Optional: visualize
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plot_Z_and_gamma(Z, gamma_est, title=f"λ={lambd}, η={eta}")
        plt.plot(gamma_true, '--', label='γ_true', color='magenta')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss)
        plt.title("Loss over Iterations")
        plt.tight_layout()
        plt.show()
