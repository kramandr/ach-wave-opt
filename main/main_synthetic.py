from src.runner import run_optimizer
from src.visualization import plot_results
from src.optimizers import rmse, relative_rmse
from data.synthetic import generate_synthetic_data, generate_realistic_synthetic_data
from utils.experiment_logger import create_experiment_folder, save_config_summary
from src.optimizers import rmse, relative_rmse
import matplotlib.pyplot as plt
import numpy as np

# --- Define Experiment Configuration ---
config = {
    "optimizer": "nag_torch",
    "collapse_method": "mean",
    "lambda": 0.1,
    "eta": 0.1,
    "max_iters": 100,
    "noise_std": 0.10,
    "seed": 94,
    "data_variant": "compare"  # choose: "original", "realistic", "compare"
}

# --- Setup Experiment Folder ---
save_dir = create_experiment_folder(config)
save_config_summary(config, save_dir)

# --- Generate & Run Optimization ---
if config["data_variant"] == "original":
    Z, gamma_true = generate_synthetic_data(noise_std=config["noise_std"], seed=config["seed"])

elif config["data_variant"] == "realistic":
    Z, gamma_true = generate_realistic_synthetic_data(seed=config["seed"], noise_std=config["noise_std"])

elif config["data_variant"] == "compare":
    Z1, g1 = generate_synthetic_data(noise_std=config["noise_std"], seed=config["seed"])
    Z2, g2 = generate_realistic_synthetic_data(seed=config["seed"], noise_std=config["noise_std"])
    gamma1, loss1 = run_optimizer(Z1, method=config["optimizer"])
    gamma2, loss2 = run_optimizer(Z2, method=config["optimizer"])

    # Compute RMSE
    rmse1 = rmse(gamma1, g1)
    rel_rmse1 = relative_rmse(gamma1, g1)
    rmse2 = rmse(gamma2, g2)
    rel_rmse2 = relative_rmse(gamma2, g2)

    # Plot comparison
    plot_results(Z1, gamma1, loss1, gamma_true=g1, title="Original Synthetic", save_dir=save_dir)
    plot_results(Z2, gamma2, loss2, gamma_true=g2, title="Realistic Synthetic", save_dir=save_dir)

    # Save results
    np.save(f"{save_dir}/gamma_original.npy", gamma1)
    np.save(f"{save_dir}/gamma_realistic.npy", gamma2)
    np.save(f"{save_dir}/gamma_true_original.npy", g1)
    np.save(f"{save_dir}/gamma_true_realistic.npy", g2)
    np.save(f"{save_dir}/loss_original.npy", loss1)
    np.save(f"{save_dir}/loss_realistic.npy", loss2)

    # Save metrics
    with open(f"{save_dir}/metrics_compare.txt", "w") as f:
        f.write("[Original Synthetic]\n")
        f.write(f"RMSE: {rmse1:.4f}\n")
        f.write(f"Relative RMSE: {rel_rmse1:.6f}\n\n")
        f.write("[Realistic Synthetic]\n")
        f.write(f"RMSE: {rmse2:.4f}\n")
        f.write(f"Relative RMSE: {rel_rmse2:.6f}\n")

    print(f"[Original Synthetic] RMSE: {rmse1:.4f}, Relative RMSE: {rel_rmse1:.6f}")
    print(f"[Realistic Synthetic] RMSE: {rmse2:.4f}, Relative RMSE: {rel_rmse2:.6f}")
    exit()

else:
    raise ValueError("Invalid data_variant. Choose 'original', 'realistic', or 'compare'.")

# --- Optimize (for single variant) ---
gamma_est, loss = run_optimizer(Z, method=config["optimizer"])
rmse_val = rmse(gamma_est, gamma_true)
rel_rmse_val = relative_rmse(gamma_est, gamma_true)
0000
print(f"[Post-Hoc] RMSE: {rmse_val:.4f}")
print(f"[Post-Hoc] Relative RMSE: {rel_rmse_val:.6f}")

with open(f"{save_dir}/metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse_val:.4f}\n")
    f.write(f"Relative RMSE: {rel_rmse_val:.6f}\n")

plot_results(Z, gamma_est, loss, gamma_true=gamma_true,
             title=f"Synthetic Data ({config['data_variant']}): γ(t)", save_dir=save_dir)

# Save raw outputs
np.save(f"{save_dir}/gamma.npy", gamma_est)
np.save(f"{save_dir}/loss.npy", loss)
np.save(f"{save_dir}/gamma_true.npy", gamma_true)