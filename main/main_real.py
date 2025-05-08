# main_real.py
from src.config import DATA_PATHS, USE_MASK, LAMBDA, ETA, MAX_ITERS
from src.preprocessing import extract_segments_from_mat, collapse_segments_to_Z
from src.runner import run_optimizer
from src.visualization import plot_results
from utils.experiment_logger import create_experiment_folder, save_config_summary
from utils.extract_data import load_entry_and_extract_segments
from src.optimizers import rmse, relative_rmse
import matplotlib.pyplot as plt

import torch
import numpy as np

# --- Define Experiment Config ---
config = {
    "experiment_type": "real",          # 'real' or 'synthetic'
    "file_mode": "grab_mec",            # 'grab_mec', 'grab_control', 'chat_control'
    "data_source": "img",               # 'img', 'foreground', or 'dFof', unspecified
    "use_mask": True,                   # True → apply ROI mask -> only for chat_control
    "collapse_method": "mean",          # 'mean', 'tucker', 'cp', 'pca
    "spatial_dim": "height",             # Collapse along 'width' or 'height'
    "time_average": True,               # Average across time steps within segment

    # Optimizer settings
    "optimizer": "gd_torch",                  # 'gd', 'momentum', 'nag', "gd_torch", "momentum_torch" , "nag_torch"
    "eta": ETA,
    "lambda": LAMBDA,
    "max_iters": MAX_ITERS,

    # Preprocessing
    "segment_length": 5,

    # File selection
    "file_subset": "all",               # e.g., 'file1_only', 'file2_only', 'file3_only', 'all'; -> only for chat_control

}

# --- Create Experiment Folder ---
experiment_path = create_experiment_folder(config)
save_config_summary(config, experiment_path)

# --- Preprocessing ---
if config["file_mode"] == "grab_mec":
    file_path = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\mec_Estim.mat"
    dfof_signal, _, x_tensor = load_entry_and_extract_segments(file_path, segment_length=config["segment_length"])
    print(dfof_signal.min(), dfof_signal.max(), dfof_signal.mean())


elif config["file_mode"] == "grab_control":
    file_path = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\control_Estim.mat"
    dfof_signal, _, x_tensor = load_entry_and_extract_segments(file_path, segment_length=config["segment_length"])
    print(dfof_signal.min(), dfof_signal.max(), dfof_signal.mean())


elif config["file_mode"] == "chat_control":
    if config["file_subset"] == "all":
        file_paths = DATA_PATHS
    elif config["file_subset"] == "file1_only":
        file_paths = [DATA_PATHS[0]]
    elif config["file_subset"] == "file2_only":
        file_paths = [DATA_PATHS[1]]
    elif config["file_subset"] == "file3_only":
        file_paths = [DATA_PATHS[2]]
    else:
        raise ValueError("Invalid file_subset option")

    x_tensor = extract_segments_from_mat(
        file_paths,
        segment_length=config["segment_length"],
        use_mask=config["use_mask"],
        data_source=config["data_source"]
    )

else:
    raise ValueError("Invalid file_mode")

print(f"Shape of resulting tensor: {x_tensor.shape}")
print(f"Data type: {x_tensor.dtype}")
print(f"Max value: {x_tensor.max().item():.2f}")
print(f"Min value: {x_tensor.min().item():.2f}")

# --- Collapse Z(x, t) ---
Z = collapse_segments_to_Z(
    x_tensor,
    spatial_dim=config["spatial_dim"],
    time_average=config["time_average"],
    method=config["collapse_method"]
)
print(f"Collapsed Z shape: {Z.shape}")

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.imshow(Z, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel("Time")
plt.ylabel("Spatial Position (X)")
plt.title("Z(x, t) - Collapsed Input Matrix")
plt.tight_layout()
plt.show()


# --- Optimization ---
gamma, loss = run_optimizer(Z.numpy(), method=config["optimizer"])

# --- Visualization & Save ---
plot_results(Z, gamma, loss, title="γ(t) over Z(x,t) - Real Data", save_dir=experiment_path)

# --- Post-Hoc RMSE Evaluation Against dF/F₀ (if available) ---
if config["file_mode"] in ["grab_control", "grab_mec"]:
    aligned_dfof = dfof_signal[0:len(gamma)] 

    if len(aligned_dfof) == len(gamma):
        rmse_val = rmse(gamma, aligned_dfof)
        rel_rmse_val = relative_rmse(gamma, aligned_dfof)

        # Save metrics to file
        with open(f"{experiment_path}/metrics.txt", "w") as f:
            f.write(f"RMSE: {rmse_val:.4f}\n")
            f.write(f"Relative RMSE: {rel_rmse_val:.6f}\n")

        print(f"[Post-Hoc] RMSE vs. dF/F₀: {rmse_val:.4f}")
        print(f"[Post-Hoc] Relative RMSE: {rel_rmse_val:.6f}")
    else:
        print(f"[WARNING] Length mismatch: dF/F₀ ({len(aligned_dfof)}) vs. γ(t) ({len(gamma)})")

plt.figure(figsize=(12, 4))
plt.plot(gamma, label='γ(t)', linewidth=2)
plt.plot(aligned_dfof, label='dF/F₀ (ground truth)', linestyle='--')
plt.title("γ(t) vs. dF/F₀")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{experiment_path}/gamma_vs_dfof.png")
plt.show()

import matplotlib.pyplot as plt
plt.plot(dfof_signal)
plt.title("Raw dF/F₀ Signal from MEC File")
plt.xlabel("Time")
plt.ylabel("dF/F₀")
plt.show()

# --- Post-Hoc Evaluation: Signal Walked by γ(t) ---
if config["file_mode"] in ["grab_control", "grab_mec"]:
    # Extract the trace walked by γ(t)
    T = len(gamma)
    H, W, _ = x_tensor.shape[1:4]
    center_y = H // 2
    walked_trace = [Z[int(gamma[t]), t].item() for t in range(len(gamma))]



    # Save trace as .npy
    np.save(f"{experiment_path}/walked_trace.npy", walked_trace)

    # Visualize γ(t) vs walked trace
    plt.figure(figsize=(12, 4))
    plt.plot(gamma, label='γ(t)', linewidth=2)
    plt.plot(walked_trace, label='Walked signal intensity', linestyle='--')
    plt.title("γ(t) vs. Intensity Along Path")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{experiment_path}/gamma_vs_walked_trace.png")
    plt.show()

# Save raw results
np.save(f"{experiment_path}/gamma.npy", gamma)
np.save(f"{experiment_path}/loss.npy", loss)





