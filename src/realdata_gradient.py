# Apply trajectory optimizer to real ACh imaging data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from optimizers import gradient_descent  # or import from a shared optimizer module

# --- Step 1: Load a .mat file ---
file_path = r"..\..\..\Work\Acetylcholine\Data\Figure_2\Figure 2\ChAt_GCamp_eStim\file1.mat"
Data = h5py.File(file_path, 'r')

# Determine control key
control_key = 'control1' if 'control1' in Data else 'control'
control = Data[control_key]
ref = control[0, 0]
entry = Data[ref]

# --- Step 2: Extract raw image and ROI ---
img = np.array(entry['img'])        # Shape: (M, N, T)
img = np.transpose(img, (1, 2, 0))  # To (H, W, T)
roi_mask = np.array(entry['foregroundoverlap']).astype(bool)  # Shape: (H, W)

print("Image shape:", img.shape)

# --- Step 3: Collapse over Y-axis to get Z(x,t) ---
masked_img = np.where(roi_mask[..., None], img, np.nan)
Z_real = np.nanmean(masked_img, axis=0)  # Shape: (W, T)

# Optional: z-score normalization
Z_real = (Z_real - np.nanmean(Z_real)) / np.nanstd(Z_real)

# --- Step 4: Run trajectory optimization ---
gamma_real, loss_real = gradient_descent(Z_real, lambd=0.1, eta=0.1, max_iters=100)

# --- Step 5: Visualize results ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(Z_real, aspect='auto', origin='lower')
plt.plot(gamma_real, color='cyan', label='Estimated γ(t)')
plt.title("Real Data: γ(t) over Z(x,t)")
plt.xlabel("Time")
plt.ylabel("Spatial Position (X)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_real)
plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()