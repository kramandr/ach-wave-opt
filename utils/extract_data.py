import h5py
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

GRAB_CONTROL = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\control_Estim.mat"
GRAB_MEC = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\mec_Estim.mat"

def extract_data_for_regression(file_path, segment_length=5):
    """
    Loads one entry from .mat file and extracts:
    - X: masked image segments, shape (N, H*W*T)
    - y: mean dF/F₀ over each time segment, shape (N,)
    """
    with h5py.File(file_path, 'r') as f:
        if 'control1' in f.keys():
            control_key = 'control1'
        elif 'control' in f.keys():
            control_key = 'control'
        else: 
            control_key = 'mec'
         
        ref = f[control_key][0, 0]
        entry = f[ref]

        img = np.array(entry["img"]).transpose(1, 2, 0)      # (H, W, T)
        roi = np.array(entry["foregroundoverlap"]).astype(bool)
        dfof = np.array(entry["dFof"]).squeeze()            # (T,)

        # Apply mask
        masked = np.where(roi[..., None], img, 0.0)          # (H, W, T)

        # Segment the masked image and corresponding dfof
        segments = []
        dfof_targets = []
        for j in range(masked.shape[-1] - segment_length):
            img_seg = masked[..., j:j+segment_length]        # (H, W, 5)
            dfof_seg = dfof[j:j+segment_length]              # (5,)
            segments.append(img_seg.flatten())               # (H*W*T,)
            dfof_targets.append(dfof_seg.mean())             # scalar target

    X = torch.tensor(np.array(segments), dtype=torch.float32)  # (N, D)
    y = torch.tensor(np.array(dfof_targets), dtype=torch.float32)  # (N,)
    return X, y

X, y = extract_data_for_regression(GRAB_CONTROL)

# Save
torch.save({'X': X, 'y': y}, 'grab_control_regression_data.pt')
print("Saved data to grab_control_regression_data.pt")


def load_entry_and_extract_segments(file_path, segment_length=5):
    """
    Loads one entry from a .mat file and extracts:
    - dF/F₀ signal
    - ROI-masked image stack
    - Segment tensor
    """
    with h5py.File(file_path, 'r') as f:
        dataset_key = 'control' if 'control' in f.keys() else 'mec'
        print(f"\nProcessing: {Path(file_path).name}")
        print(f"Using dataset key: {dataset_key}")
        
        ref = f[dataset_key][0, 0]
        entry = f[ref]
        print(f"Keys in entry: {list(entry.keys())}")

        # Extract components
        img_stack = np.array(entry["img"]).transpose(1, 2, 0)
        print(f"img shape: {img_stack.shape}")

        dfof_signal = np.array(entry["dFof"]).squeeze()
        print(f"dF/F₀ shape: {dfof_signal.shape}")

        roi_mask = np.array(entry["foregroundoverlap"]).astype(bool)
        print(f"ROI shape: {roi_mask.shape}")

        # Apply ROI to image stack
        masked_img = np.where(roi_mask[..., None], img_stack, 0.0)
        print(f"Masked img shape: {masked_img.shape}")

        # Segment the masked stack
        segments = []
        for j in range(masked_img.shape[-1] - segment_length):
            segment = masked_img[..., j:j+segment_length]  # (H, W, segment_length)
            segment = torch.from_numpy(segment).unsqueeze(0).unsqueeze(-1)  # (1, H, W, T, 1)
            segments.append(segment)
        segment_tensor = torch.cat(segments, dim=0)
        print(f"Segment tensor shape: {segment_tensor.shape}")

        # Sample outputs
        print("\n--- Sample Values ---")
        print("dF/F₀ values [0:5]:", dfof_signal[:5].flatten())

        mid_frame = masked_img[:, :, masked_img.shape[2] // 2]
        h, w = mid_frame.shape
        print("\nMasked mid-frame center 5x5 pixels:")
        print(mid_frame[h//2 - 2:h//2 + 3, w//2 - 2:w//2 + 3])

        print("\nFirst segment tensor slice [0] (shape {}):".format(segment_tensor[0].shape))
        print(segment_tensor[0][:5, :5, :, 0])  # Top-left 5x5 spatial slice

    return dfof_signal, masked_img, segment_tensor

def load_entry_and_extract_segments1(file_path, segment_length=5):
    """
    Loads one entry from a .mat file and extracts:
    - dF/F₀ signal
    - ROI-masked image stack
    - Segment tensor
    Also visualizes ROI alignment for sanity checking.
    """
    with h5py.File(file_path, 'r') as f:
        dataset_key = 'control' if 'control' in f.keys() else 'mec'
        print(f"\nProcessing: {Path(file_path).name}")
        print(f"Using dataset key: {dataset_key}")

        ref = f[dataset_key][0, 0]
        entry = f[ref]
        print(f"Keys in entry: {list(entry.keys())}")

        # Load fields
        dfof_signal = np.array(entry["dFof"]).squeeze()
        roi_mask = np.array(entry["foregroundoverlap"]).astype(bool)

        print(f"dF/F₀ shape: {dfof_signal.shape}")
        print(f"ROI mask shape: {roi_mask.shape}")

        # Try 'img', fallback to 'foreground' if shape mismatch
        img_stack = np.array(entry["img"]).transpose(1, 2, 0)
        if img_stack.shape[:2] != roi_mask.shape:
            print("ROI does not match 'img'. Trying 'foreground' as base image...")
            img_stack = np.array(entry["foreground"]).transpose(1, 2, 0)
        else:
            print("Using 'img' as base image.")

        print(f"Image stack shape (after transpose): {img_stack.shape}")

        # Sanity check visualization: ROI on mid-frame
        mid_frame = img_stack[:, :, img_stack.shape[2] // 2]
        plt.figure(figsize=(6, 6))
        plt.imshow(mid_frame, cmap="gray")
        plt.contour(roi_mask, colors='r', linewidths=1)
        plt.title("Mid Frame with ROI Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Apply ROI mask
        masked_img = np.where(roi_mask[..., None], img_stack, 0.0)
        print(f"Masked img shape: {masked_img.shape}")

        # Compute ROI-based dF/F₀ signal as average intensity across the mask
        roi_pixels = img_stack[roi_mask]  # shape: (H*W,)
        roi_pixels = roi_pixels.reshape(-1, img_stack.shape[-1])  # (num_pixels, T)
        roi_mean_trace = roi_pixels.mean(axis=0)  # (T,)
        print(f"Computed ROI-mean trace shape: {roi_mean_trace.shape}")
        # Segment the masked stack
        segments = []
        for j in range(masked_img.shape[-1] - segment_length):
            segment = masked_img[..., j:j+segment_length]  # (H, W, segment_length)
            segment = torch.from_numpy(segment).unsqueeze(0).unsqueeze(-1)  # (1, H, W, T, 1)
            segments.append(segment)

        segment_tensor = torch.cat(segments, dim=0)

        print(f"Segment tensor shape: {segment_tensor.shape}")
        print("Sample dF/F₀ values [0:5]:", dfof_signal[:5].flatten())

    return dfof_signal, masked_img, segment_tensor

# === Run for Control ===
control_path = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\control_Estim.mat"
control_dfof, control_img, control_segments = load_entry_and_extract_segments(control_path)

# === Run for Mec ===
mec_path = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\mec_Estim.mat"
mec_dfof, mec_img, mec_segments = load_entry_and_extract_segments(mec_path)
