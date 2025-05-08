import h5py
from pathlib import Path
import torch

# Data paths (from memory)
DATA_PATHS = [
    r"..\..\Work\Acetylcholine\Data\Figure_2\Figure 2\ChAt_GCamp_eStim\file1.mat",
    r"..\..\Work\Acetylcholine\Data\Figure_2\Figure 2\ChAt_GCamp_eStim\file2.mat",
    r"..\..\Work\Acetylcholine\Data\Figure_2\Figure 2\ChAt_GCamp_eStim\file3.mat"
]

GRAB_PATHS = [
    r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\control_Estim.mat",
    r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\mec_Estim.mat"
]

grab_summary = []

for path in GRAB_PATHS:
    with h5py.File(path, 'r') as f:
        top_keys = list(f.keys())
        data_key = 'control' if 'control' in f else 'mec'
        dataset = f[data_key]
        dataset = f[data_key]
        shape = dataset.shape
        entry_refs = [dataset[i, 0] for i in range(min(3, shape[0]))]  # check first few rows
        entry_structs = [list(f[ref].keys()) for ref in entry_refs]

        summary = {
            "file": Path(path).name,
            "dataset_key": data_key,
            "shape": shape,
            "sample_entry_keys": entry_structs
        }
        grab_summary.append(summary)

#print(grab_summary)

with h5py.File(GRAB_PATHS[0], 'r') as f:  # control_Estim.mat
    dataset = f['control']
    ref = dataset[0, 10]  # column 10 = index 10
    entry = f[ref]
    print("Keys at (0,10):", list(entry.keys()))


# One-liner-style exploration for control group structure
exploration_summary = []

for path in DATA_PATHS:
    with h5py.File(path, 'r') as f:
        top_keys = list(f.keys())
        control_key = 'control1' if 'control1' in f else 'control'
        control_data = f[control_key]
        shape = control_data.shape
        sample_entry_ref = control_data[0, 0]
        entry = f[sample_entry_ref]
        entry_keys = list(entry.keys())

        summary = {
            "file": Path(path).name,
            "shape": shape,
            "top_keys": top_keys,
            "entry_keys": entry_keys
        }
        exploration_summary.append(summary)

#print(exploration_summary)


with h5py.File(DATA_PATHS[0], 'r') as f:
    dataset = f['control1'] if 'control1' in f else f['control']
    ref = dataset[0, 0]
    entry = f[ref]
    print(list(entry.keys()))
    for key in entry.keys():
        print(f"{key}: {entry[key]}")


import numpy as np
import h5py
from typing import List, Tuple

def load_dataset_cells(file_path: str, dataset_key: str) -> List[h5py.Group]:
    """Return all valid entries (cells) from a control or mec dataset."""
    entries = []
    with h5py.File(file_path, 'r') as f:
        dataset = f[dataset_key]
        print(f"Dataset shape: {dataset.shape}")
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                try:
                    ref = dataset[i, j]
                    obj = f[ref]
                    if isinstance(obj, h5py.Group):
                        entries.append(obj)
                    else:
                        print(f"Entry at ({i},{j}) is not a Group: {type(obj)}")
                except Exception as e:
                    print(f"Skipping entry at ({i},{j}): {e}")
    print(f"Total valid entries loaded: {len(entries)}")
    return entries

def load_single_entry(file_path: str, dataset_key: str = "control") -> h5py.Group:
    """Accesses a single entry Group from the .mat file."""
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        key = 'control1' if 'control1' in keys else dataset_key
        print(f"Using dataset key: {key}")
        ref = f[key][0, 0]
        entry = f[ref]
        print(f"Keys in entry: {list(entry.keys())}")
        return entry  # h5py.Group


def build_segment_tensor(entry, segment_length=5, use_mask=True, data_source="img"):
    if data_source == "img":
        signal = np.array(entry['img']).transpose(1, 2, 0)  # (H, W, T)
    elif data_source == "dFof":
        signal = np.array(entry['dFof']).squeeze()[None, None, :]  # (1, 1, T)
    elif data_source == "foreground":
        signal = np.array(entry['foreground']).squeeze()[None, None, :]  # (1, 1, T)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    if use_mask and "foregroundoverlap" in entry:
        mask = np.array(entry['foregroundoverlap']).astype(bool)
        signal = np.where(mask[..., None], signal, 0.0)

    segments = []
    for j in range(signal.shape[-1] - segment_length):
        segment = signal[..., j:j+segment_length]
        segment = torch.from_numpy(segment).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 5, 1)
        segments.append(segment)

    return torch.cat(segments, dim=0)


def load_all_dfof(file_path: str, dataset_key: str) -> List[np.ndarray]:
    """Load all dF/F₀ traces from a file."""
    entries = load_dataset_cells(file_path, dataset_key)
    return [np.array(entry["dFof"]).squeeze() for entry in entries if "dFof" in entry]


def load_all_img(file_path: str, dataset_key: str, apply_roi: bool = False) -> List[np.ndarray]:
    """Load all raw img stacks, optionally applying foreground ROI masks."""
    entries = load_dataset_cells(file_path, dataset_key)
    result = []

    for entry in entries:
        if "img" in entry:
            img = np.array(entry["img"]).transpose(1, 2, 0)  # (H, W, T)
            if apply_roi and "foregroundoverlap" in entry:
                roi = np.array(entry["foregroundoverlap"]).astype(bool)
                img = np.where(roi[..., None], img, 0.0)
            result.append(img)
    return result


def load_all_foreground_rois(file_path: str, dataset_key: str) -> List[np.ndarray]:
    """Load all binary foreground ROI masks."""
    entries = load_dataset_cells(file_path, dataset_key)
    return [np.array(entry["foregroundoverlap"]).astype(bool) for entry in entries if "foregroundoverlap" in entry]




# Example file for testing (manually update path if needed)
GRAB_CONTROL = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\control_Estim.mat"
GRAB_MEC = r"C:\Users\andre\OneDrive - isu.edu\Dokumente\Spring 2025\Work\Acetylcholine\Data\Figure_2\Figure 2\GRAB_da_eStim\mec_Estim.mat"

import torch
import numpy as np
import h5py

with h5py.File(GRAB_MEC, 'r') as f:
    control_key = 'mec' if 'mec' in f.keys() else 'mec'
    print(f"Using dataset key: {control_key}")
    ref = f[control_key][0, 0]
    entry = f[ref]

    print(f"Keys in entry: {list(entry.keys())}")

    # Extract raw image (H, W, T)
    img_stack = np.array(entry["img"]).transpose(1, 2, 0)
    print(f"img shape: {img_stack.shape}")

    # Extract dF/F₀ (T,)
    dfof_signal = np.array(entry["dFof"]).squeeze()
    print(f"dF/F₀ shape: {dfof_signal.shape}")

    # Extract ROI
    roi_mask = np.array(entry["foregroundoverlap"]).astype(bool)
    print(f"ROI shape: {roi_mask.shape}")

    # Apply ROI mask to image
    masked_img = np.where(roi_mask[..., None], img_stack, 0.0)
    print(f"Masked img shape: {masked_img.shape}")

    # Extract segment tensor (optional)
    def build_segment_tensor(signal, segment_length=5):
        segments = []
        for j in range(signal.shape[-1] - segment_length):
            segment = signal[..., j:j+segment_length]
            segment = torch.from_numpy(segment).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 5, 1)
            segments.append(segment)
        return torch.cat(segments, dim=0)

    segment_tensor = build_segment_tensor(masked_img, segment_length=5)
    print(f"Segment tensor shape: {segment_tensor.shape}")

    print("\n--- Sample Values ---")

# First 5 values from dF/F₀
print("dF/F₀ values [0:5]:", dfof_signal[:5].flatten())

# Center pixel values from masked image at middle frame
mid_frame = masked_img[:, :, masked_img.shape[2] // 2]
h, w = mid_frame.shape
print("\nMasked mid-frame center 5x5 pixels:")
print(mid_frame[h//2 - 2:h//2 + 3, w//2 - 2:w//2 + 3])

# First segment sample
print("\nFirst segment tensor slice [0] (shape {}):".format(segment_tensor[0].shape))
print(segment_tensor[0][:5, :5, :, 0])  # Top-left 5x5 slice of first frame in segment






