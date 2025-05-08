import h5py
import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from sklearn.decomposition import PCA

def extract_segments_from_mat(file_paths, segment_length=5, use_mask=False, data_source="img"):
    """
    Loads .mat files, slices 5-frame segments from full image stack,
    optionally masks with foreground ROI.
    Returns: torch.Tensor of shape (num_segments, H, W, segment_length, 1)
    """
    all_segments = []

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as Data:
            keys = list(Data.keys())
            control_key = 'control1' if 'control1' in keys else 'control'
            ref = Data[control_key][0, 0]
            entry = Data[ref]

            if data_source == "img":
                signal = np.array(entry['img']).transpose(1, 2, 0)  # (H, W, T)
            elif data_source == "dFof":
                signal = np.array(entry['dFof']).transpose(1, 2, 0)  # (H, W, T)
            elif data_source == "foreground":
                signal = np.array(entry['foreground'])  # (T,)
                signal = signal[np.newaxis, np.newaxis, :]  # (1, 1, T) to broadcast like (H, W, T)
            else:
                raise ValueError(f"Unknown data_source: {data_source}")

            if use_mask:
                roi_mask = np.array(entry['foregroundoverlap']).astype(bool)
                signal = np.where(roi_mask[..., None], signal, 0.0)

            # Segmenting
            for j in range(signal.shape[-1] - segment_length):
                segment = signal[..., j:j+segment_length]  # (H, W, 5)
                segment = torch.from_numpy(segment).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 5, 1)
                all_segments.append(segment)

    x_tensor = torch.cat(all_segments, dim=0)  # (N, H, W, 5, 1)
    return x_tensor

def load_Z_from_mat(file_path, use_mask=True):
    """
    Load Z(x, t) from a .mat file containing 'img'.
    Optionally applies ROI mask and collapses Y.
    Returns a z-scored 2D array (space, time).
    """
    with h5py.File(file_path, 'r') as Data:
        control_key = 'control1' if 'control1' in Data else 'control'
        ref = Data[control_key][0, 0]
        entry = Data[ref]

        img = np.array(entry['img']).transpose(1, 2, 0)  # (H, W, T)

        if use_mask:
            roi_mask = np.array(entry['foregroundoverlap']).astype(bool)
            img = np.where(roi_mask[..., None], img, np.nan)
            Z = np.nanmean(img, axis=0)  # (W, T)
        else:
            Z = np.mean(img, axis=0)     # (W, T)

        Z = (Z - np.nanmean(Z)) / np.nanstd(Z)
        return Z
    
def collapse_segments_to_Z(
    x_tensor, spatial_dim='width', time_average=False, method="mean"
):
    """
    Converts a 5D tensor of segments (N, H, W, T, 1) to a 2D matrix Z(x, t).
    method: "mean", "tucker", "cp", "pca"
    spatial_dim: 'width' or 'height' determines which axis becomes x
    """
    if method == "mean":
        # Original logic
        if spatial_dim == 'width':
            collapsed = x_tensor.mean(dim=1)  # avg over height → (N, W, 5, 1)
            X = collapsed.shape[1]
        elif spatial_dim == 'height':
            collapsed = x_tensor.mean(dim=2)  # avg over width → (N, H, 5, 1)
            X = collapsed.shape[1]
        else:
            raise ValueError("spatial_dim must be 'width' or 'height'")

        if time_average:
            Z = collapsed.mean(dim=2).squeeze(-1)  # → (N, X)
        else:
            Z = collapsed.permute(1, 2, 0, 3).squeeze(-1)  # → (X, T, 1)

        return Z.T  # shape (X, T)

    elif method == "tucker":
        print("Applying Tucker decomposition...")
        return torch.tensor(decompose_tensor_to_Z_tucker(x_tensor))

    elif method == "cp":
        print("Applying CP decomposition...")
        return torch.tensor(decompose_tensor_to_Z_cp(x_tensor))

    elif method == "pca":
        print("Applying PCA-based collapse...")
        return torch.tensor(collapse_with_pca(x_tensor))

    else:
        raise ValueError(f"Unknown collapse method: {method}")


def decompose_tensor_to_Z_tucker(x_tensor, rank=(10, 10, 10, 5)):
    x_np = x_tensor.squeeze(-1).numpy()  # (N, H, W, L)
    tucker_tensor = tucker(x_np, rank)   # Positional for older TensorLy versions
    core = tucker_tensor.core
    factors = tucker_tensor.factors
    reconstructed = tl.tucker_to_tensor(tucker_tensor)
    Z = reconstructed.mean(axis=(1, 2))  # Collapse across H and W
    return torch.tensor(Z).T  # (X, T)


def decompose_tensor_to_Z_cp(x_tensor, rank=10):
    x_np = x_tensor.squeeze(-1).numpy()  # (N, H, W, L)
    cp_tensor = parafac(x_np, rank=rank)
    reconstructed = tl.cp_to_tensor(cp_tensor)
    Z = reconstructed.mean(axis=(1, 2))  # Collapse across H and W
    return torch.tensor(Z).T  # (X, T)

def collapse_with_pca(x_tensor, n_components=1):
    N, H, W, L, _ = x_tensor.shape
    flattened = x_tensor.view(N, -1, L).squeeze(-1).numpy()  # (N, H*W, L)
    pca_projections = []

    for i in range(N):
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(flattened[i])
        pca_signal = transformed[:, 0]  # First principal component
        pca_projections.append(pca_signal)

    Z = np.array(pca_projections)
    return torch.tensor(Z.T)  # (X, T)