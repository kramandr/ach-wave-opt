import os
import json
from datetime import datetime
import uuid

def create_experiment_folder_old(config, base_dir="results"):
    """
    Create a timestamped folder in the results/ directory based on config.
    Returns the full path to the experiment folder.
    """
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path)

    return folder_path



def create_experiment_folder_synthetic(config, base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    uid = str(uuid.uuid4())[:6]  # Use first 6 chars of UUID
    tag = f"{config['data_variant']}__{config['optimizer']}"
    folder_name = f"{tag}__exp_{uid}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path)
    return folder_path

def create_experiment_folder_real(config, base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    uid = str(uuid.uuid4())[:6]

    # Compose descriptive tag
    tag = (
        f"{config['file_mode']}"
        f"__{config['optimizer']}"
        f"__{config['collapse_method']}"
        f"__{config['spatial_dim']}"
        f"__seg{config['segment_length']}"
        f"__mask{int(config['use_mask'])}"
    )

    if config.get("file_mode") == "chat_control":
        tag += f"__{config['file_subset']}"

    folder_name = f"{tag}__exp_{uid}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path)
    return folder_path


def save_config_summary(config, folder_path):
    """
    Save the experiment configuration as a JSON file.
    """
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def get_readable_config_name(config):
    """
    Optional: Returns a string like "img_nomask_gd_eta0.1_lam0.1"
    """
    mask_str = "mask" if config.get("use_mask") else "nomask"
    return f"{config['data_source']}_{mask_str}_{config['optimizer']}_eta{config['eta']}_lam{config['lambda']}"
