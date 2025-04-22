from typing import Literal

import h5py
import torch


def save_dataset_to_hdf5(features, labels, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("labels", data=labels)


def load_dataset_from_hdf5(filename):
    with h5py.File(filename, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
    return features, labels

def get_device(accelerator: Literal["auto", "gpu", "mps"] = "auto") -> torch.device:
    # Determine the device based on the accelerator
    if accelerator == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    elif accelerator == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif accelerator == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device