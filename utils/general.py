import torch
from pathlib import Path

from typing import Union, Tuple

from models import Conv1DAutoencoder


def save_autoencoder(model: Conv1DAutoencoder, filename: Union[str, Path], data = None) -> Path:
    """
    Saves the model weights and meta information to a checkpoint file.
    :param model:
    :param filename:
    :param data:
    :return:
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_args": {
            ky: getattr(model, ky) for ky in ["n_channels", "n_depth", "stride", "len_sig", "n_channels_out_0"]
        },
        "data_args": {
            "mean": data.mean if hasattr(data, "mean") else None,
            "std": data.std if hasattr(data, "std") else None,
            "normalize_data": hasattr(data, "mean"),
            "signal_len": getattr(model, "len_sig"),
        }
    }

    filename = Path(filename).with_suffix(".pth")
    torch.save(checkpoint, filename)
    return filename


def load_autoencoder(filename: Union[str, Path]) -> Tuple[Conv1DAutoencoder, dict]:

    checkpoint = torch.load(filename)

    # Initialize the model, define the loss function and the optimizer
    model = Conv1DAutoencoder(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])

    return model, {ky: vl for ky, vl in checkpoint["data_args"].items() if vl is not None}
