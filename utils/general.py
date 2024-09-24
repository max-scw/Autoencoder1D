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
        "n_channels": model.n_channels,
        "len_sig": model.len_sig,
        "n_depth": model.n_depth,
        "sig_mean": data.mean if hasattr(data, "mean") else None,
        "sig_std": data.std if hasattr(data, "std") else None,
        "normalize_data": hasattr(data, "mean"),
    }
    filename = Path(filename).with_suffix(".pth")
    torch.save(checkpoint, filename)
    return filename


def load_autoencoder(filename: Union[str, Path]) -> Tuple[Conv1DAutoencoder, dict]:

    checkpoint = torch.load(filename)

    # Initialize the model, define the loss function and the optimizer
    model = Conv1DAutoencoder(
        n_channels=checkpoint["n_channels"],
        len_sig=checkpoint["len_sig"],
        n_depth=checkpoint["n_depth"],
    )
    model.load_state_dict(checkpoint["state_dict"])

    kwargs = {
        "signal_len": checkpoint["len_sig"],
        "normalize_data": checkpoint["normalize_sig"],
    }
    if checkpoint["sig_mean"]:
        kwargs["mean"] = checkpoint["sig_mean"]
    if checkpoint["sig_std"]:
        kwargs["std"] = checkpoint["sig_std"]

    return model, kwargs
