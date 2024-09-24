from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from tqdm import tqdm
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from models import Conv1DAutoencoder
from utils import create_dataset, load_autoencoder
from train import get_device


def calculate_rms_energy(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the root-mean-square energy for each batch in a tensor.

    Args:
        tensor (torch.Tensor): A tensor of size [batch_size, n_channels, length]

    Returns:
        torch.Tensor: A tensor of size [batch_size] containing the RMS energy for each batch
    """
    # Calculate the square of the tensor
    squared_tensor = tensor ** 2
    # Calculate the mean of the squared tensor along the last two dimensions (n_channels and length)
    mean_squared_tensor = torch.mean(squared_tensor, dim=(1, 2))

    # Calculate the square root of the mean squared tensor
    rms_energy = torch.sqrt(mean_squared_tensor)

    return rms_energy


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--data', type=str, help="Path a file that lists all training data")
    # parser.add_argument("--signal-len", type=int, default=2 ** 16,
    #                     help="Maximum length to which a signal is padded or cropped it is longer")
    # parser.add_argument("--depth", type=int, default=5,
    #                     help="Number of convolution layers of the encoder.")

    parser.add_argument("--checkpoint", type=str, help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=16, help="Total batch size (for all GPUs)")
    # parser.add_argument("--normalize", action="store_true",
    #                     help="Applies z-standardization on the input data before feeding it to the autoencoder")
    parser.add_argument("--ignore-idxs", type=int, nargs="+", default=None, help="Indices of files to ignore")

    parser.add_argument("--workers", type=int, default=2, help="Maximum number of dataloader workers")
    parser.add_argument("--device", default="cpu", help="Cuda device, i.e. 0 or 0,1,2,3, or cpu")

    parser.add_argument("--process-title", type=str, default=None, help="Names the process")

    opt = parser.parse_args()

    # Setup logging
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    # # create file wide logger
    # logger = logging.getLogger(__name__)

    device = get_device(opt.device)
    path_to_weights = Path(opt.weights)
    path_to_data = Path(opt.data)

    # load model
    model, kwargs = load_autoencoder(opt.checkpoint)
    model = model.to(device)

    dataset = create_dataset(
        opt.data,
        **kwargs,
        ignore_idxs=opt.ignore_idxs
    )
    # get one datapoint to adjust the autoencoder according to the data shape
    data_shape = dataset[0].shape
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers
    )

    model.eval()
    encoded, energy = [], []
    for x in tqdm(dataloader, desc=f"Encoding data"):

        if isinstance(x, tuple) and len(x) == 2:
            inputs = x[0].to(device)
        else:
            inputs = x.to(device)

        # forward
        with torch.no_grad():
            output = model(inputs)
        encoded += output.tolist()
        # calculate the energy (rms-value) of the signals
        energy += calculate_rms_energy(x)

    encoded_data = np.array(encoded)
    energy_data = np.array(energy)


    if isinstance(dataset.data[0], pd.DataFrame) and hasattr(dataset.data[0], "name"):
        # reconstruct names
        names = [el.name for el in dataset.data]
    else:
        names = None

    # create pandas.DataFrame object from encoded dimensions + energy value
    columns = list(range(1, encoded_data.shape[1] + 1)) + ["energy"]
    df = pd.DataFrame(np.hstack((encoded_data, energy_data.reshape(-1, 1))), index=names, columns=columns)

    # save results
    filename = Path(f"{path_to_weights.stem}_{path_to_data.stem}.csv")
    df.to_csv(filename)
