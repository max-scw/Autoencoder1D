from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from tqdm import tqdm
import logging

import numpy as np
from matplotlib import pyplot as plt

from models import Conv1DAutoencoder
from data.data import create_dataset
from train import get_device


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--data', type=str, help="Path a file that lists all training data")

    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=16, help="Total batch size (for all GPUs)")

    parser.add_argument("--workers", type=int, default=2, help="Maximum number of dataloader workers")
    parser.add_argument("--device", default="cpu", help="Cuda device, i.e. 0 or 0,1,2,3, or cpu")

    parser.add_argument("--process-title", type=str, default=None, help="Names the process")

    opt = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # create file wide logger
    logger = logging.getLogger(__name__)

    device = get_device(opt.device)
    path_to_weights = Path(opt.weights)
    path_to_data = Path(opt.data)

    dataset = create_dataset(path_to_data, opt.signal_len)
    # get one datapoint to adjust the autoencoder according to the data shape
    data_shape = dataset[0].shape
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers
    )

    # Initialize the model, define the loss function and the optimizer
    autoencoder = Conv1DAutoencoder(
        n_channels=data_shape[0],
        len_sig=data_shape[1]
    ).to(device)

    # Load the weights from a file
    checkpoint = torch.load(path_to_weights, map_location=device, weights_only=True)
    autoencoder.load_state_dict(checkpoint)

    # Make sure the model is in evaluation mode
    model = autoencoder.encoder

    model.eval()
    encoded = []
    for x in tqdm(dataloader, desc=f"Encoding data"):

        if isinstance(x, tuple) and len(x) == 2:
            inputs = x[0].to(device)
        else:
            inputs = x.to(device)

        # forward
        with torch.no_grad():
            output = model(inputs)
        encoded += output.tolist()

    encoded_data = np.array(encoded)
    # save results
    np.savetxt(f"{path_to_weights.stem}_{path_to_data.stem}.csv", encoded_data, delimiter=",")

    # plot results
    fig, ax = plt.subplots()
    ax.plot(encoded_data[:, 0], encoded_data[:, 1])
    ax.title(f"{path_to_weights.stem}: {path_to_data.name} ({len(dataset)} points)")
    ax.xlabel("Encoded dimension 1")
    ax.ylabel("Encoded dimension 2")
