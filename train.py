from argparse import ArgumentParser

import torch
import torch.nn as nn
# dataset
from torch.utils.data import DataLoader
# optimizer
from torch.optim import Adam

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer
import copy
import logging
from datetime import datetime
from pathlib import Path

from typing import Tuple, List, Dict, Any, Union

from models import Conv1DAutoencoder
from utils import create_dataset, save_autoencoder
from utils import cast_logging_level, cast


def get_device(device_str: str) -> torch.device:
    cuda = torch.cuda.is_available() and device_str.lower() != "cpu"
    return torch.device(f"cuda:{int(device_str)}" if cuda else "cpu")

def train(
    model: nn.Module,
    dataloader_training: DataLoader,
    n_epochs: int,
    dataloader_validation: DataLoader = None,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    optimizer=None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """
    Trains a PyTorch model
    :param model: PyTorch model to train
    :param dataloader_training: dataloader holding the training set
    :param n_epochs: Number of training epochs
    :param dataloader_validation: dataloader holding the validation set (to ensure that the best model is returned later)
    :param criterion: evaluation criterion (for classification usually cross-entropy loss
    :param optimizer: optimizer to calculate the training step, usually Adam
    :param device: usually CPU or GPU
    :return: best trained model
    """
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=0.001)

    t0 = default_timer()  # time the execution of the function

    # initialize best_loss and model_weights to keep track of the best model
    best_loss = torch.inf
    best_model_weights = copy.deepcopy(model.state_dict())
    history = []

    for i_epoch in range(n_epochs):
        desc = {}
        # Each epoch has a training and validation phase
        for phase in ["training", "validation"]:
            is_training = phase == "training"

            if is_training:
                model.train()  # Set model to training mode
                dataloader = dataloader_training
            else:
                model.eval()  # Set model to evaluate mode
                if dataloader_validation is None:
                    continue
                else:
                    dataloader = dataloader_validation

            running_loss = 0.0
            running_len = 0

            # Iterate over data.
            # progress bar object
            pbar = tqdm(dataloader, desc=f"Epoch {i_epoch + 1}/{n_epochs}, {phase}")
            for x in pbar:

                if isinstance(x, tuple) and len(x) == 2:
                    inputs = x[0].to(device)
                    labels = x[1].to(device)
                else:
                    inputs = x.to(device)
                    labels = inputs

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only in train
                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if is_training:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_len += len(labels)

                # update postfix of the progress bar
                pbar.set_postfix({"loss": running_loss / running_len})

            # calculate loss
            epoch_loss = running_loss / running_len
            # store information to update the bar
            desc[f"{phase}_loss"] = epoch_loss

            # deep copy the model
            if ((not is_training) or (dataloader_validation is None)) and (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        history.append(desc)

    time_elapsed = default_timer() - t0
    print(f"Training complete in {time_elapsed // 60:.0g}m {time_elapsed % 60:.0f}s. Best validation loss: {best_loss:4g}")

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, history





if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--data', type=str, help="Path a file that lists all training data")
    parser.add_argument("--signal-len", type=int, default=2 ** 16,
                        help="Maximum length to which a signal is padded or cropped it is longer")
    parser.add_argument("--depth", type=int, default=5,
                        help="Number of convolution layers of the encoder.")

    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Total batch size (for all GPUs)")
    parser.add_argument("--normalize", action="store_true",
                        help="Applies z-standardization on the input data before feeding it to the autoencoder")
    parser.add_argument("--ignore-idxs", type=int, nargs="+", default=None, help="Indices of files to ignore")
    parser.add_argument('--kwargs', nargs='*', help='Additional keyword arguments for the dataset')

    parser.add_argument("--workers", type=int, default=2, help="Maximum number of dataloader workers")
    parser.add_argument("--device", default="cpu", help="Cuda device, i.e. 0 or 0,1,2,3, or cpu")

    parser.add_argument("--process-title", type=str, default=None, help="Names the process")
    parser.add_argument("--logging-level", type=str, default="INFO", help="Logging level")

    opt = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=cast_logging_level(opt.logging_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # create file wide logger
    logger = logging.getLogger(__name__)
    logger.debug(opt)

    if opt.process_title:
        from setproctitle import setproctitle
        setproctitle(opt.process_title)

    device = get_device(opt.device)

    kwargs = dict()
    for arg in opt.kwargs:
        key, value = arg.split("=", 1)

        kwargs[key] = cast(value)

    # # create data
    # sensor_data = np.random.randn(1000, 5, 256)  # Example data
    # # Convert to PyTorch tensor
    # sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
    # # Create Dataset
    # dataset = SensorDataset(sensor_data)

    dataset = create_dataset(
        opt.data,
        opt.signal_len,
        normalize_data=opt.normalize,
        ignore_idxs=opt.ignore_idxs,
        **kwargs
    )
    # get one datapoint to adjust the autoencoder according to the data shape
    data_shape = dataset[0].shape

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    # Initialize the model, define the loss function and the optimizer
    autoencoder = Conv1DAutoencoder(
        n_channels=data_shape[0],
        len_sig=data_shape[1],
        n_depth=opt.depth
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(autoencoder.parameters(), lr=0.001)

    # Training loop
    logger.debug("Start training. ")
    autoencoder, hist = train(
        autoencoder,
        dataloader_training=dataloader,
        n_epochs=opt.epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    df = pd.DataFrame(hist)
    df.plot(xlabel="Epoch", ylabel="MSE Loss", title="Training Loss")

    plt.savefig("Autoencoder_TrainingLoss.png")
    save_autoencoder(
        model=autoencoder,
        filename=f"{datetime.now().strftime('%Y%m%d-%H%M')}_autoencoder_{Path(opt.data).stem}",
        data=dataset
    )
