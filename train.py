import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


from models import Conv1DAutoencoder
from data import SensorDataset


if __name__ == "__main__":

    # create data
    sensor_data = np.random.randn(1000, 5, 256)  # Example data

    # Convert to PyTorch tensor
    sensor_data = torch.tensor(sensor_data, dtype=torch.float32)

    # Create Dataset and DataLoader
    dataset = SensorDataset(sensor_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model, define the loss function and the optimizer
    model = Conv1DAutoencoder(n_channels=sensor_data.shape[1], len_sig=sensor_data.shape[2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 5
    for epoch in (pbar := trange(n_epochs, desc="Epoch")):
        for data in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Compute the loss
            loss = criterion(outputs, data)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        pbar.set_postfix({"Loss": round(loss.item(), 5)})
