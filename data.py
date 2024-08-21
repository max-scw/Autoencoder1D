import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class SensorDataset(Dataset):
    def __init__(self, data):
        # Add channel dimension (batch_size, 1, num_features)
        self.data = data.unsqueeze(1) if len(data.shape) < 3 else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


if __name__ == '__main__':
    # Assuming sensor_data is a NumPy array of shape (num_samples, num_features)
    sensor_data = np.random.randn(1000, 3, 128)  # Example data

    # Convert to PyTorch tensor
    sensor_data = torch.tensor(sensor_data, dtype=torch.float32)

    # Create Dataset and DataLoader
    dataset = SensorDataset(sensor_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in dataloader:
        print(data.shape)
        break
