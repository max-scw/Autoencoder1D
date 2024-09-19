import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from pathlib import Path
from tqdm import tqdm
import pandas as pd

from typing import Union

from utils import OnlineStats


class SensorDataset(Dataset):
    def __init__(self, data) -> None:
        # Add channel dimension (batch_size, 1, num_features)
        self.data = data.unsqueeze(1) if len(data.shape) < 3 else data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetCSV(Dataset):
    variables_to_ignore = []

    def __init__(
            self,
            info_file: Union[str, Path],
            signal_len: int,
            normalize: bool = False,
            **kwargs
    ) -> None:
        super().__init__()
        self.__kwargs_pandas_read_csv = kwargs
        self.__signal_len = signal_len

        if isinstance(info_file, str):
            info_file = Path(info_file)
        elif not isinstance(info_file, Path):
            raise ValueError("Input 'csv_info_file' must be a string or a Path object")

        if not info_file.exists() or not info_file.is_file():
            raise ValueError("Input 'csv_info_file' must point to an existing file.")

        # read info file
        with open(info_file, "r") as fid:
            lines = fid.readlines()
        files = [Path(el.strip()) for el in lines if len(el) > 5]

        # check files
        self.files = []
        n_corrupted = 0
        for fl in (pbar := tqdm(files, desc="Checking files")):
            file = info_file.parent / fl
            if not file.is_file():
                n_corrupted += 1
                pbar.set_postfix({"# corrupted files": n_corrupted})
            else:
                self.files.append(file)

        if len(self.files) == 0:
            raise Exception("No files found!")

        # normalize (numeric) data
        if normalize:
            stats = dict()
            for fl in tqdm(self.files, desc="Calculating statistics from files"):
                # read file
                df = pd.read_csv(fl, **self.__kwargs_pandas_read_csv)
                # calculate statistics
                for ky, vl in df.select_dtypes("number").items():
                    if ky not in stats:
                        stats[ky] = OnlineStats()
                    stats[ky].add_signal(vl)
            self.mean = pd.Series({ky: vl.mean for ky, vl in stats.items()})
            self.std = pd.Series({ky: vl.std for ky, vl in stats.items()})
        else:
            self.mean, self.std = None, None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> np.ndarray:
        # get file
        file = self.files[idx]
        # read file
        df = pd.read_csv(file, **self.__kwargs_pandas_read_csv).select_dtypes("number")

        # normalize (z-normalization)
        if (self.mean is not None) and (self.std is not None):
            # identify variables that exhibit no variance
            lg = self.std != 0
            # variables to ignore
            self.variables_to_ignore = np.array(df.columns)[np.invert(lg)].tolist()
            # crop and scale
            df = (df.loc[:, lg] - self.mean[lg]) / self.std[lg]

        # zero-padding if necessary
        n_padding = self.__signal_len - df.shape[0]
        if n_padding > 0:
            # create a DataFrame with zeros for padding
            zero_padding = pd.DataFrame(np.zeros((n_padding, df.shape[1])), columns=df.columns)
            # concatenate the original DataFrame with the zero-padding
            df_ = pd.concat([df, zero_padding], ignore_index=True)
        else:
            df_ = df[:self.__signal_len]

        return df_.to_numpy().transpose().astype(np.float32)  # expecting shape to be (n_channles, len_signal)



if __name__ == '__main__':
    # # Assuming sensor_data is a NumPy array of shape (num_samples, num_features)
    # sensor_data = np.random.randn(1000, 3, 128)  # Example data
    #
    # # Convert to PyTorch tensor
    # sensor_data = torch.tensor(sensor_data, dtype=torch.float32)

    # Create Dataset and DataLoader
    # dataset = SensorDataset(sensor_data)
    dataset = DatasetCSV(info_file=r"../CaptureDataParser/data/Testfiles.txt", signal_len=2**16)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    for element in dataloader:
        print(element.shape)
        break


