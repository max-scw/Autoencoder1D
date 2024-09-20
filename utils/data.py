from torch.utils.data import  Dataset

from pathlib import Path

from typing import Union

from utils.DatasetCSV import DatasetCSV
from utils.DatasetParquet import DatasetParquet


class DatasetSensor(Dataset):
    def __init__(self, data) -> None:
        # Add channel dimension (batch_size, 1, num_features)
        self.data = data.unsqueeze(1) if len(data.shape) < 3 else data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




def create_dataset(
        path_to_data: Union[str, Path],
        signal_len: int
) -> Union[DatasetCSV, DatasetParquet]:
    data_file = Path(path_to_data)

    if data_file.suffix == ".parquet":
        dataset = DatasetParquet(
            file=data_file,
            groupby="name",
            normalize=True,
            signal_len=signal_len,
        )
    else:
        # read from files
        dataset = DatasetCSV(
            info_file=data_file,
            signal_len=signal_len,
            normalize=True
        )

    return dataset
