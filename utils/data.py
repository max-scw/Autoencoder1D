from torch.utils.data import  Dataset

from pathlib import Path

from typing import Union, List

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
        signal_len: int,
        normalize_data: bool = True,
        ignore_idxs: List[int] = None
) -> Union[DatasetCSV, DatasetParquet]:
    data_file = Path(path_to_data)

    kwargs = {
        "normalize": normalize_data,
        "signal_len": signal_len,
        "ignore_idxs": ignore_idxs
    }
    if data_file.suffix == ".parquet":
        dataset = DatasetParquet(
            file=data_file,
            groupby="name",
            **kwargs
        )
    else:
        # read from files
        dataset = DatasetCSV(
            info_file=data_file,
            **kwargs
        )

    return dataset
