from torch.utils.data import DataLoader, Dataset
import numpy as np

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pyarrow as pa

from .general import OnlineStats, normalize_df, zero_pad_df

from typing import Union, List


def load_from_parquet(
        file: Union[str, Path],
        groupby = None
) -> Union[pa.Table, List[pd.DataFrame]]:
    if isinstance(file, str):
        file = Path(file)

    if file.is_file():
        table = pd.read_parquet(file)
        if groupby is None:
            return table
        else:
            dfs = []
            for ky, vl in pd.DataFrame(table).groupby(groupby):
                # ignore groupby column when constructing a new DataFrame. (The column indicates the filename)
                columns = [el for el in vl.columns if el != groupby]
                df = pd.DataFrame(vl[columns]).reset_index(drop=True)

                df.name = ky
                dfs.append(df)
            return dfs
    else:
        raise FileNotFoundError


class DatasetParquet(Dataset):
    def __init__(
            self,
            file: Union[str, Path],
            groupby=None,
            signal_len: int = None,
            normalize: bool = False,
            ignore_idxs: List[int] = None
    ) -> None:
        super().__init__()

        # bulk load data from file
        data_ = load_from_parquet(file=file, groupby=groupby)
        # filter data
        ignore_idxs = ignore_idxs if isinstance(ignore_idxs, list) else []
        self.data = [df for i, df in enumerate(data_) if i not in ignore_idxs]

        self._signal_len = signal_len

        # normalize (numeric) data
        if normalize or (signal_len is None):
            stats = dict()
            _signal_len = 0
            for df in tqdm(self.data, desc="Calculating statistics from files"):
                # calculate statistics
                for ky, vl in df.select_dtypes("number").items():
                    if ky not in stats:
                        stats[ky] = OnlineStats()
                    # calculate statistics
                    stats[ky].add_signal(vl)
                # length
                _signal_len = max(_signal_len, len(df))

            self.mean = pd.Series({ky: vl.mean for ky, vl in stats.items()})
            self.std = pd.Series({ky: vl.std for ky, vl in stats.items()})
            if signal_len is None:
                self._signal_len = _signal_len
        else:
            self.mean, self.std = None, None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> np.ndarray:
        df = self.data[idx]

        # normalize (z-normalization)
        if (self.mean is not None) and (self.std is not None):
            df = normalize_df(df, self.mean, self.std)

        # zero-padding if necessary
        df = zero_pad_df(df, self._signal_len)
        return df.to_numpy().transpose().astype(np.float32)

if __name__ == '__main__':

    # Create Dataset and DataLoader
    # dataset = SensorDataset(sensor_data)
    dataset = DatasetParquet(
        file=r"../../SmartSchaKu/Daten_SmartSchaKu_DB_Leipzig_V01_05_500_name.parquet",
        groupby="name",
        normalize=True
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    for element in dataloader:
        print(element.shape)
        break