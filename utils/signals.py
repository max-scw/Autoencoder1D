import pandas as pd
import numpy as np


class OnlineStats:
    def __init__(self):
        self.n = 0        # Number of elements
        self.mean = 0.0   # Mean of elements
        self.m2 = 0.0     # Sum of squares of differences from the mean
        self.min = 9e9
        self.max = -9e9
        self.signal_len = 0

    def add_signal(self, signal: pd.Series):
        """Add a new signal (pandas Series) and update the global mean and variance."""
        for x in signal:
            try:
                self.add_data_point(x)
            except Exception as ex:
                raise ex
        # length
        self.signal_len = max(self.signal_len, len(signal))

    def add_data_point(self, x):
        """Add a new data point and update the mean and standard deviation."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        # extrema
        if x > self.max:
            self.max = x
        elif x < self.min:
            self.min = x


    @property
    def variance(self) -> float:
        """Return the current variance."""
        return self.m2 / self.n if self.n > 0 else float('nan')

    @property
    def standard_deviation(self) -> float:
        """Return the current standard deviation."""
        return self.variance ** 0.5

    @property
    def std(self) -> float:
        return self.standard_deviation


def normalize_df(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    # identify variables that exhibit no variance
    lg = std != 0
    # variables to ignore
    variables_to_ignore = np.array(df.columns)[np.invert(lg)].tolist()
    # crop and scale
    return (df.loc[:, lg] - mean[lg]) / std[lg]


def zero_pad_df(df: pd.DataFrame, signal_len: int) -> pd.DataFrame:
    n_padding = signal_len - df.shape[0]
    if n_padding > 0:
        # create a DataFrame with zeros for padding
        zero_padding = pd.DataFrame(np.zeros((n_padding, df.shape[1])), columns=df.columns)
        # concatenate the original DataFrame with the zero-padding
        return pd.concat([df, zero_padding], ignore_index=True)
    else:
        return df[:signal_len]


if __name__ == '__main__':
    # Example usage
    stats = OnlineStats()
    data_points = [1, 2, 3, 4, 5]

    for point in data_points:
        stats.add_data_point(point)
        print(f"Added {point}: Mean = {stats.mean:.2f}, Std Dev = {stats.standard_deviation:.2f}")
