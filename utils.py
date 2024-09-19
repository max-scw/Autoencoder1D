import pandas as pd


class OnlineStats:
    def __init__(self):
        self.n = 0        # Number of elements
        self.mean = 0.0   # Mean of elements
        self.m2 = 0.0     # Sum of squares of differences from the mean
        self.min = 9e9
        self.max = -9e9

    def add_signal(self, signal: pd.Series):
        """Add a new signal (pandas Series) and update the global mean and variance."""
        for x in signal:
            try:
                self.add_data_point(x)
            except Exception as ex:
                raise ex

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


if __name__ == '__main__':
    # Example usage
    stats = OnlineStats()
    data_points = [1, 2, 3, 4, 5]

    for point in data_points:
        stats.add_data_point(point)
        print(f"Added {point}: Mean = {stats.mean:.2f}, Std Dev = {stats.standard_deviation:.2f}")
