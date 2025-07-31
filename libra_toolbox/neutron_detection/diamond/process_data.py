import numpy as np


class DataProcessor:
    """
    A class for reading and processing data from diamond detectors as text files

    Attributes:
        files (list of str): List of text filenames that have been read
        time_values (np.array): Array of time values from all files
        energy_values (np.array): Array of energy values from all files
    """

    def __init__(self) -> None:
        self.files = []

        self.time_values = np.array([])
        self.energy_values = np.array([])

    def add_file(
        self,
        filename: str,
        time_column: int,
        energy_column: int,
        scale_time: bool = True,
        **kwargs,
    ):
        """
        Adds a file to the data processor, reading the time and energy values from the file
        and appending them to the existing data (``time_values`` and ``energy_values`` attributes).

        Args:
            filename (str): the name of the file to read
            time_column (int): the column index of the time values in the file
            energy_column (int): the column index of the energy values in the file
            scale_time (bool, optional): if True, the time values are scaled from ps to s. Defaults to True.
        """
        self.files.append(filename)

        # Should we store the data for each file separately too?
        data = np.genfromtxt(filename, **kwargs)

        time_values = data[:, time_column]
        if scale_time:
            # convert times from ps to s
            time_values *= 1 / 1e12
        energy_values = data[:, energy_column]

        # Append time and energy values to the list
        self.time_values = np.concatenate((self.time_values, time_values))
        self.energy_values = np.concatenate((self.energy_values, energy_values))

        # sort time and energy values
        inds = np.argsort(self.time_values)
        self.time_values = np.array(self.time_values)[inds]
        self.energy_values = np.array(self.energy_values)[inds]

        print(f"Added file: {filename} containing {len(time_values)} events")

    def get_count_rate(self, bin_time: float, energy_window: tuple = None):
        """
        Calculate the count rate in a given time bin for the
        time values stored in the data processor.

        Args:
            bin_time (float): the time bin width in seconds
            energy_window (tuple, optional): If provided, the rate
                will be computed only on this energy window. Defaults to None.

        Returns:
            np.array: Array of count rates (counts per second)
            np.array: Array of time bin edges (in seconds)
        """
        time_values = self.time_values.copy()
        energy_values = self.energy_values.copy()

        time_bins = np.arange(time_values.min(), time_values[-2], bin_time)

        if energy_window is not None:
            peak_mask = (energy_values > energy_window[0]) & (
                energy_values < energy_window[1]
            )
            time_values = time_values[peak_mask]

        count_rates, count_rate_bins = np.histogram(time_values, bins=time_bins)
        count_rates = count_rates / bin_time

        return count_rates, count_rate_bins

    def get_avg_rate(self, t_min: float, t_max: float, energy_window: tuple = None):
        """
        Calculate the average count rate in a given time window for the
        time values stored in the data processor.
        Similar to ``get_count_rate`` but returns a single value for a time window.

        Args:
            t_min (float): start time of the time window
            t_max (float): end time of the time window
            energy_window (tuple, optional): If provided, the rate
                will be computed only on this energy window. Defaults to None.

        Returns:
            float: Average count rate (counts per second)
            float: Error on the average count rate (counts per second)
        """
        time_values = self.time_values.copy()
        energy_values = self.energy_values.copy()

        if energy_window is not None:
            peak_mask = (energy_values > energy_window[0]) & (
                energy_values < energy_window[1]
            )
            time_values = time_values[peak_mask]

        # Create mask to only count pulses of any energy in section time window
        idx = np.logical_and(time_values > t_min, time_values < t_max)

        counts = len(time_values[idx])
        error = np.sqrt(len(time_values[idx]))
        delta_t = t_max - t_min
        count_rate = counts / delta_t
        count_rate_err = error / delta_t

        return count_rate, count_rate_err


if __name__ == "__main__":
    pass
