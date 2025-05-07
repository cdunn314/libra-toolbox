import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, List, Union
import datetime
import uproot
import glob

import warnings


def get_channel(filename):
    """
    Extract the channel number from a given filename string.

    Parameters
    ----------
    filename : str
        The input filename string containing the channel information.
        Should look something like : "Data_CH<channel_number>@V...CSV"

    Returns
    -------
    int
        The extracted channel number.

    Example
    -------
    >>> get_channel("Data_CH4@V1725_292_Background_250322.CSV")
    4
    """
    return int(filename.split("@")[0][7:])


def sort_compass_files(directory: str) -> dict:
    """Gets Compass csv data filenames
    and sorts them according to channel and ending number.
    The filenames need to be sorted by ending number because only
    the first csv file for each channel contains a header.

    Example of sorted filenames in directory:
        1st file: Data_CH4@...22.CSV
        2nd file: Data_CH4@...22_1.CSV
        3rd file: Data_CH4@...22_2.CSV"""

    filenames = os.listdir(directory)
    data_filenames = {}
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            ch = get_channel(filename)
            # initialize filenames for each channel
            if ch not in data_filenames.keys():
                data_filenames[ch] = []

            data_filenames[ch].append(filename)
    # Sort filenames by number at end
    for ch in data_filenames.keys():
        data_filenames[ch] = np.sort(data_filenames[ch])

    return data_filenames


def get_events(directory: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    From a directory with unprocessed Compass data CSV files,
    this returns dictionaries of detector pulse times and energies
    with digitizer channels as the keys to the dictionaries.

    This function is also built to be able to read-in problematic
    Compass CSV files that have been incorrectly post-processed to
    reduce waveform data.

    Args:
        directory: directory containing CSV files with Compass data

    Returns:
        time values and energy values for each channel
    """

    time_values = {}
    energy_values = {}

    data_filenames = sort_compass_files(directory)

    for ch in data_filenames.keys():
        # Initialize time_values and energy_values for each channel
        time_values[ch] = np.empty(0)
        energy_values[ch] = np.empty(0)
        for i, filename in enumerate(data_filenames[ch]):

            # only the first file has a header
            if i == 0:
                header = 0
            else:
                header = None

            csv_file_path = os.path.join(directory, filename)

            df = pd.read_csv(csv_file_path, delimiter=";", header=header)

            # read the header and store in names
            if i == 0:
                names = df.columns.values
            else:
                # apply the column names if not the first file
                df.columns = names

            time_data = df["TIMETAG"].to_numpy()
            energy_data = df["ENERGY"].to_numpy()

            # Extract and append the energy data to the list
            time_values[ch] = np.concatenate([time_values[ch], time_data])
            energy_values[ch] = np.concatenate([energy_values[ch], energy_data])

    return time_values, energy_values


def get_start_stop_time(directory: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """Obtains count start and stop time from the run.info file."""

    info_file = Path(directory).parent / "run.info"
    if info_file.exists():
        time_format = "%Y/%m/%d %H:%M:%S.%f%z"
        with open(info_file, "r") as file:
            lines = file.readlines()
    else:
        raise FileNotFoundError(
            f"Could not find run.info file in parent directory {Path(directory).parent}"
        )

    start_time, stop_time = None, None
    for line in lines:
        if "time.start=" in line:
            # get start time string while cutting off '\n' newline
            time_string = line.split("=")[1].replace("\n", "")
            start_time = datetime.datetime.strptime(time_string, time_format)
        elif "time.stop=" in line:
            # get stop time string while cutting off '\n' newline
            time_string = line.split("=")[1].replace("\n", "")
            stop_time = datetime.datetime.strptime(time_string, time_format)

    if None in (start_time, stop_time):
        raise ValueError(f"Could not find time.start or time.stop in file {info_file}.")
    else:
        return start_time, stop_time


def get_live_time_from_root(root_filename: str, channel: int) -> Tuple[float, float]:
    """
    Gets live and real count time from Compass root file.
    Live time is defined as the difference between the actual time that
    a count is occurring and the "dead time," in which the output of detector
    pulses is saturated such that additional signals cannot be processed."""

    with uproot.open(root_filename) as root_file:
        live_count_time = root_file[f"LiveTime_{channel}"].members["fMilliSec"] / 1000
        real_count_time = root_file[f"RealTime_{channel}"].members["fMilliSec"] / 1000
    return live_count_time, real_count_time


class Detector:
    events: NDArray[Tuple[float, float]]  # type: ignore # Array of (time in ps, energy) pairs
    channel_nb: int
    live_count_time: float
    real_count_time: float

    def __init__(self, channel_nb) -> None:
        """
        Initialize a Detector object.
        Args:
            channel_nb: channel number of the detector
        """
        self.channel_nb = channel_nb
        self.events = np.empty((0, 2))  # Initialize as empty 2D array with 2 columns
        self.live_count_time = None
        self.real_count_time = None

    def get_energy_hist(
        self, bins: Union[int, str, NDArray[np.float64]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the energy histogram of the detector events.
        Args:
            bins: number of bins or "double" to use half the max energy as bin size
        Returns:
            Tuple of histogram values and bin edges
        """

        energy_values = self.events[:, 1].copy()
        time_values = self.events[:, 0].copy()

        # sort data based on timestamp
        inds = np.argsort(time_values)
        time_values = time_values[inds]
        energy_values = energy_values[inds]

        energy_values = np.nan_to_num(energy_values, nan=0)

        if isinstance(bins, (np.ndarray, int)):
            real_bins = bins
        elif bins == "double":
            real_bins = int(np.nanmax(energy_values) / 2)

        return np.histogram(energy_values, bins=real_bins)


class Measurement:
    start_time: datetime.datetime
    stop_time: datetime.datetime
    name: str
    detectors: List[Detector]

    def __init__(self, name: str) -> None:
        """
        Initialize a Measurement object.
        Args:
            name: name of the measurement
        """
        self.start_time = None
        self.stop_time = None
        self.name = name
        self.detectors = []

    @classmethod
    def from_directory(
        cls, source_dir: str, name: str, info_file_optional: bool = False
    ) -> "Measurement":
        """
        Create a Measurement object from a directory containing Compass data.
        Args:
            source_dir: directory containing Compass data
            name: name of the measurement
            info_file_optional: if True, the function will not raise an error
                if the run.info file is not found
        Returns:
            Measurement object
        """
        measurement_object = cls(name=name)

        # Get events
        time_values, energy_values = get_events(source_dir)

        # Get start and stop time
        try:
            start_time, stop_time = get_start_stop_time(source_dir)
            measurement_object.start_time = start_time
            measurement_object.stop_time = stop_time
        except FileNotFoundError:
            if info_file_optional:
                warnings.warn(
                    "run.info file not found. Assuming start and stop time are not needed."
                )

        # Create detectors
        detectors = [Detector(channel_nb=nb) for nb in time_values.keys()]

        # Get live and real count times
        all_root_filenames = glob.glob(os.path.join(source_dir, "*.root"))
        if len(all_root_filenames) == 1:
            root_filename = all_root_filenames[0]
        else:
            root_filename = None
            print("No root file found, assuming all counts are live")

        for detector in detectors:
            detector.events = np.column_stack(
                (time_values[detector.channel_nb], energy_values[detector.channel_nb])
            )

            if root_filename:
                live_count_time, real_count_time = get_live_time_from_root(
                    root_filename, detector.channel_nb
                )
                detector.live_count_time = live_count_time
                detector.real_count_time = real_count_time
            else:
                real_count_time = (stop_time - start_time).total_seconds()
                # Assume first and last event correspond to start and stop time of live counts
                # and convert from picoseconds to seconds
                ps_to_seconds = 1e-12
                live_count_time = (
                    time_values[detector.channel_nb][-1]
                    - time_values[detector.channel_nb][0]
                ) * ps_to_seconds
                detector.live_count_time = live_count_time
                detector.real_count_time = real_count_time

        measurement_object.detectors = detectors

        return measurement_object
