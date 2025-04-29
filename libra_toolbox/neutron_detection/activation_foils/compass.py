import numpy as np
import os
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict
import datetime
import uproot


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
        raise FileNotFoundError(f"Could not find run.info file in {directory}")

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


def get_live_time_from_root(filename, channel: int):
    """Gets live and real count time from Compass root file.
    Live time is defined as the difference between the actual time that
    a count is occurring and the "dead time," in which the output of detector
    pulses is saturated such that additional signals cannot be processed."""

    with uproot.open(filename) as root_file:
        live_count_time = root_file[f"LiveTime_{channel}"].members["fMilliSec"] / 1000
        real_count_time = root_file[f"RealTime_{channel}"].members["fMilliSec"] / 1000
    return live_count_time, real_count_time
