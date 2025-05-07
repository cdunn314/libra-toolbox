import numpy as np
import os
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, List
import datetime
import uproot
import glob


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


def get_live_time_from_root(root_filename, channel: int):
    """Gets live and real count time from Compass root file.
    Live time is defined as the difference between the actual time that
    a count is occurring and the "dead time," in which the output of detector
    pulses is saturated such that additional signals cannot be processed."""

    with uproot.open(root_filename) as root_file:
        live_count_time = root_file[f"LiveTime_{channel}"].members["fMilliSec"] / 1000
        real_count_time = root_file[f"RealTime_{channel}"].members["fMilliSec"] / 1000
    return live_count_time, real_count_time


class Detector:
    events: np.ndarray
    channel_nb: int
    live_count_time: float
    real_count_time: float

class Measurement:
    start_time: datetime.datetime
    stop_time: datetime.datetime
    name: str
    detectors: List[Detector]

    def __init__(self, name: str, ch: int) -> None:
        self.name = name
        self.channel_nb = ch
        self.detectors = []

    @classmethod
    def from_directory(cls, source_dir: str, name: str):
        #print('Reading in files for {}'.format(source))

        # Get events
        time_values, energy_values = get_events(source_dir)

        # Get start and stop time
        start_time, stop_time = get_start_stop_time(source_dir)

        # Get live and real count times
        root_filename = glob.glob(os.path.join(source_dir, '*.root'))[0]
        if os.path.isfile(root_filename):
            for channel in time_values.keys():
                live_count_time, real_count_time = get_live_time_from_root(root_filename, channel)
        else:
            real_count_time = (stop_time - start_time).total_seconds()
            for channel in time_values.keys():
                # Assume first and last event correspond to start and stop time of live counts
                # and convert from picoseconds to seconds
                live_count_time = (time_values[channel][-1] - time_values[channel][0]) / 1e12

        measurement_object = cls(name=name, )

        return cls()


my_check_source_measurement = Measurement.from_directory(path)

def get_all_spectra_from_raw(directories):

    data = {}

    # Iterate through all CSV files in the directory
    for source in directories.keys():
        print('Reading in files for {}'.format(source))
        data[source] = {}

        # Get events
        time_values, energy_values = get_events(directories[source])

        # Get start and stop time
        start_time, stop_time = get_start_stop_time(directories[source])

        # Get live and real count times
        root_filename = glob.glob(os.path.join(directories[source], '*.root'))[0]
        if os.path.isfile(root_filename):
            for channel in time_values.keys():
                live_count_time, real_count_time = get_live_time_from_root(root_filename, channel)
        else:
            real_count_time = (stop_time - start_time).total_seconds()
            for channel in time_values.keys():
                # Assume first and last event correspond to start and stop time of live counts
                # and convert from picoseconds to seconds
                live_count_time = (time_values[channel][-1] - time_values[channel][0]) / 1e12
            

        # Create a histogram to represent the combined energy spectrum
        for ch in time_values[source].keys():
            # sort data based on timestamp
            inds = np.argsort(time_values[source][ch])
            # convert times to seconds
            time_values[source][ch] = np.array(time_values[source][ch])[inds] /1e12
            energy_values[source][ch] = np.array(energy_values[source][ch])[inds]
            # print(np.nanmax(energy_values[source]))

            energy_values[source][ch] = np.nan_to_num(energy_values[source][ch], nan=0)

            if isinstance(bins, int):
                hist, bin_edges = np.histogram(energy_values[source][ch], bins=bins)
            elif bins=='double':
                hist, bin_edges = np.histogram(energy_values[source][ch], bins=int(np.nanmax(energy_values[source][ch])/2))
            else:
                # b = np.arange(0, max_channel[ch])
                b = np.arange(0, np.max(energy_values[source][ch]))
                hist, bin_edges = np.histogram(energy_values[source][ch], bins=b)

            total_time = np.max(time_values[source][ch]) - np.min(time_values[source][ch])
            # counts[source][ch]['count_time'] = total_time
            if np.abs(total_time - counts[source][ch]['real_count_time']) > 0.1:
                print(f'Total time = {total_time}\n', 
                        'Real Count Time (root) = {}'.format(counts[source][ch]['real_count_time']))
                raise Exception(f'Total time different from root file real count time')
            if count_rate:
                counts[source][ch]['hist'] = hist / total_time
            else:
                counts[source][ch]['hist'] = hist
            counts[source][ch]['bin_edges'] = bin_edges
        # Get count start time
        start_time, stop_time = get_start_stop_time(directories[source])
        for ch in counts[source].keys():
            counts[source][ch]['start_time'] = start_time
            counts[source][ch]['stop_time'] = stop_time
    
    
    # save data for faster opening in future
    if savefile is not None:
        with open(savefile, 'wb') as file:
            pickle.dump(counts, file)   

    return counts

def get_all_spectra(directories:dict, savefile=None):

    """ Obtain detector counts from .CSV file saved in CoMPASS."""

    get_raw_data = False
    
    if savefile is not None:
        if os.path.isfile(savefile):
            with open(savefile, 'rb') as file:
                data = pickle.load(file)
        else:
            get_raw_data = True

    if get_raw_data:
        counts = get_all_spectra_from_raw(directories)

    return counts

