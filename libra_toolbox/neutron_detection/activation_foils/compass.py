import numpy as np
import os
import pandas as pd
import datetime


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
    return int(filename.split('@')[0][7:])


def sort_compass_files(directory):
    """ Gets Compass csv data filenames
    and sorts them according to channel and ending number.
    The filenames need to be sorted by ending number because only
    the first csv file for each channel contains a header.
    
    Example of sorted filenames in directory: 
        1st file: Data_CH4@...22.CSV
        2nd file: Data_CH4@...22_1.CSV
        3rd file: Data_CH4@...22_2.CSV """

    filenames = os.listdir(directory)
    data_filenames = {}
    for filename in filenames:
        if filename.lower().endswith('.csv'):
            ch = get_channel(filename)
            # initialize filenames for each channel
            if ch not in data_filenames.keys():
                data_filenames[ch] = []

            data_filenames[ch] += [filename]
    # Sort filenames by number at end
    for ch in data_filenames.keys():
        data_filenames[ch] = np.sort(data_filenames[ch])

    return data_filenames



def get_events(directory):
    """ From a directory with unprocessed Compass data CSV files,
    this returns dictionaries of detector pulse times and energies
    with digitizer channels as the keys to the dictionaries. 
    
    This function is also built to be able to read-in problematic 
    Compass CSV files that have been incorrectly post-processed to 
    reduce waveform data. """

    time_values = {}
    energy_values = {}

    data_filenames = sort_compass_files(directory)

    for ch in data_filenames.keys():
        # Initialize time_values and energy_values for each channel
        time_values[ch] = []
        energy_values[ch] = []
        for i,filename in enumerate(data_filenames[ch]):
            # First file has a header, so skip the first row
            # Eventually, we can use the header to index the values
            # but since some csv datafiles have been changed without header info,
            # this is the code that will work
            if i==0:
                skiprows=1
            else:
                skiprows=0

            csv_file_path = os.path.join(directory, filename)

            try:
                df = pd.read_csv(csv_file_path, delimiter=';', header=None, skiprows=skiprows)
            except:
                raise Exception(f'Could not read in file: {csv_file_path}')
            
            time_column = 2
            energy_column = 3
            time_data = df[time_column].to_numpy()
            # print(time_data.shape)
            energy_data = df[energy_column].to_numpy()
            erg_max = np.nanmax(energy_data)

            # Check if time and energy are in different columns:
            # This usually is the case due to the csv file being changed previously
            # due to an attempt to reduce the size of waveform data. In this attempt, 
            # the time and energy column were accidently shifted one over.
            if erg_max > 1e5:
                print('time data may be in energy_data column for: \n', filename)
                # Need to skip first row for all problematic files, not just the first file
                df = pd.read_csv(csv_file_path, delimiter=';', header=None, skiprows=1)
                time_column = 3
                energy_column = 4
                time_data = df[time_column].to_numpy()
                energy_data = df[energy_column].to_numpy()
                erg_max = np.nanmax(energy_data)
            
            # Extract and append the energy data to the list
            time_values[ch].extend(time_data)
            # print(len(time_values[source]))
            energy_values[ch].extend(energy_data)
    return time_values, energy_values


def get_start_stop_time(run_info_filepath):
    """ Gets the start and stop time as a datetime.datetime object
    for a detector count from the run.info Compass file"""

    if os.path.isfile(run_info_filepath):
        time_format = "%Y/%m/%d %H:%M:%S.%f%z"
        with open(run_info_filepath, 'r') as file:
            continue_search = True
            while continue_search:
                line = file.readline()
                if 'time.start=' in line:
                    # get time string while cutting off '\n' newline
                    time_string = line.split('=')[1][:-1]
                    start_time = datetime.datetime.strptime(time_string, time_format)
                elif 'time.stop=' in line:
                    time_string = line.split('=')[1][:-1]
                    stop_time = datetime.datetime.strptime(time_string, time_format)
                    continue_search = False
                elif len(line) == 0:
                    # if blank line occurs, stop search
                    continue_search = False
    else:
        raise LookupError('Could not find run.info file')
    
    return start_time, stop_time

