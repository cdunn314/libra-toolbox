from typing import Tuple
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

nano_to_seconds = 1e-9


def get_timestamps_and_amplitudes(
    h5py_file: h5py.File, channel: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the amplitude (in mV) and timestamp (in seconds) of the given channel.

    Args:
        h5py_file: the h5py file object
        channel: the name of the channel in the file (eg. "Channel A")

    Returns:
        the timestamps in s, the amplitudes in mV
    """
    trigger_tist = np.array(h5py_file[f"{channel}/Amplitude-Timestamp"])
    timestamps = trigger_tist["Time [ns]"] * nano_to_seconds
    amplitudes = trigger_tist["Amplitude [mV]"]
    return timestamps, amplitudes


def load_data_from_file(filename: str | Path) -> dict:
    """
    Reads a h5 file from ROSY and returns the data for each channel (timestamps in s and
    amplitudes in mV) in a dictionary with numpy arrays.
    If a channel is not active, it will be an empty array.

    Args:
        filename: the filename

    Returns:
        a dictionary with the following format
            {"Channel A": {"timestamps": [...], "amplitudes": [...]}, ...}
    """
    with h5py.File(filename, "r") as ROSY_file:

        channels = list(ROSY_file.keys())
        print(channels)
        # general information
        active_channels = ROSY_file.attrs["Active channels"]
        print(f"Active channels: {active_channels}")

        data = {}

        for i, channel_name in enumerate(channels):
            if channel_name == "Coincidence":
                continue
            print(f"Channel {i}: {channel_name}")

            if active_channels[i]:
                channel_timestamps, channel_amplitudes = get_timestamps_and_amplitudes(
                    ROSY_file, channel_name
                )
            else:
                channel_timestamps = []
                channel_amplitudes = []

            data[channel_name] = {
                "timestamps": channel_timestamps,
                "amplitudes": channel_amplitudes,
            }
        return data


def get_count_rate(
    time_values: np.ndarray, bin_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the count rate in a given time bin for the
    time values.

    Args:
        time_values: the time values in seconds
        bin_time: the time bin in seconds

    Returns:
        count rates (counts per second), time bin edges (in seconds)
    """
    time_bins = np.arange(time_values.min(), time_values[-2], bin_time)

    count_rates, count_rate_bins = np.histogram(time_values, bins=time_bins)
    count_rates = count_rates / bin_time

    return count_rates, count_rate_bins


# TODO refactor/simplify/remove bits below that aren't needed
"""
# Coincidence Spectrum Analysis for Diamond Telescope Detector

This code calculates the coincidence energy spectrum of a Diamond Telescope Detector based on user-defined coincidence settings and time window parameters.

**Provided by:** [CIVIDEC Instrumentation GmbH ](https://cividec.at) <br>
**Contact:** [office@cividec.at](mailto:office@cividec.at) <br>
**Author:** Julian Melbinger

"""


def COINC_2(Ch1_TIME, Ch2_TIME, Ch1_AMPL, Ch2_AMPL, t_window):
    Ch1_TIME = np.asarray(Ch1_TIME)
    Ch2_TIME = np.asarray(Ch2_TIME)
    Ch1_AMPL = np.asarray(Ch1_AMPL)
    Ch2_AMPL = np.asarray(Ch2_AMPL)

    # For each Ch1 time, find window in Ch2 where match is possible
    idx_start = np.searchsorted(Ch2_TIME, Ch1_TIME - t_window, side="left")
    idx_end = np.searchsorted(Ch2_TIME, Ch1_TIME + t_window, side="right")

    # Keep only those with at least one match
    has_match = idx_start < idx_end

    matched_Ch1_idx = np.flatnonzero(has_match)
    matched_Ch2_idx = idx_start[has_match]  # First match only

    return (
        Ch1_TIME[matched_Ch1_idx],
        Ch2_TIME[matched_Ch2_idx],
        Ch1_AMPL[matched_Ch1_idx],
        Ch2_AMPL[matched_Ch2_idx],
    )


def COINC_3(Ch1_TIME, Ch2_TIME, Ch3_TIME, Ch1_AMPL, Ch2_AMPL, Ch3_AMPL, t_window):
    Ch1_TIME = np.asarray(Ch1_TIME)
    Ch2_TIME = np.asarray(Ch2_TIME)
    Ch3_TIME = np.asarray(Ch3_TIME)
    Ch1_AMPL = np.asarray(Ch1_AMPL)
    Ch2_AMPL = np.asarray(Ch2_AMPL)
    Ch3_AMPL = np.asarray(Ch3_AMPL)

    # For each Ch1 time, find window in Ch2 and Ch3
    idx_start_2 = np.searchsorted(Ch2_TIME, Ch1_TIME - t_window, side="left")
    idx_end_2 = np.searchsorted(Ch2_TIME, Ch1_TIME + t_window, side="right")

    idx_start_3 = np.searchsorted(Ch3_TIME, Ch1_TIME - t_window, side="left")
    idx_end_3 = np.searchsorted(Ch3_TIME, Ch1_TIME + t_window, side="right")

    # Valid coincidences: Ch1 has at least one match in both Ch2 and Ch3
    has_match = (idx_start_2 < idx_end_2) & (idx_start_3 < idx_end_3)
    matched_Ch1_idx = np.flatnonzero(has_match)
    matched_Ch2_idx = idx_start_2[has_match]
    matched_Ch3_idx = idx_start_3[has_match]

    return (
        Ch1_TIME[matched_Ch1_idx],
        Ch2_TIME[matched_Ch2_idx],
        Ch3_TIME[matched_Ch3_idx],
        Ch1_AMPL[matched_Ch1_idx],
        Ch2_AMPL[matched_Ch2_idx],
        Ch3_AMPL[matched_Ch3_idx],
    )


def COINC_4(
    Ch1_TIME,
    Ch2_TIME,
    Ch3_TIME,
    Ch4_TIME,
    Ch1_AMPL,
    Ch2_AMPL,
    Ch3_AMPL,
    Ch4_AMPL,
    t_window,
):
    # Convert to NumPy arrays
    Ch1_TIME = np.asarray(Ch1_TIME)
    Ch2_TIME = np.asarray(Ch2_TIME)
    Ch3_TIME = np.asarray(Ch3_TIME)
    Ch4_TIME = np.asarray(Ch4_TIME)
    Ch1_AMPL = np.asarray(Ch1_AMPL)
    Ch2_AMPL = np.asarray(Ch2_AMPL)
    Ch3_AMPL = np.asarray(Ch3_AMPL)
    Ch4_AMPL = np.asarray(Ch4_AMPL)

    # For each Ch1 event, find index range in Ch2/Ch3/Ch4 within time window
    idx_start_2 = np.searchsorted(Ch2_TIME, Ch1_TIME - t_window, side="left")
    idx_end_2 = np.searchsorted(Ch2_TIME, Ch1_TIME + t_window, side="right")

    idx_start_3 = np.searchsorted(Ch3_TIME, Ch1_TIME - t_window, side="left")
    idx_end_3 = np.searchsorted(Ch3_TIME, Ch1_TIME + t_window, side="right")

    idx_start_4 = np.searchsorted(Ch4_TIME, Ch1_TIME - t_window, side="left")
    idx_end_4 = np.searchsorted(Ch4_TIME, Ch1_TIME + t_window, side="right")

    # Valid coincidences must have at least one match in Ch2, Ch3, and Ch4
    has_match = (
        (idx_start_2 < idx_end_2)
        & (idx_start_3 < idx_end_3)
        & (idx_start_4 < idx_end_4)
    )

    matched_Ch1_idx = np.flatnonzero(has_match)
    matched_Ch2_idx = idx_start_2[has_match]
    matched_Ch3_idx = idx_start_3[has_match]
    matched_Ch4_idx = idx_start_4[has_match]

    return (
        Ch1_TIME[matched_Ch1_idx],
        Ch2_TIME[matched_Ch2_idx],
        Ch3_TIME[matched_Ch3_idx],
        Ch4_TIME[matched_Ch4_idx],
        Ch1_AMPL[matched_Ch1_idx],
        Ch2_AMPL[matched_Ch2_idx],
        Ch3_AMPL[matched_Ch3_idx],
        Ch4_AMPL[matched_Ch4_idx],
    )


def COINC_2_ANTI_1(
    Ch1_TIME, Ch2_TIME, Ch3_TIME, Ch1_AMPL, Ch2_AMPL, Ch3_AMPL, t_window
):

    pos_Ch1, pos_Ch2, pos_Ch3 = 0, 0, 0

    length_Ch1 = len(Ch1_AMPL)
    length_Ch2 = len(Ch2_AMPL)
    length_Ch3 = len(Ch3_TIME)

    aaccepted_ampl_1 = []
    aaccepted_time_1 = []

    aaccepted_ampl_2 = []
    aaccepted_time_2 = []

    while pos_Ch1 < length_Ch1 and pos_Ch2 < length_Ch2:
        min_val = min(Ch1_TIME[pos_Ch1], Ch2_TIME[pos_Ch2])
        max_val = max(Ch1_TIME[pos_Ch1], Ch2_TIME[pos_Ch2])

        CH3_IS_ANTI = True
        while Ch3_TIME[pos_Ch3] <= min_val + t_window:
            if Ch3_TIME[pos_Ch3] >= min_val:
                CH3_IS_ANTI = False
                break

            if pos_Ch3 < length_Ch3 - 1:
                pos_Ch3 += 1
            else:
                break

        if max_val - min_val <= t_window and CH3_IS_ANTI:

            aaccepted_ampl_1.append(Ch1_AMPL[pos_Ch1])
            aaccepted_time_1.append(Ch1_TIME[pos_Ch1])

            aaccepted_ampl_2.append(Ch2_AMPL[pos_Ch2])
            aaccepted_time_2.append(Ch2_TIME[pos_Ch2])

            pos_Ch1 += 1
            pos_Ch2 += 1

        else:
            if min_val == Ch1_TIME[pos_Ch1]:
                pos_Ch1 += 1
            if min_val == Ch2_TIME[pos_Ch2]:
                pos_Ch2 += 1

    return aaccepted_time_1, aaccepted_time_2, aaccepted_ampl_1, aaccepted_ampl_2


def COINC_3_ANTI_1(
    Ch1_TIME,
    Ch2_TIME,
    Ch3_TIME,
    Ch4_TIME,
    Ch1_AMPL,
    Ch2_AMPL,
    Ch3_AMPL,
    Ch4_AMPL,
    t_window,
):
    pos_Ch1, pos_Ch2, pos_Ch3, pos_Ch4 = 0, 0, 0, 0

    length_Ch1 = len(Ch1_AMPL)
    length_Ch2 = len(Ch2_AMPL)
    length_Ch3 = len(Ch3_AMPL)
    length_Ch4 = len(Ch4_TIME)

    aaccepted_ampl_1 = []
    aaccepted_time_1 = []

    aaccepted_ampl_2 = []
    aaccepted_time_2 = []

    aaccepted_ampl_3 = []
    aaccepted_time_3 = []

    while pos_Ch1 < length_Ch1 and pos_Ch2 < length_Ch2 and pos_Ch3 < length_Ch3:
        min_val = min(Ch1_TIME[pos_Ch1], Ch2_TIME[pos_Ch2], Ch3_TIME[pos_Ch3])
        max_val = max(Ch1_TIME[pos_Ch1], Ch2_TIME[pos_Ch2], Ch3_TIME[pos_Ch3])

        CH4_IS_ANTI = True
        while Ch4_TIME[pos_Ch4] <= min_val + t_window:
            if Ch4_TIME[pos_Ch4] >= min_val:
                CH4_IS_ANTI = False
                break

            if pos_Ch4 < length_Ch4 - 1:
                pos_Ch4 += 1
            else:
                break

        if max_val - min_val <= t_window and CH4_IS_ANTI:
            aaccepted_ampl_1.append(Ch1_AMPL[pos_Ch1])
            aaccepted_time_1.append(Ch1_TIME[pos_Ch1])

            aaccepted_ampl_2.append(Ch2_AMPL[pos_Ch2])
            aaccepted_time_2.append(Ch2_TIME[pos_Ch2])

            aaccepted_ampl_3.append(Ch3_AMPL[pos_Ch3])
            aaccepted_time_3.append(Ch3_TIME[pos_Ch3])

            pos_Ch1 += 1
            pos_Ch2 += 1
            pos_Ch3 += 1
        else:
            if min_val == Ch1_TIME[pos_Ch1]:
                pos_Ch1 += 1
            if min_val == Ch2_TIME[pos_Ch2]:
                pos_Ch2 += 1
            if min_val == Ch3_TIME[pos_Ch3]:
                pos_Ch3 += 1

    return (
        aaccepted_time_1,
        aaccepted_time_2,
        aaccepted_time_3,
        aaccepted_ampl_1,
        aaccepted_ampl_2,
        aaccepted_ampl_3,
    )


def COINC_2_ANTI_2(
    Ch1_TIME,
    Ch2_TIME,
    Ch3_TIME,
    Ch4_TIME,
    Ch1_AMPL,
    Ch2_AMPL,
    Ch3_AMPL,
    Ch4_AMPL,
    t_window,
):
    # Ch3 and 4 is anti

    pos_Ch1, pos_Ch2, pos_Ch3, pos_Ch4 = 0, 0, 0, 0

    length_Ch1 = len(Ch1_AMPL)
    length_Ch2 = len(Ch2_AMPL)
    length_Ch3 = len(Ch3_TIME)
    length_Ch4 = len(Ch4_TIME)

    aaccepted_ampl_1 = []
    aaccepted_time_1 = []

    aaccepted_ampl_2 = []
    aaccepted_time_2 = []

    while pos_Ch1 < length_Ch1 and pos_Ch2 < length_Ch2:
        min_val = min(Ch1_TIME[pos_Ch1], Ch2_TIME[pos_Ch2])
        max_val = max(Ch1_TIME[pos_Ch1], Ch2_TIME[pos_Ch2])

        CH3_IS_ANTI = True
        while Ch3_TIME[pos_Ch3] <= min_val + t_window:
            if Ch3_TIME[pos_Ch3] >= min_val:
                CH3_IS_ANTI = False
                break

            if pos_Ch3 < length_Ch3 - 1:
                pos_Ch3 += 1
            else:
                break

        CH4_IS_ANTI = True
        while Ch4_TIME[pos_Ch4] <= min_val + t_window:
            if Ch4_TIME[pos_Ch4] >= min_val:
                CH4_IS_ANTI = False
                break

            if pos_Ch4 < length_Ch4 - 1:
                pos_Ch4 += 1
            else:
                break

        if max_val - min_val <= t_window and CH3_IS_ANTI and CH4_IS_ANTI:

            aaccepted_ampl_1.append(Ch1_AMPL[pos_Ch1])
            aaccepted_time_1.append(Ch1_TIME[pos_Ch1])

            aaccepted_ampl_2.append(Ch2_AMPL[pos_Ch2])
            aaccepted_time_2.append(Ch2_TIME[pos_Ch2])

            pos_Ch1 += 1
            pos_Ch2 += 1
        else:
            if min_val == Ch1_TIME[pos_Ch1]:
                pos_Ch1 += 1
            if min_val == Ch2_TIME[pos_Ch2]:
                pos_Ch2 += 1

    return aaccepted_time_1, aaccepted_time_2, aaccepted_ampl_1, aaccepted_ampl_2


def calculate_coincidence(
    A_time,
    A_ampl,
    B_time,
    B_ampl,
    C_time,
    C_ampl,
    D_time,
    D_ampl,
    coincidence_window,
    coincidence_citeria,
):
    # Amplitude in mV
    # Time in s

    Channel_names = ["A", "B", "C", "D"]
    coincidence_citeria = np.array(coincidence_citeria)

    grouped_data = [
        [A_time, A_ampl],
        [B_time, B_ampl],
        [C_time, C_ampl],
        [D_time, D_ampl],
    ]

    number_of_ignore = len(np.where(np.array(coincidence_citeria) == 0)[0])
    number_of_coincidence = len(np.where(np.array(coincidence_citeria) == 1)[0])
    number_of_anti_coincidence = len(np.where(np.array(coincidence_citeria) == 2)[0])
    print(
        f"Ignore: {number_of_ignore}, Coincidence: {number_of_coincidence}, Anti-Coincidence: {number_of_anti_coincidence}"
    )

    # Coincidence between two data channels:
    if number_of_coincidence == 2 and number_of_ignore == 2:
        which_data_channels = np.where(coincidence_citeria == 1)[0]

        ch1 = Channel_names[which_data_channels[0]]
        ch2 = Channel_names[which_data_channels[1]]
        print(f"Coincidence between {ch1} and {ch2}")
        first_data = grouped_data[which_data_channels[0]]
        second_data = grouped_data[which_data_channels[1]]

        result = COINC_2(
            first_data[0],
            second_data[0],
            first_data[1],
            second_data[1],
            t_window=coincidence_window,
        )

        df = pd.DataFrame(
            {
                f"{ch1}_time [s]": np.array(result[0]),
                f"{ch1}_amplitude [mV]": np.array(result[2]),
                f"{ch2}_time [s]": np.array(result[1]),
                f"{ch2}_amplitude [mV]": np.array(result[3]),
                "Sum_amplitude [mV]": np.array(result[2]) + np.array(result[3]),
            }
        )
        return df

    # Coincidence between three data channels:
    elif number_of_coincidence == 3 and number_of_ignore == 1:
        which_data_channels = np.where(coincidence_citeria == 1)[0]

        ch1 = Channel_names[which_data_channels[0]]
        ch2 = Channel_names[which_data_channels[1]]
        ch3 = Channel_names[which_data_channels[2]]

        print(f"Coincidence between {ch1}, {ch2} and {ch3}")
        first_data = grouped_data[which_data_channels[0]]
        second_data = grouped_data[which_data_channels[1]]
        third_data = grouped_data[which_data_channels[2]]

        result = COINC_3(
            first_data[0],
            second_data[0],
            third_data[0],
            first_data[1],
            second_data[1],
            third_data[1],
            t_window=coincidence_window,
        )

        df = pd.DataFrame(
            {
                f"{ch1}_time [s]": np.array(result[0]),
                f"{ch1}_amplitude [mV]": np.array(result[3]),
                f"{ch2}_time [s]": np.array(result[1]),
                f"{ch2}_amplitude [mV]": np.array(result[4]),
                f"{ch3}_time [s]": np.array(result[2]),
                f"{ch3}_amplitude [mV]": np.array(result[5]),
                "Sum_amplitude [mV]": np.array(result[3])
                + np.array(result[4])
                + np.array(result[5]),
            }
        )
        return df

    # Coincidence between all four data channels:
    elif number_of_coincidence == 4:
        which_data_channels = np.where(coincidence_citeria == 1)[0]

        ch1 = Channel_names[which_data_channels[0]]
        ch2 = Channel_names[which_data_channels[1]]
        ch3 = Channel_names[which_data_channels[2]]
        ch4 = Channel_names[which_data_channels[3]]

        print(f"Coincidence between {ch1}, {ch2}, {ch3} and {ch4}")
        first_data = grouped_data[which_data_channels[0]]
        second_data = grouped_data[which_data_channels[1]]
        third_data = grouped_data[which_data_channels[2]]
        fourth_data = grouped_data[which_data_channels[3]]

        result = COINC_4(
            first_data[0],
            second_data[0],
            third_data[0],
            fourth_data[0],
            first_data[1],
            second_data[1],
            third_data[1],
            fourth_data[1],
            t_window=coincidence_window,
        )

        df = pd.DataFrame(
            {
                f"{ch1}_time [s]": np.array(result[0]),
                f"{ch1}_amplitude [mV]": np.array(result[4]),
                f"{ch2}_time [s]": np.array(result[1]),
                f"{ch2}_amplitude [mV]": np.array(result[5]),
                f"{ch3}_time [s]": np.array(result[2]),
                f"{ch3}_amplitude [mV]": np.array(result[6]),
                f"{ch4}_time [s]": np.array(result[3]),
                f"{ch4}_amplitude [mV]": np.array(result[7]),
                "Sum_amplitude [mV]": np.array(result[4])
                + np.array(result[5])
                + np.array(result[6])
                + np.array(result[7]),
            }
        )
        return df

    # Coincidence between two channels and anti-coincidence with a third one
    elif number_of_coincidence == 2 and number_of_anti_coincidence == 1:
        which_coinc_data_channels = np.where(coincidence_citeria == 1)[0]
        which_anti_data_channels = np.where(coincidence_citeria == 2)[0]

        # Coincidence channels:
        ch1 = Channel_names[which_coinc_data_channels[0]]
        ch2 = Channel_names[which_coinc_data_channels[1]]

        # Anti-coincidence channel
        ch3 = Channel_names[which_anti_data_channels[0]]

        print(f"Coincidence between {ch1} and {ch2} and anti-coincidence with {ch3}")
        first_data = grouped_data[which_coinc_data_channels[0]]
        second_data = grouped_data[which_coinc_data_channels[1]]

        third_data = grouped_data[which_anti_data_channels[0]]

        result = COINC_2_ANTI_1(
            first_data[0],
            second_data[0],
            third_data[0],
            first_data[1],
            second_data[1],
            third_data[1],
            t_window=coincidence_window,
        )

        df = pd.DataFrame(
            {
                f"{ch1}_time [s]": np.array(result[0]),
                f"{ch1}_amplitude [mV]": np.array(result[2]),
                f"{ch2}_time [s]": np.array(result[1]),
                f"{ch2}_amplitude [mV]": np.array(result[3]),
                "Sum_amplitude [mV]": np.array(result[2]) + np.array(result[3]),
            }
        )
        return df

    # Coincidence between three channels and anti-coincidence with a fourth one
    elif number_of_coincidence == 3 and number_of_anti_coincidence == 1:
        which_coinc_data_channels = np.where(coincidence_citeria == 1)[0]
        which_anti_data_channels = np.where(coincidence_citeria == 2)[0]

        # Coincidence channels:
        ch1 = Channel_names[which_coinc_data_channels[0]]
        ch2 = Channel_names[which_coinc_data_channels[1]]
        ch3 = Channel_names[which_coinc_data_channels[2]]

        # Anti-coincidence channel
        ch4 = Channel_names[which_anti_data_channels[0]]

        print(
            f"Coincidence between {ch1}, {ch2} and {ch3} and anti-coincidence with {ch4}"
        )
        first_data = grouped_data[which_coinc_data_channels[0]]
        second_data = grouped_data[which_coinc_data_channels[1]]
        third_data = grouped_data[which_coinc_data_channels[2]]

        fourth_data = grouped_data[which_anti_data_channels[0]]

        result = COINC_3_ANTI_1(
            first_data[0],
            second_data[0],
            third_data[0],
            fourth_data[0],
            first_data[1],
            second_data[1],
            third_data[1],
            fourth_data[1],
            t_window=coincidence_window,
        )

        df = pd.DataFrame(
            {
                f"{ch1}_time [s]": np.array(result[0]),
                f"{ch1}_amplitude [mV]": np.array(result[3]),
                f"{ch2}_time [s]": np.array(result[1]),
                f"{ch2}_amplitude [mV]": np.array(result[4]),
                f"{ch3}_time [s]": np.array(result[2]),
                f"{ch3}_amplitude [mV]": np.array(result[5]),
                "Sum_amplitude [mV]": np.array(result[3])
                + np.array(result[4])
                + np.array(result[5]),
            }
        )
        return df

    # Coincidence between two channels and anti-coincidence with the remainign two channels
    elif number_of_coincidence == 2 and number_of_anti_coincidence == 2:
        which_coinc_data_channels = np.where(coincidence_citeria == 1)[0]
        which_anti_data_channels = np.where(coincidence_citeria == 2)[0]

        # Coincidence channels:
        ch1 = Channel_names[which_coinc_data_channels[0]]
        ch2 = Channel_names[which_coinc_data_channels[1]]

        # Anti-coincidence channel
        ch3 = Channel_names[which_anti_data_channels[0]]
        ch4 = Channel_names[which_anti_data_channels[1]]

        print(
            f"Coincidence between {ch1} and {ch2} and anti-coincidence with {ch3}  and {ch4} "
        )
        first_data = grouped_data[which_coinc_data_channels[0]]
        second_data = grouped_data[which_coinc_data_channels[1]]

        third_data = grouped_data[which_anti_data_channels[0]]
        fourth_data = grouped_data[which_anti_data_channels[1]]

        result = COINC_2_ANTI_2(
            first_data[0],
            second_data[0],
            third_data[0],
            fourth_data[0],
            first_data[1],
            second_data[1],
            third_data[1],
            fourth_data[1],
            t_window=coincidence_window,
        )

        df = pd.DataFrame(
            {
                f"{ch1}_time [s]": np.array(result[0]),
                f"{ch1}_amplitude [mV]": np.array(result[2]),
                f"{ch2}_time [s]": np.array(result[1]),
                f"{ch2}_amplitude [mV]": np.array(result[3]),
                "Sum_amplitude [mV]": np.array(result[2]) + np.array(result[3]),
            }
        )
        return df
