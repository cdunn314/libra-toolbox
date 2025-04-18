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

Changes:
- vectorised functions using numpy for performance
- refactoring and abstraction of the common logic
- removed unused arguments in anti-coincidence functions
"""


def coinc_2(Ch1_TIME, Ch2_TIME, Ch1_AMPL, Ch2_AMPL, t_window):
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


def coinc_3(Ch1_TIME, Ch2_TIME, Ch3_TIME, Ch1_AMPL, Ch2_AMPL, Ch3_AMPL, t_window):
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


def coinc_4(
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


def coinc_2_ANTI_1(Ch1_TIME, Ch2_TIME, Ch3_TIME, Ch1_AMPL, Ch2_AMPL, t_window):
    Ch1_TIME = np.asarray(Ch1_TIME)
    Ch2_TIME = np.asarray(Ch2_TIME)
    Ch3_TIME = np.asarray(Ch3_TIME)
    Ch1_AMPL = np.asarray(Ch1_AMPL)
    Ch2_AMPL = np.asarray(Ch2_AMPL)

    # Step 1: Find all time differences
    time_diff = np.abs(Ch1_TIME[:, None] - Ch2_TIME[None, :])
    match_indices = np.where(time_diff <= t_window)
    i1 = match_indices[0]
    i2 = match_indices[1]

    if len(i1) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Step 2: Compute t_min and t_max for matched pairs
    t_min = np.minimum(Ch1_TIME[i1], Ch2_TIME[i2])
    t_max = t_min + t_window

    # Step 3: Use searchsorted to check if any Ch3 event is in [t_min, t_max]
    idx_start = np.searchsorted(Ch3_TIME, t_min, side="left")
    idx_end = np.searchsorted(Ch3_TIME, t_max, side="right")
    is_anticoinc = idx_start == idx_end  # True if no Ch3 event in window

    # Step 4: Return only accepted coincidences
    return (
        Ch1_TIME[i1[is_anticoinc]],
        Ch2_TIME[i2[is_anticoinc]],
        Ch1_AMPL[i1[is_anticoinc]],
        Ch2_AMPL[i2[is_anticoinc]],
    )


def coinc_3_ANTI_1(
    Ch1_TIME,
    Ch2_TIME,
    Ch3_TIME,
    Ch4_TIME,
    Ch1_AMPL,
    Ch2_AMPL,
    Ch3_AMPL,
    t_window,
):
    Ch1_TIME = np.asarray(Ch1_TIME)
    Ch2_TIME = np.asarray(Ch2_TIME)
    Ch3_TIME = np.asarray(Ch3_TIME)
    Ch4_TIME = np.sort(np.asarray(Ch4_TIME))  # must be sorted for searchsorted

    Ch1_AMPL = np.asarray(Ch1_AMPL)
    Ch2_AMPL = np.asarray(Ch2_AMPL)
    Ch3_AMPL = np.asarray(Ch3_AMPL)

    # Step 1: Coincidences between Ch1 and Ch2
    diff12 = np.abs(Ch1_TIME[:, None] - Ch2_TIME[None, :])
    i1, i2 = np.where(diff12 <= t_window)

    if len(i1) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    # Step 2: Now for each (Ch1, Ch2) pair, find matching Ch3
    t12_avg = 0.5 * (Ch1_TIME[i1] + Ch2_TIME[i2])
    diff13 = np.abs(t12_avg[:, None] - Ch3_TIME[None, :])
    i_comb, i3 = np.where(diff13 <= t_window)

    # Keep only valid triplets (Ch1[i1[i_comb]], Ch2[i2[i_comb]], Ch3[i3])
    if len(i_comb) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    final_i1 = i1[i_comb]
    final_i2 = i2[i_comb]
    final_i3 = i3

    # Step 3: Anti-coincidence with Ch4
    t_min = np.minimum.reduce(
        [Ch1_TIME[final_i1], Ch2_TIME[final_i2], Ch3_TIME[final_i3]]
    )
    t_max = t_min + t_window

    idx_start = np.searchsorted(Ch4_TIME, t_min, side="left")
    idx_end = np.searchsorted(Ch4_TIME, t_max, side="right")
    is_anticoinc = idx_start == idx_end

    # Step 4: Return accepted triples (not coincident with Ch4)
    return (
        Ch1_TIME[final_i1[is_anticoinc]],
        Ch2_TIME[final_i2[is_anticoinc]],
        Ch3_TIME[final_i3[is_anticoinc]],
        Ch1_AMPL[final_i1[is_anticoinc]],
        Ch2_AMPL[final_i2[is_anticoinc]],
        Ch3_AMPL[final_i3[is_anticoinc]],
    )


def coinc_2_ANTI_2(
    Ch1_TIME,
    Ch2_TIME,
    Ch3_TIME,
    Ch4_TIME,
    Ch1_AMPL,
    Ch2_AMPL,
    t_window,
):
    Ch1_TIME = np.asarray(Ch1_TIME)
    Ch2_TIME = np.asarray(Ch2_TIME)
    Ch3_TIME = np.asarray(Ch3_TIME)
    Ch4_TIME = np.asarray(Ch4_TIME)

    Ch1_AMPL = np.asarray(Ch1_AMPL)
    Ch2_AMPL = np.asarray(Ch2_AMPL)

    # Step 1: Find all coincidences between Ch1 and Ch2
    diff12 = np.abs(Ch1_TIME[:, None] - Ch2_TIME[None, :])
    i1, i2 = np.where(diff12 <= t_window)

    if len(i1) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    t_min = np.minimum(Ch1_TIME[i1], Ch2_TIME[i2])
    t_max = t_min + t_window

    # Step 2: Check anti-coincidence with Ch3
    idx3_start = np.searchsorted(Ch3_TIME, t_min, side="left")
    idx3_end = np.searchsorted(Ch3_TIME, t_max, side="right")
    anticoinc_3 = idx3_start == idx3_end

    # Step 3: Check anti-coincidence with Ch4
    idx4_start = np.searchsorted(Ch4_TIME, t_min, side="left")
    idx4_end = np.searchsorted(Ch4_TIME, t_max, side="right")
    anticoinc_4 = idx4_start == idx4_end

    is_anticoinc = anticoinc_3 & anticoinc_4

    final_i1 = i1[is_anticoinc]
    final_i2 = i2[is_anticoinc]

    return (
        Ch1_TIME[final_i1],
        Ch2_TIME[final_i2],
        Ch1_AMPL[final_i1],
        Ch2_AMPL[final_i2],
    )


def process_coincidence(
    grouped_data, coincidence_channels, t_window, coincidence_function
):
    """
    Process coincidence for the given channels using the specified coincidence function.

    Args:
        grouped_data: List of grouped data for all channels.
        coincidence_channels: Indices of the channels involved in coincidence.
        t_window: Time window for coincidence detection.
        coincidence_function: Function to calculate coincidence.

    Returns:
        Result of the coincidence function.
    """
    data = [grouped_data[i] for i in coincidence_channels]
    times = [d[0] for d in data]
    amplitudes = [d[1] for d in data]

    return coincidence_function(*times, *amplitudes, t_window)


def process_anti_coincidence(
    grouped_data, coincidence_channels, anti_channels, t_window, anti_function
):
    """
    Process coincidence with anti-coincidence for the given channels.

    Args:
        grouped_data: List of grouped data for all channels.
        coincidence_channels: Indices of the channels involved in coincidence.
        anti_channels: Indices of the channels involved in anti-coincidence.
        t_window: Time window for coincidence detection.
        anti_function: Function to calculate coincidence with anti-coincidence.

    Returns:
        Result of the anti-coincidence function.
    """
    coinc_data = [grouped_data[i] for i in coincidence_channels]
    anti_data = [grouped_data[i] for i in anti_channels]

    coinc_times = [d[0] for d in coinc_data]
    coinc_amplitudes = [d[1] for d in coinc_data]

    anti_times = [d[0] for d in anti_data]

    return anti_function(*coinc_times, *anti_times, *coinc_amplitudes, t_window)


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

    channel_names = ["A", "B", "C", "D"]
    coincidence_citeria = np.array(coincidence_citeria)

    grouped_data = [
        [A_time, A_ampl],
        [B_time, B_ampl],
        [C_time, C_ampl],
        [D_time, D_ampl],
    ]

    number_of_ignore = len(np.where(coincidence_citeria == 0)[0])
    number_of_coincidence = len(np.where(coincidence_citeria == 1)[0])
    number_of_anti_coincidence = len(np.where(coincidence_citeria == 2)[0])

    print(
        f"Ignore: {number_of_ignore}, Coincidence: {number_of_coincidence}, Anti-Coincidence: {number_of_anti_coincidence}"
    )

    # Get indices of coincidence and anti-coincidence channels
    coincidence_channels = np.where(coincidence_citeria == 1)[0]
    anti_channels = np.where(coincidence_citeria == 2)[0]

    # Handle different cases
    if number_of_coincidence == 2 and number_of_anti_coincidence == 0:
        result = process_coincidence(
            grouped_data, coincidence_channels, coincidence_window, coinc_2
        )
    elif number_of_coincidence == 3 and number_of_anti_coincidence == 0:
        result = process_coincidence(
            grouped_data, coincidence_channels, coincidence_window, coinc_3
        )
    elif number_of_coincidence == 4:
        result = process_coincidence(
            grouped_data, coincidence_channels, coincidence_window, coinc_4
        )
    elif number_of_coincidence == 2 and number_of_anti_coincidence == 1:
        result = process_anti_coincidence(
            grouped_data,
            coincidence_channels,
            anti_channels,
            coincidence_window,
            coinc_2_ANTI_1,
        )
    elif number_of_coincidence == 3 and number_of_anti_coincidence == 1:
        result = process_anti_coincidence(
            grouped_data,
            coincidence_channels,
            anti_channels,
            coincidence_window,
            coinc_3_ANTI_1,
        )
    elif number_of_coincidence == 2 and number_of_anti_coincidence == 2:
        result = process_anti_coincidence(
            grouped_data,
            coincidence_channels,
            anti_channels,
            coincidence_window,
            coinc_2_ANTI_2,
        )
    else:
        raise ValueError("Unsupported combination of coincidence and anti-coincidence.")

    # Generate DataFrame dynamically
    df_data = {}
    for i, ch_idx in enumerate(coincidence_channels):
        ch_name = channel_names[ch_idx]
        df_data[f"{ch_name}_time [s]"] = np.array(result[i])
        df_data[f"{ch_name}_amplitude [mV]"] = np.array(
            result[len(coincidence_channels) + i]
        )

    if number_of_anti_coincidence == 0:
        df_data["Sum_amplitude [mV]"] = np.sum(
            [
                np.array(result[len(coincidence_channels) + i])
                for i in range(len(coincidence_channels))
            ],
            axis=0,
        )

    return pd.DataFrame(df_data)
