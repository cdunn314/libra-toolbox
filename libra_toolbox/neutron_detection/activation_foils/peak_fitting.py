import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from typing import List, Dict, Union
import datetime
from libra_toolbox.neutron_detection.activation_foils.compass import (
    Detector,
    Measurement,
)


def get_peak_inputs(samples):
    default_inputs = {
        "Na22": {"prom_factor": 0.075, "width": [10, 150], "start_index": 100},
        "Co60": {"prom_factor": 0.2, "width": [10, 150], "start_index": 400},
        "Ba133": {"prom_factor": 0.1, "width": [10, 200], "start_index": 100},
        "Mn54": {"prom_factor": 0.2, "width": [10, 100], "start_index": 100},
    }

    defaults = {"prom_factor": 0.075, "width": [10, 150], "start_index": 100}

    peak_inputs = {}
    for sample in samples:
        if sample in default_inputs.keys():
            peak_inputs[sample] = default_inputs[sample]
        else:
            peak_inputs[sample] = defaults
    return peak_inputs


def get_peaks(hist: np.ndarray, source: str) -> np.ndarray:
    """Returns the peak indices of the histogram

    Args:
        hist: a histogram
        source: the type of source (eg. "Na22", "Co60", "Ba133", "Mn54")

    Returns:
        the peak indices in ``hist``
    """
    start_index = 100
    prominence = 0.10 * np.max(hist[start_index:])
    height = 0.10 * np.max(hist[start_index:])
    width = [10, 150]
    indices = None
    distance = 30
    if "na22" in source.lower():
        # find 511 keV peak first
        prominence = 0.01 * np.max(hist[start_index:])
        height = 0.9 * np.max(hist[start_index:])
        width = [10, 200]
    elif "co60" in source.lower():
        start_index = 400
        height = 0.60 * np.max(hist[start_index:])
        prominence = None
    elif "ba133" in source.lower():
        width = [10, 200]
    elif "mn54" in source.lower():
        height = 0.6 * np.max(hist[start_index:])
    peaks, peak_data = find_peaks(
        hist[start_index:],
        prominence=prominence,
        height=height,
        width=width,
        distance=distance,
    )
    peaks = np.array(peaks) + start_index
    if "na22" in source.lower():
        # Find 1275 keV peak
        peak_511 = peaks[0]
        start_index = peak_511 + 100
        prominence = 0.5 * np.max(hist[start_index:])
        height = 0.10 * np.max(hist[start_index:])

        high_peaks, peak_data = find_peaks(
            hist[start_index:],
            prominence=prominence,
            height=height,
            width=width,
            distance=distance,
        )
        high_peaks = np.array(high_peaks) + start_index
        peaks = [peak_511, high_peaks[0]]

    if indices:
        peaks = peaks[[indices]][0]

    return peaks


def get_calibration_data(
    check_source_measurements: List[Measurement],
    background_measurement: Measurement,
    channel_nb: int,
    decay_lines,
):
    background_detector = [
        detector
        for detector in background_measurement.detectors
        if detector.channel_nb == detector.channel_nb
    ][0]

    calibration_energies = []
    calibration_channels = []

    for measurement in check_source_measurements.values():
        for detector in measurement.detectors:
            if detector.channel_nb != channel_nb:
                continue

            sample = measurement.name[:-2]

            hist, bin_edges = detector.get_energy_hist_background_substract(
                background_detector, bins="double"
            )
            peaks_ind = get_peaks(hist, sample)
            peaks = bin_edges[peaks_ind]

            if len(peaks) != len(decay_lines[sample]["energy"]):
                raise ValueError(
                    f"SciPy find_peaks() found {len(peaks)} photon peaks, while {len(decay_lines[sample]["energy"])} were expected"
                )
            calibration_channels += list(peaks)
            calibration_energies += decay_lines[sample]["energy"]

    inds = np.argsort(calibration_channels)
    calibration_channels = np.array(calibration_channels)[inds]
    calibration_energies = np.array(calibration_energies)[inds]

    return calibration_channels, calibration_energies


def get_calibration_curve(
    check_source_measurements: List[Measurement],
    background_measurement: Measurement,
    channel_nb: int,
    decay_lines,
):

    calibration_channels, calibration_energies = get_calibration_data(
        check_source_measurements,
        background_measurement,
        channel_nb,
        decay_lines,
    )

    # linear fit for calibration curve
    coeff = np.polyfit(
        calibration_channels,
        calibration_energies,
        1,
    )

    return coeff


def gauss(x, b, m, *args):
    """Creates a multipeak gaussian with a linear addition of the form:
    m * x + b + Sum_i (A_i * exp(-(x - x_i)**2) / (2 * sigma_i**2)"""

    out = m * x + b
    if np.mod(len(args), 3) == 0:
        for i in range(int(len(args) / 3)):
            out += args[i * 3 + 0] * np.exp(
                -((x - args[i * 3 + 1]) ** 2) / (2 * args[i * 3 + 2] ** 2)
            )
    else:
        raise ValueError("Incorrect number of gaussian arguments given.")
    return out


def fit_peak_gauss(hist, xvals, peak_ergs, search_width=600):

    search_start = np.argmin(
        np.abs((peak_ergs[0] - search_width / (2 * len(peak_ergs))) - xvals)
    )
    search_end = np.argmin(
        np.abs((peak_ergs[-1] + search_width / (2 * len(peak_ergs))) - xvals)
    )

    slope_guess = (hist[search_end] - hist[search_start]) / (
        xvals[search_end] - xvals[search_start]
    )

    guess_parameters = [0, slope_guess]

    for i in range(len(peak_ergs)):
        peak_ind = np.argmin(np.abs((peak_ergs[i]) - xvals))
        guess_parameters += [
            hist[peak_ind],
            peak_ergs[i],
            search_width / (3 * len(peak_ergs)),
        ]

    parameters, covariance = curve_fit(
        gauss,
        xvals[search_start:search_end],
        hist[search_start:search_end],
        p0=guess_parameters,
    )

    return parameters, covariance


def get_multipeak_area(hist, bins, peak_ergs, search_width=600):
    # get midpoints of every bin
    xvals = np.diff(bins) / 2 + bins[:-1]

    parameters, covariance = fit_peak_gauss(
        hist, xvals, peak_ergs, search_width=search_width
    )

    areas = []
    peak_starts = []
    peak_ends = []
    all_peak_params = []
    # peak_amplitudes = []
    for i in range(len(peak_ergs)):
        # peak_amplitudes += [parameters[2 + 3 * i]]
        mean = parameters[2 + 3 * i + 1]
        sigma = np.abs(parameters[2 + 3 * i + 2])
        peak_start = np.argmin(np.abs((mean - 3 * sigma) - xvals))
        peak_end = np.argmin(np.abs((mean + 3 * sigma) - xvals))

        peak_starts += [peak_start]
        peak_ends += [peak_end]

        # Use unimodal gaussian to estimate counts from just one peak
        peak_params = [parameters[0], parameters[1], parameters[2 + 3 * i], mean, sigma]
        all_peak_params += [peak_params]
        gross_area = np.trapezoid(
            gauss(xvals[peak_start:peak_end], *peak_params),
            x=xvals[peak_start:peak_end],
        )

        # Cut off trapezoidal area due to compton scattering and noise
        trap_cutoff_area = np.trapezoid(
            parameters[0] + parameters[1] * xvals[peak_start:peak_end],
            x=xvals[peak_start:peak_end],
        )
        area = gross_area - trap_cutoff_area
        areas += [area]

    return areas


def group_close_values(data, threshold=200):
    # Sort the data to group values sequentially
    data.sort()

    # Initialize groups and a temporary group
    groups = []
    temp_group = [data[0]]

    for i in range(1, len(data)):
        # Check if the current value is within the threshold of the last value in the temp group
        if abs(data[i] - temp_group[-1]) < threshold:
            temp_group.append(data[i])
        else:
            # Commit the temp group to groups and start a new group
            groups.append(tuple(temp_group))
            temp_group = [data[i]]

    # Add the last group
    groups.append(tuple(temp_group))

    return groups


def get_peak_areas(hist, bins, peak_ergs, overlap_width=200, search_width=400):

    areas = []
    # organize peak energies into tuples, in which peak energies close enough
    # to have overlapping peaks will be paired together
    erg_groups = group_close_values(peak_ergs, threshold=overlap_width)
    # print(erg_groups)

    for erg_group in erg_groups:
        areas += get_multipeak_area(
            hist, bins, erg_group, search_width=len(erg_group) * search_width
        )
        # print(areas)
    return areas


# should be a method of a class called CheckSourceMeasurement
def get_expected_activity(
    check_source_data: Dict[str, Union[float, List[float], datetime.date]],
    check_source_meas: Measurement,
) -> float:
    """
    Calculates the expected activity of a check source given the
    half-life and the date of the measurement.
    The expected activity is calculated using the formula:
    .. math:: A(t) = A_0 e^{-\\lambda t}

    where :math:`A_0` is the initial activity, :math:`\\lambda` is the decay constant
    and :math:`t` is the time since the measurement date.

    Args:
        check_source_data: _description_
        check_source_meas: _description_

    Returns:
        the expected activity of the check source in Bq
    """
    # expected activity
    decay_constant = np.log(2) / check_source_data["half_life"]

    # Convert date to datetime if needed
    if isinstance(check_source_data["activity_date"], datetime.date) and not isinstance(
        check_source_data["activity_date"], datetime.datetime
    ):
        activity_datetime = datetime.datetime.combine(
            check_source_data["activity_date"], datetime.time.min
        )
        # add a timezone
        activity_datetime = activity_datetime.replace(tzinfo=datetime.timezone.utc)
    else:
        activity_datetime = check_source_data["activity_date"]

    time = (check_source_meas.start_time - activity_datetime).total_seconds()
    act_expec = check_source_data["activity"] * np.exp(-decay_constant * time)
    return act_expec


# should be a method of a class called CheckSourceMeasurement
def compute_detection_efficiency(
    check_source_detector: Detector,
    background_detector: Detector,
    check_source_meas: Measurement,
    check_source_data: Dict[str, Union[float, List[float], datetime.date]],
    calibration_coeffs: np.ndarray,
) -> Union[np.ndarray, float]:
    """
    Computes the detection efficiency of a check source given the
    check source data and the calibration coefficients.
    The detection efficiency is calculated using the formula:
    .. math:: \\eta = \\frac{A_{meas}}{A_{expec}}

    where :math:`A_{meas}` is the measured activity and :math:`A_{expec}` is the expected activity.
    The measured activity is calculated using the formula:
    .. math:: A_{meas} = \\frac{A_{peak}}{I \\cdot t_{live}}

    where :math:`A_{peak}` is the area of the peak, :math:`I` is the intensity of the check source
    and :math:`t_{live}` is the live count time of the detector.

    Args:
        check_source_detector: _description_
        background_detector: _description_
        check_source_meas: _description_
        check_source_data: _description_
        calibration_coeffs: _description_

    Returns:
        the detection efficiency
    """

    hist, bin_edges = check_source_detector.get_energy_hist_background_substract(
        background_detector, bins="double"
    )

    calibrated_bin_bedges = np.polyval(calibration_coeffs, bin_edges)

    areas = get_multipeak_area(
        hist, calibrated_bin_bedges, check_source_data["energy"], search_width=800
    )

    act_meas = np.array(areas) / (
        np.array(check_source_data["intensity"]) * check_source_detector.live_count_time
    )
    act_meas_err = np.sqrt(np.array(areas)) / (
        np.array(check_source_data["intensity"]) * check_source_detector.live_count_time
    )

    act_expec = get_expected_activity(check_source_data, check_source_meas)

    detection_efficiency = act_meas / act_expec

    return detection_efficiency
