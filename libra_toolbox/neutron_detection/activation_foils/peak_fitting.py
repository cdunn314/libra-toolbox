import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from typing import List, Dict
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


def gauss1(x, H, m, A, x0, sigma):
    return H + m * x + A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gauss2(x, H, m, A1, x1, sigma1, A2, x2, sigma2):
    out = (
        H
        + m * x
        + A1 * np.exp(-((x - x1) ** 2) / (2 * sigma1**2))
        + A2 * np.exp(-((x - x2) ** 2) / (2 * sigma2**2))
    )
    return out


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


def get_singlepeak_area(hist, bins, peak_erg, search_width=300):

    # get midpoints of every bin
    xvals = np.diff(bins) / 2 + bins[:-1]

    peak_ind = np.argmin(np.abs((peak_erg) - xvals))
    search_start = np.argmin(np.abs((peak_erg - search_width / 2) - xvals))
    search_end = np.argmin(np.abs((peak_erg + search_width / 2) - xvals))

    slope_guess = (hist[search_end] - hist[search_start]) / (
        xvals[search_end] - xvals[search_start]
    )

    guess_parameters = [0, slope_guess, hist[peak_ind], peak_erg, search_width / 6]
    # print(guess_parameters)

    parameters, covariance = curve_fit(
        gauss1,
        xvals[search_start:search_end],
        hist[search_start:search_end],
        p0=guess_parameters,
    )
    # print(parameters)

    mean = parameters[3]
    sigma = parameters[4]

    peak_start = np.argmin(np.abs((mean - 3 * sigma) - xvals))
    peak_end = np.argmin(np.abs((mean + 3 * sigma) - xvals))

    gross_area = np.trapezoid(hist[peak_start:peak_end], x=xvals[peak_start:peak_end])
    # trap_cutoff_area = (hist[peak_start] + hist[peak_end])/2 * (bins[peak_end] - bins[peak_start])
    trap_cutoff_area = np.trapezoid(
        parameters[0] + parameters[1] * xvals[peak_start:peak_end],
        x=xvals[peak_start:peak_end],
    )
    area = gross_area - trap_cutoff_area

    return area


def get_multipeak_area(hist, bins, peak_ergs, search_width=600):
    # get midpoints of every bin
    xvals = np.diff(bins) / 2 + bins[:-1]

    search_start = np.argmin(
        np.abs((peak_ergs[0] - search_width / (2 * len(peak_ergs))) - xvals)
    )
    search_end = np.argmin(
        np.abs((peak_ergs[-1] + search_width / (2 * len(peak_ergs))) - xvals)
    )

    guess_slope = (hist[search_end] - hist[search_start]) / (
        xvals[search_end] - xvals[search_start]
    )

    guess_parameters = [0, guess_slope]

    for i in range(len(peak_ergs)):
        peak_ind = np.argmin(np.abs((peak_ergs[i]) - xvals))
        guess_parameters += [
            hist[peak_ind],
            peak_ergs[i],
            search_width / (3 * len(peak_ergs)),
        ]

    # print(guess_parameters)

    parameters, covariance = curve_fit(
        gauss,
        xvals[search_start:search_end],
        hist[search_start:search_end],
        p0=guess_parameters,
    )
    # print(parameters)

    areas = []
    peak_starts = []
    peak_ends = []
    all_peak_params = []
    peak_amplitudes = []
    for i in range(len(peak_ergs)):
        peak_amplitudes += [parameters[2 + 3 * i]]
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


def energy_efficiency(
    counts,
    decay_lines,
    nuclides=None,
    overlap_width=200,
    search_width=400,
    degree=2,
    count_sum_peak=False,
):
    if nuclides is None:
        nuclides = decay_lines.keys()

    effs = {}
    eff_errs = {}
    energies = {}
    for nuc in nuclides:
        sum_peak = False
        if len(decay_lines[nuc]["energy"]) > 1 and count_sum_peak:
            peak_energies = decay_lines[nuc]["energy"] + [
                np.sum(decay_lines[nuc]["energy"])
            ]
            sum_peak = True
        else:
            peak_energies = decay_lines[nuc]["energy"]
        for ch in counts[nuc].keys():
            # initialize efficiency list
            if ch not in effs.keys():
                effs[ch] = []
                eff_errs[ch] = []
                energies[ch] = []

            # print(nuc, ' Ch ', ch )
            areas = get_peak_areas(
                counts[nuc][ch]["hist"],
                counts[nuc][ch]["calibrated_bin_edges"],
                peak_energies,
                overlap_width=overlap_width,
                search_width=search_width,
            )
            if sum_peak:
                areas = np.array(areas)[:-1] + areas[-1] / len(areas[:-1])
            # print('Peak areas: ', areas)
            # measured activity
            # I think this should be divided by live count time, but maybe it should
            # be divided by real count time?? Should go over this again
            act_meas = np.array(areas) / (
                np.array(decay_lines[nuc]["intensity"])
                * counts[nuc][ch]["live_count_time"]
            )
            # print('Activity measured: ', act_meas)
            act_meas_err = np.sqrt(np.array(areas)) / (
                np.array(decay_lines[nuc]["intensity"])
                * counts[nuc][ch]["live_count_time"]
            )
            # expected activity
            l = np.log(2) / decay_lines[nuc]["half_life"]
            # print('decay constant: ', l)
            time = (
                counts[nuc][ch]["start_time"] - decay_lines[nuc]["activity_date"]
            ).total_seconds()
            # print('count time: ', time)
            act_expec = decay_lines[nuc]["activity"] * np.exp(-l * time)
            # print('Activity expected: ', act_expec)
            # print('efficiency: ', act_meas/act_expec)

            effs[ch] += list(act_meas / act_expec)
            eff_errs[ch] += list(act_meas_err / act_expec)
            energies[ch] += decay_lines[nuc]["energy"]

    # Sort the data
    # print('effs: ', effs)
    # print('energies: ', energies)
    coeff = {}
    bounds = {}
    for ch in effs.keys():
        ind = np.argsort(energies[ch])
        energies[ch] = np.array(energies[ch])[ind]
        effs[ch] = np.array(effs[ch])[ind]
        eff_errs[ch] = np.array(eff_errs[ch])[ind]

        # Create polynomial fit
        coeff[ch] = np.polyfit(energies[ch], effs[ch], degree)

        # Get bounds of fit for interpolation
        bounds[ch] = [np.min(energies[ch]), np.max(energies[ch])]

    return effs, eff_errs, coeff, bounds
