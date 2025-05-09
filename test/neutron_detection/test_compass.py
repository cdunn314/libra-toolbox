import pytest
import numpy as np
import os
from libra_toolbox.neutron_detection.activation_foils import compass
from pathlib import Path
import datetime


@pytest.mark.parametrize(
    "filename, expected_channel",
    [
        ("Data_CH14@V1725_292_Background_250322.CSV", 14),
        ("Data_CH7@V1725_123_Background_250322.CSV", 7),
        ("Data_CH21@V1725_456_Background_250322.CSV", 21),
    ],
)
def test_get_channel(filename, expected_channel):
    ch = compass.get_channel(filename)
    assert ch == expected_channel


def create_empty_csv_files(directory, base_name, count, channel):
    """
    Creates empty CSV files in a specified directory with a specific pattern.

    Args:
        directory (str): The directory where the files will be created.
        base_name (str): The base name of the file (e.g., "Data_CH14").
        count (int): The number of files to generate.

    Returns:
        list: A list of file paths for the created CSV files.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_paths = []
    for i in range(count):
        if i == 0:
            filename = f"Data_CH{channel}@{base_name}.csv"
        else:
            filename = f"Data_CH{channel}@{base_name}_{i}.csv"
        file_path = os.path.join(directory, filename)
        with open(file_path, "w") as f:
            pass  # Create an empty file
        file_paths.append(file_path)

    return file_paths


@pytest.mark.parametrize(
    "base_name, expected_filenames",
    [
        (
            "base",
            {
                4: [
                    "Data_CH4@base.csv",
                    "Data_CH4@base_1.csv",
                    "Data_CH4@base_2.csv",
                    "Data_CH4@base_3.csv",
                ],
                1: [
                    "Data_CH1@base.csv",
                ],
            },
        ),
    ],
)
def test_sort_compass_files(tmpdir, base_name: str, expected_filenames: dict):
    for ch, list_of_filenames in expected_filenames.items():
        create_empty_csv_files(
            tmpdir, base_name, count=len(list_of_filenames), channel=ch
        )

    data_filenames = compass.sort_compass_files(tmpdir)

    assert isinstance(data_filenames, dict)

    # Check if dictionaries have the same keys, length of filenames array, and
    # the same overall filenames array
    for key in expected_filenames:
        assert key in data_filenames
        assert len(data_filenames[key]) == len(expected_filenames[key])
        for a, b in zip(data_filenames[key], expected_filenames[key]):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                assert np.array_equal(a, b)
            else:
                assert a == b


@pytest.mark.parametrize(
    "expected_time, expected_energy, expected_idx",
    [
        (6685836624, 515, 5),
        (11116032249, 568, 6),
        (1623550122, 589, -1),
        (535148093, 1237, -2),
    ],
)
def test_get_events(expected_time, expected_energy, expected_idx):
    """
    Test the get_events function from the compass module.
    Checks that specific time and energy values are returned for a given channel
    """
    test_directory = Path(__file__).parent / "compass_test_data/events"
    times, energies = compass.get_events(test_directory)
    assert isinstance(times, dict)
    assert isinstance(energies, dict)

    expected_keys = [5, 15]
    for key in expected_keys:
        assert key in times
        assert key in energies

    ch = 5
    assert times[ch][expected_idx] == expected_time
    assert energies[ch][expected_idx] == expected_energy


utc_minus5 = datetime.timezone(datetime.timedelta(hours=-5))
utc_minus4 = datetime.timezone(datetime.timedelta(hours=-4))


@pytest.mark.parametrize(
    "start_time, stop_time",
    [
        (
            datetime.datetime(
                2024, 11, 7, 15, 47, 21, microsecond=127000, tzinfo=utc_minus5
            ),
            datetime.datetime(
                2024, 11, 7, 16, 2, 21, microsecond=133000, tzinfo=utc_minus5
            ),
        ),
        (
            datetime.datetime(
                2025, 3, 18, 22, 19, 3, microsecond=947000, tzinfo=utc_minus4
            ),
            datetime.datetime(
                2025, 3, 19, 9, 21, 6, microsecond=558000, tzinfo=utc_minus4
            ),
        ),
    ],
)
def test_get_start_stop_time(tmpdir, start_time, stop_time):
    """
    Tests the get_start_stop_time function from the compass module.
    Checks that the start and stop times are correctly parsed from the run.info file.
    """
    # BUILD
    content = _run_info_content(start_time, stop_time)

    # Create another temporary directory
    tmpdir2 = os.path.join(tmpdir, "tmpdir2")

    # create an empty run.info file
    run_info_path = os.path.join(tmpdir, "run.info")

    # add some stuff
    with open(run_info_path, "w") as f:
        f.write(content)

    # RUN
    start_time_out, stop_time_out = compass.get_start_stop_time(tmpdir2)

    # TEST
    assert isinstance(start_time_out, datetime.datetime)
    assert start_time_out == start_time

    assert isinstance(stop_time_out, datetime.datetime)
    assert stop_time_out == stop_time


def _run_info_content(start_time: datetime.datetime, stop_time: datetime.datetime):
    """
    Creates a string that simulates the content of a run.info file.
    """
    return f"""id=Co60_0_872uCi_19Mar14_241107
time.start={start_time.strftime("%Y/%m/%d %H:%M:%S.%f%z")}
time.stop={stop_time.strftime("%Y/%m/%d %H:%M:%S.%f%z")}
time.real=00:15:00
board.0-14-292.readout.rate=132.731 kb/s
board.0-14-292.1.rejections.singles=0.0
board.0-14-292.1.rejections.pileup=0.0
board.0-14-292.1.rejections.saturation=1729.15
board.0-14-292.1.rejections.energy=0.0
board.0-14-292.1.rejections.psd=0.0
board.0-14-292.1.rejections.timedistribution=0.0
board.0-14-292.1.throughput=6950.66
board.0-14-292.1.icr=7424.44
board.0-14-292.1.ocr=5253.24
board.0-14-292.1.calibration.energy.c0=0.0
board.0-14-292.1.calibration.energy.c1=1.0
board.0-14-292.1.calibration.energy.c2=0.0
board.0-14-292.1.calibration.energy.uom=keV
board.0-14-292.2.rejections.singles=0.0
board.0-14-292.2.rejections.pileup=0.0
board.0-14-292.2.rejections.saturation=8.2202
board.0-14-292.2.rejections.energy=0.0
board.0-14-292.2.rejections.psd=0.0
board.0-14-292.2.rejections.timedistribution=0.0
board.0-14-292.2.throughput=3958.96
board.0-14-292.2.icr=3981.66
board.0-14-292.2.ocr=3952.89
board.0-14-292.2.calibration.energy.c0=0.0
board.0-14-292.2.calibration.energy.c1=1.0
board.0-14-292.2.calibration.energy.c2=0.0
board.0-14-292.2.calibration.energy.uom=keV
"""


def test_filenotfound_error_info():
    with pytest.raises(FileNotFoundError, match="Could not find run.info"):
        compass.get_start_stop_time(
            directory=Path(__file__).parent / "compass_test_data/events"
        )


def test_get_start_stop_time_with_notime(tmpdir):
    """Creates an empty file run.info and check that an error is raised if can't find time"""

    # Create another temporary directory

    tmpdir2 = os.path.join(tmpdir, "tmpdir2")

    # create an empty run.info file
    run_info_path = os.path.join(tmpdir, "run.info")

    # add some stuff
    with open(run_info_path, "w") as f:
        f.write("coucou\ncoucou\n")

    # run
    with pytest.raises(ValueError, match="Could not find time.start or time.stop"):
        compass.get_start_stop_time(tmpdir2)


@pytest.mark.parametrize(
    "root_filename, channel, live_time, real_time",
    [
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Co60_20241107.root",
            1,
            808.305,
            900.108,
        ),
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Co60_20241107.root",
            2,
            896.374,
            900.108,
        ),
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Zirconium_20250319.root",
            4,
            35654.785,
            39722.502,
        ),
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Zirconium_20250319.root",
            5,
            39678.458,
            39722.502,
        ),
    ],
)
def test_get_live_time_from_root(root_filename, channel, live_time, real_time):
    live_time_out, real_time_out = compass.get_live_time_from_root(
        root_filename, channel
    )
    assert live_time_out == live_time
    assert real_time_out == real_time


@pytest.mark.parametrize("no_root", [True, False])
def test_measurement_object_from_directory(no_root):
    """
    Test the Measurement object creation from a directory.
    """
    if no_root:
        test_directory = (
            Path(__file__).parent
            / "compass_test_data/complete_measurement_no_root/data"
        )
    else:
        test_directory = (
            Path(__file__).parent / "compass_test_data/complete_measurement/data"
        )

    measurement = compass.Measurement.from_directory(test_directory, name="test")

    assert len(measurement.detectors) == 1
    assert isinstance(measurement.detectors[0], compass.Detector)
    assert measurement.detectors[0].channel_nb == 1

    assert measurement.detectors[0].events.shape[1] == 2

    measurement.detectors[0].get_energy_hist(bins=None)


@pytest.mark.parametrize(
    "bins",
    [
        10,
        20,
        50,
        100,
        None,
        np.arange(0, 10, 1),
        np.linspace(0, 10, num=100),
    ],
)
def test_detector_get_energy_hist(bins):
    """
    Test the get_energy_hist method of the Detector class.
    """
    my_detector = compass.Detector(channel_nb=1)
    my_detector.events = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
    )

    my_detector.get_energy_hist(bins=bins)


@pytest.mark.parametrize(
    "counting_time_background",
    [
        0.1,
        1,
        10,
        100,
        1000,
        3000,
    ],
)
def test_background_sub(counting_time_background):
    """
    Test the background subtraction method of the Detector class.
    """
    # BUILD

    def background_spectrum(energies):
        return np.ones_like(energies)

    def measured_spectrum(energies):
        return np.cos(energies / 10) + 10

    counting_time_measured = 3600
    counting_time_background = counting_time_measured * 5

    background_rate = 100000 / (3600)
    measurement_rate = 3 * background_rate

    nb_events_background = int(background_rate * counting_time_background)
    nb_events_measured = int(measurement_rate * counting_time_measured)
    nb_events_measured_bg_contrib = int(background_rate * counting_time_measured)

    # Define energy grid for sampling
    energy_grid = np.arange(100)

    # Calculate probability distributions using the spectrum functions
    bg_probabilities = background_spectrum(energy_grid)
    bg_probabilities = bg_probabilities / np.sum(bg_probabilities)  # Normalize
    measured_probabilities = measured_spectrum(energy_grid)
    measured_probabilities = measured_probabilities / np.sum(
        measured_probabilities
    )  # Normalize

    # Sample from these distributions
    energy_events_bg = np.random.choice(
        energy_grid, size=nb_events_background, p=bg_probabilities
    )
    energy_events_measured = np.random.choice(
        energy_grid, size=nb_events_measured, p=measured_probabilities
    )
    energy_events_measured_bg_contrib = np.random.choice(
        energy_grid, size=nb_events_measured_bg_contrib, p=bg_probabilities
    )

    energy_events_measured = np.concatenate(
        (energy_events_measured, energy_events_measured_bg_contrib)
    )

    # Create the measurement objects
    ps_to_seconds = 1e-12

    measurement = compass.Measurement("test")
    detector_meas = compass.Detector(channel_nb=1)
    detector_meas.real_count_time = counting_time_measured
    measurement.detectors = [detector_meas]
    time_events_measured = np.random.uniform(
        0, counting_time_measured, nb_events_measured + nb_events_measured_bg_contrib
    )
    time_events_measured *= 1 / ps_to_seconds
    time_events_measured.sort()
    detector_meas.events = np.column_stack(
        (time_events_measured, energy_events_measured)
    )

    background_measurment = compass.Measurement("background")
    background_detector = compass.Detector(channel_nb=1)
    background_detector.real_count_time = counting_time_background
    background_measurment.detectors = [background_detector]
    background_time_events = np.random.uniform(
        0, counting_time_background, nb_events_background
    )
    background_time_events *= 1 / ps_to_seconds
    background_time_events.sort()
    background_detector.events = np.column_stack(
        (background_time_events, energy_events_bg)
    )

    # RUN
    hist_bc_sub, _ = detector_meas.get_energy_hist_background_substract(
        background_detector=background_detector
    )

    # TEST
    hist_bg, _ = background_detector.get_energy_hist()
    hist_raw, _ = detector_meas.get_energy_hist()
    expected_hist = (
        hist_raw - hist_bg / counting_time_background * counting_time_measured
    )
    assert np.allclose(hist_bc_sub, expected_hist, rtol=1e-1)
