import pytest
import numpy as np
import os
from libra_toolbox.neutron_detection.activation_foils import compass
from pathlib import Path


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
    test_directory = Path(__file__).parent / "compass_test_data"
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
