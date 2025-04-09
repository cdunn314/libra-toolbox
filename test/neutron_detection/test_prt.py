import h5py
import numpy as np
import pytest

from libra_toolbox.neutron_detection.diamond import prt

ns_to_s = 1e-9


@pytest.fixture
def h5_file_with_data(tmpdir):
    """Fixture to create an HDF5 file with structured data for testing."""
    filename = tmpdir + "/test_file.h5"

    # Create structured array for Channel A
    data_channel_a = np.zeros(
        100, dtype=[("Time [ns]", float), ("Amplitude [mV]", float)]
    )
    data_channel_a["Time [ns]"] = np.random.rand(100) * 1e9  # Example time values in ns
    data_channel_a["Amplitude [mV]"] = (
        np.random.rand(100) * 100
    )  # Example amplitude values in mV

    # Create structured array for Channel B
    data_channel_b = np.zeros(
        100, dtype=[("Time [ns]", float), ("Amplitude [mV]", float)]
    )
    data_channel_b["Time [ns]"] = np.random.rand(100) * 1e9
    data_channel_b["Amplitude [mV]"] = np.random.rand(100) * 100

    # Create HDF5 file
    with h5py.File(filename, "w") as f:
        f.attrs["Active channels"] = [True, True]

        channel_a = f.create_group("Channel A")
        channel_a.create_dataset(name="Amplitude-Timestamp", data=data_channel_a)

        channel_b = f.create_group("Channel B")
        channel_b.create_dataset(name="Amplitude-Timestamp", data=data_channel_b)

    return filename, data_channel_a, data_channel_b


def test_get_timestamps_and_amplitudes(h5_file_with_data):
    """
    Test the get_timestamps_and_amplitudes function.
    This function retrieves timestamps and amplitudes from a given HDF5 file.
    It checks if the retrieved data matches the expected data.

    Args:
        h5_file_with_data: Fixture that provides a temporary HDF5 file with structured data.
    """
    filename, data_channel_a, _ = h5_file_with_data

    # run
    with h5py.File(filename, "r") as ROSY_file:
        timestamps, amplitudes = prt.get_timestamps_and_amplitudes(
            ROSY_file, channel="Channel A"
        )

    # test
    assert np.array_equal(timestamps, data_channel_a["Time [ns]"] * ns_to_s)
    assert np.array_equal(amplitudes, data_channel_a["Amplitude [mV]"])


def test_load_data_from_file(h5_file_with_data):
    """
    Test the load_data_from_file function.
    This function loads data from a given HDF5 file and checks if the loaded data
    matches the expected data.

    Args:
        h5_file_with_data: Fixture that provides a temporary HDF5 file with structured data.
    """
    filename, data_channel_a, data_channel_b = h5_file_with_data

    data = prt.load_data_from_file(filename)
    assert "Channel A" in data
    assert "Channel B" in data
    assert np.array_equal(
        data["Channel A"]["timestamps"],
        data_channel_a["Time [ns]"] * ns_to_s,
    )
    assert np.array_equal(
        data["Channel A"]["amplitudes"], data_channel_a["Amplitude [mV]"]
    )
    assert np.array_equal(
        data["Channel B"]["timestamps"],
        data_channel_b["Time [ns]"] * ns_to_s,
    )
    assert np.array_equal(
        data["Channel B"]["amplitudes"], data_channel_b["Amplitude [mV]"]
    )


@pytest.mark.parametrize("bin_time", [1, 10, 100])
@pytest.mark.parametrize("count_rate_real", [1, 10, 100])
def test_get_count_rate(bin_time: float, count_rate_real: float):
    """
    Test the get_count_rate function.
    This function calculates the count rate from given timestamps and checks
    if the calculated count rate matches the expected count rate.
    Args:
        bin_time: The bin time in seconds.
        count_rate_real: The expected count rate in Hz.
    """
    # Example data
    total_time = 1000  # seconds
    timestamps = np.linspace(0, total_time, num=count_rate_real * total_time)

    # run
    count_rates, _ = prt.get_count_rate(timestamps, bin_time=bin_time)

    # test
    assert np.allclose(count_rates, count_rate_real)
