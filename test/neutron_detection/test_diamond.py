from libra_toolbox.neutron_detection.diamond.process_data import *
import pytest


def test_get_avg_neutron_rate(tmpdir):

    time_values = np.arange(1, 10)
    print(time_values)
    t_min = 2.5
    t_max = 9.0

    # read the data back
    processor = DataProcessor()
    processor.time_values = time_values

    avg_neutron_rate, error = processor.get_avg_rate(t_min, t_max)

    expected_number_of_counts = 6  # there are 6 counts between 2.5 and 9.0

    assert avg_neutron_rate == expected_number_of_counts / (t_max - t_min)
    assert error == np.sqrt(expected_number_of_counts) / (t_max - t_min)


@pytest.mark.parametrize("delimiter", [",", ";", "\t"])
@pytest.mark.parametrize("extention", ["csv", "CSV"])
def test_get_time_energy_values(tmpdir, delimiter, extention):

    # make data
    time_values_out = np.random.rand(10)
    energy_values_out = np.random.rand(10)
    extra_column = np.random.rand(10)

    # dump 2 columns to the same csv file
    filename = tmpdir.join(f"test.{extention}")
    np.savetxt(
        filename,
        np.column_stack((time_values_out, energy_values_out, extra_column)),
        delimiter=delimiter,
    )

    # read the data back
    processor = DataProcessor()
    processor.add_file(
        filename, time_column=0, energy_column=1, delimiter=delimiter, scale_time=False
    )

    time_values_in, energy_values_in = processor.time_values, processor.energy_values

    # test

    # sort values out based on time
    inds = np.argsort(time_values_out)
    time_values_out = time_values_out[inds]
    energy_values_out = energy_values_out[inds]

    assert np.allclose(time_values_in, time_values_out)
    assert np.allclose(energy_values_in, energy_values_out)


def test_get_count_rate(tmpdir):

    total_time_s = 100  # s
    total_time_ps = total_time_s * 1e12  # s to ps

    # peak 1
    nb_counts_peak1 = int(2e4)
    size_peak1 = nb_counts_peak1
    time_values_out_peak1 = np.random.rand(size_peak1) * total_time_ps
    mean_energy_peak1 = 4e6
    std_energy_peak1 = 0.1e6
    energy_values_peak1 = np.random.normal(
        mean_energy_peak1, std_energy_peak1, size_peak1
    )

    # make data
    nb_counts_peak2 = int(7e4)
    size_peak2 = nb_counts_peak2
    time_values_out_peak2 = np.random.rand(size_peak2) * total_time_ps
    mean_energy_peak2 = 14e6
    std_energy_peak2 = 1e6

    energy_values_peak2 = np.random.normal(
        mean_energy_peak2, std_energy_peak2, size_peak2
    )

    # import matplotlib.pyplot as plt

    # plt.hist(energy_values_peak1)
    # plt.hist(energy_values_peak2)
    # plt.show()

    filename1 = tmpdir.join(f"peak1.csv")
    np.savetxt(
        filename1,
        np.column_stack((time_values_out_peak1, energy_values_peak1)),
        delimiter=",",
    )
    filename2 = tmpdir.join(f"peak2.csv")
    np.savetxt(
        filename2,
        np.column_stack((time_values_out_peak2, energy_values_peak2)),
        delimiter=",",
    )

    # run
    bin_time = 20  # s
    processor = DataProcessor()
    processor.add_file(
        filename1, time_column=0, energy_column=1, delimiter=",", scale_time=True
    )
    processor.add_file(
        filename2, time_column=0, energy_column=1, delimiter=",", scale_time=True
    )

    count_rates_total, count_rate_bins_total = processor.get_count_rate(
        bin_time=bin_time
    )

    count_rates_peak1, count_rate_bins_peak1 = processor.get_count_rate(
        bin_time=bin_time,
        energy_window=(
            mean_energy_peak1 - std_energy_peak1 * 2,
            mean_energy_peak1 + std_energy_peak1 * 2,
        ),
    )

    count_rates_peak2, count_rate_bins_peak2 = processor.get_count_rate(
        bin_time=bin_time,
        energy_window=(
            mean_energy_peak2 - std_energy_peak2 * 2,
            mean_energy_peak2 + std_energy_peak2 * 2,
        ),
    )

    # test
    expected_count_rate_total = (nb_counts_peak1 + nb_counts_peak2) / total_time_s
    expected_count_rate_peak1 = nb_counts_peak1 / total_time_s
    expected_count_rate_peak2 = nb_counts_peak2 / total_time_s

    # check that the count rates are as expected
    assert np.allclose(count_rates_total, expected_count_rate_total, rtol=0.1)
    assert np.allclose(count_rates_peak1, expected_count_rate_peak1, rtol=0.1)
    assert np.allclose(count_rates_peak2, expected_count_rate_peak2, rtol=0.1)

    assert len(count_rate_bins_total) == total_time_s / bin_time


def test_count_rate_doesnt_ignore_data():
    """Test to catch the bug in #48"""

    # create a processor with time from -100 to 100 and energy values from -100 to 100
    processor = DataProcessor()
    processor.time_values = np.linspace(-100, 100, 1000)
    processor.energy_values = np.linspace(-100, 100, 1000)

    # calculate the count rates
    count_rates, time_bins = processor.get_count_rate(bin_time=1)

    # the first time bin should be -100
    assert np.isclose(time_bins[0], -100)
