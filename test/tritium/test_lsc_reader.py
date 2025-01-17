from libra_toolbox.tritium.lsc_measurements import LSCFileReader

from pathlib import Path


def test_lsc_reader():
    filename = Path(__file__).parent / "TEST_CSV.csv"

    csv_reader = LSCFileReader(filename)
    csv_reader.vial_labels = [
        "1-1-1",
        "1-1-2",
        "1-1-3",
        "1-1-4",
        None,
        "1-2-1",
        "1-2-2",
        "1-2-3",
        "1-2-4",
        None,
        "1-3-1",
        "1-3-2",
        "1-3-3",
        "1-3-4",
    ]
    csv_reader.read_file()
    bq1_values = csv_reader.get_bq1_values()
    print(bq1_values)
    bq1_values_with_labels = csv_reader.get_bq1_values_with_labels()
    print(bq1_values_with_labels)

    assert len(bq1_values) == 14
    assert len(bq1_values_with_labels) == 13


def test_lsc_reader_label_automated():
    """
    Test LSCFileReader with labels_column provided
    """
    filename = Path(__file__).parent / "test_lsc_file_with_labels.csv"

    csv_reader = LSCFileReader(filename, labels_column="SMPL_ID")
    csv_reader.read_file()

    bq1_values = csv_reader.get_bq1_values()
    bq1_values_with_labels = csv_reader.get_bq1_values_with_labels()

    assert len(bq1_values) == 10
    assert len(bq1_values_with_labels) == 10


def test_read_quench_set():
    """Tests reading the quench set from the LSC file"""
    filename = Path(__file__).parent / "test_lsc_file_with_labels.csv"

    csv_reader = LSCFileReader(filename, labels_column="SMPL_ID")
    csv_reader.read_file()

    assert csv_reader.quench_set == "Low Energy: 3H-UG"
