from libra_toolbox.tritium.lsc_measurements import (
    LSCSample,
    LIBRASample,
    GasStream,
    LSCFileReader,
)
from libra_toolbox.tritium.model import ureg

from pathlib import Path
import pytest


def test_lscsample_init():
    activity = 1.0 * ureg.Bq
    name = "Sample1"
    sample = LSCSample(activity, name)
    assert sample.activity == activity
    assert sample.name == name
    assert not sample.background_substracted


def test_lscsample_str():
    sample = LSCSample(1.0 * ureg.Bq, "Sample1")
    assert str(sample) == "Sample Sample1"


def test_lscsample_substract_background():
    sample = LSCSample(1.0 * ureg.Bq, "Sample1")
    background_sample = LSCSample(0.5 * ureg.Bq, "Background")
    sample.substract_background(background_sample)
    assert sample.activity == 0.5 * ureg.Bq
    assert sample.background_substracted

    with pytest.raises(ValueError, match="Background already substracted"):
        sample.substract_background(background_sample)


def test_lscsample_substract_background_when_negative():
    sample = LSCSample(0.5 * ureg.Bq, "Sample1")
    background_sample = LSCSample(1.0 * ureg.Bq, "Background")
    with pytest.warns(
        UserWarning,
        match=f"Activity of {sample.name} is negative after substracting background. Setting to zero.",
    ):
        sample.substract_background(background_sample)
    assert sample.activity == 0 * ureg.Bq
    assert sample.background_substracted


def test_lscsample_from_file():
    file_reader = LSCFileReader(
        Path(__file__).parent / "TEST_CSV.csv",
        vial_labels=[
            "1L-OV-1-0-1",
            "1L-OV-1-0-2",
            "1L-OV-1-0-3",
            "1L-OV-1-0-4",
            None,
            "1L-IV-1-0-1",
            "1L-IV-1-0-2",
            "1L-IV-1-0-3",
            "1L-IV-1-0-4",
            None,
            "Sample1",
            "1L-IV-1-1-2",
            "1L-IV-1-1-3",
            "1L-IV-1-1-4",
        ],
    )

    file_reader.read_file()
    sample = LSCSample.from_file(file_reader, "Sample1")
    assert sample.activity == 0.334 * ureg.Bq
    assert sample.name == "Sample1"


def test_lscsample_from_file_when_not_found():
    file_reader = LSCFileReader(
        Path(__file__).parent / "TEST_CSV.csv",
        vial_labels=[
            "1L-OV-1-0-1",
            "1L-OV-1-0-2",
            "1L-OV-1-0-3",
            "1L-OV-1-0-4",
            None,
            "1L-IV-1-0-1",
            "1L-IV-1-0-2",
            "1L-IV-1-0-3",
            "1L-IV-1-0-4",
            None,
            "Sample1",
            "1L-IV-1-1-2",
            "1L-IV-1-1-3",
            "1L-IV-1-1-4",
        ],
    )

    file_reader.read_file()
    with pytest.raises(
        ValueError, match="Vial coucoucou not found in the file reader."
    ):
        LSCSample.from_file(file_reader, "coucoucou")


def test_lsc_sample_has_origin_file():
    """Test that the origin file is set when creating a LSCSample from a LSCFileReader"""
    filename = Path(__file__).parent / "test_lsc_file_with_labels.csv"

    csv_reader = LSCFileReader(filename, labels_column="SMPL_ID")
    csv_reader.read_file()

    sample = LSCSample.from_file(csv_reader, "1L-IV_2-0-4")

    assert sample.origin_file == filename
