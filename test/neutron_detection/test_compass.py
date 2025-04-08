import pytest
from libra_toolbox.neutron_detection.activation_foils import compass

@pytest.mark.parametrize("filename, expected_channel", [
    ("Data_CH14@V1725_292_Background_250322.CSV", 14),
    ("Data_CH7@V1725_123_Background_250322.CSV", 7),
    ("Data_CH21@V1725_456_Background_250322.CSV", 21),
])
def test_get_channel(filename, expected_channel):
    ch = compass.get_channel(filename)
    assert ch == expected_channel
   
