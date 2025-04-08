from libra_toolbox.neutron_detection.activation_foils import compass

def test_get_channel():
    ch = compass.get_channel("Data_CH14@V1725_292_Background_250322.CSV")
    assert ch==14

