from libra_toolbox.neutron_detection.activation_foils import calibration


def test_get_decay_lines():
    decay_lines = calibration.get_decay_lines(['Na22', 'Cs137'])
    assert isinstance(decay_lines, dict)
    assert 'energy' in decay_lines['Na22'].keys()


