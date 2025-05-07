from libra_toolbox.neutron_detection.activation_foils.compass import Measurement


my_measurement = Measurement(name="check_source_from_last_friday", events=[], ch=2)


my_other_measurement = Measurement.from_directory("path_to_directory")