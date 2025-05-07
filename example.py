from libra_toolbox.neutron_detection.activation_foils.compass import Measurement


my_measurement = Measurement.from_directory(
    "250317_BABY_1L_run3/DAQ/Co60_0_872uCi_19Marc2014_250319_run3/UNFILTERED",
    name="test",
)

print(my_measurement.start_time)
print(my_measurement.stop_time)
