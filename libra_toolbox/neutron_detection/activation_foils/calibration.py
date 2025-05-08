import datetime

from libra_toolbox.neutron_detection.activation_foils import CheckSource

uCi_to_Bq = 3.7e4

check_source_ba133 = CheckSource(
    nuclide="Ba133",
    energy=[80.9979, 276.3989, 302.8508, 356.0129, 383.8485],
    intensity=[0.329, 0.0716, 0.1834, 0.6205, 0.0894],
    activity_date=datetime.date(2014, 3, 19),
    activity=1 * uCi_to_Bq,
)

check_source_co60 = CheckSource(
    nuclide="Co60",
    energy=[1173.228, 1332.492],
    intensity=[0.9985, 0.999826],
    activity_date=datetime.date(2014, 3, 19),
    activity=0.872 * uCi_to_Bq,
)
check_source_na22 = CheckSource(
    nuclide="Na22",
    energy=[511, 1274.537],
    intensity=[1.80, 0.9994],
    activity_date=datetime.date(2014, 3, 19),
    activity=5 * uCi_to_Bq,
)
check_source_cs137 = CheckSource(
    nuclide="Cs137",
    energy=[661.657],
    intensity=[0.851],
    activity_date=datetime.date(2014, 3, 19),
    activity=4.66 * uCi_to_Bq,
)
check_source_mn54 = CheckSource(
    nuclide="Mn54",
    energy=[834.848],
    intensity=[0.99976],
    activity_date=datetime.date(2016, 5, 2),
    activity=6.27 * uCi_to_Bq,
)
