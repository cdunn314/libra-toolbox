from datetime import datetime
from . import calibration

from dataclasses import dataclass


@dataclass
class CheckSource:
    nuclide: calibration.Nuclide
    activity_date: datetime.date
    activity: float


class ActivationFoil:
    nuclide: calibration.Nuclide
    mass: float


from . import explicit, settings, calculations
