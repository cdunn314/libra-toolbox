from datetime import datetime


class CheckSource:
    nuclide: str
    energy: list
    intensity: list
    activity_date: datetime.date
    activity: float

    def __init__(
        self,
        nuclide: str,
        energy: list,
        intensity: list,
        activity_date: datetime.date,
        activity: float,
    ):
        self.nuclide = nuclide
        self.energy = energy
        self.intensity = intensity
        self.activity_date = activity_date
        self.activity = activity


class ActivationFoil:
    nuclide: str
    mass: float


from . import explicit, settings, calculations
