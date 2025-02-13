import numpy as np
from settings import ureg

class Foil:
    def __init__(self, mass: float, thickness: float, name=''):
        """
        Base class for neutron activation foils.
        :param mass: Mass of the foil in grams
        :param thickness: Thickness of the foil in cm
        """
        self.mass = mass
        self.thickness = thickness
        self.name = name
        self.half_life = None
        self.decay_constant = None
        self.reaction = None
        self.branching_ratio = None
        self.photon_energies = None
        self.cross_section = None  # in barns (1 barn = 1e-24 cm^2)
        self.density = None  # in g/cm^3
        self.atomic_mass = None
        self.abundance = None
        self.mass_attenuation_coefficient = None  # in cm^2/g
    
    def get_properties(self):
        """Returns the properties of the foil."""
        return {
            "Name": self.name,
            "Mass (g)": self.mass,
            "Thickness (cm)": self.thickness,
            "Decay Constant (1/s)": self.decay_constant,
            "Reaction": self.reaction,
            "Cross Section (barns)": self.cross_section,
            "Density (g/cm³)": self.density,
            "Mass Attenuation Coefficient (cm²/g)": self.mass_attenuation_coefficient,
        }
    
    def get_decay_constant(self):
        self.decay_constant = np.log(2) / self.half_life.to(ureg.s)

    def get_atoms(self):
        self.atoms = (self.abundance 
                      * (self.mass 
                         * (6.022e23 * ureg.particle / ureg.mol) 
                         / self.atomic_mass))

class Niobium(Foil):
    def __init__(self, mass: float, thickness: float, name=''):
        super().__init__(mass, thickness, name=name)
        self.half_life = np.array([10.15]) * ureg.day
        # half life used in BABY 100 mL analysis
        # self.half_life = [10.25 * ureg.day]
        self.get_decay_constant()

        self.reaction = ["Nb93(n,2n)Nb92m"]
        self.branching_ratio = np.array([0.9915])
        # branching ratio used in BABY 100 mL analysis
        # self.branching_ratio = [0.999]
        self.photon_energies = np.array([934.44])

        # Cross section from EAF-2010 at 14 MeV
        self.cross_section = np.array([0.4729]) * ureg.barn
        # cross section used in BABY 100 mL analysis
        # self.cross_section = 0.46 * ureg.barn

        self.density = 8.582 * ureg.g / ureg.cm**3 
        self.atomic_mass = 92.90638 * ureg.g/ureg.mol
        # Natural abundance of reactant in analyzed foil reaction
        self.abundance = np.array([1.00])
        self.get_atoms()

        # Should update mass attenuation coefficient to be photon energy-dependent
        self.mass_attenuation_coefficient = np.array([0.06120]) * ureg.cm**2 / ureg.g   # at 1 MeV

class Zirconium(Foil):
    def __init__(self, mass: float, thickness: float, name=''):
        super().__init__(mass, thickness, name=name)
        self.half_life = np.array([78.41,
                                   78.41]) * ureg.hour
        self.get_decay_constant()

        self.reaction = ["Zr90(n,2n)Zr89", 
                         "Zr90(n,2n)Zr89"]
        self.branching_ratio = np.array([0.9904, 
                                         0.45])
        self.photon_energies = np.array([909.15, 
                                         511]) * ureg.keV

        # Cross sections from EAF-2010 at 14 MeV
        self.cross_section = np.array([0.5068,
                                       0.5068]) * ureg.barn
                              
        self.density = 6.505 * ureg.g / ureg.cm**3 
        self.atomic_mass = 91.222 * ureg.g / ureg.mol
        # Natural abundance of reactant in analyzed foil reaction
        self.abundance = np.array([0.5145,
                                   0.5145])
        self.get_atoms()

        #From NIST Xray Mass Attenuation Coefficients Table 3
        self.mass_attenuation_coefficient = np.array([0.08590, 
                                                      0.06156]) * ureg.cm**2 / ureg.g  # at 1 MeV

