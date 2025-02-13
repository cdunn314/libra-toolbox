from settings import *
import foils
from .calculations import n93_number, delay_time
import numpy as np
import pandas as pd
import os


# def get_foil_data(foil: dict, 
#                   filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nuclide_data.xlsx'),
#                   xslib='EAF2010'):
    
#     # Read in nuclide data
#     df = pd.read_excel(filepath, skiprows=1)
    
#     # Only get info for one nuclide
#     mask = df['Nuclide'] == foil['nuclide']

#     # Calculate number of nuclide atoms in foil
#     num_element = (foil['mass'] 
#                    / (df['Element_Atomic_Mass'][mask].item() * ureg.g / ureg.mol)
#                    * (6.022e23 * ureg.particle / ureg.mol)
#     )
    
#     foil['number'] = num_element * df['Natural_Abundance'][mask].item()

#     # Get density and mass attenuation coefficient
#     foil['density'] = df['Density'][mask].item() * ureg.g / ureg.cm**3
#     foil['mass_attenuation_coefficient'] = df['Mass_Attenuation_Coefficient'][mask].item() * (ureg.cm**2/(ureg.g))

#     # Get cross section for available reactions
#     for reaction in foil['reactions'].keys():
#         heading = reaction + '_' + xslib + '_xs'
#         if heading in df.keys():
#             foil['reactions'][reaction]['cross_section'] = df[heading][mask].item() * ureg.barn

#     return foil



def get_chain(irradiations, decay_constant=Nb92m_decay_constant):
    """
    Returns the value of
    (1 - exp(-\lambda * \Delta t_1)) * (1 - exp(-\lambda * \Delta t_2)) * ... * (1 - exp(-\lambda * \Delta t_n))
    where \Delta t_i is the time between the end of the i-th period (rest or irradiation) and the start of the next one

    Args:
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        float or pint.Quantity: the value of the chain
    """
    result = 1
    periods = [{"start": irradiations[0]["t_on"], "end": irradiations[0]["t_off"]}]
    for irr in irradiations[1:]:
        periods.append({"start": periods[-1]["end"], "end": irr["t_on"]})
        periods.append({"start": irr["t_on"], "end": irr["t_off"]})

    for period in periods:
        delta_t = period["end"] - period["start"]
        result = 1 - result * np.exp(-decay_constant * delta_t)
    return result

def get_efficiency(energies=None, coeff=None,
                   coeff_energy_bounds=None,
                   coeff_type='total', geometric_eff=1.0):
    """Calculates the total efficiency of a gamma detector
    based on provided coefficients. These coefficients could
    be for calculating the total efficiency (coeff_type='total')
    or for calculating the intrinsic efficiency (coeff_type='intrinsic')"""

    if coeff is None:
        # These are the efficiencies reported for activation foil analysis
        # of the BABY 100 mL runs
        geometric_efficiency = 0.5
        intrinsic_efficiency = 0.344917296922981
        total_efficiency = geometric_eff * intrinsic_efficiency
    else:
        # Check that efficiency is being interpolated between the bounds of the fit curve
        if energies.min() < coeff_energy_bounds.min() or energies.max() > coeff_energy_bounds.max():
            raise Warning('Efficiency is being extrapolated according to efficiency fit curve bounds.')
        if coeff_type.lower() is 'total':
            total_efficiency = np.polyval(coeff, energies)
        elif coeff_type.lower() is 'intrinsic':
            intrinsic_efficiency = np.polyval(coeff, energies)
            total_efficiency = geometric_eff * intrinsic_efficiency
    total_efficiency *= ureg.count / ureg.particle
    return total_efficiency


def get_neutron_flux(experiment: dict, irradiations: list, foil: foils.Foil):
    """calculates the neutron flux during the irradiation
    Based on Equation 1 from:
    Lee, Dongwon, et al. "Determination of the Deuterium-Tritium (D-T) Generator 
    Neutron Flux using Multi-foil Neutron Activation Analysis Method." , 
    May. 2019. https://doi.org/10.2172/1524045

    Args:
        experiment (dict): dictionary containing the experiment data
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        pint.Quantity: neutron flux
    """


    # time_between_generator_off_and_start_of_counting = delay_time(
    #     experiment["time_generator_off"], experiment["start_time_counting"]
    # )
    time_between_generator_off_and_start_of_counting = (
        experiment["start_time_counting"] - experiment["time_generator_off"]
    ).seconds * ureg.second

    if 'total_eff_coeff' in experiment.keys():
        total_efficiency = get_efficiency(experiment['total_eff_coeff'], foil.photon_energies,
                                          coeff_energy_bounds=experiment['efficiency_bounds'],
                                          coeff_type='total')
    elif 'intrinsic_eff_coeff' in experiment.keys():
        total_efficiency = get_efficiency(experiment['intrinsic_eff_coeff'], foil.photon_energies,
                                          coeff_energy_bounds=experiment['efficiency_bounds'],
                                          coeff_type='intrinsic')
    else:
        total_efficiency = get_efficiency()

    #Spectroscopic Factor to account for the branching ratio and the
    # total detection efficiency

    f_spec = total_efficiency * foil.branching_ratio

    print('total efficiency', total_efficiency)
    number_of_decays_measured = experiment["photon_counts"] / f_spec
    print('number of decays measured', number_of_decays_measured)
    print('number', foil.atoms)
    print('cross section', foil.cross_section)


    flux = (
        number_of_decays_measured
        / foil.atoms
        / foil.cross_section
    )

    f_time = (get_chain(irradiations, foil.decay_constant)
                * np.exp( -foil.decay_constant
                     * time_between_generator_off_and_start_of_counting) 
                * (1 - np.exp( -foil.decay_constant * experiment['real_count_time']))
                * (experiment['live_count_time'] / experiment['real_count_time'])
                / foil.decay_constant
    )

    print('att coeff', foil['mass_attenuation_coefficient'])
    print('density', foil['density'])
    print('thickness', foil['thickness'])

    # Correction factor of gamma-ray self-attenuation in the foil
    f_self = ( (1 - 
                    np.exp(-foil.mass_attenuation_coefficient
                           * foil.density
                           * foil.thickness))
                / (foil.mass_attenuation_coefficient
                   * foil.density
                   * foil.thickness)
    ).to('dimensionless')

    print('flux: ', flux.to_reduced_units())
    print('f_time', f_time)
    print('f_self', f_self)

    flux /= (f_time * f_self)
    


    # convert n/cm2/s to n/s
    area_of_sphere = 4 * np.pi * experiment["distance_from_center_of_target_plane"] ** 2

    flux *= area_of_sphere

    return flux


def get_neutron_flux_error(experiment: dict):
    """
    Returns the uncertainty of the neutron flux as a pint.Quantity

    Args:
        experiment (dict): dictionary containing the experiment data
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        pint.Quantity: uncertainty of the neutron flux
    """ 
    error_counts = experiment["photon_counts_uncertainty"] / experiment["photon_counts"]
    error_mass = 0.0001 * ureg.g / experiment["foil_mass"]
    error_geometric_eff = 0.025 / geometric_efficiency
    error_intrinsic_eff = 0.025 / nal_gamma_efficiency

    error = np.sqrt(
        error_counts**2
        + error_mass**2
        + error_geometric_eff**2
        + error_intrinsic_eff**2
    )
    return error.to(ureg.dimensionless).magnitude


if __name__ == "__main__":
    pass
