"Implementation of the dielectrophoretic potential in Python"
import numpy as np
from fplanck import uniform_potential
from fplanck.utility import value_to_vector

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the functions for the dielectrophoretic potential.
def dep_potential_1D_on_negative():
    "Function to define the potential in the time intervals in which we turn it on and we consider negative potential."
    eps0 = 8.854e-12

    permittivity_medium = 80.2
    permittivity_particle = 2.1
    radius = 1e-8  # radius of the particle
    k = (permittivity_particle - permittivity_medium) / (permittivity_particle + 2 * permittivity_medium)  # Clausius-Mossotti factor
    viscosity_medium = 0.001
    G = 6 * np.pi * radius * viscosity_medium
    q = 3.2e-16 

    shift_value = 0.000000
    scale_value = 0.00000000000000000000001                # 1e-20     0.035
    max_potential = - 0.000000000000000001                  # +-100nm

    def potential1(*args):
        U = np.zeros_like(args[0])
        max_value = np.full(np.shape(args[0]), max_potential)
        for i, arg in enumerate(args):
            range1 = (arg >= -1e-7) & (arg <= -0.9e-7)
            U[range1] = -7.1e-20
            range2 = (arg >= 0.9e-7) & (arg <= 1e-7)
            U[range2] = -7.1e-20
            range3 = (arg >= -0.9e-7) & (arg <= 0.9e-7)
            U[range3] = - 1e-5 * arg[range3] ** 2 - 1e-20
            U[~(range1|range2|range3)] = scale_value * (shift_value - 2 * 5 * np.pi * eps0 * permittivity_medium * q/G *  radius ** 3 * 1 * arg[~(range1|range2|range3)] ** (-4))
        return U


    
    def potential2(*args):
        U = np.zeros_like(args[0])
        max_value = np.full(np.shape(args[0]), max_potential)
        for i, arg in enumerate(args):
            print(i)
            if np.allclose(arg, 0.0):
                U += max_value
            else:
                U += scale_value * (shift_value + np.where(arg == 0.0, max_value, np.maximum(max_value, - 2 * 5 * np.pi * eps0 * permittivity_medium *
                                                                 radius ** 3 *q/G * 1 * arg ** (-4))))
        return U
    
    

    return potential2


def dep_potential_1D_on_positive():
    "Function to define the potential in the time intervals in which we turn it on and we consider positive potential."
    eps0 = 8.854e-12

    permittivity_medium = 80.2
    permittivity_particle = 2.1
    radius = 1e-8  # radius of the particle
    k = (permittivity_particle - permittivity_medium) / (permittivity_particle + 2 * permittivity_medium)  # Clausius-Mossotti factor
   # Clausius-Mossotti factor
    viscosity_medium = 0.001
    G = 6 * np.pi * radius * viscosity_medium
    q = 3.2e-16

    shift_value = 0.0
    scale_value = 0.00000000000000000000001                # 1e-20     0.035
    max_potential = 0.000000000000000001                      # +-100nm

    def potential1(*args):
        U = np.zeros_like(args[0])
        max_value = np.full(np.shape(args[0]), max_potential)

        for i, arg in enumerate(args):
            print(i)
            if np.allclose(arg, 0.0):
                U += max_value
            else:
                U += scale_value*(shift_value + np.where(arg == 0.0, max_value, np.minimum (max_value, - 2 * 5 * np.pi * eps0 * permittivity_medium *
                                                                 radius ** 3 * q/G * k * arg ** (-4))))
        return U
    
    def potential2(*args):
        U = np.zeros_like(args[0])
        for i, arg in enumerate(args):
            range1 = (arg >= -1e-7) & (arg <= -0.9e-7)
            U[range1] = 7.1e-20
            range2 = (arg >= 0.9e-7) & (arg <= 1e-7)
            U[range2] = 7.1e-20
            range3 = (arg >= -0.9e-7) & (arg <= 0.9e-7)
            U[range3] = 1e-5 * arg[range3] ** 2 + 1e-20
            U[~(range1|range2|range3)] = scale_value * (shift_value + 2 * 5 * np.pi * eps0 * permittivity_medium * q/G *  radius ** 3 * 1 * arg[~(range1|range2|range3)] ** (-4))
        return U

    return potential2


def dep_potential_1D_off():
    "Function to define the potential quhne the dielectrophoretic potential is off."
    nm = 1e-9
    return uniform_potential(lambda x: bool(1), 0)