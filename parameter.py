import numpy as np
# Parameters (All SI units)
# =============================================================================
# General
kb = 1.38e-23                   # Boltzmann constant
T = 300                         # Temperature (K)

# Particle
ro = 1e-8                       # Radius particle (m)
rho = 1000                      # Density particle (kg/m^3)
mo = rho*(4/3)*np.pi*ro**3      # mass of a polystyrene bead in kg. (Np,) array

# Actuation
#f = 500                        # Actuation frequency of electric field

# Diffusion Tensor
eta = 8.9e-4                    # Dynamic viscosity of water (PaS)
gamma0 = 6*np.pi*eta*ro         # friction coefficient (Np,) array
D0 = kb*T/gamma0                # Free space diffusion coefficient. (Np,) array
D = np.array([D0, D0, D0])

# Time
# num_steps = 1000                # Number of time steps
# s = 0.05                       # Seconds of simulation
