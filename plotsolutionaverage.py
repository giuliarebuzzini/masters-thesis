'Plot of the solution obtained averaging over the noise'

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------
# Parameters
x0 = 4e-8  # Initial value of x (m)
epsilon0 = 8.854e-12  # Permittivity of free space (F/m)
epsilon1 = 2.1  # Particle permittivity
k = 1.0  # Clausius-Mossotti factor
q = 3.2e-16  # Charge (C)
r = 1e-8  # Radius of the particle (m)
Vp = r**3  # Volume of the particle (m^3)
eta = 8.9e-4  # Dynamic viscosity (Pa·s)
gamma = 6 * np.pi * eta * r  # Drag coefficient
f = 500  # Frequency (Hz)
omega_A = 2 * np.pi * f  # Angular frequency (rad/s)

t = np.linspace(0, 1e-3, 1000)  


#------------------------------------------------------------------------------------------------------------------
# Definition of the function
def x(t, sign=1):
    term = (3/2)*t + (2/omega_A) - (np.sin(2*omega_A*t)/(4*omega_A)) - (2*np.cos(omega_A*t)/omega_A)
    factor = (2 * np.pi * epsilon0 * epsilon1 * k * Vp * q) / gamma
    inside_root = x0**6 - factor * term
    if inside_root < 0:
        return np.nan  
    return sign * np.cbrt(inside_root)

# Calculate x(t) for both positive and negative solution
x_pos = np.array([x(ti, sign=1) for ti in t])
x_neg = np.array([x(ti, sign=-1) for ti in t])

#-------------------------------------------------------------------------------------------------------------
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, x_pos, label='x(t) Positive SOlution')
plt.plot(t, x_neg, label='x(t) Negative SOlution')
plt.axhline(y=0, color='grey', linestyle='--', label='Trap position')
plt.xlabel('Time (s)')
plt.ylabel('x(t) (m)')
plt.title('Positive and negative trajectories of the particle')
plt.legend()
plt.grid(True)
plt.show()
