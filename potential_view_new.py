import numpy as np
from fplanck.utility import value_to_vector
from fplanck import (fokker_planck, boundary, gaussian_pdf, harmonic_potential, gaussian_potential, uniform_potential,
                     potential_from_data)
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from dep_potential_new import dep_potential_1D_on_positive, dep_potential_1D_on_negative

mpl.use('TkAgg')
# plt.ion()

x = np.linspace(-300e-9, 300e-9, 101)
y = np.linspace(-300e-9, 300e-9, 101)
X, Y = np.meshgrid(x, y)

U1 = dep_potential_1D_on_negative()
U2 = dep_potential_1D_on_positive()

potential_values1 = U1(x)
potential_values2 = U2(x)
#print (potential_values)

nm = 1e-9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(x / nm, potential_values1, color='k', alpha=.5)
ax1.set_title("Dielectrophoretic potential")
ax1.set_xlabel("Position (nm)")
ax1.set_ylabel("Potential (JC)")
ax2.plot(x / nm, potential_values2, color='k', alpha=.5)
ax2.set_title("Positive Potential")
ax2.set_xlabel("Position (nm)")
ax2.set_ylabel("Potential (JC)")
plt.show()