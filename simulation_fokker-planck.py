'Simulation of the Fokker-Planck equation in 1 dimension.'
# Import useful libraries and files from other codes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from fplanck import fokker_planck, boundary, gaussian_pdf, uniform_pdf
from dep_potential_new import (dep_potential_1D_off,
                           dep_potential_1D_on_positive, dep_potential_1D_on_negative)

mpl.use('TkAgg')

#---------------------------------------------------------------------------------------------------------
# Parameters and constants for the simulation
# Constants
ms = 1e-3
nm = 1e-9

# Global parameters for Fokker Planck solution
VISCOSITY = 8e-4
RADIUS = 50 * nm
DRAG = 6 * np.pi * VISCOSITY * RADIUS
TEMPERATURE = 300
EXTENT = 1000 * nm
RESOLUTION = 5 * nm

# Simulation parameters
dt_resolution = 0.01 * ms  # dt interval at simulations
init_duration = 1 * ms  # Initialisation phase duration (s)
trap_off_duration = 0.5 * ms  # potential OFF phase duration (s)
trap_on_duration = 0.1 * ms  # potential ON phase duration (s)
trap_off_min_duration = 0.01 * ms
trap_off_max_duration = 1 * ms
trap_on_min_duration = 0.01 * ms
trap_on_max_duration = 1 * ms

#-----------------------------------------------------------------------------------------------------------
# Creating two simulation objects with for different potential states, for the sates ON and OFF.
'''
# Get Fokker-Plank solution for dep_on and dep_off potentials
sim_dep_on = fokker_planck(temperature=TEMPERATURE, drag=DRAG, extent=EXTENT,
                           resolution=RESOLUTION, boundary=boundary.reflecting,
                           potential=dep_potential_1D_on_harmonic((0,0), 1e-6))
'''

sim_dep_on = fokker_planck(temperature=TEMPERATURE, drag=DRAG, extent=EXTENT,
                           resolution=RESOLUTION, boundary=boundary.reflecting,
                           potential=dep_potential_1D_on_negative())

sim_dep_off = fokker_planck(temperature=TEMPERATURE, drag=DRAG, extent=EXTENT,
                            resolution=RESOLUTION, boundary=boundary.reflecting,
                            potential=dep_potential_1D_off())

# Initial state
dep_off_center_index = np.argmin(np.abs(sim_dep_off.grid[0]))
dep_on_center_index = np.argmin(np.abs(sim_dep_on.grid[0]))
print("off_center, on_center", dep_off_center_index, dep_on_center_index )

steady_stabilisation = sim_dep_on.steady_state()

t_stabilisation, Pt_stabilisation = sim_dep_on.propagate_interval(
    initial=gaussian_pdf(center=(0.0*nm), width=20 * nm),
    tf=init_duration, dt=dt_resolution)

'''
t_stabilisation, Pt_stabilisation = sim_dep_on.propagate_interval(
    initial=uniform_pdf(lambda x: (x > -50*nm) & (x < 50*nm)),
    tf=init_duration, dt=dt_resolution)
'''

steady_dep_off = sim_dep_off.steady_state()
steady_dep_on = sim_dep_on.steady_state()

global t_dep_off, Pt_dep_off, t_dep_on, Pt_dep_on

#---------------------------------------------------------------------------------------------------
# Function to update the simulation, the state goes through ON and OFF phases
def update_simulation_step(tf_off, tf_on, initial_off):
    global t_dep_off, Pt_dep_off, t_dep_on, Pt_dep_on

    # dep_off phase
    t_dep_off, Pt_dep_off = sim_dep_off.propagate_interval(
        initial=lambda x: initial_off,
        tf=tf_off, dt=dt_resolution)

    # dep_on phase
    t_dep_on, Pt_dep_on = sim_dep_on.propagate_interval(
        initial=lambda x: Pt_dep_off[-1],
        tf=tf_on, dt=dt_resolution)

    print("off on ", Pt_dep_off[-1, dep_off_center_index], Pt_dep_on[-1, dep_on_center_index])


update_simulation_step(trap_off_duration, trap_on_duration, Pt_stabilisation[-1])

#--------------------------------------------------------------------------------------------------------------
# Plotting the animation
# animation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

def update_plot(last_dep_on_interval):
    global line2, line3

    ax2.clear()
    ax3.clear()

    # Steady state
    ax2.plot(sim_dep_off.grid[0] / nm, steady_dep_off, color='k', ls='--', alpha=.5)
    ax3.plot(sim_dep_on.grid[0] / nm, steady_dep_on, color='k', ls='--', alpha=.5)

    # Initial PDF
    ax2.plot(sim_dep_off.grid[0] / nm, Pt_dep_off[0], color='red', ls='--', alpha=.3)
    ax3.plot(sim_dep_on.grid[0] / nm, Pt_dep_on[0], color='red', ls='--', alpha=.3)

    # interval PDF
    line2, = ax2.plot(sim_dep_off.grid[0] / nm, Pt_dep_off[0], lw=2, color='C3')
    line3, = ax3.plot(sim_dep_on.grid[0] / nm, last_dep_on_interval, lw=2, color='C3')

    # Axis
    ax2.set(xlabel='x (nm)', ylabel='normalized PDF')
    ax3.set(xlabel='x (nm)', ylabel='normalized PDF')

    # Header
    ax2.set_title(f'OFF\nTime: {t_dep_off[-1]:.2e} s')
    ax3.set_title(f'ON\nTime: {t_dep_on[-1]:.2e} s')


# Steady state
ax1.plot(sim_dep_on.grid[0] / nm, steady_stabilisation, color='k', ls='--', alpha=.5)
# Initial PDF
ax1.plot(sim_dep_on.grid[0] / nm, Pt_stabilisation[0], color='red', ls='--', alpha=.3)
# interval PDF
line1, = ax1.plot(sim_dep_on.grid[0] / nm, Pt_stabilisation[0], lw=2, color='C3')
# Axis
ax1.set(xlabel='x (nm)', ylabel='normalized PDF')
ax1.set_title(f'ON\nTime: {t_stabilisation[-1]:.2e} s')


global line2, line3
update_plot(Pt_dep_on[-1])

for ax in (ax1, ax2, ax3):
    ax.set(xlabel='x (nm)', ylabel='normalized PDF')
    ax.margins(x=0)


def update_stabilisation(i):
    if i < len(t_stabilisation):
        line1.set_ydata(Pt_stabilisation[i])
        ax1.set_title(f'ON\nTime: {t_stabilisation[i]:.2e} s')
        if i == len(t_stabilisation) - 1:
            start_dep_off()
        return [line1]


def update_dep_off(i):
    if i < len(t_dep_off):
        line2.set_ydata(Pt_dep_off[i])
        ax2.set_title(f'OFF\nTime: {t_dep_off[i]:.2e} s')
        if i == len(t_dep_off) - 1:
            start_dep_on()
        return [line2]


def update_dep_on(i):
    if i < len(t_dep_on):
        line3.set_ydata(Pt_dep_on[i])
        ax3.set_title(f'ON\nTime: {t_dep_on[i]:.2e} s')
        if i == len(t_dep_on) - 1:
            iterate_simulation()
        return [line3]


def start_dep_off():
    global anim
    anim = FuncAnimation(fig, update_dep_off, frames=len(t_dep_off), interval=30, repeat=False)


def start_dep_on():
    global anim
    anim = FuncAnimation(fig, update_dep_on, frames=len(t_dep_on), interval=30, repeat=False)


def iterate_simulation():
    global anim, trap_off_duration, trap_on_duration

    # store last dep_on interval value
    last_dep_on_interval = Pt_dep_on[-1]
    trap_off_duration = slider_trap_off.val
    trap_on_duration = slider_trap_on.val
    update_simulation_step(trap_off_duration, trap_on_duration, Pt_dep_on[-1])
    update_plot(last_dep_on_interval)
    start_dep_off()


def update_all(i):
    if i < len(t_stabilisation):
        return update_stabilisation(i)
    elif i < len(t_stabilisation) + len(t_dep_off):
        return update_dep_off(i - len(t_stabilisation))
    else:
        return update_dep_on(i - len(t_stabilisation) - len(t_dep_off))


plt.subplots_adjust(bottom=.25)

ax_slider_trap_off = plt.axes((0.25, 0.1, 0.65, 0.03), facecolor='lightgoldenrodyellow')
slider_trap_off = Slider(ax_slider_trap_off, 'trap_off_duration', trap_off_min_duration, trap_off_max_duration,
                    valinit=trap_off_duration, valstep=1e-5)

ax_slider_trap_on = plt.axes((0.25, 0.05, 0.65, 0.03), facecolor='lightgoldenrodyellow')
slider_trap_on = Slider(ax_slider_trap_on, 'trap_on_duration', trap_off_min_duration, trap_off_max_duration,
                    valinit=trap_on_duration, valstep=1e-5)

anim = FuncAnimation(fig, update_stabilisation, frames=len(t_stabilisation), interval=30, repeat=False)



# plt.tight_layout()
plt.show()
