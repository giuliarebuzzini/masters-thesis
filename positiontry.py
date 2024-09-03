'Simulation of the Langevin equation for a particle subjected to Brownian Motion and dielectrophoretic force'

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from parameter import * # Library created to store all the parameters
import matplotlib.animation as animation
from plotter import * # Library created for plotting

#------------------------------------------------------------------------------------------------------------------------
# Initialization of time and variables
tt = 0.0001                                           # Final time of the simulation
dt = 0.000000005                                      # Time step 
                                   
N = int(tt/dt)                                       # Number of time steps of the simulation
t_star = N
t = np.linspace(0,tt, N) 
delta = t[2]-t[1] 
f = 500

B = np.random.normal(0, 1, (3, N))                  # Random noise term of dimension 3xN
x = np.zeros((3, N))                                # Main position vector
x2 = np.zeros((3, N))                               # Position vector to store force turned off
x[:, 0] = [4e-8, 4e-8, 4e-8]                        # Initial position
x_e = np.zeros(3)                                   # Trap position                        

trapped = np.zeros(N)                               # Storage array for trapping times
S = 1 + np.sin(2 * np.pi * f * t)                   # Dynamic traping field
distance = np.zeros(N)                              # N dimensional array to store the distance between a position and the previous one
distance2 = np.zeros(N)                             # Same as before but for the array x2
trap = 0

value_force = np.zeros(N)
value_force2 = np.zeros(N)
value_bm = np.zeros(N)


#------------------------------------------------------------------------------------------------------------------------
# Definition of useful functions
def distance_function(v1, v2):
    """This function serves the purpose of calculating the distance
        between two vectors and returns the normalized vector and distance.
        It avoids division by zero, and returning a zero vector in that case. 
        Parameters:
        v1 (array-like): first vector.
        v2 (array-like): second vector.
    
        Returns:
        tuple: containing the normalized vector and the distance between the vectors.
    """
    
    # Calculate the vector from p1 to p2
    vector = np.array(v2) - np.array(v1)
    normed_distance = np.linalg.norm(vector)

    # Check if the distance is not zero to avoid division by zero
    if normed_distance != 0:
        normalized_vector = vector / normed_distance # Vector normalization
        return normalized_vector, normed_distance
    else:
        return np.array([0, 0, 0]), normed_distance  # Returns zero vector if the distance is zero

def F_dep(radius, permittivity_particle, permittivity_medium, electric_field_gradient):
    """This function computes the dielectrophoretic force 
    without the addition of the electric field gradient.
    Parameters:
    radius (float): radius of the particle.
    permittivity_particle (float): permittivity of the particle.
    permittivity_medium (float): permittivity of the medium.
    electric_field_gradient (float): electric field gradient.
    
    Returns:
    float: dielectrophoretic force.
    """

    eps0 = 8.854e-12  # Permittivity of free space (F/m)

    # Clausius-Mossotti factor
    K = 1 # (permittivity_particle - permittivity_medium) / (permittivity_particle + 2 * permittivity_medium)

    F = (2 * np.pi * eps0 * permittivity_medium * radius ** 3 * K * electric_field_gradient)

    return F # Returns the dielectrophoretic force

def Egrad(distance):
    """ This function returns the electric field gradient.
    Parameters:
    distance (float): distance from the trap center.
    
    Returns:
    float: electric field gradient. 
    """
    return 3.2e-16/(distance **5)

def Diffusion_tensor(r, r0):
    """ This function computes the diffusion tensor for the particle.
    Parameters:
    r (array-like): current position of the particle.
    r0 (float): reference position.
    
    Returns:
    array-like: diffusion tensor.
      """

    D_tensor = np.zeros(3) # Initialization of the diffusion tensor

    z = r[2, :] 
    ro_z = r0 / z

    # Defining the components of the tensor according to the theory
    D_tensor[0, :] = D0 
    D_tensor[1, :] = D0 
    D_tensor[2, :] = D0 

    return D_tensor

def collision_detection(x_now, x_next, ro):
    """This function detects the collision of the particle against the bounday of the trap.
    Parameters:
    x_now (array-like): current position of the particle.
    x_next (array-like): next position of the particle.
    ro (float): radius of the trap.
    
    Returns:
    tuple: tuple containing the next position and the distance to the trap."""

    forceVector, distance = distance_function(x_next, np.array([0, 0, 0]))
    
    #radius = 3e-8 # Radius of the trap
    trap = 0
    if distance <= ro+3e-8:
        # Collision Detected
        x_next = x_now # If there is a collision between the wall and the particle, then the position is the previous one
        trap = 1
    return x_next, trap
    
#--------------------------------------------------------------------------------------------------------------------------
# Simulation running for every time step
# Main Simulation Loop
factor1 = np.sqrt(2 * D) * np.sqrt(delta)
factor2 = D * delta / (kb * T)

firstTime = 0
t_electrode = 0.00

for m in range(N - 1):
   
    forceVector, distance[m] = distance_function(x[:, m], x_e)
    distance2[m] = distance_function(x[:,0], x_e)[1]
    E = S[m] ** 2 * Egrad(distance[m])  # Electric field
    fdep = F_dep(ro, 2.1, 80.2, E) * forceVector  # Dielectrophoretic force
    E2 = S[m] ** 2 * Egrad(distance2[m])  # Electric field
    fdep2 = F_dep(ro, 2.1, 80.2, E2) * forceVector

    if distance[m] < ro+2e-8:
        trapped[m] = 1
        if firstTime == 0:
            t_electrode = m * dt
            firstTime = 1


    # Update of the position in time using the Euler-Maruyama method
    if m < t_star - 1:
            x[:, m + 1] = x[:, m] + (factor1 * B[:, m] + factor2 * fdep)  # Brownian Motion and force action
    else:
        x[:, m + 1] = x[:, m] + (factor1 * B[:,m]) # + fct_2 * fdep) # Euler-Maruyama method
    x[:, m + 1], trapped[m] = collision_detection(x[:, m], x[:, m + 1], ro)  # Activation of the detector of collision

    value_force[m] = np.linalg.norm(factor2 * fdep)
    value_force2[m] = np.linalg.norm(factor2 * fdep2)
    value_bm[m] = np.linalg.norm(factor1 * B[:, m])

distance[N - 1] = distance[N - 2]  # Not jump in the last time step to zero

#------------------------------------------------------------------------------------------------------------------------
# Plotting of the simulation and the displacement for time step
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialization
X, Y, Z = x[0, :], x[1, :], x[2, :]

# Plotting the simulation
line, = ax.plot(X[0:t_star-1], Y[0:t_star-1], Z[0:t_star-1], 'tomato')
line, = ax.plot(X[t_star:N-1], Y[t_star:N-1], Z[t_star:N-1], 'teal') # To emphasize the switch off of the force

# Drawing the starting point and trap point
ax.scatter(x_e[0], x_e[1], x_e[2], marker='*', label='Trap position', color='blue')
ax.scatter(x[0, 0], x[1, 0], x[2, 0], marker='*', label='Initial position', color='green')

plt.title('Action of the dielectrophoretic force \n with Brownian Motion', fontsize=18)
ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')
ax.legend()
plt.draw()

# Trap boundaries 
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = x_e[0] + 3e-8 * np.outer(np.cos(u), np.sin(v))
y = x_e[1] + 3e-8 * np.outer(np.sin(u), np.sin(v))
z = x_e[2] + 3e-8 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z, color='silver', alpha=0.5)

# Plot the displacement for every time step
fig, ax = plt.subplots()
ax.plot(t, distance, color='royalblue')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (m)')
plt.title('Distance from the trap', fontsize= 20)

# Find the first time step when the particle is trapped
print(trapped)

trapping_index = np.argmax(trapped)

# Extract the distances from the trapping time to the end of the simulation
distances_after_trapping = distance[trapping_index:]

# Compute mean and standard deviation of these distances
mean_distance = np.mean(distances_after_trapping)
std_distance = np.std(distances_after_trapping)

# Determine a characteristic trapping radius
characteristic_radius = mean_distance + std_distance

print(f"First trapping time step: {trapping_index}")
print(f"Mean distance after trapping: {mean_distance:.2e} meters")
print(f"Standard deviation of distance after trapping: {std_distance:.2e} meters")
print(f"Characteristic trapping radius: {characteristic_radius:.2e} meters")

fig, ax = plt.subplots()
ax.plot(t[trapping_index:], distances_after_trapping, color='royalblue')
ax.axhline(y=characteristic_radius, color='red', linestyle='--', label='Characteristic Trapping Radius')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (m)')
plt.title('Distance from the trap (after trapping)', fontsize=16)
plt.legend()

# Plot histogram of the mean displacement
fig, ax = plt.subplots()
ax.hist(distances_after_trapping, bins=30, color='royalblue', edgecolor='black')
ax.axvline(mean_distance, color='red', linestyle='--', label='Mean Distance')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Frequency')
plt.title('Histograms of the distances from the trap (after trapping)', fontsize=15)
plt.legend()

t = t[:-1]

# Specify the number of last time steps to plot
num_last_steps = 1000

# Slice the data to get the last num_last_steps points
t_last = t[0: num_last_steps]
value_bm_last = value_bm[0: num_last_steps]
value_force_last = value_force[0: num_last_steps]

value_bm = value_bm[:-1]
value_force = value_force[:-1]
value_force2 = value_force2[:-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(t, value_bm, color='firebrick', label='Brownian motion')
ax1.plot(t,value_force, color='rebeccapurple', label='DEP force')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Strength (N)')
ax1.legend()
ax1.set_title('All time steps')

ax2.plot(t_last, value_bm_last, color='firebrick', label='Brownian motion')
ax2.plot(t_last,value_force_last, color='rebeccapurple', label='DEP force')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Strength (N)')
ax2.legend()
ax2.set_title('First 500 steps')

fig.suptitle('Comparison between the strength of Brownian motion and the DEP force', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.95])

distance = distance[:-1]
distance_last = distance[0: num_last_steps]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,10))
ax1.plot(t_last,value_force_last, color='rebeccapurple', label='DEP force')
ax2.plot(t_last,distance_last, color='royalblue', label='distance from trap')
ax1.axvline(t[trapping_index], color='red', linestyle='--', label='Trapping time')
ax2.axvline(t[trapping_index], color='red', linestyle='--', label='Trapping time')
ax1.set_xlabel('Time(s)')
ax2.set_xlabel('Time(s)')
ax1.set_ylabel('Strength (N)')
ax2.set_ylabel('Distance (m)')
ax1.legend()
ax2.legend()

fig.suptitle('Effect of the DEP force on the distance from the trap', fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,10))
ax1.plot(t,value_force, color='rebeccapurple', label='DEP force')
ax2.plot(t,distance, color='royalblue', label='distance from trap')
ax1.set_xlabel('Time(s)')
ax2.set_xlabel('Time(s)')
ax1.set_ylabel('Strength (N)')
ax2.set_ylabel('Distance (m)')
ax1.legend()
ax2.legend()

fig.suptitle('Periodicity of the DEP force', fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Compute the maximum distance where DEP force is greater than Brownian motion
distances_with_stronger_force = distance[value_force > value_bm]

# Find the maximum of these distances
if distances_with_stronger_force.size > 0:
    max_distance_stronger_force = np.max(distances_with_stronger_force)
else:
    max_distance_stronger_force = 0

print(f"Max distance where force>bm: {max_distance_stronger_force}")


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 10))
ax2.plot(t,value_force, color='rebeccapurple', label='DEP force')
ax1.plot(t,distance, color='royalblue', label='distance from trap')
ax3.plot(t,value_force2, color='rebeccapurple', linestyle='--', label='DEP force')
ax1.axhline(y=max_distance_stronger_force, color='red', linestyle='--', label='Capturing radius')
ax3.set_xlabel('Time(s)')
ax2.set_ylabel('Strength (N)')
ax3.set_ylabel('Strength (N)')
ax1.set_ylabel('Distance (m)')
fig.legend()


fig.suptitle('Periodicity of the DEP force', fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Compute the Fourier Transform of the distance vector
fft_distance = np.fft.fft(distance)
frequencies = np.fft.fftfreq(N-1, d=dt)

# Plotting the amplitude of the FFT result on a log-log scale
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, np.abs(fft_distance), color='forestgreen')
plt.title('Log-Log Plot of Fourier Transform of the Distance Vector', fontsize=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(1e2, 0.2e7)  # Adjust the limits if needed
plt.ylim(1e-3, np.max(np.abs(fft_distance)))  # Adjust the limits if needed
plt.grid(True, which="both", ls="--")
plt.show()