'Simulation of the Langevin equation for a particle subjected to Brownian Motion and dielectrophoretic force'

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from parameter import * # Library created to store all the parameters
import matplotlib.animation as animation
from plotter import * # Library created for plotting

#------------------------------------------------------------------------------------------------------------------------
# Initialization of time and variables
tt = 0.5                                             # Final time of the simulation
dt = 0.0005                                         # Time step
N = int(tt/dt)                                       # Number of time steps of the simulation
t = np.linspace(0,tt, N) 
delta = t[2]-t[1] 

B = np.random.normal(0, 1, (3, N))                  # Random noise term of dimension 3xN
x = np.zeros((3, N))                                # Main position vector
x2 = np.zeros((3, N))                               # Position vector to store force turned off
x[:, 0] = [5e-8, 5e-8, 5e-8]                        # Initial position
x2[:, 0] = [5e-8, 5e-8, 5e-8]                       # Initial position
x_e = np.zeros(3)                                   # Trap position                        

trapped = np.zeros(N)                               # Storage array for trapping times
S = 1 + np.sin(2 * np.pi * f * t)                   # Dynamic traping field
distance = np.zeros(N)                              # N dimensional array to store the distance between a position and the previous one
distance2 = np.zeros(N)                             # Same as before but for the array x2

#------------------------------------------------------------------------------------------------------------------------
# Definition of useful functions
def distance_function(v1, v2):
    """This function serves the purpose of calculating the distance
        between two vectors and checks whether it is zero or not """
    
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
    without the addition of the electric field gradient"""

    eps0 = 8.854e-12  # Permittivity of free space (F/m)

    # Clausius-Mossotti factor
    K = (permittivity_particle - permittivity_medium) / (permittivity_particle + 2 * permittivity_medium)

    F = (2 * np.pi * eps0 * permittivity_medium * radius ** 3 * 1 * electric_field_gradient)

    return F # Returns the dielectrophoretic force

def Egrad(distance):
    """ This function returns the electric field gradient we are working with 1/(x-x_e)^5"""
    return 1e-19/(distance **5)

def Diffusion_tensor(r, r0):
    """ Function that computes the diffusion tensor and the perpendicular and parallel tensors """

    D_tensor = np.zeros(3) # Initialization of the diffusion tensor

    z = r[2, :] 
    ro_z = r0 / z

    H_parallel = 1 - (9 / 16) * (ro_z) + (1 / 8) * (ro_z) ** 3 - (45 / 256) * (ro_z) ** 4 - (1 / 16) * (ro_z) ** 5
    H_perp = (6 * z ** 2 + 2 * r0 * z) / (6 * z ** 2 + 9 * r0 * z + 2 * r0 ** 2)

    # Defining the components of the tensor according to the theory
    D_tensor[0, :] = D0 * H_parallel
    D_tensor[1, :] = D0 * H_parallel
    D_tensor[2, :] = D0 * H_perp

    return D_tensor


def collision_detection(x_now, x_next, ro):
    """Function to detect collision of the particle against the bounday ofthe trap"""

    forceVector, distance = distance_function(x_next, np.array([0, 0, 0]))

    #radius = 3e-8 # Radius of the trap

    if distance <= ro+3e-8:
        # Collision Detected
        v = x_next - x_now
        # Compute the dot product of v1 and v2
        dot_product = np.dot(v, forceVector)

        # Compute the squared magnitude of v2
        v2_magnitude_squared = np.dot(forceVector, forceVector)

        # Compute the projection of v1 onto the direction of v2
        projection = dot_product / v2_magnitude_squared * forceVector

        # Subtract the projection from v1
        result = v - projection
        x_next = x_now # If there is a collision between the wall and the particle, then the position is the previous one

    return x_next

#--------------------------------------------------------------------------------------------------------------------------
# Simulation running for every time step
# Main Simulation Loop
for m in range(N - 1):
    factor1 = np.sqrt(2 * D) * delta
    factor2 = D * delta / (kb * T)
    forceVector, distance[m] = distance_function(x[:, m], x_e)
    E = S[m] ** 2 * Egrad(distance[m])  # Electric field
    fdep = F_dep(ro, 2.1, 80.2, E) * forceVector  # Dielectrophoretic force
    firstTime = 0
    t_electrode = 0.00

    for n in range(600, N-1):
        distance2[n] = distance_function(x2[:, n], x_e)[1]
        distance[n] = distance2[n]

    if distance[m] < 2e-8:
        trapped[m] = 1
        if firstTime == 0:
            t_electrode = m * dt
            firstTime = 1

    # Update of the position in time using the Euler-Maruyama method
    x[:, m + 1] = x[:, m] + (factor1 * B[:, m] + factor2 * fdep)  # Brownian Motion and force action
    x2[:, m + 1] = x2[:, m] + (factor1 * B[:,m])# + fct_2 * fdep) # Euler-Maruyama method
    x[:, m + 1] = collision_detection(x[:, m], x[:, m + 1], ro)  # Activation of the detector of collision

    if m == N - 1:  # Fixed condition
        distance[m] = distance[m - 1]

distance[N - 1] = distance[N - 2]  # Not jump in the last time step to zero
distance2[N-1] = distance2[N-2] # not jump in the last time step to zero

#------------------------------------------------------------------------------------------------------------------------
# Plotting of the simulation and the displacement for time step
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize lists to store history
X, Y, Z = x[0, :], x[1, :], x[2, :]
X2, Y2, Z2 = x2[0, :], x2[1, :], x2[2, :]

t_star = 900

X2[t_star] = X[t_star-1]
Y2[t_star] = Y[t_star-1]
Z2[t_star] = Z[t_star-1]

# Plotting the simulation
line, = ax.plot(X[0:t_star-1], Y[0:t_star-1], Z[0:t_star-1], 'tomato')
line, = ax.plot(X2[t_star:N-1], Y2[t_star:N-1], Z2[t_star:N-1], 'teal') # To emphasize the switch off of the force

# Drawing the starting point and trap point
ax.scatter(x_e[0], x_e[1], x_e[2], marker='*', label='Trap position', color='blue')
ax.scatter(x[0, 0], x[1, 0], x[2, 0], marker='*', label='Initial position', color='red')

plt.title('Action of the dielectrophoretic force \n with Brownian Motion', fontsize=16)
ax.set_xlabel('X(m)', color='red')
ax.set_ylabel('Y(m)', color='red')
ax.set_zlabel('Z(m)', color='red')
ax.legend()
plt.draw()

# Trap boundaries 
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = x_e[0] + 3e-8 * np.outer(np.cos(u), np.sin(v))
y = x_e[1] + 3e-8 * np.outer(np.sin(u), np.sin(v))
z = x_e[2] + 3e-8 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z, color='b', alpha=0.5)
plt.show()

# Plot the displacement for every time step
fig, ax = plt.subplots()
ax.plot(t, distance, color='royalblue')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m)')
plt.title('Displacement for every time step', fontsize= 16)
plt.show()