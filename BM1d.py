'Simulation of Brownian Motion in 1 dimension'

#import useful libraries
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
# Parameters
N = 5  # Number of Brownian motion trajectories
num_steps = 1000     # Number of steps
dt = 1      # Time step size

#----------------------------------------------------------------------------------
# Definition of the function to compute Brownian motion
def brownian_motion_1d(num_steps, dt):
    random_steps = np.random.normal(0, np.sqrt(dt), num_steps)

    position = np.cumsum(random_steps)
    
    return position

#-----------------------------------------------------------------------------------
# Simulate multiple Brownian motions
trajectories = []
for i in range(N):
    trajectories.append(brownian_motion_1d(num_steps, dt))

# Plot the Brownian motions
plt.figure(figsize=(10, 6))
for trajectory in trajectories:
    plt.plot(trajectory)
plt.title('Multiple Brownian Motions in 1 dimension', fontsize=16)
plt.xlabel('Time Step (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.show()
