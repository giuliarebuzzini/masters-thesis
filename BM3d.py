'Simulation of Brownian Motion in 3 dimensions'

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#--------------------------------------------------------------------------------------------------
# Parameters
N = 5  # Number of Brownian motion trajectories
num_steps = 1000     # Number of steps
dt = 1      # Time step size

#--------------------------------------------------------------------------------------------------
# Definition of the function to compute the 3 dimensiona Brownian Motion
def brownian_motion_3d(num_steps, dt):
    random_steps_x = np.random.normal(0, np.sqrt(dt), num_steps)
    random_steps_y = np.random.normal(0, np.sqrt(dt), num_steps)
    random_steps_z = np.random.normal(0, np.sqrt(dt), num_steps)
    
    # Compute the positions
    position_x = np.cumsum(random_steps_x)
    position_y = np.cumsum(random_steps_y)
    position_z = np.cumsum(random_steps_z)
    
    return position_x, position_y, position_z

#--------------------------------------------------------------------------------------------------
# Simulate multiple Brownian motions
trajectories = []
for i in range(N):
    trajectories.append(brownian_motion_3d(num_steps, dt))

# Plot the 3D Brownian motions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for trajectory in trajectories:
    ax.plot(trajectory[0], trajectory[1], trajectory[2])
ax.set_title('Brownian Motions in 3 dimensions', fontsize=16) 
ax.scatter(trajectory[0][0], trajectory[1][0], trajectory[2][0], color='blue', s=50, marker='*') 
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.show()
