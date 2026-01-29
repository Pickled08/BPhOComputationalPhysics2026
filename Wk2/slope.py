import numpy as np
from numba import njit
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import math

#Intial Conditions
m =25 #kg
g = np.array([0,-9.81]) #kgN^-1
theta = 0.588176 #rads
mu = 1 #coefficient of friction
force_applied = 0

weight = m * g

def slope(m,g,theta,mu,force_applied,weight):

    # Slope at angle theta
    n_hat = np.array([-np.sin(theta), np.cos(theta)])   # perpendicular to slope (pointing up from slope)
    t_hat = np.array([np.cos(theta), np.sin(theta)])    # along slope (downhill)

    force_perp = np.dot(weight, n_hat) * n_hat
    force_parr = np.dot(weight, t_hat) * t_hat

    force_normal = -force_perp

    # Maximum static friction
    F_friction_max = mu * np.linalg.norm(force_normal)

    # Net force along slope **without friction**
    F_along_slope = np.dot(force_parr + force_applied * -t_hat, t_hat)

    # Determine friction
    if abs(F_along_slope) <= F_friction_max:
        # Friction cancels any tendency to move -> block stays still
        force_friction = -F_along_slope * t_hat
    else:
        # Block slides -> kinetic friction opposes motion
        direction = -np.sign(F_along_slope)  # opposite to motion
        force_friction = direction * F_friction_max * t_hat

    force_external = force_applied * t_hat

    force_resultant = force_perp+force_parr+force_friction+force_normal+force_external

    return force_parr, force_perp, force_normal, force_friction, force_external, force_resultant

def plot_forces():
    forces = slope(m, g, theta, mu, force_applied, weight)

    origin = np.zeros((len(forces), 2))  # all arrows start at 0,0

    U = np.array([f[0] for f in forces])
    V = np.array([f[1] for f in forces])

    plt.quiver(origin[:,0], origin[:,1], U, V, angles='xy', scale_units='xy', scale=1,
               color=['r','g','b','c','m','k'])
    plt.axis('equal')
    plt.grid(True)
    plt.show()

plot_forces()