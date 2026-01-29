import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
from numba import cuda

#Constants
G = 6.67430e-11 # Gravitational constant in SI units (m^3 kg^-1 s^-2)
m_target = 100000  # Mass of the Target in kg
m_satellite = 10 # Mass of the satellite in kg
eps = 1e-6  
dt = 0.001           #time step s
simulationTime = 6000.0 #sim time s
timeIterations = int(simulationTime/dt)
x0, y0 = 0, 0 # Initial target pos
x1, y1 = 10, 10 # Initial object pos
initial_velocity = np.array([0.0, 1.0]) # Initial velocity of satellite in m/s

@njit(fastmath=True)
def orbit():
    xarr = np.empty(timeIterations)
    yarr = np.empty(timeIterations)
    tarr = np.empty(timeIterations)
    varr = np.empty(timeIterations) 
    aarr = np.empty(timeIterations)
    
    X=x1
    Y=y1
    velocity = initial_velocity.copy()
    t = 0.0 #Time s

    for i in range(timeIterations):
        U = x0 - X
        V = y0 - Y

        #Create Normalised Vector Towards Target
        U = x0 - X
        V = y0 - Y
        mag = np.sqrt(U**2 + V**2)
        U = U / mag
        V = V / mag

        #Adjust Force Magnitude
        r=np.sqrt((X - x0)**2 + (Y - y0)**2)
        r = np.maximum(r, eps)
        F = G * (m_target * m_satellite) / r**2
        ffGravity = np.array([U,V])

        acceleration = ffGravity/m_satellite
        
        velocity = velocity + acceleration * dt

        X = X + velocity[0] * dt
        Y = Y + velocity[1] * dt
        t = i * dt


        xarr[i] = X
        yarr[i] = Y
        tarr[i] = t
        varr[i] = np.linalg.norm(velocity)
        aarr[i] = np.linalg.norm(acceleration)

    return tarr, xarr, yarr, varr, aarr

#Plotting Functions
def plot_orbit():
    tarr, xarr, yarr, varr, aarr = orbit()  # unpack

    plt.figure()
    plt.plot(xarr, yarr, linewidth=1)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Satellite Orbit Path")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

plot_orbit()