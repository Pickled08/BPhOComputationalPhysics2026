import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
from numba import cuda
from matplotlib.animation import FuncAnimation

#Constants
G = 6.67430e-11 # Gravitational constant in SI units (m^3 kg^-1 s^-2)
m_target = 5.7e24 # Mass of the Target in kg
m_satellite = 10 # Mass of the satellite in kg
eps = 1e-6  
dt = 0.001           #time step s
simulationTime = 6000.0 #sim time s
timeIterations = int(simulationTime/dt)
x0, y0 = 0, 0 # Initial target pos
x1, y1 = 10, 10 # Initial object pos
initial_velocity_s = np.array([1.0, 1.0]) # Initial velocity of satellite in m/s
initial_velocity_t = np.array([0.0, 0.0]) # Initial velocity of target in m/s

@njit(fastmath=True)
def orbit():
    xarr = np.empty(timeIterations)
    yarr = np.empty(timeIterations)
    tarr = np.empty(timeIterations)
    varr = np.empty(timeIterations) 
    aarr = np.empty(timeIterations)
    x2arr = np.empty(timeIterations)
    y2arr = np.empty(timeIterations)
    
    Xt = x0
    Yt = y0
    Xs=x1
    Ys=y1
    velocity_s = initial_velocity_s.copy()
    velocity_t = initial_velocity_t.copy()
    t = 0.0 #Time s

    for i in range(timeIterations):

        #Find Force on Satellite Due to Target
        Us = Xt - Xs
        Vs = Yt - Ys

        #Create Normalised Vector Towards Target
        Us = Xt - Xs
        Vs = Yt - Ys
        mag = np.sqrt(Us**2 + Vs**2)
        Us = Us / mag
        Vs = Vs / mag

        #Adjust Force Magnitude
        r=np.sqrt((Xs - x0)**2 + (Ys - y0)**2)
        r = np.maximum(r, eps)
        F = G * (m_target * m_satellite) / r**2
        ffGravity = np.array([Us,Vs]) * F

        acceleration_s = ffGravity/m_satellite
        
        velocity_s = velocity_s + acceleration_s * dt

        Xs = Xs + velocity_s[0] * dt
        Ys = Ys + velocity_s[1] * dt

        #Find Force on Target Due to Satellite
        Us = Xs - Xt
        Vs = Ys - Yt

        #Create Normalised Vector Towards Target
        Ut = Xs - Xt
        Vt = Ys - Yt
        mag = np.sqrt(Ut**2 + Vt**2)
        Ut = Ut / mag
        Vt = Vt / mag

        #Adjust Force Magnitude
        r=np.sqrt((Xs - x0)**2 + (Ys - y0)**2)
        r = np.maximum(r, eps)
        F = G * (m_target * m_satellite) / r**2
        ffGravity = np.array([Ut,Vt]) * F

        acceleration_t = ffGravity/m_target
        
        velocity_t = velocity_t + acceleration_t * dt

        Xt = Xt + velocity_t[0] * dt
        Yt = Yt + velocity_t[1] * dt

        t = i * dt


        xarr[i] = Xs
        yarr[i] = Ys
        x2arr[i] = Xt
        y2arr[i] = Yt
        tarr[i] = t
        varr[i] = np.linalg.norm(velocity_s)
        aarr[i] = np.linalg.norm(acceleration_s)

    return tarr, xarr, yarr, varr, aarr, x2arr, y2arr

#Plotting Functions
def animate_orbit():
    tarr, xarr, yarr, varr, aarr, x2arr, y2arr = orbit()

    # --- downsample for animation ---
    step = 500          # <-- key fix (adjust for smoothness vs speed)
    tarr = tarr[::step]
    xarr = xarr[::step]
    yarr = yarr[::step]
    x2arr = x2arr[::step]
    y2arr = y2arr[::step]

    fig, ax = plt.subplots()
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Satellite Orbit Path")
    ax.axis("equal")
    ax.grid(True)

    ax.set_xlim(min(xarr.min(), x2arr.min()), max(xarr.max(), x2arr.max()))
    ax.set_ylim(min(yarr.min(), y2arr.min()), max(yarr.max(), y2arr.max()))

    trail, = ax.plot([], [], lw=1)
    satellite, = ax.plot([], [], 'o')
    body2, = ax.plot([], [], 'o', color='red')

    def init():
        trail.set_data([], [])
        satellite.set_data([], [])
        body2.set_data([], [])
        return trail, satellite, body2

    def update(i):
        trail.set_data(xarr[:i], yarr[:i])
        satellite.set_data([xarr[i]], [yarr[i]])
        body2.set_data([x2arr[i]], [y2arr[i]])
        return trail, satellite, body2


    ani = FuncAnimation(
        fig,
        update,
        frames=len(tarr),
        init_func=init,
        interval=20,
        blit=True
    )

    plt.show()

animate_orbit()