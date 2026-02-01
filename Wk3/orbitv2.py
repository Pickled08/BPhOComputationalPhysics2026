import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
from numba import cuda
from matplotlib.animation import FuncAnimation

#Constants
G = 10 # Gravitational constant in SI units (m^3 kg^-1 s^-2)
m_target = 10000  # Mass of the Target in kg
m_satellite = 1 # Mass of the satellite in kg
eps = 1e-6  
dt = 0.00001           #time step s
simulationTime = 60.0 #sim time s
timeIterations = int(simulationTime/dt)
target_pos_init = np.array([0.0,0.0])
satellite_pos_init = np.array([10.0,0.0])
initial_velocity_s = np.array([0.0, 100.0]) # Initial velocity of satellite in m/s
initial_velocity_t = np.array([0.0, 0.0]) # Initial velocity of target in m/s

@njit(fastmath=True)
def norm2(v):
    return np.sqrt(np.dot(v, v))

@njit(fastmath=True)
def orbit():
    xarr = np.empty(timeIterations)
    yarr = np.empty(timeIterations)
    tarr = np.empty(timeIterations)
    varr = np.empty(timeIterations) 
    aarr = np.empty(timeIterations)
    x2arr = np.empty(timeIterations)
    y2arr = np.empty(timeIterations)

    target_pos   = np.empty_like(target_pos_init)
    target_pos[:] = target_pos_init

    satellite_pos = np.empty_like(satellite_pos_init)
    satellite_pos[:] = satellite_pos_init

    velocity_s = np.empty_like(initial_velocity_s)
    velocity_s[:] = initial_velocity_s

    velocity_t = np.empty_like(initial_velocity_t)
    velocity_t[:] = initial_velocity_t

    t = 0.0 #Time s

    for i in range(timeIterations):

        vector_to_target = target_pos - satellite_pos
        r = norm2(vector_to_target)

        if r > 0:
            unit_vector_to_target = vector_to_target / r
            ffgravity_satellite = unit_vector_to_target * (G * m_satellite * m_target / r**2)
        else:
            ffgravity_satellite = np.zeros_like(vector_to_target)

        acceleration_s = ffgravity_satellite / m_satellite

        velocity_s = velocity_s + acceleration_s * dt
        satellite_pos = satellite_pos + velocity_s * dt
        t = t + dt

        xarr[i] = satellite_pos[0]
        yarr[i] = satellite_pos[1]
        tarr[i] = t
        varr[i] = norm2(velocity_s)
        aarr[i] = norm2(acceleration_s)
        x2arr[i] = target_pos[0]
        y2arr[i] = target_pos[1]
    return tarr, xarr, yarr, varr, aarr, x2arr, y2arr




# --- Plotting Functions ---
def animate_orbit():
    tarr, xarr, yarr, varr, aarr, x2arr, y2arr = orbit()

    # --- downsample for animation ---
    step = 500  # adjust for smoothness vs speed
    tarr = tarr[::step]
    xarr = xarr[::step]
    yarr = yarr[::step]
    x2arr = x2arr[::step]
    y2arr = y2arr[::step]

    fig, ax = plt.subplots()
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Satellite Orbit Path")
    ax.grid(True)

    # --- Center on satellite initial position ---
    x0, y0 = xarr[0], yarr[0]
    max_range = max(
        xarr.max() - xarr.min(),
        yarr.max() - yarr.min(),
        x2arr.max() - x2arr.min(),
        y2arr.max() - y2arr.min()
    )
    half_range = max_range / 2

    ax.set_xlim(x0 - half_range, x0 + half_range)
    ax.set_ylim(y0 - half_range, y0 + half_range)

    # --- make XY scale equal ---
    ax.set_aspect('equal', adjustable='box')

    # --- plot elements ---
    trail, = ax.plot([], [], lw=1)
    satellite, = ax.plot([], [], 'o', color='blue')
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