import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
from numba import cuda
from matplotlib.animation import FuncAnimation

#Constants
title = "Halley's Comet Orbit Simulation Using Velocity Verlet Integration"
G = 6.674e-11  # Gravitational constant in SI units (m^3 kg^-1 s^-2)

# Halley's comet has extreme elliptical orbit (e=0.967)
m_target = 1.989e30    # Mass of Sun (kg, real value)
m_satellite = 2.2e14   # Mass of Halley's comet (kg, real value)

dt = 10000.0           # time step s
simulationTime = 5e9   # sim time s (~1.5 years to see the ellipse)
timeIterations = int(simulationTime/dt)

target_pos_init = np.array([0.0, 0.0])          # Sun at origin
satellite_pos_init = np.array([8.77e10, 0.0])   # Start at perihelion (0.586 AU in meters)

# At perihelion: v = sqrt(G*M*(2/r - 1/a))
# Semi-major axis a = 2.67e12 m
# v = sqrt(6.674e-11 * 1.989e30 * (2/8.77e10 - 1/2.67e12)) ≈ 54,550 m/s
initial_velocity_t = np.array([0.0, 0.0])       # Sun stationary
initial_velocity_s = np.array([0.0, 54550.0])   # Fast at closest approach

# --- Numba JIT-compiled Functions ---
@njit(fastmath=True)
def norm2(v):
    return np.sqrt(np.dot(v, v))

@njit(fastmath=True)
def orbit():
    # Preallocate arrays for storing data
    xarr = np.empty(timeIterations)
    yarr = np.empty(timeIterations)
    tarr = np.empty(timeIterations)
    varr = np.empty(timeIterations) 
    aarr = np.empty(timeIterations)
    x2arr = np.empty(timeIterations)
    y2arr = np.empty(timeIterations)

    # Initialize Positions and Velocities
    target_pos   = np.empty_like(target_pos_init)
    target_pos[:] = target_pos_init

    satellite_pos = np.empty_like(satellite_pos_init)
    satellite_pos[:] = satellite_pos_init

    velocity_s = np.empty_like(initial_velocity_s)
    velocity_s[:] = initial_velocity_s

    velocity_t = np.empty_like(initial_velocity_t)
    velocity_t[:] = initial_velocity_t

    t = 0.0 #Time s

    # Main Simulation Loop
    for i in range(timeIterations):
        
        # ===== STEP 1: Calculate CURRENT accelerations =====
        vector_to_target = target_pos - satellite_pos
        r_s = norm2(vector_to_target)
        
        if r_s > 0:
            unit_vector_to_target = vector_to_target / r_s
            ffgravity_satellite = unit_vector_to_target * (G * m_satellite * m_target / r_s**2)
        else:
            ffgravity_satellite = np.zeros_like(vector_to_target)
        
        ffgravity_target = -ffgravity_satellite  # Newton's 3rd law
        
        acceleration_s = ffgravity_satellite / m_satellite
        acceleration_t = ffgravity_target / m_target
        
        
        # ===== STEP 2: Update positions using current velocities + accelerations =====
        satellite_pos = satellite_pos + velocity_s * dt + 0.5 * acceleration_s * dt**2
        target_pos = target_pos + velocity_t * dt + 0.5 * acceleration_t * dt**2
        
        
        # ===== STEP 3: Calculate NEW accelerations at NEW positions =====
        vector_to_target_new = target_pos - satellite_pos
        r_s_new = norm2(vector_to_target_new)
        
        if r_s_new > 0:
            unit_vector_to_target_new = vector_to_target_new / r_s_new
            ffgravity_satellite_new = unit_vector_to_target_new * (G * m_satellite * m_target / r_s_new**2)
        else:
            ffgravity_satellite_new = np.zeros_like(vector_to_target_new)
        
        ffgravity_target_new = -ffgravity_satellite_new
        
        acceleration_s_new = ffgravity_satellite_new / m_satellite
        acceleration_t_new = ffgravity_target_new / m_target
        
        
        # ===== STEP 4: Update velocities using AVERAGE of old and new accelerations =====
        velocity_s = velocity_s + 0.5 * (acceleration_s + acceleration_s_new) * dt
        velocity_t = velocity_t + 0.5 * (acceleration_t + acceleration_t_new) * dt
        
        
        # ===== STEP 5: Update time and store data =====
        t = t + dt
        
        xarr[i] = satellite_pos[0]
        yarr[i] = satellite_pos[1]
        tarr[i] = t
        varr[i] = norm2(velocity_s)
        aarr[i] = norm2(acceleration_s_new)  # Store the final acceleration
        x2arr[i] = target_pos[0]
        y2arr[i] = target_pos[1]

    return tarr, xarr, yarr, varr, aarr, x2arr, y2arr




# --- Plotting Functions ---
def animate_orbit():
    tarr, xarr, yarr, varr, aarr, x2arr, y2arr = orbit()

    # --- downsample for animation ---
    step = 100  # adjust for smoothness vs speed
    tarr = tarr[::step]
    xarr = xarr[::step]
    yarr = yarr[::step]
    x2arr = x2arr[::step]
    y2arr = y2arr[::step]

    fig, ax = plt.subplots()
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(title)
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
    satellite_trail, = ax.plot([], [], lw=1, color='blue')
    satellite_dot, = ax.plot([], [], 'o', color='blue')
    target_trail, = ax.plot([], [], lw=1, color='red')  # red line for target
    target_dot, = ax.plot([], [], 'o', color='red')     # red dot for target

    def init():
        satellite_trail.set_data([], [])
        satellite_dot.set_data([], [])
        target_trail.set_data([], [])
        target_dot.set_data([], [])
        return satellite_trail, satellite_dot, target_trail, target_dot

    def update(i):
        satellite_trail.set_data(xarr[:i], yarr[:i])
        satellite_dot.set_data([xarr[i]], [yarr[i]])
        target_trail.set_data(x2arr[:i], y2arr[:i])  # trail for target
        target_dot.set_data([x2arr[i]], [y2arr[i]])
        return satellite_trail, satellite_dot, target_trail, target_dot

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