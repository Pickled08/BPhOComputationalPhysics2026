import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
from numba import cuda
from matplotlib.animation import FuncAnimation

# Constants
title = "Slow Binary Spin with Later Intruder"
G = 6.674e-11  

m_obj1 = 5.0e24
m_obj2 = 5.0e24
m_obj3 = 7.0e24   # intruder

dt = 2000.0
simulationTime = 5e9   # total sim time
timeIterations = int(simulationTime/dt)

# --- Binary positions ---
obj1_init = np.array([-2.0e10, 0.0])
obj2_init = np.array([ 2.0e10, 0.0])

# --- Intruder starts closer so it interacts near end ---
obj3_init = np.array([0.0, 5.0e11])

# --- Velocities ---
# Slow binary for visible orbit
initial_velocity_obj1 = np.array([0.0, 40.0])
initial_velocity_obj2 = np.array([0.0, -40.0])

# Slow angled intruder, so it reaches binary after a few spins
initial_velocity_obj3 = np.array([0.0, -100.0])


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
    x3arr = np.empty(timeIterations)
    y3arr = np.empty(timeIterations)

    # Initialize Positions and Velocities
    obj1_pos  = np.empty_like(obj1_init)
    obj1_pos[:] = obj1_init

    obj2_pos = np.empty_like(obj2_init)
    obj2_pos[:] = obj2_init
    
    obj3_pos = np.empty_like(obj3_init)
    obj3_pos[:] = obj3_init
    
    velocity_obj1 = np.empty_like(initial_velocity_obj1)
    velocity_obj1[:] = initial_velocity_obj1
    
    velocity_obj2 = np.empty_like(initial_velocity_obj2)
    velocity_obj2[:] = initial_velocity_obj2
    
    velocity_obj3 = np.empty_like(initial_velocity_obj3)
    velocity_obj3[:] = initial_velocity_obj3

    t = 0.0 #Time s

    # Main Simulation Loop
    for i in range(timeIterations):
        
        #Object 1
        vector_to_obj2 = obj2_pos - obj1_pos
        r_12 = norm2(vector_to_obj2)
        if r_12 > 0:
            unit_vector_to_obj2 = vector_to_obj2 / r_12
            ffgravity_obj1_from_obj2 = unit_vector_to_obj2 * (G * m_obj1 * m_obj2 / r_12**2)
        else:
            ffgravity_obj1_from_obj2 = np.zeros_like(vector_to_obj2)
            
        vector_to_obj3 = obj3_pos - obj1_pos
        r_13 = norm2(vector_to_obj3)
        if r_13 > 0:
            unit_vector_to_obj3 = vector_to_obj3 / r_13
            ffgravity_obj1_from_obj3 = unit_vector_to_obj3 * (G * m_obj1 * m_obj3 / r_13**2)
        else:
            ffgravity_obj1_from_obj3 = np.zeros_like(vector_to_obj3)
        
        ffgravity_obj1 = ffgravity_obj1_from_obj2 + ffgravity_obj1_from_obj3    
        
        #Object 2
        vector_to_obj1 = obj1_pos - obj2_pos
        r_21 = norm2(vector_to_obj1)
        if r_21 > 0:
            unit_vector_to_obj1 = vector_to_obj1 / r_21
            ffgravity_obj2_from_obj1 = unit_vector_to_obj1 * (G * m_obj2 * m_obj1 / r_21**2)
        else:
            ffgravity_obj2_from_obj1 = np.zeros_like(vector_to_obj1)
            
        vector_to_obj3 = obj3_pos - obj2_pos
        r_23 = norm2(vector_to_obj3)
        if r_23 > 0:
            unit_vector_to_obj3 = vector_to_obj3 / r_23
            ffgravity_obj2_from_obj3 = unit_vector_to_obj3 * (G * m_obj2 * m_obj3 / r_23**2)
        else:
            ffgravity_obj2_from_obj3 = np.zeros_like(vector_to_obj3)
            
        ffgravity_obj2 = ffgravity_obj2_from_obj1 + ffgravity_obj2_from_obj3
        
        #Object 3
        vector_to_obj1 = obj1_pos - obj3_pos
        r_31 = norm2(vector_to_obj1)
        if r_31 > 0:
            unit_vector_to_obj1 = vector_to_obj1 / r_31
            ffgravity_obj3_from_obj1 = unit_vector_to_obj1 * (G * m_obj3 * m_obj1 / r_31**2)
        else:
            ffgravity_obj3_from_obj1 = np.zeros_like(vector_to_obj1)
            
        vector_to_obj2 = obj2_pos - obj3_pos
        r_32 = norm2(vector_to_obj2)
        if r_32 > 0:
            unit_vector_to_obj2 = vector_to_obj2 / r_32
            ffgravity_obj3_from_obj2 = unit_vector_to_obj2 * (G * m_obj3 * m_obj2 / r_32**2)
        else:
            ffgravity_obj3_from_obj2 = np.zeros_like(vector_to_obj2)
            
        ffgravity_obj3 = ffgravity_obj3_from_obj1 + ffgravity_obj3_from_obj2
        
        # Update accelerations
        acceleration_obj1 = ffgravity_obj1 / m_obj1
        acceleration_obj2 = ffgravity_obj2 / m_obj2
        acceleration_obj3 = ffgravity_obj3 / m_obj3
        
        # Update positions using current velocities + accelerations
        obj1_pos = obj1_pos + velocity_obj1 * dt + 0.5 * acceleration_obj1 * dt**2
        obj2_pos = obj2_pos + velocity_obj2 * dt + 0.5 * acceleration_obj2 * dt**2
        obj3_pos = obj3_pos + velocity_obj3 * dt + 0.5 * acceleration_obj3 * dt**2
        # Calculate NEW accelerations at NEW positions

        #Object 1 New
        vector_to_obj2_new = obj2_pos - obj1_pos
        r_12_new = norm2(vector_to_obj2_new)
        if r_12_new > 0:
            unit_vector_to_obj2_new = vector_to_obj2_new / r_12_new
            ffgravity_obj1_from_obj2_new = unit_vector_to_obj2_new * (G * m_obj1 * m_obj2 / r_12_new**2)
        else:
            ffgravity_obj1_from_obj2_new = np.zeros_like(vector_to_obj2_new)
            
        vector_to_obj3_new = obj3_pos - obj1_pos
        r_13_new = norm2(vector_to_obj3_new)
        if r_13_new > 0:
            unit_vector_to_obj3_new = vector_to_obj3_new / r_13_new
            ffgravity_obj1_from_obj3_new = unit_vector_to_obj3_new * (G * m_obj1 * m_obj3 / r_13_new**2)
        else:
            ffgravity_obj1_from_obj3_new = np.zeros_like(vector_to_obj3_new)
            
        ffgravity_obj1_new = ffgravity_obj1_from_obj2_new + ffgravity_obj1_from_obj3_new
        
        #Object 2 New
        vector_to_obj1_new = obj1_pos - obj2_pos
        r_21_new = norm2(vector_to_obj1_new)
        if r_21_new > 0:
            unit_vector_to_obj1_new = vector_to_obj1_new / r_21_new
            ffgravity_obj2_from_obj1_new = unit_vector_to_obj1_new * (G * m_obj2 * m_obj1 / r_21_new**2)
        else:
            ffgravity_obj2_from_obj1_new = np.zeros_like(vector_to_obj1_new)
            
        vector_to_obj3_new = obj3_pos - obj2_pos
        r_23_new = norm2(vector_to_obj3_new)
        if r_23_new > 0:
            unit_vector_to_obj3_new = vector_to_obj3_new / r_23_new
            ffgravity_obj2_from_obj3_new = unit_vector_to_obj3_new * (G * m_obj2 * m_obj3 / r_23_new**2)
        else:
            ffgravity_obj2_from_obj3_new = np.zeros_like(vector_to_obj3_new)
            
        ffgravity_obj2_new = ffgravity_obj2_from_obj1_new + ffgravity_obj2_from_obj3_new
        #Object 3 New
        vector_to_obj1_new = obj1_pos - obj3_pos
        r_31_new = norm2(vector_to_obj1_new)
        if r_31_new > 0:
            unit_vector_to_obj1_new = vector_to_obj1_new / r_31_new
            ffgravity_obj3_from_obj1_new = unit_vector_to_obj1_new * (G * m_obj3 * m_obj1 / r_31_new**2)
        else:
            ffgravity_obj3_from_obj1_new = np.zeros_like(vector_to_obj1_new)
            
        vector_to_obj2_new = obj2_pos - obj3_pos
        r_32_new = norm2(vector_to_obj2_new)
        if r_32_new > 0:
            unit_vector_to_obj2_new = vector_to_obj2_new / r_32_new
            ffgravity_obj3_from_obj2_new = unit_vector_to_obj2_new * (G * m_obj3 * m_obj2 / r_32_new**2)
        else:
            ffgravity_obj3_from_obj2_new = np.zeros_like(vector_to_obj2_new)
            
        ffgravity_obj3_new = ffgravity_obj3_from_obj1_new + ffgravity_obj3_from_obj2_new
        # Update velocities using AVERAGE of old and new accelerations
        velocity_obj1 = velocity_obj1 + 0.5 * (acceleration_obj1 + ffgravity_obj1_new / m_obj1) * dt
        velocity_obj2 = velocity_obj2 + 0.5 * (acceleration_obj2 + ffgravity_obj2_new / m_obj2) * dt
        velocity_obj3 = velocity_obj3 + 0.5 * (acceleration_obj3 + ffgravity_obj3_new / m_obj3) * dt
        
        # Update time and store data
        t = t + dt
        xarr[i] = obj1_pos[0]
        yarr[i] = obj1_pos[1]
        tarr[i] = t
        varr[i] = norm2(velocity_obj1)
        aarr[i] = norm2(acceleration_obj1)  # Store the final acceleration
        x2arr[i] = obj2_pos[0]
        y2arr[i] = obj2_pos[1]
        x3arr[i] = obj3_pos[0]
        y3arr[i] = obj3_pos[1]
        
    return tarr, xarr, yarr, varr, aarr, x2arr, y2arr, x3arr, y3arr




# --- Plotting Functions ---
def animate_orbit():
    tarr, xarr, yarr, varr, aarr, x2arr, y2arr, x3arr, y3arr = orbit()

    # --- downsample for animation ---
    step = 2000  # adjust for smoothness vs speed
    tarr = tarr[::step]
    xarr = xarr[::step]
    yarr = yarr[::step]
    x2arr = x2arr[::step]
    y2arr = y2arr[::step]
    x3arr = x3arr[::step]
    y3arr = y3arr[::step]

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
        y2arr.max() - y2arr.min(),
        x3arr.max() - x3arr.min(),
        y3arr.max() - y3arr.min()
    )
    half_range = max_range / 2

    ax.set_xlim(x0 - half_range, x0 + half_range)
    ax.set_ylim(y0 - half_range, y0 + half_range)

    # --- make XY scale equal ---
    ax.set_aspect('equal', adjustable='box')

    # --- plot elements ---
    obj1_trail, = ax.plot([], [], lw=1, color='blue')
    obj1_dot, = ax.plot([], [], 'o', color='blue')
    obj2_trail, = ax.plot([], [], lw=1, color='red')  
    obj2_dot, = ax.plot([], [], 'o', color='red')     
    obj3_trail, = ax.plot([], [], lw=1, color='green')  
    obj3_dot, = ax.plot([], [], 'o', color='green') 

    def init():
        obj1_trail.set_data([], [])
        obj1_dot.set_data([], [])
        obj2_trail.set_data([], [])
        obj2_dot.set_data([], [])
        obj3_trail.set_data([], [])
        obj3_dot.set_data([], [])
        return obj1_trail, obj1_dot, obj2_trail, obj2_dot, obj3_trail, obj3_dot

    def update(i):
        obj1_trail.set_data(xarr[:i], yarr[:i])
        obj1_dot.set_data([xarr[i]], [yarr[i]])
        obj2_trail.set_data(x2arr[:i], y2arr[:i])  
        obj2_dot.set_data([x2arr[i]], [y2arr[i]])     
        obj3_trail.set_data(x3arr[:i], y3arr[:i])  
        obj3_dot.set_data([x3arr[i]], [y3arr[i]]) 
        return obj1_trail, obj1_dot, obj2_trail, obj2_dot, obj3_trail, obj3_dot

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