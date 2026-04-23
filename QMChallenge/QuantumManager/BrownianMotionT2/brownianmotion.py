import numpy as np # Numpy: Math functions and arrays 
import matplotlib.pyplot as plt # Matplotlib: plotting
import math
import time

def brownian_motion(n, m , r, v, M, R, xmax, ymax,kappa, dt):
    rng = np.random.default_rng()

    particle_pos = rng.uniform(low=0, high=[xmax, ymax], size=(n, 2))
    
    particle_angle = rng.uniform(0, 2 * np.pi, size=n)
    
    noise = rng.normal(0.0, 1.0 / (kappa + 1e-9), size=n)
    particle_angle += noise
    

    # Clean up the plot
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Particle Positions")
    plt.ion()
    
    plt.scatter(particle_pos[:, 0], particle_pos[:, 1], s=10)
    
    next_tick = time.perf_counter()
    
    while True:
        
        noise = rng.normal(0.0, 1.0 / (kappa + 1e-9), size=n)
        particle_angle += noise 
        step_size = v * dt
        particle_pos[:, 0] += step_size * np.cos(particle_angle) # Update X
        particle_pos[:, 1] += step_size * np.sin(particle_angle) # Update Y
        
        plt.cla() 

        plt.scatter(particle_pos[:, 0], particle_pos[:, 1], s=10)

        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        plt.pause(0.001) 
        
        next_tick += dt
        sleep_time = next_tick - time.perf_counter()
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()
        
    
    
    
    
n=500
m=1
r=1
v=20
M=1000
R=10
xmax=100
ymax=100
kappa = 10
dt=0.01

brownian_motion(n, m , r, v, M, R, xmax, ymax, kappa, dt)
    