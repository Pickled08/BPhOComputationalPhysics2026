import numpy as np # Numpy: Math functions and arrays 
import matplotlib.pyplot as plt # Matplotlib: plotting
import tkinter as tk
from tkinter import ttk
import time

global running
running = True

def on_closing():
    global running
    running = False
    root.destroy()  # Closes the Tkinter window
    plt.close('all') # Closes any open Matplotlib windows
    exit()
    
def brownian_motion(n, m , r, v, M, R, xmax, ymax,kappa, dt, traceMode, title):
        
    rng = np.random.default_rng()

    particle_pos = rng.uniform(low=0, high=[xmax, ymax], size=(n, 2))
    
    particle_angle = rng.uniform(0, 2 * np.pi, size=n)
    
    diffusion_coefficient = 1.0 / (kappa + 1e-9)
    noise = rng.normal(0.0, diffusion_coefficient * np.sqrt(dt), size=n)
    particle_angle += noise
    
    v_array = np.full(n, v)
    
    
    large_particle_pos = np.array([xmax / 2, ymax / 2])
    
    large_particle_v = np.array([0.0,0.0])
    
    large_particle_history = []
    
    count = 1

    # Clean up the plot 
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.ion()
    
    plt.scatter(particle_pos[:, 0], particle_pos[:, 1], s=10)
    
    render_dt = 0.016  # Draw at ~60 FPS
    steps_per_frame = int(render_dt / dt)
    sim_elapsed = 0.0
    start_time = time.perf_counter()
    while running == True:
        for _ in range(steps_per_frame):
            #Check if particles and large particles interact
            distances = np.linalg.norm(particle_pos - large_particle_pos, axis=1)
            inside_indices = np.where(distances < R + r)[0]
            
            for idx in inside_indices:
                # Get vector pointing from large particle to small particle
                relative_vec = particle_pos[idx] - large_particle_pos
                
                if distances[idx] < 1e-9: # Avoid division by zero
                    particle_pos[idx] = large_particle_pos + np.array([R, 0])
                else:
                            #push to just outside the boundary (R + r)
                            buffer = r
                            particle_pos[idx] = large_particle_pos + (relative_vec / distances[idx]) * (R + buffer)
                
                #Convert to vector
                theta = particle_angle[idx] 
                particle_vec_vel_init = v_array[idx]*np.array([np.cos(theta),np.sin(theta)])
                
                relative_pos = large_particle_pos - particle_pos[idx]
                true_distance = np.linalg.norm(relative_pos)

                # Prevent division by zero just in case they are exactly on top of each other
                if true_distance == 0: 
                    true_distance = 0.001

                normal_vec = relative_pos / true_distance
                
                tan_vec = np.array([-normal_vec[1], normal_vec[0]])
                
                #Tracks
                collision_1 = np.dot(particle_vec_vel_init, normal_vec)
                side_1 = np.dot(particle_vec_vel_init, tan_vec)
                
                collision_2 = np.dot(large_particle_v, normal_vec)
                side_2 = np.dot(large_particle_v, tan_vec)
                
                new_collision_1 = ((m - M) * collision_1 + 2 * M * collision_2) / (m + M)
                new_collision_2 = ((M - m) * collision_2 + 2 * m * collision_1) / (m + M)
                
                # Recombine the updated tracks into final [x, y] velocity vectors
                new_particle_v = (new_collision_1 * normal_vec) + (side_1 * tan_vec)
                new_large_v    = (new_collision_2 * normal_vec) + (side_2 * tan_vec)
                
                theta_new = np.arctan2(new_particle_v[1], new_particle_v[0])
                new_particle_speed = np.linalg.norm(new_particle_v)
                
                v_array[idx] = new_particle_speed
                particle_angle[idx] = theta_new
                
                large_particle_v = new_large_v  
                
                    
            #Movement Step
            diffusion_coefficient = 1.0 / (kappa + 1e-9)
            noise = rng.normal(0.0, diffusion_coefficient * np.sqrt(dt), size=n)
            particle_angle += noise 
            particle_angle = (particle_angle + np.pi) % (2 * np.pi) - np.pi
            step_size = v_array * dt
            particle_pos[:, 0] += step_size * np.cos(particle_angle) # Update X
            particle_pos[:, 1] += step_size * np.sin(particle_angle) # Update Y
            
            out_x = (particle_pos[:, 0] < 0) | (particle_pos[:, 0] > xmax)
            particle_angle[out_x] = np.pi - particle_angle[out_x]
            particle_pos[:, 0] = np.clip(particle_pos[:, 0], 0, xmax)

            out_y = (particle_pos[:, 1] < 0) | (particle_pos[:, 1] > ymax)
            particle_angle[out_y] = -particle_angle[out_y]
            particle_pos[:, 1] = np.clip(particle_pos[:, 1], 0, ymax)
            
            # 2. Keep large particle inside the box
            if large_particle_pos[0] - R < 0 or large_particle_pos[0] + R > xmax:
                large_particle_v[0] *= -1
                large_particle_pos[0] = np.clip(large_particle_pos[0], R, xmax - R)
            if large_particle_pos[1] - R < 0 or large_particle_pos[1] + R > ymax:
                large_particle_v[1] *= -1
                large_particle_pos[1] = np.clip(large_particle_pos[1], R, ymax - R)
            
            large_particle_pos += large_particle_v * dt
            sim_elapsed += dt
            
        if traceMode == True:
            count = (count % 2) + 1
            if count == 1:
                large_particle_history.append(large_particle_pos.copy())      
                
        plt.cla() 

        plt.scatter(particle_pos[:, 0], particle_pos[:, 1], s=10)
        
        # Create the circle object
        circle = plt.Circle(large_particle_pos, R, color='red', fill=False)
        
        if traceMode == True and len(large_particle_history) > 1:
            history_array = np.array(large_particle_history)
            plt.plot(history_array[:, 0], history_array[:, 1], color='red', linestyle='-', linewidth=0.8, alpha=0.3)

        # Add it to the plot
        plt.gca().add_patch(circle)

        plt.xlim(0, xmax)
        plt.ylim(0, ymax)
        
        plt.xlabel("X Position (nm)")
        plt.ylabel("Y Position (nm)")
        plt.title(title)
        
        actual_elapsed = time.perf_counter() - start_time
        time_to_wait = sim_elapsed - actual_elapsed
        if time_to_wait > 0:
            plt.pause(time_to_wait)
        else:
            plt.pause(0.001) # Force GUI update even if lagging
            

# --- GUI SETUP ---
root = tk.Tk()
root.title("Brownian Motion GUI")

frame = ttk.Frame(root, padding=10)
frame.grid()

# Variables
n_var = tk.StringVar(value="500")
m_var = tk.StringVar(value="1")
r_var = tk.StringVar(value="0.1")
v_var = tk.StringVar(value="400")
M_var = tk.StringVar(value="10")
R_var = tk.StringVar(value="10")
xmax_var = tk.StringVar(value="100")
ymax_var = tk.StringVar(value="100")
kappa_var = tk.StringVar(value="500")
dt_var = tk.StringVar(value="0.001")
traceMode = tk.BooleanVar(value=True)
pico_second_mode = tk.BooleanVar(value=True)
pico_second_map = tk.StringVar(value="100")

# Inputs

ttk.Label(frame, text="Particle Count").grid(row=5, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=n_var).grid(row=5, column=1, padx=5, pady=2)

ttk.Label(frame, text="Mass of Particles (Daltons)").grid(row=6, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=m_var).grid(row=6, column=1, padx=5, pady=2)

ttk.Label(frame, text="Radius of Particles (nm)").grid(row=7, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=r_var).grid(row=7, column=1, padx=5, pady=2)

ttk.Label(frame, text="Inital Velocity (m/s)").grid(row=8, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=v_var).grid(row=8, column=1, padx=5, pady=2)

ttk.Label(frame, text="Mass of Large Particle (Daltons)").grid(row=9, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=M_var).grid(row=9, column=1, padx=5, pady=2)

ttk.Label(frame, text="Radius of Large Particle (nm)").grid(row=10, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=R_var).grid(row=10, column=1, padx=5, pady=2)

ttk.Label(frame, text="Max Width (nm)").grid(row=11, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=xmax_var).grid(row=11, column=1, padx=5, pady=2)

ttk.Label(frame, text="Max Height (nm)").grid(row=12, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=ymax_var).grid(row=12, column=1, padx=5, pady=2)

ttk.Label(frame, text="Physics Kappa").grid(row=13, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=kappa_var).grid(row=13, column=1, padx=5, pady=2)

ttk.Label(frame, text="Trace Mode").grid(row=14, column=0, sticky="w", padx=5, pady=2)
ttk.Checkbutton(frame, text="On/Off", variable=traceMode).grid(row=14, column=1, sticky="w", padx=5, pady=2)

ttk.Label(frame, text="Picosecond Mode").grid(row=15, column=0, sticky="w", padx=5, pady=2)
ttk.Checkbutton(frame, text="On/Off", variable=pico_second_mode).grid(row=15, column=1, sticky="w", padx=5, pady=2)

ttk.Label(frame, text="Picoseconds").grid(row=16, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=pico_second_map).grid(row=16, column=1, padx=5, pady=2)

if pico_second_mode.get() == True:
    sim_time_map = f"{pico_second_map.get()}ps"
else:
        sim_time_map = "1ns"

title = f"Brownian Motion Simulation, 1s simulation time = {sim_time_map} "

print(title)
ttk.Button(
    frame, 
    text="Run Simulation", 
    command=lambda: brownian_motion(
        int(n_var.get()),
        float(m_var.get()),
        float(r_var.get()),
        float(v_var.get()) * (float(int(pico_second_map.get())/1000)) if pico_second_mode.get() else float(v_var.get()),
        float(M_var.get()),
        float(R_var.get()),
        float(xmax_var.get()),
        float(ymax_var.get()),
        float(kappa_var.get()),
        float(dt_var.get()),
        bool(traceMode.get()),
        f"Brownian Motion Simulation, 1s simulation time = {f'{pico_second_map.get()}ps' if pico_second_mode.get() else '1ns'}"
    )
).grid(row=17, column=0, columnspan=2, pady=10)

# Matplotlib interactive mode
plt.ion()
plt.figure()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()