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
    
    v_array = np.full(n, v)
    
    
    large_particle_pos = np.array([xmax / 2, ymax / 2])
    
    large_particle_v = np.array([0.0,0.0])

    # Clean up the plot
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Particle Positions")
    plt.ion()
    
    plt.scatter(particle_pos[:, 0], particle_pos[:, 1], s=10)
    
    render_dt = 0.016  # Draw at ~60 FPS
    steps_per_frame = int(render_dt / dt)
    sim_elapsed = 0.0
    start_time = time.perf_counter()
    while True:
        for _ in range(steps_per_frame):
            #Check if particles and large particles interact
            distances = np.linalg.norm(particle_pos - large_particle_pos, axis=1)
            inside_indices = np.where(distances < R)[0]
            
            for idx in inside_indices:
                # Get vector pointing from large particle to small particle
                relative_vec = particle_pos[idx] - large_particle_pos
                
                if distances[idx] < 1e-9: # Avoid division by zero
                    particle_pos[idx] = large_particle_pos + np.array([R, 0])
                else:
                            #push to just outside the boundary (R + a tiny buffer)
                            buffer = 0.1 
                            particle_pos[idx] = large_particle_pos + (relative_vec / distances[idx]) * (R + buffer)
                            
                            # reflect the angle
                            normal_angle = np.arctan2(relative_vec[1], relative_vec[0])
                            particle_angle[idx] = 2 * normal_angle - particle_angle[idx]
                            
                #Get velocity vector
                theta = particle_angle[idx]
                particle_vec_v = np.array([np.cos(theta),np.sin(theta)])*v_array[idx]
                
                collision_vector = large_particle_pos - particle_pos[idx]
                normal = collision_vector / np.linalg.norm(collision_vector)

                # 2. Relative velocity vector
                relative_v = particle_vec_v - large_particle_v

                # 3. Calculate scalar velocity along the normal
                # (Only the component of velocity along this line changes)
                v_along_normal = np.dot(relative_v, normal)

                # Do not process if particles are already moving apart
                if v_along_normal > 0:
                    # 4. Calculate impulse scalar (Elastic Collision)
                    # Total Momentum Change = (2 * v_normal) / (1/m + 1/M)
                    impulse = (2 * v_along_normal) / (1/m + 1/M)

                    # 5. Apply the impulse to update velocities
                    # Particle loses momentum, Large Particle gains it
                    particle_vec_v -= (impulse / m) * normal
                    large_particle_v += (impulse / M) * normal
                    
                v_array[idx] = np.linalg.norm(particle_vec_v)
                
                
                    
            #Movement Step
            noise = rng.normal(0.0, 1.0 / (kappa + 1e-9), size=n)
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
            
        plt.cla() 

        plt.scatter(particle_pos[:, 0], particle_pos[:, 1], s=10)
        
        # Create the circle object
        circle = plt.Circle(large_particle_pos, R, color='red', fill=False)

        # Add it to the plot
        plt.gca().add_patch(circle)

        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        print(np.mean(v_array))
        
        actual_elapsed = time.perf_counter() - start_time
        time_to_wait = sim_elapsed - actual_elapsed
        if time_to_wait > 0:
            plt.pause(time_to_wait)
        else:
            plt.pause(0.001) # Force GUI update even if lagging
            

#Defaults
n=500
m=1
r=1
v=200
M=0.5
R=10
xmax=100
ymax=100
kappa = 5
dt=0.001

brownian_motion(n, m , r, v, M, R, xmax, ymax, kappa, dt)
    