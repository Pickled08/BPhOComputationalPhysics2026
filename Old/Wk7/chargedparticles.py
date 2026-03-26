import numpy as np
import matplotlib.pyplot as plt

pfs = 8.85e-12  # Permittivity of free space
k = 1 / (4 * np.pi * pfs)  # Coulomb's constant

# Define a class for charged particles
class ChargedParticle:
    def __init__(self, charge, mass, position):
        self.charge = charge
        self.mass = mass
        self.position = position

particles = [
    ChargedParticle(charge=1.0, mass=1.0, position=np.array([0.0, 0.0])), ChargedParticle(charge=-1.0, mass=1.0, position=np.array([2.0, 0.0])), ChargedParticle(charge=1.0, mass=1.0, position=np.array([0.0, 2.0])), ChargedParticle(charge=-1.0, mass=1.0, position=np.array([-2.0, 0.0])), ChargedParticle(charge=1.0, mass=1.0, position=np.array([0.0, -2.0]))

]

def electric_field(xmin, xmax, ymin, ymax, grid_pacing, line_density):

    # Grid setup
    x = np.linspace(xmin, xmax, grid_spacing)
    y = np.linspace(ymin, ymax, grid_spacing)
    X, Y = np.meshgrid(x, y)

    for particle in particles:
        r = np.sqrt((X - particle.position[0])**2 + (Y - particle.position[1])**2) # Distance from particle to each point in the grid
        E = k * particle.charge / r**2 # Electric field magnitude from the particle
        Ex = E * (X - particle.position[0]) / r # Electric field component in x direction
        Ey = E * (Y - particle.position[1]) / r # Electric field component in y direction

        # Update the total electric field
        if 'total_Ex' in locals(): # Check if total_Ex and total_Ey already exist
            total_Ex += Ex
            total_Ey += Ey
        else:
            total_Ex = Ex
            total_Ey = Ey

    U = total_Ex
    V = total_Ey

    # Plot
    plt.figure(figsize=(6,6))
    #Draw Particles
    for particle in particles:
        plt.scatter(particle.position[0], particle.position[1], color='red' if particle.charge > 0 else 'blue', s=100)
    #Draw Field Lines
    plt.streamplot(X, Y, U, V, color='blue', density=line_density, linewidth=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vector Field')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    
xmin, xmax, ymin, ymax, grid_spacing, line_density = -10, 10, -10, 10, 500, 3

electric_field(xmin, xmax, ymin, ymax, grid_spacing, line_density)
