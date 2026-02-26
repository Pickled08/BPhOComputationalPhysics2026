import numpy as np
import matplotlib.pyplot as plt

class ChargedParticle:
    def __init__(self, charge, mass, position):
        self.charge = charge
        self.mass = mass
        self.position = position

particles = [
    ChargedParticle(charge=1.0, mass=1.0, position=np.array([0.0, 0.0])),

]

def electric_field(xmin, xmax, ymin, ymax, spacing, density):

    # Grid
    x = np.linspace(xmin, xmax, spacing)
    y = np.linspace(ymin, ymax, spacing)
    X, Y = np.meshgrid(x, y)
    

    # Extract components
    U = vectors[0]
    V = vectors[1]

    # Plot
    plt.figure(figsize=(6,6))
    plt.streamplot(X, Y, U, V, color='blue', density=density, linewidth=1)
    plt.quiver(X, Y, U, V, color='red', alpha=0.5)  # optional arrows
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vector Field')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    
xmin, xmax, ymin, ymax, spacing, density = -5, 5, -5, 5, 20, 1.5   

electric_field(xmin, xmax, ymin, ymax, spacing, density)