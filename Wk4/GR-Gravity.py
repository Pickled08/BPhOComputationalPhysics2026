import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  

from matplotlib.animation import FuncAnimation

# Constants
G = 6.674e-11
m_target = 1.989e30 # mass creating the field

target_pos = np.array([0.0, 0.0])


# Grid
x = np.linspace(-20, 20, 40)
y = np.linspace(-20, 20, 40)
X, Y = np.meshgrid(x, y)

Z = -G * m_target / np.sqrt(X**2 + Y**2 + 1e-3)
    
# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title("Gravity Field of a 2D Point Mass")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Lock axis scales
ax.set_box_aspect([1, 1, 1])  # Make X, Y, Z same scale

fig.colorbar(surf, shrink=0.5, aspect=10)

def update(frame):
    ax.clear()  # clear previous surface
    
    # Move the mass in a circle
    target_pos = np.array([10 * np.cos(frame/10), 10 * np.sin(frame/10)])
    
    Z = -G * m_target / np.sqrt((X - target_pos[0])**2 + (Y - target_pos[1])**2 + 1e-3)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    
    ax.set_title("Gravity Field of a 2D Point Mass")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential')
    ax.set_box_aspect([1,1,1])
    
    return surf,

# Create animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()



