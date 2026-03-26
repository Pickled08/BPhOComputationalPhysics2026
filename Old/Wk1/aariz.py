import numpy as np #numbers
import matplotlib.pyplot as plt #plotting
import time
import random

# This is a random walk demo. Take a fixed step on L metres.
# Between each step change direction randomly by an angle between 0 and 2pi radians.
# After N steps plot the path taken. This model can be used to model diffusion or gas particles.

#Time your function
def timer(func):
    def wrapper_function(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Took: {(end - start) * 1_000_000:.2f} microseconds")
        return result
    return wrapper_function


def randomwalk (L,N):
    x = 0
    y = 0
    x_positions = [x]
    y_positions = [y]


    for i in range (int(N)):
        theta = random.uniform(0,2*3.141592653589793) # Random angle between 0 and 2pi radians
        dx = np.cos(theta)*L # Change in x position
        dy = np.sin(theta)*L # Change in y position
        x = x + dx # New x position
        y = y + dy # New y position
        x_positions.append(x) # Append new x position to list
        y_positions.append(y) # Append new y position to list
    return [x_positions, y_positions]



@timer
def random_walk_demo(): # Main function to run the random walk demo.
 
    N = 1000000 # Total number of steps taken.
    L = 1.0 # Size of each step in metres.
    m = 10 # Number of random walks to simulate.
    for i in range(int(m)):
    
        x,y = randomwalk (L,N)
        RGB = np.random.uniform(0,1,3)

        plt.plot(x,y,color=RGB,linewidth = 1) # Plot the path taken in a blue line

    #plt.savefig('rwg.png', dpi = 300)


    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("2D Random Walk")
    plt.show() # Show the plot

random_walk_demo()
