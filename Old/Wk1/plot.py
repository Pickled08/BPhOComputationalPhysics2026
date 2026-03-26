import numpy as np # Numpy: Math functions and arrays 
import matplotlib.pyplot as plt # Matplotlib: plotting
import time

#Time your function
def timer(func):
    def wrapper_function(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func} Took: {(end - start) * 1_000_000:.2f} microseconds")
        return result
    return wrapper_function

#Function you wish to graph
from randomwalknumba import NRandomWalks

@timer
def graphfunc():
    return NRandomWalks(100, 0)

saveImg = False

@timer
def plot():

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("2D Random Walk")

    for x, y, label in graphfunc():

        RGB = np.random.uniform(0, 1, 3)  # (r, g, b)
        plt.plot(x,y,color=RGB,linewidth = 1, label=label) # Plot the Graph
        print(f"Plotted: {label}")

    if saveImg:
        plt.savefig('graph-output.png', dpi = 300)

    

if __name__ == "__main__":
    plot()
    plt.show()


