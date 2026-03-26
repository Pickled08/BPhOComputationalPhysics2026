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
from rocket import rocketGraph

@timer
def graphfunc():
    return rocketGraph()

saveImg = False

import matplotlib.pyplot as plt

def plot():
    plt.ion()  # Turn on interactive mode

    for x, y, label in graphfunc():
        title = ["Height vs Time", "Velocity vs Time", "Acceleration vs Time"]
        yaxis = ["Height (m)", "Velocity (ms^-1)", "Acceleration (ms^-2)"]
        plt.figure()
        plt.plot(x, y, linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel(yaxis[int(label)-1])
        plt.title(title[int(label)-1])

        print(f"Plotted: {label}")

        if saveImg:
            plt.savefig(f'graph-output-{label}.png', dpi=300)

    # Keep all figures open until the user closes them manually
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show all figures at once

if __name__ == "__main__":
    plot()
    


