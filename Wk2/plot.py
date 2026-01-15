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
from bounce import bounceGraph

@timer
def graphfunc():
    return bounceGraph()

saveImg = False

@timer
def plot():

    plt.xlabel("Time (s)")
    plt.ylabel("Hight (m)")
    plt.title("2D Random Walk")

    for x, y, label in graphfunc():

        plt.plot(x,y,linewidth = 1, label=label) # Plot the Graph
        print(f"Plotted: {label}")

    if saveImg:
        plt.savefig('graph-output.png', dpi = 300)
    plt.show()

    

if __name__ == "__main__":
    plot()
    


