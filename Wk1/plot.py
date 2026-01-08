import math
import numpy as np
import numbers
import time
from functools import wraps
from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 

from randomwalk import NRadomWalks

window = Tk()

def timer(func):
    def wrapper_function(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Took: {(end - start) * 1_000_000:.2f} microseconds")
        return result
    return wrapper_function


#EXAMPLE graph function
#def graphfunc():
#    x = numpy.linspace(-10, 10, 1000)
#
#    y1 = numpy.sin(x)
#    y2 = numpy.cos(x)
#    y3 = x ** 2
#
#    return [
#        (x, y1, "sin(x)"),
#        (x, y2, "cos(x)"),
#        (x, y3, "x²"),
#    ]
#

def graphfunc():
    return NRadomWalks(10)

print("Graphing Module Loaded")

def plotOnGraph():
    fig = Figure(figsize=(20, 6.8), dpi=100)
    ax = fig.add_subplot(111)

    for x, y, label in graphfunc():
        ax.plot(x, y, label=label)

    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()


@timer
def main():
    print("Loading...")
    plotOnGraph()


if __name__ == "__main__":
    window.title("Maths and Numbers Software")
    window.configure(background="black")
    window.maxsize(1280, 720)
    window.minsize(1280,720)

    main() 

    window.mainloop()