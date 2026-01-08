import math
import numpy
import numbers
import time
from functools import wraps
from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 

window = Tk()

def timer(func):
    def wrapper_function(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Took: {(end - start) * 1_000_000:.2f} microseconds")
        return result
    return wrapper_function




def graphfunc():
    x = [0,1,1,0,0]
    y = [1,1,0,0,1]
    return x, y

print("Graphing Module Loaded")

def plot(): 
  
    # the figure that will contain the plot 
    fig = Figure(figsize = (20, 6.8), 
                 dpi = 100) 
  
    #
    x, y = graphfunc()
    plot1 = fig.add_subplot(111)
    plot1.plot(x, y)
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                               master = window)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   window) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack() 
@timer
def main():
    print("Loading...")
    plot()


if __name__ == "__main__":
    window.title("Maths and Numbers Software")
    window.configure(background="black")
    window.maxsize(1280, 720)
    window.minsize(1280,720)

    main() 

    window.mainloop()