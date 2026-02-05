import matplotlib.pyplot as plt
import numpy as np
import time
from numba import njit

#Constants
C=299792458 # Speed of light in m/s

resolution = 1000
v = np.linspace(0, C, resolution) # Velocity array from 0 to speed of light

@njit(fastmath=True)
def lorentzfactor():
    xarr = np.empty(resolution)
    yarr = np.empty(resolution)
    for i in range(resolution):
        lf = 1/(np.sqrt(1-((v[i]**2)/(C**2))))

        xarr[i] = v[i]
        yarr[i] = lf
    
    return xarr, yarr

def plot():
    x, y = lorentzfactor()
    plt.plot(x,y)
    plt.show()

plot()
