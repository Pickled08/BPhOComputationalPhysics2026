import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

SPEED_OF_LIGHT = scipy.constants.c
H_PLANCK = scipy.constants.h
KB = scipy.constants.k
R = scipy.constants.R

def plot_molar_heat_capacity(fE_values, elements):
    plt.close("all")
    for fE in fE_values:
        T = np.linspace(1, 1000, 1000)
        x = (H_PLANCK*fE)/(KB*T)
        C = 3 * R * (((x**2)*np.exp(x))/((np.exp(x)-1)**2))
        
        #plot and add label for each material
        plt.plot(T, C, label=f"{elements[fE_values.index(fE)]} (fᴇ={fE:.2e} Hz)")
    plt.title("Molar Heat Capacity vs Temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Molar Heat Capacity (J/mol·K)")
    plt.legend()
    plt.show()