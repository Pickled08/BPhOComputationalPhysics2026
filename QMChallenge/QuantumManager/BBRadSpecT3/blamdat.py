import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

SPEED_OF_LIGHT = scipy.constants.c
H_PLANCK = scipy.constants.h
KB = scipy.constants.k

def plot_blackbody_spectrum(T_values):
    plt.close("all")
    lamda = np.linspace(1e-9, 2e-6, 1000)
    for T in T_values:
        B = (2 * H_PLANCK * SPEED_OF_LIGHT**2) / (lamda**5 * (np.exp((H_PLANCK * SPEED_OF_LIGHT) / (lamda * KB * T)) - 1))
        plt.plot(lamda * 1e9, B, label=f"T={T}K")
        
    plt.title("Blackbody Radiation Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral Radiance (W m⁻² sr⁻¹ Hz⁻¹)")
    plt.legend()
    plt.show()

T_values = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]