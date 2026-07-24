import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import tkinter as tk

EPISLON_0 = scipy.constants.epsilon_0
H_PLANCK = scipy.constants.h
C = scipy.constants.c
M_ELECTRON = scipy.constants.m_e
E_CHARGE = scipy.constants.e

Z = 1  # Atomic number for hydrogen

def photon_lamda(n, m):
    result = ((8 * (EPISLON_0**2) * (H_PLANCK**3) * C) / (M_ELECTRON * (Z**2) * (E_CHARGE**4))) / ((1 / (m**2)) - (1 / (n**2)))
    return result

def photon_energy(lamda): #eV
    result = (H_PLANCK * C) / (lamda * E_CHARGE)
    return result

def plot_hydrogen_emission_spectrum(n_max):
    plt.close("all")
    #Lyman
    m = 1
    n_values = np.arange(2, n_max)
    lamda_values = photon_lamda(n_values, m)
    energy_values = photon_energy(lamda_values)
    plt.scatter(lamda_values * 1e9, energy_values, color="magenta", marker=".", label="Lyman Series")
    y_max = plt.gca().get_ylim()[1]
    plt.vlines(lamda_values * 1e9, 0, y_max, color="magenta", linewidth=0.2, linestyle="--")
    
    #Balmer
    m = 2
    n_values = np.arange(3, n_max)
    lamda_values = photon_lamda(n_values, m)
    energy_values = photon_energy(lamda_values)
    plt.scatter(lamda_values * 1e9, energy_values, color="red", marker=".", label="Balmer Series")
    plt.vlines(lamda_values * 1e9, 0, y_max, color="red", linewidth=0.2, linestyle="--")
    
    #Paschen
    m = 3
    n_values = np.arange(4, n_max)
    lamda_values = photon_lamda(n_values, m)
    energy_values = photon_energy(lamda_values)
    plt.scatter(lamda_values * 1e9, energy_values, color="blue", marker=".", label="Paschen Series")
    plt.vlines(lamda_values * 1e9, 0, y_max, color="blue", linewidth=0.2, linestyle="--")
    
    #Brackett
    m = 4
    n_values = np.arange(5, n_max)
    lamda_values = photon_lamda(n_values, m)
    energy_values = photon_energy(lamda_values)
    plt.scatter(lamda_values * 1e9, energy_values, color="lime", marker=".", label="Brackett Series")
    plt.vlines(lamda_values * 1e9, 0, y_max, color="lime", linewidth=0.2, linestyle="--")
    
    #Pfund
    m = 5
    n_values = np.arange(6, n_max)
    lamda_values = photon_lamda(n_values, m)
    energy_values = photon_energy(lamda_values)
    plt.scatter(lamda_values * 1e9, energy_values, color="black", marker=".", label="Pfund Series")
    plt.vlines(lamda_values * 1e9, 0, y_max, color="black", linewidth=0.2, linestyle="--")

    plt.title("Bohr Model of Hydrogen Emission Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Photon Energy (eV)")
    plt.legend()
    
    plt.show()
    
root = tk.Tk()

tk.Label(root, text="Bohr Model of Hydrogen Emission Spectrum").pack()
tk.Label(root, text="Enter the maximum value of n (n > 1):").pack()
n_max_entry = tk.Entry(root)
n_max_entry.pack()

def hes_plot():
    n_max = int(n_max_entry.get())
    plot_hydrogen_emission_spectrum(n_max)

tk.Button(root, text="Plot Spectrum", command=hes_plot).pack()

def on_closing():
    global running
    running = False
    root.destroy()  # Closes the Tkinter window
    plt.close("all") # Closes any open Matplotlib windows
    exit()

root.mainloop()