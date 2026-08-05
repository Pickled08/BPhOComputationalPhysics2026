import tkinter as tk
from tkinter import ttk
import numpy as np
from electronrecoilangle_theta import plot_electron_recoil_angle_vs_theta
from electronrecoilspeed_theta import plot_electron_recoil_speed_vs_theta
from wavelengthratio_theta import plot_wavelength_ratio_vs_theta


photon_energies_eV = np.array([50000, 100000, 200000, 500000, 1000000])
root = tk.Tk()

# Setting some window properties
root.title("Compton Scattering Analysis")
root.configure(background="white")
#Style
style = ttk.Style()
style.configure("TButton",
                font=("Arial", 12),
                padding=6)

#GUI interface to launch and confingure each program from
ttk.Label(root, text="Compton Scattering Analysis").pack()

#Buttons to launch each program
ttk.Button(
    root,
    text="Plot Electron Recoil Angle",
    command=lambda: plot_electron_recoil_angle_vs_theta(photon_energies_eV)
).pack(pady=10)

ttk.Button(
    root,
    text="Plot Electron Recoil Speed",
    command=lambda: plot_electron_recoil_speed_vs_theta(photon_energies_eV)
).pack(pady=10)

ttk.Button(
    root,
    text="Plot Wavelength Ratio",
    command=lambda: plot_wavelength_ratio_vs_theta(photon_energies_eV)
).pack(pady=10)

def add_custom_photon_energy():
    custom_energy = float(custom_energy_entry.get())
    global photon_energies_eV
    photon_energies_eV = np.append(photon_energies_eV, custom_energy)
    custom_energy_entry.delete(0, tk.END)
    

#Add custom value label and entry field
ttk.Label(root, text="Enter Custom Photon Energy (eV):").pack(pady=10)

custom_energy_entry = ttk.Entry(root)
custom_energy_entry.pack(pady=10)

ttk.Button(
    root,
    text="Add",
    command=add_custom_photon_energy
).pack(pady=10)

root.mainloop()
