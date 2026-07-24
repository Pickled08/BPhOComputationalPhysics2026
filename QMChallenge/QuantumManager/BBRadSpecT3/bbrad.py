import tkinter as tk
from tkinter import ttk
from blamdat import plot_blackbody_spectrum
from molarhc import plot_molar_heat_capacity

fE_values = [0.2855e13,0.5769e13,0.7054e13,0.7188e13,0.7893e13,1.0832e13,3.7451e13]
elements = ["Au","Cu","Ti","Al","Fe","Si","C"]

T_values = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

root = tk.Tk()

# Setting some window properties
root.title("Blackbody Radiation and Molar Heat Capacity")
root.configure(background="white")
#Style
style = ttk.Style()
style.configure("TButton",
                font=("Arial", 12),
                padding=6)

#GUI interface to launch and confingure each program from
ttk.Label(root, text="Blackbody Radiation and Molar Heat Capacity").pack()

#Buttons to launch each program
ttk.Button(
    root,
    text="Plot Blackbody Spectrum",
    command=lambda: plot_blackbody_spectrum(T_values)
).pack(pady=10)

ttk.Button(
    root,
    text="Plot Molar Heat Capacity",
    command=lambda: plot_molar_heat_capacity(fE_values, elements)
).pack(pady=10)

root.mainloop()
