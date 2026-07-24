import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_SP_F(workfunction):
    if workfunction is None:
        print("Error: Work function is not defined.")
        return
    
    frequencies = np.linspace(0, 2e15, 200)  # Frequency range from 10^14 to 10^15 Hz
    photon_energy = frequencies * 6.62607015e-34
    photon_energy_eV = photon_energy / 1.60217663e-19
    electron_kinetic_energy = photon_energy_eV - workfunction
    stopping_potential = electron_kinetic_energy

    y_above = np.ma.masked_where(stopping_potential < 0, stopping_potential)
    y_below = np.ma.masked_where(stopping_potential >= 0, stopping_potential)

    #Visible Light Range Freq
    visible_red = 4.3e14
    visible_orange = 5e14
    visible_yellow = 5.3e14
    visible_green = 6e14
    visible_blue = 6.5e14
    visible_indigo = 7e14
    visible_violet = 7.5e14

    mpl.rcParams["toolbar"] = "None"

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, y_above, label="Stopping Potential vs Frequency", color="blue")
    plt.plot(frequencies, y_below, color="blue", linestyle="--")
    plt.axhline(0, color="red", linestyle="--", label="Zero Stopping Potential")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Stopping Potential (V)")
    plt.title("Photoelectric Effect: Stopping Potential vs Frequency \nWork Function = {:.2f} eV".format(workfunction))

    plt.vlines(visible_red, -100, 100, colors="red", linewidth=0.5)
    plt.vlines(visible_orange, -100, 100, colors="orange", linewidth=0.5)
    plt.vlines(visible_yellow, -100, 100, colors="yellow", linewidth=0.5)
    plt.vlines(visible_green, -100, 100, colors="green", linewidth=0.5)
    plt.vlines(visible_blue, -100, 100, colors="blue", linewidth=0.5)
    plt.vlines(visible_indigo, -100, 100, colors="#4B0082", linewidth=0.5)
    plt.vlines(visible_violet, -100, 100, colors="#8A2BE2", linewidth=0.5)
    
    plt.legend()
    plt.grid()
    plt.xlim(1e14, 2e15)
    plt.ylim(-0.5, max(stopping_potential) + 0.5)
    plt.show()
    
if __name__ == "__main__":
    plot_SP_F(2.0)  # Example work function in eV