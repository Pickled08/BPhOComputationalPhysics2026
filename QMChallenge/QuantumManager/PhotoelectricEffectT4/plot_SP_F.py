import matplotlib.pyplot as plt
import numpy as np

def plot_SP_F(workfunction):
    if workfunction is None:
        print("Error: Work function is not defined.")
        return
    # Sample data for demonstration
    frequencies = np.linspace(0, 2e15, 200)  # Frequency range from 10^14 to 10^15 Hz
    photon_energy = frequencies * 6.62607015e-34
    photon_energy_eV = photon_energy / 1.60217663e-19
    electron_kinetic_energy = photon_energy_eV - workfunction
    stopping_potential = electron_kinetic_energy

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, stopping_potential, label='Stopping Potential vs Frequency', color='blue')
    plt.axhline(0, color='red', linestyle='--', label='Zero Stopping Potential')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Stopping Potential (V)')
    plt.title('Photoelectric Effect: Stopping Potential vs Frequency \nWork Function = {:.2f} eV'.format(workfunction))
    plt.legend()
    plt.grid()
    plt.xlim(1e14, 1e15)
    plt.ylim(-0.5, max(stopping_potential) + 0.5)
    plt.show()
    
if __name__ == "__main__":
    plot_SP_F(2.0)  # Example work function in eV