import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_current_V(wavelength, work_function, intensity, quantum_efficiency, stopping_potential, intensity_lamp):
    if work_function is None:
        print("Error: Work function is not defined.")
        return
    #draw linear part between 0 and stopping potential, then current is 0 after stopping potential
    if wavelength > 0 and intensity > 0:
        photons_per_second = (intensity_lamp * (intensity / 100.0)) / (6.626e-34 * 3e8 / (wavelength * 1e-9))
        photons_per_second_displayed = photons_per_second / 1e16
        interval = 1.0 / photons_per_second_displayed if photons_per_second_displayed > 0 else float('inf')
    else:
        interval = float('inf')
    electron_charge = 1.6021766e-19
    electrons_per_second = photons_per_second * (1 if work_function is not None and (6.6260702e-34 * (299792458 / (wavelength * 1e-9))) >= (work_function * electron_charge) else 0) * quantum_efficiency
    current_max = electrons_per_second * electron_charge
    current = current_max
    voltages = np.linspace(-10, 10, 200)
    #For voltages between 0 and stopping potential, current decreases linearly from current_max to 0. For voltages above stopping potential, current is 0. For voltages below 0, current is current_max.
    currents = []
    for voltage in voltages:
        if voltage < 0:
            currents.append(current_max)
        elif stopping_potential is not None and voltage >= stopping_potential:
            currents.append(0.0)
        else:
            currents.append((-current_max/stopping_potential * voltage) + current_max)
            
    mpl.rcParams['toolbar'] = 'None'
    
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, currents, label='Current vs Voltage', color='blue')
    #draw y axis at 0
    plt.axvline(0, color='black', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('Photoelectric Effect: Current vs Voltage \nWork Function = {:.2f} eV'.format(work_function))
    plt.legend()
    plt.grid()
    plt.xlim(-10, 10)
    plt.ylim(-0.01, current_max + 0.01)
    plt.show()
    
if __name__ == "__main__":
    plot_current_V(100, 2.0, 100, 0.1, 1.5, 1)  # Example parameters: wavelength in nm, work function in eV, photons per second, quantum efficiency, stopping potential in V, intensity of the light source