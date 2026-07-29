import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

H_PLANCK = scipy.constants.h
C = scipy.constants.c
M_ELECTRON = scipy.constants.m_e

e_CHARGE = scipy.constants.e

HC = H_PLANCK * C

def plot_electron_recoil_speed_vs_theta(photon_energies_eV):
    theta = np.linspace(1e-5, 180, 1000)
    theta_rad = np.radians(theta)
    lamda = (H_PLANCK * C) / (photon_energies_eV * e_CHARGE)
    dlamda = ((H_PLANCK)/(M_ELECTRON*C))*(1 - np.cos(theta_rad))
    for i, w in enumerate(lamda):
        lamda_prime = w + dlamda
        v = C * np.sqrt(1 - (((M_ELECTRON * C * C)/( (HC/w) - (HC/lamda_prime) + (M_ELECTRON * C * C) ))**2) )
        v_c = v / C
        plt.plot(theta, v_c, label=f'Photon Energy: {photon_energies_eV[i]} eV')
        
    plt.title('Electron Recoil Speed vs Scattering Angle')
    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Electron Recoil Speed (v/c)')
    plt.legend()
    plt.grid()
    plt.show()
    

   
if __name__ == "__main__":
    photon_energies_eV = np.array([50000, 100000, 200000, 500000, 1000000])  # Photon energies in eV
    plot_electron_recoil_speed_vs_theta(photon_energies_eV) 