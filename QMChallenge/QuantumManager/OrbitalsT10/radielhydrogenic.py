import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as sp


#Universal Constants
PERMITTIVITY_FREE_SPACE = scipy.constants.epsilon_0
ELEMENTARY_CHARGE = scipy.constants.e
PLANCK_CONSTANT = scipy.constants.h
REDUCED_PLANCK_CONSTANT = scipy.constants.hbar
ELECTRON_MASS = scipy.constants.m_e
PI = np.pi
ATOMIC_MASS_UNIT = scipy.constants.u
BOHR_RADIUS = scipy.constants.physical_constants['Bohr radius'][0]

#input parameters
Z = 6  # Atomic number for hydrogen
A = 12  # Mass number (for hydrogen, A=1)
n = 4  # Principal quantum number
l = 2  # Azimuthal quantum number




#Start of Computation

M = A * ATOMIC_MASS_UNIT  # in kg

reduced_mass = (ELECTRON_MASS * M) / (ELECTRON_MASS + M)

hydrogenic_atomic_radius = BOHR_RADIUS * (ELECTRON_MASS / reduced_mass) / Z

r = np.linspace(0, 25 * hydrogenic_atomic_radius, 1000)

def laguer_polynomial(x, n, l):
    
    laguer_poly=sum( ( ((scipy.special.factorial((l + n), exact=True))*((-x)**k))/((scipy.special.factorial((2*l + 1 + k), exact=True))*(scipy.special.factorial((n - l - 1 - k), exact=True)) * scipy.special.factorial((k), exact=True)) ) for k in range(0, n - l))
    
    return laguer_poly

def radial_hydrogenic(r, n, l):
    
    x = (2 * r) / (hydrogenic_atomic_radius * n)
    
    laguer_poly = laguer_polynomial(x, n, l)
    
    normalization = np.sqrt(
            scipy.special.factorial((n - l - 1), exact=True) / 
            (2 * n * scipy.special.factorial((n + l), exact=True))
        ) * ((2 / (hydrogenic_atomic_radius * n))**(1.5))
        
    radial = normalization * (x**l) * np.exp(-x / 2) * laguer_poly
    return radial
    

def plot_radial_wavefunction(r, n, l):
    radial_values = radial_hydrogenic(r, n, l)
    plt.plot(r, radial_values)
    plt.title(f"Radial Wavefunction for n={n}, l={l}")
    plt.xlabel("r (m)")
    plt.ylabel("R(r)")
    plt.grid()
    plt.show()


def plot_linear_probability_density(r, n, l):
    radial_values = radial_hydrogenic(r, n, l)
    
    density_at_a_point = radial_values**2
    
    # Convert meters to Angstroms for the x-axis plot
    r_angstroms = r / 1e-10
    
    plt.figure(figsize=(7, 5.5))
    plt.plot(r_angstroms, density_at_a_point, color='red', linewidth=2)
    plt.title(f"Hydrogenic atom: Z={Z}, A={A}, n={n}, L={l}")
    plt.xlabel("radius /Angstroms")
    plt.ylabel("Probability density")
    plt.xlim(0, 4) 
    plt.grid(True, color='lightgray')
    plt.show()

# Adjust your linspace range to cleanly cover up to 4 Angstroms
r = np.linspace(0, 4e-10, 1000)
plot_linear_probability_density(r, n, l)