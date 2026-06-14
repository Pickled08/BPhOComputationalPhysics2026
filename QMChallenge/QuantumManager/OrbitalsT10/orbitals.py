import numpy as np
import matplotlib.pyplot as plt
import scipy
from numba import njit

#Universal Constants
PERMITTIVITY_FREE_SPACE = scipy.constants.epsilon_0
ELEMENTARY_CHARGE = scipy.constants.e
PLANCK_CONSTANT = scipy.constants.h
REDUCED_PLANCK_CONSTANT = scipy.constants.hbar
ELECTRON_MASS = scipy.constants.m_e
PI = np.pi
EULER_NUM = np.e
ATOMIC_MASS_UNIT = scipy.constants.u
BOHR_RADIUS = scipy.constants.physical_constants['Bohr radius'][0]

#input parameters
Z = 1  # Atomic number (Hydrogen)
A = 1  # Mass number
n = 2  # Principal quantum number
l = 0  # Azimuthal quantum number
m = 0 # Magnetic quantum number


#Start of Computation

M = A * ATOMIC_MASS_UNIT  # in kg

reduced_mass = (ELECTRON_MASS * M) / (ELECTRON_MASS + M)

hydrogenic_atomic_radius = BOHR_RADIUS * (ELECTRON_MASS / reduced_mass) / Z

def laguer_polynomial(x, n, l):
    
    laguer_poly=sum( ( ((scipy.special.factorial((l + n), exact=True))*((-x)**k))/((scipy.special.factorial((2*l + 1 + k), exact=True))*(scipy.special.factorial((n - l - 1 - k), exact=True)) * scipy.special.factorial((k), exact=True)) ) for k in range(0, n - l))
    
    return laguer_poly

def radial_wavefunction(r, n, l):
    
    x = (2 * r) / (hydrogenic_atomic_radius * n)
    
    laguer_poly = laguer_polynomial(x, n, l)
    
    normalization = np.sqrt(
            scipy.special.factorial((n - l - 1), exact=True) / 
            (2 * n * scipy.special.factorial((n + l), exact=True))
        ) * ((2 / (hydrogenic_atomic_radius * n))**(1.5))
        
    radial = normalization * (x**l) * np.exp(-x / 2) * laguer_poly
    return radial

def spharm(theta, phi, l, m):
    abs_m = abs(m)
    y = scipy.special.lpmv(abs_m, l, np.cos(theta))
    
    normalization = ((-1)**abs_m) * (
        ((2*l + 1) / (4 * PI)) *
        (scipy.special.factorial(l - abs_m, exact=True) /
         scipy.special.factorial(l + abs_m, exact=True))
    ) ** 0.5
    
    if m >= 0:
        Y = normalization * y * np.exp(1j * m * phi)
    else:
        Y = normalization * y * np.exp(1j * m * phi) * ((-1)**m)

    return Y

def angular_wavefunction(theta, phi, l, m):
    if m > 0:
        Y = (spharm(theta, phi, l, -m) + ((-1)**m) * spharm(theta, phi, l, m)) / np.sqrt(2)
    elif m < 0:
        Y = (1j / np.sqrt(2)) * (spharm(theta, phi, l, m) - ((-1)**m) * spharm(theta, phi, l, -m))
    else:
        Y = spharm(theta, phi, l, 0)
    
    return np.real(Y)
    
def wavefunction(r, theta, phi):
    global n, l, m
    
    psi = radial_wavefunction(r, n, l)*angular_wavefunction(theta, phi, l, m)
    
    return psi

def plot_linear_probability_density(r, n, l):
    radial_values = radial_wavefunction(r, n, l)
    
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

#Plot probability density vs x,y as color map
def plot_probability_density_2d(n, l, m):
    # Build a Cartesian grid directly
    extent = 25 * hydrogenic_atomic_radius
    num_points = 2000
    
    x_vals = np.linspace(-extent, extent, num_points)
    z_vals = np.linspace(-extent, extent, num_points)
    X, Z = np.meshgrid(x_vals, z_vals)
    
    # Convert each Cartesian point to spherical coordinates
    R = np.sqrt(X**2 + Z**2)
    THETA = np.arctan2(np.sqrt(X**2), Z)  # polar angle from z-axis
    PHI = np.ones_like(R) * 0 # azimuthal angle in x-y plane
    
    # Avoid r=0 singularity
    R = np.where(R == 0, 1e-20, R)
    
    # Evaluate wavefunction on the Cartesian grid
    psi = wavefunction(R, THETA, PHI)
    probability_density = np.abs(psi)**2

    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(X, Z, probability_density, shading='auto', cmap='hot')
    plt.colorbar(mesh, ax=ax, label='Probability Density')
    ax.set_title(f'Probability Density (XZ cross-section) n={n}, l={l}, m={m}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
   
plot_probability_density_2d(n, l, m)
   
    
    