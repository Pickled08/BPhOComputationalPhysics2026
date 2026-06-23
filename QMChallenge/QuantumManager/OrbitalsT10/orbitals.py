import numpy as np
import matplotlib.pyplot as plt
import scipy
from numba import njit
import pyvista as pv

import tkinter as tk
from tkinter import ttk
import threading

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

#Setup input GUI 

root = tk.Tk()
root.title("Orbital")
root.geometry("500x300")

Z_inp = tk.DoubleVar()
A_inp = tk.DoubleVar()
n_inp = tk.DoubleVar()
l_inp = tk.DoubleVar()
m_inp = tk.DoubleVar()

# Set default float values
Z_inp.set(1.0)  # Atomic number
A_inp.set(1.0)  # Mass number
n_inp.set(5.0)  # Principal quantum number
l_inp.set(4.0)  # Azimuthal quantum number
m_inp.set(0.0)  # Magnetic quantum number

#Calculate orbital Shape

def calculate_radius():

    A = float(A_inp.get())
    Z = float(Z_inp.get())


    M = A * ATOMIC_MASS_UNIT  # in kg

    reduced_mass = (ELECTRON_MASS * M) / (ELECTRON_MASS + M)

    hydrogenic_atomic_radius = BOHR_RADIUS * (ELECTRON_MASS / reduced_mass) / Z

    return hydrogenic_atomic_radius

def laguer_polynomial(x, n, l):
    
    laguer_poly=sum( ( ((scipy.special.factorial((l + n), exact=True))*((-x)**k))/((scipy.special.factorial((2*l + 1 + k), exact=True))*(scipy.special.factorial((n - l - 1 - k), exact=True)) * scipy.special.factorial((k), exact=True)) ) for k in range(0, n - l))
    
    return laguer_poly

def radial_wavefunction(r, n, l):

    hydrogenic_atomic_radius = calculate_radius()
    
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

    hydrogenic_atomic_radius = calculate_radius()

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
    mesh = ax.pcolormesh(X, Z, probability_density, shading='auto', cmap='inferno')
    plt.colorbar(mesh, ax=ax, label='Probability Density')
    ax.set_title(f'Probability Density (XZ cross-section) n={n}, l={l}, m={m}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def gen_points_3d_cloud(threshold, range_input, num_range, render_type, noise=False):

    hydrogenic_atomic_radius = calculate_radius()

    # Generate Points
    range_extent = range_input * hydrogenic_atomic_radius

    x = np.linspace(-range_extent, range_extent, num_range)
    y = np.linspace(-range_extent, range_extent, num_range)
    z = np.linspace(-range_extent, range_extent, num_range)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')    
    
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    R_pts     = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
    THETA_pts = np.arccos(np.clip(points[:,2] / (R_pts + 1e-30), -1, 1))# polar angle from Z
    PHI_pts   = np.arctan2(points[:,1], points[:,0])# azimuthal angle
    
    psi = wavefunction(R_pts, THETA_pts, PHI_pts)
    
    pd_values = np.abs(psi)**2
    pd_normalised = pd_values / pd_values.max()  # scale to [0, 1]
    
    threshold_mask = pd_normalised > threshold
    random_mask = np.random.random(len(pd_normalised)) < pd_normalised
    
    if render_type == "voxel" or noise != True:
        random_mask=threshold_mask

    combined_mask = threshold_mask & random_mask

    cloud = pv.PolyData(points[combined_mask])
    cloud["probability_density"] = pd_normalised[combined_mask]
        
    return cloud

def gen_points_3d_monte_carlo(N,range_input):

    hydrogenic_atomic_radius = calculate_radius()
    
    range_extent = range_input * hydrogenic_atomic_radius
    
    points = np.random.uniform(-range_extent, range_extent, size=(N, 3))

    R_pts     = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
    THETA_pts = np.arccos(np.clip(points[:,2] / (R_pts + 1e-30), -1, 1))
    PHI_pts   = np.arctan2(points[:,1], points[:,0])

    psi = wavefunction(R_pts, THETA_pts, PHI_pts)

    pd_values = np.abs(psi)**2
    pd_normalised = pd_values / pd_values.max()

    random_mask = np.random.random(N) < pd_normalised

    monte_carlo = pv.PolyData(points[random_mask])
    monte_carlo["probability_density"] = pd_normalised[random_mask]
    
    return monte_carlo
    
    
def plot_probability_density_3d(n, l, m, range_input, num_range, threshold, cmap, sim_type, render_type, noise=False):

    hydrogenic_atomic_radius = calculate_radius()
    
    range_extent = range_input * hydrogenic_atomic_radius
    
    N = 1000000

    if sim_type == "cloud":    
        data=gen_points_3d_cloud(threshold, range_input ,num_range, render_type, noise)
    elif sim_type == "monte_carlo":
        data=gen_points_3d_monte_carlo(N, range_input)
    else:
        print("Please select sim type from list of supported types")
        return
    
    plotter = pv.Plotter(window_size=(900, 700))
    
    if render_type == "voxel":
        cube = pv.Cube()
        glyphs = data.glyph(geom=cube, scale=False, orient=False, factor=range_extent/(num_range/2))
        plotter.add_mesh(glyphs, scalars="probability_density", cmap=cmap, opacity=0.85)
    else:#Default to scatter
        plotter.add_mesh(
        data,
        point_size=8,#sphere radius in pixels
        scalars="probability_density",
        cmap=cmap,
        render_points_as_spheres=True,
        opacity=0.85,
        )
    
    plotter.add_text(
        f"3-D Probability Density: n={n}, l={l}, m={m}",
        position="upper_edge",
        font_size=12,
        color="white",
    )

    plotter.show_bounds(
        grid='back',
        location='outer',
        color='white',
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
        font_size=10,
        fmt="%.1e",
    )
        
    plotter.add_axes(interactive=True)
    
    plotter.background_color = 'black'
    plotter.camera_position = "iso" # isometric starting view

    plotter.show()
    
<<<<<<< HEAD
#plot_probability_density_3d(n, l, m, 50, 200, 0.1, "rainbow", "monte_carlo", "voxel") #f0 orbital


#GUI
frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="nsew")

ttk.Label(frame, text="Inputs").grid(row=0, column=0, sticky="w", padx=5, pady=2)

# Input field layout using grid
ttk.Label(frame, text="Atomic number").grid(row=1, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=Z).grid(row=1, column=1, padx=5, pady=2)

ttk.Label(frame, text="Atomic Mass").grid(row=2, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=A).grid(row=2, column=1, padx=5, pady=2)

ttk.Label(frame, text="Principal quantum number").grid(row=3, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=n).grid(row=3, column=1, padx=5, pady=2)

ttk.Label(frame, text="Azimuthal quantum number").grid(row=4, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=l).grid(row=4, column=1, padx=5, pady=2)

# Button layout converted from .pack() to .grid() to prevent layout freeze
render_btn = ttk.Button(
    frame,
    text="Render",
    command=lambda: threading.Thread(
        target=plot_probability_density_3d, 
        args=(float(n.get()), float(l.get()), float(m.get()), 50, 200, 0.1, "rainbow", "monte_carlo", "voxel"),
        daemon=True
    ).start()
)
render_btn.grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)

root.mainloop()
=======

plot_probability_density_3d(n, l, m, 50, 200, 0.1, "rainbow", "monte_carlo", "cube")
>>>>>>> 83a929dc04db13ee5fe3858016e9fd6e00e5c0ea
