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
root.geometry("500x500")

Z_inp = tk.IntVar()
A_inp = tk.IntVar()
n_inp = tk.IntVar()
l_inp = tk.IntVar()
m_inp = tk.IntVar()

grid_zoom = tk.IntVar()
resolution = tk.IntVar()
cutoff_threshold = tk.DoubleVar()

cmap_var = tk.StringVar(value="rainbow")
sim_var = tk.StringVar(value="monte_carlo")
render_var = tk.StringVar(value="voxel")

preset_var = tk.StringVar(value="Custom")

# Set default values
# Set default values
Z_inp.set(1)   # Atomic number
A_inp.set(1)   # Mass number
n_inp.set(4)   # 4th shell
l_inp.set(3)   # f orbital
m_inp.set(0)   # m=0 component

grid_zoom.set(40)
resolution.set(200)
cutoff_threshold.set(0.1)

PRESETS = {
    "1s": {"n": 1, "l": 0, "m": 0, "zoom": 8,  "cutoff": 0.05,  "cmap": "rainbow"},
    "2s": {"n": 2, "l": 0, "m": 0, "zoom": 9, "cutoff": 0.005,  "cmap": "rainbow"},
    "2p": {"n": 2, "l": 1, "m": 0, "zoom": 12, "cutoff": 0.05,  "cmap": "rainbow"},
    "3s": {"n": 3, "l": 0, "m": 0, "zoom": 16, "cutoff": 0.005, "cmap": "rainbow"},
    "3p": {"n": 3, "l": 1, "m": 0, "zoom": 20, "cutoff": 0.03, "cmap": "rainbow"},
    "3d": {"n": 3, "l": 2, "m": 0, "zoom": 20, "cutoff": 0.1, "cmap": "rainbow"},
    "4s": {"n": 4, "l": 0, "m": 0, "zoom": 24, "cutoff": 0.001, "cmap": "rainbow"},
    "4p": {"n": 4, "l": 1, "m": 0, "zoom": 40, "cutoff": 0.01, "cmap": "rainbow"},
    "4d": {"n": 4, "l": 2, "m": 0, "zoom": 40, "cutoff": 0.03, "cmap": "rainbow"},
    "4f": {"n": 4, "l": 3, "m": 0, "zoom": 40, "cutoff": 0.1, "cmap": "rainbow"},
}

def apply_preset(event=None):
    if preset_var.get() == "Custom":
        return

    p = PRESETS[preset_var.get()]

    n_inp.set(p["n"])
    l_inp.set(p["l"])
    m_inp.set(p["m"])

    grid_zoom.set(p["zoom"])
    cutoff_threshold.set(p["cutoff"])
    cmap_var.set(p["cmap"])

#Calculate orbital Shape

def calculate_radius():

    A = int(A_inp.get())
    Z = int(Z_inp.get())


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
    n=int(n_inp.get())
    m=int(m_inp.get())
    l=int(l_inp.get())
    
    psi = radial_wavefunction(r, n, l)*angular_wavefunction(theta, phi, l, m)
    
    return psi

def plot_linear_probability_density(r, n, l):
    
    A = int(A_inp.get())
    Z = int(Z_inp.get())
    
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
    max_pd = pd_values.max()

    if max_pd > 0:
        pd_normalised = pd_values / max_pd
    else:
        pd_normalised = pd_values
    
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
    
#plot_probability_density_3d(n, l, m, 50, 200, 0.1, "rainbow", "monte_carlo", "voxel") #f0 orbital


#GUI
frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="nsew")

ttk.Label(frame, text="Inputs").grid(row=0, column=0, sticky="w", padx=5, pady=2)

# Input field layout using grid
ttk.Label(frame, text="Atomic number").grid(row=1, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=Z_inp).grid(row=1, column=1, padx=5, pady=2)

ttk.Label(frame, text="Atomic Mass").grid(row=2, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=A_inp).grid(row=2, column=1, padx=5, pady=2)

ttk.Label(frame, text="Principal quantum number").grid(row=3, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=n_inp).grid(row=3, column=1, padx=5, pady=2)

ttk.Label(frame, text="Azimuthal quantum number").grid(row=4, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=l_inp).grid(row=4, column=1, padx=5, pady=2)

ttk.Label(frame, text="Magnetic quantum number").grid(row=5, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=m_inp).grid(row=5, column=1, padx=5, pady=2)

ttk.Label(frame, text="------------------------------------").grid(row=6, column=0, sticky="w", padx=5, pady=2)

ttk.Label(frame, text="Render Settings").grid(row=7, column=0, sticky="w", padx=5, pady=2)

ttk.Label(frame, text="Grid Zoom").grid(row=8, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=grid_zoom).grid(row=8, column=1, padx=5, pady=2)


ttk.Label(frame, text="Resolution").grid(row=9, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=resolution).grid(row=9, column=1, padx=5, pady=2)

ttk.Label(frame, text="Cutoff Threshold").grid(row=10, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(frame, textvariable=cutoff_threshold).grid(row=10, column=1, padx=5, pady=2)

ttk.Label(frame, text="Colour Map").grid(row=11, column=0, sticky="w", padx=5, pady=2)

all_cmaps = sorted(plt.colormaps())

favourites = ["inferno", "hot", "rainbow", "viridis", "plasma", "magma", "cividis"]
colourmaps = favourites + [c for c in all_cmaps if c not in favourites]

cmap_dropdown = ttk.Combobox(
    frame,
    textvariable=cmap_var,
    values=colourmaps,
    state="readonly",
    width=20
)
cmap_dropdown.grid(row=11, column=1, padx=5, pady=2, sticky="ew")

ttk.Label(frame, text="Simulation Type").grid(row=12, column=0, sticky="w", padx=5, pady=2)

sim_dropdown = ttk.Combobox(
    frame,
    textvariable=sim_var,
    values=["cloud", "monte_carlo"],
    state="readonly",
    width=20
)
sim_dropdown.grid(row=12, column=1, padx=5, pady=2, sticky="ew")

ttk.Label(frame, text="Render Type").grid(row=13, column=0, sticky="w", padx=5, pady=2)

render_dropdown = ttk.Combobox(
    frame,
    textvariable=render_var,
    values=["voxel", "scatter"],
    state="readonly",
    width=20
)
render_dropdown.grid(row=13, column=1, padx=5, pady=2, sticky="ew")

ttk.Label(frame, text="------------------------------------").grid(row=14, column=0, sticky="w", padx=5, pady=2)

ttk.Label(frame, text="Preset").grid(
    row=15, column=0, sticky="w", padx=5, pady=2
)

preset_dropdown = ttk.Combobox(
    frame,
    textvariable=preset_var,
    values=["Custom"] + list(PRESETS.keys()),
    state="readonly",
    width=20
)

preset_dropdown.grid(
    row=15,
    column=1,
    padx=5,
    pady=2,
    sticky="ew"
)

preset_dropdown.bind("<<ComboboxSelected>>", apply_preset)

def render_with_error_popup():
    try:
        plot_probability_density_3d(
            int(n_inp.get()),
            int(l_inp.get()),
            int(m_inp.get()),
            int(grid_zoom.get()),
            int(resolution.get()),
            float(cutoff_threshold.get()),
            cmap_var.get(),
            sim_var.get(),
            render_var.get()
        )
    except Exception as e:
        from tkinter import messagebox
        messagebox.showerror(
            "Render Error",
            f"An error occurred while rendering:\n\n{type(e).__name__}: {e}"
        )


render_btn = ttk.Button(
    frame,
    text="Render",
    command=lambda: threading.Thread(
        target=render_with_error_popup,
        daemon=True
    ).start()
)

render_btn.grid(row=16, column=0, columnspan=2, sticky="ew", pady=5)

root.mainloop()

