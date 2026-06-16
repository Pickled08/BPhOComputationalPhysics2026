"""
PyVista 3D Scatter Plot Example
================================
Generates a 3D scatter plot of 500 random points coloured by a scalar field
(distance from origin). Run this locally with:

    pip install pyvista
    python pyvista_scatter.py

PyVista docs: https://docs.pyvista.org
"""

import numpy as np
import pyvista as pv

# ── 1. Generate random 3-D point cloud ────────────────────────────────────────
rng = np.random.default_rng(42)
n_points = 500

x = rng.normal(0, 1, n_points)
y = rng.normal(0, 1, n_points)
z = rng.normal(0, 1, n_points)

points = np.column_stack((x, y, z))          # shape (N, 3)

# ── 2. Build a PyVista PolyData object ────────────────────────────────────────
cloud = pv.PolyData(points)

print(cloud)  # shows number of points, cells, and bounds

# Attach a scalar array – distance from the origin – used for colouring

# ── 4. Plot ───────────────────────────────────────────────────────────────────
plotter = pv.Plotter(window_size=(900, 700))

plotter.add_mesh(
    cloud,
    point_size=8,              # sphere radius in pixels
    render_points_as_spheres=True,         # matplotlib colourmap
    opacity=0.85,
)

# Bounding box grid lines for spatial reference
plotter.show_grid(
    xlabel="X",
    ylabel="Y",
    zlabel="Z",
    font_size=10,
)

# Axes orientation widget (bottom-left corner)
plotter.add_axes(interactive=True)

# Title annotation
plotter.add_text(
    "3-D Scatter — coloured by distance from origin",
    position="upper_edge",
    font_size=12,
    color="white",
)

plotter.background_color = "#1a1a2e"   # dark navy
plotter.camera_position = "iso"        # isometric starting view

plotter.show()