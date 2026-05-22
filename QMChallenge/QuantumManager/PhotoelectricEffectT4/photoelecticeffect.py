import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import json
import os

# Load program configurations
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "materials.json")

# Safely load materials.json
default_materials = [
    {"name": "Silver", "workfunction": 4.5},
    {"name": "Gold", "workfunction": 5.285},
    {"name": "Copper", "workfunction": 4.815},
]

materials = []
if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            materials = json.load(f)
            if not isinstance(materials, list):
                raise ValueError("materials.json must contain a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: failed to parse materials.json ({json_path}): {e}")
        materials = default_materials
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(materials, f, indent=2)
                print(f"Wrote fallback defaults to {json_path}")
        except OSError:
            pass
    except OSError as e:
        print(f"Warning: cannot open materials.json ({json_path}): {e}")
        materials = default_materials
else:
    print(f"Warning: materials.json not found at {json_path}, using defaults.")
    materials = default_materials

def wavelength_changed(value):
    wavelength_label.config(text=f"Wavelength: {int(float(value))}nm")

def intensity_changed(value):
    intensity_label.config(text=f"Intensity: {int(float(value))}%")

def voltage_changed(value):
    voltage_label.config(text=f"Voltage: {float(value):.1f}V")

def material_changed(value):
    #get the work function of the selected material
    work_function = next((m["workfunction"] for m in materials if m["name"] == value), None)
    if work_function is not None:
        work_function_label.config(text=f"Work Function: {work_function} eV")
    material_label.config(text=f"Material: {value}")

# Create the main window
root = tk.Tk()
root.title("Photoelectric Effect Simulation")
root.geometry("900x600") # Widened the window slightly to fit both sides comfortably

# ==========================================
# LAYOUT STRUCTURE (The Left/Right Split)
# ==========================================

# 1. Left Frame to hold all sliders and labels
left_frame = tk.Frame(root)
left_frame.pack(side="left", fill="y", padx=20, pady=20)

# 2. Right Frame (The Animation Box)
# We use LabelFrame to draw a visible border/box with a title
animation_box = tk.LabelFrame(root, text="Simulation Animation", padx=10, pady=10)
animation_box.pack(side="right", fill="both", expand=True, padx=20, pady=20)

# Optional placeholder label inside the animation box so you can see it
placeholder_label = tk.Label(animation_box, text="[Animation Component Goes Here]", fg="gray")
placeholder_label.pack(expand=True)


# ==========================================
# WIDGETS (Now packed into left_frame)
# ==========================================

# --- Wavelength Group ---
wavelength_slider = tk.Scale(
    left_frame, # Changed parent to left_frame
    from_=100, 
    to=850, 
    orient="horizontal", 
    command=wavelength_changed
)
wavelength_slider.pack(anchor="w", pady=(10, 2))

wavelength_label = tk.Label(left_frame, text="Wavelength: 0nm")
wavelength_label.pack(anchor="w", pady=(0, 15))

# --- Intensity Group ---
intensity_slider = tk.Scale(
    left_frame, # Changed parent to left_frame
    from_=0, 
    to=100, 
    orient="horizontal", 
    command=intensity_changed
)
intensity_slider.pack(anchor="w", pady=(10, 2))

intensity_label = tk.Label(left_frame, text="Intensity: 0%")
intensity_label.pack(anchor="w", pady=(0, 15))

# --- Voltage Group ---
voltage_slider = tk.Scale(
    left_frame, # Changed parent to left_frame
    from_=-10, 
    to=10, 
    resolution=0.1,
    orient="horizontal", 
    command=voltage_changed # Connected your missing callback here
)
voltage_slider.pack(anchor="w", pady=(10, 2))

voltage_label = tk.Label(left_frame, text="Voltage: 0V")
voltage_label.pack(anchor="w", pady=(0, 15))

# --- Material Group ---
material_dropdown = tk.StringVar(root)
material_dropdown.set("Select Material") 

material_options = [material["name"] for material in materials]
material_menu = tk.OptionMenu(left_frame, material_dropdown, *material_options, command=material_changed)
material_menu.pack(anchor="w", pady=(15, 2))

material_label = tk.Label(left_frame, text="Material: None")
material_label.pack(anchor="w", pady=(0, 15))

work_function_label = tk.Label(left_frame, text="Work Function: N/A")
work_function_label.pack(anchor="w", pady=(0, 15))

root.mainloop()