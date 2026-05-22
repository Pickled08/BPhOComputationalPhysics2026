import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import json
import os

# Load program configurations
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "materials.json")

# Work function values in electronvolts (eV)
# Source: Chemistry LibreTexts - B1: Workfunction Values (Reference Table)
# URL: https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Bulk_Properties/B1%3A_Workfunction_Values_(Reference_Table)

# Safely load materials.json
default_materials = [
    {"name": "Silver", "workfunction": 4.5},
    {"name": "Gold", "workfunction": 5.285},
    {"name": "Copper", "workfunction": 4.815},
    {"name": "Please check that materials.json exists to load more materials", "workfunction": 0.0}
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
    v = float(value)
    voltage_label.config(text=f"Voltage: {v:.1f}V")
    
    # Only show/scale text if voltage is strictly positive (> 0)
    if v > 0:
        # Max voltage is 10. Max font size is 16.
        # This formula scales the font size proportionally from 1 up to 16, but in reverse since it's negative.
        # 1. Map the voltage to a fraction between 0.0 and 1.0
        fraction = abs(v) / 10.0
        
        # 2. Apply sine wave scaling: sin(fraction * pi / 2)
        # When fraction is 1.0, sin(pi/2) = 1.0 (Maximum growth rate)
        # This makes the font size grow quickly at first, then level off smoothly
        sinusoidal_scale = np.sin(fraction * (np.pi / 2))
        
        # 3. Multiply by your max font size (16)
        dynamic_size = int(sinusoidal_scale * 16)
        
        # Ensure the font size is at least 1 so Tkinter doesn't throw an error
        if dynamic_size < 1: 
            dynamic_size = 1
            
        # Update the text and make its font size dynamic
        canvas.itemconfig(metal_text, text="+\n+\n+\n+\n+\n+\n+\n+\n+\n+", font=("Arial", dynamic_size, "bold"), fill="#FF3333")
    elif v < 0:
        # Max voltage is -10. Max font size is 16.
        # This formula scales the font size proportionally from 1 up to 16, but in reverse since it's negative.
        # 1. Map the voltage to a fraction between 0.0 and 1.0
        fraction = abs(v) / 10.0
        
        # 2. Apply sine wave scaling: sin(fraction * pi / 2)
        # When fraction is 1.0, sin(pi/2) = 1.0 (Maximum growth rate)
        # This makes the font size grow quickly at first, then level off smoothly
        sinusoidal_scale = np.sin(fraction * (np.pi / 2))
        
        # 3. Multiply by your max font size (16)
        dynamic_size = int(sinusoidal_scale * 16)
        
        # Ensure the font size is at least 1 so Tkinter doesn't throw an error
        if dynamic_size < 1: 
            dynamic_size = 1
            
        # Update the text and make its font size dynamic
        canvas.itemconfig(metal_text, text="-\n-\n-\n-\n-\n-\n-\n-\n-\n-", font=("Arial", dynamic_size, "bold"), fill="#007BFF")
    else:
        # If voltage is 0 or negative, hide the text by clearing it
        canvas.itemconfig(metal_text, text="")

def material_changed(value):
    #get the work function of the selected material
    work_function = next((m["workfunction"] for m in materials if m["name"] == value), None)
    color = next((m["color"] for m in materials if m["name"] == value), "#000000")
    if color is not None:
        canvas.itemconfig(1, fill=color)  # Update the rectangle's fill color
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
left_frame = tk.Frame(root, width=200, height=600)
left_frame.pack(side="left", fill="y", padx=20, pady=20)

# 2. Right Frame (The Animation Box)
# We use LabelFrame to draw a visible border/box with a title
animation_box = tk.LabelFrame(root, text="Simulation Animation", padx=10, pady=10)
animation_box.pack(side="right", fill="both", expand=True, padx=20, pady=20)

canvas = tk.Canvas(animation_box, bg="white", highlightthickness=1, highlightbackground="gray")
canvas.pack(fill="both", expand=True)

# Draw a rectangle on the canvas
canvas.create_rectangle(
    50, 100, 100, 350,   # Position and dimensions
    fill="lightgray",    # Inside color of the rectangle
    outline="black",     # Border color
    width=2              # Border thickness
)

metal_text = canvas.create_text(
    75, 225, 
    text="",  # Starts empty because initial voltage is 0
    font=("Arial", 1, "bold"), 
    justify="center",
    fill="#FF3333" 
)


# ==========================================
# WIDGETS (Now packed into left_frame)
# ==========================================

# Wavelength Group 
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
material_menu.config(width=15)
material_menu.pack(anchor="w", fill="x", pady=(15, 2))

material_label = tk.Label(left_frame, text="Material: None")
material_label.pack(anchor="w", pady=(0, 15))

work_function_label = tk.Label(left_frame, text="Work Function: N/A")
work_function_label.pack(anchor="w", pady=(0, 15))

root.mainloop()