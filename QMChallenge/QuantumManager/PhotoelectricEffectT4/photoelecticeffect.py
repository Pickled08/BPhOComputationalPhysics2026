import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import json
import os
from PIL import Image, ImageTk, ImageDraw
import threading
import time

from plot_SP_F import plot_SP_F

# Load program configurations
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "materials.json")

# Global variables for image manipulation
resized_lamp = None
resized_battery = None
battery_image = None
resized_image = None
light_beam_id = None  # Tracking ID for the canvas transparent overlay
BATTERY_ROTATION_ANGLE = 0
LAMP_ROTATION_ANGLE = -60

RUN_SIMULATION = False  # Set to True to enable the main simulation loop, False to only show the GUI without photon animation

# Simulation state variables
wavelength = 100 #wavelength in nm, default to 100nm (UV range)
intensity = 0 #intensity as a percentage (0-100), default to 0%
voltage = 0.0 #voltage in volts, default to 0V
work_function = None #work function in eV, default to None until a material is selected
photons_per_second = 0.0 #calculated photons per second based on wavelength and intensity

polygon_points = [(300, 200), (300, 450), (750, 150), (650, 50)]

quantum_efficiency = 0.1 #Assume 10% of photons that hit the surface cause electron emission for simplicity

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
    
def wavelength_to_hex(wavelength: float, gamma: float = 1.00) -> str:
    """
    Converts a given wavelength of light (in nanometers) to a hex color string.
    Wavelengths outside the visible spectrum (380 nm to 750 nm) return black ("#000000").
    
    Based on Dan Bruton's algorithm
    """
    #Determine base RGB components and intensity attenuation (factor)
    if 380 <= wavelength <= 439:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
        factor = 0.3 + 0.7 * (wavelength - 380) / (439 - 380)
    elif 440 <= wavelength <= 489:
        r = 0.0
        g = (wavelength - 440) / (489 - 440)
        b = 1.0
        factor = 1.0
    elif 490 <= wavelength <= 509:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
        factor = 1.0
    elif 510 <= wavelength <= 579:
        r = (wavelength - 510) / (579 - 510)
        g = 1.0
        b = 0.0
        factor = 1.0
    elif 580 <= wavelength <= 644:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
        factor = 1.0
    elif 645 <= wavelength <= 750:
        r = 1.0
        g = 0.0
        b = 0.0
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
    else:
        # Non-visible spectrum (100-379 nm and 751-850 nm)
        return "#000001"

    #Apply intensity factor and gamma correction, scaling to 0-255 range
    r_int = int(round(255 * (r * factor) ** gamma)) if r > 0 else 0
    g_int = int(round(255 * (g * factor) ** gamma)) if g > 0 else 0
    b_int = int(round(255 * (b * factor) ** gamma)) if b > 0 else 0

    #Convert integer values to a hexadecimal string format
    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"

def update_beam_visual():
    global light_beam_id
    
    if light_beam_id is not None:
        canvas.delete(light_beam_id)
        
    fraction = float(intensity) / 100.0
    alpha_pct = np.sin(fraction * (np.pi / 2)) / 3.0
    current_hex = wavelength_to_hex(wavelength)
    
    if alpha_pct == 0 or current_hex == "#000000":
        light_beam_id = None
        return

    # Update canvas architecture layouts safely
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    
    # Fail-safe catch if window sizes haven't completed rendering yet
    if width <= 1 or height <= 1:
        width, height = 1100, 700 

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    hex_color = current_hex.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    alpha_val = int(alpha_pct * 255)
    
    draw.polygon(polygon_points, fill=(r, g, b, alpha_val))
    tk_img = ImageTk.PhotoImage(img)
    
    light_beam_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.beam_image_ref = tk_img
    canvas.tag_lower(light_beam_id)

def wavelength_changed(value):
    global wavelength
    wavelength = int(float(value))
    wavelength_label.config(text=f"Wavelength: {wavelength}nm")
    update_beam_visual()

def intensity_changed(value):
    global intensity
    intensity = int(float(value))
    intensity_label.config(text=f"Intensity: {int(float(value))}%")
    update_beam_visual()

def voltage_changed(value):
    global voltage
    v = float(value)
    voltage = v
    voltage_label.config(text=f"Voltage: {v:.1f}V")
    
    if resized_image is not None:
        # If voltage is negative, flip the battery 180 degrees
        angle = BATTERY_ROTATION_ANGLE + 180 if v < 0 else BATTERY_ROTATION_ANGLE
        rotated = resized_image.rotate(angle, expand=True)
        battery_image = ImageTk.PhotoImage(rotated)
        
        # Update the existing canvas image item configuration
        canvas.itemconfig(battery_item_id, image=battery_image)
        canvas.image = battery_image  # Save structural reference

    # Only show/scale text if voltage is strictly positive (> 0)
    if v > 0:
        # Max voltage is 10. Max font size is 16.
        #This formula scales the font size proportionally from 1 up to 16
        # Map the voltage to a fraction between 0.0 and 1.0
        fraction = abs(v) / 10.0
        
        #Apply sine wave scaling: sin(fraction * pi / 2)
        # When fraction is 1.0, sin(pi/2) = 1.0 (Maximum growth rate)
        # This makes the font size grow quickly at first, then level off smoothly
        sinusoidal_scale = np.sin(fraction * (np.pi / 2))
        
        #Multiply by your max font size (16)
        dynamic_size = int(sinusoidal_scale * 16)
        
        # Ensure the font size is at least 1
        if dynamic_size < 1: 
            dynamic_size = 1
            
        # Update the text and make its font size dynamic
        canvas.itemconfig(left_metal_charge_text, text="+\n+\n+\n+\n+\n+\n+\n+\n+\n+", font=("Arial", dynamic_size, "bold"), fill="#FF3333")
        canvas.itemconfig(right_metal_charge_text, text="-\n-\n-\n-\n-\n-\n-\n-\n-\n-", font=("Arial", dynamic_size, "bold"), fill="#007BFF")
    elif v < 0:
        # Max voltage is -10. Max font size is 16.
        # This formula scales the font size proportionally from 1 up to 16
        # Map the voltage to a fraction between 0.0 and 1.0
        fraction = abs(v) / 10.0
        
        #Apply sine wave scaling: sin(fraction * pi / 2)
        # When fraction is 1.0, sin(pi/2) = 1.0 (Maximum growth rate)
        # This makes the font size grow quickly at first, then level off smoothly
        sinusoidal_scale = np.sin(fraction * (np.pi / 2))
        
        #Multiply by your max font size (16)
        dynamic_size = int(sinusoidal_scale * 16)
        
        # Ensure the font size is at least 1 so Tkinter doesn't throw an error
        if dynamic_size < 1: 
            dynamic_size = 1
            
        # Update the text and make its font size dynamic
        canvas.itemconfig(left_metal_charge_text, text="-\n-\n-\n-\n-\n-\n-\n-\n-\n-", font=("Arial", dynamic_size, "bold"), fill="#007BFF")
        canvas.itemconfig(right_metal_charge_text, text="+\n+\n+\n+\n+\n+\n+\n+\n+\n+", font=("Arial", dynamic_size, "bold"), fill="#FF3333")
    else:
        # If voltage is 0 or negative, hide the text by clearing it
        canvas.itemconfig(left_metal_charge_text, text="")
        canvas.itemconfig(right_metal_charge_text, text="")
    
def material_changed(value):
    #get the work function of the selected material
    global work_function
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
root.geometry("1400x800")

# ==========================================
# LAYOUT STRUCTURE (The Left/Right Split)
# ==========================================

#Left Frame to hold all sliders and labels
left_frame = tk.Frame(root, width=200, height=600)
left_frame.pack(side="left", fill="y", padx=20, pady=20)

#Right Frame (The Animation Box)
animation_box = tk.LabelFrame(root, text="Simulation Animation", padx=10, pady=10)
animation_box.pack(side="right", fill="both", expand=True, padx=20, pady=20)

canvas = tk.Canvas(animation_box, bg="white", highlightthickness=1, highlightbackground="gray")
canvas.pack(fill="both", expand=True)

# 1. Left Electrode (Emitter)
left_electrode = canvas.create_rectangle(
    250, 200, 300, 450,   # Position and dimensions
    fill="lightgray",     # Inside color of the rectangle
    outline="black",      # Border color
    width=2               # Border thickness
)

# 2. Right Electrode (Collector)
right_electrode = canvas.create_rectangle(
    750, 200, 800, 450,   # Position and dimensions
    fill="lightgray",     # Inside color of the rectangle
    outline="black",      # Border color
    width=2               # Border thickness
)

left_metal_charge_text = canvas.create_text(
    275, 325,             
    text="",  
    font=("Arial", 12, "bold"), 
    justify="center",
    fill="#FF3333" 
)

right_metal_charge_text = canvas.create_text(
    775, 325,             
    text="",
    font=("Arial", 12, "bold"),
    justify="center",
    fill="#007BFF"
)

#Draw Wires
canvas.create_line(250, 325, 200, 325, fill="black", width=2)  # Left wire
canvas.create_line(800, 325, 850, 325, fill="black", width=2)  # Right wire
canvas.create_line(200, 325, 200, 600, fill="black", width=2)  # Left vertical wire
canvas.create_line(850, 325, 850, 600, fill="black", width=2)  # Right vertical wire
canvas.create_line(200, 600, 850, 600, fill="black", width=2)  # Bottom horizontal wire

#Draw Ammeter
center_x = 850
center_y = 462.5
r = 25 

# Draw the white circle with a black outline
canvas.create_oval(
    center_x - r, 
    center_y - r, 
    center_x + r, 
    center_y + r, 
    fill="white", 
    outline="black", 
    width=2
)

#Draw Ammeter Label
canvas.create_text(
    center_x, 
    center_y, 
    text="A", 
    font=("Arial", 16, "bold"), 
    fill="black"
)

#Current Lable next to Ammeter
current_label = canvas.create_text(
    center_x + 110, 
    center_y,
    text="Current: 0.00 A",
    font=("Arial", 14, "bold"),
    fill="black"
)
try:
    original_image = Image.open(os.path.join(base_dir, "Images/Lamp.png"))
    resized_image = original_image.resize((300, 300))
    
    # Process base rotation configuration
    rotated_image = resized_image.rotate(LAMP_ROTATION_ANGLE, expand=True)
    lamp_image = ImageTk.PhotoImage(rotated_image)

    # Saved to lamp_item_id so voltage_changed can modify it later
    lamp_item_id = canvas.create_image(700, 100, image=lamp_image)
    canvas.image = lamp_image 

except FileNotFoundError:
    lamp_item_id = canvas.create_text(525, 200, text="Lamp image file not found.", fill="red")


try:
    original_image = Image.open(os.path.join(base_dir, "Images/Battery.png"))
    resized_image = original_image.resize((150, 150))
    
    # Process base rotation configuration
    rotated_image = resized_image.rotate(BATTERY_ROTATION_ANGLE, expand=True)
    battery_image = ImageTk.PhotoImage(rotated_image)

    # Saved to battery_item_id so voltage_changed can modify it later
    battery_item_id = canvas.create_image(525, 600, image=battery_image)
    canvas.image = battery_image 

except FileNotFoundError:
    battery_item_id = canvas.create_text(525, 600, text="Battery image file not found.", fill="red")

update_beam_visual()

# ==========================================
# WIDGETS (Now packed into left_frame)
# ==========================================

# Wavelength Group
wavelength_slider = tk.Scale(
    left_frame,
    from_=100, 
    to=850, 
    orient="horizontal", 
    command=wavelength_changed,
    length=250,  # horizontal length in pixels
    width=25,    # thickness of the slider track
)
wavelength_slider.pack(anchor="w", pady=(10, 0))
try:
    spectra_original = Image.open(os.path.join(base_dir, "Images/Spectra.png"))
    spectra_resized = spectra_original.resize((250, 25))
    spectra_photo = ImageTk.PhotoImage(spectra_resized)
    spectra_label = tk.Label(left_frame, image=spectra_photo)
    spectra_label.image = spectra_photo  # Prevent garbage collection
    spectra_label.pack(anchor="w", pady=(0, 5))
except FileNotFoundError:
    tk.Label(left_frame, text="Spectra image not found.", fg="red").pack(anchor="w", pady=(15, 5))
    

wavelength_label = tk.Label(left_frame, text="Wavelength: 0nm")
wavelength_label.pack(anchor="w", pady=(0, 15))

# --- Intensity Group ---
intensity_slider = tk.Scale(
    left_frame,
    from_=0, 
    to=100, 
    orient="horizontal", 
    command=intensity_changed,
    length=250,  # horizontal length in pixels
    width=25,    # thickness of the slider track
)
intensity_slider.pack(anchor="w", pady=(10, 2))

intensity_label = tk.Label(left_frame, text="Intensity: 0%")
intensity_label.pack(anchor="w", pady=(0, 15))

# --- Voltage Group ---
voltage_slider = tk.Scale(
    left_frame,
    from_=-10, 
    to=10, 
    resolution=0.1,
    orient="horizontal", 
    command=voltage_changed,
    length=250,  # horizontal length in pixels
    width=25,    # thickness of the slider track
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

#Button to start the simulation loop (for testing purposes, can be removed or hidden in final version)
def toggle_simulation():
    global RUN_SIMULATION
    RUN_SIMULATION = not RUN_SIMULATION
    if RUN_SIMULATION:
        toggle_button.config(text="Stop Simulation")
    else:
        toggle_button.config(text="Start Simulation")
        
toggle_button = ttk.Button(left_frame, text="Start Simulation", command=toggle_simulation)
toggle_button.pack(fill="x", pady=(10, 20))

tk.Label(
    left_frame,
    text="Graph Tools"
).pack(pady=(30, 5))

ttk.Button(
    left_frame,
    text="Draw Stop Potential vs Frequency Graph",
    command=lambda: threading.Thread(target=plot_SP_F, args=(work_function,), daemon=True).start()
).pack(fill="x", pady=5)
ttk.Button(
    left_frame,
    text="Draw KE vs Frequency Graph",
).pack(fill="x", pady=5)

    
def hit_left_electrode(photon_coords, wavelength_photon):
    
    # Check if work function is defined for the selected material
    if work_function is None:
        return
    #10% chance to emit an electron if the photon hits the left electrode and has enough energy
    if np.random.randint(0, 100) < 90:
        return
    
    frequency = 299792458 / (wavelength_photon * 1e-9)
    photon_energy = 6.6260702e-34 * frequency
    work_function_joules = work_function * 1.6021766e-19
    
    if photon_energy < work_function_joules:
        return
    else:
        electron_kinetic_energy = photon_energy - work_function_joules
        
    speed_of_electron = np.sqrt(2 * electron_kinetic_energy / 9.1093837e-31)
    
    
    electron = canvas.create_oval(300, photon_coords[1], 310, photon_coords[1] + 10, fill="yellow", outline="black")
    
    def animate_electron():
        
        speed_display = speed_of_electron / 10000.0
        dt = 0.01
        step = speed_display * dt
        
        for _ in range(50000):
            
            canvas.move(electron, step, 0)  # Move left towards the right electrode
            time.sleep(dt)  # Pause for a short time to create animation effect
            if canvas.coords(electron)[0] > 735: # If the electron has hit the right electrode
                break
        canvas.delete(electron)  # Remove the electron after it moves across the screen
    threading.Thread(target=animate_electron).start()

def create_starburst(canvas, x, y, color, size=8):
    points = []
    for i in range(8):
        angle = i * (np.pi / 4)
        # Alternate between outer and inner radius
        r = size if i % 2 == 0 else size * 0.4
        points.extend([x + r * np.cos(angle), y + r * np.sin(angle)])
    return canvas.create_polygon(points, fill=color, outline="")

# This function creates a photon at the lamp's position and animates it moving towards the left electrode.
# The photon's color is determined by its wavelength using the wavelength_to_hex function. The animation runs in a separate thread to keep the GUI responsive.
def create_photon(wavelength_photon):
    #Draw circle at 700, 100 (the position of the lamp) and animate it moving towards the right electrode at 750, 325
    #Randomize the starting position slightly to create a more dynamic effect
    start_x = 700 + np.random.randint(-30, 30)
    start_y = 100 + np.random.randint(-30, 30)
    photon = create_starburst(canvas, start_x, start_y, wavelength_to_hex(wavelength_photon), size=8)
    def animate_photon():
        speed = 299.792458  # Adjust this value to make the photon move faster or slower
        angle = 155 * (np.pi / 180)  # Convert angle to radians
        dt = 0.01  # Time delay between each movement step (in seconds)
        step = np.array([speed * np.cos(angle) * dt, speed * np.sin(angle) * dt])
        
        for _ in range(150):
            canvas.move(photon, step[0], step[1])  # Move right and slightly down to follow the path towards the right electrode
            canvas.tag_lower(photon, left_electrode)
            time.sleep(dt)  # Pause for a short time to create animation effect
            if canvas.coords(photon)[0] < 300: # If the photon has hit the left electrode
                hit_left_electrode(canvas.coords(photon),wavelength_photon) # Trigger the electron animation
                break
        canvas.delete(photon)  # Remove the photon after it moves across the screen
    threading.Thread(target=animate_photon).start()

def photoelectric_effect():
    
    global photons_per_second
    global RUN_SIMULATION
    
    intensity_lamp = 1 #1Wm^-2 treat this as 1w output from the lamp for simplicity
    last_photon_time = 0.0
    
    while True:
        while RUN_SIMULATION:
            
            #print current values of wavelength, intensity, voltage, and material to the console for debugging
            #print("Current Simulation Parameters:")
            #print(f"Wavelength: {wavelength} nm")
            #print(f"Intensity: {intensity} %")
            #print(f"Voltage: {voltage} V")
            #print(f"Work Function: {work_function} eV")
            
            now = time.time()

            # Recalculate interval every tick using current globals
            if wavelength > 0 and intensity > 0:
                photons_per_second = (intensity_lamp * (intensity / 100.0)) / (6.626e-34 * 3e8 / (wavelength * 1e-9))
                photons_per_second_displayed = photons_per_second / 1e16
                interval = 1.0 / photons_per_second_displayed if photons_per_second_displayed > 0 else float('inf')
            else:
                interval = float('inf')

            if interval != float('inf') and (now - last_photon_time) >= interval:
                create_photon(wavelength)
                last_photon_time = now

            time.sleep(0.01)  # Tight loop — checks globals 100x/sec
        time.sleep(0.1)  # Sleep briefly when simulation is not running to prevent tight loop
        
def calculate_current():
    global RUN_SIMULATION
    while True:
        while RUN_SIMULATION:
            electron_charge = 1.6021766e-19
            electrons_per_second = photons_per_second * (1 if work_function is not None and (6.6260702e-34 * (299792458 / (wavelength * 1e-9))) >= (work_function * electron_charge) else 0) * quantum_efficiency
            current = electrons_per_second * electron_charge
            canvas.itemconfig(current_label, text=f"Current: {current:.3f} A")
            time.sleep(0.1)  # Update current every 0.1 seconds
        time.sleep(0.1)  # Sleep briefly when simulation is not running to prevent tight loop
        
#Start a separate thread to calculate current every second
threading.Thread(target=calculate_current, daemon=True).start()

#Start the main photoelectric effect simulation loop in a separate thread to keep the GUI responsive
thread = threading.Thread(target=photoelectric_effect)

thread.daemon = True 
thread.start()

root.mainloop()