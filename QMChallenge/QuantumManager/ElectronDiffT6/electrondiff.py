import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.constants

C = scipy.constants.c
M_ELECTRON = scipy.constants.m_e
E_CHARGE = scipy.constants.e
H = scipy.constants.h


def calculate_wavelength(voltage):
    return H / np.sqrt(2 * M_ELECTRON * E_CHARGE * voltage * (1 + (E_CHARGE * voltage) / (2 * M_ELECTRON * C**2)))


def calculate_angle(wavelength, d):
    return np.arcsin(wavelength/(2*d))


def calculate_ring_radius(angle, screen_distance):
    return screen_distance * np.sin(2*angle)


# Constants
screen_distance = 65e-3

# Initial voltage
voltage = 3000

# Create figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)


def get_rings(voltage):
    wavelength = calculate_wavelength(voltage)

    r1 = calculate_ring_radius(
        calculate_angle(wavelength, 0.123e-9),
        screen_distance
    )

    r2 = calculate_ring_radius(
        calculate_angle(wavelength, 0.213e-9),
        screen_distance
    )

    return r1, r2


# Draw initial rings
r1, r2 = get_rings(voltage)


theta = np.linspace(0, 2*np.pi, 500)

ring1, = ax.plot(
    r1*np.cos(theta),
    r1*np.sin(theta),
    label="d = 0.123 nm"
)

ring2, = ax.plot(
    r2*np.cos(theta),
    r2*np.sin(theta),
    label="d = 0.213 nm"
)


ax.set_aspect("equal")
ax.set_xlabel("Screen position (m)")
ax.set_ylabel("Screen position (m)")
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.05, 1)
)


# Slider
slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])

voltage_slider = Slider(
    slider_ax,
    "Voltage (V)",
    1000,
    5000,
    valinit=voltage
)


def update(val):
    voltage = voltage_slider.val

    r1, r2 = get_rings(voltage)

    ring1.set_xdata(r1*np.cos(theta))
    ring1.set_ydata(r1*np.sin(theta))

    ring2.set_xdata(r2*np.cos(theta))
    ring2.set_ydata(r2*np.sin(theta))

    fig.canvas.draw_idle()
    
voltage_slider.on_changed(update)

plt.show()

#Validation graph

voltage_array = np.linspace(1000, 5000, 1000)

wavelength_array = calculate_wavelength(voltage_array)

# Use graphite spacing
d = 0.123e-9

phi_array = calculate_angle(
    wavelength_array,
    d
)

plt.figure()

plt.plot(
    1/voltage_array,
    (np.sin(0.5 * phi_array)**2)
)

plt.xlabel("1/V (1/V)")
plt.ylabel("sin²(0.5φ)")
plt.title("Diffraction verification")

plt.grid()

plt.show()
