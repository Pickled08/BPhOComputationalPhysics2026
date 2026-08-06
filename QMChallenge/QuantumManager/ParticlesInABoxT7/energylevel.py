import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy.constants

hbar = scipy.constants.hbar
pi = scipy.constants.pi
L = 1e-10
m_e = scipy.constants.m_e

def calculate_energy(n):
    if n<= 0:
        return 0
    energy_j = (n**2 * pi**2 * hbar**2) / (2 * m_e * L**2)
    energy_ev = energy_j / scipy.constants.e
    return energy_ev

fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)
initial_max_n = 5
x_data = np.arange(1, initial_max_n + 1)
y_data = [calculate_energy(n) for n in x_data]
ax.plot(x_data, y_data, 'o-', label='Energy Levels')
ax.set_xlabel('n')
ax.set_ylabel('Energy (eV)')
ax.set_title('Quantum Well Energy Levels')
ax.legend()
ax.grid(True)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
line = ax.get_lines()[0]
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
n_slider = Slider(
    ax=ax_slider,
    label='Max n',
    valmin=1,
    valmax=20,
    valinit=initial_max_n,
    valstep=1
)

def update(val):
    current_max_n = int(n_slider.val)
    new_x = np.arange(1, current_max_n + 1)
    new_y = [calculate_energy(n) for n in new_x]
    line.set_data(new_x, new_y)
    ax.set_xlim(0.5, current_max_n + 0.5)
    ax.set_ylim(0, max(new_y) * 1.1)
    fig.canvas.draw_idle()

n_slider.on_changed(update)
plt.show()