import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import math

e = math.e
hbar = scipy.constants.hbar
pi = scipy.constants.pi
m = scipy.constants.m_e
L = 1e-10
x = np.linspace(0, L, 1000)
quantum_numbers = np.array([1,2,3])

plt.figure(figsize=(8, 5))

for n in quantum_numbers:
  prob_density = (2 / L) * (np.sin(n * pi * x / L)) ** 2
  plt.plot(x, prob_density, label=f'n = {n}', linewidth=2)
plt.title('Probability Density vs Position (Particle in a Box)', fontsize=10)
plt.xlabel('Position (x)', fontsize=10)
plt.ylabel('Probability Density $|\psi_n(x)|^2$', fontsize=10)
plt.xlim(0, L)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.show()