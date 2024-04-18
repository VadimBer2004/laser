import numpy as np
import matplotlib.pyplot as plt
from utils import Gaussian, QuadPotential, quad_solution

hbar = 1
mass = 1
potential = QuadPotential(1, 0, 0)
wave = Gaussian(1, -1, 0.5, 0, hbar)

snapshots = []
gammas = np.full(100, 0, dtype=complex)
dt = 0.1
steps = np.arange(0, 100)
for i in steps:
    snapshot = quad_solution(i*dt, mass, wave, potential)
    snapshots.append(snapshot)
    gammas[i] = snapshot.gamma

fig, ax = plt.subplots()
ax.plot(steps*dt, np.real(gammas), c="red")
ax.plot(steps*dt, np.imag(gammas), c="blue")
plt.show()

'''
phase = np.linspace(0, 4*np.pi, 1_000, dtype=complex)
value = 0.5*np.log((1j)*np.sin(phase) + np.cos(phase))
#value = (1j)*np.sin(phase) + np.cos(phase)
fig, ax = plt.subplots()
ax.plot(phase, np.real(value), c="red")
ax.plot(phase, np.imag(value), c="blue")
plt.show()
'''