# Simulate a Gaussian wavepacket in a quadratic potential

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import QuadPotential, Gaussian, free_solution, quad_solution, compute_correlation, compute_spectrum, possible_energy_transitions
from utils import calculate_energy_transition



def free_update(frame, ax, grid):
    """Animation of a free Gaussian wavepacket"""

    x0 = 5
    p0 = -1
    alpha0 = 1
    m = 1
    hbar = 1
    t = frame/15
    wave = Gaussian(x0, p0, alpha0, 0, hbar)
    new_wave = free_solution(t, m, wave)
    
    ax.clear()
    new_wave.draw(ax, grid)
    ax.legend(["density", "Re", "Im"])
    ax.set_xlabel("x")
    ax.set_ylabel("Psi(x, t)")
    ax.set_title("Free Wavepacket")


def quad_update(frame, ax, grid):
    """Animation of a Gaussian wavepacket in a quadratic potential"""

    x0 = 5
    x_pot = 5
    p0 = -1
    alpha0 = 0.1 #coherent alpha0 = m*omega/2/hbar 
    m = 1
    omega = 0.1
    hbar = 1
    t = frame/2
    wave = Gaussian(x0, p0, alpha0, 0, hbar)
    pot = QuadPotential(m*omega*omega, x_pot, 0.0)
    
    new_wave = quad_solution(t, m, wave, pot)
    
    ax.clear()
    new_wave.draw(ax, grid)
    pot.draw(ax, grid)
    ax.legend(["density", "Re", "Im", "V(x)"])
    ax.set_xlabel("x")
    ax.set_ylabel("Psi(x, t)")
    ax.set_title("Quadratic Potential")


'''
#Draw Gaussian motion
#grid = np.linspace(-20, 20, 1000)
#fig, ax = plt.subplots()
#ani = animation.FuncAnimation(fig=fig, func=quad_update, frames=150, interval=50, fargs=(ax, grid))
#plt.show()
'''



# TODO: Atomic units
# [energy] - hartree, 1 hartree = 27.211386 eV
# hbar = 1
# electron_mass = 1
# [time] - hbar per hartree
# [length] - Bohr radius
# 


# Set initial conditions for CO molecule
# Reference: https://iopscience.iop.org/article/10.1088/0253-6102/59/2/11/pdf
#grid = np.linspace(-1e-10, 5e-10, 10_000) # meters
grid = np.linspace(-1, 7, 10_000) # Bohr radii
#omega1 = 2141.7 * 2*np.pi * 3e10 # rad/s
omega1 = 2141.7 * 2*np.pi * 3e10 * (2.4189e-17) # rad * hartree / hbar
#omega2 = 1743.41 * 2*np.pi * 3e10 # rad/s
omega2 = 1743.41 * 2*np.pi * 3e10 * (2.4189e-17) # rad * hartree / hbar
#x1 = 1.1327e-10 # meters
x1 = 1.1327e-10 / (5.2918e-11) # Bohr radii
#x2 = 1.2105e-10 # meters
x2 = 1.2105e-10 / (5.2918e-11) # Bohr radii
#e1 = (-11.18)*1.6e-19 # J
e1 = (-11.18) / 27.211 # hartree
#e2 = (-5.1456)*1.6e-19 # J
e2 = (-5.1456) / 27.211 # hartree
#mass = 1/1823*1.66e-27 # electron mass in kg
mass = 1 # electron mass
#mu = 12 * 16 / (12 + 16) * 1.66e-27 # reduced atomic mass in kg
mu = 12 * 16 / (12 + 16) * 1836.153 # reduced mass in electron mass
#hbar = 1.054e-34 # J*s
hbar = 1 # atomic units
init_potential = QuadPotential(mu*omega1**2, x1, e1)
new_potential = QuadPotential(mu*omega2**2, x2, e2)

# Compute stationary gaussian
# Serves as initial wavefunction
init_wave = init_potential.get_stationary(mass, 0, hbar)
print(init_wave.x0, init_wave.p0, init_wave.hbar, init_wave.gamma, init_wave.alpha)


period = 2*np.pi/new_potential.find_omega(mass) # period of oscillations
N_periods = 250 # how many periods we want to compute
points_per_period = 100 # how many points we want per period
time_steps = np.arange(0, N_periods*points_per_period) # how many time steps we take (spectrum resolution)
dt = period/points_per_period # dt per time step (correlations resolution)
T = period * N_periods

# Compute gaussians at each time stamp
snapshots = []
gammas = np.full(time_steps.size, 1j)
for i in range(len(time_steps)):
    step = time_steps[i]
    packet = quad_solution(step*dt, mass, init_wave, new_potential)
    snapshots.append(packet)
    gammas[i] = packet.gamma


# Phase unwrapping
unwrapped_gammas = np.unwrap(np.real(gammas), period=np.pi) + (1j)*np.unwrap(np.imag(gammas), period=np.pi)
for i in range(len(snapshots)):
    snapshots[i].gamma = unwrapped_gammas[i]

# Compute corelations for each time stamp
correlations = []
for wave in snapshots:
    correlations.append(compute_correlation(init_wave, wave, grid))

"""
fig, ax = plt.subplots()black
ax.plot(time_steps, np.real(correlations), label="correlations")
ax.plot(time_steps, np.real(np.exp(-1j/hbar * time_steps * dt * 0.5*init_potential.find_omega(mass))), label="expected")
ax.legend()
plt.show()
exit()
"""
# Compute spectrum from correlations
spectrum = compute_spectrum(correlations)
spectrum_shifted = np.abs(np.fft.fftshift(spectrum))
sample_freq = np.fft.fftshift(np.fft.fftfreq(time_steps.size, d=dt))*2*np.pi - 0.5*init_potential.find_omega(mass) - e1/hbar
center_ind = points_per_period*N_periods//2 
max_i = np.argmax(spectrum_shifted[center_ind:])
left_lim = center_ind-1_000
right_lim = center_ind+2_500
print(f"Maximum E_omega is {sample_freq[center_ind:][max_i]*hbar} Hartree")
print(f"Maximum E_omega should be {calculate_energy_transition(init_potential, new_potential, 0, 0, mass, hbar)} Hartree")
print(f"Second maximum should be {calculate_energy_transition(init_potential, new_potential, 0, 1, mass, hbar)} Hartree")
# Plot the spectrum
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].plot(sample_freq[left_lim:right_lim]*hbar, spectrum_shifted[left_lim:right_lim])
#axs[0].plot(sample_freq, spectrum_shifted)
#axs[0].plot(time_steps, spectrum)
axs[0].set_xlabel("$\hbar \omega$ [$a.u.$]")
axs[0].set_ylabel("$\sigma(\omega)$")
axs[0].set_title("Spectrum")

#axs[1].plot(time_steps[0:int(len(time_steps)/time_scale)], np.abs(correlations)[0:int(len(time_steps)/time_scale)])
axs[1].plot((time_steps*dt/period)[:4*points_per_period], np.abs(correlations)[:4*points_per_period], c="black")
#axs[1].plot((time_steps*dt/period)[:4*points_per_period], np.real(correlations)[:4*points_per_period], c="red")
#axs[1].plot((time_steps*dt/period)[:4*points_per_period], np.imag(correlations)[:4*points_per_period], c="blue")
#axs[1].plot((time_steps*dt/period)[:2*points_per_period], np.real(gammas)[:2*points_per_period], c="red")
#axs[1].plot((time_steps*dt/period)[:2*points_per_period], np.imag(gammas)[:2*points_per_period], c="blue")
axs[1].set_xlabel("$t$ [periods]")
axs[1].set_ylabel("| $< \psi(0) | \psi(t) > $ |")
axs[1].set_title("Correlation")

axs[2].set_ylim([-5, 20])
axs[2].set_xlim([-0.1, 5])
init_potential.draw(axs[2], grid, label="$X^1 \Sigma^+$ (initial)", color="orange")
init_potential.draw_levels(axs[2], grid, mass, hbar, 1, n_levels=0, label="levels", color="orange")
new_potential.draw(axs[2], grid, label="$a^3 \Pi$ (new)", color="green")
new_potential.draw_levels(axs[2], grid, mass, hbar, 1, n_levels=10, label="levels", color="green")
#axs[2].plot([0, 0], [np.real(init_potential.v0), -100*np.real(init_potential.v0)], linestyle="dotted", c="black", label="$R=0$")
axs[2].plot(spectrum_shifted[left_lim:right_lim]*5,
            sample_freq[left_lim:right_lim]*hbar + 0.5*hbar*init_potential.find_omega(mass) + e1,
            label="transition levels") # Hooray!
init_wave.draw(axs[2], grid, labels=["$p_{init}$", "Re_init", "Im_init"], colors=["red", None, None], draw_flags=[True, False, False])
axs[2].plot(grid, np.full(grid.size, 0), linestyle="dashed", color="grey", label="dissociation energy")
#snapshots[400].draw(axs[2], grid, labels=["p 1", "Re 1", "Im 1"], colors=["red", None, None], draw_flags=[True, False, False])
#snapshots[500].draw(axs[2], grid, labels=["p 2", "Re 2", "Im 2"], colors=["green", None, None], draw_flags=[True, False, False])
#snapshots[550].draw(axs[2], grid, labels=["p 3", "Re 3", "Im 3"], colors=["blue", None, None], draw_flags=[True, False, False])
# Sanity check
#quad_solution(50, mass, init_wave, init_potential).draw(axs[2], grid, draw_flags=[True, True, False])
axs[2].legend()
axs[2].set_xlabel("$R$ [$a_0$]")
axs[2].set_ylabel("$E_{pot}$ [$a.u.$]")
axs[2].set_title("Potentials")

fig.suptitle("CO molecule transition")
plt.tight_layout()
plt.show()