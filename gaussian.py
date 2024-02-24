# Simulate a Gaussian wavepacket in a quadratic potential

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import QuadPotential, Gaussian, free_solution, quad_solution, compute_correlation, compute_spectrum



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



#Draw Gaussian motion
#grid = np.linspace(-20, 20, 1000)
#fig, ax = plt.subplots()
#ani = animation.FuncAnimation(fig=fig, func=quad_update, frames=150, interval=50, fargs=(ax, grid))
#plt.show()




# Set initial conditions for CO molecule
grid = np.linspace(0.8e-10, 2e-10, 10_000) # meters
omega1 = 2141.7 * 2*np.pi * 3e10 # rad/s
omega2 = 1743.41 * 2*np.pi * 3e10 # rad/s
x1 = 1.1327e-10 # meters
x2 = 1.2105e-10 # meters
e1 = -11.18*1.6e-19 # J
e2 = -5.1456*1.6e-19 # J
#mass = 1/1823*1.66e-27 # kg
mu = 12 * 16 / (12 + 16) * 1.66e-27 # kg
mass = mu
hbar = 1.054e10-34 # J*s
init_potential = QuadPotential(mu*omega1**2/(1.6e-19)/(27.21), x1, e1/(1.6e-19)/(27.21)-112.8)
new_potential = QuadPotential(mu*omega2**2/(1.6e-19)/(27.21), x2, e2/(1.6e-19)/(27.21)-112.8)

# Compute stationary gaussian
# Serves as initial wavefunction
init_wave = init_potential.get_stationary(mass, 0, hbar)

period = 2*np.pi/new_potential.find_omega(mass) # period of oscillations
N_periods = 5 # how many periods we want to compute
points_per_period = 2_000 # how many points we want per period
time_steps = np.arange(0, N_periods*points_per_period) # how many time steps we take (spectrum resolution)
time_scale = points_per_period/period # how many gaussian computations per time step (correlations resolution)

# Compute gaussians at each time stamp
snapshots = []
for step in time_steps:
    packet = quad_solution(step/time_scale, mass, init_wave, new_potential)
    snapshots.append(packet)

# Compute corelations for each time stamp
correlations = []
for wave in snapshots:
    correlations.append(compute_correlation(init_wave, wave, grid))

# Compute spectrum from correlations
spectrum = compute_spectrum(correlations)
max_i = np.argmax(spectrum)
# Plot the spectrum
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#axs[0].plot(time_steps[max(max_i-300, 0):max_i+300], np.abs(spectrum)[max(max_i-300, 0):max_i+300])
#axs[0].plot(np.fft.fftfreq(time_steps.size, d=1/time_scale), np.abs(spectrum))
axs[0].plot(time_steps, np.abs(spectrum))
axs[0].set_xlabel("omega")
axs[0].set_ylabel("sigma(omega)")
axs[0].set_title("Spectrum")

#axs[1].plot(time_steps[0:int(len(time_steps)/time_scale)], np.abs(correlations)[0:int(len(time_steps)/time_scale)])
axs[1].plot(time_steps/time_scale/period, np.abs(correlations))
axs[1].set_xlabel("time [periods]")
axs[1].set_ylabel("correlation")
axs[1].set_title("Correlation")

init_potential.draw(axs[2], grid)
new_potential.draw(axs[2], grid, color="green")
#init_wave.draw(axs[2], grid)
# Sanity check
#draw_gaussian(axs[2],
#              quad_solution(50, init_potential["x"], init_potential["omega"], init_x, init_p, init_alpha, mass, grid, hbar=1),
#              grid)
axs[2].legend(["initial V(x)", "new V(x)", "inital p", "inital Re", "initial Im"])
axs[2].set_xlabel("x")
axs[2].set_ylabel("Psi(x, 0)")
axs[2].set_title("Potentials")
plt.show()