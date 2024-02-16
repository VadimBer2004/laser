# Simulate a Gaussian wavepacket in a quadratic potential

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def compute_gaussian(x_cent, alpha, p, gamma, grid, hbar=1):
    """
    Compute a gaussian at given coordinates using given parameters
    
    x_cent: complex - center of the wavepacket
    alpha: complex - quadratic coefficient
    p: complex - momentum of the wavepacket
    gamma: complex - linear coefficient
    grid: np.array - positions where to compute wavepacket
    hbar (optional): complex - hbar constant
    """
    # Tannor page 24

    N = np.power((2*np.real(alpha)/np.pi), 0.25) * np.exp(np.imag(gamma)/hbar)

    grid_shifted = grid - x_cent
    exponent = -alpha*np.square(grid_shifted) + (1j/hbar)*p*grid_shifted + (1j/hbar)*gamma
    gaussian = N*np.exp(exponent)

    return gaussian


def draw_gaussian(ax, gaussian, grid):
    """Draw a wavefunction on given matplotlib axes"""

    prob_density = np.abs(gaussian)
    real = np.real(gaussian)
    imag = np.imag(gaussian)

    ax.set_ylim([-0.5, 1])
    ax.plot(grid, prob_density, c="black")
    ax.plot(grid, real, c="red")
    ax.plot(grid, imag, c="blue")


def free_solution(t, x0, p0, alpha0, m, grid, hbar=1):
    """Find Gaussian wavepacket in free motion at a given time"""

    # Tannor page 25
    x0, p0, alpha0, t = complex(x0), complex(p0), complex(alpha0), complex(t)
    p = p0
    x = x0 + p0/m * t
    c = 2*1j*hbar*alpha0/m # temp constant
    alpha = alpha0/(complex(1) + c * t)
    gamma = (p0*p0/2/m)*t + (1j*hbar)/2*np.log(complex(1) + c * t)

    gaussian = compute_gaussian(x, alpha, p, gamma, grid, hbar=hbar)
    return gaussian

def quad_solution(t, x_pot, omega, x_init, p0, alpha0, m, grid, hbar=1):
    """Find Gaussian wavepacket in quadratic potential at a given time"""

    # Tannor pages 29-31
    x0, p0, alpha0, omega, t = complex(x_init-x_pot), complex(p0), complex(alpha0), complex(omega), complex(t)

    sin = np.sin(omega*t)
    cos = np.cos(omega*t)
    x = x0*cos + (p0/m/omega)*sin
    p = p0*cos - m*omega*x0*sin

    a = m*omega/2/hbar
    num = alpha0*cos + (1j)*a*sin
    denom = (1j)*alpha0*sin + a*cos
    alpha = a * num / denom

    gamma = (p*x - p0*x0)/2 + (1j)*hbar/2*np.log(denom / a)

    gaussian = compute_gaussian(x+x_pot, alpha, p, gamma, grid, hbar=hbar)
    return gaussian 


def free_update(frame, ax, grid):
    """Animation of a free Gaussian wavepacket"""

    x0 = 5
    p0 = -1
    alpha0 = 1
    m = 1
    hbar = 1
    t = frame/15

    gaussian = free_solution(t, x0, p0, alpha0, m, grid, hbar=hbar)
    ax.clear()
    draw_gaussian(ax, gaussian, grid)
    ax.legend(["density", "Re", "Im"])
    ax.set_xlabel("x")
    ax.set_ylabel("Psi(x, t)")
    ax.set_title("Free Wavepacket")


def draw_quad_potential(ax, x_pot, omega, mass, grid, color="orange"):
    """ Draw given quadratic potential on the axes"""

    pot = 0.5*mass*(omega**2)*np.square(grid-x_pot)
    ax.plot(grid, pot, c=color, linestyle="dashed")

def quad_update(frame, ax, grid):
    """Animation of a Gaussian wavepacket in a quadratic potential"""

    x0 = 10
    x_pot = 10
    p0 = 0
    alpha0 = 0.05 #coherent alpha0 = m*omega/2/hbar 
    m = 1
    omega = 0.1
    hbar = 1
    t = frame/2

    gaussian = quad_solution(t, x_pot, omega, x0, p0, alpha0, m, grid, hbar=hbar)
    ax.clear()
    draw_gaussian(ax, gaussian, grid)
    ax.plot(grid, 0.5*m*omega*omega*np.square(grid-x_pot), linestyle="dashed", c="orange")
    ax.legend(["density", "Re", "Im", "V(x)"])
    ax.set_xlabel("x")
    ax.set_ylabel("Psi(x, t)")
    ax.set_title("Quadratic Potential")


'''
Draw Gaussian motion
grid = np.linspace(-20, 20, 1000)
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig=fig, func=quad_update, frames=150, interval=50, fargs=(ax, grid))
plt.show()
'''

def compute_correlation(wave_1, wave_2, grid):
    """" Compute correlation function of 2 wavefunctions"""

    #assert wave_1.size == wave_2.size
    return np.trapz(np.conj(wave_1)*wave_2, x=grid)


def compute_spectrum(correlations, times):
    """Compute spectrum from correlation function of time"""
    #assert correlations.size == times.size
    # Assume times are evenly spaced
    return np.fft.ifft(correlations)

# Set initial conditions
grid = np.linspace(-10, 10, 10_000)
init_potential = {"x": 0.0, "omega": 1.0}
new_potential = {"x": 0.0, "omega": 1.0}
mass = 1

# Compute stationary gaussian
# Serves as initial wavefunction
init_alpha = 0.5 * init_potential["omega"] * mass * 1
init_x = init_potential["x"]
init_p = 0
init_time = 0
init_solution = quad_solution(init_time, init_potential["x"], init_potential["omega"], init_x, init_p, init_alpha, mass, grid, hbar=1)

period = 2*np.pi/new_potential["omega"] # period of oscillations
N_periods = 10 # how many periods we want to compute
points_per_period = 10_000 # how many points we want per period
time_steps = np.arange(0, N_periods*points_per_period) # how many time steps we take (spectrum resolution)
time_scale = points_per_period/period # how many gaussian computations per time step (correlations resolution)

# Compute gaussians at each time stamp
snapshots = []
for step in time_steps:
    packet = quad_solution(step/time_scale, new_potential["x"], new_potential["omega"], init_x, init_p, init_alpha, mass, grid, hbar=1)
    snapshots.append(packet)

# Compute corelations for each time stamp
correlations = []
for wave in snapshots:
    correlations.append(compute_correlation(init_solution, wave, grid))

# Compute spectrum from correlations
spectrum = compute_spectrum(correlations, time_steps)
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

draw_quad_potential(axs[2], init_potential["x"], init_potential["omega"], mass, grid)
draw_quad_potential(axs[2], new_potential["x"], new_potential["omega"], mass, grid, color="green")
draw_gaussian(axs[2], snapshots[0], grid)
# Sanity check
#draw_gaussian(axs[2],
#              quad_solution(50, init_potential["x"], init_potential["omega"], init_x, init_p, init_alpha, mass, grid, hbar=1),
#              grid)
axs[2].legend(["initial V(x)", "new V(x)", "inital p", "inital Re", "initial Im"])
axs[2].set_xlabel("x")
axs[2].set_ylabel("Psi(x, 0)")
axs[2].set_title("Potentials")

plt.show()
