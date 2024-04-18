""" Potential classes """

import numpy as np

class Gaussian:
    def __init__(self, x0, p0, alpha, gamma, hbar):
        """
        x_cent: complex - center of the wavepacket
        alpha: complex - quadratic coefficient
        p: complex - momentum of the wavepacket
        gamma: complex - linear coefficient
        grid: np.array - positions where to compute wavepacket
        hbar (optional): complex - hbar constant
        """
        self.x0 = complex(x0)
        self.p0 = complex(p0)
        self.alpha = complex(alpha)
        self.gamma = complex(gamma)
        self.hbar = complex(hbar)

    def compute(self, grid):
        """
        Compute a gaussian at given coordinates using given parameters
        """
        # Tannor page 24

        N = np.power((2*np.real(self.alpha)/np.pi), 0.25) * np.exp(np.imag(self.gamma)/self.hbar)

        grid_shifted = grid - self.x0
        exponent = -self.alpha*np.square(grid_shifted) + (1j/self.hbar)*self.p0*grid_shifted + (1j/self.hbar)*self.gamma
        gaussian = N*np.exp(exponent)

        return gaussian
    
    def draw(self, ax, grid, labels=["p", "Re", "Im"], colors=["black", "red", "blue"], draw_flags=[True, True, True]):
        """Draw a wavefunction on given matplotlib axes"""
        gaussian = self.compute(grid)
        prob_density = np.abs(gaussian)
        real = np.real(gaussian)
        imag = np.imag(gaussian)

        #ax.set_ylim([-0.5, 1])
        if draw_flags[0]: ax.plot(grid, prob_density, c=colors[0], label=labels[0])
        if draw_flags[1]: ax.plot(grid, real, c=colors[1], label=labels[1])
        if draw_flags[2]: ax.plot(grid, imag, c=colors[2], label=labels[2])



class QuadPotential:
    def __init__(self, k, x0, v0):
        self.k = complex(k)
        self.x0 = complex(x0)
        self.v0 = complex(v0)
    
    def find_omega(self, m):
        return np.sqrt(self.k / m)

    def compute(self, grid):
        return 0.5 * self.k * np.square(grid-self.x0) + self.v0
    
    def draw(self, ax, grid, label=None, color="orange"):
        pot = self.compute(grid)
        ax.plot(grid, pot, c=color, label=label)
    
    def draw_levels(self, ax, grid, m, hbar, size, n_levels=10, label=None, color="orange"):
        mask = np.isclose(grid, self.x0)
        for i in range(n_levels):
            energy = (i + 0.5)*hbar*self.find_omega(m) + self.v0
            if i == 0:
                ax.plot([self.x0-size/2, self.x0+size/2], [energy, energy], c=color, label=label)
            else:
                ax.plot([self.x0-size/2, self.x0+size/2], [energy, energy], c=color)

    def calc_energy_level(self, n, mass, hbar):
        return (0.5 + n) * self.find_omega(mass) * hbar + self.v0
    def get_stationary(self, m, gamma, hbar):
        """ Get a stationary Gaussian wave within the potential """
        x0 = self.x0
        p0 = 0
        alpha = 0.5 * self.find_omega(m) * m / hbar
        stat_wave = Gaussian(x0, p0, alpha, gamma, hbar)
        return stat_wave



def free_solution(t, m, wave:Gaussian):
    """Find Gaussian wavepacket in free motion at a given time"""

    # Tannor page 25
    p = wave.p0
    x = wave.x0 + wave.p0/m * t
    c = 2*1j*wave.hbar*wave.alpha/m # temp constant
    alpha = wave.alpha/(complex(1) + c * t)
    gamma = (wave.p0*wave.p0/2/m)*t + (1j*wave.hbar)/2*np.log(complex(1) + c * t)
    moved_wave = Gaussian(x, p, alpha, gamma, wave.hbar)
    return moved_wave


def quad_solution(t, m, wave:Gaussian, pot:QuadPotential):
    """Find Gaussian wavepacket in quadratic potential at a given time"""

    # Tannor pages 29-31
    x0, p0, alpha0, omega, t, hbar = wave.x0 - pot.x0, wave.p0, wave.alpha, pot.find_omega(m), complex(t), wave.hbar

    sin = np.sin(omega*t)
    cos = np.cos(omega*t)
    x = x0*cos + (p0/m/omega)*sin
    p = p0*cos - m*omega*x0*sin

    a = m*omega/2/hbar
    num = alpha0*cos + (1j)*a*sin
    denom = (1j)*alpha0*sin + a*cos
    alpha = a * num / denom

    turn = np.real(omega*t + np.pi) // (2*np.pi)
    #turn = 0

    gamma = (p*x - p0*x0)/2 + (1j)*hbar/2*(np.log(denom / a)+(1j)*2*np.pi*turn) - pot.v0*t

    new_wave = Gaussian(pot.x0+x, p, alpha, gamma, hbar)
    return new_wave


def compute_correlation(wave1:Gaussian, wave2:Gaussian, grid):
    """" Compute correlation function of 2 wavefunctions"""

    #assert wave_1.size == wave_2.size
    gaussian1 = wave1.compute(grid)
    gaussian2 = wave2.compute(grid)
    return np.trapz(np.conj(gaussian1)*gaussian2, x=grid)


def compute_spectrum(correlations):
    """Compute spectrum from correlation function of time"""
    """Time intervals must be evenly spaced"""
    #assert correlations.size == times.size
    # Assume times are evenly spaced
    # Returns linear freuqency
    return np.fft.ifft(correlations)


def calculate_energy_transition(init_pot: QuadPotential, new_pot: QuadPotential, n, m, mass, hbar):

    init_omega = init_pot.find_omega(mass)
    new_omega = new_pot.find_omega(mass)
    e_init = (0.5 + n)*init_omega*hbar + init_pot.v0
    e_new = (0.5 + m)*new_omega*hbar + new_pot.v0
    delta_e = e_new - e_init
    return delta_e

def possible_energy_transitions(init_pot: QuadPotential, new_pot: QuadPotential, mass, hbar, n_init_lim=10, n_new_lim=10):

    init_omega = init_pot.find_omega(mass)
    new_omega = new_pot.find_omega(mass)

    levels = []
    for i in range(n_init_lim):
        for j in range(n_new_lim):
            e_init = (0.5 + i)*init_omega*hbar + init_pot.v0
            e_new = (0.5 + j)*new_omega*hbar + new_pot.v0
            delta_e = e_new - e_init
            levels.append(delta_e)
    levels.sort()

    return levels
