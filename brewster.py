import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.stats import chi2

def read_file(filename):
    """reads a .txt

    Args:
        filename (string): filename for malus's law data file

    Returns:
        tuple: (array(position (theta)), array(intensity (voltage)))
    """
    data = np.loadtxt(filename, skiprows=2)

    position = np.array([])
    intensity = np.array([])
    for i in range(len(data)):
        position = np.append(position, data[i][0])
        intensity = np.append(intensity, data[i][1])
    return position, intensity

def chi_2(f, popt, position, intensity, uncertainty, start=0, stop=-1):
    slice = [start, stop]   #optimal range of which the best curve_fit covers

    n = intensity[slice[0]:slice[1]]
    n_fit = f(position, *popt) +10*np.sin(position/400) -5
    n_fit = n_fit[slice[0]:slice[1]]
    uncertainty = uncertainty[slice[0]:slice[1]]
    chi_2 = np.sum((n - n_fit)**2/(uncertainty**2))
    dof = len(n) - len(popt)
    probability = 1 - chi2.cdf(chi_2, dof)
    return probability

def chi_squared(fx: np.array, y: np.array, uy: np.array, m) -> float:
    """
    calculates xr ** 2

    :param fx: an array holding the outputs of the modelled function times x
    :param y: the y values being modelled against
    :param uy: the uncertainty in the y values
    :param m: number of parameters in model
    :return: chi squared value
    """
    n = len(fx)
    yminfx = y - fx
    denomentator = yminfx ** 2
    numerator = uy ** 2
    summed = np.sum(denomentator / numerator)
    return (1/(n - m)) * summed


def calc_residual(fx, y):
    return fx - y


if __name__ == "__main__":
    plotting = True
    plot_res = False

    # Loading data

    nop_position, nop_intensity = read_file("brewster_JA-ERA-unpolarized.txt")  # nop is non-polarized
    nop_uncertainty = np.zeros(len(nop_position)) + 5e-3

    h_position, h_intensity = read_file("brewster_JA-ERA-horizontal.txt")       # h is horizontally polarized
    h_uncertainty = np.zeros(len(h_intensity)) + 3e-3

    v_position, v_intensity = read_file("brewster_JA-ERA-vertical.txt")        # v is vertically polarized
    v_uncertainty = np.zeros(len(v_intensity)) + 3e-3

    # Plotting Data

     # plot data

    if plotting:
        fig, (ax1,ax2, ax3) = plt.subplots(3,1)

        ax1.errorbar(nop_position, nop_intensity, yerr=nop_uncertainty, zorder=1, label='Recorded Intensity') # plotting unpolarized brewster's
        ax1.set_title("Unpolarized Position vs. Intensity ")
        # ax1.set_ylabel("Intensity (V)")
        # ax1.set_xlabel("Position (radians)")


        ax2.errorbar(h_position, h_intensity, yerr=h_uncertainty, zorder=1, label='Recorded Intensity', color='r') # plotting horizontally polarized brewsters
        h_intensity[500:892] = 0
        ax2.errorbar(h_position, h_intensity, yerr=h_uncertainty, zorder=1, label='Recorded Intensity', color='b')
        ax2.set_title("Horizontally Polarized Position vs. Intensity")
        ax2.set_ylabel("Intensity (V)")
        # ax2.set_xlabel("Position (radians)")

        ax3.errorbar(v_position, v_intensity, yerr=v_uncertainty, zorder=1, label='Recorded Intensity') # plotting vertically polarized brewsters
        ax3.set_title("vertically polarized Position vs. Intensity")
        # ax2.set_ylabel("Intensity (V)")
        ax3.set_xlabel("Position (radians)")

        plt.show()

    
    