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
    # data = np.reshape(data, (2, len(data)))
    # data = np.invert(data)
    # position = data[0] # analyzer position in radians
    # intensity = data[1] # intensity in volts
    position = np.array([])
    intensity = np.array([])
    for i in range(len(data)):
        position = np.append(position, data[i][0])
        intensity = np.append(intensity, data[i][1])
    return position, intensity


def malus_law_2(phi, I0, phase_shift, offset):
    """calculates the intensity 

    Args:
        theta (_type_): _description_
        I0 (_type_): _description_
        offset (_type_): _description_

    Returns:
        np.array: calculated intensity
    """
    intensity = I0*np.square(np.cos(phi + phase_shift)) + offset
    return intensity

def malus_law_3(theta, I1, phase_shift, offset):
    """_summary_

    Args:
        theta (_type_): _description_
        I1 (_type_): _description_
        phase_shift (_type_): _description_
        offset (_type_): _description_

    Returns:
        _type_: _description_
    """
    intensity = (I1/4) * np.square(np.sin(2*((np.pi/4) + theta) + phase_shift)) + offset
    return intensity

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

    #loading data

    # print(read_file("malus_law_JA-ERA-2polarizers.txt"))
    position2, intensity2 = read_file("malus_law_JA-ERA-2polarizers.txt")
    position2 = position2[153:504]
    intensity2 = intensity2[153:504]

    position3, intensity3 = read_file("malus_law_JA-ERA-3polarizers.txt")

    uncertainty2 = np.zeros(len(intensity2)) + 5e-3
    uncertainty3 = np.zeros(len(intensity3)) + 3e-3
    # print(np.min(intensity3))

    # plot data

    if plotting:
        fig, (ax1,ax2) = plt.subplots(2,1)

        ax1.errorbar(position2, intensity2, yerr=uncertainty2, zorder=1, label='Recorded Intensity') # plotting 2 polaroids
        ax1.set_title("Position vs. Intensity with Two Polaroids")
        ax1.set_ylabel("Intensity (V)")
        # ax1.set_xlabel("Position (radians)")


        ax2.errorbar(position3, intensity3, yerr=uncertainty3, zorder=1, label='Recorded Intensity') # plotting 3 polaroids
        ax2.set_title("Position vs. Intensity with Three Polaroids")
        ax2.set_ylabel("Intensity (V)")
        ax2.set_xlabel("Position (radians)")
        
    
    # fit data, including chi squared analysis

    popt2, pcov2 = op.curve_fit(malus_law_2, position2, intensity2, p0=(3.5, -1.6, 0.01133))    # fitting 2 polaroids
    

    fx_2 = malus_law_2(position2, popt2[0], popt2[1], popt2[2])

    chi_squared_2 = chi_squared(fx_2, intensity2, uncertainty2, 3)
    print(chi_squared_2)
    # chi_2_2 = chi_2(malus_law_2, popt2, position2, intensity2, uncertainty2, 0, len(intensity2)+1)
    # print(chi_2_2)

    popt3, pcov3 = op.curve_fit(malus_law_3, position3, intensity3, p0=(3.5, np.pi/4, 0.01213))    # fitting 3 polaroids
    

    fx_3 = malus_law_3(position3, popt3[0], popt3[1], popt3[2])

    chi_squared_3 = chi_squared(fx_3, intensity3, uncertainty3, 3)
    print(chi_squared_3)
    # chi_2_3 = chi_2(malus_law_3, popt3, position3, intensity3, uncertainty3, 0, len(intensity3)+1)
    # print(chi_2_3)

    if plotting:
        ax1.plot(position2, fx_2, zorder=2, label='Modelled Fit')
        ax2.plot(position3, fx_3, zorder=2, label = 'Modelled Fit')

        ax1.legend()
        ax2.legend()

        
        plt.show()

    # residual plot showing difference between fit and measurement
    residual_2 = calc_residual(fx_2, intensity2)
    residual_3 = calc_residual(fx_3, intensity3)

    if plot_res:
        fig, (ax1,ax2) = plt.subplots(2,1)

        bl_2 = np.zeros(len(position2))
        ax1.plot(position2, bl_2, linestyle="--")
        ax1.set_title("2 Polaroids")
        ax1.errorbar(position2, residual_2, yerr=uncertainty2, linestyle="none", fmt='.')
        
        bl_3 = np.zeros(len(position3))
        ax2.plot(position3, bl_3, linestyle="--")
        ax2.plot(position3, uncertainty3, color='grey')
        ax2.plot(position3, uncertainty3* (-1), color='grey')
        ax2.set_title("3 Polaroids")
        # ax2.errorbar(position3, residual_3, yerr=uncertainty3, linestyle='none', fmt='.')
        ax2.plot(position3, residual_3, '.', linestyle='none')




        plt.show()

    