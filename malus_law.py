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


def malus_law_2(phi, I0, phase_shift, phase_multiplier):
    """calculates the intensity 

    Args:
        theta (_type_): _description_
        I0 (_type_): _description_
        offset (_type_): _description_

    Returns:
        np.array: calculated intensity
    """
    intensity = I0*np.square(np.cos(phi * phase_multiplier + phase_shift)) # + offset
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

    # ANALYSIS FOR 2 POLARIZERS

    plotting_2 = True

    #loading data

    # print(read_file("malus_law_JA-ERA-2polarizers.txt"))
    position2_0, intensity2_0 = read_file("malus_law_JA-ERA-2polarizers.txt")

    position2 = position2_0[170:490]
    intensity2 = intensity2_0[170:490]

    uncertainty2 = np.zeros(len(intensity2)) + 8e-2
    ux_2 = np.zeros(len(position2)) + 5e-4

    uncertainty2_0 = np.zeros(len(intensity2_0)) + 8e-2
    ux_2_0 = np.zeros(len(position2_0)) + 5e-4
    
    # print(np.min(intensity3))
    
    # fit data, including chi squared analysis

    popt2, pcov2 = op.curve_fit(malus_law_2, position2, intensity2, p0=(3.5, -1.6, 0.6))    # fitting 2 polaroids
    perr2 = np.sqrt(np.diag(pcov2))
    print('for 2 polarizers, I0 is' + str(popt2[0]) + '+/-' + str(perr2[0]))

    fx_2 = malus_law_2(position2, popt2[0], popt2[1], popt2[2])
    # fx_2 = malus_law_2(position2, popt2[0], popt2[1])

    chi_squared_2 = chi_squared(fx_2, intensity2, uncertainty2, 3)
    print('for 2 polarizers, chi squared is' + str(chi_squared_2))

    # residual plot showing difference between fit and measurement
    residual_2 = calc_residual(fx_2, intensity2)

    if plotting_2:
        fig, (ax1,ax2,) = plt.subplots(2,1)

        ax1.errorbar(position2_0, intensity2_0, xerr=ux_2_0, yerr=uncertainty2_0, zorder=0, label='Original data', color='black')

        ax1.errorbar(position2, intensity2, xerr=ux_2, yerr=uncertainty2, zorder=1, label='Recorded Intensity') # plotting 2 polaroids
        ax1.set_title("Position vs. Intensity with Two Polaroids")
        ax1.set_ylabel("Intensity (V)")
        ax1.plot(position2, fx_2, zorder=2, label='Modelled Fit')

        # Bullshit excursion
        # cos_2_theta = np.square(np.cos(position2))
        # print(cos_2_theta)
        # ax2.errorbar(np.arange(len(intensity2)), intensity2, yerr=uncertainty2)
        # I_0 = 3.48
        # ax2.plot(np.arange(len(intensity2)), I_0*cos_2_theta)

        bl_2 = np.zeros(len(position2))
        ax2.plot(position2, bl_2, linestyle="--", color='black')
        ax2.set_title("2 Polaroids")
        # ax2.errorbar(position2, residual_2, yerr=uncertainty2, linestyle="none", fmt='.')
        ax2.plot(position2, uncertainty2, color='grey', label='Uncertainty Bounds')
        ax2.plot(position2, uncertainty2* (-1), color='grey')
        ax2.plot(position2, residual_2, '.', linestyle='none', label="Fit Minus Measurement")
        ax2.set_ylabel("Intensity (V)")
        ax2.set_xlabel("Position (radians)")
        

        ax1.legend()
        ax2.legend()

        
        plt.show()

    
    
        
        

    # ANALYSIS FOR 3 POLARIZERS

    position3, intensity3 = read_file("malus_law_JA-ERA-3polarizers.txt")
    uncertainty3 = np.zeros(len(intensity3)) + 3e-3
    ux_3 = np.zeros(len(position3)) + 5e-4
    

    plotting_3 = False

    popt3, pcov3 = op.curve_fit(malus_law_3, position3, intensity3, p0=(3.5, np.pi/4, 0.01213))    # fitting 3 polaroids
    perr3 = np.sqrt(np.diag(pcov3))
    

    fx_3 = malus_law_3(position3, popt3[0], popt3[1], popt3[2])

    chi_squared_3 = chi_squared(fx_3, intensity3, uncertainty3, 3)
    print(chi_squared_3)

    residual_3 = calc_residual(fx_3, intensity3)

    print('for 3 polarizers, I0 is' + str(popt3[0]) + '+/-' + str(perr3[0]))
    print('for 3 polarizers, chi squared is' + str(chi_squared_3))

    if plotting_3:
        fig, (ax1,ax2) = plt.subplots(2,1)
        ax1.errorbar(position3, intensity3, xerr=ux_3, yerr=uncertainty3, zorder=1, label='Recorded Intensity') # plotting 3 polaroids
        ax1.set_title("Position vs. Intensity with Three Polaroids")
        ax1.set_ylabel("Intensity (V)")
        
        ax1.plot(position3, fx_3, zorder=2, label = 'Modelled Fit')

        bl_3 = np.zeros(len(position3))
        ax2.plot(position3, bl_3, linestyle="--", color='black')
        ax2.plot(position3, uncertainty3, color='grey', label='Uncertainty Bounds')
        ax2.plot(position3, uncertainty3* (-1), color='grey')
        ax2.set_title("Residual Plot with 3 Polaroids")
        ax2.plot(position3, residual_3, '.', linestyle='none', label='Fit Minus Measurement')
        ax2.set_xlabel("Position (radians)")
        ax1.set_ylabel("Intensity (V)")

        ax1.legend()
        ax2.legend()


        plt.show()