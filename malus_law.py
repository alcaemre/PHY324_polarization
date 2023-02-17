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

def chi_square(f, popt, position, intensity, uncertainty, start=0, stop=-1):
    # omega = omega1
    # slice1 = [11, 62]   #optimal range of which the best curve_fit covers
    # n1 = trial1[1][slice1[0]:slice1[1]]
    # n1_fit = fitting_function(trial1[0], *popt1) +10*np.sin(trial1[0]/400) -5
    # n1_fit = n1_fit[slice1[0]:slice1[1]]
    # trial1_error = trial1_error[slice1[0]:slice1[1]]
    # chi2_1 = np.sum((n1 - n1_fit)**2/(trial1_error**2))
    # dof_1 = len(n1) - len(popt1)
    # prob1 = 1 - chi2.cdf(chi2_1, dof_1)

    
    slice = [start, stop]   #optimal range of which the best curve_fit covers

    n = intensity[slice[0]:slice[1]]
    n_fit = f(position, *popt) +10*np.sin(position/400) -5
    n_fit = n_fit[slice[0]:slice[1]]
    uncertainty = uncertainty[slice[0]:slice[1]]
    chi_2 = np.sum((n - n_fit)**2/(uncertainty**2))
    dof = len(n) - len(popt)
    probability = 1 - chi2.cdf(chi_2, dof)
    return probability


if __name__ == "__main__":
    plotting = False

    #loading data

    # print(read_file("malus_law_JA-ERA-2polarizers.txt"))
    position2, intensity2 = read_file("malus_law_JA-ERA-2polarizers.txt")
    position3, intensity3 = read_file("malus_law_JA-ERA-3polarizers.txt")
    # print(np.min(intensity3))

    # plot data

    if plotting:
        fig, (ax1,ax2) = plt.subplots(2,1)

        ax1.plot(position2, intensity2) # plotting 2 polaroids
        ax1.set_title("two polaroids")

        ax2.plot(position3, intensity3) # plotting 3 polaroids
        ax2.set_title("three polaroids")
        
    
    # fit data, including chi squared analysis

    popt2, pcov2 = op.curve_fit(malus_law_2, position2, intensity2, p0=(-3, 1, -0.01133))    # fitting 2 polaroids
    uncertainty2 = np.zeros(len(intensity2)) + 5e-2
    chi_squre_2 = chi_square(malus_law_2, popt2, position2, intensity2, uncertainty2, 0, len(intensity2)+1)
    print(chi_squre_2)

    popt3, pcov3 = op.curve_fit(malus_law_3, position3, intensity3, p0=(3.5, np.pi/4, 0.01213))    # fitting 3 polaroids
    uncertainty3 = np.zeros(len(intensity3)) + 5e-2
    chi_squre_3 = chi_square(malus_law_3, popt3, position3, intensity3, uncertainty3, 0, len(intensity3)+1)
    print(chi_squre_2)

    if plotting:
        ax1.plot(position2, malus_law_2(position2, popt2[0], popt2[1], popt2[2]))
        ax2.plot(position3, malus_law_3(position3, popt3[0], popt3[1], popt3[2]))


    # residual plot showing difference between fit and measurement

    plt.show()