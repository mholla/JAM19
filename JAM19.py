import multiprocessing
import string
import warnings
from math import *

import numpy

from JAM19_subroutines import *

warnings.simplefilter('ignore')

def solve(H_m_start, segment, mode_name, mode_factor, params, output_options):
    """ parallel calculations of threshold values for each wrinkling mode

    Parameters
    ----------
    H_m_start :
    segment :
    mode_name :
    mode_factor :
    params : 

    Returns
    -------
    None

    Notes
    -----
    Here we use four processing cores in parallel for each wrinkling mode to decrease the solution time.
    There are total of 200 H_m values, divided into four parts.
    Each part returns the critical and threshold values for all beta values and only 50 values of H_m.

    """
    [findroots, plotroots, printoutput] = output_options

    [H_l, H_ms, betas, wavelengths, npts, tol] = params

    critical_strains = numpy.zeros((len(wavelengths), int(int(len(betas)) * int(len(H_ms) / 4))))
    thresh_strains = numpy.zeros((len(betas), int(len(H_ms) / 4)))
    thresh_wavelengths = numpy.zeros((len(betas), int(len(H_ms) / 4)))

    column = 0
    for j in range(0, 50, 1):

        # start from the correct array component of H_m every time "solve" function is called
        H_m_start = H_m_start + 1

        for i in range(len(betas)):
            beta = betas[i]
            H_m = H_ms[H_m_start]
            crit_strains = find_critical_values(
                mode_factor,
                H_l,
                H_m,
                beta,
                wavelengths,
                npts,
                plotroots,
                findroots,
                printoutput,
                tol
            )

            critical_strains[:, column] = crit_strains[:]
            column = column + 1
            [thresh_wavelengths, thresh_strains] = find_threshold_values(
                wavelengths,
                crit_strains,
                j,
                i,
                thresh_wavelengths,
                thresh_strains
            )

    # save all critical strains (1000x10000 matrix)
    # each row corresponds to a certain value of normalized wavelength
    # for every 200 columns (number of beta values) the value of H_m changes
    numpy.savetxt('{name}_critical_strain_{number}.txt'.format(name=mode_name, number=segment),
                  critical_strains, fmt='%.8f')

    # save threshold values for every combination of beta and H_m
    numpy.savetxt('{name}_threshold_strain_{number}.txt'.format(name=mode_name, number=segment),
                  thresh_strains, fmt='%.8f')
    numpy.savetxt('{name}_threshold_wavelength_{number}.txt'.format(name=mode_name, number=segment),
                  thresh_wavelengths, fmt='%.8f')


if __name__ == '__main__':

    ####################################################################################
    # parameters
    ####################################################################################
    npts = 100  # (number of points between 0.1 and 1.1 to look for the existence of roots)
    n_wavelengths = 1500
    n_betas = 200
    n_H_m = 200
    tol = 1.e-12  # this is a default value

    wavelengths = numpy.logspace(-1., 3., num=n_wavelengths)  # "L" in the paper
    wavelengths = wavelengths[::-1]  # start at right end of the graph (where the distance between roots is larger)

    H_l = 1  # thickness of each layer
    H_ms_1 = numpy.linspace(0.25, 10, int(n_H_m / 2))
    H_ms_2 = numpy.linspace(10.2, 30, int(n_H_m / 2))
    H_ms = numpy.concatenate((H_ms_1, H_ms_2), axis=0)
    # round some values of H_ms array for certain figures
    H_ms_rounded = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]  # Rounded H_m values
    ind_rounded = [0, 3, 5, 8, 18, 28, 38, 48, 58, 68, 79, 89, 99, 124, 149]
    for i in range(15): 
        H_ms[ind_rounded[i]] = H_ms_rounded[i]

    betas_1 = numpy.linspace(0.02, 0.98, int(n_betas / 2))  # softer layer (beta < 1)
    betas_2 = numpy.linspace(1.02, 2, int(n_betas / 2))  # stiffer layer (beta > 1)
    betas = numpy.concatenate((betas_1, betas_2), axis=0)
    # round some values in betas array for certain figures
    beta_rounded = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    ind_rounded = [9, 19, 29, 39, 49, 60, 70, 81, 91, 109, 118, 128, 138, 148, 159, 169, 179, 189]
    for i in range(18):
        betas[ind_rounded[i]] = beta_rounded[i]

    params = [H_l, H_ms, betas, wavelengths, npts, tol]

    # parameters for output
    findroots = True  # only set to false for troubleshooting, using plotroots below
    plotroots = False  # save plot of absolute value of determinant at each n_wavelengths
    printoutput = False  # print every root found at every n_wavelengths

    output_options = [findroots, plotroots, printoutput]

    ####################################################################################
    # symmetric mode
    ####################################################################################

    F_sym1 = multiprocessing.Process(target=solve, args=(0, 1, 'sym', 1, params, output_options))
    F_sym1.start()
    #
    F_sym2 = multiprocessing.Process(target=solve, args=(49, 2, 'sym', 1, params,npts))
    F_sym2.start()

    F_sym3 = multiprocessing.Process(target=solve, args=(99, 3, 'sym', 1, params, npts))
    F_sym3.start()

    F_sym4 = multiprocessing.Process(target=solve, args=(149, 4, 'sym', 1, params, npts))
    F_sym4.start()
    #
    # ####################################################################################
    # # antisymmetric mode
    # ####################################################################################
    #
    F_antisym1 = multiprocessing.Process(target=solve, args=(0, 1, 'antisym', -1, params, npts))
    F_antisym1.star()

    F_antisym2 = multiprocessing.Process(target=solve, args=(49, 2, 'antisym', -1, params, npts))
    F_antisym2.start()

    F_antisym3 = multiprocessing.Process(target=solve, args=(99, 3, 'antisym', -1, params, npts))
    F_antisym3.start()

    F_antisym4 = multiprocessing.Process(target=solve, args=(149, 4, 'antisym', -1, params, npts))
    F_antisym4.start()
