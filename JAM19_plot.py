import string
import warnings
from math import *

import numpy

from JAM19_subroutines_plot import *

if __name__ == '__main__':

    ####################################################################################
    # parameters
    ####################################################################################
    
    n_values = 100

    betas_1 = numpy.linspace(0.02, 0.98, n_values)  # softer layer (beta < 1)
    betas_2 = numpy.linspace(1.02, 2, n_values)  # stiffer layer (beta > 1)
    betas = numpy.concatenate((betas_1, betas_2), axis=0)
    # round some values in betas array for certain figures
    beta_rounded = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    ind_rounded = [9, 19, 29, 39, 49, 60, 70, 81, 91, 109, 118, 128, 138, 148, 159, 169, 179, 189]
    for i in range(18):
        betas[ind_rounded[i]] = beta_rounded[i]

    H_ms_1 = numpy.linspace(0.25, 10, int(n_H_m / 2))
    H_ms_2 = numpy.linspace(10.2, 30, int(n_H_m / 2))
    H_ms = numpy.concatenate((H_ms_1, H_ms_2), axis=0)
    # round some values of H_ms array for certain figures
    H_ms_rounded = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]  # Rounded H_m values
    ind_rounded = [0, 3, 5, 8, 18, 28, 38, 48, 58, 68, 79, 89, 99, 124, 149]
    for i in range(15): 
        H_ms[ind_rounded[i]] = H_ms_rounded[i]

    [critical_strains, threshold_strain, threshold_wavelength, anti_threshold_strain, anti_threshold_wavelength] = read_data()

    cmap = plt.get_cmap('viridis')
    new_cmap = truncate_colormap(cmap, 0.2, 1)

    plot_fig3(betas, H_ms, threshold_strain, anti_threshold_strain)
    plot_fig4a(betas, H_ms, threshold_strain)
    plot_fig4b(betas, H_ms, threshold_wavelength)
    plot_fig5a(critical_strains, betas)
    plot_fig5b(critical_strains)
    plot_fig6(critical_strains)
    plot_fig7a(H_ms, threshold_strain, anti_threshold_strain)
    plot_fig7b(threshold_wavelength, anti_threshold_wavelength)


