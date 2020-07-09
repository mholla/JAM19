import numpy
import multiprocessing
import matplotlib.pyplot as plt
from math import *
import string
import warnings
from JAM19_Subroutine import *

warnings.simplefilter('ignore')

# To decrease the solution time, the code uses 4 processing cores in parallel for each wrinkling mode (Total of 8 cores are needed).
# There are total of 200 Hm values. The results are devided into 4 parts; each part returns the critical and threshold values for all beta vaalues and only 50 values of Hm.
def solve(Hm_counter,parts_number,output_name,mode,H_l,beta_main,Hm_main,critical_strains,wavelengths,thresh_strain,thresh_wavelength):
    column = 0
    beta=0
    for j in range(0, 50, 1):
        Hm_counter = Hm_counter + 1 # recieves Hm_counter for each part
                                    # makes sure that Hm starts from the correct array component every time the "solve" function is being called .
        for i in range(len(beta_main)):
            beta = beta_main[i]
            Hm = Hm_main[Hm_counter]
            crit_strains = find_critical_values(H_l, Hm, plotroots, findroots, printoutput, beta, wavelengths, npts, tol, mode)
            # Saving all the critical strains
            # it is a 1000x10000 matrix. Each row corresponds to a certain value of normalized wavelength
            # and for every 200 columns (number of beta values) the value of Hm changes
            critical_strains[:, column] = strains_all[:]
            column = column + 1
            [thresh_wavelength, thresh_strain] = find_threshold_values(wavelengths, crit_strains, j, i, thresh_wavelength,thresh_strain)
    # Saving critical and threshold values for every combination of beta and Hm
    numpy.savetxt('%s_critical_strain_%d.txt' %(output_name,parts_number), critical_strains, fmt='%.18f')
    numpy.savetxt('%s_Threshold_strain_Hm_beta_%d.txt' %(output_name,parts_number), thresh_strain, fmt='%.18f')
    numpy.savetxt('%s_Threshold_wavelength_Hm_beta_%d.txt' %(output_name,parts_number), thresh_wavelength, fmt='%.18f')

if __name__ == '__main__':

    ####################################################################################
    # parameters
    ####################################################################################
    npts = 100  # (number of points between 0.1 and 1.1 to look for the existence of roots)
    n_wavelengths = 1500
    tol = 1.e-12 # this is a default value
    n_betas = 200
    n_Hm = 200
    wavelengths = numpy.logspace(-1., 3., num=n_wavelengths) # Known as "L" in the paper
    wavelengths = wavelengths[::-1]  # start at right end of the graph (where the distance between roots is larger)
    H_l = 1 # thicknes of each layer
    beta_main_1 = numpy.linspace(0.02, 0.98, int(n_betas/2))  # Softer layer (beta<1)
    beta_main_2 = numpy.linspace(1.02, 2, int(n_betas/2))     # Stiffer layer (beta>1)
    beta_main = numpy.concatenate((beta_main_1, beta_main_2), axis=0)
    Hm_main_1 = numpy.linspace(0.25, 10, int(n_Hm/2))
    Hm_main_2 = numpy.linspace(10.2, 30, int(n_Hm/2))
    Hm_main = numpy.concatenate((Hm_main_1, Hm_main_2), axis=0)
    ####################################################################
    # Rounding some of the values in beta_main and Hm_main arrays for generatring certain figures
    B = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9] # Rounded beta values
    counter = 0
    for i in [9, 19, 29, 39, 49, 60, 70, 81, 91, 109, 118, 128, 138, 148, 159, 169, 179, 189]: # The position of each rounded values in beta array
        beta_main[i] = B[counter]
        counter = counter + 1

    H = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # Rounded Hm values
    counter = 0
    for i in [0, 3, 5, 8, 18, 28, 38, 48, 58, 68, 79, 89, 99, 124, 149]: # The position of each rounded values in Hm array
        Hm_main[i] = H[counter]
        counter = counter + 1
    ####################################################################
    # parameters for output
    findroots = True  # only set to false for troubleshooting, using plotroots below
    plotroots = False  # save plot of absolute value of determinant at each n_wavelengths
    findminimum = True  # It is for when we need to find the threshold values
    printoutput = False  # print every root found at every n_wavelengths
    modes = [+1,-1] # Symmetric Determinant (+1) # Antiymmetric Determinant (-1)
    output_mode=['Sym','Antisym'] # Defines the name of result files
    critical_strains = numpy.zeros((n_wavelengths, int(int(n_betas) * int(n_Hm/4))))
    thresh_strain = numpy.zeros((n_betas, int(n_Hm/4)))
    thresh_wavelength = numpy.zeros((n_betas, int(n_Hm/4)))
    ####################################################################################
    # Solve for Symmetric mode
    # Each one runs on a separate processing core.
    F_Sym1 = multiprocessing.Process(target=solve, args=(0,1,output_mode[0],modes[0],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_sym1.start() # runs the "solve" function for values of Hm in Hm_main array from 0 to 49 components and all beta values
                   # and save the critical and thresold strains
    F_sym2 = multiprocessing.Process(target=solve, args=(49,2,output_mode[0],modes[0],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_sym2.start() # runs the "solve" function for values of Hm in Hm_main array from 49 to 99 components and all beta values
                   # and save the critical and thresold strains
    F_sym3 = multiprocessing.Process(target=solve, args=(99,3,output_mode[0],modes[0],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_sym3.start() # runs the "solve" function for values of Hm in Hm_main array from 100 to 149 components and all beta values
                   # and save the critical and thresold strains
    F_sym4 = multiprocessing.Process(target=solve, args=(149,4,output_mode[0],modes[0],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_sym4.start() # runs the "solve" function for values of Hm in Hm_main array from 149 to 199 components and all beta values
                   # and save the critical and thresold strains
    ####################################################################################
    # Solve for Antisymmetric mode
    # Each one runs on a separate processing core.
    F_Antisym1 = multiprocessing.Process(target=solve, args=(0,1,output_mode[1],modes[1],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_Antisym1.start()
    F_Antisym2 = multiprocessing.Process(target=solve, args=(49,2,output_mode[1],modes[1],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_Antisym2.start()
    F_Antisym3 = multiprocessing.Process(target=solve, args=(99,3,output_mode[1],modes[1],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_Antisym3.start()
    F_Antisym4 = multiprocessing.Process(target=solve, args=(149,4,output_mode[1],modes[1],H_l,beta_main,Hm_main,critical_strain,wavelengths,thresh_strain,thresh_wavelength))
    F_Antisym4.start()