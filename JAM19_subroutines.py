import string
import warnings
from math import *

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy

warnings.simplefilter('ignore')

def determinant(mode_factor, H_l, H_m, beta, wavelength, lam):
    """ Return determinant of 6x6 coefficient matrix

    Parameters
    ----------
    mode_factor : integer
        determines symmetric or antisymmetric modes of wrinkling configuration
    H_l :
    H_m : float
        half of the distance between two layers
    beta : float
        stiffness ratio (layer/matrix)
    wavelength : float
        wavelength
    lam : float
        values of compression in 1 direction

    Returns
    -------
    dd : float
        determinant of coefficient matrix.

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """

    L = wavelength
    LAMBDA = 1. + lam ** 4.
    AA = numpy.zeros((8, 8), dtype='float64')

    try:
        AA[0][0] = -exp(-2. * H_m / L * pi / lam ** 2.)
        AA[0][1] = +exp(+2. * H_m / L * pi / lam ** 2.)
        AA[0][2] = -(lam ** 2.) * exp(-2. * H_m / L * pi)
        AA[0][3] = +(lam ** 2.) * exp(+2. * H_m / L * pi)
        AA[0][4] = exp(-2. * H_m / L * pi / lam ** 2.) - exp(2. * H_m / L * pi / lam ** 2.) * mode_factor
        AA[0][5] = (lam ** 2.) * (exp(-2. * H_m / L * pi) - exp(2. * H_m / L * pi) * mode_factor)
        AA[0][6] = 0.
        AA[0][7] = 0.

        AA[1][0] = -exp(-2. * H_m / L * pi / lam ** 2.)
        AA[1][1] = -exp(+2. * H_m / L * pi / lam ** 2.)
        AA[1][2] = -exp(-2. * H_m / L * pi)
        AA[1][3] = -exp(+2. * H_m / L * pi)
        AA[1][4] = (exp(-2. * H_m / L * pi / lam ** 2.) + exp(2. * H_m / L * pi / lam ** 2.) * mode_factor)
        AA[1][5] = (exp(-2. * H_m / L * pi) + exp(2. * H_m / L * pi) * mode_factor)
        AA[1][6] = 0.
        AA[1][7] = 0.

        AA[2][0] = 4. * pi / L / lam ** 3. * beta * exp(-2. * H_m / L * pi / lam ** 2.)
        AA[2][1] = 4. * pi / L / lam ** 3. * beta * exp(+2. * H_m / L * pi / lam ** 2.)
        AA[2][2] = 2. * pi / L / lam ** 3. * LAMBDA * beta * exp(-2. * H_m / L * pi)
        AA[2][3] = 2. * pi / L / lam ** 3. * LAMBDA * beta * exp(+2. * H_m / L * pi)
        AA[2][4] = -4. * pi / L / lam ** 3. * (
                    exp(-2. * H_m / L * pi / lam ** 2.) + exp(2. * H_m / L * pi / lam ** 2.) * mode_factor)
        AA[2][5] = -2. * pi / L / lam ** 3. * LAMBDA * (exp(-2. * H_m / L * pi) + exp(2. * H_m / L * pi) * mode_factor)
        AA[2][6] = 0.
        AA[2][7] = 0.

        AA[3][0] = -2. * pi / L / lam ** 3. * LAMBDA * beta * exp(-2. * H_m / L * pi / lam ** 2.)
        AA[3][1] = +2. * pi / L / lam ** 3. * LAMBDA * beta * exp(+2. * H_m / L * pi / lam ** 2.)
        AA[3][2] = -4. * pi / L / lam * beta * exp(-2. * H_m / L * pi)
        AA[3][3] = +4. * pi / L / lam * beta * exp(+2. * H_m / L * pi)
        AA[3][4] = 2. * pi / L / lam ** 3. * LAMBDA * (
                    exp(-2. * H_m / L * pi / lam ** 2.) - exp(2. * H_m / L * pi / lam ** 2.) * mode_factor)
        AA[3][5] = 4. * pi / L / lam * (exp(-2. * H_m / L * pi) - exp(2. * H_m / L * pi) * mode_factor)
        AA[3][6] = 0.
        AA[3][7] = 0.

        AA[4][0] = -exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[4][1] = +exp(+2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[4][2] = -(lam ** 2.) * exp(-2. * (H_m + H_l) / L * pi)
        AA[4][3] = +(lam ** 2.) * exp(+2. * (H_m + H_l) / L * pi)
        AA[4][4] = 0.
        AA[4][5] = 0.
        AA[4][6] = exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[4][7] = lam ** 2. * exp(-2. * (H_m + H_l) / L * pi)

        AA[5][0] = -exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[5][1] = -exp(+2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[5][2] = -exp(-2. * (H_m + H_l) / L * pi)
        AA[5][3] = -exp(+2. * (H_m + H_l) / L * pi)
        AA[5][4] = 0.
        AA[5][5] = 0.
        AA[5][6] = exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[5][7] = exp(-2. * (H_m + H_l) / L * pi)

        AA[6][0] = 4. * pi / L / lam ** 3. * beta * exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[6][1] = 4. * pi / L / lam ** 3. * beta * exp(+2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[6][2] = 2. * pi / L / lam ** 3. * LAMBDA * beta * exp(-2. * (H_m + H_l) / L * pi)
        AA[6][3] = 2. * pi / L / lam ** 3. * LAMBDA * beta * exp(+2. * (H_m + H_l) / L * pi)
        AA[6][4] = 0.
        AA[6][5] = 0.
        AA[6][6] = -4. * pi / L / lam ** 3. * exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[6][7] = -2. * pi / L / lam ** 3. * LAMBDA * exp(-2. * (H_m + H_l) / L * pi)

        AA[7][0] = -2. * pi / L / lam ** 3. * LAMBDA * beta * exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[7][1] = +2. * pi / L / lam ** 3. * LAMBDA * beta * exp(+2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[7][2] = -4. * pi / L / lam * beta * exp(-2. * (H_m + H_l) / L * pi)
        AA[7][3] = +4. * pi / L / lam * beta * exp(+2. * (H_m + H_l) / L * pi)
        AA[7][4] = 0.
        AA[7][5] = 0.
        AA[7][6] = 2. * pi / L / lam ** 3. * LAMBDA * exp(-2. * (H_m + H_l) / L * pi / lam ** 2.)
        AA[7][7] = 4. * pi / L / lam * exp(-2. * (H_m + H_l) / L * pi)

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print("infinity at lam1 = ", lam1)
    except (OverflowError):
        dd = None
        # print("overflow at lam1 = ", lam1)
    return dd


def Ridder(mode_factor, H_l, H_m, beta, wavelength, a, b, determinant, tol):
    """ Uses Ridders' method to find critical strain (between a and b) for given wavelength kh

    Parameters
    ----------
    mode_factor :
    H_l :
    H_m : float
        half of the distance between two layers
    beta : float
        stiffness ratio (layer/matrix)
    wavelength : float
        wavelength
    a, b : float
        upper and lower brackets of lambda for Ridders' method
    determinant : function
        return values of determinat for given parameters
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
    nmax : int
        maximum number of iterations before exiting

    Returns
    -------
    x_ridder : float
        value of axial compression, lambda, that satisfies Eq. 24
    i : int
        number of iterations before lambda_crit was found

    Notes
    -----
    Based on based on https://en.wikipedia.org/wiki/Ridders%27_method
    """

    nmax = 50

    fa = determinant(mode_factor, H_l, H_m, beta, wavelength, a)
    fb = determinant(mode_factor, H_l, H_m, beta, wavelength, b)

    if fa == 0.0:
        # print("lower bracket is root")
        return a, 0
    if fb == 0.0:
        # print("upper bracket is root")
        return b, 0
    if fa * fb > 0.0:
        # print("Root is not bracketed")
        return None, None

    for i in range(nmax):
        c = 0.5 * (a + b)
        fc = determinant(mode_factor, H_l, H_m, beta, wavelength, c)

        s = sqrt(fc ** 2. - fa * fb)
        if s == 0.0:
            return None, i

        dx = (c - a) * fc / s
        if (fa - fb) < 0.0: dx = -dx
        x_ridder = c + dx

        fx = determinant(mode_factor, H_l, H_m, beta, wavelength, x_ridder)

        # check for convergence
        if i > 0:
            if abs(x_ridder - xOld) < tol * max(abs(x_ridder), 1.0):
                return x_ridder, i
        xOld = x_ridder

        # rebracket root
        if fc * fx > 0.0:
            if fa * fx < 0.0:
                b = x_ridder
                fb = fx
            else:
                a = x_ridder
                fa = fx
        else:
            a = c
            b = x_ridder
            fa = fc
            fb = fx

    res = abs(x_ridder - xOld) / max(abs(x_ridder), 1.0)

    print("Too many iterations, res = {res}".format(res=res))
    return None, nmax


def find_critical_values(mode_factor, H_l, H_m, beta, wavelengths, npts, plotroots, findroots, printoutput, tol):
    """ Finds critical strain for each specified wavelength

    Parameters
    ----------
    mode_factor : integer
        determines symmetric or antisymmetric modes of wrinkling configuration
    H_l :
    H_m :
    beta : float
        stiffness ratio (layer/matrix)
    wavelengths : list of floats
        list of wavelengths for which to calculate determinant
    npts : int
        number of strain values to consider when checking for existence of roots
    plotroots : boolean
        whether or not to plot lines showing positive or negative value at all npts for each wavelength
    findroots : boolean
        whether or not to find the values of each root (set to False and plotroots to True to see root plots)
    printoutput : boolean
        whether or not to print every root found at every wavelength
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance

    Returns
    -------
    strains : list of floats
        list of all critical strains which satisfy Eq. 24 for one beta and one H_m values.

    Notes
    -----
    Called by JAM19.py
    """

    lam_min = 0.01
    lam_max = 0.97
    strains = []

    for wavelength in wavelengths:

        [rootexists, a, c] = check_roots(mode_factor, H_l, H_m, beta, wavelength, lam_min, lam_max, npts, plotroots, determinant)

        if findroots:
            strains = find_roots(mode_factor, H_l, H_m, beta, wavelength, strains, rootexists, a, c, printoutput, tol, determinant)

    if findroots:
        return strains


def check_roots(mode_factor, H_l, H_m, beta, wavelength, lam_min, lam_max, npts, plotroots, determinant):
    """ Calculates the value and/or sign of the determinant at every lambda 

    Parameters
    ----------
    mode_factor : integer
        determines symmetric or antisymmetric modes of wrinkling configuration
    H_l: float
        thickness of each layer
    H_m: float
        half of the distance between the layers
    beta : float
        stiffness ratio (layer/matrix)
    wavelength : float
        wavelength
    lam_min, lam_max : float
        minimum and maximum values of lambda to check for existence of a root
    npts : int
        number of points between lam_min and lam_max at which to calculate determinant
    plotroots : boolean
        plot lines showing positive or negative value at all npts
    determinant : function
        return values of determinant for given parameters

    Returns
    -------
    rootexists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    a : float
        lower bracket
    c : float
        upper bracket
    """

    lams = numpy.linspace(lam_min, lam_max, npts)[::-1]
    dds = numpy.zeros(npts, dtype='float64')
    dds_abs = numpy.zeros(npts, dtype='float64')

    rootexists = False
    a = 0.  # lower bracket
    c = 0.  # upper bracket

    # move backwards, in order of decreasing strain and report values bracking highest root
    for i in range(npts):

        lam = lams[i]
        dds[i] = determinant(mode_factor, H_l, H_m, beta, wavelength, lam)

        # if encountering NaN before finding root:
        if isnan(dds[i]) and not rootexists:
            dds[i] = 0.
            rootexists = False
            a = 0.
            c = 0.

        # otherwise, step through
        if i > 0 and not rootexists:
            dds_abs[i] = dds[i] / abs(dds[i])
            # if encountering root
            if dds[i] * dds[i - 1] < 0.:  # sign change
                rootexists = True
                a = lams[i]
                c = lams[i - 1]
                break

    if plotroots:
        plt.figure()
        plt.axis([0, 1.1, -100000, 100000])
        plt.xlabel('$\lambda_1$')
        plt.ylabel('energy')
        plt.axvline(x=1., linestyle='--', color='k')
        plt.axhline(y=0., linestyle='--', color='k')
        plt.plot(lams, dds, color='b', linestyle='-')

    return rootexists, a, c


def find_roots(mode_factor, H_l, H_m, beta, wavelength, strains, rootexists, a, c, printoutput, tol, determinant):
    # returns compressive strain (1 - lam) at which buckling occurs for the given parameters

    if printoutput:
        print("\nx = %0.2f, a = %f, c = %f" % (wavelength, a, c))

    if rootexists:
        [lam1, n] = Ridder(mode_factor, H_l, H_m, beta, wavelength, a, c, determinant, tol)
        if printoutput: print("lam = %0.5f, n = %d" % (lam1, n))
    else:  # no root means that system is infinitely stable
        lam1 = 0.
        if printoutput: print("wavelength = %0.2f, no root" % wavelength)

    strains.append(1. - lam1)

    return strains


def find_threshold_values(wavelengths, crit_strains, j, i, thresh_wavelength, thresh_strain):
    """ Finds threshold critical strain and corresponding threshold wavelength

    Parameters
    ----------
    wavelengths : list of floats
        list of wavelengths
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength
    j :
    i :
    thresh_wavelength :
    thresh_strain :

    Returns
    -------
    thresh_wavelength : float
        critical wavelength (corresponding to critical strain)
    thresh_strain : float
        minimum critical strain
    
    Notes
    -----
    If there is no true minimum, it returns the zero for the wavelength and the zero wavelength strain
    """

    index = 0

    if crit_strains[0] == min(crit_strains):
        start = False
    else:
        start = True

    for x in range(1, len(wavelengths) - 1):
        if crit_strains[x] == 1.0 and start:
            index = x + 1
        elif crit_strains[x] == 1.0:
            index = index
        elif crit_strains[x] < crit_strains[index]:
            index = x
            start = False

    # remove no-root results
    strains_masked = numpy.ma.masked_greater(crit_strains, 0.999)
    if max(strains_masked) - min(strains_masked) < 0.001:
        index = 0

    # if very small wavelength is dominating, check to see if infinity is equally favorable
    if index > 0.95 * len(wavelengths):
        if abs(crit_strains[0] - strains[index]) < 0.001:
            index = 0

    thresh_wavelength[i, j] = wavelengths[index]
    thresh_strain[i, j] = crit_strains[index]

    return thresh_wavelength, thresh_strain
