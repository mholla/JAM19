import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib as mpl
from math import *
import string
import warnings
import matplotlib.colors as colors

warnings.simplefilter('ignore')

def determinant(H_l,Hm,lam,beta,wavelength,mode):
    """ Return determinant of 6x6 coefficient matrix

    Parameters
    ----------
    lam : float
        values of compression in 1 direction
    beta : float
        stiffness ratio (layer/matrix)
    Hm : float
        half of the distance between two layers
    wavelength : float
        wavelength
    mode : integer
        determines symmetric or antisymmetric modes of wrinkling configuration

    Returns
    -------
    dd : float
        determinant of coefficient matrix.

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """
    L=wavelength
    LAMBDA = 1. + lam**4.
    AA = numpy.zeros((8,8), dtype='float64')

    try:
        AA[0][0] = -                     exp(-2. * Hm/L * pi / lam**2. )
        AA[0][1] =                       exp( 2. * Hm/L * pi / lam**2. )
        AA[0][2] = - (lam**2.) *         exp(-2. * Hm/L * pi )
        AA[0][3] =   (lam**2.) *         exp( 2. * Hm/L * pi)
        AA[0][4] =   exp(-2. * Hm/L * pi / lam**2.)    - exp(2. * Hm/L * pi / lam**2.)*mode
        AA[0][5] =   (lam**2.) * ( exp(-2. * Hm/L * pi)- exp(2. * Hm/L * pi)*mode)
        AA[0][6] =   0.
        AA[0][7] =   0.

        AA[1][0] = - exp(-2. * Hm/L * pi / lam**2. )
        AA[1][1] = - exp( 2. * Hm/L * pi / lam**2. )
        AA[1][2] = - exp(-2. * Hm/L * pi )
        AA[1][3] = - exp( 2. * Hm/L * pi )
        AA[1][4] =   ( exp(-2. * Hm/L * pi / lam**2. )+ exp( 2. * Hm/L * pi / lam**2. )*mode )
        AA[1][5] =   ( exp(-2. * Hm/L * pi )          + exp( 2. * Hm/L * pi )*mode )
        AA[1][6] =   0.
        AA[1][7] =   0.

        AA[2][0] =   4. * pi/L / lam**3. * beta *          exp(-2. * Hm/L * pi / lam**2. )
        AA[2][1] =   4. * pi/L / lam**3. * beta *          exp( 2. * Hm/L * pi / lam**2. )
        AA[2][2] =   2. * pi/L / lam**3. * LAMBDA * beta * exp(-2. * Hm/L * pi )
        AA[2][3] =   2. * pi/L / lam**3. * LAMBDA * beta * exp( 2. * Hm/L * pi )
        AA[2][4] = - 4. * pi/L / lam**3. * ( exp(-2. * Hm/L * pi / lam**2. )+ exp( 2. * Hm/L * pi / lam**2. )*mode )
        AA[2][5] = - 2. * pi/L / lam**3. * LAMBDA * ( exp(-2. * Hm/L * pi ) + exp( 2. * Hm/L * pi )*mode )
        AA[2][6] =   0.
        AA[2][7] =   0.

        AA[3][0] = - 2. * pi/L / lam**3. * LAMBDA    * beta *     exp(-2. * Hm/L * pi / lam**2. )
        AA[3][1] =   2. * pi/L / lam**3. * LAMBDA    * beta *     exp( 2. * Hm/L * pi / lam**2. )
        AA[3][2] = - 4. * pi/L / lam     * beta      *  exp(-2. * Hm/L * pi )
        AA[3][3] =   4. * pi / L / lam     * beta      * exp(2. * Hm / L * pi)
        AA[3][4] =   2. * pi/L / lam**3. * LAMBDA    * (exp(-2. * Hm/L * pi / lam**2. )- exp( 2. * Hm/L * pi / lam**2. )*mode)
        AA[3][5] =   4. * pi / L / lam     * (exp(-2. * Hm / L * pi) - exp(2. * Hm / L * pi)*mode)
        AA[3][6] =   0.
        AA[3][7] =   0.

        AA[4][0] = - exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[4][1] =   exp( 2. * (Hm+H_l)/L * pi / lam**2. )
        AA[4][2] = - (lam**2.) * exp(-2. * (Hm+H_l)/L * pi )
        AA[4][3] =   (lam**2.) * exp( 2. * (Hm+H_l)/L * pi )
        AA[4][4] =   0.
        AA[4][5] =   0.
        AA[4][6] =   exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[4][7] =   lam**2. * exp(-2. * (Hm+H_l)/L * pi )


        AA[5][0] = - exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[5][1] = - exp( 2. * (Hm+H_l)/L * pi / lam**2. )
        AA[5][2] = - exp(-2. * (Hm+H_l)/L * pi )
        AA[5][3] = - exp( 2. * (Hm+H_l)/L * pi )
        AA[5][4] =   0.
        AA[5][5] =   0.
        AA[5][6] =   exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[5][7] =   exp(-2. * (Hm+H_l)/L * pi )

        AA[6][0] =   4. * pi/L /lam**3. * beta          * exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[6][1] =   4. * pi/L /lam**3. * beta          * exp( 2. * (Hm+H_l)/L * pi / lam**2. )
        AA[6][2] =   2. * pi/L /lam**3. * LAMBDA * beta * exp(-2. * (Hm+H_l)/L * pi )
        AA[6][3] =   2. * pi/L /lam**3. * LAMBDA * beta * exp( 2. * (Hm+H_l)/L * pi )
        AA[6][4] =   0.
        AA[6][5] =   0.
        AA[6][6] = - 4. * pi/L /lam**3.          * exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[6][7] = - 2. * pi/L /lam**3. * LAMBDA * exp(-2. * (Hm+H_l)/L * pi )

        AA[7][0] = - 2. * pi/L /lam**3. * LAMBDA * beta    * exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[7][1] =   2. * pi/L /lam**3. * LAMBDA * beta    * exp( 2. * (Hm+H_l)/L * pi / lam**2. )
        AA[7][2] = - 4. * pi / L / lam  * beta * exp(-2. * (Hm + H_l) / L * pi)
        AA[7][3] =   4. * pi / L / lam    * beta * exp(2. * (Hm + H_l) / L * pi)
        AA[7][4] =   0.
        AA[7][5] =   0.
        AA[7][6] =   2. * pi/L /lam**3. * LAMBDA    * exp(-2. * (Hm+H_l)/L * pi / lam**2. )
        AA[7][7] =   4. * pi / L / lam  * exp(-2. * (Hm + H_l) / L * pi)

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print("infinity at lam1 = ", lam1)
    except (OverflowError):
        dd = None
        # print("overflow at lam1 = ", lam1)
    return dd

def Ridder(H_l,Hm,a,b,determinant,beta,wavelength,tol,mode):
    """ Uses Ridders' method to find critical strain (between a and b) for given wavelength kh

    Parameters
    ----------
    a,b : float
        upper and lower brackets of lambda for Ridders' method
    determinant : function
        return values of determinat for given parameters
    beta : float
        stiffness ratio (layer/matrix)
    wavelength : float
        wavelength
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
    nmax : int
        maximum number of iterations before exiting

    Returns
    -------
    x_riddar : float
        value of axial compression, lambda, that satisfies Eq. 24
    i : int
        number of iterations before lambda_crit was found

    Notes
    -----
    Based on based on https://en.wikipedia.org/wiki/Ridders%27_method
    """

    nmax = 50

    fa = determinant(H_l,Hm,a,beta,wavelength,mode)
    fb = determinant(H_l,Hm,b,beta,wavelength,mode)

    if fa == 0.0: 
        # print("lower bracket is root")
        return a, 0
    if fb == 0.0: 
        # print("upper bracket is root")
        return b, 0
    if fa*fb > 0.0: 
        # print("Root is not bracketed")
        return None, None

    for i in range(nmax):
        c = 0.5*(a + b)
        fc = determinant(H_l,Hm,c,beta,wavelength,mode)
        
        s = sqrt(fc**2. - fa*fb)
        if s == 0.0: 
            return None, i

        dx = (c - a)*fc/s
        if (fa - fb) < 0.0: dx = -dx
        x_riddar = c + dx

        fx = determinant(H_l,Hm,x_riddar,beta,wavelength,mode)

        # check for convergence
        if i > 0: 
            if abs(x_riddar - xOld) < tol*max(abs(x_riddar),1.0):
                return x_riddar, i
        xOld = x_riddar

        # rebracket root
        if fc*fx > 0.0:
            if fa*fx < 0.0: b = x_riddar; fb = fx
            else:           a = x_riddar; fa = fx
        else:
            a = c; b = x_riddar; fa = fc; fb = fx

    res = abs(x_riddar-xOld)/max(abs(x_riddar),1.0)

    print('Too many iterations, res = %e' %res)
    return None, nmax

def find_critical_values(H_l,Hm,plotroots,findroots,output,beta,wavelengths,npts,tol,mode):
    """ Finds critical strain for each specified wavelength

    Parameters
    ----------
    mode : integer
        determines symmetric or antisymmetric modes of wrinkling configuration
    beta : float
        stiffness ratio (layer/matrix)
    wavelengths : list of floats
        list of wavelengths for which to calculate determinant
    lam_min, lam_max : floats
        min and max min_strains to consider when checking for existence of roots
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
        list of all critical starins which satisfy Eq. 24 for one beta and one Hm values.

    Notes
    -----
    Called by 19JAM.py

    """
    lam_min = 0.01
    lam_max = 0.97
    strains = []

    for wavelength in wavelengths:

        [root,a,c] = check_roots(H_l,Hm,wavelength,lam_min,lam_max,npts,beta,plotroots,determinant,mode)

        if findroots:
            strains = find_roots(H_l,Hm,wavelength,output,root,strains,a,c,beta,tol,determinant,mode)

    if findroots:
        return strains

def check_roots(H_l,Hm,wavelength,lam_min,lam_max,npts,beta,plotroots,determinant,mode):
    """ Calculates the value and/or sign of the determinant at every lambda 

    Parameters
    ----------
    lam_min, lam_max : float
        minimum and maximum values of lambda to check for existence of a root
    npts : int
        number of points between lam_min and lam_max at which to calculate determinant
    determinant : function
        return values of determinat for given parameters
    beta : float
        stiffness ratio (layer/matrix)
    wavelength : float
        wavelength
    H_l: float
        thinkess of each layer
    Hm: float
        half of the distance between the layers
    mode : integer
        determines symmetric or antisymmetric modes of wrinkling configuration
    plotroots : boolean
        plot lines showing positive or negative value at all npts

    Returns
    -------
    a : float
        lower bracket
    c : float
        upper bracker
    root : boolean
        boolean value indicating whether or not a root (sign change) was detected
    """

    lams = numpy.linspace(lam_min,lam_max,npts)[::-1]
    dds = numpy.zeros(npts, dtype='float64')
    dds_abs = numpy.zeros(npts, dtype='float64')

    root = False 
    a = 0.      # lower bracket
    c = 0.      # upper bracker
    
    # move backwards, in order of decreasing strain and report values bracking highest root
    for i in range(n):

        lam = lams[i]
        dds[i] = determinant(H_l,Hm,lam,beta,wavelength,mode)
        
        # if encountering NaN before finding root:
        if isnan(dds[i]) and not root:
            dds[i] = 0.
            root = False
            a = 0.
            c = 0.
            
        # otherwise, step through
        if i > 0 and not root:
            dds_abs[i] = dds[i]/abs(dds[i])
            # if encountering root
            if dds[i]*dds[i-1] < 0.: # sign change
                root = True
                a = lams[i]
                c = lams[i-1]

    if plotroots:     
        plt.figure()
        plt.axis([0, 1.1, -100000, 100000])
        plt.xlabel('$\lambda_1$')
        plt.ylabel('energy')
        plt.axvline(x=1., linestyle='--', color='k')
        plt.axhline(y=0., linestyle='--', color='k')
        plt.plot(lams,dds,color='b', linestyle='-')

    return root, a, c


def find_roots(H_l,Hm,wavelength,output,root,strains,a,c,beta,tol,determinant,mode):
    # returns compressive strain (1 - lam) at which buckling occurs for the given parameters

    minimum = a

    if output: 
        print("\nx = %0.2f, a = %f, c = %f" %(wavelength, a, c))
    
    if root:
        [lam1, n] = Ridder(H_l,Hm,a, c, determinant, beta, wavelength, tol,mode)
        if output: print("lam = %0.5f, n = %d" %(lam1, n))
    else: # no root means there is no strain that can cause this system to buckle.  Give it a high strain
        lam1 = 0.
        n = 1.
        if output: print("wavelength = %0.2f, no root" %wavelength)

    strains.append(1. - lam1)

    return strains

def find_threshold_values(wavelengths, crit_strains, j,i,thresh_wavelength,thresh_strain):
    """ Finds threshold critical strain and corresponding threshold wavelength

    Parameters
    ----------
    wavelengths : list of floats
        list of wavelengths
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength

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

    if crit_strains[0] == min(crit_strains):  # we're done
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

    thresh_wavelength[i,j] = wavelengths[index]
    thresh_strain[i,j] = crit_strains[index]

    return thresh_wavelength, thresh_strain