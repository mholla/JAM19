import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def read_data(): 
    # load the results for each parts and concatenate them.

    crit_strain_1 = numpy.loadtxt('sym_critical_strain_1.txt')
    crit_strain_2 = numpy.loadtxt('sym_critical_strain_2.txt')
    crit_strain_3 = numpy.loadtxt('sym_critical_strain_3.txt')
    crit_strain_4 = numpy.loadtxt('sym_critical_strain_4.txt')
    critical_strains = numpy.concatenate((crit_strain_1, crit_strain_2, crit_strain_3, crit_strain_4), axis=1)

    thresh_strain_1 = numpy.loadtxt('sym_threshold_strain_1.txt')
    thresh_strain_2 = numpy.loadtxt('sym_threshold_strain_2.txt')
    thresh_strain_3 = numpy.loadtxt('sym_threshold_strain_3.txt')
    thresh_strain_4 = numpy.loadtxt('sym_threshold_strain_4.txt')
    threshold_strain = numpy.concatenate((thresh_strain_1, thresh_strain_2, thresh_strain_3, thresh_strain_4), axis=1)
    threshold_strain = numpy.transpose(threshold_strain)

    thresh_wavelength_1 = numpy.loadtxt('sym_threshold_wavelength_1.txt')
    thresh_wavelength_2 = numpy.loadtxt('sym_threshold_wavelength_2.txt')
    thresh_wavelength_3 = numpy.loadtxt('sym_threshold_wavelength_3.txt')
    thresh_wavelength_4 = numpy.loadtxt('sym_threshold_wavelength_4.txt')
    threshold_wavelength = numpy.concatenate((thresh_wavelength_1, thresh_wavelength_2, thresh_wavelength_3, thresh_wavelength_4), axis=1)
    threshold_wavelength = numpy.transpose(threshold_wavelength)

    anti_thresh_strain_1 = numpy.loadtxt('antisym_threshold_strain_1.txt')
    anti_thresh_strain_2 = numpy.loadtxt('antisym_threshold_strain_2.txt')
    anti_thresh_strain_3 = numpy.loadtxt('antisym_threshold_strain_3.txt')
    anti_thresh_strain_4 = numpy.loadtxt('antisym_threshold_strain_4.txt')
    anti_threshold_strain = numpy.concatenate((anti_thresh_strain_1, anti_thresh_strain_2, anti_thresh_strain_3, anti_thresh_strain_4), axis=1)
    anti_threshold_strain = numpy.transpose(anti_threshold_strain)

    anti_thresh_wavelength_1 = numpy.loadtxt('antisym_threshold_wavelength_1.txt')
    anti_thresh_wavelength_2 = numpy.loadtxt('antisym_threshold_wavelength_2.txt')
    anti_thresh_wavelength_3 = numpy.loadtxt('antisym_threshold_wavelength_3.txt')
    anti_thresh_wavelength_4 = numpy.loadtxt('antisym_threshold_wavelength_4.txt')
    anti_threshold_wavelength = numpy.concatenate((anti_thresh_wavelength_1, anti_thresh_wavelength_2, anti_thresh_wavelength_3, anti_thresh_wavelength_4), axis=1)
    anti_threshold_wavelength = numpy.transpose(anti_threshold_wavelength)

    return critical_strains, threshold_strain, threshold_wavelength, anti_threshold_strain, anti_threshold_wavelength


def plot_fig3(betas, H_ms, threshold_strain, anti_threshold_strain): 
    Diff_Strain = numpy.zeros((200, 200))
    # iterate through rows
    for i in range(len(strain)):
        # iterate through columns
        for j in range(len(strain[0])):
            Diff_Strain[i][j] = anti_threshold_strain[i][j] - threshold_strain[i][j]
            if Diff_Strain[i][j] == 0:
                Diff_Strain[i][j] = 0.0001

    fig3 = plt.figure(3, figsize=(20, 8))
    ax = fig3.add_subplot(111)
    ax.contourf(betas, H_ms, Diff_Strain, 20, cmap=cm.viridis, vmin=0, vmax=0.2, interpolation='none')
    ax.minorticks_on()
    minorLocator = MultipleLocator(0.05)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.axvline(x=1, color='w', linewidth=5)
    ax.set_xlim(0.01, 2)
    ax.set_ylim(0, 20)
    ax.yaxis.set_tick_params(labelsize=24, size=5)
    ax.xaxis.set_tick_params(labelsize=24, size=5)
    ax.yaxis.set_ticks([0, 5, 10, 15, 20])
    ax.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=24)
    ax.set_ylabel('$ \mathit{\\bar H_m} $', rotation=0, fontsize=24, labelpad=20)
    # Plotting the colorbar
    ax3 = fig3.add_axes([0.92, 0.11, 0.015, 0.77])
    norm = mpl.colors.Normalize(vmin=0, vmax=0.3)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cm.viridis, norm=norm, orientation='vertical')
    cb1.set_ticks([0., 0.1, 0.2, 0.3])
    ax3.text(2.3, 0.5, '$ \mathit{\\Delta\\epsilon^{th}_c} $', rotation=0, fontsize=24)
    ax3.yaxis.set_tick_params(labelsize=24, size=8)
    ax3.minorticks_on()
    fig3.savefig('fig_3.png', dpi=350)


def plot_fig4a(betas, H_ms, threshold_strain):
    fig_4a = plt.figure(4, figsize=(20, 10))
    ax = fig_4a.add_subplot(111)
    ax.contourf(betas, H_ms, threshold_strain, 20, cmap=cm.viridis, vmax=1., vmin=0., interpolation='none')
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.axvline(x=1, color='w', linewidth=3)
    ax.set_xlim(0.09, 2)
    ax.set_ylim(0, 20)
    ax.yaxis.set_tick_params(labelsize=20, size=5)
    ax.xaxis.set_tick_params(labelsize=20, size=5)
    ax.yaxis.set_ticks([0, 5, 10, 15, 20])
    ax.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=20)
    ax.set_ylabel('$ \mathit{\\bar H_m} $', rotation=0, fontsize=20, labelpad=20)
    # Plotting the colorbar
    ax2 = fig_4a.add_axes([0.92, 0.11, 0.015, 0.77])
    ax2.yaxis.set_tick_params(labelsize=20, size=5)
    ax2.minorticks_on()
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cm.viridis, norm=norm, orientation='vertical')
    cb1.set_label('$ \mathit{\\epsilon^{th}_c} $', fontsize=20, rotation=0)
    fig_4a.savefig('fig_4a.png', dpi=350)


def plot_fig4b(betas, H_ms, threshold_wavelength):
    fig_4b = plt.figure(5, figsize=(20, 10))
    ax = fig_4b.add_subplot(111)
    ax.contourf(betas, H_ms, threshold_wavelength, 20, cmap=cm.viridis, vmax=150., vmin=0.,
                interpolation='nearest')
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.axvline(x=1., color='w', linewidth=3)
    ax.set_xlim(0.09, 2)
    ax.set_ylim(0, 20)
    ax.yaxis.set_tick_params(labelsize=20, size=5)
    ax.xaxis.set_tick_params(labelsize=20, size=5)
    ax.yaxis.set_ticks([0, 5, 10, 15, 20])
    ax.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=20)
    ax.set_ylabel('$ \mathit{\\bar H_m} $', rotation=0, fontsize=20, labelpad=20)
    # Plotting the colorbar
    ax2 = fig_4b.add_axes([0.92, 0.11, 0.015, 0.77])
    ax2.yaxis.set_tick_params(labelsize=20, size=5)
    ax2.minorticks_on()
    norm = mpl.colors.Normalize(vmin=0, vmax=150)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cm.viridis, norm=norm, orientation='vertical')
    cb1.set_ticks([0, 25, 50, 75, 100, 125, 150])
    cb1.set_label('$ \mathit{\\bar L^{th}_c} $', fontsize=20, rotation=0)
    fig_4b.savefig('fig_4b.png', dpi=350)


def plot_fig5a(critical_strains, betas):
    forming_crit_strain_fig5b(critical_strains)
    fig_5a = plt.figure(6, figsize=(12, 8))
    ax = fig_5a.add_subplot(111)
    ax.set_prop_cycle('color', plt.cm.viridis(
        [0.2, 0.25, 0.255, 0.3, 0.35, 0.355, 0.4, 0.45, 0.455, 0.4555, 0.5, 0.6, 0.7, 0.85, 1]))
    ax.minorticks_on()
    ax.yaxis.set_tick_params(labelsize=15, size=5)
    ax.xaxis.set_tick_params(labelsize=15, size=5)
    ax.xaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=20)
    ax.set_ylabel('$ \mathit{\\bar L^{th}_c} $', rotation=0, fontsize=20, labelpad=15)
    ax.set_ylim(0., 140.)
    ax.set_xlim(0.1, 1.)
    for j in [0, 3, 5, 8, 18, 28, 38, 48, 58, 68, 79, 89, 99, 124, 149]:
        ax.plot(betas[:], wavelength[j, :], linestyle='-', linewidth=3)
    # Plotting the colorbar
    ax2 = fig_5a.add_axes([0.92, 0.11, 0.015, 0.77])
    ax2.text(0., 1.04, '$ \mathit{\\bar H_m} $', rotation=0, fontsize=20)
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.minorticks_on()
    norm = mpl.colors.Normalize(vmin=0.25, vmax=20)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=new_cmap, norm=norm, orientation='vertical')
    cb1.set_ticks([0.25, 5, 10, 15, 20])
    fig_5a.savefig('fig_5a.png', dpi=350)


def plot_fig5b(critical_strains):
    crit_strain_fig5b = forming_crit_strain_fig5b(critical_strains)
    fig_5b = plt.figure(7, figsize=(12, 8))
    ax = fig_5b.add_subplot(111)
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(labelsize=15, size=5)
    ax.xaxis.set_tick_params(labelsize=15, size=5)
    ax.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    ax.set_ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)
    ax.set_xlim([0., 100.])
    ax.set_ylim([0.1, 0.6])
    ax.set_prop_cycle('color', plt.cm.viridis(
        [0.2, 0.25, 0.3, 0.35, 0.355, 0.4, 0.45, 0.455, 0.5, 0.55, 0.555, 0.6, 0.7, 0.8, 1]))
    for i in range(15):
        [xs_new, strains_new] = eliminate_strain_eq_one(crit_strain_fig5b, i)
        ax.plot(xs_new, strains_new, linestyle='-', linewidth=3)
    # Plotting the colorbar
    ax2 = fig_5b.add_axes([0.92, 0.11, 0.015, 0.77])
    ax2.text(0., 1.04, '$ \mathit{\\bar H_m} $', rotation=0, fontsize=20)
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.minorticks_on()
    norm = mpl.colors.Normalize(vmin=0.25, vmax=20)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=new_cmap, norm=norm, orientation='vertical')
    cb1.set_ticks([0.25, 5, 10, 15, 20])

    fig_5b_inset = plt.figure(8, figsize=(16, 12))
    ax3 = fig_5b_inset.add_subplot(111)
    ax3.minorticks_on()
    ax3.yaxis.set_ticks_position('both')
    ax3.yaxis.set_tick_params(labelsize=35, size=10)
    ax3.xaxis.set_tick_params(labelsize=35, size=10)
    ax3.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=35)
    ax3.set_ylabel('$ \mathit{\\epsilon\_c} $', rotation=0, fontsize=35, labelpad=10)
    ax3.set_ylim([0.5, 0.6])
    ax3.set_xlim([0., 150.])
    ax3.set_prop_cycle('color', plt.cm.viridis(numpy.linspace(0.25, 20, 350)))
    ax3.set_prop_cycle('color', plt.cm.viridis([0.45, 0.7, 1]))
    for i in [7, 12, 14]:
        [xs_new, strains_new] = eliminate_strain_eq_one(crit_strain_fig5b, i)
        ax3.plot(xs_new, strains_new, linestyle='-', linewidth=4.5)
    ax3.axhline(y=0.5503479335646, color='black', linewidth=2, linestyle='-.')
    fig_5b.savefig('fig_5b.png', dpi=350)
    fig_5b_inset.savefig('fig_5b_inset.png', dpi=350)


def plot_fig6(critical_strains):
    crit_strain_fig6 = forming_crit_strain_fig6(critical_strains)
    fig_6 = plt.figure(9, figsize=(18, 7))
    ax = fig_6.add_subplot(121)
    ax.set_prop_cycle('color', plt.cm.viridis([0.1, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]))
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(labelsize=15, size=5)
    ax.xaxis.set_tick_params(labelsize=15, size=5)
    ax.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    ax.set_ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., 100.)
    for i in range(9):
        [xs_new, strains_new] = eliminate_strain_eq_one(crit_strain_fig6, i)
        ax.plot(xs_new, strains_new, linestyle='-', linewidth=3)

    ax2 = fig_6.add_subplot(122)
    ax2.set_prop_cycle('color', plt.cm.viridis([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1]))
    ax2.minorticks_on()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    ax2.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    ax2.set_ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)
    ax2.set_ylim(0., 1.)
    ax2.set_xlim(0., 100.)
    for i in range(9, 18, 1):
        [xs_new, strains_new] = eliminate_strain_eq_one(crit_strain_fig6, i)
        ax2.plot(xs_new, strains_new, linestyle='-', linewidth=3)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
    # Plotting the colorbar
    ax3 = fig_6.add_axes([0.92, 0.11, 0.015, 0.77])
    ax3.text(0.1, -0.10, '$ \mathit{\\beta} $', rotation=0, fontsize=20)
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    norm = mpl.colors.Normalize(vmin=0.1, vmax=2)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cm.viridis, norm=norm, orientation='vertical')
    cb1.set_ticks([0.1, 0.5, 1, 1.5, 2])
    fig_6.savefig('fig_6.png', dpi=350)


def plot_fig7a(H_ms, threshold_strain, anti_threshold_strain):
    fig7a = plt.figure(10, figsize=(18, 7))
    ax = fig7a.add_subplot(121)
    ax.set_prop_cycle('color', plt.cm.viridis([0.2, 0.2, 0.35, 0.35, 0.5, 0.5, 0.6, 0.6]))
    ax.set_ylim(0, 1)
    ax.set_xlim(0.25, 30)
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(labelsize=15, size=5)
    ax.xaxis.set_tick_params(labelsize=15, size=5)
    ax.set_xlabel('$ \mathit{\\bar H_m} $', rotation=0, fontsize=20)
    ax.text(-4.5, 0.5, '$ \mathit{\\epsilon^{th}_c} $', rotation=0, fontsize=20)

    ax2 = fig7a.add_subplot(122)
    ax2.set_prop_cycle('color', plt.cm.viridis([0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]))
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0.25, 30)
    ax2.minorticks_on()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    ax2.set_xlabel('$ \mathit{\\bar H_m} $', rotation=0, fontsize=20)
    ax2.text(-4.5, 0.5, '$ \mathit{\\epsilon^{th}_c} $', rotation=0, fontsize=20)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
    # Plotting the colorbar
    ax3 = fig7a.add_axes([0.92, 0.11, 0.015, 0.77])
    ax3.text(0.1, -0.10, '$ \mathit{\\beta} $', rotation=0, fontsize=20)
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    norm = mpl.colors.Normalize(vmin=0.1, vmax=2)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=new_cmap, norm=norm, orientation='vertical')
    cb1.set_ticks([0.1, 0.5, 1, 1.5, 2])

    for j in [9, 29, 49, 70]:
        ax.axhline(y=strain[199, j], color='gray', linewidth=3.5, linestyle=':')  # plot the IJSS results
        ax.plot(H_ms[:], threshold_strain[:, j], linestyle='-', linewidth=3)
        ax.plot(H_ms[:], anti_threshold_strain[:, j], linestyle='--', linewidth=3)

    for j in [128, 148, 169, 189]:
        ax2.axhline(y=strain[199, j], color='gray', linewidth=3.5, linestyle=':')  # plot the IJSS results
        ax2.plot(H_ms[:], threshold_strain[:, j], linestyle='-', linewidth=3)
        ax2.plot(H_ms[:], anti_threshold_strain[:, j], linestyle='--', linewidth=3)
    fig7a.savefig('fig_7a.png', dpi=350)


def plot_fig7b(threshold_wavelength, anti_threshold_wavelength):
    fig7b = plt.figure(11, figsize=(18, 7))
    ax = fig7b.add_subplot(121)
    ax.set_prop_cycle('color', plt.cm.viridis([0.2, 0.2, 0.35, 0.35, 0.5, 0.5, 0.6, 0.6]))
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(labelsize=15, size=5)
    ax.xaxis.set_tick_params(labelsize=15, size=5)
    ax.yaxis.set_ticks([0, 25, 50, 75, 100, 125, 150, 175])
    ax.set_xlabel('$ \mathit{\\bar H_m} $', rotation=0, fontsize=20)
    ax.text(-4.5, 175 / 2, '$ \mathit{\\bar L^{th}_c} $', rotation=0, fontsize=20)
    ax.set_ylim(0, 175)
    ax.set_xlim(0.25, 30)

    ax2 = fig7b.add_subplot(122)
    ax2.set_prop_cycle('color', plt.cm.viridis([0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]))
    ax2.minorticks_on()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    ax2.yaxis.set_ticks([0, 25, 50, 75, 100, 125, 150, 175])
    ax2.set_xlabel('$ \mathit{\\bar H_m} $', rotation=0, fontsize=20)
    ax2.text(-4.5, 175 / 2, '$ \mathit{\\bar L^{th}_c} $', rotation=0, fontsize=20)
    ax2.set_ylim(0, 175)
    ax2.set_xlim(0.25, 30)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
    # Plotting the colorbar
    ax3 = fig7b.add_axes([0.92, 0.11, 0.015, 0.77])
    norm = mpl.colors.Normalize(vmin=0.1, vmax=2)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=new_cmap, norm=norm, orientation='vertical')
    cb1.set_ticks([0.1, 0.5, 1, 1.5, 2])
    ax3.text(0.1, -0.10, '$ \mathit{\\beta} $', rotation=0, fontsize=20)
    ax3.yaxis.set_tick_params(labelsize=15, size=5)

    for j in [9, 29, 49, 70]:
        ax.axhline(y=wavelength[199, j], color='gray', linewidth=3.5, linestyle=':')  # plot the IJSS results
        ax.plot(H_ms[:], threshold_wavelength[:, j], linestyle='-', linewidth=3)
        ax.plot(H_ms[:], anti_threshold_wavelength[:, j], linestyle='--', linewidth=3)

    for j in [128, 148, 169, 189]:
        ax2.axhline(y=wavelength[199, j], color='gray', linewidth=3.5, linestyle=':')  # plot the IJSS results
        ax2.plot(H_ms[:], threshold_wavelength[:, j], linestyle='-', linewidth=3)
        ax2.plot(H_ms[:], anti_threshold_wavelength[:, j], linestyle='--', linewidth=3)
    fig7b.savefig('fig_7b.png', dpi=350)


def eliminate_strain_eq_one(crit_strain, column):
# eliminate the strain values that is equal to 1 and their corresponding normalized wavelengths.
    n_wavelengths = 1500
    counter = []
    wavelengths = numpy.logspace(-1., 2.5, num=n_wavelengths)
    wavelengths = wavelengths[::-1]
    wavelengths_new = numpy.zeros(len(wavelengths))
    strains_save = numpy.zeros(len(wavelengths))
    strains_save = crit_strain[:, column]
    for j in range(len(wavelengths)):
        if strains_save[j] == 1:
            counter.append(j)
    strains_save = numpy.delete(strains_save, counter)
    wavelengths_new = numpy.delete(wavelengths, counter)
    return wavelengths_new, strains_save


def forming_crit_strain_fig5b(critical_strains):
# separate the critical strains corresponding to beta=0.2 and Hm=[0.25,0.5,0.75,1,2,3,4,5,6,7,8,9,10.2,15,20]
    crit_strain_fig5b = numpy.zeros((1500, 15))
    column_counter = 0
    for i in [0, 3, 5, 8, 18, 28, 38, 48, 58, 68, 79, 89, 100, 124, 149]:
        crit_strain_fig5b[:, column_counter] = critical_strains[:, 19 + 200 * i]
        column_counter = column_counter + 1
    return crit_strain_fig5b


def forming_crit_strain_fig6(critical_strains):
# separate the critical strains corresponding to Hm=0.5 and beta=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    crit_strain_fig6 = numpy.zeros((1500, 18))
    column_counter = 0
    for i in [9, 19, 29, 39, 49, 60, 70, 81, 91, 109, 118, 128, 138, 148, 159, 169, 179, 189]:
        crit_strain_fig6[:, column_counter] = critical_strains[:, i + 200 * 3]
        column_counter = column_counter + 1
    return crit_strain_fig6


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(numpy.linspace(minval, maxval, n)))
    return new_cmap