import numpy as np
#from mpi4py import MPI

import matplotlib as mpl
import matplotlib.pyplot as plt
from dedalus import public as de
#import h5py
#import time
import pickle
#import logging
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import os

# logger = logging.getLogger(__name__)
#
# comm = MPI.COMM_WORLD
#
plt.style.use('methods_paper.mplstyle')




def gen_markers_list():
    marker_list = ['o', 'v', 's', 'h', 'P', '*', 'X', 'D', 'p','^', '<', '>', 'd' ]
    return marker_list

def create_color_list(CMAP_NAME, NUM_COLORS, offset=0.0, reverse=False):

    color_interval = (1.-offset)/(NUM_COLORS)
    cm = plt.get_cmap(CMAP_NAME)
    color_list = [cm(i) for i in np.linspace(offset+0.5*color_interval, 1.-0.5*color_interval, num=NUM_COLORS)]
    if reverse is True:
        color_list.reverse()

    return color_list

def create_cmap_discrete(CMAP_NAME, NUM_COLORS, offset=0.0):
    cmaplist = create_color_list(CMAP_NAME, NUM_COLORS, offset)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, NUM_COLORS)
    return cmap

def make_legend_label(X, legend_label_vars, math_strings=None):
    # X is a dictionary of parameter values
    # legend_label_vars is a list of parameters to be included in the label

    print(legend_label_vars)
    f = mticker.ScalarFormatter(useMathText=True)
    f.set_powerlimits((-3,10))

    mathmode = True

    if math_strings is None:
        math_strings = legend_label_vars
        mathmode = False


    plot_label = ''
    if len(legend_label_vars) == len(math_strings):

        for count in range(len(legend_label_vars)):

            par_key = legend_label_vars[count]
            math_string = math_strings[count]

            if isinstance(X[par_key], np.ndarray):
                par_val = X[par_key][0]
            else:
                par_val = X[par_key]

            if par_val == 0:
                par_val_formatted = "{0}"
            else:
                par_val_formatted = "{}".format(f.format_data(par_val))

            if count == 0:
                plc = '{}={}'.format( math_string, par_val_formatted)
            else:
                plc = ', {}={}'.format( math_string, par_val_formatted)

            plot_label = plot_label+plc

        if mathmode == True:
            plot_label = r'$'+plot_label+'$'

    else:
        print('Error: mismatching number of legend label and strings')
        plot_label = ''

    return plot_label

def make_legend_standalone(cases_list, legend_label_vars, math_strings, FIGSIZE=(8,1), legend_type='patches',
                            n_col = None, color_list=None, marker_list=None):

    legend_entries_list = []
    NCASES = len(cases_list)

    if color_list is None:
        color_list = create_color_list('viridis', NCASES)

    if marker_list is None:
        marker_list = gen_markers_list()

    MARKERSIZE = 12
    MARKEREDGECOLOR ='black'


    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ii in range(NCASES):
        case_ii = cases_list[ii]
        color_ii = color_list[ii]
        marker_ii = marker_list[ii % len(marker_list)]
        case_label = make_legend_label(case_ii, legend_label_vars, math_strings)
        if legend_type=='lines':
            case_plot_entity = mlines.Line2D([],[],color=color_ii, label=case_label)
        elif legend_type=='markers':
            case_plot_entity = mlines.Line2D([],[],color=color_ii, marker=marker_ii,markersize=MARKERSIZE,markeredgecolor=MARKEREDGECOLOR, label=case_label)
        elif legend_type=='patches':
            case_plot_entity = mpatches.Patch(color=color_ii, label=case_label)

        legend_entries_list.append(case_plot_entity)

    ax.axis(False)
    if n_col is None:
        n_col=NCASES

    fig.legend(handles=legend_entries_list, ncol=n_col, loc='center', fontsize=16)

    return fig, ax


def plotSpecified(solution_data, VAR_LIST, fig=None, ax_tupple=None, math_titles=None, color=None, fontsize=18):
    z = solution_data['z']

    n_var = len(VAR_LIST)

    max_col = 4

    n_row = int(np.ceil(n_var / max_col))
    n_col = int(np.min([n_var, max_col]))
    n_fig = n_var
    fig_height = 3.5
    fig_width = 2.5

    if fig is None:
        fig, axr = plt.subplots(n_row,n_col,figsize=(n_col*fig_width, n_row*fig_height),sharey=True)
        fig.subplots_adjust(hspace=0)
    else:
        axr = ax_tupple

    FONTSIZE=fontsize

    mathmode = True
    if math_titles is None:
        math_titles = VAR_LIST

    for count in range(n_var):
        key = VAR_LIST[count]
        if isinstance(key, tuple):
            VAR = 1.
            for ele in key:
                VAR = VAR*solution_data[ele]
        elif key=='rho':
            lr = solution_data['parameters']['lambda_rho']
            rhoS = solution_data['rhoS']
            rhoL = solution_data['rhoL']
            fS = solution_data['fS']
            VAR = lr*fS*rhoS + (1-fS)*rhoL
        elif key=='T_pert':
            theta = solution_data['parameters']['theta']
            T = solution_data['T']
            T_diffusion = 1 + theta*(1-z)
            VAR = T-T_diffusion
        elif key=='Tz_pert':
            theta = solution_data['parameters']['theta']
            Tz = solution_data['Tz']
            VAR = Tz+theta
        elif key=='solid_flux':
            lr = solution_data['parameters']['lambda_rho']
            rhoS = solution_data['rhoS']
            fS = solution_data['fS']
            uS = solution_data['uS']
            VAR = lr*rhoS*fS*uS
        elif key=='liquid_flux':

            rhoL = solution_data['rhoL']
            fL = 1.0-solution_data['fS']
            uL = solution_data['uL']
            VAR = rhoL*fL*uL

        elif key=='J_Xi':
            T = solution_data['T']
            fL = 1- solution_data['fS']
            Xi = solution_data['Xi']
            Xi_z = solution_data['Xi_z']
            Pz = solution_data['Pz']
            tau = solution_data['parameters']['tau']
            B = solution_data['parameters']['B']
            Pr = solution_data['parameters']['Pr']
            D = solution_data['parameters']['D']
            chi = solution_data['parameters']['chi']
            alpha_xi = solution_data['parameters']['alpha_xi']

            c4 = tau*(B*Pr)**-1
            c5 = alpha_xi*chi*D
            Xi_diffusion = c4*(Xi_z)
            Barodiffusion = c4*c5*(Xi*Pz/T)
            VAR = rhoL*fL*(Xi_diffusion + Barodiffusion)

        else:
            VAR = solution_data[key]


        title_str = math_titles[count]
        if mathmode:
            title_str =  r'$' + title_str +'$'

        if (n_row == 1) and (n_col == 1):
            axr.plot( VAR, z, linestyle='solid', color=color)
            axr.set_title(title_str, fontsize=FONTSIZE)
            axr.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
        elif n_row == 1:
            axr[count].plot( VAR, z, linestyle='solid', color=color)
            axr[count].set_title(title_str, fontsize=FONTSIZE)
            axr[count].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
        else:
            r, c = np.divmod(count, n_col)
            axr[r, c].plot( VAR, z, linestyle='solid', color=color)
            axr[r, c].set_title(title_str, fontsize=FONTSIZE)
            axr[r, c].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

    if (n_row == 1) and (n_col == 1):
        axr.set_ylabel(r'$z$', fontsize=FONTSIZE)
    elif n_row == 1:
        axr[0].set_ylabel(r'$z$', fontsize=FONTSIZE)
    else:
        for r in range(n_row):
            axr[r,0].set_ylabel(r'$z$', fontsize=FONTSIZE)

    fig.tight_layout()

    return fig, axr



def plotQuick(solution_data, fig=None, ax_tupple=None, saving=False, fname=None, dname=None):
    z = solution_data['z']
    T = solution_data['T']
    Tz = solution_data['Tz']
    rhoS = solution_data['rhoS']
    rhoL = solution_data['rhoL']

    uS = solution_data['uS']
    uS_z = solution_data['uS_z']
    uL = solution_data['uL']
    uL_z = solution_data['uL_z']
    fS = solution_data['fS']
    fS_z = solution_data['fS_z']
    Pz = solution_data['Pz']
    mDotS = solution_data['mDotS']
    nz = solution_data['parameters']['nz']


    if "Xi" in solution_data:
        Xi = solution_data['Xi']
    else:
        Xi = 0*z

    z_basis = de.Chebyshev('z', nz, interval=(0,1))
    domain = de.Domain([z_basis], grid_dtype=np.float64)
    Xi_field = domain.new_field(name='Xi')
    Xi_field['g'] = Xi
    Xi_z_field = Xi_field.differentiate(z=1)
    Xi_z = Xi_z_field['g']

    Pz_field = domain.new_field(name='Pz')
    Pz_field['g'] = Pz
    Pz_z_field = Pz_field.differentiate(z=1)
    Pz_z = Pz_z_field['g']

    Tz_field = domain.new_field(name='Tz')
    Tz_field['g'] = Tz
    Tz_z_field = Tz_field.differentiate(z=1)
    Tz_z = Tz_z_field['g']

    rhoS_field = domain.new_field(name='rhoS')
    rhoS_field['g'] = rhoS
    rhoS_z_field = rhoS_field.differentiate(z=1)
    rhoS_z = rhoS_z_field['g']

    rhoL_field = domain.new_field(name='rhoL')
    rhoL_field['g'] = rhoL
    rhoL_z_field = rhoL_field.differentiate(z=1)
    rhoL_z = rhoL_z_field['g']


    lr = solution_data['parameters']['lambda_rho']

    fL = 1 - fS
    fL_z = - fS_z
    rho = lr*fS*rhoS + (1-fS)*rhoL
    rho_z = lr*(fS_z*rhoS + fS*rhoS_z) + fL_z*rhoL + fL*rhoL_z
    w = uS-uL
    jz = fS*fL*rhoS*rhoL*w/(rho)





    if fig is None:
        fig, axr = plt.subplots(3, 4,figsize=(15,15))
        ax_tupple = axr
    else:
        axr = ax_tupple


    FONTSIZE = 18
    axr[0, 1].plot( rho_z, z, linestyle='solid')
    axr[0, 1].set_title(r"$d\rho/dz$", fontsize=FONTSIZE)
    axr[0, 0].plot( jz, z, linestyle='solid')
    axr[0, 0].set_title(r"$j_z$", fontsize=FONTSIZE)
    axr[0, 2].plot( uS, z, linestyle='solid')
    axr[0, 2].set_title(r"$u^s$", fontsize=FONTSIZE)
    axr[0, 3].plot( uL, z, linestyle='solid')
    axr[0, 3].set_title(r"$u^l$", fontsize=FONTSIZE)
    axr[1, 0].plot( Tz, z, linestyle='solid')
    axr[1, 0].set_title(r"$dT/dz$", fontsize=FONTSIZE)
    axr[1, 1].plot( fS_z, z, linestyle='solid')
    axr[1, 1].set_title(r"$d\phi^s/dz$", fontsize=FONTSIZE)
    axr[1, 2].plot( fS, z, linestyle='solid')
    axr[1, 2].set_title(r"$\phi^s$", fontsize=FONTSIZE)
    axr[1, 3].plot( rho, z, linestyle='solid')
    axr[1, 3].set_title(r"$\rho$", fontsize=FONTSIZE)
    axr[2, 0].plot( mDotS, z, linestyle='solid')
    axr[2, 0].set_title(r"$\Gamma^s$", fontsize=FONTSIZE)
    axr[2, 1].plot( Xi_z, z, linestyle='solid')
    axr[2, 1].set_title(r"$d\xi^L/dz$", fontsize=FONTSIZE)
    axr[2, 2].plot( Xi, z, linestyle='solid')
    axr[2, 2].set_title(r"$\xi^L$", fontsize=FONTSIZE)
    axr[2, 3].plot( T, z, linestyle='solid')
    axr[2, 3].set_title(r"$T$", fontsize=FONTSIZE)


    fig.tight_layout()

    if saving is True:
        if fname is None:
            fname = genFilename(solution_data['parameters'])

        if dname is None:
            save_path = fname
        else:
            if not os.path.exists(dname):
                os.mkdir(dname)
            save_path = os.path.join(dname, fname)

        fig.savefig(save_path+'.png', dpi=300)

    return fig, ax_tupple




def showAll(solution_data, fig=None, ax=None):

    z = solution_data['z']

    keys = list(solution_data.keys())
    keys.remove('z')
    keys.remove('parameters')

    Nvar = len(keys)


    ncol = int(np.ceil(np.sqrt(Nvar)))
    nrow = int(np.ceil(Nvar/ncol))

    cols = np.tile(np.arange(ncol), (nrow, 1))
    rows = np.tile(np.arange(nrow), (ncol, 1)).T

    col_idx = np.ravel(cols)
    row_idx = np.ravel(rows)

    w = 4
    h = 4


    if ax is None:
        fig, ax = plt.subplots(nrow, ncol, figsize=(w*ncol,h*nrow))


    for ii in range(Nvar):
        key = keys[ii]

        ax[row_idx[ii],col_idx[ii]].plot(solution_data[key], z)
        ax[row_idx[ii],col_idx[ii]].set_title(key, fontsize=12)

    fig.tight_layout()


    return fig, ax
