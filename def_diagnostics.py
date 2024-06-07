import numpy as np
import pickle
import glob
import os
from matplotlib import pyplot as plt
from def_analysis import diag_fluxes

import matplotlib.lines as mlines

from def_plotting import create_color_list
from def_plotting import make_legend_label
from def_plotting import gen_markers_list



def get_file_list( par_var_dict, dirname):

    key_list=list(par_var_dict.keys())
    flist_master = []

    list_of_lists = []

    flist_master = glob.glob(dirname+'/*')

    for key in key_list:
        parval = "{:g}".format(par_var_dict[key])
        parval_p = parval.replace('.', 'p') # change periods '.' into 'p'
        search_string = "*_{}_{}_*".format(key, parval_p)

        print(search_string)
        flist_key =  glob.glob(dirname+'/'+search_string)

        intersection_list = list(set(flist_master).intersection(flist_key))
        flist_master = intersection_list

    return flist_master



def get_diagnostics(par_var_dict, x_names, dirname, Xi_true=False):
    file_list = get_file_list( par_var_dict, dirname)
    X, Y = diagnostics(file_list, x_names, Xi_true)
    return X, Y, file_list

def diagnostics(file_list, x_names, Xi_true=False):

    N = len(file_list)
    x_axis = dict()
    y_axis = dict()

    for x_name in x_names:
        x_axis[x_name] = np.zeros(N)
        x_axis[x_name][:] = np.nan



    if Xi_true == True:
        diagnostics_list = ['Qc_left','Qc_right','Qc_ratio','Ql','Qp','QvS','QvL','Qf','uS_top', 'jz_avg',  'jz_scaled', 'mDotS_avg',
                            'uL_top', 'fS_top','fL_top', 'uS_avg', 'uL_avg', 'uS_max', 'z_uS_max', 'uL_max', 'z_uL_max', 'uS_scaled', 'uL_scaled',
                            'ReS_max', 'ReS_avg', 'ReL_max', 'ReL_avg', 'gradRho_max', 'z_gradRho_max',
                            'rSfSuS_avg',  'rLfLuL_avg','rho_top', 'rho_bot', 'R_rho',
                            'fS_avg', 'fS_max', 'z_fS_max', 'fS_min', 'z_fS_min','inertiaL', 'PzL', 'buoyL', 'frictionL', 'phaseCh', 'viscL',
                                            'inertiaS', 'PzS', 'buoyS', 'viscS',
                                            'H_xi_avg', 'Xi_bot', 'Xi_avg','Xi_z_avg','Xi_z_bot','Xi_z_top', 'J_Xi_avg', 'J_Xi_scaled']
    else:
        diagnostics_list = ['Qc_left','Qc_right','Qc_ratio','Ql','Qp','QvS','QvL','Qf','uS_top', 'jz_avg',  'jz_scaled', 'mDotS_avg',
                            'uL_top', 'fS_top','fL_top', 'uS_avg', 'uL_avg', 'uS_max', 'z_uS_max', 'uL_max', 'z_uL_max', 'uS_scaled', 'uL_scaled',
                            'ReS_max', 'ReS_avg', 'ReL_max', 'ReL_avg', 'gradRho_max', 'z_gradRho_max','rho_top', 'rho_bot', 'R_rho',
                             'rSfSuS_avg',  'rLfLuL_avg',
                            'fS_avg', 'fS_max', 'z_fS_max', 'fS_min', 'z_fS_min','inertiaL', 'PzL', 'buoyL', 'frictionL', 'phaseCh', 'viscL',
                                            'inertiaS', 'PzS', 'buoyS', 'viscS']


    for flux in diagnostics_list:
        y_axis[flux]=np.zeros(N)
        y_axis[flux][:] = np.nan

    for count, filepath in enumerate(file_list):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                f_data = pickle.load(f)
                fluxes = diag_fluxes(f_data)
                for x_name in x_names:
                    x_axis[x_name][count] = f_data['parameters'][x_name]
                for flux in diagnostics_list:
                    y_axis[flux][count] = fluxes[flux]

        else:
            print(filepath+ ' does not exist')

    return x_axis, y_axis








def plot_compare_diagnostic(Xlist, Ylist, xvar, yvar, x_offset=0, fig=None, axr=None,
                            legend_label_vars=None, xlabel_str=None, ylabel_str=None, colorlist=None,
                            XSCALE='log', YSCALE='log'):
    N = len(Ylist)

    if colorlist is None:
        color_list = create_color_list('viridis', N)
    else:
        color_list = colorlist

    marker_list = gen_markers_list()

    for n in range(N):
        X = Xlist[n]
        Y = Ylist[n]
        marker_select = marker_list[n % len(marker_list)]
        fig, axr = plot_specific_diagnostic(X, Y, xvar, yvar, x_offset, fig, axr, legend_label_vars, xlabel_str=xlabel_str, ylabel_str=ylabel_str,
                                            color_in=color_list[n], marker_in=marker_select, XSCALE=XSCALE, YSCALE=YSCALE)
    return fig, axr


def plot_specific_diagnostic(X, Y, xvar, yvar, x_offset=0, fig=None, axr=None,
                                legend_label_vars=None, xlabel_str=None, ylabel_str=None, color_in=None, marker_in='o', XSCALE='log', YSCALE='log'):
    if fig is None:
        fig, axr = plt.subplots(1,1, figsize=(5,5))

    if legend_label_vars is None:
        plot_label = None
    else:
        plot_label = make_legend_label(X, legend_label_vars)

    if xlabel_str is None:
        xlabel_str = r'${}$'.format(xvar)

    if ylabel_str is None:
        ylabel_str = yvar

    if yvar == 'fL_avg':
        ydata = 1.0 - np.abs(Y['fS_avg'])
    elif yvar == 'null':
        ydata = np.nan*np.ones_like(X[xvar])
    elif yvar == 'uS_avg' or yvar == 'uS_scaled' or yvar == 'jz_avg' or yvar == 'jz_scaled':
        ydata = np.abs(Y[yvar])
    else:
        ydata = Y[yvar]

    idx_sort = np.argsort(X[xvar])

    Xs = X[xvar][idx_sort]

    MARKERSIZE=8
    MARKEVERY=0.01
    MARKEREDGECOLOR='black'
    ALPHA = 1

    if color_in is None:
        axr.plot(X[xvar][idx_sort]-x_offset, ydata[idx_sort],  marker=marker_in, markersize=MARKERSIZE,markeredgecolor=MARKEREDGECOLOR, markevery=MARKEVERY, alpha=ALPHA, label=plot_label)
    else:
        axr.plot(X[xvar][idx_sort]-x_offset, ydata[idx_sort],  marker=marker_in, markersize=MARKERSIZE,markeredgecolor=MARKEREDGECOLOR, markevery=MARKEVERY, color=color_in, alpha=ALPHA, label=plot_label)

    axr.set_title(ylabel_str, fontsize=16)
    axr.set_xlabel(xlabel_str, fontsize=16)

    axr.set_xscale(XSCALE)
    axr.set_yscale(YSCALE)

    if plot_label is not None:
        if yvar == 'null':
            axr.legend(loc='center',fontsize=16)
            axr.set_xticks([],[])
            axr.set_yticks([],[])
            axr.set_xlabel('')
        else:
            axr.legend()

    return fig, axr
