#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 08:43:14 2022

@author: fryderyk
"""
import numpy as np
import pickle
import glob
import matplotlib
from matplotlib import pyplot as plt

from def_diagnostics import *
from def_plotting import *

from def_utilities import *


#%% 



sol_dir = 'solutions_binary_LMOB'

x_param = ['B', 'M', 'K', 'xi_core', 'theta']




QLIST = [ 
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 10}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 10}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 100}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 1000}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 1000}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 10000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10000}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 10000}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 100000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100000}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 100000}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 1e+06}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 1e+06}),
#dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 1e+06}),
]


# QLIST = [ 
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 10}),
# dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 10}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 100}),
# dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 100}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 1000}),
# dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 1000}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 1000}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 10000}),
# dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10000}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 10000}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 100000}),
# dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100000}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 100000}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'xc': 0.01, 'M': 1e+06}),
# dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 1e+06}),
# #dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'xc': 0.01, 'M': 1e+06}),
# ]


QLIST = [ 
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 10}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 10}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 10000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 10000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 100000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 100000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100000})
]



QLIST = [ 
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 10}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 10000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'M': 100000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 10}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 10000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.02, 'M': 100000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 10000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.05, 'M': 100000})
]


Xlist = []
Ylist = []

xnames = ['B', 'M', 'K', 'xi_core', 'theta']

for qcase in QLIST:
    Xcase, Ycase, flist = get_diagnostics(qcase, xnames, dirname=sol_dir, Xi_true=True)
    Xlist.append(Xcase)
    Ylist.append(Ycase)



#%%
nc = 5

redPallete = create_color_list('Reds', nc)
greenPallete = create_color_list('Greens', nc)
bluePallete = create_color_list('Blues', nc)
purplePallete = create_color_list('Purples', nc)
orangePallete = create_color_list('Oranges', nc)
greyPallete = create_color_list('Greys', nc)

COLOR_LIST = bluePallete+redPallete+purplePallete+greyPallete

#COLOR_LIST = create_color_list('viridis', len(QLIST))


#%%

x_axis_param='B'

x_axis_str=r'$B$'

f1,a1 = plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'fS_top',  ylabel_str=r'$\phi^s(z=1)$',xlabel_str=x_axis_str, colorlist=COLOR_LIST)
f2,a2 = plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'rLfLuL_avg', ylabel_str=r'$|<\lambda_\rho \rho^s \phi^s u_z^s>|$', xlabel_str=x_axis_str, colorlist=COLOR_LIST)
f3,a3 = plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'J_Xi_avg', ylabel_str=r'$<\rho^l \phi^l \xi^l u_z^l>$', xlabel_str=x_axis_str,colorlist=COLOR_LIST, YSCALE='log')
f4,a4= plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'uS_avg',  ylabel_str=r'$|<u_z^s>|$', xlabel_str=x_axis_str, colorlist=COLOR_LIST, YSCALE='log')
f5,a5 = plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'uL_avg',  ylabel_str=r'$<u_z^l>$', xlabel_str=x_axis_str, colorlist=COLOR_LIST)
#f6,a6 = plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'Qc_ratio',  ylabel_str=r'$R_Q$', xlabel_str=x_axis_str,colorlist=COLOR_LIST, YSCALE='linear')
f7,a7 = plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'R_rho',  ylabel_str=r'$R_\rho$', xlabel_str=x_axis_str,colorlist=COLOR_LIST, YSCALE='linear')


#a7.plot([0, 0.1], [0, 0], linestyle='dashed', color='black')

label_vars = [ 'xi_core', 'M']
label_strings = [ '\\xi_{core}', '\\lambda_\\eta/B']

fl, al = make_legend_standalone(Xlist, legend_label_vars=label_vars, 
                                math_strings=label_strings,
                                FIGSIZE=(8,1), n_col = 3, color_list=COLOR_LIST,legend_type='markers' )



def scaling_curve(x,p,x0,y0):
    y =  y0*(x/x0)**p
    return y

xf = np.array([1e3, 8e3])
xf2 = np.array([1e3, 9e3])

yf = scaling_curve(xf, -1, xf[0], 5.5e-3)
yf2 = scaling_curve(xf2, -1, xf2[0], 0.1)
yf3 = scaling_curve(xf2, -1, xf2[0], 0.65)

#a1.loglog(xf, yf, linestyle='dashed', color='black')
#a1.loglog(xf2, yf2, linestyle='dashed', color='black')
#a1.loglog(xf2, yf3, linestyle='dashed', color='black')


yf = scaling_curve(xf, -1, xf[0], 2e-6)
yf2 = scaling_curve(xf2, -1, xf2[0], 2e-8)
yf3 = scaling_curve(xf2, -1, xf2[0], 2e-6)
#a5.loglog(xf, yf, linestyle='dashed', color='black')
#a3.loglog(xf2, yf2, linestyle='dashed', color='black')
#a2.loglog(xf2, yf3, linestyle='dashed', color='black')



#f1.savefig('figure4_B_variation_fs_top', dpi=300, bbox_inches='tight')
#f2.savefig('figure4_B_variation_solid_flux', dpi=300, bbox_inches='tight')
#f3.savefig('figure4_xi_variation_jxi', dpi=300, bbox_inches='tight')
#f4.savefig('figure4_xi_variation_us', dpi=300, bbox_inches='tight')
#f5.savefig('figure4_xi_variation_ul', dpi=300, bbox_inches='tight')
# f6.savefig('figure4_xi_variation_R_Q', dpi=300, bbox_inches='tight')
#f7.savefig('figure4_xi_variation_R_rho', dpi=300, bbox_inches='tight')


#fl.savefig('figure4_legend', dpi=300, bbox_inches='tight')
