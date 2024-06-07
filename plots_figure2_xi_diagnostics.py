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
#from def_analysis import *
from def_diagnostics import *
from def_plotting import *

from def_utilities import *


#%% 
lr = 1.1
lm = 5000
K = 1000
par_vars = dict({'K': K, 'lm': lm})


sol_dir = 'solutions_binary'

x_param = ['B', 'lambda_mu', 'K', 'xi_core', 'theta']



QLIST = [ 
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'lm': 10000, 'B': 3000}),
dict({'nz': 512, 'theta': 0.05, 'lr': 1.1,'K': 100, 'lm': 10000, 'B': 3000})
]



QLIST = [ 
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'lm': 10000, 'B': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'lm': 10000, 'B': 500}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'lm': 10000, 'B': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 100, 'lm': 10000, 'B': 3000}),
dict({'nz': 512, 'theta': 0.05, 'lr': 1.1,'K': 100, 'lm': 10000, 'B': 3000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'lm': 10000, 'B': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'lm': 10000, 'B': 500}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'lm': 10000, 'B': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'lm': 10000, 'B': 3000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'lm': 10000, 'B': 100}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'lm': 10000, 'B': 500}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'lm': 10000, 'B': 1000}),
dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 10000, 'lm': 10000, 'B': 3000})
]



Xlist = []
Ylist = []

xnames = ['B', 'lambda_mu', 'K', 'xi_core', 'theta']

for qcase in QLIST:
    Xcase, Ycase, flist = get_diagnostics(qcase, xnames, dirname=sol_dir, Xi_true = True)
    Xlist.append(Xcase)
    Ylist.append(Ycase)



#%%
redPallete = create_color_list('Reds', 4)
greenPallete = create_color_list('Greens', 4)
bluePallete = create_color_list('Blues', 4)
purplePallete = create_color_list('Purples', 1)


COLOR_LIST = bluePallete+purplePallete+redPallete+greenPallete
#COLOR_LIST = 


#%%

x_axis_param='xi_core'
f1,a1 = plot_compare_diagnostic(Xlist,Ylist,'xi_core', 'fS_top',  ylabel_str=r'$\phi^s(z=1)$',xlabel_str=r'$\xi_{core}$', colorlist=COLOR_LIST)
f2,a2 = plot_compare_diagnostic(Xlist,Ylist,'xi_core', 'R_rho', ylabel_str=r'$R_\rho$', xlabel_str=r'$\xi_{core}$', colorlist=COLOR_LIST, YSCALE='linear')
f3,a3 = plot_compare_diagnostic(Xlist,Ylist,x_axis_param, 'J_Xi_scaled', ylabel_str=r'$<j_\xi>$', xlabel_str=r'$\xi_{core}$',colorlist=COLOR_LIST, YSCALE='log')
f4,a4= plot_compare_diagnostic(Xlist,Ylist,'xi_core', 'uS_avg',  ylabel_str=r'$|<u_z^s>|$', xlabel_str=r'$\xi_{core}$', colorlist=COLOR_LIST, YSCALE='log')
f5,a5 = plot_compare_diagnostic(Xlist,Ylist,'xi_core', 'uL_top',  ylabel_str=r'$u_z^l(z=1)$', xlabel_str=r'$\xi_{core}$', colorlist=COLOR_LIST)
f6,a6 = plot_compare_diagnostic(Xlist,Ylist,'xi_core', 'Qc_ratio',  ylabel_str=r'$R_Q$', xlabel_str=r'$\xi_{core}$',colorlist=COLOR_LIST, YSCALE='linear')
f7,a7 = plot_compare_diagnostic(Xlist,Ylist,'xi_core', 'gradRho_max',  ylabel_str=r'$max(d\rho/dz)$', xlabel_str=r'$\xi_{core}$',colorlist=COLOR_LIST, YSCALE='linear')


a7.plot([0, 0.1], [0, 0], linestyle='dashed', color='black')

a3.set_ylim([1e-7, 1e-2])


label_vars = ['theta', 'K', 'B']
label_strings = ['\\theta','K', 'B']

fl, al = make_legend_standalone(Xlist, legend_label_vars=label_vars, 
                                math_strings=label_strings,
                                FIGSIZE=(6,1), n_col = 3, color_list=COLOR_LIST,legend_type='markers' )



def scaling_curve(x,p,x0,y0):
    y =  y0*(x/x0)**p
    return y

xf = np.array([0.002, 0.02])
xf2 = np.array([0.04, 0.1])

yf = scaling_curve(xf, -1, xf[0], 1.5e-3)
yf2 = scaling_curve(xf2, -1, xf2[0], 0.35)

a1.loglog(xf, yf, linestyle='dashed', color='black')
a1.loglog(xf2, yf2, linestyle='dashed', color='black')


yf = scaling_curve(xf, -1, xf[0], 1.5e-6)
yf2 = scaling_curve(xf2, -1, xf2[0], 0.08e-4)
a5.loglog(xf, yf, linestyle='dashed', color='black')
a5.loglog(xf2, yf2, linestyle='dashed', color='black')



f1.savefig('figure2_xi_variation_fs', dpi=300, bbox_inches='tight')
f2.savefig('figure2_xi_variation_R_rho', dpi=300, bbox_inches='tight')
f3.savefig('figure2_xi_variation_jxi_log', dpi=300, bbox_inches='tight')
f4.savefig('figure2_xi_variation_us', dpi=300, bbox_inches='tight')
f5.savefig('figure2_xi_variation_ul', dpi=300, bbox_inches='tight')
f6.savefig('figure2_xi_variation_R_Q', dpi=300, bbox_inches='tight')
f7.savefig('figure2_xi_variation_gradRho', dpi=300, bbox_inches='tight')


fl.savefig('figure2_xi_variation_legend', dpi=300, bbox_inches='tight')
