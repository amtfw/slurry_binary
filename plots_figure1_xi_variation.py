import numpy as np

import matplotlib.pyplot as plt
from dedalus import public as de



from def_plotting import *
from def_diagnostics import *
from def_utilities import *
from def_termbyterm import *


read_dir = 'solutions_binary/'

plot_dir='plots_analysis/'



LABEL_VARS = ['xi_core']
math_strings = ['\\xi_{core}']


LIST_CASES = [dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0, 'lm': 10000, 'B': 3000}),
               dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.0001, 'lm': 10000, 'B': 3000}),
               dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.001, 'lm': 10000, 'B': 3000}),
               dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.005, 'lm': 10000, 'B': 3000}),
               dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.01, 'lm': 10000, 'B': 3000})]




FILE_LIST = []

for CASE in LIST_CASES:
    flist = get_file_list(CASE, dirname=read_dir)
    if not flist:
        print(CASE)
        print('No such case found')
    else:
        FILE_LIST.extend(flist)


#%%
        
        
NUM_COLORS = len(FILE_LIST)

color_list = create_color_list('viridis_r', NUM_COLORS, offset=0.0)

color_count = 0
CASE_PARAMS_LIST = []
h_fig=None
h_ax=None

h_fig_u =None
h_ax_u = None


VAR_LIST = ['Xi','fS', 'uS', 'uL', 'mDotS', 'Tz_pert', 'rho', 'J_Xi']
TITLE_LIST = ['\\xi^l','\phi^s',  'u_z^s', 'u_z^l', '\\Gamma_\\rho^s',  'd \hat{T}/dz', '\\bar\\rho', '\\rho^l \phi^l \\xi^l u_z^l']




for sd_filename in FILE_LIST: 
    sd_data = read_solution_from_file(sd_filename)
    CASE_PARAMS_LIST.append(sd_data['parameters'])
    color_select = color_list[color_count]
    h_fig, h_ax = plotSpecified(sd_data, VAR_LIST, math_titles=TITLE_LIST, fig=h_fig, ax_tupple=h_ax, color=color_select)
    #h_fig, h_ax = showAll(sd_data, fig=h_fig, ax=h_ax)
    #h_fig, h_ax = plot_balance(sd_data, 'concentration2', fig=h_fig, ax=h_ax, color=color_select)
  
    
    
    color_count += 1
    
h_fig_leg, h_ax_leg  = make_legend_standalone(CASE_PARAMS_LIST, LABEL_VARS, math_strings=math_strings, color_list=color_list)

h_ax[0,3].set_xscale('log')
h_ax[0,1].set_xscale('log')
h_ax[1,1].set_xscale('log')


fig_string = 'figure1_xi_profiles'

#h_fig.savefig(fig_string, dpi=300, bbox_inches='tight')
#h_fig_leg.savefig(fig_string+'_legend', dpi=300, bbox_inches='tight')
