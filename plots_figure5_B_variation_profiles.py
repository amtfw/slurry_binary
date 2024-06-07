import numpy as np

import matplotlib.pyplot as plt
from dedalus import public as de
import pickle


from def_plotting import *
from def_utilities import *



read_dir = 'solutions_binary_LMOB/'#'solutions_binary_p/'

plot_dir='plots_analysis/'

# Load starting point solution


sd ='sd_nz_512_theta_0p04_Pr_0p1_St_4_D_0p1_K_1000_alpha_0p05_beta_0p005_la_1_lb_1_lr_1p1_lc_1_lk_1_chi_1p38_ax_1p1_tau_0p001_xc_0p01_M_1000_B_1000_ds'
sol_data = read_solution_from_file(os.path.join(read_dir,sd+'.data'))

    
#fig, ax = plot_balance(sol_data, equation_label='temperature')




CASE_PARAMETERS = sol_data['parameters'].copy()


LABEL_VARS = ['B']

math_strings = ['B']
par_name = 'B'

par_values = [100, 500, 1000, 5000, 10000]




#par_values.reverse()

LIST_CASES = []

for pval in par_values:
    pdict = dict({par_name: pval})
    LIST_CASES.append(pdict)



#LIST_CASES = [ dict({'theta': 0.04}),dict({'theta': 0.0405}),dict({'theta': 0.041})
#              ,dict({'theta': 0.0415}),dict({'theta': 0.042}), dict({'theta': 0.0425}),
#              dict({'theta': 0.043})] 

FILE_LIST = []

for CASE in LIST_CASES:
    for key in list(CASE.keys()):
        CASE_PARAMETERS[key] = CASE[key]
        
    fname = genFilename(CASE_PARAMETERS)+'.data'
    
    
    if not os.path.exists( os.path.join(read_dir, fname)):
        print(fname)
        print('No such case found')
    else:
        FILE_LIST.append(os.path.join(read_dir, fname))
        
        
        
NUM_COLORS = len(FILE_LIST)

color_list = create_color_list('viridis_r', NUM_COLORS, offset=0.0)

color_count = 0
CASE_PARAMS_LIST = []
h_fig=None
h_ax=None

h_fig_u =None
h_ax_u = None


VAR_LIST = ['mDotS','fS', 'uS', 'uL']
TITLE_LIST = ['\\Gamma_\\rho^s','\phi^s',  'u_z^s', 'u_z^l']




for sd_filename in FILE_LIST: 
    sd_data = read_solution_from_file(sd_filename)
    CASE_PARAMS_LIST.append(sd_data['parameters'])
    color_select = color_list[color_count]
    #h_fig, h_ax = plot_balance(sd_data, equation_label='momentumS3', fig=h_fig, ax=h_ax, color=color_select)
    #h_fig, h_ax = plotSingle(sd_data, 'fS',math_titles='$\\phi^s$', fig=h_fig, ax_tupple=h_ax, color=color_select, XSCALE='linear', YSCALE='linear')
    h_fig, h_ax = plotSpecified(sd_data, VAR_LIST, math_titles=TITLE_LIST, fig=h_fig, ax_tupple=h_ax, color=color_select)
    #h_fig, h_ax = plotStability(sd_data, fig=h_fig, ax_tupple=h_ax, color=color_select)
    
    color_count += 1
    
h_fig_leg, h_ax_leg  = make_legend_standalone(CASE_PARAMS_LIST, LABEL_VARS, math_strings=math_strings, color_list=color_list)

#h_ax[0].set_xscale('log')
h_ax[1].set_xscale('log')
h_ax[3].set_xscale('log')


fig_string = 'figure5_B_profiles'

h_fig.savefig(fig_string, dpi=300, bbox_inches='tight')
h_fig_leg.savefig(fig_string+'_legend', dpi=300, bbox_inches='tight')
