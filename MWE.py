import numpy as np

import matplotlib.pyplot as plt
from dedalus import public as de


from def_equations_binary import *
from def_plotting import *
from def_diagnostics import *
from def_utilities import *


read_dir = 'solutions_binary/'
sol_dir = 'solutions_binary/'








CASE = dict({'nz': 512, 'theta': 0.04, 'lr': 1.1,'K': 1000, 'xc': 0.001, 'lm': 10000, 'B': 500})





flist = get_file_list(CASE, dirname=read_dir)
if not flist:
    print(CASE)
    print('No such case found')
else:
    sd_data = read_solution_from_file(flist[0])
    
    
#%% plot every variable in solution file    

h_fig, h_ax = showAll(sd_data)

#%% plot specified variables

VAR_LIST = ['Xi','fS', 'uS', 'uL']
TITLE_LIST = ['\\xi^l','\phi^s',  'u_z^s', 'u_z^l']


h_fig2, h_ax2 = plotSpecified(sd_data, VAR_LIST, math_titles=TITLE_LIST)

#%% Use loaded solution as initial condition 

init_data = sd_data
parameters = sd_data['parameters'].copy()
parameters['xi_core'] = 0.0015

solution, error = binaryslurryODEs(parameters, init_data)


if error > 1:
    print("---------------------------------------------------")
    print("Solver failed to converge")
    print("---------------------------------------------------")
else:
    data_p = extractVars(solution, parameters, dname=sol_dir, saving=True)
    h_fig, h_ax = showAll(data_p, fig=h_fig, ax=h_ax )


    figname = genFilename(parameters)
    h_fig.savefig(figname+'.png', dpi=300)
    #plt.close(h_fig)
    
    
