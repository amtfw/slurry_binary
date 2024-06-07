import numpy as np
import pickle
import os
from dedalus import public as de


# def load_from_file(filepath):
#     with open(filepath, 'rb') as f:
#         data = pickle.load(f)
#     return data

def read_solution_from_file(filepath):
    with open(filepath, 'rb') as f:
        solution_data = pickle.load(f)
    return solution_data


def save_data_to_file(data_dictionary, fname=None, dname=None):

    if fname is None:
        fname = genFilename(data_dictionary['parameters'])+'.data'

    if dname is None:
        save_path = fname
    else:
        if not os.path.exists(dname):
            os.mkdir(dname)
        save_path = os.path.join(dname, fname)

    with open(save_path, 'wb') as f:
        pickle.dump(data_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Solution data saved")

    return


def extractVars(solver, parameters, saving=False, fname=None, dname=None):

    z = solver.domain.grid(0, scales=solver.domain.dealias)
    data_out = dict()
    data_out['parameters'] = parameters.copy()
    data_out['z'] = z

    fields = solver.state.field_dict
    for key in fields.keys():
        data_out[key] = fields[key]['g']

    if saving is True:
        save_data_to_file(data_out, fname=fname, dname=dname)

    return data_out


def genFilename(params):
    dict_keys = list(params.keys())
    if "Lz" in dict_keys:
        dict_keys.remove('Lz')

    fstr="sd"
    for name in dict_keys:
        if '_' in name:
            pos = name.find('_')
            name_save = name[0]+name[pos+1]
        else:
            name_save = name

        parval = "{:g}".format(params[name])
        parval_p = parval.replace('.', 'p') # change periods '.' into 'p'
        temp = "_{}_{}".format(name_save, parval_p)
        fstr += temp

    fstr += '_ds'
    return fstr


def logrange(start_power, stop_power, dt=1.0, lowerbound=None, upperbound=None):
    N = stop_power - start_power + 1
    powers = np.linspace(start_power, stop_power, num=N, endpoint=True)
    rp_list=[];
    for p in powers:
        rp = np.arange(1,10,dt)*10**p
        rp_list.append(rp)
    rp_list.append(10**(p+1))
    rp_array = np.hstack(rp_list)

    if lowerbound is not None:
        trim_left = rp_array < lowerbound
        rp_array_trimmed = rp_array[~trim_left]
        rp_array = rp_array_trimmed

    if upperbound is not None:
        trim_right = rp_array > upperbound
        rp_array_trimmed = rp_array[~trim_right]
        rp_array = rp_array_trimmed

    return rp_array
