import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import h5py
import time
import pickle
import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD

def binaryslurryODEs(parameters, init_guess):
    # Parameters
    tolerance = 1e-7
    # Unpack parameters
    nz = parameters['nz']
    theta = parameters['theta']
    B = parameters['B']
    Pr = parameters['Pr']
    St = parameters['St']
    D = parameters['D']
    K = parameters['K']
    tau = parameters['tau']
    chi = parameters['chi']
    xi_core = parameters['xi_core']
    alpha = parameters['alpha']
    alpha_xi = parameters['alpha_xi']
    beta = parameters['beta']
    lambda_alpha = parameters['lambda_alpha']
    lambda_beta = parameters['lambda_beta']
    #lambda_mu = parameters['lambda_mu']
    M = parameters['M']
    lambda_rho = parameters['lambda_rho']
    U_ICB = parameters['U_ICB']

    # Domain

    z_basis = de.Chebyshev('z', nz, interval=(0,1))
    domain = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)
    z = domain.grid(0)
    # Problem
    problem = de.NLBVP(domain, variables=['T','Tz','Pz','rhoS', 'rhoL', 'uS', 'uS_z','uL','uL_z', 'fS', 'fS_z', 'mDotS','Xi','Xi_z', 'Sigma_Xi', 'Xi_bc_res', 'uL_bc_res'], ncc_cutoff=1e-9)
    problem.parameters['c0'] = 1.-1/lambda_rho
    problem.parameters['c1'] = 1/(St*D)
    problem.parameters['c2'] = B*Pr
    problem.parameters['c3'] = 0.75*B
    problem.parameters['c4'] = tau*(B*Pr)**-1
    problem.parameters['c5'] = alpha_xi*chi*D

    problem.parameters['St'] = St
    problem.parameters['K'] = K
    problem.parameters['D'] = D
    problem.parameters['la'] = lambda_alpha
    problem.parameters['lb'] = lambda_beta
    problem.parameters['lm'] = M*B
    problem.parameters['lr'] = lambda_rho
    problem.parameters['alpha'] = alpha
    problem.parameters['beta'] = beta
    problem.parameters['theta'] = theta
    problem.parameters['chi'] = chi
    problem.parameters['alpha_xi'] = alpha_xi
    problem.parameters['tau'] = tau
    problem.parameters['xi_core'] = xi_core
    problem.parameters['U_ICB'] = U_ICB


    #problem.substitutions['c0'] = "1.0/rhoL - 1.0/(lr*rhoS)"
    problem.substitutions['w'] = "(uS-uL)"
    problem.substitutions['fL'] = "(1-fS)"
    problem.substitutions['rhoMix'] = "(lr*fS*rhoS + fL*rhoL)"
    problem.substitutions['rhoIntersect'] = "(rhoS*rhoL/rhoMix)"
    problem.substitutions['rhoU'] =" (lr*fS*rhoS*uS + fL*rhoL*uL) "
    problem.substitutions['heatAdvection'] =  '(-lr*U_ICB*(Tz-alpha*D*T*Pz))'
    problem.substitutions['heatPipe'] = "lr*fS*rhoS*fL*rhoL*w*Tz/(St*rhoMix*T)"
    problem.substitutions['heatFriction'] = "D*K*fS*fL*rhoIntersect*w*w"
    problem.substitutions['heatViscous'] = "(D/c3)*(lm*fS*uS_z*uS_z + fL*uL_z*uL_z)"
    problem.substitutions['heatReaction'] = "(chi*tau/c2)*(fL*Xi/T)*( T*Xi_z/(chi*Xi) + alpha_xi*D*Pz )*( T*Xi_z/(chi*Xi) + alpha_xi*D*Pz )"

    problem.substitutions['mSbyfS'] = "lr*(rhoS*uS_z + uS*dz(rhoS) + rhoS*uS*fS_z/fS)"
    problem.substitutions['mSbyfL'] = "(-rhoL*uL_z - uL*dz(rhoL) + rhoL*uL*fS_z/fL)"
    problem.substitutions['dlnfS'] = "(fS_z/fS)"
    problem.substitutions['dlnfL'] = "(-fS_z/fL)"



    problem.add_equation("mDotS  =  lr*(rhoS*uS*fS_z + fS*rhoS*uS_z + fS*uS*dz(rhoS))")
    problem.add_equation("mDotS  =  rhoL*uL*fS_z -(1-fS)*( rhoL*uL_z + uL*dz(rhoL))")
    problem.add_equation("dz(fS)- fS_z = 0", tau=0)
    problem.add_equation("dz(rhoS) = rhoS*(lb*beta*Pz - la*alpha*Tz)")
    problem.add_equation("dz(rhoL) = rhoL*(beta*Pz - alpha*Tz - alpha_xi*Xi_z  )")
    problem.add_equation("Pz = ((c1/T)*Tz + (alpha_xi*Xi*Pz + T*Xi_z/(chi*D)) )/c0 ")
    problem.add_equation("dz(T) - Tz = 0")
    problem.add_equation("dz(Sigma_Xi) = 0")
    problem.add_equation("Xi_bc_res = Sigma_Xi - Xi*fL*U_ICB")
    problem.add_equation("dz(Xi) - Xi_z = 0")  #
    problem.add_equation("Xi_z = Xi*uL/c4 - (Sigma_Xi/fL)/c4 - c5*Xi*Pz/T")
    problem.add_equation("dz(Tz) = c2*(heatAdvection + heatPipe -mDotS/St - heatFriction -heatViscous - heatReaction ) ")
    problem.add_equation("dz(uS) - uS_z = 0")
    problem.add_equation("dz(uS_z) - (c3/lm)*( lr*rhoS + Pz ) = -dlnfS*uS_z + (c3/lm)*( lr*rhoS*uS*uS_z + fL*rhoIntersect*K*w + 0.5*mSbyfS*w)")
    problem.add_equation("dz(uL) - uL_z = 0")
    problem.add_equation("dz(uL_z) - c3*(rhoL + Pz) = -dlnfL*uL_z + c3*( rhoL*uL*uL_z  - fS*rhoIntersect*K*w + 0.5*mSbyfL*w)")
    problem.add_equation("uL_bc_res = fL*uL")

    problem.add_bc("left(Xi_bc_res)  = 0")
    problem.add_bc("left(rhoS) = 1")
    problem.add_bc("left(rhoL) = 1")
    problem.add_bc("right(T)   = 1")
    problem.add_bc("left(Tz)  = -theta")
    problem.add_bc("left(uL_bc_res)   = 0")
    problem.add_bc("left(uS)   = 0")
    problem.add_bc("right(uL_z)  = 0")
    problem.add_bc("right(uS_z)  = 0")
    problem.add_bc("right(Xi)  = xi_core")


    # Initial guess
    solver = problem.build_solver()
    z = domain.grid(0, scales=domain.dealias)

    T = solver.state['T']
    T.set_scales(domain.dealias)
    Tz = solver.state['Tz']
    Tz.set_scales(domain.dealias)
    Pz = solver.state['Pz']
    Pz.set_scales(domain.dealias)
    rhoS = solver.state['rhoS']
    rhoL = solver.state['rhoL']
    rhoS.set_scales(domain.dealias)
    rhoL.set_scales(domain.dealias)
    uS = solver.state['uS']
    uL = solver.state['uL']
    uS.set_scales(domain.dealias)
    uL.set_scales(domain.dealias)
    uS_z = solver.state['uS_z']
    uL_z = solver.state['uL_z']
    uS_z.set_scales(domain.dealias)
    uL_z.set_scales(domain.dealias)
    mDotS = solver.state['mDotS']
    mDotS.set_scales(domain.dealias)
    fS = solver.state['fS']
    fS.set_scales(domain.dealias)
    fS_z = solver.state['fS_z']
    fS_z.set_scales(domain.dealias)
    Xi = solver.state['Xi']
    Xi.set_scales(domain.dealias)
    Xi_z = solver.state['Xi_z']
    Xi_z.set_scales(domain.dealias)
    #Sigma_Xi = solver.state['Sigma_Xi']
    #Sigma_Xi.set_scales(domain.dealias)
    #Xi_bc_res = solver.state['Xi_bc_res']
    #Xi_bc_res.set_scales(domain.dealias)

    z_guess = init_guess['z']

    if np.array_equal(z, z_guess):
        print('z grids are matching')
        T['g'] = init_guess['T']
        Tz['g'] = init_guess['Tz']
        rhoS['g'] = init_guess['rhoS']
        rhoL['g'] = init_guess['rhoL']
        uS['g'] = init_guess['uS']
        uS_z['g'] = init_guess['uS_z']
        uL['g'] = init_guess['uL']
        uL_z['g'] = init_guess['uL_z']
        fS['g'] = init_guess['fS']
        fS_z['g'] = init_guess['fS_z']
        mDotS['g'] = init_guess['mDotS']
        Pz['g'] = init_guess['Pz']
        if "Xi" in init_guess:
            Xi['g'] = init_guess['Xi']
        if "Xi_z" in init_guess:
            Xi_z['g'] = init_guess['Xi_z']

    else:
        T['g'] = np.interp(z, z_guess, init_guess['T'])
        Tz['g'] = np.interp(z, z_guess, init_guess['Tz'])
        rhoS['g'] = np.interp(z, z_guess, init_guess['rhoS'])
        rhoL['g'] = np.interp(z, z_guess, init_guess['rhoL'])
        uS['g'] = np.interp(z, z_guess, init_guess['uS'])
        uS_z['g'] = np.interp(z, z_guess, init_guess['uS_z'])
        uL['g'] = np.interp(z, z_guess, init_guess['uL'])
        uL_z['g'] = np.interp(z, z_guess, init_guess['uL_z'])
        fS['g'] = np.interp(z, z_guess, init_guess['fS'])
        fS_z['g'] = np.interp(z, z_guess, init_guess['fS_z'])
        mDotS['g'] = np.interp(z, z_guess, init_guess['mDotS'])
        Pz['g'] = np.interp(z, z_guess, init_guess['Pz'])
        if "Xi" in init_guess:
            Xi['g'] = np.interp(z, z_guess, init_guess['Xi'])
        if "Xi_z" in init_guess:
            Xi_z['g'] = np.interp(z, z_guess, init_guess['Xi_z'])

    # Iterations
    pert = solver.perturbations.data
    pert.fill(1+tolerance)

    solver.evaluator.evaluate_group("diagnostics")

    iter = 0
    max_iter = 200
    start_time = time.time()
    try:
        while np.sum(np.abs(pert)) > tolerance and np.sum(np.abs(pert)) < 1e6 and iter < max_iter:
            solver.newton_iteration()
            pert_norm = np.sum(np.abs(pert))
            logger.info('Perturbation norm: {:g}'.format(pert_norm))
            iter += 1
    except:
        raise

    if iter == max_iter:
        pert_norm = 1e6

    end_time = time.time()
    logger.info("Finished in {:d} iter and in {:g} seconds".format(iter, end_time-start_time))

    return solver, pert_norm
