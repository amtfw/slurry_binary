#--------------------------------
# Calculations of diagnostic quantities
#--------------------------------

import numpy as np
from dedalus import public as de



def diag_fluxes(solution_data):
    # load variables
    z = solution_data['z']
    rhoS = solution_data['rhoS']
    rhoL = solution_data['rhoL']
    uS = solution_data['uS']
    uL = solution_data['uL']
    uS_z = solution_data['uS_z']
    uL_z = solution_data['uL_z']
    mDotS = solution_data['mDotS']
    fS = solution_data['fS']
    fS_z = solution_data['fS_z']
    T = solution_data['T']
    Tz = solution_data['Tz']
    Pz = solution_data['Pz']
    # load parameters
    nz = solution_data['parameters']['nz']
    theta = solution_data['parameters']['theta']

    B = solution_data['parameters']['B']
    Pr = solution_data['parameters']['Pr']
    St = solution_data['parameters']['St']
    D = solution_data['parameters']['D']
    K = solution_data['parameters']['K']

    alpha = solution_data['parameters']['alpha']
    beta = solution_data['parameters']['beta']
    la = solution_data['parameters']['lambda_alpha']
    lb = solution_data['parameters']['lambda_beta']

    lr = solution_data['parameters']['lambda_rho']
    if 'lambda_mu' in solution_data['parameters']:
        lm = solution_data['parameters']['lambda_mu']
    else:
        lm = B*solution_data['parameters']['M']


    Ra = Pr*B**2 ###

    fL = 1-fS
    fL_z = -fS_z
    rhoMix = (lr*fS*rhoS + fL*rhoL)
    rhoIntersect=fS*rhoS*fL*rhoL/rhoMix
    w = uS-uL


    jz = w*fS*rhoS*fL*rhoL/rhoMix

    uS_top = uS[-1]
    uL_top = uL[-1]
    fS_top = fS[-1]
    fL_top = 1-fS[-1]
    rhoS_top = rhoS[-1]

    fS_max_idx = fS.argmax()
    z_fS_max = z[fS_max_idx]
    fS_max = fS[fS_max_idx]

    fS_min_idx = fS.argmin()
    z_fS_min = z[fS_min_idx]
    fS_min = fS[fS_min_idx]

    uL_max_idx = uL.argmax()
    z_uL_max = z[uL_max_idx]
    uL_max = uL[uL_max_idx]

    uS_max_idx = abs(uS).argmax()
    z_uS_max = z[uS_max_idx]
    uS_max = uS[uS_max_idx]

    z_basis = de.Chebyshev('z', nz, interval=(0,1))
    domain = de.Domain([z_basis], grid_dtype=np.float64)



    fS_field = domain.new_field(name='fS')
    fS_field['g'] = fS
    fS_integral = fS_field.integrate('z')
    fS_avg = fS_integral['g'][0]

    uS_field = domain.new_field(name='uS')
    uS_field['g'] = uS
    uS_integral = uS_field.integrate('z')
    uS_avg = uS_integral['g'][0]

    uS_scaled = uS_avg*K/(lr-1)

    uL_field = domain.new_field(name='uL')
    uL_field['g'] = uL
    uL_integral = uL_field.integrate('z')
    uL_avg = uL_integral['g'][0]
    uL_scaled = uL_avg*K/(lr-1)

    rhoS_field = domain.new_field(name='rhoS')
    rhoS_field['g'] = rhoS
    rhoS_z_field = rhoS_field.differentiate(z=1)
    rhoS_z = rhoS_z_field['g']

    rhoL_field = domain.new_field(name='rhoL')
    rhoL_field['g'] = rhoL
    rhoL_z_field = rhoL_field.differentiate(z=1)
    rhoL_z = rhoL_z_field['g']

    rho = (lr*fS*rhoS + fL*rhoL)

    rho_top = rho[-1]
    rho_bot = rho[0]

    R_rho = rho_top/rho_bot

    rho_z = lr*(fS_z*rhoS + fS*rhoS_z) + fL_z*rhoL + fL*rhoL_z


    gradRho_max_idx = rho_z.argmax()
    z_gradRho_max = z[gradRho_max_idx]
    gradRho_max = rho_z[gradRho_max_idx]


    rSfSuS_field = domain.new_field(name='rSfSuS')
    rSfSuS_field['g'] = rhoS*fS*uS
    rSfSuS_integral = rSfSuS_field.integrate('z')
    rSfSuS_avg = rSfSuS_integral['g'][0]


    rLfLuL_field = domain.new_field(name='rLfLuL')
    rLfLuL_field['g'] = rhoL*fL*uL
    rLfLuL_integral = rLfLuL_field.integrate('z')
    rLfLuL_avg = rLfLuL_integral['g'][0]


    jz_field = domain.new_field(name='jz')
    jz_field['g'] = jz
    jz_integral = jz_field.integrate('z')

    mDotS_field = domain.new_field(name='mDotS')
    mDotS_field['g'] = mDotS
    mDotS_integral = mDotS_field.integrate('z')

    heatPipe_field = domain.new_field(name='heatPipe')
    heatPipe_field['g'] = rhoIntersect*Tz*w/T
    heatPipe_integral = heatPipe_field.integrate('z')

    QvS_field = domain.new_field(name='QvS')
    QvS_field['g'] = (4./3)*D*Pr*lm*fS*uS_z**2
    QvS_integral = QvS_field.integrate('z')

    QvL_field = domain.new_field(name='QvL')
    QvL_field['g'] = (4./3)*D*Pr*fL*uL_z**2
    QvL_integral = QvL_field.integrate('z')

    Qf_field = domain.new_field(name='Qf')
    Qf_field['g'] = (Ra*Pr)**0.5 *D*K*rhoIntersect* w**2
    Qf_integral = Qf_field.integrate('z')

    Qc_left = -Tz[0]
    Qc_right = -Tz[-1]
    Qc_ratio = -Tz[-1]/theta


    mDotS_avg = mDotS_integral['g'][0]
    Ql = (Ra*Pr)**0.5 * mDotS_integral['g'][0]/St
    Qp = -(Ra*Pr)**0.5 * lr * heatPipe_integral['g'][0]/St
    QvS = QvS_integral['g'][0]
    QvL = QvL_integral['g'][0]
    Qf = Qf_integral['g'][0]

    jz_avg = jz_integral['g'][0]
    jz_scaled = jz_avg*K

    ReL_avg = (Pr/Ra)**-0.5  * uL_avg
    ReL_max = (Pr/Ra)**-0.5  * uL_max
    ReS_avg = (Pr/Ra)**-0.5  * uS_avg /lm
    ReS_max = (Pr/Ra)**-0.5  * uS_max /lm

    fluxes = dict({'Qc_left': Qc_left,'Qc_right': Qc_right,'Qc_ratio': Qc_ratio,'Ql':Ql,
                    'Qp':Qp,'QvS':QvS,'QvL':QvL,'Qf':Qf,'mDotS_avg':mDotS_avg,
                    'uS_top': uS_top, 'uL_top':uL_top, 'fS_top':fS_top, 'fL_top':fL_top,  'jz_avg':jz_avg,'jz_scaled': jz_scaled,
                    'uS_avg': uS_avg, 'uL_avg':uL_avg, 'fS_avg':fS_avg, 'uS_scaled': uS_scaled, 'uL_scaled': uL_scaled,
                    'rSfSuS_avg': rSfSuS_avg, 'rLfLuL_avg': rLfLuL_avg,
                    'uS_max': uS_max, 'z_uS_max': z_uS_max, 'ReS_max': ReS_max, 'ReS_avg': ReS_avg,
                    'uL_max': uL_max, 'z_uL_max': z_uL_max, 'ReL_max': ReL_max, 'ReL_avg': ReL_avg,
                    'fS_max': fS_max, 'z_fS_max': z_fS_max,
                    'fS_min': fS_min, 'z_fS_min': z_fS_min,
                    'gradRho_max': gradRho_max, 'z_gradRho_max': z_gradRho_max,
                    'rho_top': rho_top, 'rho_bot':rho_bot, 'R_rho':R_rho})

    momentum_diags = diag_momentum(solution_data)

    fluxes.update(momentum_diags)


    if 'Xi' in solution_data:
        concentration_diags = diag_concentration(solution_data)
        fluxes.update(concentration_diags)

    return fluxes


def diag_momentum(solution_data):
    # load variables
    z = solution_data['z']
    rhoS = solution_data['rhoS']
    rhoL = solution_data['rhoL']
    uS = solution_data['uS']
    uL = solution_data['uL']
    uS_z = solution_data['uS_z']
    uL_z = solution_data['uL_z']
    mDotS = solution_data['mDotS']
    fS = solution_data['fS']
    fS_z = solution_data['fS_z']
    Pz = solution_data['Pz']
    # load parameters
    nz = solution_data['parameters']['nz']
    B = solution_data['parameters']['B']
    Pr = solution_data['parameters']['Pr']
    K = solution_data['parameters']['K']
    lr = solution_data['parameters']['lambda_rho']
    if 'lambda_mu' in solution_data['parameters']:
        lm = solution_data['parameters']['lambda_mu']
    else:
        lm = B*solution_data['parameters']['M']

    fL = 1-fS
    fL_z = -fS_z
    rhoMix = (lr*fS*rhoS + fL*rhoL)
    w = uS-uL
    c3 = 0.75*B

    z_basis = de.Chebyshev('z', nz, interval=(0,1))
    domain = de.Domain([z_basis], grid_dtype=np.float64)
    uL_z_field = domain.new_field(name='uL_z')
    uL_z_field['g'] = uL_z
    uL_zz_field = uL_z_field.differentiate(z=1)
    uS_z_field = domain.new_field(name='uS_z')
    uS_z_field['g'] = uS_z
    uS_zz_field = uS_z_field.differentiate(z=1)

    uL_zz = uL_zz_field['g']
    uS_zz = uS_zz_field['g']

    inertiaL = fL*rhoL*uL*uL_z
    inertiaS = lr*fS*rhoS*uS*uS_z
    PzS = -fS*Pz
    PzL = -fL*Pz
    buoyL = -fL*rhoL
    buoyS = -lr*fS*rhoS
    frictionS = -K*fS*fL*rhoS*rhoL*w/(rhoMix)
    frictionL = -frictionS
    phaseCh = -0.5*mDotS*w
    viscL = (1./c3)*(fL_z*uL_z + fL*uL_zz)
    viscS = (lm/c3)*(fS_z*uS_z + fS*uS_zz)


    diag_names = ['inertiaL', 'PzL', 'buoyL', 'frictionL', 'phaseCh', 'viscL',
                    'inertiaS', 'PzS', 'buoyS', 'viscS']

    diag_data = [inertiaL, PzL, buoyL, frictionL, phaseCh, viscL, inertiaS, PzS, buoyS, viscS]

    momentum_diags = dict()

    for count, name_count in enumerate(diag_names):
        temp_grid_data = diag_data[count]
        temp_field = domain.new_field(name=name_count)
        temp_field['g'] = temp_grid_data
        temp_integral = temp_field.integrate('z')
        temp_avg = temp_integral['g'][0]
        momentum_diags[name_count] = temp_avg

    return momentum_diags


def diag_concentration(solution_data):
    # load variables
    z = solution_data['z']

    rhoL = solution_data['rhoL']

    fS = solution_data['fS']

    T = solution_data['T']
    Tz = solution_data['Tz']
    Pz = solution_data['Pz']

    # load parameters
    nz = solution_data['parameters']['nz']
    theta = solution_data['parameters']['theta']
    B = solution_data['parameters']['B']
    Pr = solution_data['parameters']['Pr']

    D = solution_data['parameters']['D']

    fL = 1-fS




    chi = solution_data['parameters']['chi']
    alpha_xi = solution_data['parameters']['alpha_xi']
    tau = solution_data['parameters']['tau']
    xi_core = solution_data['parameters']['xi_core']
    Xi = solution_data['Xi']
    Xi_z = solution_data['Xi_z']

    c4 = tau*(B*Pr)**-1
    c5 = alpha_xi*chi*D

    Xi_diffusion = c4*(Xi_z)
    Barodiffusion = c4*c5*(Xi*Pz/T)

    J_Xi = rhoL*fL*( Xi_diffusion + Barodiffusion )


    z_basis = de.Chebyshev('z', nz, interval=(0,1))
    domain = de.Domain([z_basis], grid_dtype=np.float64)

    H_xi = Xi_z/Xi
    H_xi_field = domain.new_field(name='H_xi')
    H_xi_field['g'] = H_xi
    H_xi_integral = H_xi_field.integrate('z')
    H_xi_avg = H_xi_integral['g'][0]

    Xi_field = domain.new_field(name='Xi')
    Xi_field['g'] = Xi
    Xi_integral = Xi_field.integrate('z')
    Xi_avg = Xi_integral['g'][0]

    Xi_z_field = domain.new_field(name='Xi_z')
    Xi_z_field['g'] = Xi_z
    Xi_z_integral = Xi_z_field.integrate('z')
    Xi_z_avg = Xi_z_integral['g'][0]

    J_Xi_field = domain.new_field(name='J_Xi')
    J_Xi_field['g'] = J_Xi
    J_Xi_integral = J_Xi_field.integrate('z')
    J_Xi_avg = J_Xi_integral['g'][0]

    J_Xi_scaled = J_Xi_avg/c4


    heatReaction= (chi*tau/c2)*(fL*Xi/T)*( T*Xi_z/(chi*Xi) + alpha_xi*D*Pz )*( T*Xi_z/(chi*Xi) + alpha_xi*D*Pz )
    Qxi_field = domain.new_field(name='Qxi')
    Qxi_field['g'] = heatReaction
    Qxi_integral = Qxi_field.integrate('z')
    Qxi_avg = Qxi_integral['g'][0]

    Xi_bot = Xi[0]
    Xi_z_bot = Xi_z[0]
    Xi_z_top = Xi_z[-1]

    fluxes = dict({'H_xi_avg': H_xi_avg, 'Xi_bot':Xi_bot, 'Xi_z_bot':Xi_z_bot, 'Xi_z_top':Xi_z_top,
                    'Xi_avg':Xi_avg,'Xi_z_avg':Xi_z_avg, 'J_Xi_avg':J_Xi_avg, 'J_Xi_scaled': J_Xi_scaled, 'Qxi_avg': Qxi_avg})

    return fluxes
