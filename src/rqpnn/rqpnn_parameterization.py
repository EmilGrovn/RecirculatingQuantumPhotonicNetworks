#rqpnn_parameterization.py
#in this script, we define how to parameterize the rates and matrices in the RQPNN system

import jax.numpy as jnp
import jax.nn as jnn                #for jax neural network operations

#########
### produce physical properties of the RQPNN from parameterization
#########
#input: trainable parameters (datastructure that we choose) and fixed parameters, (parameterization to choose)
#output: C, S_in, Delta_cav, Delta_em, kappa, times
def generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict):
    
    params_ham_dict={
        "coeff_mat": None,
        "S_in_mat": None,
        "tau": None,
        "delta_cav": None,
        "delta_em": None,
        "kappa": None
    }
    #matrix parameterization
    
    C=build_coefficient_matrix(spec, params_train_dict['coeff_vec']) #parameters for training C
    S_in=S_in_general_from_C(spec,C,params_train_dict["S_in_Rmat"] )
    #assert jnp.allclose(S_in@(jnp.conjugate(S_in).T), jnp.eye(spec.M_cav), atol=1e-6), "S_in matrix is not unitary"
    params_ham_dict["coeff_mat"]=C
    params_ham_dict["S_in_mat"]=S_in  #parameters for training R#NOTE: S_in_mat is not used for any results in the paper

    #tau
    params_tau=params_train_dict['tau'] #trainable parameters for tau
    max_tau=params_amp_dict['max_tau'] #fixed parameters for tau
    min_tau=params_amp_dict['min_tau'] #fixed parameters for tau
    rates_tau=_generate_rates_tau(params_tau,max_tau,min_tau)
    params_ham_dict['tau']=rates_tau
    
    #delta_cav
    params_delta_cav=params_train_dict['delta_cav'] #trainable parameters for delta_cav
    amps_delta_cav=params_amp_dict['delta_cav'] #fixed amplitude
    rates_delta_cav=_generate_rates_delta_cav(params_delta_cav,amps_delta_cav)
    params_ham_dict['delta_cav']=rates_delta_cav

    #delta_em
    if spec.has_emitters:
        params_delta_em=params_train_dict['delta_em'] #trainable parameters for delta_em
        amps_delta_em=params_amp_dict['delta_em'] #fixed amplitude
        offset_delta_em=params_amp_dict['delta_em_offset'] #fixed offset
        rates_delta_em=_generate_rates_delta_em(params_delta_em,amps_delta_em,offset_delta_em)
        params_ham_dict['delta_em']=rates_delta_em

    #kappa
    params_kappa=params_train_dict['kappa'] #trainable parameters for kappa
    amps_kappa=params_amp_dict['kappa'] #fixed amplitude
    rates_kappa=_generate_rates_kappa(params_kappa,amps_kappa)
    params_ham_dict['kappa']=rates_kappa

    return params_ham_dict


### coefficient matrix - VERIFIED in test_matrix_parameterization.py
def build_coefficient_matrix(spec,coeff_vec):
    '''
    coeff_vec: jnp.vector of length N_cav*(N_cav-1)/2

    example: coeff_vec = jnp.array([1, 2, 3, 4])-> C = [[0, 1, 2, 3],
                                                     [1, 0, 4, 5],
                                                     [2, 4, 0, 6],
                                                     [3, 5, 6, 0]]
    '''
    A = jnp.zeros((spec.M_cav,spec.M_cav))
    i_upper, j_upper = jnp.triu_indices(spec.M_cav, k=1)
    A = A.at[i_upper,j_upper].set(coeff_vec)
    C=A+A.T
    return C
#Verified in test_matrix_parameterization.py
def S_in_sym_from_C(C):
    # C: real symmetric (n,n)
    lam, Q = jnp.linalg.eigh(C)                       # C = Q diag(lam) Q^T
    s = (2*lam+1j) / jnp.sqrt(1.0+4.0*lam**2)             #eigenvalues of s (could use sqrt1p)
    S_in = Q @ jnp.diag(s) @ Q.T                      # complex symmetric S
    return S_in
#Verified in test_matrix_parameterization.py
def generate_R(spec,para_mat):
    '''
    para_mat: jnp.array of dimension (N_cav*(N_cav-1)/2,)
    output:
    R: jnp.array of dimension (N_cav,N_cav), RR^T=I
    '''
    # generate K^T=-K
    A = jnp.zeros((spec.M_cav,spec.M_cav),dtype=jnp.complex64)
    i_upper, j_upper = jnp.triu_indices(spec.M_cav, k=1)
    A = A.at[i_upper,j_upper].set(para_mat)
    K=A-A.T
    # generate R
    B1=jnp.eye(K.shape[0])-K
    B2=jnp.eye(K.shape[0])+K
    #B1 @ R= B2 (=> R= jnp.linalg.inv(B1) @ B2) => R= jnp.linalg.solve(B1,B2)
    R=jnp.linalg.solve(B1,B2)
    return R
#Verified in test_matrix_parameterization.py
def S_in_general_from_C(spec,C,para_mat):
    '''
    C: jnp.array of dimension (N_cav,N_cav), real symmetric coefficient matrix
    para_mat: jnp.array of dimension (2,N_cav*(N_cav-1)/2)
    output:
    S_in: jnp.array of dimension (N_cav,N_cav), complex unitary scattering matrix
    '''
    R=generate_R(spec,para_mat)
    S_in_sym=S_in_sym_from_C(C)
    S_in=S_in_sym@R
    return S_in

### rates
## cavity detuning (VERIFIED in test_rates_parameterization.py)
def _generate_rates_delta_cav(params_delta,amps_delta):
    '''
    input:
    params_delta: jax array of shape (M_cav,N_steps)
    amps_delta: jax array of shape (N_steps,)
    output:
    rates_delta_mat: jax array of shape (M_cav,N_steps)
    '''
    rates_delta_mat=jnp.arctan(params_delta)*amps_delta/jnp.pi
    return rates_delta_mat

## emitter detuning
def _generate_rates_delta_em(params_delta,amps_delta,offset_delta):
    '''
    input:
    params_delta: jax array of shape (M_cav,N_steps)
    amps_delta: jax array of shape (N_steps,)
    offset_delta: constant jax array of shape (N_steps,), offset for delta_em
    output:
    rates_delta_mat: jax array of shape (M_cav,N_steps)
    '''
    rates_delta_mat=jnp.arctan(params_delta)*amps_delta/jnp.pi+offset_delta
    return rates_delta_mat

## cavity coupling (VERIFIED in test_rates_parameterization.py)
def _generate_rates_kappa(params_kappa,amps_kappa):
    rates_kappa_mat=(jnp.pi/2+jnp.arctan(params_kappa))*amps_kappa/jnp.pi
    return rates_kappa_mat

## time steps - verified in test_rates_parameterization.py
def _generate_rates_tau(params_tau,max_tau,min_tau):
    '''
    input:
    params_tau: jax array of shape (N_steps,)
    max_tau: constant jax array of shape (N_steps,), maximum time step
    min_tau: constant jax array of shape (N_steps,), minimum time step
    output:
    rates_tau: jax array of shape (N_steps,)
    '''
    rates_tau=jnn.sigmoid(params_tau)*(max_tau-min_tau)+min_tau
    return rates_tau