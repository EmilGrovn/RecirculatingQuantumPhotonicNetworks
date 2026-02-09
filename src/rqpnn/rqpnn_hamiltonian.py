### 
# rqpnn_hamiltonian.py
import dynamiqs as dq
import jax
import jax.numpy as jnp
from jax import Array as jaxArray
from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params

# main function
def build_H_func(spec,dict_ham_ops,N_pieces,params_amp_dict):
    #time grid
    print("N_pieces",N_pieces)
    t_grid=jnp.linspace(0,N_pieces,N_pieces+1)#essentatial

    ## prepare operator dict
    dict_ops_build_ham={}
    
    if spec.M_em==0:
        #tau
        chi3_op=dict_ham_ops["chi3"][0]
        for i, ops_i in enumerate(dict_ham_ops["chi3"][1:]):
            chi3_op+=ops_i
        dict_ops_build_ham["tau"]=-chi3_op#NOTE: WE HAVE NOW CHANGED THE SIGN HERE!
        
        dict_ops_build_ham["delta_cav"]=dict_ham_ops["delta_cav"]
        dict_ops_build_ham["int"]=dict_ham_ops["int"]
    
    elif spec.M_em==spec.M_cav:
        cav_em_op=dict_ham_ops["g_lm"][0]
        for i, ops_i in enumerate(dict_ham_ops["g_lm"][1:]):
            cav_em_op+=ops_i
        dict_ops_build_ham["tau"]=cav_em_op

        dict_ops_build_ham["delta_cav"]  =dict_ham_ops["delta_cav"]
        dict_ops_build_ham["delta_em"]   =dict_ham_ops["delta_em"]
        dict_ops_build_ham["int"]        =dict_ham_ops["int"]

    else:
        raise ValueError("Require M_em==0 or M_em==M_cav")

    if spec.M_em==0:
        def H_func(params_train_dict):
            '''
            Function that returns the Hamiltonian for the cavity network, given the optimizable parameters as input
            '''
            print("building hamiltonian")

            params_ham_dict=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)

            H=_H_pwc_init_tau(params_ham_dict["tau"],t_grid,dict_ops_build_ham["tau"])
            H+=_H_pwc_init_delta(params_ham_dict["delta_cav"],t_grid,dict_ops_build_ham["delta_cav"])
            H+=_H_pwc_init_coupling(params_ham_dict["kappa"],params_ham_dict["coeff_mat"],t_grid,dict_ops_build_ham["int"])
            return H
    elif spec.M_em==spec.M_cav:
        def H_func(params_train_dict):
            '''
            Function that returns the Hamiltonian for the cavity network, given the optimizable parameters as input
            '''

            print("building hamiltonian")
            params_ham_dict=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
            
            H=_H_pwc_init_tau(params_ham_dict["tau"],t_grid,dict_ops_build_ham["tau"])
            H+=_H_pwc_init_delta(params_ham_dict["delta_cav"],t_grid,dict_ops_build_ham["delta_cav"])
            H+=_H_pwc_init_delta(params_ham_dict["delta_em"],t_grid,dict_ops_build_ham["delta_em"])
            H+=_H_pwc_init_coupling(params_ham_dict["kappa"],params_ham_dict["coeff_mat"],t_grid,dict_ops_build_ham["int"])
            return H
    return H_func #jax.jit(H_func)

### --------- END build parametrized hamiltonian (for a fixed scattering matrix and pwc Hamiltonian)

#helpers
##tau
# tau init hamiltonian
def _H_pwc_init_tau(tau_vec, t_grid, op) -> dq.TimeQArray:
    H = dq.pwc(t_grid, tau_vec, op)
    return H

##delta_cav
#delta_cav hamiltonian
def _H_pwc_init_delta(rate_mat, t_grid, op_list) -> dq.TimeQArray:
    '''
    parameters:
    ----------
    - params: 2D jaxArray, (N_cav, N_steps)
    - ops: list of number operator for each cavity (n_cavities elements)
    - n_grid: number of grid points for the time-varying signals (int)
    returns:
    -------
    - H: time-dependent Hamiltonian (object: dynamiqs.qarrays.pwc_qarray.PWCQArray)
    '''

    H=dq.pwc(t_grid, rate_mat[0], op_list[0])
    for idx,op_delta_n in enumerate(op_list[1:]):
        H += dq.pwc(t_grid, rate_mat[idx+1], op_delta_n)

    return H


#interaction hamiltonian init
#kappa and scattering
def _H_pwc_init_coupling(kappa_mat, coeff_mat, t_grid, op_nestedlist) -> dq.TimeQArray:
    H=dq.pwc(t_grid, jnp.zeros_like(kappa_mat[0]), op_nestedlist[0][0])   #initialize Hamiltonian to zero
    #begin
    for idx_cav_n,ops_list_n in enumerate(op_nestedlist):
        rates_kappa_n=kappa_mat[idx_cav_n]
        for idx_cav_m,op_nm in enumerate(ops_list_n):
            #coefficient matrix
            coeff_nm=coeff_mat[idx_cav_n,idx_cav_m]
            #calculate prefactor
            prefactor_nm=jnp.sqrt(rates_kappa_n.conj()*kappa_mat[idx_cav_m])*coeff_nm
            #calculate H_nm
            H_nm=dq.pwc(t_grid, prefactor_nm, op_nm)
            H+=H_nm
    
    return H


