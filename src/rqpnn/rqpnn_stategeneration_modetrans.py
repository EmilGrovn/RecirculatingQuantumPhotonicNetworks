#rqpnn_stategeneration_modetrans.py
import dynamiqs as dq
import jax.numpy as jnp
import math

#generate states using excitation numbers and basis - VERIFIED using unit tests for all cases in thesis
def _generate_product_number_state_using_basis(excitation_numbers,basis_conserved_mat):
    ''' 
    Input is a list of excitation numbers for subspace.
    Output is a state vector in dynamiqs.
    '''
    if type(excitation_numbers) is list:
        excitation_numbers=jnp.asarray(excitation_numbers)
        print("excitation_numbers was list")
    bool_vec=jnp.all(excitation_numbers==basis_conserved_mat,axis=1)
    hilbertdim_conserved=basis_conserved_mat.shape[0]
    state=jnp.zeros((hilbertdim_conserved,1))
    if not all(excitation_number>=0 for excitation_number in excitation_numbers):
        return state
    if jnp.sum(bool_vec)>1:
        raise ValueError("There should be only one state for the given excitation numbers")
    if jnp.sum(bool_vec)==0:
        raise ValueError("No state found for the given excitation numbers")
    state=state.at[bool_vec].set(1.0) #set state to 1 for the given excitation numbers
    return dq.asqarray(state.astype('complex64')).asdense()
#use raising operators - VERIFIED using unit tests for all cases in thesis
def _generate_product_state_using_raising_operators(exc_idx_list,exc_number_list,raising_op_list, vacuum_state,basis_switch_matrix):
    '''
    input:
    exc_idx_list: list of indices for the excitation numbers M+1.
    exc_number_list: list of excitation numbers.
    raising_op_list: list of raising operators for each mode. (including identity in zeroth position)
    vacuum_state: vacuum state vector (jax.array hilbertdim_conserved x 1)
    basis_switch_matrix: switch matrix to go from restricted basis to conserved basis (jax.array hilbertdim_conserved x hilbertdim_restricted)
    '''
    init_state = vacuum_state
    for idx in exc_idx_list:
        raising_op = raising_op_list[idx] #idx starts from 1, so we need to subtract 1
        init_state = raising_op @ init_state
    init_state= basis_switch_matrix @ init_state #transform to conserved basis
    factorial = 1.0
    for n in exc_number_list:
        #print("n in exc_number_list:", n,type(n))
        factorial *= 1/math.sqrt(math.factorial(n))
    return init_state * factorial
###helper: generate exc_idx_basis - VERIFIED by unit tests for all cases in thesis
def _generate_exc_idx_basis(basis_conserved_mat,N_exc_max):
    '''
    Generate a list of indices for the excitation numbers.
    E.g. for excitation_numbers=[2,1,0,3] and N_exc_max=7 -> exc_idx_list=[1,1,2,4,4,4,0]
    '''
    exc_idx_basis=[]
    for row in range(basis_conserved_mat.shape[0]):
        exc_number_list=basis_conserved_mat[row]
        exc_idx_list=[]
        sum=0
        for m in range(1,len(exc_number_list)+1):
            exc_number=int(exc_number_list[m-1])
            sum+=exc_number
            for n in range(exc_number):
                exc_idx_list.append(m)
        for n in range(N_exc_max-sum):
            exc_idx_list.append(0)
        exc_idx_basis.append(exc_idx_list)
    return exc_idx_basis

# transform raising operators
def _transform_raising_operators(spec,raising_op_list,S_in):
    '''
    Transform the raising operators to the conserved basis using the switch matrix.
    input:
    spec: RQPNNSpec object
    raising_op_list: list of raising operators for each mode.
    S_in: scattering matrix (jax array M_cav x M_cav)
    '''
    assert len(raising_op_list)==spec.M_cav+spec.M_em, "Number of raising operators does not match number of cavities"
    assert S_in.shape[0]==spec.M_cav, "Switch matrix does not match number of cavities"
    assert S_in.shape[1]==spec.M_cav
    raising_op_list_transformed = []
    for idx in range(spec.M_cav):
        raising_op_idx=S_in[0,idx]*raising_op_list[0]
        for jdx in range(1,spec.M_cav):
            raising_op_idx+=S_in[jdx,idx]*raising_op_list[jdx]
        raising_op_list_transformed.append(raising_op_idx)
    #append untransformed emitter raising operators
    for idx in range(spec.M_cav,spec.M_cav+spec.M_em):
        raising_op_list_transformed.append(raising_op_list[idx])
    return raising_op_list_transformed

#build modeswitch matrix
def _build_mode_switch_matrix(spec,raising_op_list,S_in,basis_conserved_mat,exc_idx_basis,vacuum_state,basis_switch_matrix):
    # Ensure complex dtypes everywhere you combine values
    S_in = S_in.astype(jnp.complex64)
    vacuum_state = vacuum_state.astype(jnp.complex64)
    basis_switch_matrix = basis_switch_matrix.astype(jnp.complex64)
    raising_op_list = [op.astype(jnp.complex64) for op in raising_op_list]    
    #transform modes
    raising_op_list_transformed=_transform_raising_operators(spec,raising_op_list,S_in)
    raising_op_list_transformed.insert(0,jnp.eye(raising_op_list_transformed[0].shape[0], dtype=jnp.complex64))
    #   
    Unitary_transform_mat=jnp.zeros((basis_conserved_mat.shape[0],basis_conserved_mat.shape[0]),dtype=jnp.complex64)
    for idx in range(basis_conserved_mat.shape[0]):
        excitation_numbers=basis_conserved_mat[idx]
        exc_idx_list=exc_idx_basis[idx]
        state=_generate_product_state_using_raising_operators(exc_idx_list,excitation_numbers,raising_op_list_transformed, vacuum_state,basis_switch_matrix)        
        Unitary_transform_mat=Unitary_transform_mat.at[:,idx].set(dq.to_jax(state).flatten())
    return dq.asqarray(Unitary_transform_mat.astype('complex64'))