# rqpnn_raising_ops.py
#necessary imports
import jax.numpy as jnp

#generate raising operators for restricted basis
def _generate_raising_operators_for_restricted_basis(M_cav,M_em,basis_restricted_mat,hilbertdim_restricted):
    '''
    Generate raising operators for the restricted basis.
    Returns a list of raising operators for each cavity and emitter. First all cavities, then all emitters
    list length =M_cav+M_em, element shape (hilbertdim_restricted,hilbertdim_restricted) jnp array
    '''
    #check input
    if not isinstance(M_cav, int) or M_cav <= 0:
        raise ValueError("M_cav must be a positive integer")
    if not isinstance(M_em, int) or M_em < 0:
        raise ValueError("M_em must be a non-negative integer")
    if M_cav!=M_em and M_em!=0:
        raise ValueError("M_em must be either 0 or equal to M_cav")
    #generate raising operators
    raising_op_list=[]
    #instantiate matrix of zeros for the raising operator of each subspace (boson or fermion)
    for i in range(M_cav+M_em):
        raising_op_list.append(jnp.zeros((hilbertdim_restricted,hilbertdim_restricted)))
    #for each basis state, for each cav/em, add an excitation and find the new basis state, if available
    for idx_col in range(hilbertdim_restricted):
        basis_state=basis_restricted_mat[idx_col]
        for idx_op,raise_op in enumerate(raising_op_list):
            basis_state_new=basis_state.copy()
            basis_state_new=basis_state_new.at[idx_op].add(1)
            mask = jnp.all(basis_restricted_mat == basis_state_new, axis=1)
            raising_op_list[idx_op]=raise_op.at[mask,idx_col].set(jnp.sqrt(basis_state_new[idx_op]))
    return raising_op_list