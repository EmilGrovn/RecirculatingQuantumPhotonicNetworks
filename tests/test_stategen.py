import numpy as np
import jax.numpy as jnp

from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_system import RQPNNSystem
from rqpnn.rqpnn_stategeneration_modetrans import _generate_exc_idx_basis, _generate_product_state_using_raising_operators, _generate_product_number_state_using_basis
#test that the conserved and restricted basis are correct for all examples that we consider in the thesis
def test_2Mcav():
    spec = RQPNNSpec(2, 0, (1,))
    sys = RQPNNSystem(spec)
    _check_exc_idx_basis(sys)
    _check_state_generation_using_raising_ops_2Mcav(sys)
def test_2McavMem():
    spec = RQPNNSpec(2, 2, (1,))
    sys = RQPNNSystem(spec)
    _check_exc_idx_basis(sys)   
    _check_state_generation_using_raising_ops_2Mcav(sys)
def test_4Mcav():
    spec = RQPNNSpec(4, 0, (2,))
    sys = RQPNNSystem(spec)
    _check_exc_idx_basis(sys)
    _check_state_generation_using_raising_ops_2Mcav(sys)
def test_4McavMem():
    spec = RQPNNSpec(4, 4, (2,))
    sys = RQPNNSystem(spec)
    _check_exc_idx_basis(sys)   
    _check_state_generation_using_raising_ops_2Mcav(sys)
def test_6Mcav():
    spec = RQPNNSpec(6, 0, (3,))
    sys = RQPNNSystem(spec)
    _check_exc_idx_basis(sys)
    _check_state_generation_using_raising_ops_2Mcav(sys)
def test_3McavRep():
    spec = RQPNNSpec(3, 0, (4,5,))
    sys = RQPNNSystem(spec)
    _check_exc_idx_basis(sys)
    _check_state_generation_using_raising_ops_2Mcav(sys)
def test_2McavMemRep():
    spec = RQPNNSpec(2, 2, (5,6,))
    sys = RQPNNSystem(spec)
    _check_exc_idx_basis(sys)
    _check_state_generation_using_raising_ops_2Mcav(sys)

def _check_exc_idx_basis(RQPNNSys):
    '''
    Unit test to check that the excitation numbers in our basis are correct'''
    exc_idx_basis=_generate_exc_idx_basis(RQPNNSys.basis_conserved_mat,max(RQPNNSys.spec.N_exc_list))
    if RQPNNSys.spec.M_cav==2:
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==1:
            assert exc_idx_basis==[[2],[1]], "excitation numbers are not correct"
            print("1qubit gate dual rail with chi3 encoding can be used safely")
        if RQPNNSys.spec.M_em==RQPNNSys.spec.M_cav and max(RQPNNSys.spec.N_exc_list)==1:
            assert exc_idx_basis==[[2],[1],[3],[4]], "excitation numbers are not correct"
            print("1qubit gate dual rail with emitter encoding can be used safely")
        if RQPNNSys.spec.M_em==RQPNNSys.spec.M_cav and max(RQPNNSys.spec.N_exc_list)==6:
            assert exc_idx_basis==[
                                            [2,2,2,2,2,0],[1,2,2,2,2,0],[1,1,2,2,2,0],[1,1,1,2,2,0],[1,1,1,1,2,0],[1,1,1,1,1,0],
                                            [2,2,2,2,3,0],[2,2,2,2,4,0],[1,2,2,2,3,0],[1,2,2,2,4,0],[1,1,2,2,3,0],[1,1,2,2,4,0],[1,1,1,2,3,0],[1,1,1,2,4,0],[1,1,1,1,3,0],[1,1,1,1,4,0],
                                            [2,2,2,3,4,0],[1,2,2,3,4,0],[1,1,2,3,4,0],[1,1,1,3,4,0], #5 excitations
                                            [2,2,2,2,2,2],[1,2,2,2,2,2],[1,1,2,2,2,2],[1,1,1,2,2,2],[1,1,1,1,2,2],[1,1,1,1,1,2],[1,1,1,1,1,1],
                                            [2,2,2,2,2,3],[2,2,2,2,2,4],[1,2,2,2,2,3],[1,2,2,2,2,4],[1,1,2,2,2,3],[1,1,2,2,2,4],[1,1,1,2,2,3],[1,1,1,2,2,4],[1,1,1,1,2,3],[1,1,1,1,2,4],
                                            [1,1,1,1,1,3],[1,1,1,1,1,4],
                                            [2,2,2,2,3,4],[1,2,2,2,3,4],[1,1,2,2,3,4],[1,1,1,2,3,4],[1,1,1,1,3,4],#6 excitations
                                            ], "excitation numbers are not correct"
            print("2mode bosonic protecting 2 mode loss can be used safely")
    if RQPNNSys.spec.M_cav==3:
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==5:
            assert exc_idx_basis==[[3,3,3,3,0],[2,3,3,3,0],[2,2,3,3,0],[2,2,2,3,0],[2,2,2,2,0],[1,3,3,3,0],[1,2,3,3,0],
                                            [1,2,2,3,0],[1,2,2,2,0],[1,1,3,3,0],[1,1,2,3,0],[1,1,2,2,0],[1,1,1,3,0],[1,1,1,2,0],[1,1,1,1,0],#4 excitations
                                            [3,3,3,3,3],[2,3,3,3,3],[2,2,3,3,3],[2,2,2,3,3],[2,2,2,2,3],[2,2,2,2,2],[1,3,3,3,3],
                                            [1,2,3,3,3],[1,2,2,3,3],[1,2,2,2,3],[1,2,2,2,2],[1,1,3,3,3],[1,1,2,3,3],
                                            [1,1,2,2,3],[1,1,2,2,2],[1,1,1,3,3],[1,1,1,2,3],[1,1,1,2,2],[1,1,1,1,3],[1,1,1,1,2],[1,1,1,1,1]], "excitation numbers are not correct"
        print("2mode bosonic encoding with chi3 protecting 1mode loss can be used safely")
    if RQPNNSys.spec.M_cav==4:    
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==2:
            assert exc_idx_basis==[
                                            [4,4],[3,4],[3,3],[2,4],
                                            [2,3],[2,2],[1,4],[1,3],
                                            [1,2],[1,1]], "excitation numbers are not correct"
            print("2qubit gate dual rail with chi3 encoding can be used safely")
        if RQPNNSys.spec.M_em==RQPNNSys.spec.M_cav and max(RQPNNSys.spec.N_exc_list)==2:     
            assert exc_idx_basis==[[4,4],[3,4],[3,3],[2,4],
                                            [2,3],[2,2],[1,4],[1,3],
                                            [1,2],[1,1],
                                            [4,5],[4,6],[4,7],[4,8],
                                            [3,5],[3,6],[3,7],[3,8],
                                            [2,5],[2,6],[2,7],[2,8],
                                            [1,5],[1,6],[1,7],[1,8],
                                            [5,6],[5,7],[5,8],
                                            [6,7],[6,8],[7,8]], "excitation numbers are not correct"
            print("2qubit gate dual rail with emitter encoding can be used safely")
    if RQPNNSys.spec.M_cav==6:
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==3:
            assert exc_idx_basis==[[6,6,6],[5,6,6],[5,5,6],[5,5,5],
                                            [4,6,6],[4,5,6],[4,5,5],[4,4,6],[4,4,5],[4,4,4],
                                            [3,6,6],[3,5,6],[3,5,5],[3,4,6],[3,4,5],[3,4,4],
                                            [3,3,6],[3,3,5],[3,3,4],[3,3,3],
                                            [2,6,6],[2,5,6],[2,5,5],[2,4,6],[2,4,5],
                                            [2,4,4],[2,3,6],[2,3,5],[2,3,4],[2,3,3],[2,2,6],
                                            [2,2,5],[2,2,4],[2,2,3],[2,2,2],
                                            [1,6,6],[1,5,6],[1,5,5],[1,4,6],[1,4,5],
                                            [1,4,4],[1,3,6],[1,3,5],[1,3,4],[1,3,3],
                                            [1,2,6],[1,2,5],[1,2,4],[1,2,3],[1,2,2],
                                            [1,1,6],[1,1,5],[1,1,4],[1,1,3],[1,1,2],[1,1,1]], "excitation numbers are not correct"
            print("3qubit gate dual rail encoding with chi3 can be used safely")    
    return 

#check state generation using raising operators
def _check_state_generation_using_raising_ops_2Mcav(RQPNNSys):
    exc_idx_basis=_generate_exc_idx_basis(RQPNNSys.basis_conserved_mat,max(RQPNNSys.spec.N_exc_list))
    raising_op_list0=RQPNNSys.raising_op_list
    raising_op_list0.insert(0,jnp.eye(raising_op_list0[0].shape[0])) #append identity operator for vacuum
    vacuum_state=RQPNNSys.vacuum_restricted_jax
    basis_switch_matrix=RQPNNSys.switch_matrix_rest_to_cons
    for idx in range(RQPNNSys.basis_conserved_mat.shape[0]):
        excitation_numbers=RQPNNSys.basis_conserved_mat[idx]
        exc_idx_list=exc_idx_basis[idx]
        state_using_basis=_generate_product_number_state_using_basis(excitation_numbers,RQPNNSys.basis_conserved_mat)
        state_using_basis_jax=state_using_basis.to_jax()
        state_using_raising_ops=_generate_product_state_using_raising_operators(exc_idx_list,excitation_numbers,raising_op_list0, vacuum_state,basis_switch_matrix)
        assert jnp.allclose(state_using_basis_jax,state_using_raising_ops), "State generation using raising operators is not correct"