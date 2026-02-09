import numpy as np
import jax.numpy as jnp
import dynamiqs as dq

from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_system import RQPNNSystem
from rqpnn.rqpnn_stategeneration_modetrans import _generate_product_number_state_using_basis

def test_2Mcav():
    spec = RQPNNSpec(2, 0, (1,))
    sys = RQPNNSystem(spec)
    _check_ham_operators(sys)
def test_2McavMem():
    spec = RQPNNSpec(2, 2, (1,))
    sys = RQPNNSystem(spec)
    _check_ham_operators(sys)   
def test_4Mcav():
    spec = RQPNNSpec(4, 0, (2,))
    sys = RQPNNSystem(spec)
    _check_ham_operators(sys)
def test_4McavMem():
    spec = RQPNNSpec(4, 4, (2,))
    sys = RQPNNSystem(spec)
    _check_ham_operators(sys)   
def test_6Mcav():
    spec = RQPNNSpec(6, 0, (3,))
    sys = RQPNNSystem(spec)
    _check_ham_operators(sys)
def test_3McavRep():
    spec = RQPNNSpec(3, 0, (4,5,))
    sys = RQPNNSystem(spec)
    _check_ham_operators(sys)
def test_2McavMemRep():
    spec = RQPNNSpec(2, 2, (5,6,))
    sys = RQPNNSystem(spec)
    _check_ham_operators(sys)  

def _check_ham_operators(RQPNNSys):
    '''
    Unit test to check that the Hamiltonian operators are correct
    '''
    hilbertdim_conserved=RQPNNSys.basis_conserved_mat.shape[0]
    for idx in range(hilbertdim_conserved):
        print("idx",idx)
        excitation_numbers=RQPNNSys.basis_conserved_mat[idx]    #excitation numbers for the given basis state
        state=_generate_product_number_state_using_basis(excitation_numbers,RQPNNSys.basis_conserved_mat)#state corresponding the excitation numbers
        print("state",state)
        for key in RQPNNSys.dict_ham_ops.keys():
            print("op",key)
            op_list=RQPNNSys.dict_ham_ops[key]
            if key=="delta_cav":
                for i,n_op in enumerate(op_list):
                    #print(n_op,n_op@state,state*photon_numbers[i])
                    assert jnp.isclose(dq.to_jax(n_op@state), dq.to_jax(state*excitation_numbers[i]),1e-6).all(), "number operator does not do what it should"
            if key=="chi3":
                for i,n2mn_op in enumerate(op_list):
                    #print(n_op,n_op@state,state*photon_numbers[i])
                    assert jnp.isclose(dq.to_jax(n2mn_op@state), dq.to_jax(state*(excitation_numbers[i]**2-excitation_numbers[i])),1e-6).all(), "n^2-n operator does not do what it should"
            if key=="delta_em":
                for i,em_op in enumerate(op_list):
                    #print(em_op,em_op@state,state*photon_numbers[i])
                    assert jnp.isclose(dq.to_jax(em_op@state), dq.to_jax(state*excitation_numbers[i+len(op_list)]),1e-6).all(), "emitter operator does not do what it should"
            if key=="int":
                for i,ham_op_row in enumerate(op_list):
                    for j,ham_op in enumerate(ham_op_row):
                        #print("state, int@state,",state,ham_op@state,state*excitation_numbers[i])
                        if i==j:
                            assert jnp.isclose(dq.to_jax(ham_op@state), dq.to_jax(state*excitation_numbers[i]),1e-6).all(), "coupling operator does not do what it should"
                        else:
                            photon_numbers_new=excitation_numbers.copy().at[i].add(1).at[j].add(-1)
                            print(excitation_numbers,photon_numbers_new)
                            try:
                                state_new=_generate_product_number_state_using_basis(photon_numbers_new,RQPNNSys.basis_conserved_mat)
                                stateExist=True
                            except ValueError:
                                print("state does not exist in basis, skipping check")
                                stateExist=False
                                continue
                            if stateExist:
                                assert jnp.isclose(dq.to_jax(ham_op@state), dq.to_jax(state_new*jnp.sqrt((excitation_numbers[i]+1)*excitation_numbers[j])),1e-6).all(), "coupling operator does not do what it should"
                            #print(photon_numbers,photon_numbers_new,state,ham_op@state,state_new*jnp.sqrt((photon_numbers[i]+1)*photon_numbers[j]))
            if key=="g_lm":
                for i,em_cav_op in enumerate(op_list):                            
                    #print(em_op,em_op@state,state*photon_numbers[i])
                    photon_numbers_add_photon=excitation_numbers.copy().at[i].add(1).at[i+len(op_list)].add(-1)
                    photon_numbers_add_exciton=excitation_numbers.copy().at[i].add(-1).at[i+len(op_list)].add(1)
                    #print("add photon state",photon_numbers_add_photon)
                    #print("add exciton state",photon_numbers_add_exciton)
                    try:
                        state_new_add_photon=_generate_product_number_state_using_basis(photon_numbers_add_photon,RQPNNSys.basis_conserved_mat)
                        state_new_add_exciton=_generate_product_number_state_using_basis(photon_numbers_add_exciton,RQPNNSys.basis_conserved_mat)
                        stateExist=True
                    except ValueError:
                        print("state does not exist in basis, skipping check")
                        stateExist=False
                        continue
                    #print(photon_numbers,photon_numbers_add_photon,photon_numbers_add_exciton)
                    #print(state,em_cav_op@state,state_new_add_photon*jnp.sqrt(photon_numbers[i]+1)+state_new_add_exciton*jnp.sqrt(photon_numbers[i]),state_new_add_photon,state_new_add_exciton)
                    if stateExist:
                        assert jnp.isclose(dq.to_jax(em_cav_op@state), dq.to_jax(state_new_add_photon*jnp.sqrt(excitation_numbers[i]+1)*jnp.sqrt(excitation_numbers[i+len(op_list)])+state_new_add_exciton*jnp.sqrt(excitation_numbers[i])*jnp.sqrt(1+excitation_numbers[i+len(op_list)])),1e-6).all(), "emitter operator does not do what it should"
    print("Hamiltonian operators perform the correct operation on all basis states!")