#rqpnn_system.py
from functools import cached_property
from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_basis import _generate_basises_and_switch_matrix
from rqpnn.rqpnn_raising_ops import _generate_raising_operators_for_restricted_basis
from rqpnn.rqpnn_ham_ops import _generate_hamiltonian_operators
from rqpnn.rqpnn_stategeneration_modetrans import _generate_exc_idx_basis, _build_mode_switch_matrix
from rqpnn.rqpnn_hamiltonian import build_H_func

import jax.numpy as jnp
import dynamiqs as dq

class RQPNNSystem:
    def __init__(self, spec: RQPNNSpec):
        #immutable specs
        self.spec = spec

        #basis related attributes
        (self.basis_conserved_mat,self.basis_restricted_mat,
         self.switch_matrix_rest_to_cons,self.hilbertdim_restricted
                                                            )=self._generate_basis
        self.hilbertdim_conserved=self.basis_conserved_mat.shape[0] #dimension of the conserved basis

        #raising operators
        self.raising_op_list=self._generate_raise_ops

        #hamiltonian operators
        self.dict_ham_ops=self._generate_ham_ops_dict

        #vacuum state in restricted basis
        self.vacuum_restricted_jax=jnp.zeros((self.hilbertdim_restricted,1)).at[0].set(1) #vacuum state in jax array

        #excitation index basis
        self.exc_idx_basis=self._generate_exc_idx_basis

    ### methods when instantiating the class ###
    #generate basis (ONLY used for generating raising operators)
    @cached_property
    def _generate_basis(self):
        basis_conserved_mat,basis_restricted_mat,switch_matrix_rest_to_cons,hilbertdim_restricted = _generate_basises_and_switch_matrix(
            self.spec.M_cav, self.spec.M_em, self.spec.N_exc_list
        )
        return basis_conserved_mat,basis_restricted_mat,switch_matrix_rest_to_cons,hilbertdim_restricted

    #generate raising operators (these now reveal the basis!)
    @cached_property
    def _generate_raise_ops(self):
        return _generate_raising_operators_for_restricted_basis(
            self.spec.M_cav, self.spec.M_em, self.basis_restricted_mat,
            self.hilbertdim_restricted)

    #generate hamiltonian operators (from raising operators)
    @cached_property
    def _generate_ham_ops_dict(self):
        return _generate_hamiltonian_operators(self.spec,self.raising_op_list,self.switch_matrix_rest_to_cons)
    
    #generate excitation index basis
    @cached_property
    def _generate_exc_idx_basis(self):
        return _generate_exc_idx_basis(self.basis_conserved_mat,max(self.spec.N_exc_list))
    ### end methods when instantiating the class ###

    ### methods that define transformations of the RQPNN system ###
    #generate mode switch matrix
    def build_mode_switch_unitary(self,S_in):
        '''
        Build the mode switch unitary matrix for the given input switch matrix S_in.
        S_in: jax array of shape (M_cav,M_cav)
        returns: dynamiqs array of shape (hilbertdim_conserved,hilbertdim_conserved)
        '''
        return _build_mode_switch_matrix(self.spec,self.raising_op_list,S_in,self.basis_conserved_mat,self.exc_idx_basis,self.vacuum_restricted_jax,self.switch_matrix_rest_to_cons)

    #generate hamiltonian for RQPNN time evolution
    def _build_H_func(self,N_pieces,params_amp_dict):
        '''
        Build the function that returns the Hamiltonian for the cavity network, given the optimizable parameters as input
        N_pieces: number of pieces in piecewise constant Hamiltonian (int)
        params_amp_dict: dictionary containing amplitude parameters (dict)
        returns: function that takes in a dictionary of trainable parameters and returns a dynamiqs TimeQArray
        '''
        return build_H_func(self.spec,self.dict_ham_ops,N_pieces,params_amp_dict)