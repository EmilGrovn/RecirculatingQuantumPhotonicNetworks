import numpy as np
import jax.numpy as jnp

from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_system import RQPNNSystem
#test that the conserved and restricted basis are correct for all examples that we consider in the thesis
def test_2Mcav():
    spec = RQPNNSpec(2, 0, (1,))
    sys = RQPNNSystem(spec)
    _check_basis_restricted(sys,sys.basis_restricted_mat.tolist())
    _check_basis_conserved(sys)
def test_2McavMem():
    spec = RQPNNSpec(2, 2, (1,))
    sys = RQPNNSystem(spec)
    _check_basis_restricted(sys,sys.basis_restricted_mat.tolist())
    _check_basis_conserved(sys)    
def test_4Mcav():
    spec = RQPNNSpec(4, 0, (2,))
    sys = RQPNNSystem(spec)
    _check_basis_restricted(sys,sys.basis_restricted_mat.tolist())
    _check_basis_conserved(sys)
def test_4McavMem():
    spec = RQPNNSpec(4, 4, (2,))
    sys = RQPNNSystem(spec)
    _check_basis_restricted(sys,sys.basis_restricted_mat.tolist())
    _check_basis_conserved(sys)    
def test_6Mcav():
    spec = RQPNNSpec(6, 0, (3,))
    sys = RQPNNSystem(spec)
    _check_basis_restricted(sys,sys.basis_restricted_mat.tolist())
    _check_basis_conserved(sys)
def test_3McavRep():
    spec = RQPNNSpec(3, 0, (4,5,))
    sys = RQPNNSystem(spec)
    _check_basis_restricted(sys,sys.basis_restricted_mat.tolist())
def test_2McavMemRep():
    spec = RQPNNSpec(2, 2, (5,6,))
    sys = RQPNNSystem(spec)
    _check_basis_restricted(sys,sys.basis_restricted_mat.tolist())
    _check_basis_conserved(sys)    

def _check_basis_restricted(RQPNNSys,basis_restricted_list):
    '''
    Unit test to check that the excitation numbers in our basis are correct'''
    if RQPNNSys.spec.M_cav==2:
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==1:
            assert basis_restricted_list==[[0,0],[0,1],[1,0]], "excitation numbers are not correct"
            print("1qubit gate dual rail with chi3 encoding can be used safely")
        if RQPNNSys.spec.M_em==RQPNNSys.spec.M_cav and max(RQPNNSys.spec.N_exc_list)==1:
            assert basis_restricted_list==[[0,0,0,0],[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]], "excitation numbers are not correct"
            print("1qubit gate dual rail with emitter encoding can be used safely")
        if RQPNNSys.spec.M_em==RQPNNSys.spec.M_cav and max(RQPNNSys.spec.N_exc_list)==6:
            assert basis_restricted_list==[[0, 0, 0, 0], 
                                            [0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1],
                                            [0,2,0,0],[1,1,0,0],[2,0,0,0],[0,1,1,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,0,1,1],#2 excitations
                                            [0,3,0,0],[1,2,0,0],[2,1,0,0],[3,0,0,0],[0,2,1,0],[0,2,0,1],[1,1,1,0],[1,1,0,1],[2,0,1,0],[2,0,0,1],
                                            [0,1,1,1],[1,0,1,1],#3 excitations
                                            [0,4,0,0],[1,3,0,0],[2,2,0,0],[3,1,0,0],[4,0,0,0],
                                            [0,3,1,0],[0,3,0,1],[1,2,1,0],[1,2,0,1],[2,1,1,0],[2,1,0,1],[3,0,1,0],[3,0,0,1],
                                            [0,2,1,1],[1,1,1,1],[2,0,1,1],#4 excitations
                                            [0,5,0,0],[1,4,0,0],[2,3,0,0],[3,2,0,0],[4,1,0,0],[5,0,0,0],
                                            [0,4,1,0],[0,4,0,1],[1,3,1,0],[1,3,0,1],[2,2,1,0],[2,2,0,1],[3,1,1,0],[3,1,0,1],[4,0,1,0],[4,0,0,1],
                                            [0,3,1,1],[1,2,1,1],[2,1,1,1],[3,0,1,1], #5 excitations
                                            [0,6,0,0],[1,5,0,0],[2,4,0,0],[3,3,0,0],[4,2,0,0],[5,1,0,0],[6,0,0,0],
                                            [0,5,1,0],[0,5,0,1],[1,4,1,0],[1,4,0,1],[2,3,1,0],[2,3,0,1],[3,2,1,0],[3,2,0,1],[4,1,1,0],[4,1,0,1],
                                            [5,0,1,0],[5,0,0,1],
                                            [0,4,1,1],[1,3,1,1],[2,2,1,1],[3,1,1,1],[4,0,1,1],#6 excitations
                                            ], "excitation numbers are not correct"
            print("2mode bosonic protecting 2 mode loss can be used safely")
    if RQPNNSys.spec.M_cav==3:
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==5:
            assert basis_restricted_list==[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], #0 and 1 excitation
                                            [0,0,2],[0,1,1],[0,2,0],[1,0,1],[1,1,0],[2,0,0], #2 excitations
                                            [0,0,3],[0,1,2],[0,2,1],[0,3,0],[1,0,2],[1,1,1],
                                            [1,2,0],[2,0,1],[2,1,0],[3,0,0],#3 excitations
                                            [0,0,4],[0,1,3],[0,2,2],[0,3,1],[0,4,0],[1,0,3],[1,1,2],
                                            [1,2,1],[1,3,0],[2,0,2],[2,1,1],[2,2,0],[3,0,1],[3,1,0],[4,0,0],#4 excitations
                                            [0,0,5],[0,1,4],[0,2,3],[0,3,2],[0,4,1],[0,5,0],[1,0,4],
                                            [1,1,3],[1,2,2],[1,3,1],[1,4,0],[2,0,3],[2,1,2],
                                            [2,2,1],[2,3,0],[3,0,2],[3,1,1],[3,2,0],[4,0,1],[4,1,0],[5,0,0]], "excitation numbers are not correct"
        print("2mode bosonic encoding with chi3 protecting 1mode loss can be used safely")
    if RQPNNSys.spec.M_cav==4:    
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==2:
            assert basis_restricted_list==[[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],
                                            [0,0,0,2],[0,0,1,1],[0,0,2,0],[0,1,0,1],
                                            [0,1,1,0],[0,2,0,0],[1,0,0,1],[1,0,1,0],
                                            [1,1,0,0],[2,0,0,0]], "excitation numbers are not correct"
            print("2qubit gate dual rail with chi3 encoding can be used safely")
        if RQPNNSys.spec.M_em==RQPNNSys.spec.M_cav and max(RQPNNSys.spec.N_exc_list)==2:
            assert basis_restricted_list==[[0, 0, 0, 0, 0, 0, 0, 0], 
                                            [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                                            [0,0,0,2, 0, 0, 0, 0],[0,0,1,1, 0, 0, 0, 0],[0,0,2,0, 0, 0, 0, 0],[0,1,0,1, 0, 0, 0, 0],
                                            [0,1,1,0, 0, 0, 0, 0],[0,2,0,0, 0, 0, 0, 0],[1,0,0,1, 0, 0, 0, 0],[1,0,1,0, 0, 0, 0, 0],
                                            [1,1,0,0, 0, 0, 0, 0],[2,0,0,0, 0, 0, 0, 0],
                                            [0,0,0,1,1,0,0,0],[0,0,0,1,0,1,0,0],[0,0,0,1,0,0,1,0],[0,0,0,1,0,0,0,1],
                                            [0,0,1,0,1,0,0,0],[0,0,1,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,1,0,0,0,0,1],
                                            [0,1,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,0,1,0],[0,1,0,0,0,0,0,1],
                                            [1,0,0,0,1,0,0,0],[1,0,0,0,0,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],
                                            [0,0,0,0,1,1,0,0],[0,0,0,0,1,0,1,0],[0,0,0,0,1,0,0,1],
                                            [0,0,0,0,0,1,1,0],[0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,1]], "excitation numbers are not correct"
            print("2qubit gate dual rail with emitter encoding can be used safely")
    if RQPNNSys.spec.M_cav==6:
        if RQPNNSys.spec.M_em==0 and max(RQPNNSys.spec.N_exc_list)==3:
            assert basis_restricted_list==[[0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0],[0,0,0,1,0,0],
                                            [0,0,1,0,0,0],[0,1,0,0,0,0],[1,0,0,0,0,0],#1 excitation
                                            [0,0,0,0,0,2],[0,0,0,0,1,1],[0,0,0,0,2,0],
                                            [0,0,0,1,0,1],[0,0,0,1,1,0],[0,0,0,2,0,0],
                                            [0,0,1,0,0,1],[0,0,1,0,1,0],[0,0,1,1,0,0],[0,0,2,0,0,0],
                                            [0,1,0,0,0,1],[0,1,0,0,1,0],[0,1,0,1,0,0],[0,1,1,0,0,0],[0,2,0,0,0,0],
                                            [1,0,0,0,0,1],[1,0,0,0,1,0],[1,0,0,1,0,0],[1,0,1,0,0,0],[1,1,0,0,0,0],[2,0,0,0,0,0],#2 excitations
                                            [0,0,0,0,0,3],[0,0,0,0,1,2],[0,0,0,0,2,1],[0,0,0,0,3,0],
                                            [0,0,0,1,0,2],[0,0,0,1,1,1],[0,0,0,1,2,0],[0,0,0,2,0,1],[0,0,0,2,1,0],[0,0,0,3,0,0],
                                            [0,0,1,0,0,2],[0,0,1,0,1,1],[0,0,1,0,2,0],[0,0,1,1,0,1],[0,0,1,1,1,0],[0,0,1,2,0,0],
                                            [0,0,2,0,0,1],[0,0,2,0,1,0],[0,0,2,1,0,0],[0,0,3,0,0,0],
                                            [0,1,0,0,0,2],[0,1,0,0,1,1],[0,1,0,0,2,0],[0,1,0,1,0,1],[0,1,0,1,1,0],
                                            [0,1,0,2,0,0],[0,1,1,0,0,1],[0,1,1,0,1,0],[0,1,1,1,0,0],[0,1,2,0,0,0],[0,2,0,0,0,1],
                                            [0,2,0,0,1,0],[0,2,0,1,0,0],[0,2,1,0,0,0],[0,3,0,0,0,0],
                                            [1,0,0,0,0,2],[1,0,0,0,1,1],[1,0,0,0,2,0],[1,0,0,1,0,1],[1,0,0,1,1,0],
                                            [1,0,0,2,0,0],[1,0,1,0,0,1],[1,0,1,0,1,0],[1,0,1,1,0,0],[1,0,2,0,0,0],
                                            [1,1,0,0,0,1],[1,1,0,0,1,0],[1,1,0,1,0,0],[1,1,1,0,0,0],[1,2,0,0,0,0],
                                            [2,0,0,0,0,1],[2,0,0,0,1,0],[2,0,0,1,0,0],[2,0,1,0,0,0],[2,1,0,0,0,0],[3,0,0,0,0,0]], "excitation numbers are not correct"
            print("3qubit gate dual rail encoding with chi3 can be used safely")
def _check_basis_conserved(RQPNNSys):
    '''
    Unit test to check that the excitation numbers in our basis are correct'''
    hilbertdim_conserved=RQPNNSys.basis_conserved_mat.shape[0]
    print("hilbert space dim conserved",hilbertdim_conserved)
    print("basis conserved",RQPNNSys.basis_conserved_mat)
    #print("basis restricted taken down",self.basis_restricted_mat[-self.hilbertdim_conserved:,:])
    np.testing.assert_allclose(RQPNNSys.basis_conserved_mat,RQPNNSys.basis_restricted_mat[-hilbertdim_conserved:,:], rtol=1e-3, atol=1e-5), "basis conserved does not match basis restricted"
    print("basis conserved is correct")