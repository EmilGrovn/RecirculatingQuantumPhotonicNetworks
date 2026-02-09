#test each function of matrix parameterization

from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_parameterization import build_coefficient_matrix,S_in_sym_from_C, generate_R, S_in_general_from_C
import jax.numpy as jnp

def test_build_coefficient_matrix():
    spec = RQPNNSpec(2, 0, (1,))
    coeff_vec = jnp.array([1])
    C = build_coefficient_matrix(spec, coeff_vec)
    expected_C = jnp.array([[0, 1],
                            [1, 0]])
    assert jnp.array_equal(C, expected_C), f"Expected {expected_C}, but got {C}"
    spec = RQPNNSpec(4, 0, (1,))
    coeff_vec = jnp.array([1, 2, 3, 4, 5, 6])#length 6 = 4*3/2
    C = build_coefficient_matrix(spec, coeff_vec)
    expected_C = jnp.array([[0, 1, 2, 3],
                            [1, 0, 4, 5],
                            [2, 4, 0, 6],
                            [3, 5, 6, 0]])
    assert jnp.array_equal(C, expected_C), f"Expected {expected_C}, but got {C}"

def test_derive_S_in_sym():
    spec = RQPNNSpec(4, 0, (1,))
    coeff_vec = jnp.array([1, 2, 3, 4, 5, 6])#length 6 = 4*3/2
    C = build_coefficient_matrix(spec, coeff_vec)
    S_in_sym=S_in_sym_from_C(C)
    print("S_in_sym",S_in_sym)
    assert jnp.allclose(S_in_sym, S_in_sym.T), "S_in_sym is not symmetric"
    print(jnp.imag(S_in_sym@S_in_sym.T@jnp.linalg.inv(jnp.eye(S_in_sym.shape[0])-S_in_sym@S_in_sym.T)))
    C_conventional_reconstruct=jnp.imag(S_in_sym@S_in_sym.T@jnp.linalg.inv(jnp.eye(S_in_sym.shape[0])-S_in_sym@S_in_sym.T))
    C_other_reconstruct=-0.5j*jnp.linalg.solve(jnp.eye(S_in_sym.shape[0])-S_in_sym@S_in_sym.T,jnp.eye(S_in_sym.shape[0])+S_in_sym@S_in_sym.T)
    assert jnp.allclose(C_conventional_reconstruct, C,atol=1e-5), "S_in_sym is not consistent with C"
    print(C_other_reconstruct)#needs a bit higher tolerance
    assert jnp.allclose(C_other_reconstruct, C,atol=1e-4), "S_in_sym is not consistent with C"

    #simple case
    spec = RQPNNSpec(2, 0, (1,))
    coeff_vec = jnp.array([0.5])
    C = build_coefficient_matrix(spec, coeff_vec)
    S_in_sym=S_in_sym_from_C(C)
    print("S_in_sym",S_in_sym)#[[0.       +0.7071067j 0.7071067+0.j       ]
                                #[0.7071067+0.j        0.       +0.7071067j]]
                                #nice symmetrical matrix that we know
    assert jnp.allclose(S_in_sym, S_in_sym.T), "S_in_sym is not symmetric"

def test_R_and_S_in_general():
    spec = RQPNNSpec(3, 0, (1,))
    para_mat = jnp.array([0,0,0])#length 6 = 3*2/2*2
    R=generate_R(spec, para_mat)
    assert jnp.allclose(R, jnp.eye(R.shape[0],dtype=R.dtype),atol=1e-6), "R is not identity in the special case"
    spec = RQPNNSpec(4, 0, (1,))
    para_mat = jnp.array([1, 2, 3, 4, 5, 6])#length 6 = 4*3/2
    R=generate_R(spec, para_mat)
    print("R",R)
    print("R@R.T",R@R.T)
    assert jnp.allclose(R@R.T, jnp.eye(R.shape[0]),atol=1e-6), "R is not orthogonal"
    
    coeff_vec = jnp.array([1, 2, 3, 4, 5, 6])#length 6 = 4*3/2
    C = build_coefficient_matrix(spec, coeff_vec)
    S_in_sym=S_in_sym_from_C(C)
    S_in=S_in_general_from_C(spec,C,para_mat)
    print("S_in",S_in)
    assert jnp.allclose(S_in@S_in.T, S_in_sym@S_in_sym.T,atol=1e-6), "S_in general does not reproduce S_round"
    assert jnp.allclose(S_in@S_in.T, jnp.linalg.solve(2j*C+jnp.eye(S_in.shape[0]),2j*C-jnp.eye(S_in.shape[0])),atol=1e-6), "S_in does not reproduce C"

