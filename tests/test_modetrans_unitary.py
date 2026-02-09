from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_system import RQPNNSystem
from rqpnn.rqpnn_stategeneration_modetrans import _transform_raising_operators, _generate_product_number_state_using_basis

import jax.numpy as jnp

#transform operators
def test_transform_operators_2cav():
    spec = RQPNNSpec(2, 0, (1,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[0.0,1.0],
                    [1.0,0.0]])
    spec = RQPNNSpec(2, 0, (2,))
    sys = RQPNNSystem(spec)
    sys.raising_op_list
    raising_op_list_transformed=_transform_raising_operators(spec,sys.raising_op_list,S_in)
    print(raising_op_list_transformed[0],type(raising_op_list_transformed[0]))
    print(sys.raising_op_list[1],type(sys.raising_op_list[1]))
    assert jnp.allclose(raising_op_list_transformed[0], sys.raising_op_list[1]), "Raising operators not transformed correctly"
    assert jnp.allclose(raising_op_list_transformed[1], sys.raising_op_list[0]), "Raising operators not transformed correctly"

def test_transform_operators_2cav2em():
    spec = RQPNNSpec(2, 2, (1,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[0.0,1.0],
                    [1.0,0.0]])
    sys.raising_op_list
    raising_op_list_transformed=_transform_raising_operators(spec,sys.raising_op_list,S_in)
    print(raising_op_list_transformed[0],type(raising_op_list_transformed[0]))
    print(sys.raising_op_list[1],type(sys.raising_op_list[1]))
    assert jnp.allclose(raising_op_list_transformed[0], sys.raising_op_list[1]), "Raising operators not transformed correctly"
    assert jnp.allclose(raising_op_list_transformed[1], sys.raising_op_list[0]), "Raising operators not transformed correctly"
    assert jnp.allclose(raising_op_list_transformed[2], sys.raising_op_list[2]), "Raising operators not transformed correctly"
    assert jnp.allclose(raising_op_list_transformed[3], sys.raising_op_list[3]), "Raising operators not transformed correctly"

def test_transform_operators_3cav_circulator():
    spec = RQPNNSpec(3, 0, (1,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[0.0,0.0,1.0],
                    [1.0,0.0,0.0],
                    [0.0,1.0,0.0]])
    sys.raising_op_list
    raising_op_list_transformed=_transform_raising_operators(spec,sys.raising_op_list,S_in)
    assert jnp.allclose(raising_op_list_transformed[0], sys.raising_op_list[1]), "Raising operators not transformed correctly"
    assert jnp.allclose(raising_op_list_transformed[1], sys.raising_op_list[2]), "Raising operators not transformed correctly"
    assert jnp.allclose(raising_op_list_transformed[2], sys.raising_op_list[0]), "Raising operators not transformed correctly"

def test_HongOuMandel_2cav():
    spec = RQPNNSpec(2, 0, (2,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),1/jnp.sqrt(2)],
                    [1/jnp.sqrt(2),-1/jnp.sqrt(2)]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([1,1],sys.basis_conserved_mat)
    psi_expected=1/jnp.sqrt(2)*(_generate_product_number_state_using_basis([2,0],sys.basis_conserved_mat)-_generate_product_number_state_using_basis([0,2],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print("conserved basis",sys.basis_conserved_mat)
    print("psi11",psi11,type(psi11))
    print("U_mat",U_mat,type(U_mat))
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"

def test_HongOuMandel_4cav():
    spec = RQPNNSpec(4, 0, (2,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),0,1/jnp.sqrt(2),0],
                    [0,1,0,0],
                    [1/jnp.sqrt(2),0,-1/jnp.sqrt(2),0],
                    [0,0,0,1]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([1,0,1,0],sys.basis_conserved_mat)
    psi_expected=1/jnp.sqrt(2)*(_generate_product_number_state_using_basis([2,0,0,0],sys.basis_conserved_mat)-_generate_product_number_state_using_basis([0,0,2,0],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"

def test_HongOuMandel_6cav():
    spec = RQPNNSpec(6, 0, (3,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),0,0,1/jnp.sqrt(2),0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0],
                    [1/jnp.sqrt(2),0,0,-1/jnp.sqrt(2),0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([1,1,0,1,0,0],sys.basis_conserved_mat)
    psi_expected=1/jnp.sqrt(2)*(_generate_product_number_state_using_basis([2,1,0,0,0,0],sys.basis_conserved_mat)-_generate_product_number_state_using_basis([0,1,0,2,0,0],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print(psi11)
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"

def test_HongOuMandel_2cav2em():
    spec = RQPNNSpec(2, 2, (2,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),1/jnp.sqrt(2)],
                    [1/jnp.sqrt(2),-1/jnp.sqrt(2)]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([1,1,0,0],sys.basis_conserved_mat)
    psi_expected=1/jnp.sqrt(2)*(_generate_product_number_state_using_basis([2,0,0,0],sys.basis_conserved_mat)-_generate_product_number_state_using_basis([0,2,0,0],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print("conserved basis",sys.basis_conserved_mat)
    print("psi11",psi11,type(psi11))
    print("U_mat",U_mat,type(U_mat))
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"

def test_HongOuMandel_4cav4em():
    spec = RQPNNSpec(4, 4, (4,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1/jnp.sqrt(2),1/jnp.sqrt(2)],
                    [0,0,1/jnp.sqrt(2),-1/jnp.sqrt(2)]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([0,1,1,1,0,0,1,0],sys.basis_conserved_mat)
    psi_expected=1/jnp.sqrt(2)*(_generate_product_number_state_using_basis([0,1,2,0,0,0,1,0],sys.basis_conserved_mat)-_generate_product_number_state_using_basis([0,1,0,2,0,0,1,0],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print("conserved basis",sys.basis_conserved_mat)
    print("psi11",psi11,type(psi11))
    print("U_mat",U_mat,type(U_mat))
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"

#complex-valued S_in
#transformed operators
def test_transform_operators_2cav_complex():
    S_in=jnp.array([[0.0,1.0j],
                    [1.0,0.0]])
    spec = RQPNNSpec(2, 0, (2,))
    sys = RQPNNSystem(spec)
    sys.raising_op_list
    raising_op_list_transformed=_transform_raising_operators(spec,sys.raising_op_list,S_in)
    print(raising_op_list_transformed[0],type(raising_op_list_transformed[0]))
    print(sys.raising_op_list[1],type(sys.raising_op_list[1]))
    assert jnp.allclose(raising_op_list_transformed[0], sys.raising_op_list[1]), "Raising operators not transformed correctly"
    assert jnp.allclose(raising_op_list_transformed[1], 1j*sys.raising_op_list[0]), "Raising operators not transformed correctly"

def test_complexS_in_2cav():
    spec = RQPNNSpec(2, 0, (1,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),1j/jnp.sqrt(2)],
                    [-1j/jnp.sqrt(2),1/jnp.sqrt(2)]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi10=_generate_product_number_state_using_basis([1,0],sys.basis_conserved_mat)
    psi01=_generate_product_number_state_using_basis([0,1],sys.basis_conserved_mat)
    psi_expected=1/jnp.sqrt(2)*(_generate_product_number_state_using_basis([1,0],sys.basis_conserved_mat)-1j*_generate_product_number_state_using_basis([0,1],sys.basis_conserved_mat))
    psi_out=U_mat@psi10
    print("conserved basis",sys.basis_conserved_mat)
    print("psi10",psi10,type(psi10))
    print("psi01",psi01,type(psi01))
    print("U_mat",U_mat,type(U_mat))
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"

#
def test_complexHongOuMandel_2cav():
    spec = RQPNNSpec(2, 0, (2,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),1j/jnp.sqrt(2)],
                    [1j/jnp.sqrt(2),1/jnp.sqrt(2)]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([1,1],sys.basis_conserved_mat)
    psi_expected=1j/jnp.sqrt(2)*(_generate_product_number_state_using_basis([2,0],sys.basis_conserved_mat)+_generate_product_number_state_using_basis([0,2],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print("conserved basis",sys.basis_conserved_mat)
    print("psi11",psi11,type(psi11))
    print("U_mat",U_mat,type(U_mat))
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"

def test_complexHongOuMandel_2cav2em():
    spec = RQPNNSpec(2, 2, (3,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),1j/jnp.sqrt(2)],
                    [1j/jnp.sqrt(2),1/jnp.sqrt(2)]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([1,1,0,1],sys.basis_conserved_mat)
    psi_expected=1j/jnp.sqrt(2)*(_generate_product_number_state_using_basis([2,0,0,1],sys.basis_conserved_mat)+_generate_product_number_state_using_basis([0,2,0,1],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print("conserved basis",sys.basis_conserved_mat)
    print("psi11",psi11,type(psi11))
    print("U_mat",U_mat,type(U_mat))
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"


def test_HongOuMandel_6cav_complex():
    spec = RQPNNSpec(6, 0, (3,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[1/jnp.sqrt(2),0,0,1/jnp.sqrt(2),0,0],
                    [0,1j,0,0,0,0],
                    [0,0,1,0,0,0],
                    [1/jnp.sqrt(2),0,0,-1/jnp.sqrt(2),0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi11=_generate_product_number_state_using_basis([1,1,0,1,0,0],sys.basis_conserved_mat)
    psi_expected=1j/jnp.sqrt(2)*(_generate_product_number_state_using_basis([2,1,0,0,0,0],sys.basis_conserved_mat)-_generate_product_number_state_using_basis([0,1,0,2,0,0],sys.basis_conserved_mat))
    psi_out=U_mat@psi11
    print(psi11)
    print("psi_out",psi_out)
    print("psi expected",psi_expected)
    assert jnp.allclose(psi_out.to_jax(),psi_expected.to_jax(),atol=1e-6), "Hong-Ou-Mandel test failed"


def test_circulator():
    spec = RQPNNSpec(3, 0, (1,))
    sys = RQPNNSystem(spec)
    S_in=jnp.array([[0.0,0.0,1.0],
                    [1.0j,0.0,0.0],
                    [0.0,1.0j,0.0]])
    U_mat=sys.build_mode_switch_unitary(S_in)
    psi100=_generate_product_number_state_using_basis([1,0,0],sys.basis_conserved_mat)
    psi010=_generate_product_number_state_using_basis([0,1,0],sys.basis_conserved_mat)
    psi001=_generate_product_number_state_using_basis([0,0,1],sys.basis_conserved_mat)
    psi_expected1=1j*(_generate_product_number_state_using_basis([0,1,0],sys.basis_conserved_mat))
    psi_expected2=1j*(_generate_product_number_state_using_basis([0,0,1],sys.basis_conserved_mat))
    psi_expected3=1.0*(_generate_product_number_state_using_basis([1,0,0],sys.basis_conserved_mat))
    assert jnp.allclose((U_mat@psi100).to_jax(),psi_expected1.to_jax(),atol=1e-6), "Circulator test failed"
    assert jnp.allclose((U_mat@psi010).to_jax(),psi_expected2.to_jax(),atol=1e-6), "Circulator test failed"
    assert jnp.allclose((U_mat@psi001).to_jax(),psi_expected3.to_jax(),atol=1e-6), "Circulator test failed"