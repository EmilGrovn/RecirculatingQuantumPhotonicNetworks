from qip_task.qiptask_init_target_states import _generate_initial_and_target_states
from rqpnn.rqpnn_stategeneration_modetrans import _generate_product_number_state_using_basis
from rqpnn.rqpnn_system import RQPNNSystem
from rqpnn.rqpnn_spec import RQPNNSpec
import jax.numpy as jnp
import dynamiqs as dq
#test all the different initial and target state generations
#1qubit dual rail
def test_hadamard_gate_1qubit_cav():
    spec = RQPNNSpec(2, 0, (1,))
    sys = RQPNNSystem(spec)
    transformation_type="Hadamard gate 1qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi10=_generate_product_number_state_using_basis([1,0],sys.basis_conserved_mat)
    psi01=_generate_product_number_state_using_basis([0,1],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi10))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi01))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax((psi10+psi01)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax((psi10-psi01)/jnp.sqrt(2)))

def test_hadamard_gate_1qubit_em():
    spec = RQPNNSpec(2, 2, (1,))
    sys = RQPNNSystem(spec)
    transformation_type="Hadamard gate 1qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi10=_generate_product_number_state_using_basis([1,0,0,0],sys.basis_conserved_mat)
    psi01=_generate_product_number_state_using_basis([0,1,0,0],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi10))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi01))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax((psi10+psi01)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax((psi10-psi01)/jnp.sqrt(2)))

def test_p4_gate_1qubit_cav():
    spec = RQPNNSpec(2, 0, (1,))
    sys = RQPNNSystem(spec)
    transformation_type="Pi/4-Phase gate 1qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi10=_generate_product_number_state_using_basis([1,0],sys.basis_conserved_mat)
    psi01=_generate_product_number_state_using_basis([0,1],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi10))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi01))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi10))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi01*jnp.exp(1j*jnp.pi/4)))

def test_p4_gate_1qubit_em():
    spec = RQPNNSpec(2, 2, (1,))
    sys = RQPNNSystem(spec)
    transformation_type="Pi/4-Phase gate 1qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi10=_generate_product_number_state_using_basis([1,0,0,0],sys.basis_conserved_mat)
    psi01=_generate_product_number_state_using_basis([0,1,0,0],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi10))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi01))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi10))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi01*jnp.exp(1j*jnp.pi/4)))
#2qubit dual rail
def test_CZ_gate_2qubit_cav():
    spec = RQPNNSpec(4, 0, (2,))
    sys = RQPNNSystem(spec)
    transformation_type="CZ gate 2qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi1010=_generate_product_number_state_using_basis([1,0,1,0],sys.basis_conserved_mat)
    psi0110=_generate_product_number_state_using_basis([0,1,1,0],sys.basis_conserved_mat)
    psi1001=_generate_product_number_state_using_basis([1,0,0,1],sys.basis_conserved_mat)
    psi0101=_generate_product_number_state_using_basis([0,1,0,1],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(initial_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(initial_states[3]), dq.to_jax(psi0101))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(target_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(target_states[3]), dq.to_jax(-psi0101))

def test_CZ_gate_2qubit_em():
    spec = RQPNNSpec(4, 4, (2,))
    sys = RQPNNSystem(spec)
    transformation_type="CZ gate 2qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi1010=_generate_product_number_state_using_basis([1,0,1,0,0,0,0,0],sys.basis_conserved_mat)
    psi0110=_generate_product_number_state_using_basis([0,1,1,0,0,0,0,0],sys.basis_conserved_mat)
    psi1001=_generate_product_number_state_using_basis([1,0,0,1,0,0,0,0],sys.basis_conserved_mat)
    psi0101=_generate_product_number_state_using_basis([0,1,0,1,0,0,0,0],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(initial_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(initial_states[3]), dq.to_jax(psi0101))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(target_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(target_states[3]), dq.to_jax(-psi0101))

def test_CNot_gate_2qubit_cav():
    spec = RQPNNSpec(4, 0, (2,))
    sys = RQPNNSystem(spec)
    transformation_type="CNot gate 2qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi1010=_generate_product_number_state_using_basis([1,0,1,0],sys.basis_conserved_mat)
    psi0110=_generate_product_number_state_using_basis([0,1,1,0],sys.basis_conserved_mat)
    psi1001=_generate_product_number_state_using_basis([1,0,0,1],sys.basis_conserved_mat)
    psi0101=_generate_product_number_state_using_basis([0,1,0,1],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(initial_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(initial_states[3]), dq.to_jax(psi0101))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(target_states[2]), dq.to_jax(psi0101))
    assert jnp.allclose(dq.to_jax(target_states[3]), dq.to_jax(psi0110))

def test_CNot_gate_2qubit_em():
    spec = RQPNNSpec(4, 4, (2,))
    sys = RQPNNSystem(spec)
    transformation_type="CNot gate 2qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi1010=_generate_product_number_state_using_basis([1,0,1,0,0,0,0,0],sys.basis_conserved_mat)
    psi0110=_generate_product_number_state_using_basis([0,1,1,0,0,0,0,0],sys.basis_conserved_mat)
    psi1001=_generate_product_number_state_using_basis([1,0,0,1,0,0,0,0],sys.basis_conserved_mat)
    psi0101=_generate_product_number_state_using_basis([0,1,0,1,0,0,0,0],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(initial_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(initial_states[3]), dq.to_jax(psi0101))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(target_states[2]), dq.to_jax(psi0101))
    assert jnp.allclose(dq.to_jax(target_states[3]), dq.to_jax(psi0110))

def test_BSA_2qubit_cav():
    spec = RQPNNSpec(4, 0, (2,))
    sys = RQPNNSystem(spec)
    transformation_type="BSA 2qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi1010=_generate_product_number_state_using_basis([1,0,1,0],sys.basis_conserved_mat)
    psi0110=_generate_product_number_state_using_basis([0,1,1,0],sys.basis_conserved_mat)
    psi1001=_generate_product_number_state_using_basis([1,0,0,1],sys.basis_conserved_mat)
    psi0101=_generate_product_number_state_using_basis([0,1,0,1],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax((psi1010+psi0101)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax((psi1001+psi0110)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(initial_states[2]), dq.to_jax((psi1010-psi0101)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(initial_states[3]), dq.to_jax((psi1001-psi0110)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(target_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(target_states[3]), dq.to_jax(psi0101))

def test_BSA_2qubit_em():
    spec = RQPNNSpec(4, 4, (2,))
    sys = RQPNNSystem(spec)
    transformation_type="BSA 2qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi1010=_generate_product_number_state_using_basis([1,0,1,0,0,0,0,0],sys.basis_conserved_mat)
    psi0110=_generate_product_number_state_using_basis([0,1,1,0,0,0,0,0],sys.basis_conserved_mat)
    psi1001=_generate_product_number_state_using_basis([1,0,0,1,0,0,0,0],sys.basis_conserved_mat)
    psi0101=_generate_product_number_state_using_basis([0,1,0,1,0,0,0,0],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax((psi1010+psi0101)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax((psi1001+psi0110)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(initial_states[2]), dq.to_jax((psi1010-psi0101)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(initial_states[3]), dq.to_jax((psi1001-psi0110)/jnp.sqrt(2)))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi1010))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi1001))
    assert jnp.allclose(dq.to_jax(target_states[2]), dq.to_jax(psi0110))
    assert jnp.allclose(dq.to_jax(target_states[3]), dq.to_jax(psi0101))

def test_Toffoli_gate_3qubit_cav():
    spec = RQPNNSpec(6, 0, (3,))
    sys = RQPNNSystem(spec)
    transformation_type="Tofolli gate 3qubit"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    psi101010=_generate_product_number_state_using_basis([1,0,1,0,1,0],sys.basis_conserved_mat)
    psi101001=_generate_product_number_state_using_basis([1,0,1,0,0,1],sys.basis_conserved_mat)
    psi011010=_generate_product_number_state_using_basis([0,1,1,0,1,0],sys.basis_conserved_mat)
    psi011001=_generate_product_number_state_using_basis([0,1,1,0,0,1],sys.basis_conserved_mat)
    psi100110=_generate_product_number_state_using_basis([1,0,0,1,1,0],sys.basis_conserved_mat)
    psi100101=_generate_product_number_state_using_basis([1,0,0,1,0,1],sys.basis_conserved_mat)
    psi010110=_generate_product_number_state_using_basis([0,1,0,1,1,0],sys.basis_conserved_mat)
    psi010101=_generate_product_number_state_using_basis([0,1,0,1,0,1],sys.basis_conserved_mat)
    assert jnp.allclose(dq.to_jax(initial_states[0]), dq.to_jax(psi101010))
    assert jnp.allclose(dq.to_jax(initial_states[1]), dq.to_jax(psi101001))
    assert jnp.allclose(dq.to_jax(initial_states[2]), dq.to_jax(psi100110))
    assert jnp.allclose(dq.to_jax(initial_states[3]), dq.to_jax(psi100101))
    assert jnp.allclose(dq.to_jax(initial_states[4]), dq.to_jax(psi011010))
    assert jnp.allclose(dq.to_jax(initial_states[5]), dq.to_jax(psi011001))
    assert jnp.allclose(dq.to_jax(initial_states[6]), dq.to_jax(psi010110))
    assert jnp.allclose(dq.to_jax(initial_states[7]), dq.to_jax(psi010101))
    assert jnp.allclose(dq.to_jax(target_states[0]), dq.to_jax(psi101010))
    assert jnp.allclose(dq.to_jax(target_states[1]), dq.to_jax(psi101001))
    assert jnp.allclose(dq.to_jax(target_states[2]), dq.to_jax(psi100110))
    assert jnp.allclose(dq.to_jax(target_states[3]), dq.to_jax(psi100101))
    assert jnp.allclose(dq.to_jax(target_states[4]), dq.to_jax(psi011010))
    assert jnp.allclose(dq.to_jax(target_states[5]), dq.to_jax(psi011001))
    assert jnp.allclose(dq.to_jax(target_states[6]), dq.to_jax(psi010101))
    assert jnp.allclose(dq.to_jax(target_states[7]), dq.to_jax(psi010110))

#repeaters - der er ikke så meget at teste. Skal bare have indtastet de rigtige exc numbers i koden
def test_repeater_1loss_cav():
    spec = RQPNNSpec(3, 0, (4,5))
    sys = RQPNNSystem(spec)
    transformation_type="1loss 3cav repeater"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    for i in range(len(initial_states)):
        print(dq.to_jax(initial_states[i].dag()@initial_states[i]))
        print(dq.to_jax(target_states[i].dag()@target_states[i]))
        assert jnp.allclose(dq.to_jax(initial_states[i].dag()@initial_states[i]),1.0,atol=1e-6)
        assert jnp.allclose(dq.to_jax(target_states[i].dag()@target_states[i]),1.0,atol=1e-6)
    #assert False
def test_repeater_2loss_em():
    spec = RQPNNSpec(2, 2, (5,6))
    sys = RQPNNSystem(spec)
    transformation_type="2loss emitter coherent repeater"
    initial_states,target_states= _generate_initial_and_target_states(spec,transformation_type,sys.basis_conserved_mat)
    print("initial_states", initial_states)
    print("target_states", target_states)
    for i in range(len(initial_states)):
        print(dq.to_jax(initial_states[i].dag()@initial_states[i]))
        print(dq.to_jax(target_states[i].dag()@target_states[i]))
        assert jnp.allclose(dq.to_jax(initial_states[i].dag()@initial_states[i]),1.0,atol=1e-6)
        assert jnp.allclose(dq.to_jax(target_states[i].dag()@target_states[i]),1.0,atol=1e-6)
    #assert False