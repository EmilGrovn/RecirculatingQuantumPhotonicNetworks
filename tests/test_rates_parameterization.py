# test_rates_parameterization.py for testing rqpnn_parameterization.py
from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_parameterization import _generate_rates_tau, _generate_rates_delta_cav, _generate_rates_kappa, _generate_rates_delta_em
import jax.numpy as jnp

def test_generate_rates_tau():
    spec = RQPNNSpec(2, 0, (1,))
    params_tau = jnp.array([0.0, -jnp.log(3.0), jnp.log(2.0)])  # Example parameters
    max_tau = 1.0*jnp.ones_like(params_tau)
    min_tau = 0.0*jnp.ones_like(params_tau)
    print("params_tau",params_tau,"shape",params_tau.shape)#shape (3,)
    rates_tau = _generate_rates_tau(params_tau, max_tau, min_tau)
    expected_rates = jnp.array([0.5, 0.25, 2/3.0])
    assert jnp.allclose(rates_tau, expected_rates), f"Expected {expected_rates}, but got {rates_tau}"

    max_tau = 10.0*jnp.ones_like(params_tau)
    min_tau = 2.0*jnp.ones_like(params_tau)
    expected_rates = jnp.array([6.0, 4.0, 2+8*2/3.0])
    rates_tau = _generate_rates_tau(params_tau, max_tau, min_tau)
    assert jnp.allclose(rates_tau, expected_rates), f"Expected {expected_rates}, but got {rates_tau}"

def test_generate_rates_delta_cav():
    spec = RQPNNSpec(4, 0, (1,))
    params_delta = jnp.array([[0.0, 0.0, 0.0],[1.0, 1.0,1.0],
                              [-1.0,-1.0,-1.0],[1/jnp.sqrt(3),-jnp.sqrt(3),1.0]])  # Example parameters
    amps_delta = jnp.array([2.0, 4.0, 6.0])  # Example amplitudes
    print("params_delta",params_delta,"shape",params_delta.shape)#shape (4,3)
    print("amps_delta",amps_delta,"shape",amps_delta.shape)#shape (3,)
    rates_delta = _generate_rates_delta_cav(params_delta, amps_delta)
    expected_rates = jnp.array([[0.0,0.0,0.0],[0.5,1.0,1.5],[-0.5,-1.0,-1.5],[2.0/6.0,-4.0/3,6.0/4.0]])
    assert jnp.allclose(rates_delta, expected_rates), f"Expected {expected_rates}, but got {rates_delta}"

def test_generate_rates_kappa():
    spec = RQPNNSpec(4, 0, (1,))
    params_kappa = jnp.array([[0.0, 0.0, 0.0],[1.0, 1.0,1.0],
                              [-1.0,-1.0,-1.0],[1/jnp.sqrt(3),-jnp.sqrt(3),1.0]])  # Example parameters
    amps_kappa = jnp.array([2.0, 4.0, 6.0])  # Example amplitudes
    print("params_kappa",params_kappa,"shape",params_kappa.shape)#shape (4,3)
    print("amps_kappa",amps_kappa,"shape",amps_kappa.shape)#shape (3,)
    rates_kappa = _generate_rates_kappa(params_kappa, amps_kappa)
    expected_rates = jnp.array([[1.0,2.0,3.0],[1.5,3.0,4.5],[0.5,1.0,1.5],[1.0+2.0/6.0,2.0-4.0/3,3.0+6.0/4.0]])
    assert jnp.allclose(rates_kappa, expected_rates), f"Expected {expected_rates}, but got {rates_kappa}"

def test_generate_rates_delta_em():
    spec = RQPNNSpec(4, 0, (1,))
    params_delta = jnp.array([[0.0, 0.0, 0.0],[1.0, 1.0,1.0],
                              [-1.0,-1.0,-1.0],[1/jnp.sqrt(3),-jnp.sqrt(3),1.0]])  # Example parameters
    amps_delta = jnp.array([2.0, 4.0, 6.0])  # Example amplitudes
    offset_delta=jnp.array([1.0,0.0,2.0])
    print("params_delta",params_delta,"shape",params_delta.shape)#shape (4,3)
    print("amps_delta",amps_delta,"shape",amps_delta.shape)#shape (3,)
    rates_delta = _generate_rates_delta_em(params_delta, amps_delta,offset_delta)
    expected_rates = jnp.array([[1.0,0.0,2.0],[1.5,1.0,3.5],[0.5,-1.0,0.5],[1.0+2.0/6.0,-4.0/3,2.0+6.0/4.0]])
    assert jnp.allclose(rates_delta, expected_rates), f"Expected {expected_rates}, but got {rates_delta}"
