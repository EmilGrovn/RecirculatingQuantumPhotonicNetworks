# Test big function parameterization
from rqpnn.rqpnn_spec import RQPNNSpec
import jax.numpy as jnp

spec= RQPNNSpec(4, 4, (1,))

params_train_dict={
    'coeff_vec': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    'S_in_Rmat': jnp.array([0.1,0.2,0.3,0.4,0.6,0.9]), #length 12 = 4*3/2*2
    'tau': jnp.array([0.0, -jnp.log(3.0), jnp.log(2.0)]) ,
    'delta_cav': jnp.array([[-1.0,-1.0,-1.0],[1/jnp.sqrt(3),-jnp.sqrt(3),1.0],
                            [0.0, 0.0, 0.0],[1.0, 1.0,1.0]]),
    'delta_em': jnp.array([[0.0, 0.0, 0.0],[1.0, 1.0,1.0],
                              [-1.0,-1.0,-1.0],[1/jnp.sqrt(3),-jnp.sqrt(3),1.0]]),
    'kappa': jnp.array([[0.0, 0.0, 0.0],[1.0, 1.0,1.0],
                              [-1.0,-1.0,-1.0],[1/jnp.sqrt(3),-jnp.sqrt(3),1.0]])
}
params_amp_dict={
    'max_tau': 10.0*jnp.ones(3),'min_tau': 2.0*jnp.ones(3),
    'delta_cav': jnp.array([1.0, 0.0, 3.0]),
    'delta_em': jnp.array([2.0, 4.0, 6.0]),
    'kappa': jnp.array([2.0, 4.0, 6.0]),
    'delta_em_offset': jnp.array([0.0, 2.0, 0.0])
}
def test_generate_ham_properties_from_params():
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    params_ham_dict=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    #coeff matrix
    assert jnp.allclose(params_ham_dict["coeff_mat"],jnp.array([[0, 1, 2, 3],
                                            [1, 0, 4, 5],
                                            [2, 4, 0, 6],
                                            [3, 5, 6, 0]]))
    #S_in_mat
    S_in=params_ham_dict["S_in_mat"]
    assert jnp.allclose(jnp.imag(S_in@S_in.T@jnp.linalg.inv(jnp.eye(S_in.shape[0])-S_in@S_in.T)), params_ham_dict["coeff_mat"],atol=1e-4), "S_in is not consistent with C" 
    assert jnp.allclose(S_in@(jnp.conjugate(S_in).T), jnp.eye(spec.M_cav), atol=1e-6), "S_in matrix is not unitary"
    #tau
    assert jnp.allclose(params_ham_dict['tau'], jnp.array([6.0, 4.0, 2+8*2/3.0]))
    #delta_cav
    expected_rates = jnp.array([[-0.25,0.0,-0.75],[2.0/6.0/2,0.0,6.0/4.0/2],[0.0,0.0,0.0],[0.25,0.0,0.75]])
    assert jnp.allclose(params_ham_dict['delta_cav'], expected_rates), f"Expected {expected_rates}, but got {params_ham_dict['delta_cav']}"
    #delta_em
    expected_rates = jnp.array([[0.0,2.0,0.0],[0.5,3.0,1.5],[-0.5,1.0,-1.5],[2.0/6.0,2.0-4.0/3,6.0/4.0]])
    assert jnp.allclose(params_ham_dict['delta_em'], expected_rates), f"Expected {expected_rates}, but got {params_ham_dict['delta_em']}"
    #kappa
    expected_rates = jnp.array([[1.0,2.0,3.0],[1.5,3.0,4.5],[0.5,1.0,1.5],[1.0+2.0/6.0,2.0-4.0/3,3.0+6.0/4.0]])
    assert jnp.allclose(params_ham_dict['kappa'], expected_rates), f"Expected {expected_rates}, but got {params_ham_dict['kappa']}"
