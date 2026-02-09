#test hamiltonian

from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_system import RQPNNSystem
import jax.numpy as jnp
#

#test helpers
def test_H_pwc_init_tau():
    spec = RQPNNSpec(2, 0, (2,))
    sys = RQPNNSystem(spec)
    N_pieces=3
    amp_dict={
        'max_tau': 10.0*jnp.ones(3),'min_tau': 2.0*jnp.ones(3),
        'delta_cav': jnp.array([0.0,0.0,0.0]),
        'delta_em': jnp.array([0.0,0.0,0.0]),
        'kappa': jnp.array([0.0,0.0,0.0])
    }
    build_H=sys._build_H_func(N_pieces, amp_dict)
    #
    train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([0.0]), #length 1 = 2*1/2
        'tau': jnp.array([0.0, -jnp.log(3.0), jnp.log(2.0)]) ,
        'delta_cav': jnp.array([[0.0,0.0,0.0],[0.0,0.0,0.0]]),
        'kappa': jnp.array([[0.0,0.0,0.0],[0.0,0.0,0.0]])
    }
    H_t=build_H(train_dict)
    import dynamiqs as dq
    print(H_t(0))
    H0=dq.to_jax(H_t(0))
    H0_expected=dq.to_jax((2+8/2.0)*(-sys.dict_ham_ops["chi3"][0]-sys.dict_ham_ops["chi3"][1]))
    H1_expected=dq.to_jax((2+8/4.0)*(-sys.dict_ham_ops["chi3"][0]-sys.dict_ham_ops["chi3"][1]))
    H2_expected=dq.to_jax((2+8*2/3.0)*(-sys.dict_ham_ops["chi3"][0]-sys.dict_ham_ops["chi3"][1]))    
    print("H1",dq.to_jax(H_t(1.0)))
    print("H1 expected",H1_expected)
    print("H3",dq.to_jax(H_t(3.0)))
    assert jnp.allclose(H0, H0_expected,atol=1e-5), "H at t=0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(0.5)), H0_expected,atol=1e-5), "H at t=0.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.0)), H1_expected,atol=1e-5), "H at t=1.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.5)), H1_expected,atol=1e-5), "H at t=1.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.0)), H2_expected,atol=1e-5), "H at t=2.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.5)), H2_expected,atol=1e-5), "H at t=2.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(3.0)), jnp.zeros(H0.shape),atol=1e-5), "H at t=3.0 is not correct"
    # Check that H has the correct shape

def test_H_pwc_init_delta_cav():
    spec = RQPNNSpec(2, 0, (2,))
    sys = RQPNNSystem(spec)
    N_pieces=3
    amp_dict={
        'max_tau': 0.0*jnp.ones(3),'min_tau': 0.0*jnp.ones(3),
        'delta_cav': jnp.array([1.0,3.0,2.0]),
        'delta_em': jnp.array([0.0,0.0,0.0]),
        'kappa': jnp.array([0.0,0.0,0.0])
    }
    build_H=sys._build_H_func(N_pieces, amp_dict)
    #
    train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([0.0]), #length 1 = 2*1/2
        'tau': jnp.array([0.0, 0.0, 0.0]) ,
        'delta_cav': jnp.array([[0.0,1.0,-1/jnp.sqrt(3)],[1/jnp.sqrt(3),0.0,jnp.sqrt(3)]]),
        'kappa': jnp.array([[0.0,0.0,0.0],[0.0,0.0,0.0]])
    }
    H_t=build_H(train_dict)
    import dynamiqs as dq
    print(H_t(0))
    H0=dq.to_jax(H_t(0))
    H0_expected=dq.to_jax(0.0*sys.dict_ham_ops["delta_cav"][0]+1/6*sys.dict_ham_ops["delta_cav"][1])
    H1_expected=dq.to_jax(3/4*sys.dict_ham_ops["delta_cav"][0]+0.0*sys.dict_ham_ops["delta_cav"][1])
    H2_expected=dq.to_jax(-2/6*sys.dict_ham_ops["delta_cav"][0]+2/3*sys.dict_ham_ops["delta_cav"][1])   
    print("H0 expected",H0_expected)
    print("H1",dq.to_jax(H_t(1.0)))
    print("H1 expected",H1_expected)
    print("H3",dq.to_jax(H_t(3.0)))
    assert jnp.allclose(H0, H0_expected,atol=1e-5), "H at t=0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(0.5)), H0_expected,atol=1e-5), "H at t=0.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.0)), H1_expected,atol=1e-5), "H at t=1.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.5)), H1_expected,atol=1e-5), "H at t=1.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.0)), H2_expected,atol=1e-5), "H at t=2.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.5)), H2_expected,atol=1e-5), "H at t=2.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(3.0)), jnp.zeros(H0.shape),atol=1e-5), "H at t=3.0 is not correct"

def test_H_pwc_init_kappa():
    spec = RQPNNSpec(2, 0, (2,))
    sys = RQPNNSystem(spec)
    N_pieces=3
    amp_dict={
        'max_tau': 0.0*jnp.ones(3),'min_tau': 0.0*jnp.ones(3),
        'delta_cav': jnp.array([0.0,0.0,0.0]),
        'delta_em': jnp.array([0.0,0.0,0.0]),
        'kappa': jnp.array([1.0,3.0,2.0])
    }
    build_H=sys._build_H_func(N_pieces, amp_dict)
    #
    train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([0.0]), #length 1 = 2*1/2
        'tau': jnp.array([0.0, 0.0, 0.0]) ,
        'delta_cav': jnp.array([[0.0,0.0,0.0],[0.0,0.0,0.0]]),
        'kappa': jnp.array([[jnp.sqrt(3),1.0,-1/jnp.sqrt(3)],[1/jnp.sqrt(3),0.0,1.0]])
    }
    H_t=build_H(train_dict)
    import dynamiqs as dq
    #print(H_t(0))
    H0=dq.to_jax(H_t(0))
    H0_expected=dq.to_jax((jnp.sqrt((0.5+1/6)*(0.5+1/3)))*(sys.dict_ham_ops['int'][0][1]+sys.dict_ham_ops["int"][1][0]))
    H1_expected=dq.to_jax(3*jnp.sqrt(0.5*(0.5+1/4))*(sys.dict_ham_ops['int'][0][1]+sys.dict_ham_ops["int"][1][0]))
    H2_expected=dq.to_jax(2*jnp.sqrt((0.5+1/4)*(0.5-1/6))*(sys.dict_ham_ops['int'][0][1]+sys.dict_ham_ops["int"][1][0])) 
    print("H0",dq.to_jax(H_t(0)))
    print("H0 expected",H0_expected)
    print("H1",dq.to_jax(H_t(1.0)))
    print("H1 expected",H1_expected)
    print("H2",dq.to_jax(H_t(2.0)))
    print("H2 expected",H2_expected)
    print("H3",dq.to_jax(H_t(3.0)))
    assert jnp.allclose(dq.to_jax(H_t(0)), H0_expected,atol=1e-5), "H at t=0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(0.5)), H0_expected,atol=1e-5), "H at t=0.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.0)), H1_expected,atol=1e-5), "H at t=1.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.5)), H1_expected,atol=1e-5), "H at t=1.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.0)), H2_expected,atol=1e-5), "H at t=2.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.5)), H2_expected,atol=1e-5), "H at t=2.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(3.0)), jnp.zeros(H0.shape),atol=1e-5), "H at t=3.0 is not correct"

def test_entire_ham():
    spec = RQPNNSpec(2, 2, (2,))
    sys = RQPNNSystem(spec)
    N_pieces=3
    amp_dict={
        'max_tau': 10.0*jnp.ones(3),'min_tau': 2.0*jnp.ones(3),
        'delta_cav': jnp.array([1.0,3.0,2.0]),
        'delta_em': jnp.array([2.0,1.0,4.0]),
        'kappa': jnp.array([1.0,3.0,2.0]),
        'delta_em_offset': jnp.array([0.0,0.0,0.0])
    }
    build_H=sys._build_H_func(N_pieces, amp_dict)
    #
    train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([0.0]), #length 1 = 2*1/2
        'tau': jnp.array([0.0, -jnp.log(3.0), jnp.log(2.0)]) ,
        'delta_cav': jnp.array([[0.0,1.0,-1/jnp.sqrt(3)],[1/jnp.sqrt(3),0.0,jnp.sqrt(3)]]),
        'delta_em': jnp.array([[1.0,-1.0,1/jnp.sqrt(3)],[-1/jnp.sqrt(3),jnp.sqrt(3),0.0]]),
        'kappa': jnp.array([[jnp.sqrt(3),1.0,-1/jnp.sqrt(3)],[1/jnp.sqrt(3),0.0,1.0]])
    }
    H_t=build_H(train_dict)
    import dynamiqs as dq
    H0_expected_tau=dq.to_jax((2+8/2.0)*(sys.dict_ham_ops["g_lm"][0]+sys.dict_ham_ops["g_lm"][1]))
    H1_expected_tau=dq.to_jax((2+8/4.0)*(sys.dict_ham_ops["g_lm"][0]+sys.dict_ham_ops["g_lm"][1]))
    H2_expected_tau=dq.to_jax((2+8*2/3.0)*(sys.dict_ham_ops["g_lm"][0]+sys.dict_ham_ops["g_lm"][1])) 
    H0_expected_delta_cav=dq.to_jax(0.0*sys.dict_ham_ops["delta_cav"][0]+1/6*sys.dict_ham_ops["delta_cav"][1])
    H1_expected_delta_cav=dq.to_jax(3/4*sys.dict_ham_ops["delta_cav"][0]+0.0*sys.dict_ham_ops["delta_cav"][1])
    H2_expected_delta_cav=dq.to_jax(-2/6*sys.dict_ham_ops["delta_cav"][0]+2/3*sys.dict_ham_ops["delta_cav"][1]) 
    H0_expected_kappa=dq.to_jax((jnp.sqrt((0.5+1/6)*(0.5+1/3)))*(sys.dict_ham_ops['int'][0][1]+sys.dict_ham_ops["int"][1][0]))
    H1_expected_kappa=dq.to_jax(3*jnp.sqrt(0.5*(0.5+1/4))*(sys.dict_ham_ops['int'][0][1]+sys.dict_ham_ops["int"][1][0]))
    H2_expected_kappa=dq.to_jax(2*jnp.sqrt((0.5+1/4)*(0.5-1/6))*(sys.dict_ham_ops['int'][0][1]+sys.dict_ham_ops["int"][1][0]))
    H0_expected_delta_em=dq.to_jax(2.0*0.25*sys.dict_ham_ops["delta_em"][0]-2.0/6*sys.dict_ham_ops["delta_em"][1])
    H1_expected_delta_em=dq.to_jax(-1.0*0.25*sys.dict_ham_ops["delta_em"][0]+1/3*sys.dict_ham_ops["delta_em"][1])
    H2_expected_delta_em=dq.to_jax(4/6*sys.dict_ham_ops["delta_em"][0]+0.0*sys.dict_ham_ops["delta_em"][1])
    H0_expected=H0_expected_tau+H0_expected_delta_cav+H0_expected_kappa+H0_expected_delta_em
    H1_expected=H1_expected_tau+H1_expected_delta_cav+H1_expected_kappa+H1_expected_delta_em
    H2_expected=H2_expected_tau+H2_expected_delta_cav+H2_expected_kappa+H2_expected_delta_em
    print("H0",dq.to_jax(H_t(0)))
    print("H0 expected",H0_expected)
    print("H1",dq.to_jax(H_t(1.0)))
    print("H1 expected",H1_expected)
    print("H2",dq.to_jax(H_t(2.0)))
    print("H2 expected",H2_expected)
    print("H3",dq.to_jax(H_t(3.0)))
    assert jnp.allclose(dq.to_jax(H_t(0)), H0_expected,atol=1e-5), "H at t=0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(0.5)), H0_expected,atol=1e-5), "H at t=0.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.0)), H1_expected,atol=1e-5), "H at t=1.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(1.5)), H1_expected,atol=1e-5), "H at t=1.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.0)), H2_expected,atol=1e-5), "H at t=2.0 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(2.5)), H2_expected,atol=1e-5), "H at t=2.5 is not correct"
    assert jnp.allclose(dq.to_jax(H_t(3.0)), jnp.zeros(dq.to_jax(H_t(0)).shape),atol=1e-5), "H at t=3.0 is not correct"
