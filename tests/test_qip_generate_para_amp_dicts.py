from qip_task.qiptask_generate_params import generate_params_train
from qip_task.qiptask_generate_params import generate_params_amp
import jax.numpy as jnp
from rqpnn.rqpnn_spec import RQPNNSpec
def test_generate_para_dicts_custom():

    spec= RQPNNSpec(2, 2, (1,))
    #dictionary that determines trainable_parameters
    #if custom=True, then we are simply providing params_train_dict
    params_train_dict={'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([[0.0],[0.0]]), #length 1 = 2*1/2
        'tau': jnp.array([0.0, -jnp.log(3.0), jnp.log(2.0)]) ,
        'delta_cav': jnp.array([[0.0,1.0,-1/jnp.sqrt(3)],[1/jnp.sqrt(3),0.0,jnp.sqrt(3)]]),
        #'delta_em': jnp.array([[1.0,-1.0,1/jnp.sqrt(3)],[-1/jnp.sqrt(3),jnp.sqrt(3),0.0]]),
        'kappa': jnp.array([[jnp.sqrt(3),1.0,-1/jnp.sqrt(3)],[1/jnp.sqrt(3),0.0,1.0]])}
    train_para_spec={
        "custom": True,
        "run_id": "9dpqr3lc"
        }
    out=generate_params_train(spec,train_para_spec)
    print("out",out)
    print("params_train_dict",params_train_dict)#NOTE: same dict, if we change something in one, it is also changed in the other
    assert params_train_dict.keys()==out.keys()

def test_generate_para_dicts_notcustom():
    spec= RQPNNSpec(2, 2, (1,))
    #dictionary that determines trainable_parameters
    #if custom=False, then we are to generate params_train_dict from few specifications
    train_para_spec={
        "custom": False,
        "N_steps": 3, #number of time pieces
        "key_seed": 43,
        "tau": {"offset": 1.0,"noise": 0.0},
        "delta_cav": {"offset": 2.0,"noise": 0.0},
        "delta_em": {"offset": 3.0,"noise": 0.0},
        "kappa": {"offset": 4.0,"noise": 0.0},
        "coeff_vec": {"offset": 5.0,"noise": 0.0},
        "S_in_Rmat": {"offset": 6.0,"noise": 0.0}
        }
    out=generate_params_train(spec,train_para_spec)
    print("out",out)
    assert jnp.allclose(out['tau'], jnp.array([1.0,1.0,1.0]))
    assert jnp.allclose(out['delta_cav'], jnp.array([[2.0,2.0,2.0],[2.0,2.0,2.0]]))
    assert jnp.allclose(out['delta_em'], jnp.array([[3.0,3.0,3.0],[3.0,3.0,3.0]]))
    assert jnp.allclose(out['kappa'], jnp.array([[4.0,4.0,4.0],[4.0,4.0,4.0]]))
    assert jnp.allclose(out['coeff_vec'], jnp.array([5.0]))
    assert jnp.allclose(out['S_in_Rmat'], jnp.array([6.0]))
    #Option: Print noise parameters
    train_para_spec={
        "custom": False,
        "N_steps": 3, #number of time pieces
        "key_seed": 43,
        "tau": {"offset": 0.0,"noise": 1.0},
        "delta_cav": {"offset": 0.0,"noise": 1.0},
        "delta_em": {"offset": 0.0,"noise": 1.0},
        "kappa": {"offset": 0.0,"noise": 0.5},
        "coeff_vec": {"offset": 0.0,"noise": 0.5},
        "S_in_Rmat": {"offset": 0.0,"noise": 0.2}
        }
    out=generate_params_train(spec,train_para_spec)
    print("out",out)
    #assert False, "Just wanted to print some noisy parameters"

#########
#amplitude
def test_generate_para_dicts_ampCustom():
    spec= RQPNNSpec(2, 2, (1,))
    #dictionary that determines amplitude parameters
    #if custom=True, then we are simply providing params_amp_dict
    params_amp_dict={
        'max_tau': 10.0*jnp.ones(3),'min_tau': 2.0*jnp.ones(3),
        'delta_cav': jnp.array([1.0,3.0,2.0]),
        #'delta_em': jnp.array([2.0,1.0,4.0]),
        #'delta_em_offset': jnp.array([0.0,1.0,0.0]),
        'kappa': jnp.array([1.0,3.0,2.0])
    }
    amp_para_spec={
        "custom": True,
        "run_id": "9dpqr3lc"
        }
    out=generate_params_amp(amp_para_spec,spec)
    print("out",out)
    print("params_amp_dict",params_amp_dict)
    assert params_amp_dict.keys()==out.keys()

def test_generate_para_dicts_ampNotCustom():
    spec= RQPNNSpec(2, 2, (1,))
    #dictionary that determines amplitude parameters
    #if custom=False, then we are to generate params_amp_dict from few specifications
    amp_para_spec={
        "custom": False,
        "N_steps": 3, #number of time pieces
        "max_tau": 10.0,
        "min_tau": 2.0,
        "delta_cav": {"shape": "uniform", "amp_physical": 1.0, "ramp_up_speed_physical": 1.0},
        "delta_em": {"shape": "uniform", "amp_physical": 2.0, "ramp_up_speed_physical": 1.0,
                     "offset_shape": "uniform", "offset_size_physical": 1.0},
        "kappa": {"shape": "uniform", "amp_physical": 3.0, "ramp_up_speed_physical": 1.0}
        }
    out=generate_params_amp(amp_para_spec,spec)
    print("out",out)
    #NOTE: amplitudes of rates are multiplied by min_tau!!
    assert jnp.allclose(out['max_tau'], jnp.array([10.0,10.0,10.0]))
    assert jnp.allclose(out['min_tau'], jnp.array([2.0,2.0,2.0]))
    assert jnp.allclose(out['delta_cav'], jnp.array([1.0,1.0,1.0])*2.0)
    assert jnp.allclose(out['delta_em'], jnp.array([2.0,2.0,2.0])*2.0)
    assert jnp.allclose(out['delta_em_offset'], jnp.array([1.0,1.0,1.0])*10.0)
    assert jnp.allclose(out['kappa'], jnp.array([3.0,3.0,3.0])*2.0)

def test_generate_para_dicts_ramp():
    spec= RQPNNSpec(2, 2, (1,))
    #dictionary that determines amplitude parameters
    #if custom=False, then we are to generate params_amp_dict from few specifications
    amp_para_spec={
        "custom": False,
        "N_steps": 6, #number of time pieces
        "max_tau": 6.0,
        "min_tau": 2.0,
        "delta_cav": {"shape": "ramp", "amp_physical": 10.0, "ramp_up_speed_physical": 1.0},
        "delta_em": {"shape": "uniform", "amp_physical": 2.0, "ramp_up_speed_physical": 3.0,
                     "offset_shape": "ramp", "offset_size_physical": 20.0},
        "kappa": {"shape": "ramp", "amp_physical": 7.0, "ramp_up_speed_physical": 1.0}
        }
    out=generate_params_amp(amp_para_spec,spec)
    print("out",out)
    #NOTE: amplitudes of rates are multiplied by min_tau!!
    assert jnp.allclose(out['max_tau'], jnp.array([6.0,6.0,6.0,6.0,6.0,6.0]))
    assert jnp.allclose(out['min_tau'], jnp.array([2.0,2.0,2.0,2.0,2.0,2.0]))
    assert jnp.allclose(out['delta_cav'], jnp.array([2.0,6.0,10.0,10.0,6.0,2.0])*2.0)
    assert jnp.allclose(out['delta_em'], jnp.array([2.0,2.0,2.0,2.0,2.0,2.0])*2.0)
    assert jnp.allclose(out['delta_em_offset'], jnp.array([108.0,84.0,60.0,60.0,84.0,108.0]))
    assert jnp.allclose(out['kappa'], jnp.array([2.0,6.0,7.0,7.0,6.0,2.0])*2.0)