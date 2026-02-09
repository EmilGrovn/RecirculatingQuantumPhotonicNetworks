
from qip_task.qiptask_costfunc import compute_physical_rates, compute_timeduration_penalty, compute_bandwidth_penalty
from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_system import RQPNNSystem
import jax.numpy as jnp

spec= RQPNNSpec(2, 2, (1,))
RQPNN=RQPNNSystem(spec)
basis_cons_mat=RQPNN.basis_conserved_mat
N_p=3
params_amp_dict={
    'max_tau': 2.0*jnp.ones(N_p),'min_tau': 1.0*jnp.ones(N_p),
    'delta_cav': jnp.array([1.0,2.0,1.0]),
    'delta_em': jnp.array([1.0,1.0,1.0]),
    'kappa': jnp.array([3.0,4.0,2.0]),
    'delta_em_offset': jnp.array([1.0,0.0,1.0]),
    'delta_em_rest': 2.0
}
params_train_dict={
    'coeff_vec': jnp.array([1.0]),
    'S_in_Rmat': jnp.array([0.0]), #length 12 = 4*3/2*2
    'tau': jnp.array([0.0,jnp.log(2),-jnp.log(3)]),
    'delta_cav': jnp.array([[0.0,1.0,1.0],[-1.0,-jnp.sqrt(3),1.0]]),#atan(1)/pi=0.25
    'delta_em': jnp.array([[0.0,1.0,0.0],[0.0,-1.0,-jnp.sqrt(3)]]),
    'kappa': jnp.array([[0.0,1.0,1/jnp.sqrt(3)],[-1.0,0.0,-jnp.sqrt(3)]])
}

def test_physical_rate_gen():

    rates=compute_physical_rates(spec,params_train_dict,params_amp_dict)
    print("tau",rates[0])
    print("delta_cav phys",rates[1])
    print("kappa",rates[2])
    assert jnp.allclose(rates[0],jnp.array([1.5,5/3,1.25]),atol=1e-5)
    assert jnp.allclose(rates[1],jnp.array([[0,0.3,0.2],[-1/6,-0.4,0.2]]))
    assert jnp.allclose(rates[2],jnp.array([[1.0,9/5,16/15],[0.5,6/5,4/15]]))
    assert jnp.allclose(rates[3],jnp.array([[2/3,3/20,4/5],[2/3,-3/20,8/15]]))
import numpy as np
def test_compute_duration_penalty():
    rates=compute_physical_rates(spec,params_train_dict,params_amp_dict)
    time_pen,gate_duration=compute_timeduration_penalty(rates[0], 0.5)
    print("time_pen",time_pen)
    print("expected",53/12*0.5)
    assert np.allclose(time_pen,53/12*0.5)

def test_bwpen_energy_gradint():
    rate_phys_matrix=jnp.array([[2.0,1.0,4.0],[1.0,-1.0,3.0]])
    tau_vec=jnp.array([1.0,2.0,4.0])
    weight_smooth_pen=10.0
    offset_rest=2.0
    loss, energy, grad_int, eff_bandwidth = compute_bandwidth_penalty(rate_phys_matrix, tau_vec,weight_smooth_pen,offset_rest)
    print("energy",energy)
    print("grad_int",grad_int)
    energy_val=18.0
    grad_int_val=97/12.0
    assert np.allclose(energy,energy_val)
    assert np.allclose(grad_int,grad_int_val)
    assert np.allclose(loss, 10*grad_int_val/(1.0+energy_val))
    assert np.allclose(eff_bandwidth, jnp.sqrt(grad_int_val/energy_val))


def test_full_loss_calc():
    #test full stack
    spec= RQPNNSpec(2, 2, (1,))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    N_p=3
    params_amp_dict={
        'max_tau': 2.0*jnp.ones(N_p),'min_tau': 1.0*jnp.ones(N_p),
        'delta_cav': jnp.array([2.0,2.0,2.0]),
        'delta_em': jnp.array([1.0,1.0,1.0]),
        'kappa': jnp.array([2.0,2.0,2.0]),
        'delta_em_offset': jnp.array([1.0,1.0,1.0]),
        'delta_em_rest': 1.0
    }
    params_train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([0.0]), #length 12 = 4*3/2*2
        'tau': jnp.array([0.0,0.0,0.0]),
        'delta_cav': jnp.array([[1.0,1.0,1.0],[-1.0,-1.0,-1.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[1.0,1.0,1.0],[0.0,0.0,0.0]]),
        'kappa': jnp.array([[1.0,1.0,1.0],[0.0,0.0,0.0]])
    }
    from qip_task.qiptask_costfunc import build_loss
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.1,
        'weight_bw_deltacav_pen': 0.2,
        'weight_bw_kappa_pen': 0.3,
        'weight_bw_deltaem_pen': 0.4
    }
    from qip_task.qiptask_evolve import _build_evolve
    include_time_evo=True
    include_mode_evo=False
    from qip_task.qiptask_init_target_states import _generate_initial_and_target_states
    initial_states,target_states=_generate_initial_and_target_states(spec,"Pi/4-Phase gate 1qubit",basis_cons_mat)
    tsave=jnp.linspace(0.0,1.0,10) #time points to save states
    import dynamiqs as dq
    solver=dq.method.Expm()
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    _evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
            spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    loss=build_loss(spec,params_amp_dict,pen_dict,_evolve,target_array)
    out=loss(params_train_dict)
    print("loss",out)
    loss,aux=out
    loss_tot,infidelity,inner_prods,gate_duration,energy_list,gradint_list,eff_bandwidth_list=aux
    rates=compute_physical_rates(spec,params_train_dict,params_amp_dict)
    loss_dc, energy_dc, grad_int_dc, eff_bandwidth_dc = compute_bandwidth_penalty(rates[1], rates[0],0.2,0.0)
    loss_kappa, energy_kappa, grad_int_kappa, eff_bandwidth_kappa = compute_bandwidth_penalty(rates[2], rates[0],0.3,0.0)
    loss_de, energy_de, grad_int_de, eff_bandwidth_de = compute_bandwidth_penalty(rates[3], rates[0],0.4,1.0)
    time_pen, gate_duration_val=compute_timeduration_penalty(rates[0], 0.1)
    expected_loss= jnp.log(infidelity)*1.0 + time_pen + loss_dc + loss_kappa + loss_de
    assert np.allclose(loss, expected_loss)
    assert np.allclose(gate_duration, gate_duration_val)
    assert np.allclose(energy_list[0], energy_dc)
    assert np.allclose(energy_list[1], energy_kappa)
    assert np.allclose(energy_list[2], energy_de)
    assert np.allclose(gradint_list[0], grad_int_dc)
    assert np.allclose(gradint_list[1], grad_int_kappa)
    assert np.allclose(gradint_list[2], grad_int_de)
    assert np.allclose(eff_bandwidth_list[0], eff_bandwidth_dc)
    assert np.allclose(eff_bandwidth_list[1], eff_bandwidth_kappa)
    assert np.allclose(eff_bandwidth_list[2], eff_bandwidth_de)

def test_handle_kappa_for_forward():
    from qip_task.qiptask_costfunc import handle_kappa_for_forward
    #case 1: kappa shape (1,N_p) and M_cav>1
    spec= RQPNNSpec(2, 2, (1,))
    params_train_dict={
        'delta_cav': jnp.array([[1.0,2.0,3.0],[-1.0,-2.0,-3.0]]),
        'kappa': jnp.array([[1.0,2.0,3.0]])
    }
    params_eff=handle_kappa_for_forward(spec, params_train_dict)
    print("params_eff",params_eff)
    assert jnp.allclose(params_eff["kappa"], jnp.array([[1.0,2.0,3.0],[1.0,2.0,3.0]]))
    #case 2: kappa shape (M_cav,N_p)
    params_train_dict={
        'kappa': jnp.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    }
    params_eff=handle_kappa_for_forward(spec, params_train_dict)
    print("params_eff",params_eff)
    assert jnp.allclose(params_eff["kappa"], jnp.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]))