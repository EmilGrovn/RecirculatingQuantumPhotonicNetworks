#we will test evolve for various known examples

from rqpnn.rqpnn_spec import RQPNNSpec
from rqpnn.rqpnn_system import RQPNNSystem
from qip_task.qiptask_evolve import _build_evolve
from qip_task.qiptask_init_target_states import _generate_initial_and_target_states
import jax.numpy as jnp
import dynamiqs as dq
from qip_task.qiptask_costfunc import build_loss
#Pi/4 phase gate time evolution test
def test_evolve_pi4_gate_time():
    include_time_evo=True
    include_mode_evo=False
    spec= RQPNNSpec(2, 0, (1,))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    QIPtask="Pi/4-Phase gate 1qubit"
    tsave=jnp.linspace(0.0,1.0,10) #time points to save states
    solver=dq.method.Expm()
    initial_states,target_states=_generate_initial_and_target_states(spec,QIPtask,basis_cons_mat)
    params_amp_dict={
        'max_tau': 2.0*jnp.ones(1),'min_tau': 1.0*jnp.ones(1),
        'delta_cav': jnp.array([jnp.pi]),
        'delta_em': jnp.array([2.0]),
        'kappa': jnp.array([0.0]),
        'delta_em_offset': jnp.array([0.0])
    }
    N_p=1
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    params_train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([0.0]), #length 12 = 4*3/2*2
        'tau': jnp.array([0.0]) ,
        'delta_cav': jnp.array([[0.0],[-1.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[0.0],[0.0]]),
        'kappa': jnp.array([[0.0],[1.0]])
    }    
    #test
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    ham_properties=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    print("ham_properties",ham_properties)

    evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    
    final_states=evolve(params_train_dict)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    print("final_states",final_states)
    print("target_states",target_array)
    assert jnp.allclose(dq.to_jax(final_states),dq.to_jax(target_array),atol=1e-4), "Time-Evolved states do not match target states for pi/4 phase gate time evolution"
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.0,
        'weight_bw_deltacav_pen': 0.0,  
        'weight_bw_kappa_pen': 0.0,
        'weight_bw_deltaem_pen': 0.0
    }
    loss=build_loss(spec,params_amp_dict,pen_dict,evolve,target_array)
    loss,aux=loss(params_train_dict)
    inf=aux[1]
    print("infidelity",inf)
    assert inf<1e-4, "Loss is too high for pi/4 phase gate time evolution"

#Pi/4 phase gate time evolution test
def test_evolve_hadamard_gate_time():
    include_time_evo=True
    include_mode_evo=False
    spec= RQPNNSpec(2, 0, (1,))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    QIPtask="Hadamard gate 1qubit"
    solver=dq.method.Expm()
    initial_states,target_states=_generate_initial_and_target_states(spec,QIPtask,basis_cons_mat)
    params_amp_dict={
        'max_tau': 2.0*jnp.ones(3),'min_tau': 1.0*jnp.ones(3),
        'delta_cav': jnp.array([jnp.pi*2,jnp.pi*2,jnp.pi*2]),
        'delta_em': jnp.array([2.0]),
        'kappa': jnp.array([jnp.pi/2,jnp.pi/2,jnp.pi/2]),
        'delta_em_offset': jnp.array([0.0])
    }
    N_p=3
    tsave=jnp.linspace(0.0,N_p*1.0,10) #time points to save states
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    params_train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.array([0.0]), #length 12 = 4*3/2*2
        'tau': jnp.array([0.0,0.0,0.0]) ,
        'delta_cav': jnp.array([[0.0,0.0,0.0],[-1.0,0.0,-1.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[0.0],[0.0]]),
        'kappa': jnp.array([[-100000.0,0.0,-100000.0],[-100000.0,0.0,-100000.0]])#0.5*amp_max
    }    
    #test
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    ham_properties=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    print("ham_properties",ham_properties)

    evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    
    final_states=evolve(params_train_dict)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    print("final_states",final_states,final_states.shape)
    norm=final_states.dag()@final_states
    print("final states norm",norm,norm.shape)
    print("target_states",target_array)#I will ad a global Pi phase
    assert jnp.allclose(dq.to_jax(final_states),dq.to_jax(target_array),atol=1e-4), "Time-Evolved states do not match target states for Hadamard gate time evolution"
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.0,
        'weight_bw_deltacav_pen': 0.0,  
        'weight_bw_kappa_pen': 0.0,
        'weight_bw_deltaem_pen': 0.0
    }
    loss_func=build_loss(spec,params_amp_dict,pen_dict,evolve,target_array)
    loss,aux=loss_func(params_train_dict)
    inf=aux[1]
    print("infidelity",inf)
    assert inf<1e-4, "Loss is too high for pi/4 phase gate time evolution"

def test_evolve_CZ_gate_time():
    include_time_evo=True
    include_mode_evo=False
    spec= RQPNNSpec(4, 0, (2,))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    QIPtask="CZ gate 2qubit"
    solver=dq.method.Expm()
    initial_states,target_states=_generate_initial_and_target_states(spec,QIPtask,basis_cons_mat)
    N_p=1
    m1=36
    m2=80
    params_amp_dict={
        'max_tau': 10.0*jnp.ones(N_p),'min_tau': jnp.array([jnp.pi*(1+m2*2.0)]),
        'delta_cav': 0.0*jnp.ones(N_p),
        'delta_em': jnp.array([2.0]),
        'kappa': 2*m1*2*jnp.pi*jnp.ones(N_p),
        'delta_em_offset': jnp.array([0.0])
    }
    tsave=jnp.linspace(0.0,N_p*1.0,10) #time points to save states
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    params_train_dict={
        'coeff_vec': jnp.array([0.0,0.0,0.0,0.0,1.0,0.0]),
        'S_in_Rmat': jnp.zeros((6)), #length 12 = 4*3/2*2
        'tau': jnp.array([-100000.0]) ,
        'delta_cav': jnp.array([[0.0],[0.0],[0.0],[0.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[0.0],[0.0]]),
        'kappa': jnp.array([[-100000.0],[0.0],
                            [-100000.0],[0.0]])#0.5*amp_max
    }    
    #test
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    ham_properties=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    print("ham_properties",ham_properties)

    evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    
    final_states=evolve(params_train_dict)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    print("final_states",final_states)
    print("target_states",target_array)
    assert jnp.allclose(dq.to_jax(final_states),dq.to_jax(target_array),atol=1e-2), "Time-Evolved states do not match target states for CZ gate time evolution"
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.0,
        'weight_bw_deltacav_pen': 0.0,  
        'weight_bw_kappa_pen': 0.0,
        'weight_bw_deltaem_pen': 0.0
    }
    loss_func=build_loss(spec,params_amp_dict,pen_dict,evolve,target_array)
    loss,aux=loss_func(params_train_dict)
    inf=aux[1]
    print("infidelity",inf)
    assert inf<1e-5, "Loss is too high for CZ gate time evolution"

def test_2rep_em():
    include_time_evo=True
    include_mode_evo=False
    spec= RQPNNSpec(2, 2, (5,6))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    QIPtask="2loss emitter coherent repeater"
    solver=dq.method.Expm()
    initial_states,target_states=_generate_initial_and_target_states(spec,QIPtask,basis_cons_mat)
    N_p=1
    params_amp_dict={
        'max_tau': 3.0*jnp.ones(N_p),'min_tau': 0.0*jnp.ones(N_p),
        'delta_cav': 0.0*jnp.ones(N_p),
        'delta_em': jnp.array([2.0]),
        'kappa': 2*jnp.pi*jnp.ones(N_p),
        'delta_em_offset': jnp.array([0.0]),
        'delta_em_rest': 0.0
    }
    tsave=jnp.linspace(0.0,N_p*1.0,10) #time points to save states
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    params_train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.zeros((1)), #length 12 = 4*3/2*2
        'tau': jnp.array([-100000.0]) ,
        'delta_cav': jnp.array([[0.0],[0.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[0.0],[0.0]]),
        'kappa': jnp.array([[-100000.0],[-100000.0]])#0.5*amp_max
    }    
    #test
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    ham_properties=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    print("ham_properties",ham_properties)

    evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    
    final_states=evolve(params_train_dict)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    # print("final_states",final_states)
    # print("target_states",target_array)
    for i in range(2):
        assert jnp.allclose(dq.to_jax(final_states[i].dag()@final_states[i]),1.0,atol=1e-3), f"Time-Evolved state {i} is not normalized"
        #print(f"State {i} norm:",dq.to_jax(final_states[i].dag()@final_states[i]))
        print(f"Final state {i}",final_states[i])
        print(f"Target state {i}",target_array[i])
        assert jnp.allclose(dq.to_jax(final_states[i]),dq.to_jax(target_array[i]),atol=1e-2), f"Time-Evolved state {i} does not match target state for 2-repeater emitter coherent repeater"
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.0,
        'weight_bw_deltacav_pen': 0.0,  
        'weight_bw_kappa_pen': 0.0,
        'weight_bw_deltaem_pen': 0.0
    }
    loss_func=build_loss(spec,params_amp_dict,pen_dict,evolve,target_array)
    loss,aux=loss_func(params_train_dict)
    inf=aux[1]
    print("infidelity",inf)
    inf_expect=1-(2/6)**2
    print("expected infidelity",inf_expect)
    assert inf==inf_expect

def test_Rabi_1photon_nodetuning():
    from rqpnn.rqpnn_stategeneration_modetrans import _generate_product_number_state_using_basis
    include_time_evo=True
    include_mode_evo=False
    spec= RQPNNSpec(2, 2, (1,))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    solver=dq.method.Expm()
    initial_states=[_generate_product_number_state_using_basis([1,0,0,0],basis_cons_mat)]
    target_states=[-_generate_product_number_state_using_basis([1,0,0,0],basis_cons_mat)]
    N_p=1
    params_amp_dict={
        'max_tau': 2*jnp.pi*jnp.ones(N_p),'min_tau': 0.0*jnp.ones(N_p),
        'delta_cav': 0.0*jnp.ones(N_p),
        'delta_em': jnp.array([2.0*jnp.sqrt(3.0)]),
        'kappa': jnp.array([3.0]),
        'delta_em_offset': jnp.array([0.0]),
        'delta_em_rest': 0.0
    }
    tsave=jnp.linspace(0.0,N_p*1.0,10) #time points to save states
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    params_train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.zeros((1)), #length 12 = 4*3/2*2
        'tau': jnp.array([0.0]) ,
        'delta_cav': jnp.array([[0.0],[0.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[0.0],[0.0]]),
        'kappa': jnp.array([[-100000.0],[-100000.0]])#0.5*amp_max
    }    
    #test
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    ham_properties=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    print("ham_properties",ham_properties)

    evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    
    final_states=evolve(params_train_dict)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    # print("final_states",final_states)
    # print("target_states",target_array)
    for i in range(2):
        assert jnp.allclose(dq.to_jax(final_states[i].dag()@final_states[i]),1.0,atol=1e-3), f"Time-Evolved state {i} is not normalized"
        #print(f"State {i} norm:",dq.to_jax(final_states[i].dag()@final_states[i]))
        print(f"Final state {i}",final_states[i])
        print(f"Target state {i}",target_array[i])
        assert jnp.allclose(dq.to_jax(final_states[i]),dq.to_jax(target_array[i]),atol=1e-2), f"Time-Evolved state {i} does not match target state for 2-repeater emitter coherent repeater"
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.0,
        'weight_bw_deltacav_pen': 0.0,  
        'weight_bw_kappa_pen': 0.0,
        'weight_bw_deltaem_pen': 0.0
    }
    loss_func=build_loss(spec,params_amp_dict,pen_dict,evolve,target_array)
    loss,aux=loss_func(params_train_dict)
    inf=aux[1]
    print("infidelity",inf)
    inf_expect=0.0
    print("expected infidelity",inf_expect)
    #assert inf==inf_expect
def test_Rabi_2photon_nodetuning():
    from rqpnn.rqpnn_stategeneration_modetrans import _generate_product_number_state_using_basis
    include_time_evo=True
    include_mode_evo=False
    spec= RQPNNSpec(2, 2, (2,))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    solver=dq.method.Expm()
    initial_states=[_generate_product_number_state_using_basis([2,0,0,0],basis_cons_mat)]
    target_states=[-1j*_generate_product_number_state_using_basis([1,0,1,0],basis_cons_mat)]
    N_p=1
    params_amp_dict={
        'max_tau': 1/jnp.sqrt(2)*jnp.pi*jnp.ones(N_p),'min_tau': 0.0*jnp.ones(N_p),
        'delta_cav': 0.0*jnp.ones(N_p),
        'delta_em': jnp.array([2.0*jnp.sqrt(3.0)]),
        'kappa': jnp.array([3.0]),
        'delta_em_offset': jnp.array([0.0]),
        'delta_em_rest': 0.0
    }
    tsave=jnp.linspace(0.0,N_p*1.0,10) #time points to save states
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    params_train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.zeros((1)), #length 12 = 4*3/2*2
        'tau': jnp.array([0.0]) ,
        'delta_cav': jnp.array([[0.0],[0.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[0.0],[0.0]]),
        'kappa': jnp.array([[-100000.0],[-100000.0]])#0.5*amp_max
    }    
    #test
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    ham_properties=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    print("ham_properties",ham_properties)

    evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    
    final_states=evolve(params_train_dict)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    # print("final_states",final_states)
    # print("target_states",target_array)
    for i in range(2):
        assert jnp.allclose(dq.to_jax(final_states[i].dag()@final_states[i]),1.0,atol=1e-3), f"Time-Evolved state {i} is not normalized"
        #print(f"State {i} norm:",dq.to_jax(final_states[i].dag()@final_states[i]))
        print(f"Final state {i}",final_states[i])
        print(f"Target state {i}",target_array[i])
        assert jnp.allclose(dq.to_jax(final_states[i]),dq.to_jax(target_array[i]),atol=1e-2), f"Time-Evolved state {i} does not match target state for 2-repeater emitter coherent repeater"
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.0,
        'weight_bw_deltacav_pen': 0.0,  
        'weight_bw_kappa_pen': 0.0,
        'weight_bw_deltaem_pen': 0.0
    }
    loss_func=build_loss(spec,params_amp_dict,pen_dict,evolve,target_array)
    loss,aux=loss_func(params_train_dict)
    inf=aux[1]
    print("infidelity",inf)
    inf_expect=0.0
    print("expected infidelity",inf_expect)

def test_Rabi_1photon_detuning():
    from rqpnn.rqpnn_stategeneration_modetrans import _generate_product_number_state_using_basis
    include_time_evo=True
    include_mode_evo=False
    spec= RQPNNSpec(2, 2, (1,))
    RQPNN=RQPNNSystem(spec)
    basis_cons_mat=RQPNN.basis_conserved_mat
    solver=dq.method.Expm()
    initial_states=[_generate_product_number_state_using_basis([1,0,0,0],basis_cons_mat),
    _generate_product_number_state_using_basis([0,1,0,0],basis_cons_mat)]
    N_p=1
    g=jnp.pi/2
    dc=jnp.sqrt(3)*g
    de=jnp.sqrt(3)*g
    params_amp_dict={
        'max_tau': 2*g*jnp.ones(N_p),'min_tau': 0.0*jnp.ones(N_p),
        'delta_cav': 4.0*dc*jnp.ones(N_p),
        'delta_em': jnp.array([4.0*de]),
        'kappa': jnp.array([3.0]),
        'delta_em_offset': jnp.array([0.0]),
        'delta_em_rest': 0.0
    }
    tsave=jnp.linspace(0.0,N_p*1.0,10) #time points to save states
    H_func_para=RQPNN._build_H_func(N_p,params_amp_dict)
    params_train_dict={
        'coeff_vec': jnp.array([1.0]),
        'S_in_Rmat': jnp.zeros((1)), #length 12 = 4*3/2*2
        'tau': jnp.array([0.0]) ,
        'delta_cav': jnp.array([[100000.0],[0.0]]),#atan(1)/pi=0.25
        'delta_em': jnp.array([[0.0],[-1000000.0]]),
        'kappa': jnp.array([[-100000.0],[-100000.0]])#0.5*amp_max
    }   
    target_states=[-jnp.exp(-1j*dc)*_generate_product_number_state_using_basis([1,0,0,0],basis_cons_mat),
    -jnp.exp(1j*de)*_generate_product_number_state_using_basis([0,1,0,0],basis_cons_mat)] 
    #NOTE: the formula says -de, but we flip de-sign through our parameterization
    #test
    from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
    ham_properties=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    print("ham_properties",ham_properties)

    evolve=_build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,RQPNN.build_mode_switch_unitary,H_func_para,tsave,solver)
    final_states=evolve(params_train_dict)
    target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))
    # print("final_states",final_states)
    # print("target_states",target_array)
    for i in range(2):
        assert jnp.allclose(dq.to_jax(final_states[i].dag()@final_states[i]),1.0,atol=1e-3), f"Time-Evolved state {i} is not normalized"
        #print(f"State {i} norm:",dq.to_jax(final_states[i].dag()@final_states[i]))
        print(f"Final state {i}",final_states[i])
        print(f"Target state {i}",target_array[i])
        assert jnp.allclose(dq.to_jax(final_states[i]),dq.to_jax(target_array[i]),atol=1e-2), f"Time-Evolved state {i} does not match target state for 2-repeater emitter coherent repeater"
    pen_dict={
        'weight_inf': 1.0,
        'weight_time_pen': 0.0,
        'weight_bw_deltacav_pen': 0.0,  
        'weight_bw_kappa_pen': 0.0,
        'weight_bw_deltaem_pen': 0.0
    }
    loss_func=build_loss(spec,params_amp_dict,pen_dict,evolve,target_array)
    loss,aux=loss_func(params_train_dict)
    inf=aux[1]
    print("infidelity",inf)
    inf_expect=0.0
    print("expected infidelity",inf_expect)