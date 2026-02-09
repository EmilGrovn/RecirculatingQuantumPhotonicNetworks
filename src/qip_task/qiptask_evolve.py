#qiptask_evolve.py
#evolution should be defined as function of trainable parameters
#step 1: build outgoing and incoming unitary transformation from trainable parameters
from rqpnn.rqpnn_parameterization import build_coefficient_matrix,S_in_general_from_C
import jax.numpy as jnp
def compute_unitary_in_out(spec,build_mode_switch_unitary,params_train_dict):
    #NOTE: reusing two lines from rqpnn_parameterization.py
    #NOTE: this is not used in the final RQPN design and not used in the paper
    C=build_coefficient_matrix(spec, params_train_dict['coeff_vec']) #parameters for training C
    S_in_mat=S_in_general_from_C(spec,C,params_train_dict['S_in_Rmat'] )
    #assert jnp.allclose(S_in_mat@jnp.conjugate(S_in_mat).T, jnp.eye(spec.N_cav), atol=1e-6), "S_in matrix is not unitary"
    #find inout and output unitary
    S_out_mat=S_in_mat.T
    U_in=build_mode_switch_unitary(S_in_mat)
    U_out=build_mode_switch_unitary(S_out_mat)
    return U_in, U_out
def mode_transform_states(input_states,mode_switch_unitary):
    #NOTE: this is not used in the final RQPN design and not used in the paper
    transformed_states=mode_switch_unitary@input_states
    print("shape states mode evo",transformed_states.shape)
    return transformed_states
#step 2: build time evolution from trainable parameters
import dynamiqs as dq
def _time_evolution(H_func_para,params_train_dict,input_states,tsave,solver):
    H=H_func_para(params_train_dict)
    res=dq.sesolve(
        H,
        input_states,#self.initial_states,
        tsave,
        method = solver)
    #shape (n_states,n_time_steps,n_hilbert,1)
    print("out state shape",res.states.shape)
    final_states=res.states[:,-1,:,:]
    #print("states after time evo",final_states)
    #print("shape states after time evo",final_states.shape)
    return final_states
#step 3: build evolve function
def _build_evolve(include_time_evo,include_mode_evo,initial_states,
        spec,build_mode_switch_unitary,H_func_para,tsave,solver):
    '''
    inputs:
    - include_time_evo: bool, whether to include time evolution
    - include_mode_evo: bool, whether to include mode transformation
    - initial_states: list of dynamiqs.qarrays.QArray, initial states to evolve
    - spec: RQPNNSpec object, specification of the RQPNN system
    - build_mode_switch_unitary: function, function to build the mode switch unitary, of S_in
    - H_func_para: function, function to build the Hamiltonian for time evolution, of train_params_dict
    - tsave: jax array, time points to save the states
    - solver: str, solver to use for time evolution
    returns:
    - _evolve: function, function that takes in params_train_dict and returns the final states after evolution
    '''
    if include_time_evo==True and include_mode_evo==False:
        #NOTE: this is not used in the final RQPN design
        #@partial(jax.jit, static_argnames=('self',))
        print("only time evolution included")
        def _evolve(params_train_dict):
            states=_time_evolution(H_func_para,params_train_dict,initial_states,tsave,solver)
            print(states.dag()@states)
            print("norm",dq.to_jax(states.dag()@states))
            #assert jnp.allclose(dq.to_jax(states.dag()@states),1.0,atol=1e-4), "States are not normalized after time evolution"
            return states
        return _evolve
    elif include_time_evo==False and include_mode_evo==True:
        print("only mode evolution included")
        def _evolve(params_train_dict):
            #mode transformation
            U_in, U_out=compute_unitary_in_out(spec,build_mode_switch_unitary,params_train_dict)
            states=mode_transform_states(initial_states,U_in)
            states=mode_transform_states(states,U_out)
            print("norm",dq.to_jax(states.dag()@states))
            #assert jnp.allclose(dq.to_jax(states.dag()@states),1.0,atol=1e-6), "States are not normalized after time evolution"
            return states
        return _evolve            
    elif include_time_evo==True and include_mode_evo==True:
        #NOTE: this is not used in the final RQPN design
        print("time and mode evolution included")
        def _evolve(params_train_dict):
            #mode transformation
            U_in, U_out=compute_unitary_in_out(spec,build_mode_switch_unitary,params_train_dict)
            states=mode_transform_states(initial_states,U_in)
            states=_time_evolution(H_func_para,params_train_dict,states,tsave,solver)
            states=mode_transform_states(states,U_out)
            print("norm",dq.to_jax(states.dag()@states))        
            #assert jnp.allclose(dq.to_jax(states.dag()@states),1.0,atol=1e-6), "States are not normalized after time evolution"
            return states
        return _evolve