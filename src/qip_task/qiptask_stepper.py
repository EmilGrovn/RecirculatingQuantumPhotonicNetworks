#qiptask_stepper.py
import jax
import optax
### build function to take optimization step
def _build_stepper(_loss_func,optimizer,jit_stepper):
    '''
    inputs:
    - _loss_func: function, function that takes in parameters and returns loss and auxiliary info
    - optimizer: optax optimizer, optimizer to use for updating parameters
    - jit_stepper: bool, whether to jit the step function
    returns:
    - _step: function, function that takes in parameters and optimizer state and returns 
                -updated parameters, 
                -gradients, 
                -auxiliary info (loss_tot,infidelity,inner_prods,gate_duration,
                energy_list,grad_int_list,eff_bandwidth_list),
                -updated optimizer state
    '''
    #@partial(jax.jit, static_argnames=('self',))
    #jax.jit
    def _step(_parameters,_opt_state):
        # compute losses and their gradients with `jax.value_and_grad`
        fidelity_grad,aux = jax.grad(_loss_func,has_aux=True)(_parameters)
        #print("fidelity grad",fidelity_grad)
        # update amplitudes with optimizer
        updates, _opt_state = optimizer.update(fidelity_grad, _opt_state)
        _parameters = optax.apply_updates(_parameters, updates)

        return _parameters, fidelity_grad, aux, _opt_state
    if jit_stepper==True:
        print("JITTING step function")
        return jax.jit(_step, donate_argnums=(0, 1))
    else:
        return _step