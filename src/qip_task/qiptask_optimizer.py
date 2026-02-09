#qiptask_optimizer.py
import dynamiqs as dq
import jax
import jax.numpy as jnp
import optax
from qip_task.qiptask_init_target_states import _generate_initial_and_target_states
from qip_task.qiptask_evolve import _build_evolve
from qip_task.qiptask_costfunc import build_loss
from qip_task.qiptask_stepper import _build_stepper
from qip_task.qiptask_optimize import optimize_params
import qip_task.qiptask_generate_params as qiptask_generate_params  #for generating trainable parameters
from qip_task.qiptask_build_masked_optimizer import make_adam_with_mask
from rqpnn.rqpnn_system import RQPNNSystem

class OptimizerQIPtask:
    def __init__(self, spec, dict_opti,params_train_dict,basis_cons_mat):
        #spec of RQPNN
        self.spec = spec
        #define task
        self.initial_states,target_states = _generate_initial_and_target_states(self.spec,dict_opti["QIPtrans"],basis_cons_mat)
        self.target_array=dq.asqarray(jnp.stack(dq.to_jax(target_states),axis=0))   #target states as array for easy vectorization
        self.initial_states=dq.asqarray(jnp.stack(dq.to_jax(self.initial_states),axis=0))   #initial states as array for easy vectorization
        ### ---- define evolution
        # physical evolution
        self.include_time_evolution=dict_opti["include_time_evolution"]
        self.include_mode_evolution=dict_opti["include_mode_evolution"]
        # numerics
        self.jit_stepper=dict_opti["jit_stepper"]   #whether to jit
        if dict_opti["solver"]=='Expm':
            self.solver=dq.method.Expm()
        else:
            raise ValueError(f"Solver {dict_opti['solver']} not recognized")
        
        ### ---- define cost function weights
        self.pen_dict={
            "weight_inf": dict_opti["cost"]["inf"],
            "weight_time_pen": dict_opti["cost"]["time"],
            "weight_bw_deltacav_pen": dict_opti["cost"]["bw_deltacav"],
            "weight_bw_kappa_pen": dict_opti["cost"]["bw_kappa"]
        }
        if spec.has_emitters:
            self.pen_dict["weight_bw_deltaem_pen"]=dict_opti["cost"]["bw_deltaem"]

        ### ---- define optimizer approach
        self.opt_spec=dict_opti["optimizer"]
        if self.opt_spec['type']=='adam':
            #mask
            mask_dict = self.opt_spec["mask"]
            print("mask",mask_dict)
            mask = {}
            for key in mask_dict.keys():
                if key == "delta_em" and spec.has_emitters==False:
                    print("No emitters in system -> do not  include delta_em")
                else:
                    mask[key] = mask_dict[key]
            self.optimizer,mask= make_adam_with_mask(
                    train_mask=mask,
                    params_tree= params_train_dict,
                    lr=self.opt_spec['parameters']['lr'],
                    b1=self.opt_spec['parameters']['b1'],
                    b2=self.opt_spec['parameters']['b2'])
        
        ### --- define optimization stop criteria
        self.train_spec=dict_opti["training"]
        

    ### --- main method: optimize
    def optimizeRQPNN(self,build_mode_switch_unitary, build_H_func, N_pieces, amp_spec_dict,params_train_dict):
        #build stepper
        params_amp_dict=qiptask_generate_params.generate_params_amp(amp_spec_dict,self.spec)
        H_func_para=build_H_func(N_pieces,params_amp_dict)
        tsave=jnp.linspace(0.0,N_pieces,N_pieces+1) #time points to save states
        evolve=self.build_evolve(build_mode_switch_unitary, H_func_para, tsave)
        loss_func = self.build_loss(params_amp_dict, evolve)
        take_step = self.build_stepper(loss_func)

        if amp_spec_dict["custom"]==True:
            run_id = amp_spec_dict["run_id"]
            final_params, amp_dict, opt_state, cfg_para,dict_system, dict_opti=qiptask_generate_params.load_params_and_amps(run_id)
            print("using saved opt_state")
            opt_state    = jax.tree_util.tree_map(jax.device_put, opt_state)
            take_step = self.build_stepper(loss_func)
        else:
            opt_state = self.optimizer.init(params_train_dict)#params_train_dict,opt_state,take_step,sys_spec,params_amp_dict,
        message= optimize_params(params_train_dict, opt_state, take_step, self.spec,params_amp_dict,
                                 epochs=self.train_spec["epochs"],plot_period=self.train_spec["plot_period"],
                                 plot_periodically=self.train_spec["plot_periodically"],
                                 log_wandb=self.train_spec["log_wandb"],
                                 cost_tol=self.train_spec["cost_tol"],
                                 log_inf_tol=self.train_spec["log_inf_tol"],
                                 epoch_interval=self.train_spec["epoch_interval"],
                                 inf_tol=self.train_spec["inf_tol"],
                                 approach_continuous=self.train_spec["approach_continuous"])
        if message==0:
            print("Consider checking if continuous limit was reached")
        elif message==1:
            print("fidelity goal was reached")
        elif message==2:
            print("cost function improved too slowly")
        elif message==3:
            print("infidelity improved too slowly")
        elif message==4:
            print("exhausted all epochs")



    ### --- internal methods
    #evolution
    def build_evolve(self,build_mode_switch_unitary,H_func_para,tsave):
        '''
        Builds the evolution function for the QIP task.
        inputs:
        - build_mode_switch_unitary: function (method from RQPNNsystem), function to build the mode switch unitary, of S_in
        - H_func_para: function (built using RQPNNsystem), function to build the Hamiltonian for time evolution, of train_params_dict
        - tsave: jax array, time points to save the states (we will specify this based on N_pieces)
        returns:
        - _evolve: function that takes in params_train_dict and returns the states at t_save times
        '''
        return _build_evolve(self.include_time_evolution,self.include_mode_evolution,self.initial_states,
        self.spec,build_mode_switch_unitary,H_func_para,tsave,self.solver)
    
    #cost function
    def build_loss(self,params_amp_dict,_evolve):
        '''
        inputs:
        - params_amp_dict: dict, dictionary of amplitude parameters for the physical rates
        - _evolve: function (made using build-evolve-method), function that takes in params_train_dict and returns the final states after evolution
        returns:
        - loss function: function that takes in params_train_dict and returns the loss and other diagnostic quantities
        '''
        return build_loss(self.spec,params_amp_dict,self.pen_dict,_evolve,self.target_array)

    #stepper
    def build_stepper(self,_loss_func):
        '''
        inputs:
        - _loss_func: function (made using build-loss-method), function that takes in params_train_dict and returns the loss and other diagnostic quantities
        returns:
        - stepper: function that takes in opt_state and params_train_dict and returns updated opt_state and params_train_dict
        '''
        return _build_stepper(_loss_func,self.optimizer,self.jit_stepper)