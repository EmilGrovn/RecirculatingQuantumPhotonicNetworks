#RQPNN_workflow.py
### ---------------------- ###
### PURPOSE OF THE CODE: setup QPNN workflow
### ---------------------- ###
### --------------------------------------------------------------------------------- ###
### ---------------------- ###
# SECTION 1: Import libraries and modules
### ---------------------- ###
print("RQPNN_workflow.py: starting!")
## module for logging
import wandb
import os#to access api key

import jax

###Import custom modules
import rqpnn.rqpnn_spec as rqpnn_spec        #for defining rqpnn-specs
import rqpnn.rqpnn_system as rqpnn_system    #for defining rqpnn
import qip_task.qiptask_optimizer as qiptask_optimizer  #for optimizing rqpnn for a given QIP task
import qip_task.qiptask_generate_params as qiptask_generate_params  #for generating trainable parameters
print("RQPNN_workflow.py: imports done!")
### --------------------------------------------------------------------------------- ###
### ---------------------- ###
# SECTION 2: Define RQPNN optimization workflow class
### ---------------------- ###
#Class definition
class RQPNNWorkflow:
    '''
    Contains all parameters and methods for training an RQPNN for a given QIP task

    parameters:
    -----------
    dict_sysSpec: dictionary
        defines the spec of the RQPNN
    dict_para: dictionary
        specifies the (optionally trainable) parameters of the system
    dict_amp:
        specifies the always fixed amplitude parameters of the system
    dict_opti: dictionary
        specifies the optimization to perform and optimizer parameters
    wandb_specs: dictionary
        specifies the wandb project and run name

    methods:
    -----------
    External method:
    train: trains the model

    Internal methods, are used when initializing the class:
    _initialize_wandb: initializes wandb for logging
    '''
    def __init__(self, dict_sys, dict_para_spec, dict_amp_spec, dict_opti, wandb_specs):
        print("Initializing RQPNN workflow...")
        #initialize specifications
        self.dict_sys = dict_sys
        self.dict_para_spec = dict_para_spec
        self.dict_amp_spec = dict_amp_spec
        self.dict_opti = dict_opti
        self.wandb_specs = wandb_specs

        #initialize wandb for logging
        self._initialize_wandb()
        print("run func!")
        #define RQPNN specs
        print("N_exc",dict_sys["N_exc_list"],type(dict_sys["N_exc_list"]))
        self.rqpnnSpec=rqpnn_spec.RQPNNSpec(dict_sys["M_cav"],dict_sys["M_em"],tuple(dict_sys["N_exc_list"]))
        #instantiate RQPNN system
        self.rqpnnSystem=rqpnn_system.RQPNNSystem(self.rqpnnSpec)
        #instantiate RQPNN optimizer
        params_train_dict=qiptask_generate_params.generate_params_train(self.rqpnnSpec,self.dict_para_spec)
        self.optimizerQIPtask=qiptask_optimizer.OptimizerQIPtask(self.rqpnnSpec,self.dict_opti,params_train_dict,self.rqpnnSystem.basis_conserved_mat)

    ### ----------------  Initialize wandb ---------------- ###
    def _initialize_wandb(self):
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            print("Must be running interactively then")
            #raise ValueError("WANDB_API_KEY environment variable not set.")
        wandb.login(key=api_key)#using my api key in a non-interactive environment
        wandb.init(
            project=self.wandb_specs['project'],  #name of the project
            config={"task": self.wandb_specs['group']+"_"+self.wandb_specs['sweep_name']},                          # Configuration initialized
            group=self.wandb_specs['group'], #group name of the project
            name=self.wandb_specs['group']+"_"+self.wandb_specs['sweep_name'],                #name of the run
        )

        wandb.config.update({                   #update configuration
                'system_spec': self.dict_sys,
                'parameterization_spec': self.dict_para_spec,
                'amplitude_spec': self.dict_amp_spec,
                'optimization_spec': self.dict_opti,
                'device'      : jax.devices()[0]#, #"CPU",#self.device,
                #'wandb_specs' : self.wandb_specs   
            }
        )
    

    ### ----------------  Run the training ---------------- ###
    def train(self):
        print("Starting training...")
        #generate parameterization and amplitude parameters
        params_amp_dict=qiptask_generate_params.generate_params_amp(self.dict_amp_spec,self.rqpnnSpec)
        params_train_dict=qiptask_generate_params.generate_params_train(self.rqpnnSpec,self.dict_para_spec)
        N_pieces=self.dict_para_spec["N_steps"]
        
        #optimize the RQPNN for the given QIP task
        self.optimizerQIPtask.optimizeRQPNN(self.rqpnnSystem.build_mode_switch_unitary, 
                                        self.rqpnnSystem._build_H_func, 
                                        N_pieces, self.dict_amp_spec,params_train_dict)
        print("Training finished. See wandb for results")
        wandb.finish()
        
    ### ---------------- END OF CLASS --------------------- ###

### --------------------------------------------------------------------------------- ###
