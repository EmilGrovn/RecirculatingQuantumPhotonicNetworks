#qiptask_generate_params.py
#in this script we want to generate the datastructure
#of trainable parameters and amplitude parameters
#which you put into hamiltonian

import jax.numpy as jnp
import jax

import wandb, os, pickle, pathlib
def load_params_and_amps(run_id,entity   = "SPECIFY WANDB ID",project  = "SPECIFY PROJECT"):
    # --- 1. Connect to the public REST API (no wandb.init() needed) -------------
    api      = wandb.Api()

    run      = api.run(f"{entity}/{project}/{run_id}")


    final_params = None
    amp_dict     = None

    # the run logged exactly one artifact called “optimized_state”
    for art in run.logged_artifacts():
        if art.name.startswith("optimized_state"):
            art_dir = pathlib.Path(art.download())        # tmp dir with both files
            with open(art_dir / "final_params.pkl", "rb") as f:
                final_params = pickle.load(f)             # ← numpy array / list / …
            with open(art_dir / "amp_dict.pkl", "rb") as f:
                amp_dict = pickle.load(f)                 # ← **plain Python dict**
            # NEW: load optimizer state
            with open(art_dir / "opt_state.pkl", "rb") as f:
                opt_state = pickle.load(f)
            
            break

    print("final parameters:", type(final_params), "\n", final_params)
    print("\namp_dict:", type(amp_dict), "\n", amp_dict)
    #create slim para builder
    cfg = run.config
    cfg_para= cfg["parameterization_spec"] 
    dict_system = cfg["system_spec"]
    dict_opti=cfg["optimization_spec"]
    cfg_para["use saved parameter dict"]=True
    return final_params, amp_dict, opt_state, cfg_para,dict_system, dict_opti

### generate trainable parameters
def generate_params_train(spec,train_para_spec):
    if train_para_spec["custom"]==True:
        run_id = train_para_spec["run_id"]
        raise ValueError("To use custom==True you must modify entity and project in load_params_and_amps")
            
        #final_params, amp_dict, opt_state, cfg_para,dict_system, dict_opti=load_params_and_amps(run_id)
        #return final_params
    elif train_para_spec["custom"]==False:
        return initialize_params_train(train_para_spec,spec)

def initialize_params_train(train_para_spec,spec):
    params_train_dict={}
    N_steps=train_para_spec["N_steps"]
    seed=train_para_spec["key_seed"]
    base = jax.random.PRNGKey(int(seed))
    k_tau   = jax.random.fold_in(base, 0)
    k_dc    = jax.random.fold_in(base, 1)
    k_kappa = jax.random.fold_in(base, 2)
    k_de    = jax.random.fold_in(base, 3)
    k_coeff = jax.random.fold_in(base, 4)
    k_Sin   = jax.random.fold_in(base, 5)

    #tau

    params_train_dict["tau"]=initialize_tau_vec(N_steps,k_tau,
                                                train_para_spec["tau"]["offset"],train_para_spec["tau"]["noise"]) #initialize tau parameters to zero

    #delta cavity
    params_train_dict["delta_cav"]=initilize_rateparam_mat(spec.M_cav,N_steps,k_dc,
                                         train_para_spec["delta_cav"]["offset"],train_para_spec["delta_cav"]["noise"])

    if spec.M_cav>2:
        params_train_dict["kappa"]=initilize_rateparam_mat(spec.M_cav,N_steps,k_kappa,
                                         train_para_spec["kappa"]["offset"],train_para_spec["kappa"]["noise"])
    else:
        print("identical kappas for both cavities!!")
        params_train_dict["kappa"]=initilize_rateparam_mat(1,N_steps,k_dc,
                                         train_para_spec["kappa"]["offset"],train_para_spec["delta_cav"]["noise"])
        print(params_train_dict["kappa"])
    #kappa (cavity coupling)


    #delta emitters
    if spec.has_emitters:
        params_train_dict["delta_em"]=initilize_rateparam_mat(spec.M_cav,N_steps,k_de,
                                             train_para_spec["delta_em"]["offset"],train_para_spec["delta_em"]["noise"])

    #coefficient matrix
    params_train_dict["coeff_vec"]=initilize_coeff_vec(spec.M_cav,k_coeff,
                                         train_para_spec["coeff_vec"]["offset"],train_para_spec["coeff_vec"]["noise"])

    #S_in matrix NOTE: S_in has not been used in the RQPN evolution and results
    params_train_dict["S_in_Rmat"]=initilize_S_inTrans_mat(spec.M_cav,k_Sin,
                                         train_para_spec["S_in_Rmat"]["offset"],train_para_spec["S_in_Rmat"]["noise"])

    return params_train_dict

def initialize_tau_vec(N_steps,subkey,offset,noise):
    return jnp.ones(N_steps,dtype=jnp.float32)*offset+jax.random.normal(subkey, shape=(N_steps,),dtype=jnp.float32) * noise

def initilize_rateparam_mat(M_cav,N_steps,subkey,offset,noise):
    #NOTE: overvej om vi vil lave individualiserede offsets til hver kavitet eller emitter
    return jnp.ones((M_cav,N_steps),dtype=jnp.float32)*offset+jax.random.normal(subkey, shape=(M_cav,N_steps,),dtype=jnp.float32) * noise

def initilize_coeff_vec(M_cav,subkey,offset,noise):
    N_coeff=int(M_cav*(M_cav-1)/2)
    return jnp.ones(N_coeff,dtype=jnp.float32)*offset+jax.random.normal(subkey, shape=(N_coeff,),dtype=jnp.float32) * noise

def initilize_S_inTrans_mat(M_cav,subkey,offset,noise):
    N_coeff=int(M_cav*(M_cav-1)/2)
    return jnp.ones((N_coeff),dtype=jnp.float32)*offset+jax.random.normal(subkey, shape=(N_coeff,),dtype=jnp.float32) * noise

### generate amplitude parameters
def generate_params_amp(amp_para_spec,spec):
    if amp_para_spec["custom"]==True:
        run_id = amp_para_spec["run_id"]
        raise ValueError("To use custom==True you must modify entity and project in load_params_and_amps")
        # final_params, amp_dict, opt_state, cfg_para,dict_system, dict_opti=load_params_and_amps(run_id)
        # return amp_dict
    elif amp_para_spec["custom"]==False:
        return initialize_params_amp(amp_para_spec,spec)

def initialize_params_amp(amp_para_spec,spec):
    params_amp_dict={}
    N_steps=amp_para_spec["N_steps"]
    #tau
    params_amp_dict["max_tau"]=amp_para_spec["max_tau"]*jnp.ones(N_steps)
    params_amp_dict["min_tau"]=amp_para_spec["min_tau"]*jnp.ones(N_steps)

    #delta cavity
    shape=amp_para_spec["delta_cav"]["shape"]
    amp=amp_para_spec["delta_cav"]["amp_physical"]*amp_para_spec["min_tau"]
    ramp_up_speed=amp_para_spec["delta_cav"]["ramp_up_speed_physical"]
    params_amp_dict["delta_cav"]=_generate_amplitude_segment(N_steps,amp,
                                            shape,ramp_up_speed,amp_para_spec["min_tau"],
                                            amp_para_spec["max_tau"])
    #kappa
    shape=amp_para_spec["kappa"]["shape"]
    amp=amp_para_spec["kappa"]["amp_physical"]*amp_para_spec["min_tau"]
    ramp_up_speed=amp_para_spec["kappa"]["ramp_up_speed_physical"]
    params_amp_dict["kappa"]=_generate_amplitude_segment(N_steps,amp,
                                            shape,ramp_up_speed,amp_para_spec["min_tau"],
                                            amp_para_spec["max_tau"])
    #delta emitters
    if spec.has_emitters:
        shape=amp_para_spec["delta_em"]["shape"]
        amp=amp_para_spec["delta_em"]["amp_physical"]*amp_para_spec["min_tau"]
        ramp_up_speed=amp_para_spec["delta_em"]["ramp_up_speed_physical"]
        params_amp_dict["delta_em"]=_generate_amplitude_segment(N_steps,amp,
                                                shape,ramp_up_speed,amp_para_spec["min_tau"],
                                            amp_para_spec["max_tau"])
        offset_shape=amp_para_spec["delta_em"]["offset_shape"]
        offset_size=amp_para_spec["delta_em"]["offset_size_physical"]*amp_para_spec["max_tau"]
        #NOTE: We have only used an amplitude offset of zero in our results 
        params_amp_dict["delta_em_offset"]=_generate_offset_segment(N_steps,offset_size,
                                                offset_shape,ramp_up_speed,amp_para_spec["min_tau"],
                                                amp_para_spec["max_tau"])
        params_amp_dict["delta_em_rest"]=offset_size
    return params_amp_dict

def _generate_amplitude_segment(N_steps,amp,shape,ramp_up_speed,tau_min,tau_max):
    '''
    amp_max: maximum amplitude
    amp_t: amplitude type
    pieces: number of pieces in segment (integer)
    generate the amplitude of a given segment
    returns: 1D jax array of shape (pieces,) containing the amplitudes for the segment
    '''
    if shape=="uniform":
        return jnp.ones((N_steps))*amp
    #NOTE: We have only used "uniform" in our results
    elif shape=="ramp":     
        #only meant to guide training -> ultimately regularization during optimization ensures smoothness
        const_amp=jnp.ones((N_steps))*amp
        up_amp=(tau_min+tau_max)/2*ramp_up_speed*tau_min*jnp.arange(1,N_steps+1,1)
        up_amp+=-0.5*(tau_min+tau_max)/2*ramp_up_speed*tau_min#only half-step initially
        down_amp=jnp.flip(up_amp)
        return jnp.min(jnp.stack([up_amp,const_amp,down_amp]), axis=0)

def _generate_offset_segment(N_steps,offset,shape,ramp_up_speed,tau_min,tau_max):
    if shape=="uniform":
        return jnp.ones((N_steps),dtype=jnp.float32)*offset
    elif shape=="ramp":
        return jnp.ones((N_steps),dtype=jnp.float32)*offset-_generate_amplitude_segment(N_steps,offset,shape,ramp_up_speed,tau_min,tau_max)