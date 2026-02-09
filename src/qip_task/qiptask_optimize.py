#qiptask_optimize.py

import jax
import jax.numpy as jnp
import optax
import time
import wandb
import pickle
from qip_task.qiptask_generateplots import generate_plots
from qip_task.qiptask_generateplots import plot_ham_and_physical_rates,plot_coeffmatrix,plot_unitarymatrix

#we want to build the optimization of the RQPNN for a given QIP task
def optimize_params(params_train_dict,opt_state,take_step,sys_spec,params_amp_dict,
                    epochs=10000,plot_period=500,plot_periodically=True,log_wandb=True,
                 cost_tol=1e-7,log_inf_tol=1e-7,epoch_interval=500,inf_tol=10e-4,
                 approach_continuous=False):
    #initialization
    current_params=params_train_dict
    parameter_list=[]
    parameter_list.append(current_params)
    loss_tot_list=[]
    infidelity_list=[]
    inner_product_list=[]
    t0=time.time()

    #initial plotting
    plot_ham_and_physical_rates(sys_spec,params_train_dict,params_amp_dict)
    #plot_timevolve_logbasis_one(parameter_list[-1])
    plot_coeffmatrix(sys_spec,params_train_dict,params_amp_dict)
    #plot_unitarymatrix(sys_spec,params_train_dict,params_amp_dict)

    for epoch in range(1,epochs+1):
        print("epoch",epoch)
        #take step
        updated_params,fidelity_grad,aux,opt_state=take_step(current_params,opt_state)

        #update current parameters
        current_params=updated_params
        #unpack auxiliary info for logging   
        parameter_list.append(updated_params)
        loss_tot,infidelity,inner_prods,gate_duration,energy_list,grad_int_list,eff_bandwidth_list=aux
        loss_tot_list.append(loss_tot)
        infidelity_list.append(infidelity)
        inner_product_list.append(inner_prods)

        #time for epoch
        t1=time.time()
        #prepare info on improvement speed for stopping/logging
        if epoch>2*epoch_interval:
            loss_improve_speed=-(loss_tot_list[-1]-loss_tot_list[-1-epoch_interval])/epoch_interval
            log_inf_improve_speed=-(jnp.log(infidelity_list[-1])-jnp.log(infidelity_list[-1-epoch_interval]))/epoch_interval
        else:
            loss_improve_speed=cost_tol*2
            log_inf_improve_speed=inf_tol*2

        #log in wandb
        if log_wandb==True:
            wandb.log({"loss": loss_tot, 
                    "infidelity": infidelity,
                    "time_duaration_log": gate_duration,
                    "energy_delta_cav1": energy_list[0],
                    "energy_kappa1": energy_list[1],
                    "energy_delta_em1": energy_list[2],
                    "gradint_delta_cav": grad_int_list[0],
                    "gradint_kappa": grad_int_list[1],
                    "gradint_delta_em": grad_int_list[2],
                    "eff_bandwidth_delta_cav1": eff_bandwidth_list[0],
                    "eff_bandwidth_kappa1": eff_bandwidth_list[1],
                    "eff_bandwidth_delta_em1": eff_bandwidth_list[2],
                    "loss_improve_speed1": loss_improve_speed,
                    "log_inf_improve_speed1": log_inf_improve_speed,
                    "epoch-time": t1-t0})
        #restart timer
        t0=t1
        
        ###check stop criteria
        #check if infidelity has been reached
        if infidelity<inf_tol:
            print("infidelity",infidelity,"has reached below inf_tol",inf_tol)
            print("therefore we save the parameters and stop")
            #check convergence of solver 
            try:
                generate_plots(inner_product_list,parameter_list,sys_spec,params_amp_dict)
            except ValueError:
                print("could not generate plots due to ValueError")
            _log_summary(sys_spec,opt_state,params_amp_dict,infidelity_list,parameter_list,gate_duration,energy_list,grad_int_list,eff_bandwidth_list,
                    loss_improve_speed,log_inf_improve_speed,epoch,epochs)              
            if approach_continuous==True:
                print("we should check if we have reached convergence in continuous limit")
                return 0
            elif approach_continuous==False:
                return 1
        
        #check if loss improvement speed is below tolerance
        if loss_improve_speed<cost_tol:
            print("loss improvement speed",loss_improve_speed,"is below cost_tol",cost_tol)
            generate_plots(inner_product_list,parameter_list,sys_spec,params_amp_dict)
            _log_summary(sys_spec,opt_state,params_amp_dict,infidelity_list,parameter_list,gate_duration,energy_list,grad_int_list,eff_bandwidth_list,
                    loss_improve_speed,log_inf_improve_speed,epoch,epochs)              
            return 2
        #check if log infidelity improvement speed is below tolerance
        if log_inf_improve_speed<log_inf_tol:
            print("log infidelity improvement speed",log_inf_improve_speed,"is below log_inf_tol",log_inf_tol)
            print("therefore we save the parameters and stop")
            generate_plots(inner_product_list,parameter_list,sys_spec,params_amp_dict)
            _log_summary(sys_spec,opt_state,params_amp_dict,infidelity_list,parameter_list,gate_duration,energy_list,grad_int_list,eff_bandwidth_list,
                    loss_improve_speed,log_inf_improve_speed,epoch,epochs)
            return 3
        
        #periodic plotting
        if epoch%plot_period==0 and epoch>0 and plot_periodically==True:
            generate_plots(inner_product_list,parameter_list,sys_spec,params_amp_dict)
    #end of epoch loop
    print("exhausted all epochs")
    #check convergence of solver 
    _log_summary(sys_spec,opt_state,params_amp_dict,infidelity_list,parameter_list,gate_duration,energy_list,grad_int_list,eff_bandwidth_list,
                    loss_improve_speed,log_inf_improve_speed,epoch,epochs)

    return 4

#---
#log summary
def _log_summary(sys_spec,opt_state,params_amp_dict,infidelity_list,parameter_list,gate_duration,energy_list,grad_int_list,eff_bandwidth_list,
                    loss_improve_speed,log_inf_improve_speed,epoch,epochs,inf_tol=1e-3):
    run=wandb.run
    run.summary["infidelity"]=infidelity_list[-1]
    run.summary["time_duration"]=gate_duration
    if sys_spec.has_emitters==True:
        run.summary["mean_energy"]= (energy_list[0]+ energy_list[1]+energy_list[2])/3.0
        run.summary["mean_gradint_delta_cav"]=(grad_int_list[0]+grad_int_list[1]+grad_int_list[2])/3.0
        run.summary["mean_eff_bandwidth_delta_cav"]=(eff_bandwidth_list[0]+eff_bandwidth_list[1]+eff_bandwidth_list[2])/3.0
    else:
        run.summary["mean_energy"]= (energy_list[0]+ energy_list[1])/2.0
        run.summary["mean_gradint_delta_cav"]=(grad_int_list[0]+grad_int_list[1])/2.0
        run.summary["mean_eff_bandwidth_delta_cav"]=(eff_bandwidth_list[0]+eff_bandwidth_list[1])/2.0
    run.summary["loss_improve_speed"]=loss_improve_speed
    run.summary["log_inf_improve_speed"]=log_inf_improve_speed
    run.summary["infidelity_reached"]=infidelity_list[-1]<inf_tol
    run.summary["epochs_exhausted"]=epoch>epochs
    run.summary["epochs_used"]=epoch
    run.summary["final_parameters"]=parameter_list[-1]

    # 1)  Pickle both dicts to disk
    to_save = {
        "final_params": parameter_list[-1],
        "amp_dict":     params_amp_dict,
        "opt_state":    jax.device_get(opt_state)
    }
    for name, obj in to_save.items():
        fname = f"{name}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(obj, f)

    # 2)  Put both files in one artifact
    art = wandb.Artifact(
        name="optimized_state",          # ← anything   (one artifact, two files)
        type="model-params"
    )
    for fname in to_save.keys():
        art.add_file(f"{fname}.pkl")     # attach both pickles

    # 3)  Log the artifact (after all add_file calls)
    run.log_artifact(art)
