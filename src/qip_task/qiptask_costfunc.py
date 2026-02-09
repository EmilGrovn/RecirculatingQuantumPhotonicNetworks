#qiptask_costfunc.py
#cost function should be defined as function of trainable parameters

import dynamiqs as dq
import jax.numpy as jnp

#calculate smootness penalties on physical rates
#step 1: get physical rates from trainable parameters
from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
def compute_physical_rates(spec,params_train_dict,params_amp_dict):
    ham_properties = generate_ham_properties_from_params(spec, params_train_dict, params_amp_dict)
    tau_vec= ham_properties['tau']
    delta_cav_phys= ham_properties['delta_cav']/tau_vec
    kappa_phys= ham_properties['kappa']/tau_vec
    if spec.has_emitters:
        delta_em_phys= ham_properties['delta_em']/tau_vec
        return tau_vec, delta_cav_phys, kappa_phys, delta_em_phys
    else:
        return tau_vec, delta_cav_phys, kappa_phys
#step 2: compute loss penalties on physical rates
#time duration penalty
def compute_timeduration_penalty(tau_vec, weight_time_pen):
    gate_duration=jnp.sum(tau_vec) 
    return weight_time_pen*gate_duration,gate_duration
#bandwidth penalty
def compute_bandwidth_penalty(rate_phys_matrix, tau_vec,weight_smooth_pen,offset_rest):
    energy=mean_energy_alternative_int_array(rate_phys_matrix,tau_vec,offset_rest)
    grad_int=mean_grad_int_array(rate_phys_matrix,tau_vec,offset_rest)
    loss=weight_smooth_pen*grad_int/(1.0+energy)
    eff_bandwidth=jnp.sqrt(grad_int/energy)
    return loss, energy, grad_int, eff_bandwidth
#energy
def _energy_alternative_int(rate_vec,tau_vec,offset_rest):
    tau_vec_pad=jnp.pad(tau_vec, 1,mode="constant", constant_values=0.0)
    #rate_vec_pad=jnp.pad(rate_vec, 1, mode="constant", constant_values=offset_rest)-offset_rest
    delta_tau_vec=(tau_vec_pad[1:]+tau_vec_pad[:-1])*0.5
    h_vec=(delta_tau_vec[:-1]+delta_tau_vec[1:])*0.5
    rate_vec_zero=rate_vec-offset_rest
    #avg_rate_vec=(rate_vec_pad[1:]+rate_vec_pad[:-1])*0.5
    return jnp.sum(jnp.abs(rate_vec_zero)**2*h_vec)
def mean_energy_alternative_int_array(rate_array,tau_vec,offset_rest):
    return jnp.mean(jnp.array([_energy_alternative_int(rate_vec,tau_vec,offset_rest) for rate_vec in rate_array]))
#grad_int
def _grad_int(rate_vec,tau_vec,offset_rest):
    tau_vec_pad=jnp.pad(tau_vec, 1,mode="constant", constant_values=0.0)
    delta_tau_vec=(tau_vec_pad[1:]+tau_vec_pad[:-1])*0.5
    return jnp.sum(jnp.abs(jnp.diff(jnp.pad(rate_vec,1, mode="constant", constant_values=offset_rest)))**2/delta_tau_vec)
def mean_grad_int_array(rate_array,tau_vec,offset_rest):
    return jnp.mean(jnp.array([_grad_int(rate_vec,tau_vec,offset_rest) for rate_vec in rate_array]))

### calculate coherent infidelity
def _coherent_infidelity(_evolve,target_array,params_train_dict):
    states=_evolve(params_train_dict)
    inner_products=dq.to_jax(target_array.dag()@states)
    print("inner products",inner_products, inner_products.shape)
    coherent_infidelity=1-jnp.abs(jnp.mean(inner_products))**2
    #return coherent_infidelity, inner_products
    return coherent_infidelity, inner_products

### calculate total loss
def build_loss(spec,params_amp_dict,pen_dict,_evolve,target_array):
    def _loss(params_train_dict):
        params_eff=handle_kappa_for_forward(spec, params_train_dict)
        #infidelity
        infidelity,inner_prods=_coherent_infidelity(_evolve,target_array,params_eff)
        loss_tot=jnp.log(infidelity)*pen_dict['weight_inf']#initialize loss tot
        #penalty on rates
        rate_tuple=compute_physical_rates(spec,params_eff,params_amp_dict)
        #tau
        tau_vec=rate_tuple[0]
        loss_time, gate_duration=compute_timeduration_penalty(tau_vec, pen_dict['weight_time_pen'])
        loss_tot+=loss_time
        #physical rates
        #delta_cav
        delta_cav=rate_tuple[1]
        loss_delta_cav, energy_delta_cav, grad_int_delta_cav, eff_bandwidth_delta_cav=compute_bandwidth_penalty(delta_cav,tau_vec,pen_dict['weight_bw_deltacav_pen'],0.0)
        loss_tot+=loss_delta_cav
        #kappa
        kappa=rate_tuple[2]
        loss_kappa, energy_kappa, grad_int_kappa, eff_bandwidth_kappa=compute_bandwidth_penalty(kappa,tau_vec,pen_dict['weight_bw_kappa_pen'],0.0)
        loss_tot+=loss_kappa
        #delta_em
        if spec.has_emitters:
            delta_em=rate_tuple[3]
            loss_delta_em, energy_delta_em, grad_int_delta_em, eff_bandwidth_delta_em=compute_bandwidth_penalty(delta_em,tau_vec,pen_dict['weight_bw_deltaem_pen'],params_amp_dict["delta_em_rest"])
            loss_tot+=loss_delta_em
            return loss_tot,(loss_tot,infidelity,inner_prods,gate_duration,
                [energy_delta_cav, energy_kappa, energy_delta_em],
                [grad_int_delta_cav, grad_int_kappa, grad_int_delta_em],
                [eff_bandwidth_delta_cav, eff_bandwidth_kappa, eff_bandwidth_delta_em])
        else:
            return loss_tot,(loss_tot,infidelity,inner_prods,gate_duration,
                [energy_delta_cav, energy_kappa,0.0],
                [grad_int_delta_cav, grad_int_kappa,0.0],
                [eff_bandwidth_delta_cav, eff_bandwidth_kappa,0.0])
    
    return _loss

#special case for kappa (this function was generated using AI - tested in test_qip_loss)
def handle_kappa_for_forward(spec, params):
    k = params["kappa"]  # expected shape (d,) for the trainable vector

    if spec.M_cav == 2:
        # Build 2 x d with identical rows for the forward pass
        if k.ndim == 1:                       # (d,) -> (1,d) -> (2,d)
            k2 = jnp.broadcast_to(k[None, :], (2, k.shape[-1]))
        elif k.ndim == 2 and k.shape[0] == 1: # (1,d) -> (2,d)
            k2 = jnp.broadcast_to(k, (2, k.shape[-1]))
        elif k.ndim == 2 and k.shape[0] == 2: # already (2,d)
            k2 = k
        else:
            raise ValueError(f"Unexpected kappa shape: {k.shape}")
    else:
        # Use as-is (or generalize to M_cav > 2 similarly)
        k2 = k

    # Build an *effective* params dict for the forward only.
    # This does NOT change the params PyTree shape seen by the optimizer.
    params_eff = {**params, "kappa": k2}
    return params_eff