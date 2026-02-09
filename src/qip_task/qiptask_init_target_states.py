#qiptask_init_target_states.py
import jax.numpy as jnp

from rqpnn.rqpnn_stategeneration_modetrans import _generate_product_number_state_using_basis

## ---- generate initial and target states
def _generate_initial_and_target_states(spec,transformation_type,basis_conserved_mat):
    if transformation_type=="Pi/4-Phase gate 1qubit":
        assert spec.M_cav==2, spec.N_exc_list==(1,)
        initial_states,target_states=generate_single_qubit_piquarter_phase_gate_init_target_states(spec,basis_conserved_mat)
    elif transformation_type=="Hadamard gate 1qubit":
        assert spec.M_cav==2, spec.N_exc_list==(1,)
        initial_states,target_states=generate_single_qubit_hadamard_gate_init_target_states(spec,basis_conserved_mat)
    elif transformation_type=="CZ gate 2qubit":
        assert spec.M_cav==4, spec.N_exc_list==(2,)
        initial_states,target_states=generate_two_qubit_csign_gate_init_target_states(spec,basis_conserved_mat) #target states
    elif transformation_type=="CZ gate 2qubit qutrit":
        assert spec.M_cav==4, spec.N_exc_list==(1,2)
        assert spec.M_em==0
        initial_states,target_states=generate_two_qubit_csign_gate_qutrit_init_target_states(spec,basis_conserved_mat) 
    elif transformation_type=="CZ gate 5m qutrit":
        assert spec.M_cav==5, spec.N_exc_list==(2,)
        assert spec.M_em==0
        initial_states,target_states=generate_two_qubit_csign_gate_5m_qutrit_init_target_states(spec,basis_conserved_mat) 

    elif transformation_type=="CNot gate 2qubit":
        assert spec.M_cav==4, spec.N_exc_list==(2,)
        initial_states,target_states=generate_two_qubit_cnot_gate_init_target_states(spec,basis_conserved_mat) #target states
    elif transformation_type=="BSA 2qubit":
        assert spec.M_cav==4, spec.N_exc_list==(2,)
        initial_states,target_states=generate_two_qubit_BSA_init_target_states(spec,basis_conserved_mat) #target states
    elif transformation_type=="GHZ 3qubit":
        assert spec.M_cav==6, spec.N_exc_list==(3,)
        initial_states,target_states=generate_three_qubit_GHZ_init_target_states(spec,basis_conserved_mat) #target states
    elif transformation_type=="Tofolli gate 3qubit":
        assert spec.M_cav==6, spec.N_exc_list==(3,)
        initial_states,target_states=generate_three_qubit_tofolli_gate_init_target_states(spec,basis_conserved_mat)
    elif transformation_type=="Autoencoder 4qubit":
        assert spec.M_cav==6, spec.N_exc_list==(4,)
        initial_states,target_states=generate_four_qubit_autoencoder_init_target_states(spec,basis_conserved_mat)
    elif transformation_type=="1loss 3cav repeater":
        assert spec.M_cav==3, spec.N_exc_list==(4,5)
        initial_states,target_states=generate_1loss_3cav_repeater_init_target_states(basis_conserved_mat)
    elif transformation_type=="1loss m2 3cav repeater":
        assert spec.M_cav==3, spec.N_exc_list==(4,5)
        initial_states,target_states=generate_1loss_m2_3cav_repeater_init_target_states(basis_conserved_mat)
    elif transformation_type=="2loss 4cav coherent repeater":
        assert spec.M_cav==4, spec.N_exc_list==(5,6)
        initial_states,target_states=generate_2loss_4cav_coherent_repeater_init_target_states(basis_conserved_mat)
    elif transformation_type=="1loss emitter repeater":
        initial_states,target_states=generate_1loss_em_repeater_init_target_states(basis_conserved_mat)
    elif transformation_type=="2loss emitter coherent repeater":
        assert spec.M_cav==2 and spec.M_em==2 and spec.N_exc_list==(5,6)
        initial_states,target_states=generate_2loss_em_coherent_repeater_init_target_states(basis_conserved_mat)
    else:
        raise ValueError("transformation type not recognized")
    return initial_states,target_states

#dual-rail logical states
def _generate_logical_dualrail_basisstates(spec,basis_conserved_mat,encoding = "dual-rail"):
    if encoding == "dual-rail":
        assert spec.M_cav % 2 == 0, "Dual-rail encoding is only supported for mod-2 cavities"

        n_qubits = int(spec.M_cav/2)
        logical_states = []
        log0 = [1,0]
        log1 = [0,1]
        logical_states.append(log0)
        logical_states.append(log1)
        for i in range(1,n_qubits):
            logical_states_new = []
            for j in range(len(logical_states)):
                add_log0 = logical_states[j]+log0
                add_log1 = logical_states[j]+log1
                logical_states_new.append(add_log0)
                logical_states_new.append(add_log1)
            logical_states = logical_states_new
        #print(logical_states)
        psi_logical_basis = []
        for i,logical_state in enumerate(logical_states):
            if spec.M_em==spec.M_cav:
                logical_state+=[0]*spec.M_em
            #TODO: import appropriate funtion!!!
            psi_i_conserved = _generate_product_number_state_using_basis(logical_state,basis_conserved_mat)#no longer tuple input!
            psi_logical_basis.append(psi_i_conserved)
        return psi_logical_basis

### GENERATE SPECIFIC GATES --------------------------------------------
#single qubit phase gate
def generate_single_qubit_piquarter_phase_gate_init_target_states(spec,basis_conserved_mat):
    '''
    Generate input-target states for a single qubit Pi/4 gate (Pi/4-phase shift)
    For dual rail encoding
    '''
    #assert self.n_cavities==0, "Single qubit phase gate is only supported for 2 cavities"
    dual_rail_logical_states = _generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi0=dual_rail_logical_states[0]
    psi1=dual_rail_logical_states[1]
    target_states = [psi0,jnp.exp(1j*jnp.pi/4)*psi1]
    return dual_rail_logical_states, target_states
#single qubit hadamard gate
def generate_single_qubit_hadamard_gate_init_target_states(spec,basis_conserved_mat):
    '''
    Generate input-target states for a single qubit hadamard gate
    For dual rail encoding
    '''
    #assert self.n_cavities==0, "Single qubit phase gate is only supported for 2 cavities"
    dual_rail_logical_states = _generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi0=dual_rail_logical_states[0]
    psi1=dual_rail_logical_states[1]
    target_states = [(psi0+psi1)/jnp.sqrt(2),(psi0-psi1)/jnp.sqrt(2)]
    return dual_rail_logical_states, target_states

#two qubit c-phase gate
def generate_two_qubit_csign_gate_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a two qubit c-sign gate
    For dual rail encoding
    """
    dual_rail_logical_states = _generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi00=dual_rail_logical_states[0]
    psi01=dual_rail_logical_states[1]
    psi10=dual_rail_logical_states[2]
    psi11=dual_rail_logical_states[3]
    target_states = [psi00,psi01,psi10,-psi11]
    return dual_rail_logical_states, target_states
#two qubit c-phase gate qutrit
def generate_two_qubit_csign_gate_qutrit_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a two qubit c-sign gate
    For dual rail encoding
    """
    dual_rail_logical_states = _generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi00=dual_rail_logical_states[0]
    psi01=dual_rail_logical_states[1]
    psi10=dual_rail_logical_states[2]
    psi11=dual_rail_logical_states[3]
    #
    psi1000=_generate_product_number_state_using_basis([1,0,0,0],basis_conserved_mat)
    psi0100=_generate_product_number_state_using_basis([0,1,0,0],basis_conserved_mat)
    init_states=[psi00,psi01,psi10,psi11,psi1000,psi0100]
    target_states = [psi00,psi01,psi10,-psi11,psi1000,psi0100]
    return init_states, target_states
#two qubit c-phase gate qutrit
def generate_two_qubit_csign_gate_5m_qutrit_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a two qubit c-sign gate
    For dual rail encoding
    """
    psi00=_generate_product_number_state_using_basis([1,0,1,0,0],basis_conserved_mat)
    psi01=_generate_product_number_state_using_basis([1,0,0,1,0],basis_conserved_mat)
    psi10=_generate_product_number_state_using_basis([0,1,1,0,0],basis_conserved_mat)
    psi11=_generate_product_number_state_using_basis([0,1,0,1,0],basis_conserved_mat)
    #
    psi1000=_generate_product_number_state_using_basis([1,0,0,0,1],basis_conserved_mat)
    psi0100=_generate_product_number_state_using_basis([0,1,0,0,1],basis_conserved_mat)
    init_states=[psi00,psi01,psi10,psi11,psi1000,psi0100]
    target_states = [psi00,psi01,psi10,-psi11,psi1000,psi0100]
    return init_states, target_states
#two qubit cnot gate
def generate_two_qubit_cnot_gate_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a two qubit cnot gate
    For dual rail encoding
    """
    dual_rail_logical_states = _generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi00=dual_rail_logical_states[0]
    psi01=dual_rail_logical_states[1]
    psi10=dual_rail_logical_states[2]
    psi11=dual_rail_logical_states[3]
    target_states = [psi00,psi01,psi11,psi10]
    return dual_rail_logical_states, target_states
## generate specific quantum information processors
#two qubit: bell state analyzer (BSA)
def generate_two_qubit_BSA_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a two qubit Bell state analyzer (BSA)
    For dual rail encoding
    """
    dual_rail_logical_states=_generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi00=dual_rail_logical_states[0]
    psi01=dual_rail_logical_states[1]
    psi10=dual_rail_logical_states[2]
    psi11=dual_rail_logical_states[3]
    init_states = [(psi00+psi11)/jnp.sqrt(2),(psi01+psi10)/jnp.sqrt(2),(psi00-psi11)/jnp.sqrt(2),(psi01-psi10)/jnp.sqrt(2)]
    return init_states, dual_rail_logical_states
# #three qubit: GHZ state generator
def generate_three_qubit_GHZ_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a three qubit GHZ state generator
    For dual rail encoding
    """
    dual_rail_logical_states=_generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi000=dual_rail_logical_states[0]
    # psi001=self.psi_logical_basis[1]
    # psi010=self.psi_logical_basis[2]
    # psi011=self.psi_logical_basis[3]
    # psi100=self.psi_logical_basis[4]
    # psi101=self.psi_logical_basis[5]
    # psi110=self.psi_logical_basis[6]
    psi111=dual_rail_logical_states[7]
    target_state = [(psi000+psi111)/jnp.sqrt(2)]
    return [dual_rail_logical_states[0]], target_state
# ###
def generate_three_qubit_tofolli_gate_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a three qubit Toffoli gate
    For dual rail encoding
    """
    dual_rail_logical_states=_generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi000=dual_rail_logical_states[0]
    psi001=dual_rail_logical_states[1]
    psi010=dual_rail_logical_states[2]
    psi011=dual_rail_logical_states[3]
    psi100=dual_rail_logical_states[4]
    psi101=dual_rail_logical_states[5]
    psi110=dual_rail_logical_states[6]
    psi111=dual_rail_logical_states[7]
    target_state = [psi000,psi001,psi010,psi011,psi100,psi101,psi111,psi110]
    return dual_rail_logical_states, target_state
#4 qubit
def generate_four_qubit_autoencoder_init_target_states(spec,basis_conserved_mat):
    """
    Generate input-target states for a four qubit autoencoder
    For dual rail encoding
    """
    dual_rail_logical_states=_generate_logical_dualrail_basisstates(spec,basis_conserved_mat)
    psi0000=dual_rail_logical_states[0]
    psi0001=dual_rail_logical_states[1]
    # psi0010=self.psi_logical_basis[2]
    psi0011=dual_rail_logical_states[3]
    # psi0100=self.psi_logical_basis[4]
    # psi0101=self.psi_logical_basis[5]
    # psi0110=self.psi_logical_basis[6]
    # psi0111=self.psi_logical_basis[7]
    # psi1000=self.psi_logical_basis[8]
    # psi1001=self.psi_logical_basis[9]
    # psi1010=self.psi_logical_basis[10]
    # psi1011=self.psi_logical_basis[11]
    psi1100=dual_rail_logical_states[12]
    # psi1101=self.psi_logical_basis[13]
    # psi1110=self.psi_logical_basis[14]
    # psi1111=self.psi_logical_basis[15]
    initial_states = [psi0011,psi1100] 
    target_states = [psi0000,psi0001]
    return initial_states,target_states
#one way repeater
#one mode loss
def generate_1loss_3cav_repeater_init_target_states(basis_conserved_mat):
    """
    Generate input-target states for one-way repeater correcting photon loss on one mode
    For two-mode 4-photon encoding, using photon ancillas
    """
    #no loss
    init_0_noloss=(_generate_product_number_state_using_basis([4,0,1],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,1],basis_conserved_mat))/jnp.sqrt(2)
    target_0_noloss=init_0_noloss
    init_1_noloss=_generate_product_number_state_using_basis([2,2,1],basis_conserved_mat)
    target_1_noloss=init_1_noloss
    #loss mode 1 m1loss
    init_0_m1loss=_generate_product_number_state_using_basis([3,0,1],basis_conserved_mat)
    target_0_m1loss=(_generate_product_number_state_using_basis([4,0,0],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,0],basis_conserved_mat))/jnp.sqrt(2)
    init_1_m1loss=_generate_product_number_state_using_basis([1,2,1],basis_conserved_mat)
    target_1_m1loss=_generate_product_number_state_using_basis([2,2,0],basis_conserved_mat)
    #loss mode 2 m2loss do nothing
    init_0_m2loss=_generate_product_number_state_using_basis([0,3,1],basis_conserved_mat)
    target_0_m2loss=init_0_m2loss
    init_1_m2loss=_generate_product_number_state_using_basis([2,1,1],basis_conserved_mat)#NOTE: we have changed this task too!
    target_1_m2loss=init_1_m2loss
    initial_state=[init_0_noloss,init_1_noloss,init_0_m1loss,init_1_m1loss,init_0_m2loss,init_1_m2loss]
    target_state=[target_0_noloss,target_1_noloss,target_0_m1loss,target_1_m1loss,target_0_m2loss,target_1_m2loss]
    return initial_state,target_state
#one mode loss mode 2
def generate_1loss_m2_3cav_repeater_init_target_states(basis_conserved_mat):
    """
    Generate input-target states for one-way repeater correcting photon loss on one mode
    For two-mode 4-photon encoding, using photon ancillas
    """
    #no loss
    init_0_noloss=(_generate_product_number_state_using_basis([4,0,1],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,1],basis_conserved_mat))/jnp.sqrt(2)
    target_0_noloss=init_0_noloss
    init_1_noloss=_generate_product_number_state_using_basis([2,2,1],basis_conserved_mat)
    target_1_noloss=init_1_noloss
    #loss mode 2 m2loss do nothing
    init_0_m2loss=_generate_product_number_state_using_basis([0,3,1],basis_conserved_mat)
    target_0_m2loss=(_generate_product_number_state_using_basis([4,0,0],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,0],basis_conserved_mat))/jnp.sqrt(2)
    init_1_m2loss=_generate_product_number_state_using_basis([2,1,1],basis_conserved_mat)
    target_1_m2loss=_generate_product_number_state_using_basis([2,2,0],basis_conserved_mat)
    initial_state=[init_0_noloss,init_1_noloss,init_0_m2loss,init_1_m2loss]
    target_state=[target_0_noloss,target_1_noloss,target_0_m2loss,target_1_m2loss]
    return initial_state,target_state
#two mode loss, coherent
def generate_2loss_4cav_coherent_repeater_init_target_states(basis_conserved_mat):
    """
    Generate input-target states for one-way repeater correcting photon loss on two modes
    For two-mode 4-photon encoding, using photon ancillas
    """
    init_0_noloss=(_generate_product_number_state_using_basis([4,0,1,1],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,1,1],basis_conserved_mat))/jnp.sqrt(2)
    target_0_noloss=init_0_noloss
    init_1_noloss=_generate_product_number_state_using_basis([2,2,1,1],basis_conserved_mat)
    target_1_noloss=init_1_noloss
    #loss mode 1 m1loss
    init_0_m1loss=_generate_product_number_state_using_basis([3,0,1,1],basis_conserved_mat)
    target_0_m1loss=(_generate_product_number_state_using_basis([4,0,0,1],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,0,1],basis_conserved_mat))/jnp.sqrt(2)
    init_1_m1loss=_generate_product_number_state_using_basis([1,2,1,1],basis_conserved_mat)
    target_1_m1loss=_generate_product_number_state_using_basis([2,2,0,1],basis_conserved_mat)
    #loss mode 2 m2loss
    init_0_m2loss=_generate_product_number_state_using_basis([0,3,1,1],basis_conserved_mat)
    target_0_m2loss=(_generate_product_number_state_using_basis([4,0,1,0],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,1,0],basis_conserved_mat))/jnp.sqrt(2)
    init_1_m2loss=_generate_product_number_state_using_basis([2,1,1,1],basis_conserved_mat)
    target_1_m2loss=_generate_product_number_state_using_basis([2,2,1,0],basis_conserved_mat)
    #
    initial_states=[init_0_noloss,init_1_noloss,init_0_m1loss,init_1_m1loss,init_0_m2loss,init_1_m2loss]
    target_states=[target_0_noloss,target_1_noloss,target_0_m1loss,target_1_m1loss,target_0_m2loss,target_1_m2loss]
    return initial_states,target_states
#one mode loss, emitters
def generate_1loss_em_repeater_init_target_states(basis_conserved_mat):
    """
    Generate input-target states for one-way repeater correcting photon loss on one mode
    For two-mode 4-photon encoding, using emitter ancillas
    """
    #no loss
    init_0_noloss=(_generate_product_number_state_using_basis([4,0,1,0],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,1,0],basis_conserved_mat))/jnp.sqrt(2)
    target_0_noloss=init_0_noloss
    init_1_noloss=_generate_product_number_state_using_basis([2,2,1,0],basis_conserved_mat)
    target_1_noloss=init_1_noloss
    #loss mode 1 m1loss
    init_0_m1loss=_generate_product_number_state_using_basis([3,0,1,0],basis_conserved_mat)
    target_0_m1loss=(_generate_product_number_state_using_basis([4,0,0,0],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,0,0],basis_conserved_mat))/jnp.sqrt(2)
    init_1_m1loss=_generate_product_number_state_using_basis([1,2,1,0],basis_conserved_mat)
    target_1_m1loss=_generate_product_number_state_using_basis([2,2,0,0],basis_conserved_mat)
    initial_state=[init_0_noloss,init_1_noloss,init_0_m1loss,init_1_m1loss]
    target_state=[target_0_noloss,target_1_noloss,target_0_m1loss,target_1_m1loss]
    return initial_state,target_state
#two mode loss, coherent
def generate_2loss_em_coherent_repeater_init_target_states(basis_conserved_mat):
    """
    Generate input-target states for one-way repeater correcting photon loss on two modes
    For two-mode 4-photon encoding, using emitter ancillas
    """
    init_0_noloss=(_generate_product_number_state_using_basis([4,0,1,1],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,1,1],basis_conserved_mat))/jnp.sqrt(2)
    target_0_noloss=init_0_noloss
    init_1_noloss=_generate_product_number_state_using_basis([2,2,1,1],basis_conserved_mat)
    target_1_noloss=init_1_noloss
    #loss mode 1 m1loss
    init_0_m1loss=_generate_product_number_state_using_basis([3,0,1,1],basis_conserved_mat)
    target_0_m1loss=(_generate_product_number_state_using_basis([4,0,0,1],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,0,1],basis_conserved_mat))/jnp.sqrt(2)
    init_1_m1loss=_generate_product_number_state_using_basis([1,2,1,1],basis_conserved_mat)
    target_1_m1loss=_generate_product_number_state_using_basis([2,2,0,1],basis_conserved_mat)
    #loss mode 2 m2loss
    init_0_m2loss=_generate_product_number_state_using_basis([0,3,1,1],basis_conserved_mat)
    target_0_m2loss=(_generate_product_number_state_using_basis([4,0,1,0],basis_conserved_mat)+_generate_product_number_state_using_basis([0,4,1,0],basis_conserved_mat))/jnp.sqrt(2)
    init_1_m2loss=_generate_product_number_state_using_basis([2,1,1,1],basis_conserved_mat)
    target_1_m2loss=_generate_product_number_state_using_basis([2,2,1,0],basis_conserved_mat)
    #
    initial_states=[init_0_noloss,init_1_noloss,init_0_m1loss,init_1_m1loss,init_0_m2loss,init_1_m2loss]
    target_states=[target_0_noloss,target_1_noloss,target_0_m1loss,target_1_m1loss,target_0_m2loss,target_1_m2loss]
    return initial_states,target_states   