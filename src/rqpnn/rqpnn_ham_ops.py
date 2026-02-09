#rqpnn_ham_ops.py
#necessary imports
import jax.numpy as jnp
import dynamiqs as dq
#generate dictionary of hamiltonian operators for RQPNN system
def _generate_hamiltonian_operators(spec,raising_op_list,switch_matrix_rest_to_cons):
    if spec.has_emitters==True:
        #cavity emitter spaces
        cav_raising_op_list=raising_op_list[0:spec.M_cav]
        em_raising_op_list=raising_op_list[spec.M_cav:spec.M_cav+spec.M_em]
        n_op_list=_generate_number_op_list(cav_raising_op_list,switch_matrix_rest_to_cons)      #number operator for each cavity
        ee_op_list=_generate_number_op_list(em_raising_op_list,switch_matrix_rest_to_cons)            #excitation number operator for each emitter
        em_cav_int_op_list=_generate_cav_em_op_list(cav_raising_op_list,em_raising_op_list,switch_matrix_rest_to_cons)#cavity-emitter interaction operator for each emitter in a cavity
        Ham_int_op_nestedlist=_generate_Ham_int_ops(cav_raising_op_list,switch_matrix_rest_to_cons)#interaction hamiltonian operators between all pairs of cavities
        dict_ops={"delta_cav":n_op_list,"delta_em":ee_op_list,"g_lm":em_cav_int_op_list,"int":Ham_int_op_nestedlist}
    elif spec.has_emitters==False:
        #cavity only spaces
        cav_raising_op_list=raising_op_list
        n_op_list=_generate_number_op_list(cav_raising_op_list,switch_matrix_rest_to_cons)
        n2mn_op_list=_generate_number_squared_minus_number_op_list(n_op_list)
        Ham_int_op_nestedlist=_generate_Ham_int_ops(cav_raising_op_list,switch_matrix_rest_to_cons)
        dict_ops={"delta_cav":n_op_list,"chi3":n2mn_op_list,"int":Ham_int_op_nestedlist} #dictionary of operators
    return dict_ops

### ----------- generate hamilton operators!!
#cavity number
def _generate_number_op_list(raising_op_list,switch_matrix_rest_to_cons):
    '''
    Generate number operator for cavity or emitter i (linear diagonal term)
    --------
    returns:
    list of n_op_i for each cavity or emitter i
    n_op_i: number operator for cavity or emitter i (object: dynamiqs.qarrays.sparsedia_qarray.SparseDIAQArray)
    '''
    n_op_list=[]
    for idx in range(len(raising_op_list)):    
        a_i_dag=raising_op_list[idx]     #access system operator for chosen cavity
        a_i=a_i_dag.T
        n_op_i=a_i_dag@a_i          #generate linear hamiltonian term (number operator)
        n_op_i_conserved=_transform_op_jax_to_reduced_space(n_op_i,switch_matrix_rest_to_cons)
        n_op_i_dq_conserved=_convert_jaxnumpy_to_dq_op(n_op_i_conserved) #convert to qarray
        n_op_list.append(n_op_i_dq_conserved)            #append to list of number operators
    return n_op_list

## ---------- generate each non-linear cavity operator
def _generate_number_squared_minus_number_op_list(n_op_list):
    '''
    Generate normally ordered squared number operator for cavity i (non-linear diagonal term)
    --------
    returns:
    list of n2mn_op_i for each cavity i
    n2mn_op_i: n^2-n operator for cavity i (object: dynamiqs.qarrays.sparsedia_qarray.SparseDIAQArray)
    '''
    n2mn_op_list=[]
    for idx in range(len(n_op_list)):
        n_i=n_op_list[idx]
        n2mn_op_i=n_i@(n_i-dq.eye_like(n_i))
        n2mn_op_list.append(n2mn_op_i)
    return n2mn_op_list

#generate cav-emitter interaction operators (i.e. non-interacting hamiltonian)
def _generate_cav_em_op_list(cav_raising_op_list,em_raising_op_list,switch_matrix_rest_to_cons):
    '''
    Generate the cavity-emitter interaction operator for emitter i in cavity i (non-interacting hamiltonian term)
    --------
    output:
    list of a_dag_i@sigma_i + sigma_dag_i@a_i for each emitter i in cavity i
    (object: dynamiqs.qarrays.sparsedia_qarray.SparseDIAQArray)
    '''
    em_cav_int_op_list=[]
    assert len(em_raising_op_list)==len(cav_raising_op_list), "number of emitters must be equal to number of cavities"
    for idx in range(len(cav_raising_op_list)):
        a_dag_i=cav_raising_op_list[idx]
        sigma_dag_i=em_raising_op_list[idx]
        sigma_i=sigma_dag_i.T
        a_i=a_dag_i.T
        em_cav_int_op_jax=a_dag_i@sigma_i+sigma_dag_i@a_i
        em_cav_int_op_jax_cons=_transform_op_jax_to_reduced_space(em_cav_int_op_jax,switch_matrix_rest_to_cons)
        ee_op_i_dq_conserved=_convert_jaxnumpy_to_dq_op(em_cav_int_op_jax_cons) #convert to qarray
        em_cav_int_op_list.append(ee_op_i_dq_conserved)      #append to list of number operators
    return em_cav_int_op_list
# generate ALL interaction hamiltonian operators
def _generate_Ham_int_ops(cav_raising_op_list,switch_matrix_rest_to_cons):
    '''
    Generate the interaction hamiltonian operators between all cavities (WITHOUT coupling and scattering!)
    --------
    returns:
    H_int_ops_nested_list: nested list of interaction hamiltonian operators
                    len(H_int_ops_nested_list),len(H_int_ops_nested_list[0])=n_cavities,n_cavities
                    element [i][j]=a_dag_i@a_j (object: dynamiqs.qarrays.sparsedia_qarray.SparseDIAQArray)
    '''
    H_int_ops_nested_list=[]                           #initialize nested list of interaction hamiltonian operators
    for i in range(0,len(cav_raising_op_list)):
        H_int_row_list=[]
        for j in range(0,len(cav_raising_op_list)):
            a_dag_i=cav_raising_op_list[i]          #access system operator for chosen cavity
            a_j=cav_raising_op_list[j].T            #access system operator for chosen cavity
            H_int_ij=a_dag_i@a_j
            H_int_ij_jax_cons=_transform_op_jax_to_reduced_space(H_int_ij,switch_matrix_rest_to_cons)
            H_int_ij_dq_cons=_convert_jaxnumpy_to_dq_op(H_int_ij_jax_cons) #convert to qarray
            H_int_row_list.append(H_int_ij_dq_cons)                
        H_int_ops_nested_list.append(H_int_row_list)
    return H_int_ops_nested_list
    
### ---------- helpers
#transform to reduced space
def _transform_op_jax_to_reduced_space(op_jax,trans_mat):
    '''
    Transform jax numpy array operator from restricted to conserved space using the transformation matrix
    '''
    #transform to reduced space
    op_jax_cons=trans_mat@op_jax@trans_mat.T
    return op_jax_cons

#
def _convert_jaxnumpy_to_dq_op(op_jax):
    '''
    Convert (jax) numpy array operators to dynamical quantum operators
    '''
    return dq.asqarray(op_jax.astype('complex64')).asdense()
