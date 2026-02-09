# src/rqpnn/rqpnn_basis.py

#Consider system (defined in RQPNNSpec) with either M bosonic modes or M bosonic modes and M fermionic modes
#can be imported as: from rqpnn.spec import RQPNNSpec #import the RQPNNSpec class
    #syntax works because rqpnn is on our Python path as a proper package


#For this system: generate restricted and conserved basis and basis switch matrix 
# Restricted basis: all states with excitation number up to N_exc_max
# Conserved basis: all states with excitation number in N_exc_list
# conserved basis = basis_switch_matrix * restricted basis

#necessary imports
import jax.numpy as jnp             #for numpy operations in jax
import math                         #for combinatorics
from itertools import combinations  #for choosen N out of M elements

#generate restricted and conserved basis and basis switch matrix
def _generate_basises_and_switch_matrix(M_cav: int,M_em: int,N_exc_list: list[int]):
    """
    Generate conserved and restricted basis and basis-switch matrix from restricted to conserved basis.
    Restricted basis: all states with excitation number up to N_exc_max
    Conserved basis: all states with excitation number in N_exc_list
    input:
    M_cav: int, number of cavities (positive integer)
    M_em: int, number of emitters (0 or equal to M_cav)
    N_exc_list: tuple of non-negative integers in ascending order, excitation numbers to include in the conserved basis
    output:
    -------
    basis_conserved_mat: jnp-array, each row gives the excitation numbers of a basisstate
    basis_restricted_mat: jnp-array, each row gives the excitation numbers of a basisstate
    switch_matrix_rest_to_cons: jax array of shape (hilbertdim_conserved,hilbertdim_restricted)
    hilbertdim_restricted: integer, dimension of restricted Hilbert space
    """
    basis_conserved_list=_generate_basis_conserved(M_cav,M_em,N_exc_list)
    basis_restricted_list,index_list=_generate_basis_restricted(M_cav,M_em,max(N_exc_list))
    basis_conserved_mat = jnp.asarray(basis_conserved_list)
    basis_restricted_mat = jnp.asarray(basis_restricted_list)
    hilbertdim_restricted=index_list[-1]#extract dimension of restricted Hilbert space
    
    #produce mask that is 1 for the indices of the restricted basis which we will include in the conserved basis
    mask=[0]*hilbertdim_restricted #mask of zeros
    for N_exc_i,index_saved in enumerate(index_list[:-1]):#loop over excitation numbers (0,1,2,..) and corresponding index in the restricted basis where the new excitation number starts
        if N_exc_i in N_exc_list:
            index_next=index_list[N_exc_i+1]
            for i in range(index_saved,index_next):
                mask[i]=1
    
    #mask
    mask = jnp.asarray(mask).astype(bool)#IMPORTANT: convert to jax array and ensure bool
    switch_matrix_rest_to_cons=jnp.eye(hilbertdim_restricted)[mask,:] #generate basis switch matrix rows,cols=hilbertdim_conserved,hilbertdim_restricted
    return basis_conserved_mat,basis_restricted_mat,switch_matrix_rest_to_cons,hilbertdim_restricted

def _generate_basis_restricted(M_cav: int,M_em: int,N_exc_max: int) -> list[list[int]]:
    """
    Generate the restricted basis for the RQPNN and given maximum excitation number.
    output:
    - basis_tot_list: list of lists, each sublist gives the excitation numbers of a basisstate
    - index_list: list of integers, each integer gives the index in basis_tot_list where a new excitation subspace starts, including (as the last element) the dimension of the hilbert space (number of indices)
    """
    #check input that is not already checked in generate total basis function
    if not isinstance(N_exc_max, int) or N_exc_max < 0:
        raise ValueError("N_exc must be a non-negative integer")
    #generate basis
    basis_tot_list=[]
    index_list=[]
    index_list.append(0)
    for N_exc in range(0,N_exc_max+1):
        basis_tot_list+=_generate_total_basis_for_single_excitation_number(M_cav,M_em,N_exc)
        index_list.append(len(basis_tot_list))

    return basis_tot_list,index_list

#generate basis for list of excitation numbers
def _generate_basis_conserved(M_cav: int,M_em: int,N_exc_list: tuple[int]) -> list[list[int]]:
    """
    Generate the conserved basis for given RQPNN system and a list of excitation numbers.
    """
    #check input that is not already checked in generate total basis function
    if not isinstance(N_exc_list, tuple):
        raise ValueError("N_exc_list must be a tuple")
    if len(N_exc_list) == 0:
        raise ValueError("N_exc_list must not be empty.")
    if any(n < 0 for n in N_exc_list) or not all(isinstance(n, int) for n in N_exc_list):
        raise ValueError("All elements of N_exc_list must be non-negative integers.")
    if not sorted(N_exc_list) == list(N_exc_list):
        raise ValueError("Elements of N_exc_list must be sorted in ascending order.")
    

    #generate basis
    basis_tot_list=[]
    for N_exc in N_exc_list:
        basis_tot_list+=_generate_total_basis_for_single_excitation_number(M_cav,M_em,N_exc)
    return basis_tot_list   


### ---- helpers: basis generation for single excitation number ---- ###
#function which chooses the proper space and generate basis for single excitation number
def _generate_total_basis_for_single_excitation_number(M_cav,M_em,N_exc):
    """
    Generate the total basis for given RQPNN system and a single excitation number.
    """
    #generate basis
    if M_em==0:
        return _generate_bosonic_basis_for_excitation_number(M_cav,N_exc)
    elif M_em==M_cav:
        return _generate_combined_boson_fermion_basis(M_cav,N_exc)
    #raise error if M_em is not 0 or M_cav
    else:
        raise ValueError("Our architecture requires M_em==0 or M_em==M_cav")



### ---- helpers: bosonic and fermionic and combined basis generation ---- ###

#combined bosonic and fermionic system
def _generate_combined_boson_fermion_basis(M,N_exc, *, check: bool = False ) -> list[list[int]]:
    """
    Generate the combined basis for a system with both bosonic and fermionic excitations.
    input: number of modes M (M bosonic and M fermionic), number of total excitations
    output: list of lists, each sublist gives the excitation numbers (bosonic and fermionic) of a basisstate
    """

    #check input
    if not isinstance(M, int) or M <= 0:
        raise ValueError("M must be a positive integer")
    if not isinstance(N_exc, int) or N_exc < 0:
        raise ValueError("N_exc must be a non-negative integer")

    #generate combined basis
    N_em_list=range(0,min(M,N_exc)+1)
    basis_tot_list=[]
    dimension=0#check dimension
    for N_em in N_em_list:
        N_cav=N_exc-N_em
        basis_bosons_list=_generate_bosonic_basis_for_excitation_number(M,N_cav)
        basis_fermions_list=_generate_fermionic_basis_for_excitation_number(M,N_em)    
        for idx1,basis_boson_state in enumerate(basis_bosons_list):
            for idx2,basis_fermion_state in enumerate(basis_fermions_list):
                basis_tot_list.append(basis_boson_state+basis_fermion_state)

        #check
        dimension+=math.comb(M+N_cav-1,N_cav)*math.comb(M,N_em)
        #print("dim",dimension)
        #optional check for internal consistency
        if check:
            if dimension !=len(basis_tot_list):
                raise ValueError("Not the correct number of states")
    
    return basis_tot_list    


#bosonic basis generation
def _generate_bosonic_basis_for_excitation_number(M_cav: int, N_exc: int, *, check: bool = False ) -> list[list[int]]:
    """
    Generate all ways to distribute N indistinguishable items (excitations)
    into M_cav distinguishable bins (modes) using backtracking.
    input: number of modes, number of photons
    output: list of lists, each sublist gives the photon numbers of a basisstate
    """
    #check input
    if not isinstance(M_cav, int) or M_cav <= 0:
        raise ValueError("M_cav must be a positive integer")
    if not isinstance(N_exc, int) or N_exc < 0:
        raise ValueError("N_exc must be a non-negative integer")

    #generate basis
    basis_list = []
    for i,dist in enumerate(_distributions_backtracking(M_cav,N_exc)):
        basis_list.append(dist)

    #check internal consistency
    if check:
        #check that back_tracking works properly
        expected_dim = math.comb(M_cav+N_exc-1,N_exc)
        if expected_dim !=len(basis_list):
            raise ValueError("Not the correct number of states")
    return basis_list

#create function to generate bosonic distributions
def _distributions_backtracking(M_cav,N):
    """
    Return function to generate all ways to distribute N indistinguishable items (excitations)
    into M_cav distinguishable bins (modes) using backtracking.
    """
    def backtrack(remaining, bins_left):
        # If there's only one bin left, it must contain all remaining items
        if bins_left == 1:
            yield [remaining]
        else:
            # Try all possible counts for the current bin from 0..remaining
            for count in range(remaining + 1):
                for tail in backtrack(remaining - count, bins_left - 1):
                    yield [count] + tail
    
    # Call the recursive generator
    yield from backtrack(N, M_cav)

#fermionic generation
def _generate_fermionic_basis_for_excitation_number(M_em: int, N_exc: int, *, check: bool = False ) -> list[list[int]]:
    """
    Generate all ways to choose N_exc from M_em distinguishable emitters.
    input: number of emitters, number of excitations
    output: list of lists, each sublist gives the excitation number of each emitter which constitutes a basisstate
    """
    #checks of input
    if not isinstance(M_em, int) or M_em <= 0:
        raise ValueError("M_em must be a positive integer")
    if not isinstance(N_exc, int) or N_exc < 0:
        raise ValueError("N_exc must be a non-negative integer")
    if N_exc > M_em:
        raise ValueError("N_exc must be less than or equal to M_em")

    #generate fermionic basis
    basis_fermion_list = []
    for chosen_modes_positions in combinations(range(M_em), N_exc):
        vec = [0] * M_em
        for idx in chosen_modes_positions:
            vec[idx] = 1
        basis_fermion_list.append(vec)


    #check of code internal consistency
    if check:
        expected_dim = math.comb(M_em,N_exc)
        if expected_dim !=len(basis_fermion_list):
            raise ValueError("Not the correct number of states")
    return basis_fermion_list