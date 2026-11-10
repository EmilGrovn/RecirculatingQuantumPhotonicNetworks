"""
    RQPNDelay

Delay-resolved model of a recirculating quantum photonic network (RQPN) with a finite round-trip
propagation delay τ (= T_prop of the paper), in the time-bin (collision-model) discretisation used by
WaveguideQED.jl, restricted to the one- and two-photon sectors, with exact adjoint gradients.

Model (units χ3 = 1, times in 1/χ3):
* M cavities a_m with detuning δ_m(t), coupling κ_m(t) and Kerr term −χ a_m†a_m†a_m a_m.
* Cavity m couples to its own waveguide m. The waveguides run through the mixing circuit to the
  mirrors and back. All paths have the same round-trip delay τ, and the net linear map
  (circuit → mirror → circuit) is the symmetric unitary R.
* Time bins of width dt, N_b = τ/dt bins per round trip, stored as a ring buffer (the same as
  `WaveguideBasis(Np, M, times)` with `length(times) == N_b` in WaveguideQED.jl).
* Step k (k = 0, 1, …) with active ring position p = k mod N_b:
      1. E_k = exp(−i dt H_k),  H_k = Σ δ_m n_m − χ Σ n_m(n_m − 1)
                                    + Σ i√(κ_m/dt)(a_m w_{m,p}† − a_m† w_{m,p})
         (the WaveguideQED.jl step Hamiltonian);
      2. the just-emitted bins w_{·,p} are mixed by R. They re-enter the cavities N_b steps later.
* τ → 0 reproduces the SLH Hamiltonian of the paper (Eq. 1),
      H = Σ δ_m n_m + Σ C_nm √(κ_n κ_m) a_n†a_m − χ Σ a_m†a_m†a_m a_m,
  with C = Im[R(I − R)⁻¹]  ⇔  R = (2C + i)(2C − i)⁻¹ (Cayley transform).

State representation
* One photon: vector ψ over D = M(1 + N_b) modes. Mode order: cavities 1..M, then bin (j, p) ↦ M + p·M + j.
* Two photons: symmetric D×D matrix Ψ with |Ψ⟩ = (1/√2) Σ_ij Ψ_ij c_i† c_j† |0⟩ and ⟨Ψ|Φ⟩ = Σ conj(Ψ_ij) Φ_ij.
  Linear maps U act as Ψ ↦ U Ψ Uᵀ. The Kerr term acts on the diagonal cavity entries Ψ_mm (amplitude of |2_m⟩).

Files
* `network.jl`   C ↔ R, time grid, piecewise-constant controls, propagators, one-/two-photon time stepping
* `tasks.jl`     gate definitions (CZ, CNOT) and the gate fidelity for arbitrary piecewise-constant controls
* `gradient.jl`  optimisation parameters and the fidelity with its exact (adjoint) gradient
* `slh.jl`       SLH (τ = 0) reference model
* `optimize.jl`  time-grid choice, control resampling, L-BFGS driver, smoothness penalty
* `seeds.jl`     initial controls: analytic release/recapture protocol, the paper's SLH-optimised CZ, random
* `io.jl`        JSON input/output of solutions and of the SLH CZ solution
"""
module RQPNDelay

using LinearAlgebra, Printf, Random
using Optim, JSON

include("network.jl")
include("tasks.jl")
include("gradient.jl")
include("slh.jl")
include("optimize.jl")
include("seeds.jl")
include("io.jl")

# network
export roundtrip_from_C, C_from_roundtrip, R_angles, angles_from_R, Grid, make_grid, PWC,
       step_propagators, simulate1, simulate2, cavity_state1, cavity_state2
# tasks
export Amp, cz_task_mirror, cz_task_blocks, cz_task_full, cnot_task_full, gate_fidelity
# parameterisation and gradient
export Params, nparams, unpack, pack, fidelity_and_grad, κmap, δmap, κinv, δinv
# SLH reference
export slh_propagators, slh_gate_fidelity
# optimisation
export choose_grid, resample_pwc, midcut_pwc, reoptimize, smoothness, refined_infidelity
# seeds
export hadamard_controls, hadamard_R, seed_hadamard, seed_slh, seed_random
# input/output
export load_slh_cz, slh_cz_blocks, mirror_blocks, save_solution, load_solution

end # module
