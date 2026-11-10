# Finite propagation delay: the delay-resolved RQPN model (Julia)

This folder contains the code and data for the paper's figure on a finite round-trip propagation delay through the
mixing circuit (τ = T_prop) and for the corresponding section of the Supplemental Material. The SLH model in `src/`
(Python) assumes that this delay is negligible. Here the recirculating network is modelled with the delay resolved, in
the time-bin (collision-model) discretisation on which [WaveguideQED.jl](https://github.com/qojulia/WaveguideQED.jl)
is built. The model is restricted to the one- and two-photon sectors and provides exact gradients. With it we

1. evaluate the SLH-optimised CZ gate at finite delay without changing its controls,
2. re-optimise the controls κ_m(t), δ_m(t) and the round-trip matrix inside the delay-resolved model, and determine
   the minimal gate time T_min(τ) for a gate infidelity ≤ 10⁻³,
3. check the model and the optimised gates against WaveguideQED.jl.

All quantities are in units of the self-phase-modulation (SPM, Kerr) strength, χ3 = 1: times in 1/χ3, rates in χ3.
Notation of the code vs the Supplemental Material: round-trip matrix R = **S S**ᵀ, N_b = N_τ bins per round trip,
"Kerr term" = SPM term.

## Installation

Install Julia ≥ 1.10 (tested with 1.12.2), e.g. with [juliaup](https://julialang.org/install/). From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. test/runtests.jl                       # consistency checks, about 1 min
```

The comparison with WaveguideQED.jl has its own environment, `waveguideqed/`, because WaveguideQED.jl and
QuantumOptics.jl take long to install and precompile (about 20 min on a Windows laptop) and are not needed for the
model itself:

```
julia --project=waveguideqed -e "using Pkg; Pkg.instantiate()"
```

`Manifest.toml` and `waveguideqed/Manifest.toml` record the exact package versions that we used (WaveguideQED.jl
0.2.14, QuantumOptics.jl 1.2.10, Optim.jl, JSON.jl). All commands below are run from this folder. The Python scripts
need numpy, scipy and matplotlib, which are part of the repository's Python environment.

## Quick start

```julia
using RQPNDelay
s = load_solution("results/solutions/smoothed/cz_k15_tau0.0010_T0.8350_smoothed.json")   # τ = 0.001, T = 0.835
F, grad, _ = fidelity_and_grad(s.pr, [s.θ], [s.C0], s.g, s.task)   # 1 − F = 9.733e-4, 839 gradient entries
refined_infidelity(s.pr, [s.θ], [s.C0], s.g, s.task; r=2)          # same controls on dt/2: 9.825e-4
```

## The model

Each cavity m (annihilation operator a_m, detuning δ_m(t), Kerr term −χ3 a_m†a_m†a_m a_m) couples with rate κ_m(t) to
its own waveguide. The waveguides run through the mixing circuit to the mirrors and back. All paths have the same
round-trip delay τ, and the round trip (circuit → mirrors → circuit) is a symmetric unitary matrix R.
In the input-output convention b_out = b_in + √κ a, the field returning to the cavities is b_in(t) = R b_out(t − τ).

**Discretisation.** Time bins of width dt hold the waveguide field, N_b = τ/dt bins per round trip, stored as a
ring buffer. In step k the cavities interact with the bins at ring position p = k mod N_b through
exp(−i dt H_k), with H_k = Σ δ_m n_m − χ3 Σ n_m(n_m − 1) + Σ i√(κ_m/dt)(a_m w_{m,p}† − a_m† w_{m,p}) (the
WaveguideQED.jl step Hamiltonian). The bins just emitted are then mixed by R and return to the cavities N_b steps later.

**SLH limit.** For τ → 0 the model reduces to the SLH Hamiltonian of the paper,
H = Σ δ_m n_m + Σ C_nm √(κ_n κ_m) a_n†a_m − χ3 Σ a_m†a_m†a_m a_m, with

    C = Im[R (I − R)⁻¹]   ⇔   R = (2C + i)(2C − i)⁻¹.

For two cavities, R = R_angles(θ, α, φ) = e^{iφ} [e^{iα}cos θ, i sin θ; i sin θ, e^{−iα}cos θ] = D B(θ) D: phase
shifters D = diag(e^{i(φ+α)/2}, e^{i(φ−α)/2}) and a coupler B(θ). Physically, light passes phase shifters and a coupler
B(θ/2) on the way to the mirror and again on the way back.

**Fidelity.** As in the SLH optimisation, F = |(1/N) Σ_n s_n ⟨target_n ⊗ vac|U(T)|input_n⟩|²: at t = T all photons
must be back in the cavities with the right phases, and the waveguides must be empty.

**Mirror-symmetric CZ.** All delay-aware CZ results use two independent two-cavity blocks, A = {1, 4} and B = {2, 3},
as in the paper's CZ solution (its C is block diagonal). Block B is the mirror image of block A: κ_3 = κ_1,
κ_2 = κ_4 and R_B = σx R_A σx. With s1 = ⟨1,0|U_A|1,0⟩, s2 = ⟨0,1|U_A|0,1⟩ and d = ⟨1,1|U_A|1,1⟩ the gate fidelity is
F = |(s1² + 2d − s2²)/4|², so only block A is simulated (`cz_task_mirror`).

**Implementation.** The one-photon state is a vector over D = M(1 + N_b) modes and the two-photon state a symmetric
D×D matrix. A step changes only the rows and columns of the cavities and of the active bins, and the Kerr term acts
only inside this block. Gradients with respect to all controls and the R angles come from a backward sweep that undoes
each (unitary) step, so the trajectory is not stored. One fidelity + gradient evaluation of the mirror CZ takes
0.07 s on one core for the κmax = 15, τ = 0.001 solution of the figure (N_b = 8, 6688 steps, 839 parameters) and 5 s
for the κmax = 240, τ = 0.26 one (N_b = 1248, 6624 steps, 26 499 parameters).

**Optimisation.** Controls are piecewise constant on control bins of equal width (an integer number of time steps).
κ = κmax sin²x and δ = δmax (2/π) atan y keep the bounds; L-BFGS (Optim.jl) minimises log(1 − F) and stops at
1 − F < 10⁻⁴.

## Layout

| path | content |
|---|---|
| `src/RQPNDelay.jl` | module: `network.jl` (R ↔ C, grid, step propagators, time stepping), `tasks.jl` (gates, `gate_fidelity`), `gradient.jl` (parameterisation, `fidelity_and_grad`), `slh.jl` (SLH reference), `optimize.jl` (grids, L-BFGS, smoothness penalty), `seeds.jl` (initial controls), `io.jl` (JSON files) |
| `scripts/convert_slh_cz.py` | paper's SLH CZ (`src/results/CZ_Chi3_res.pkl`) → `results/slh_cz_chi3.json` in physical units |
| `scripts/fixed_controls.jl` | infidelity of the SLH CZ vs τ, controls fixed |
| `scripts/optimize_point.jl` | one delay-aware CZ optimisation at (T, τ) |
| `scripts/tmin_scan.jl` | continuation towards shorter T at fixed τ |
| `scripts/smooth.jl` | post-optimisation towards smooth controls |
| `scripts/evaluate.jl` | infidelity of saved solutions on their grid and on refined grids dt/r, extrapolation dt → 0 |
| `scripts/trajectories.jl` | photon-number dynamics and gate fidelity F(t) of a solution, in the delay model and in the SLH model with the same controls |
| `scripts/plot_delay_figure.py` | the paper's figure: T_min(τ) and the two smoothed solutions |
| `test/runtests.jl` | consistency checks |
| `waveguideqed/` | comparison with WaveguideQED.jl (own environment) |
| `results/` | the T_min table and solutions, the two smoothed solutions of the figure and their trajectories, the fixed-control sweep, two test solutions |

New runs write to `output/` (ignored by git).

## Reproducing the results

### 1. The SLH-optimised CZ at finite delay

```
python scripts/convert_slh_cz.py          # optional: results/slh_cz_chi3.json is included
julia --project=. scripts/fixed_controls.jl
```

The paper's CZ (T = 0.7737, SLH infidelity 9.997 × 10⁻⁴) is split into its blocks {1, 4} and {2, 3} with
R = Cayley(C). Without re-optimisation, its infidelity is 2.5 × 10⁻³ at τ = 10⁻⁶, 0.23 at τ = 10⁻⁴ and above 0.95 for
τ ≳ 2 × 10⁻³ (`results/fixed_controls_slh_cz.json`, about 1 min).

### 2. Minimal gate time T_min(τ)

`results/tmin.json` lists T_min(τ) for κmax = 15 and κmax = 240 (|δ| ≤ 7.5) and points to the solution at each
T_min in `results/solutions/tmin/`. For every point it also gives the infidelity on refined grids (dt/2 … dt/16),
the extrapolation I_inf = 2I(dt/16) − I(dt/8), and the best infidelity found at the next lower T of the scan
(all above 10⁻³). The infidelities of all solutions can be recomputed in about 1 min:

```
julia --project=. scripts/evaluate.jl results/solutions/tmin                 # grid infidelities, vs stored values
julia --project=. scripts/evaluate.jl --r 1,2,4,8,16 results/solutions/tmin/cz_k15_tau0.2600_T1.8800.json
```

The time-grid error is first order in κ dt. The refinement keeps the piecewise-constant controls and the simulated
delay N_b dt. When the dense two-photon matrix (16 D² bytes) exceeds `--maxgb`, `evaluate.jl` switches to an
equivalent evaluator with O(D) memory; at κmax = 240 a solution at dt/16 then takes 15–90 min.

**How the solutions were found.** Each point is the shortest T on the scan grid (steps of 0.01; 0.005 for κmax = 240,
τ ≤ 0.003) at which an optimisation reached 1 − F ≤ 10⁻³ in the continuum limit. The scans (`scripts/tmin_scan.jl`)
start either from the analytic release/recapture seed (`hadamard<c>`: sech pulses, 50:50 round trip, used for most
κmax = 240 points) or from optimised solutions at neighbouring (T, τ) (`file:<solution.json>`), and we ran several
rounds in which the best solutions seeded scans at neighbouring delays. The exact search history is therefore not a
single command. The `seed` field of each solution records its starting point, for example:

```
# κmax = 240, τ = 0.26, from T = 1.46 down in steps of 0.02 (the T_min solution of results/ came from such a scan)
RQPN_DTMAX=2.0833e-4 julia --project=. scripts/tmin_scan.jl 0.26 240 1.46 0.02 6 hadamard24
# κmax = 15, continue from a stored solution
julia --project=. scripts/tmin_scan.jl 0.26 15 1.90 0.01 4 file:results/solutions/tmin/cz_k15_tau0.3000_T2.1400.json
```

Long L-BFGS runs are sensitive to rounding, so a rerun reproduces a result closely but not bit for bit. For example,
`RQPN_BINW=0.0155 julia --project=. scripts/optimize_point.jl 2.0 0.1 random3` (about 7 min) starts from the same
controls (1 − F = 0.963) and ends at 1 − F = 1.13 × 10⁻³, versus 1.12 × 10⁻³ for the stored
`results/solutions/validation/cz_k15_tau0.1000_T2.0000.json`. The two solutions in `results/solutions/validation/` are
test cases for this rerun and for `waveguideqed/check_solution.jl`; they are not shown in the paper.

Grid settings of the stored solutions (all with at least N_b = 8 bins per round trip):

| κmax | control-bin width | time step dt | script defaults that reproduce it |
|---|---|---|---|
| 15 | 0.004 | 10⁻³ (τ/8 for τ < 0.008) | defaults |
| 240 | 0.05/κmax = 2.08 × 10⁻⁴ | 1.04 × 10⁻⁴ (τ ≤ 0.03), 6.9 × 10⁻⁵ (τ = 0.05), 2.08 × 10⁻⁴ (τ ≥ 0.075) | defaults; `RQPN_DTMAX=2.0833e-4` for the last group |

### 3. Smoothed solutions and the figure

Panels (b) and (c) of the figure show T_min solutions post-optimised towards smooth controls at fixed T and τ
(`scripts/smooth.jl`): L-BFGS on (1 − F) + λS, where S sums the squared differences of κ_m/κmax and δ_m/δmax between
neighbouring control bins, with λ raised from stage to stage as long as the extrapolated infidelity stays ≤ 10⁻³.

| panel | solution | smoothing | 1 − F (grid) | 1 − F (dt → 0) |
|---|---|---|---|---|
| (b) | κmax = 15, τ = 0.001, T = T_min = 0.835 | first differences + √κ term (25 stages), then second differences + √κ term (5 stages) | 9.73 × 10⁻⁴ | 9.90 × 10⁻⁴ |
| (c) | κmax = 240, τ = 0.26, T = T_min = 1.38 | first differences (12 stages) | 4.09 × 10⁻⁴ | 6.31 × 10⁻⁴ |

For (b), the √κ term (`RQPN_SG=1`) removes jitter of switched-off couplings, which the κ term cannot reach
(dκ/dx → 0 at κ → 0), and second differences (`RQPN_SORDER=2`) penalise wiggles but not the steep ramps that are
needed at T_min. The commands are listed in the header of `scripts/smooth.jl`; the `seed` field of each solution
describes its smoothing. The solutions are in `results/solutions/smoothed/`, their trajectories in `results/trajectories/`:

```
RQPN_NSAMPLE=1000000 julia --project=. scripts/trajectories.jl results/solutions/smoothed/cz_k15_tau0.0010_T0.8350_smoothed.json
RQPN_NSAMPLE=2500 julia --project=. scripts/trajectories.jl results/solutions/smoothed/cz_k240_tau0.2600_T1.3800_smoothed.json
python scripts/plot_delay_figure.py                 # figures/fig_delay.pdf
```

The trajectories also contain the same controls in the SLH model (τ = 0, C = Im[R(I − R)⁻¹]); for (b) its
infidelity is 1.5 × 10⁻², and its F(t) is drawn as symbols. ⟨n_WG⟩ in (b) is averaged with a Gaussian of rms width 4τ:
the piecewise-constant controls excite loop modes, which add a fast ripple (about 35% of the maximum, converged in dt)
to the instantaneous value. The averaging needs the trajectory at every time step (`RQPN_NSAMPLE=1000000`).

### 4. Validation

`test/runtests.jl` checks the R ↔ C map, the SLH model against an independent numpy calculation of the paper's CZ,
the convergence of the delay model to the SLH model for τ = dt → 0, the adjoint gradient against finite differences,
the mirror-symmetric task against the full two-block CZ, the solution file format, and the smoothness penalty (value
stored with the smoothed solutions, gradient against finite differences).

In `waveguideqed/` the same network is built from WaveguideQED.jl operators (cavities with Fock cutoff 2, two-photon
`WaveguideBasis` with N_b bins, the round trip as one element or as the explicit circuit):

| script | what | result |
|---|---|---|
| `check_amplitudes.jl` | one- and two-photon amplitudes of the paper's CZ blocks at τ = 0.01 (N_b = 4) and 0.05 (N_b = 6); round trip as one element, and as phase shifters, coupler, mirror, coupler, phase shifters at their own delays (ODE solver, abstol = reltol = 10⁻¹⁰; 11 min) | max \|Δa\| = 1.4 × 10⁻⁸ (one element), 3.4 × 10⁻⁸ (circuit) |
| `check_solution.jl [file] [k]` | gate fidelity of an optimised solution on a k-times coarser grid (ODE solver; default T = 2.0, τ = 0.1, k = 4: 3.5 min) | F = 0.996744 (WaveguideQED.jl) vs 0.996746 |

Run them as `julia --project=waveguideqed waveguideqed/<script>.jl`.

## Solution files

`results/solutions/*/*.json` (written by `save_solution`, read by `load_solution`) describe one mirror-symmetric
two-cavity block:

| key | meaning |
|---|---|
| `T`, `dt`, `Nb`, `P`, `nsub` | gate time, time step, bins per round trip, number of control bins, time steps per control bin |
| `tau`, `tau_nominal` | simulated delay N_b dt, and the delay that was asked for |
| `kappa`, `delta` | piecewise-constant κ, δ on the P control bins; row 1 = κ_1 = κ_3, row 2 = κ_4 = κ_2 |
| `R_angles`, `R_real`, `R_imag` | round trip of block A, (θ, α, φ) and as a matrix |
| `kappa_max`, `delta_max` | bounds used in the optimisation |
| `infidelity`, `infidelity_dt2` | 1 − F on the grid and on dt/2 |
| `seed`, `source` | starting point of the optimisation; internal identifier |
| `smoothing_lambda`, `smoothness`, `smoothing_sqrt_kappa`, `smoothing_order`, `infidelity_est` | smoothed solutions only: final λ and S, the S options (√κ term, first or second differences), continuum estimate I(dt/2) + [I(dt/2) − I(dt)]/2 |

## Pitfalls

* Use at least N_b = 8 bins per round trip. With N_b = 2–3 the optimiser found controls tuned to the discretisation
  of the delay: a solution with 1 − F = 10⁻⁴ on its grid (T = 2.4, τ = 0.001) had 1 − F = 7.5 × 10⁻³ on a grid refined
  by four.
* Always re-evaluate optimised infidelities on refined grids (`refined_infidelity`, `scripts/evaluate.jl`). At
  κmax = 240 the infidelity on the optimisation grid (κ dt = 0.025–0.05) is about half the continuum value.
* The simulated delay is N_b dt, which can differ slightly from the requested τ (`tau` vs `tau_nominal`).
