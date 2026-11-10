# optimize_point.jl — one delay-aware optimisation of the mirror-symmetric CZ gate at gate time T and delay τ.
#
# usage: julia --project=. scripts/optimize_point.jl T tau seed [kappa_max=15] [iters=1500] [out.json]
#   seed ∈ hadamard<c>      analytic release/recapture protocol, pulse length c/κmax (e.g. hadamard18)
#        | random<n>        smooth random controls and round trip, RNG seed n
#        | slh              the paper's SLH-optimised CZ (results/slh_cz_chi3.json), centred in [0, T]
#        | file:<sol.json>  a previous solution (any T, P), mapped onto [0, T] by cutting/inserting time in the middle
# env:   RQPN_DTMAX  max time step (default min(1e-3, 0.025/κmax))
#        RQPN_BINW   control-bin width (default 0.004 for κmax ≤ 15, else 0.05/κmax)
#        RQPN_NBMIN  min bins per round trip (default 8)    RQPN_DMAX  detuning bound (default 7.5)
#        RQPN_TARGET stop at this infidelity (default 1e-4)
# output: output/opt/cz_k<κmax>_tau<τ>_T<T>_<seed>.json (default); the infidelity is also evaluated at dt/2.
using RQPNDelay, Printf, Random

T = parse(Float64, ARGS[1])
τ = parse(Float64, ARGS[2])
seed = ARGS[3]
κmax = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 15.0
iters = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1500
dtmax = parse(Float64, get(ENV, "RQPN_DTMAX", string(min(1e-3, 0.025 / κmax))))
binw = parse(Float64, get(ENV, "RQPN_BINW", string(κmax <= 15 ? 0.004 : 0.05 / κmax)))
nbmin = parse(Int, get(ENV, "RQPN_NBMIN", "8"))
δmax = parse(Float64, get(ENV, "RQPN_DMAX", "7.5"))
target = parse(Float64, get(ENV, "RQPN_TARGET", "1e-4"))
out = length(ARGS) >= 6 ? ARGS[6] :
      joinpath(@__DIR__, "..", "output", "opt", @sprintf("cz_k%g_tau%.4f_T%.4f_%s.json", κmax, τ, T, startswith(seed, "file:") ? "file" : seed))

P = max(20, round(Int, T / binw))
nsub, dt, Nb = choose_grid(T, P, τ; dtmax=dtmax, Nbmin=nbmin)
pr = Params(M=2, P=P, nsub=nsub, κmax=κmax, δmax=δmax, rmode=:angles)
g = Grid(2, dt, Nb, P * nsub)
task = cz_task_mirror()
C0s = [zeros(2, 2)]
θs = if startswith(seed, "hadamard")
    seed_hadamard(T, P, τ, pr; c=parse(Float64, seed[9:end]))
elseif startswith(seed, "random")
    seed_random(T, P, pr, MersenneTwister(parse(Int, seed[7:end])))
elseif seed == "slh"
    e = load_slh_cz()
    seed_slh(e, slh_cz_blocks(e), T, P, pr)[1:1]          # block A; block B is its mirror image
elseif startswith(seed, "file:")
    s = load_solution(seed[6:end])
    κn, δn = midcut_pwc(s.κ, s.δ, s.T, T, P)
    [pack(pr, clamp.(κn, 0, κmax), clamp.(δn, -0.999δmax, 0.999δmax), zeros(2, 2); angles=s.θ[end-2:end])]
else
    error("unknown seed $seed")
end
F0, _, _ = fidelity_and_grad(pr, θs, C0s, g, task; want_grad=false)
@printf("T = %.4f  tau = %.4e (Nb = %d, dt = %.2e, %d steps)  P = %d  kappa_max = %g  seed = %s  start I = %.4e\n",
        T, g.tau, Nb, dt, g.nsteps, P, κmax, seed, 1 - F0)
flush(stdout)
t = @elapsed θn, F, trace = reoptimize(pr, θs, C0s, g, task; iters=iters, target=target)
Iref = refined_infidelity(pr, θn, C0s, g, task; r=2)
_, _, κ, δ, R, _, _ = unpack(pr, θn[1], C0s[1])
@printf("final I = %.4e,  I(dt/2) = %.4e   (%d evaluations, %.0f s)   |R12| = %.4f  max kappa = %.2f  max |delta| = %.2f\n",
        1 - F, Iref, length(trace), t, abs(R[1, 2]), maximum(κ), maximum(abs.(δ)))
save_solution(out, pr, θn[1], g; tau_nominal=τ, infidelity=1 - F, infidelity_dt2=Iref, seed=seed, evaluations=length(trace))
println("saved ", out)
