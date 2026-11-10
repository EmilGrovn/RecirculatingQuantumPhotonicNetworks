# tmin_scan.jl — continuation in the gate time T at fixed delay τ and coupling bound κmax: optimise at T_start, then
# repeatedly shorten T by dT (time cut out of the middle of the previous optimum, `midcut_pwc`) and re-optimise.
# Every optimum is re-evaluated on the time grid refined by 2 (same controls, same τ).
#
# usage: julia --project=. scripts/tmin_scan.jl tau kappa_max T_start dT nT seed [iters=1500] [tag]
#   seed ∈ hadamard<c>      analytic release/recapture protocol, pulse length c/κmax (e.g. hadamard18, hadamard24)
#        | file:<sol.json>  a previous solution (any T, P)
# env:   RQPN_DTMAX, RQPN_BINW, RQPN_NBMIN, RQPN_DMAX, RQPN_TARGET as in optimize_point.jl;
#        RQPN_STOP: stop after two consecutive T with I(dt/2) above this value (default 3e-2)
# output: output/tmin/cz_k<κmax>_tau<τ>_T<T>_<seed><tag>.json, one file per T
#
# The T_min values of the paper are the smallest T at which any scan reached I ≤ 1e-3 after extrapolation to dt → 0
# (scripts/evaluate.jl); the scans were started from analytic seeds and from the solutions at neighbouring τ.
using RQPNDelay, Printf

τ = parse(Float64, ARGS[1])
κmax = parse(Float64, ARGS[2])
T0 = parse(Float64, ARGS[3])
dT = parse(Float64, ARGS[4])
nT = parse(Int, ARGS[5])
seed = ARGS[6]
iters = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 1500
tag = length(ARGS) >= 8 ? ARGS[8] : ""
dtmax = parse(Float64, get(ENV, "RQPN_DTMAX", string(min(1e-3, 0.025 / κmax))))
binw = parse(Float64, get(ENV, "RQPN_BINW", string(κmax <= 15 ? 0.004 : 0.05 / κmax)))
nbmin = parse(Int, get(ENV, "RQPN_NBMIN", "8"))
δmax = parse(Float64, get(ENV, "RQPN_DMAX", "7.5"))
target = parse(Float64, get(ENV, "RQPN_TARGET", "1e-4"))
stopI = parse(Float64, get(ENV, "RQPN_STOP", "3e-2"))
seedtag = startswith(seed, "file:") ? "file" : seed
outdir = joinpath(@__DIR__, "..", "output", "tmin")
task = cz_task_mirror()
C0s = [zeros(2, 2)]
bound(I) = π / 4 - asin(sqrt(I))      # T_NL(I): fastest Kerr phase π/2 at infidelity I (τ = 0)

prev = nothing                         # (T, κ, δ, R angles) of the previous optimum
if startswith(seed, "file:")
    s = load_solution(seed[6:end])
    prev = (s.T, s.κ, s.δ, s.θ[end-2:end])
end
nbad = 0
@printf("tmin_scan: tau = %.4f  kappa_max = %g  T = %.4f - k*%.4f (k < %d)  seed = %s   T_NL(1e-3) + 2tau = %.4f\n",
        τ, κmax, T0, dT, nT, seed, bound(1e-3) + 2τ)
for k in 0:nT-1
    global prev, nbad
    T = T0 - k * dT
    P = max(20, round(Int, T / binw))
    nsub, dt, Nb = choose_grid(T, P, τ; dtmax=dtmax, Nbmin=nbmin)
    pr = Params(M=2, P=P, nsub=nsub, κmax=κmax, δmax=δmax, rmode=:angles)
    g = Grid(2, dt, Nb, P * nsub)
    θs = if prev === nothing
        seed_hadamard(T, P, τ, pr; c=parse(Float64, seed[9:end]))
    else
        κn, δn = midcut_pwc(prev[2], prev[3], prev[1], T, P)
        [pack(pr, clamp.(κn, 0, κmax), clamp.(δn, -0.999δmax, 0.999δmax), zeros(2, 2); angles=prev[4])]
    end
    F0, _, _ = fidelity_and_grad(pr, θs, C0s, g, task; want_grad=false)
    t = @elapsed θn, F, trace = reoptimize(pr, θs, C0s, g, task; iters=iters, target=target, verbose=false)
    Iref = refined_infidelity(pr, θn, C0s, g, task; r=2)
    _, _, κ, δ, R, _, _ = unpack(pr, θn[1], C0s[1])
    @printf("T = %.4f (T - 2tau - T_NL = %+.4f)  P = %d dt = %.2e Nb = %d : I0 = %.3e -> I = %.3e  I(dt/2) = %.3e  (%d evals, %.0f s)  |R12| = %.3f  max kappa = %.1f\n",
            T, T - 2τ - bound(1e-3), P, dt, Nb, 1 - F0, 1 - F, Iref, length(trace), t, abs(R[1, 2]), maximum(κ))
    flush(stdout)
    save_solution(joinpath(outdir, @sprintf("cz_k%g_tau%.4f_T%.4f_%s%s.json", κmax, τ, T, seedtag, tag)), pr, θn[1], g;
                  tau_nominal=τ, infidelity=1 - F, infidelity_dt2=Iref, seed=seed, evaluations=length(trace))
    prev = (T, κ, δ, θn[1][end-2:end])
    nbad = Iref > stopI ? nbad + 1 : 0
    nbad >= 2 && (println("stopping: two consecutive T with I > $stopI"); break)
end
