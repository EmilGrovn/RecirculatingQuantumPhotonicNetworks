# smooth.jl — post-optimisation of a solution towards smooth controls (same T, τ and grid):
#     J = (1 − F) + λ S,   S = Σ_m Σ_p [((κ_{m,p+1} − κ_{m,p})/κmax)² + ((δ_{m,p+1} − δ_{m,p})/δmax)²]
# (a discrete ∫κ̇², ∫δ̇² penalty: large jumps are expensive, gentle ramps cheap; `smoothness` in src/optimize.jl).
# Continuation in λ: start at λ0 (default 0.2 I0/S0) and multiply λ by f = 2 after every accepted stage. Every stage is
# re-evaluated at dt/2 with the continuum estimate I_est = I(dt/2) + [I(dt/2) − I(dt)]/2. A stage with I_est > Imax is
# rejected and repeated from the last accepted controls with a smaller step (f → √f; λ → λ/4 if no stage has been
# accepted yet). The continuation stops after nstages accepted stages or 4 consecutive rejections.
#
# usage: julia --project=. scripts/smooth.jl <solution.json> [nstages=12] [iters=300] [Imax=1e-3] [lambda0=auto] [tag]
# env:   RQPN_SG=1      add Σ_m Σ_p ((√κ_{m,p+1} − √κ_{m,p})/√κmax)² to S (removes jitter of switched-off couplings)
#        RQPN_SORDER=2  second instead of first differences in every term of S (wiggles cost, straight ramps do not)
# output: output/smooth/<name><tag>_s<stage>.json (one file per accepted stage)
#
# The smoothed solutions of the paper's figure (results/solutions/smoothed/) were obtained as
#   κmax = 240, τ = 0.26 (stage 12 of):  scripts/smooth.jl results/solutions/tmin/cz_k240_tau0.2600_T1.3800.json 14
#   κmax = 15, τ = 0.001 (two passes):   RQPN_SG=1 scripts/smooth.jl results/solutions/tmin/cz_k15_tau0.0010_T0.8350.json 25 600 9.9e-4 2e-5 _sg
#     then on its stage 25:              RQPN_SG=1 RQPN_SORDER=2 scripts/smooth.jl output/smooth/cz_k15_tau0.0010_T0.8350_sg_s25.json 15 600 9.9e-4 1e-3 _c2
#     (stage 5 of the second pass; 4 consecutive rejections ended it).
using RQPNDelay, Printf, Optim

f = ARGS[1]
nstages = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
iters = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 300
Imax = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-3
λarg = length(ARGS) >= 5 ? ARGS[5] : "auto"
tag = length(ARGS) >= 6 ? ARGS[6] : ""
sqrtk = get(ENV, "RQPN_SG", "0") == "1"
order = parse(Int, get(ENV, "RQPN_SORDER", "1"))
s = load_solution(f)
pr, g, task, C0s = s.pr, s.g, s.task, [s.C0]
smooth(θ) = smoothness(pr, θ; sqrtk=sqrtk, order=order)

"""largest jump between neighbouring control bins, in units of the bounds (κ/κmax, δ/δmax)"""
function maxjump(θ)
    _, _, κ, δ, _, _, _ = unpack(pr, θ, s.C0)
    maximum(abs.(diff(κ, dims=2))) / pr.κmax, maximum(abs.(diff(δ, dims=2))) / pr.δmax
end

function main()
    θ = copy(s.θ)
    F0, _, _ = fidelity_and_grad(pr, [θ], C0s, g, task; want_grad=false)
    S0, _ = smooth(θ)
    λ = λarg == "auto" ? 0.2 * (1 - F0) / S0 : parse(Float64, λarg)
    outdir = joinpath(@__DIR__, "..", "output", "smooth")
    base = splitext(basename(f))[1] * tag
    jk, jd = maxjump(θ)
    @printf("%s: T = %.3f tau = %.4f  I = %.3e  I(dt/2) = %.3e  S = %.3f (sqrt(kappa) term %s, order %d)  max jump kappa %.3f delta %.3f (of bound)\n",
            base, s.T, g.tau, 1 - F0, refined_infidelity(pr, [θ], C0s, g, task), S0, sqrtk ? "on" : "off", order, jk, jd)
    flush(stdout)
    λok, fac, fails, stage = 0.0, 2.0, 0, 0
    while stage < nstages
        cache = Ref{Any}((nothing, 0.0, Float64[]))       # memoised (θ, J, ∇J): Optim asks for f and ∇f separately
        function comp(th)
            if cache[][1] === nothing || cache[][1] != th
                F, grads, _ = fidelity_and_grad(pr, [th], C0s, g, task)
                S, gS = smooth(th)
                cache[] = (copy(th), (1 - F) + λ * S, -grads[1] .+ λ .* gS)
            end
            cache[]
        end
        t = @elapsed res = optimize(th -> comp(th)[2], (G, th) -> (G .= comp(th)[3]), θ, LBFGS(m=20),
                                    Optim.Options(iterations=iters, g_tol=1e-12, f_reltol=0.0, allow_f_increases=true))
        θn = Optim.minimizer(res)
        F, _, _ = fidelity_and_grad(pr, [θn], C0s, g, task; want_grad=false)
        I = 1 - F
        Iref = refined_infidelity(pr, [θn], C0s, g, task; r=2)
        Iest = Iref + max(0.0, Iref - I) / 2
        S, _ = smooth(θn)
        if Iest > Imax
            fails += 1
            @printf("rejected lambda = %.2e : I = %.3e  I_est = %.3e  S = %.4f  (%.0f s)\n", λ, I, Iest, S, t)
            flush(stdout)
            if fails >= 4
                println("stopping: 4 consecutive rejections (last accepted stage = $stage)")
                break
            end
            if λok > 0
                fac = sqrt(fac); λ = λok * fac
            else
                λ /= 4
            end
            continue
        end
        stage += 1
        fails = 0
        jk, jd = maxjump(θn)
        @printf("stage %2d  lambda = %.2e : I = %.3e  I(dt/2) = %.3e  I_est = %.3e  S = %.4f  max jump kappa %.3f delta %.3f  (%.0f s)\n",
                stage, λ, I, Iref, Iest, S, jk, jd, t)
        flush(stdout)
        save_solution(joinpath(outdir, @sprintf("%s_s%02d.json", base, stage)), pr, θn, g; tau_nominal=s.meta["tau_nominal"],
                      infidelity=I, infidelity_dt2=Iref, infidelity_est=Iest, seed="smooth:" * basename(f),
                      smoothing_lambda=λ, smoothness=S, smoothing_sqrt_kappa=sqrtk, smoothing_order=order)
        θ = θn
        λok = λ
        λ *= fac
    end
end
main()
