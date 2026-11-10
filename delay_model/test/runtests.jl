# runtests.jl — consistency checks of RQPNDelay (≈ 1 min):  julia --project=. test/runtests.jl
# The comparison with WaveguideQED.jl is in waveguideqed/ (separate environment).
using RQPNDelay, Test, LinearAlgebra, Random
import RQPNDelay: dR_angles

const RES = joinpath(@__DIR__, "..", "results")

@testset "RQPNDelay" begin

@testset "round trip R <-> coupling matrix C" begin
    rng = MersenneTwister(1)
    C = randn(rng, 4, 4); C = C + C'
    R = roundtrip_from_C(C)
    @test norm(R - transpose(R)) < 1e-12                     # symmetric
    @test norm(R * R' - I) < 1e-12                           # unitary
    @test norm(C_from_roundtrip(R) - C) < 1e-10
    # two cavities: R_angles covers every symmetric unitary
    R2 = roundtrip_from_C([0.3 -1.2; -1.2 0.7])
    @test norm(R_angles(angles_from_R(R2)...) - R2) < 1e-12
    # ∂R/∂angles
    a, h = [0.4, -0.9, 1.3], 1e-6
    for k in 1:3
        e = zeros(3); e[k] = h
        @test norm((R_angles((a .+ e)...) - R_angles((a .- e)...)) / 2h - dR_angles(a...)[k]) < 1e-8
    end
end

@testset "SLH model reproduces the paper's CZ" begin
    e = load_slh_cz()
    F4, _ = slh_gate_fidelity([(e.C, e.pwc)], cz_task_full())
    @test isapprox(1 - F4, e.raw["slh_infidelity_numpy"]; rtol=1e-8)   # independent numpy result
    Fb, _ = slh_gate_fidelity(slh_cz_blocks(e), cz_task_blocks())
    @test abs(F4 - Fb) < 1e-6                                          # inter-block couplings |C| < 1e-3 are negligible
end

@testset "tau -> 0 limit: delay model -> SLH model" begin
    rng = MersenneTwister(7)
    M, P, T = 2, 10, 1.0
    C = [0.2 0.6; 0.6 -0.3]
    c = PWC(fill(T / P, P), 5 .* rand(rng, M, P), 4 .* rand(rng, M, P) .- 2)
    blk = [(C, c)]
    task = [(+1, [(1, Amp(1, (1, 0), (1, 0)))]), (+1, [(1, Amp(1, (1, 0), (2, 0)))]),
            (+1, [(1, Amp(2, (1, 2), (1, 2)))]), (+1, [(1, Amp(2, (1, 2), (1, 1)))])]
    _, t_slh = slh_gate_fidelity(blk, task)
    err = map((10, 40, 160, 640)) do nsub
        dt = T / (P * nsub)
        _, t = gate_fidelity([(make_grid(M, T, dt, dt), roundtrip_from_C(C), c)], task)   # N_b = 1, τ = dt
        maximum(abs.(t .- t_slh))
    end
    @test issorted(err; rev=true) && err[end] < 2e-3 && err[1] / err[end] > 30   # ≈ first order in dt
end

@testset "adjoint gradient vs finite differences" begin
    rng = MersenneTwister(1)
    M, P, nsub, T = 2, 6, 5, 0.6
    dt = T / (P * nsub)
    g = make_grid(M, T, dt, 3dt)
    task = cz_task_blocks()
    C0s = [[0.0 1.3; 1.3 0.0], [0.2 -0.8; -0.8 0.1]]
    for pr in (Params(M=M, P=P, nsub=nsub, rmode=:C, Cmask=Bool[1 1; 0 1]), Params(M=M, P=P, nsub=nsub, rmode=:angles))
        θs = [randn(rng, nparams(pr)) for _ in 1:2]
        F, grads, _ = fidelity_and_grad(pr, θs, C0s, g, task)
        for b in 1:2, i in [1, 3, 7, 13, 20, 25, nparams(pr) - 1, nparams(pr)]
            h = 1e-6
            θp = deepcopy(θs); θp[b][i] += h
            θm = deepcopy(θs); θm[b][i] -= h
            fd = (fidelity_and_grad(pr, θp, C0s, g, task; want_grad=false)[1] -
                  fidelity_and_grad(pr, θm, C0s, g, task; want_grad=false)[1]) / 2h
            @test abs(fd - grads[b][i]) <= 1e-5 * max(abs(fd), 1e-3)
        end
        bl = map(1:2) do b
            _, _, κ, δ, R, _, _ = unpack(pr, θs[b], C0s[b])
            (g, R, PWC(fill(T / P, P), κ, δ))
        end
        @test abs(F - gate_fidelity(bl, task)[1]) < 1e-12
    end
end

@testset "shipped solution: stored infidelity, mirror symmetry, save/load" begin
    f = joinpath(RES, "solutions", "smoothed", "cz_k15_tau0.0010_T0.8350_smoothed.json")
    s = load_solution(f)
    F, _, _ = fidelity_and_grad(s.pr, [s.θ], [s.C0], s.g, s.task; want_grad=false)
    @test abs((1 - F) - s.meta["infidelity"]) < 1e-11
    # the four-cavity gate on two blocks (B = mirror image of A) gives the same fidelity
    Ffull, _ = gate_fidelity([(s.g, R, c) for (R, c) in mirror_blocks(s.R, PWC(fill(s.T / s.pr.P, s.pr.P), s.κ, s.δ))],
                             cz_task_blocks())
    @test abs(F - Ffull) < 1e-11
    tmp = joinpath(mktempdir(), "sol.json")
    save_solution(tmp, s.pr, s.θ, s.g; tau_nominal=s.meta["tau_nominal"], infidelity=1 - F)
    s2 = load_solution(tmp)
    @test s2.θ ≈ s.θ && s2.g == s.g && s2.T ≈ s.T
end

@testset "smoothness penalty of the shipped smoothed solutions" begin
    for f in readdir(joinpath(RES, "solutions", "smoothed"); join=true)
        s = load_solution(f)
        S, _ = smoothness(s.pr, s.θ; sqrtk=s.meta["smoothing_sqrt_kappa"], order=s.meta["smoothing_order"])
        @test isapprox(S, s.meta["smoothness"]; rtol=1e-9)
    end
end

@testset "seeds and a short optimisation" begin
    T, τ = 1.6, 0.1
    P = 40
    nsub, dt, Nb = choose_grid(T, P, τ; dtmax=4e-3, Nbmin=8)
    pr = Params(M=2, P=P, nsub=nsub, κmax=15.0, rmode=:angles)
    g = Grid(2, dt, Nb, P * nsub)
    task, C0s = cz_task_mirror(), [zeros(2, 2)]
    e = load_slh_cz()
    for θs in (seed_hadamard(T, P, τ, pr), seed_random(T, P, pr, MersenneTwister(3)), seed_slh(e, slh_cz_blocks(e), T, P, pr)[1:1])
        @test length(θs) == 1 && length(θs[1]) == nparams(pr)
    end
    θs = seed_hadamard(T, P, τ, pr)
    F0, _, _ = fidelity_and_grad(pr, θs, C0s, g, task; want_grad=false)
    _, F, _ = reoptimize(pr, θs, C0s, g, task; iters=30, verbose=false)
    @test F > F0 + 0.1                                                  # 30 L-BFGS iterations improve the seed
    θr = seed_random(T, P, pr, MersenneTwister(5))[1]
    for (sqrtk, order) in ((false, 1), (true, 1), (true, 2)), i in (1, 6, P + 3, 2P + 7, 2P * 2 - 1)
        S, gS = smoothness(pr, θr; sqrtk=sqrtk, order=order)
        h = 1e-6; e1 = zeros(length(θr)); e1[i] = h
        fd = (smoothness(pr, θr .+ e1; sqrtk=sqrtk, order=order)[1] - smoothness(pr, θr .- e1; sqrtk=sqrtk, order=order)[1]) / 2h
        @test abs(fd - gS[i]) < 1e-6 * max(1, abs(fd))
    end
end

end
