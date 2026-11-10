# check_solution.jl — recompute the CZ fidelity of an optimised (mirror-symmetric) solution in WaveguideQED.jl
#
# usage: julia --project=waveguideqed waveguideqed/check_solution.jl [solution.json] [k=4]      (from delay_model/)
#   default solution: results/solutions/validation/cz_k15_tau0.1000_T2.0000.json
#   k ≥ 1 coarsens the time grid (dt → k dt, N_b → round(τ/(k dt))) to make the WaveguideQED.jl run cheaper; RQPNDelay is
#   evaluated on the same grid. The coarser grid moves the simulated delay slightly, so F differs from the stored
#   optimum; the comparison is like for like. k = 1 (the optimisation grid itself) takes hours.
#   Results: T = 2.0, τ = 0.1, k = 1: F = 0.998881 (RQPNDelay) vs 0.998869 (WaveguideQED.jl);
#            T = 3.2, τ = 0.3, k = 8: 0.996784 vs 0.996782.
using RQPNDelay, LinearAlgebra, Printf
include(joinpath(@__DIR__, "wqed_model.jl"))

file = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "results", "solutions", "validation", "cz_k15_tau0.1000_T2.0000.json")
k = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
s = load_solution(file)
pr = s.pr
@assert pr.nsub % k == 0 "the coarsening factor must divide nsub = $(pr.nsub)"
dt = s.g.dt * k
Nb = max(2, round(Int, s.tau / dt))
c = PWC(fill(s.T / pr.P, pr.P), s.κ, s.δ)
@printf("%s: T = %.3f, tau = %.4f -> dt = %.3e, Nb = %d (simulated tau = %.4f); stored I = %.4e\n",
        basename(file), s.T, s.tau, dt, Nb, Nb * dt, s.meta["infidelity"])
g = Grid(2, dt, Nb, pr.P * (pr.nsub ÷ k))
Ffast, _ = gate_fidelity([(g, s.R, c)], s.task)
@printf("RQPNDelay on this grid: F = %.6f\n", Ffast)
flush(stdout)
amps = unique([a for (_, lst) in s.task for (_, a) in lst])
vals = Dict{Amp,ComplexF64}()
for a in amps
    t = @elapsed begin
        ψ, bcs, bw = wqed_final_state(2, dt, Nb, s.R, c, s.T, occupation(a, a.input, 2); abstol=1e-9, reltol=1e-9)
        vals[a] = cavity_amplitude(ψ, bcs, bw, occupation(a, a.target, 2))
    end
    @printf("  %d photon(s) %s -> %s: WaveguideQED %.8f%+.8fi  (%.0f s)\n", a.kind, a.input, a.target, real(vals[a]), imag(vals[a]), t)
    flush(stdout)
end
terms = [sg * prod(vals[a] for (_, a) in lst) for (sg, lst) in s.task]
Fw = abs2(sum(terms) / length(terms))
@printf("WaveguideQED.jl: F = %.6f   RQPNDelay: F = %.6f   |dF| = %.1e\n", Fw, Ffast, abs(Fw - Ffast))
