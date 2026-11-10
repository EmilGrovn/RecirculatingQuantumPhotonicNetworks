# fixed_controls.jl — infidelity of the paper's SLH-optimised CZ (results/slh_cz_chi3.json, T = 0.7737) versus the
# round-trip delay τ when the controls are NOT re-optimised.
#
# The four-cavity network splits into the blocks A = {1, 4} and B = {2, 3} (C_14 = −3.64, C_23 = 3.45, all other
# |C_nm| < 1e-3 are dropped); each block's round trip is R = Cayley(C). At τ → 0 the model reduces to the SLH model.
#
# usage: julia --project=. scripts/fixed_controls.jl [tau ...]      (default: 31 values, 1e-6 … 0.3)
# output: results/fixed_controls_slh_cz.json
using RQPNDelay, Printf, JSON

taus = length(ARGS) >= 1 ? parse.(Float64, ARGS) : 10 .^ range(-6, log10(0.3), length=31)
Nbmin, dtmax = 4, 1e-3

e = load_slh_cz()
blocks = [(roundtrip_from_C(C), c) for (C, c) in slh_cz_blocks(e)]
task = cz_task_blocks()
Fslh_full, _ = slh_gate_fidelity([(e.C, e.pwc)], cz_task_full())
Fslh, _ = slh_gate_fidelity(slh_cz_blocks(e), task)
@printf("SLH (tau = 0): infidelity %.4e (four cavities, full C), %.4e (two blocks)\n", 1 - Fslh_full, 1 - Fslh)
infs, tausim = Float64[], Float64[]
for τ in taus
    dt = min(dtmax, τ / Nbmin)
    Nb = max(1, round(Int, τ / dt))
    g = Grid(2, dt, Nb, round(Int, e.T / dt))
    t = @elapsed F, _ = gate_fidelity([(g, R, c) for (R, c) in blocks], task)
    push!(infs, 1 - F); push!(tausim, g.tau)
    @printf("  tau = %.3e  (Nb = %4d, dt = %.2e)  infidelity = %.4e   (%.1f s)\n", g.tau, Nb, dt, 1 - F, t)
    flush(stdout)
end
out = joinpath(@__DIR__, "..", "results", "fixed_controls_slh_cz.json")
open(io -> JSON.print(io, Dict("description" => "Paper's SLH-optimised CZ (T = 0.7737), controls fixed, vs round-trip delay tau",
                               "T" => e.T, "slh_infidelity" => 1 - Fslh, "tau" => tausim, "infidelity" => infs), 1), out, "w")
println("wrote ", out)
