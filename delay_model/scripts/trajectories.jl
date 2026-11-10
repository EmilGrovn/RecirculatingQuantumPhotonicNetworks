# trajectories.jl — dynamics of a mirror-symmetric CZ solution (block A = cavities 1, 4) in the delay model and, with the
# same controls, in the SLH model (τ = 0, C = Im[R(I − R)⁻¹]).
#
# usage: julia --project=. scripts/trajectories.jl <solution.json> [out.json]
#   Inputs: one photon in cavity 1 ("single1") or cavity 2 ("single2") of the block, one photon in each ("double").
#   Stored at every sample time t: ⟨n_1⟩, ⟨n_2⟩ (the waveguide loop holds the rest, e.g. 1 − ⟨n_1⟩ − ⟨n_2⟩), the overlap
#   a(t) = ⟨input ⊗ vac|ψ(t)⟩ with the input cavity state ("a_re", "a_im"; the CZ target is the input up to a sign) and,
#   for "double", the probability of two photons in one cavity. The gate fidelity at time t is
#   F(t) = |(a_1² + 2a_12 − a_2²)/4|² (a_1, a_2, a_12: overlaps of single1, single2, double), and F(T) = 1 − I.
#   "slh": the same quantities in the SLH model (no loop, ⟨n_1⟩ + ⟨n_2⟩ = photon number).
# env:  RQPN_NSAMPLE   approximate number of stored time points (default 600; at least the number of steps stores
#                      every step, which the figure uses for the small-τ panel)
#       RQPN_REFINE=r  evaluate on the time grid refined by r (dt/r, same controls and τ)
# output: results/trajectories/<name>.json (default); values rounded to 8 significant digits
using RQPNDelay, Printf, JSON, LinearAlgebra
import RQPNDelay: step1!, step2!, slh_h

file = ARGS[1]
out = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results", "trajectories", splitext(basename(file))[1] * ".json")
s = load_solution(file)
rf = parse(Int, get(ENV, "RQPN_REFINE", "1"))
g = Grid(2, s.g.dt / rf, s.g.Nb * rf, s.g.nsteps * rf)
nsub = s.pr.nsub * rf                                      # time steps per control bin
R, C = s.R, C_from_roundtrip(s.R)
c = PWC(fill(s.T / s.pr.P, s.pr.P), s.κ, s.δ)
every = max(1, g.nsteps ÷ parse(Int, get(ENV, "RQPN_NSAMPLE", "600")))
sampled(k) = k % every == 0 || k == g.nsteps - 1
r8(v) = round.(v, sigdigits=8)
pack_traj(d) = Dict(k => r8(v) for (k, v) in d)
fgate(a1, a2, a12) = abs2((a1^2 + 2a12 - a2^2) / 4)

"""Delay model: the three inputs of block A; returns the sampled data and the final overlaps."""
function delay_model()
    U1s, U2s = step_propagators(g, c)
    res, fin = Dict{String,Any}(), Dict{String,ComplexF64}()
    for m in 1:2
        ψ = cavity_state1(g, m)
        d = Dict(k => Float64[] for k in ("t", "n1", "n2", "a_re", "a_im"))
        for k in 0:g.nsteps-1
            step1!(ψ, U1s[k+1], R, 2, mod(k, g.Nb))
            sampled(k) || continue
            push!(d["t"], (k + 1) * g.dt); push!(d["n1"], abs2(ψ[1])); push!(d["n2"], abs2(ψ[2]))
            push!(d["a_re"], real(ψ[m])); push!(d["a_im"], imag(ψ[m]))
        end
        res["single$m"], fin["single$m"] = pack_traj(d), ψ[m]
    end
    Ψ = cavity_state2(g, 1, 2)
    bufA = zeros(ComplexF64, g.D, 4); bufB = zeros(ComplexF64, g.D, 2)
    d = Dict(k => Float64[] for k in ("t", "n1", "n2", "p_double", "a_re", "a_im"))
    for k in 0:g.nsteps-1
        step2!(Ψ, U1s[k+1], U2s[k+1], R, 2, mod(k, g.Nb), bufA, bufB)
        sampled(k) || continue
        push!(d["t"], (k + 1) * g.dt)
        push!(d["n1"], 2 * sum(abs2, view(Ψ, 1, :))); push!(d["n2"], 2 * sum(abs2, view(Ψ, 2, :)))
        push!(d["p_double"], abs2(Ψ[1, 1]) + abs2(Ψ[2, 2]))
        push!(d["a_re"], real(sqrt(2) * Ψ[1, 2])); push!(d["a_im"], imag(sqrt(2) * Ψ[1, 2]))   # ⟨1_1 1_2|Ψ⟩
    end
    res["double"], fin["double"] = pack_traj(d), sqrt(2) * Ψ[1, 2]
    res, fgate(fin["single1"], fin["single2"], fin["double"])
end

"""SLH model with the same piecewise-constant controls, sampled after the same time steps."""
function slh_model()
    M = 2
    Id = Matrix{ComplexF64}(I, M, M)
    U1, U2 = Matrix{ComplexF64}[], Matrix{ComplexF64}[]
    for p in 1:s.pr.P
        h = Matrix{ComplexF64}(slh_h(C, s.κ[:, p], s.δ[:, p]))
        G = kron(Id, h) + kron(h, Id)
        for m in 1:M
            G[(m-1)*M+m, (m-1)*M+m] -= 2                    # SPM −χ3 a†a†aa on |2_m⟩ (χ3 = 1)
        end
        push!(U1, exp(-im * g.dt * Hermitian(h))); push!(U2, exp(-im * g.dt * Hermitian(G)))
    end
    res, fin = Dict{String,Any}(), Dict{String,ComplexF64}()
    for m in 1:M
        ψ = zeros(ComplexF64, M); ψ[m] = 1
        d = Dict(k => Float64[] for k in ("t", "n1", "n2", "a_re", "a_im"))
        for k in 0:g.nsteps-1
            ψ = U1[k÷nsub+1] * ψ
            sampled(k) || continue
            push!(d["t"], (k + 1) * g.dt); push!(d["n1"], abs2(ψ[1])); push!(d["n2"], abs2(ψ[2]))
            push!(d["a_re"], real(ψ[m])); push!(d["a_im"], imag(ψ[m]))
        end
        res["single$m"], fin["single$m"] = pack_traj(d), ψ[m]
    end
    Ψ = ComplexF64[0 1; 1 0] ./ sqrt(2)
    d = Dict(k => Float64[] for k in ("t", "n1", "n2", "p_double", "a_re", "a_im"))
    for k in 0:g.nsteps-1
        Ψ = reshape(U2[k÷nsub+1] * vec(Ψ), M, M)
        sampled(k) || continue
        push!(d["t"], (k + 1) * g.dt)
        push!(d["n1"], 2 * sum(abs2, view(Ψ, 1, :))); push!(d["n2"], 2 * sum(abs2, view(Ψ, 2, :)))
        push!(d["p_double"], abs2(Ψ[1, 1]) + abs2(Ψ[2, 2]))
        push!(d["a_re"], real(sqrt(2) * Ψ[1, 2])); push!(d["a_im"], imag(sqrt(2) * Ψ[1, 2]))
    end
    res["double"], fin["double"] = pack_traj(d), sqrt(2) * Ψ[1, 2]
    F = fgate(fin["single1"], fin["single2"], fin["double"])
    Fref, _ = slh_gate_fidelity([(C, c)], cz_task_mirror())
    @assert abs(F - Fref) < 1e-10 "SLH trajectory inconsistent with slh_gate_fidelity"
    res["infidelity"] = 1 - F
    res
end

res, F = delay_model()
rf == 1 && @assert abs((1 - F) - s.meta["infidelity"]) < 1e-10 "trajectory inconsistent with the stored infidelity"
slh = slh_model()
merge!(res, Dict{String,Any}("solution" => basename(file), "T" => s.T, "tau" => g.tau, "dt" => g.dt, "infidelity" => 1 - F,
                             "t_ctrl" => collect(((1:s.pr.P) .- 0.5) .* s.T / s.pr.P), "kappa" => [s.κ[1, :], s.κ[2, :]],
                             "delta" => [s.δ[1, :], s.δ[2, :]], "C" => [C[1, :], C[2, :]], "slh" => slh))
mkpath(dirname(abspath(out)))
open(io -> JSON.print(io, res), out, "w")
@printf("wrote %s  (T = %.3f, tau = %.4f, I = %.3e, SLH model with the same controls: I = %.3e, %d samples)\n",
        out, s.T, g.tau, 1 - F, slh["infidelity"], length(res["single1"]["t"]))
