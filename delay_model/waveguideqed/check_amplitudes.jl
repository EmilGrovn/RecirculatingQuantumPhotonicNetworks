# check_amplitudes.jl — RQPNDelay vs WaveguideQED.jl: one- and two-photon transition amplitudes of the two blocks
# of the paper's SLH-optimised CZ (controls fixed, T = 0.7737) at finite delay, and the round trip as a single
# element vs the explicit circuit of phase shifters, couplers and mirrors.
#
# usage: julia --project=waveguideqed waveguideqed/check_amplitudes.jl        (from delay_model/; about 10 min)
using RQPNDelay, LinearAlgebra, Printf
include(joinpath(@__DIR__, "wqed_model.jl"))

e = load_slh_cz()
blocks = slh_cz_blocks(e)
P = 50
# The 50 bins of the SLH solution have different lengths; resample onto P uniform bins so that every control edge falls on the dt grid
function uniform(c::PWC, T, P)
    edges = cumsum(c.dur)
    idx = [min(searchsortedfirst(edges, t), length(c.dur)) for t in ((1:P) .- 0.5) .* (T / P)]
    PWC(fill(T / P, P), c.κ[:, idx], c.δ[:, idx])
end
tol = (abstol=1e-10, reltol=1e-10)
maxdiff = 0.0
ins = ([1, 0], [0, 1], [1, 1])                      # one photon in cavity 1, in cavity 2, one photon in each
outs = Dict([1, 0] => ([1, 0], [0, 1]), [0, 1] => ([1, 0], [0, 1]), [1, 1] => ([1, 1], [2, 0], [0, 2]))
pairidx(v) = (ix = findall(>(0), v); length(ix) == 1 ? (v[ix[1]] == 2 ? (ix[1], ix[1]) : (ix[1], 0)) : (ix[1], ix[2]))

"""all amplitudes ⟨out|U|in⟩ of the fast model on grid g"""
function fast_amplitudes(g, R, c)
    U1s, U2s = step_propagators(g, c)
    amps = Dict{Tuple{Vector{Int},Vector{Int}},ComplexF64}()
    for inp in ins, out in outs[inp]
        if sum(inp) == 1
            amps[(inp, out)] = simulate1(g, R, U1s, cavity_state1(g, pairidx(inp)[1]))[pairidx(out)[1]]
        else
            Ψ = simulate2(g, R, U1s, U2s, cavity_state2(g, pairidx(inp)...))
            amps[(inp, out)] = sum(conj.(cavity_state2(g, pairidx(out)...)) .* Ψ)
        end
    end
    amps
end

println("== 1. Paper's SLH CZ, blocks A = {1,4} and B = {2,3}, round trip R = Cayley(C) as one element")
for (τ, Nb) in ((0.01, 4), (0.05, 6))
    global maxdiff
    dt = τ / Nb
    nsub = round(Int, e.T / P / dt)
    T = P * nsub * dt
    @printf("tau = %.3f, Nb = %d, dt = %.3e, T = %.4f\n", τ, Nb, dt, T)
    for (b, (C, c)) in enumerate(blocks)
        cr = uniform(c, T, P)
        R = roundtrip_from_C(C)
        fa = fast_amplitudes(Grid(2, dt, Nb, P * nsub), R, cr)
        for inp in ins
            t = @elapsed ψ, bcs, bw = wqed_final_state(2, dt, Nb, R, cr, T, inp; tol...)
            for out in outs[inp]
                w = cavity_amplitude(ψ, bcs, bw, out)
                f = fa[(inp, out)]
                maxdiff = max(maxdiff, abs(w - f))
                @printf("  block %s  %s -> %s:  RQPNDelay % .8f%+.8fi   WaveguideQED % .8f%+.8fi   |diff| %.1e  (%.0f s)\n",
                        b == 1 ? "A" : "B", inp, out, real(f), imag(f), real(w), imag(w), abs(w - f), t)
            end
        end
    end
end

println("\n== 2. Round trip as the explicit circuit (phase shifters, coupler, mirror, coupler, phase shifters at delays 4…0)")
τ, Nb = 0.05, 6
dt = τ / Nb
nsub = round(Int, e.T / P / dt)
T = P * nsub * dt
C, c = blocks[1]
cr = uniform(c, T, P)
R = roundtrip_from_C(C)
ang = RQPNDelay.angles_from_R(R)
els = cz_circuit_elements(ang...)
Rcirc = prod(exp(-im * h) for (_, h) in els)        # palindromic: the order of the product does not matter
@printf("||R_circuit - R|| = %.1e  (theta = %.4f, alpha = %.4f, phi = %.4f)\n", norm(Rcirc - R), ang...)
fa = fast_amplitudes(Grid(2, dt, Nb, P * nsub), R, cr)
for inp in ins
    global maxdiff
    t = @elapsed ψ, bcs, bw = wqed_final_state(2, dt, Nb, R, cr, T, inp; elements=els, tol...)
    for out in outs[inp]
        w = cavity_amplitude(ψ, bcs, bw, out)
        f = fa[(inp, out)]
        maxdiff = max(maxdiff, abs(w - f))
        @printf("  block A  %s -> %s:  RQPNDelay % .8f%+.8fi   WaveguideQED circuit % .8f%+.8fi   |diff| %.1e  (%.0f s)\n",
                inp, out, real(f), imag(f), real(w), imag(w), abs(w - f), t)
    end
end
@printf("\nlargest |amplitude difference| = %.1e (WaveguideQED.jl: ODE solver with abstol = reltol = 1e-10)\n", maxdiff)
