# wqed_model.jl — the delay-resolved RQPN built directly with WaveguideQED.jl (independent reference for RQPNDelay)
#
# * M cavities (Fock cutoff 2) and M waveguides stored as a ring buffer of N_b = τ/dt time bins
#   (`WaveguideBasis(2, M, times)` with length(times) = N_b);
# * cavity m interacts with waveguide m through operators with delay N_b − 1, i.e. a photon emitted into a bin
#   returns to the cavities N_b steps later;
# * the mixing circuit + mirrors act as linear elements H_el = (1/dt) Σ_jl h_jl w_j†w_l at smaller delays, with
#   exp(−i h) the element's scattering matrix. By default one element h = i log R at delay 0 represents the whole round
#   trip; `elements` = [(delay, h), …] places the individual beam splitters, phase shifters and mirrors instead
#   (the element with the largest delay acts first);
# * time-dependent controls enter through a TimeDependentSum and the state is propagated with
#   QuantumOptics.timeevolution.schroedinger_dynamic (adaptive ODE solver).
using QuantumOptics, WaveguideQED, LinearAlgebra
using RQPNDelay: PWC

"""Time-dependent evolution over [0, T] that sets the waveguide time index and the control time in every bin."""
function wqed_evolve(T, psi, H; tstops=Float64[], kwargs...)
    ops = get_waveguide_operators(H)
    dt = WaveguideQED.get_dt(H.basis_l)
    function get_hamiltonian(time, psi)
        tidx = round(Int, time / dt, RoundDown) + 1
        set_waveguidetimeindex!(ops, tidx)
        set_time!(H, time)
        return H
    end
    disc = sort(unique(vcat(collect(0:dt:T), tstops)))
    _, ψ = timeevolution.schroedinger_dynamic([0.0, T], psi, get_hamiltonian; d_discontinuities=disc, tstops=disc, kwargs...)
    ψ[end]
end

"""
    build_wqed(M, dt, Nb, R, κfun, δfun; χ=1.0, elements=nothing) -> (H, cavity bases, waveguide basis)

κfun[m](t), δfun[m](t): control functions. `elements` (optional): list of (delay, h) linear elements replacing R.
"""
function build_wqed(M, dt, Nb, R, κfun, δfun; χ=1.0, elements=nothing)
    @assert Nb >= 2 "the WaveguideQED model needs at least 2 bins per round trip"
    bcs = [FockBasis(2) for _ in 1:M]
    bc = tensor(bcs...)
    Ic = identityoperator(bc)
    single(op, m) = tensor([i == m ? op(bcs[i]) : identityoperator(bcs[i]) for i in 1:M]...)
    a = [single(destroy, m) for m in 1:M]
    ad = [single(create, m) for m in 1:M]
    bw = WaveguideBasis(2, M, range(0, step=dt, length=Nb))
    Iw = identityoperator(bw)
    w = [destroy(bw, m; delay=Nb - 1) for m in 1:M]
    wd = [create(bw, m; delay=Nb - 1) for m in 1:M]
    static = Any[(-χ * sum(ad[m] * ad[m] * a[m] * a[m] for m in 1:M)) ⊗ Iw]          # Kerr
    els = elements === nothing ? [(0, Matrix(im * log(Matrix{ComplexF64}(R))))] : elements
    for (d, h) in els
        @assert 0 <= d < Nb - 1 "elements must sit between emission and re-absorption (delay 0 … Nb−2)"
        for j in 1:M, l in 1:M
            abs(h[j, l]) < 1e-14 && continue
            push!(static, (h[j, l] / dt) * (Ic ⊗ (create(bw, j; delay=d) * destroy(bw, l; delay=d))))
        end
    end
    coeffs = Any[1.0]
    ops = Any[LazySum(static...)]
    for m in 1:M
        push!(coeffs, t -> sqrt(max(κfun[m](t), 0.0) / dt))
        push!(ops, im * (a[m] ⊗ wd[m]) + (-im) * (ad[m] ⊗ w[m]))     # i√(κ/dt)(a w† − a† w)
        push!(coeffs, t -> δfun[m](t))
        push!(ops, (ad[m] * a[m]) ⊗ Iw)
    end
    TimeDependentSum(coeffs, ops), bcs, bw
end

"""Piecewise-constant lookup: value of the segment that contains t (0 after the last segment)."""
function pwc_function(dur, vals)
    edges = cumsum(dur)
    t -> (i = searchsortedfirst(edges, t * (1 - 1e-12)); i > length(vals) ? 0.0 : vals[i])
end

"""
    wqed_final_state(M, dt, Nb, R, c::PWC, T, input; χ=1, elements=nothing, kwargs...) -> (ψ(T), cavity bases, waveguide basis)

input: vector of cavity photon numbers, waveguides empty at t = 0.
"""
function wqed_final_state(M, dt, Nb, R, c::PWC, T, input; χ=1.0, elements=nothing, kwargs...)
    κf = [pwc_function(c.dur, c.κ[m, :]) for m in 1:M]
    δf = [pwc_function(c.dur, c.δ[m, :]) for m in 1:M]
    H, bcs, bw = build_wqed(M, dt, Nb, R, κf, δf; χ=χ, elements=elements)
    ψ0 = tensor([fockstate(bcs[m], input[m]) for m in 1:M]...) ⊗ zerophoton(bw)
    wqed_evolve(T, ψ0, H; tstops=cumsum(c.dur), kwargs...), bcs, bw
end

"""⟨target ⊗ vac|ψ⟩ for a vector of cavity photon numbers `target`."""
cavity_amplitude(ψ, bcs, bw, target) = dagger(tensor([fockstate(bcs[m], target[m]) for m in eachindex(bcs)]...) ⊗ zerophoton(bw)) * ψ

"""Photon-number vector of an RQPNDelay.Amp input or target (cavity indices)."""
occupation(a, which, M) = (v = zeros(Int, M); a.kind == 1 ? (v[which[1]] = 1) : (v[which[1]] += 1; v[which[2]] += 1); v)

"""
Elements of the two-cavity round trip R_angles(θ, α, φ) as the physical circuit: phase shifters D', coupler B(θ/2),
mirror (−1), coupler B(θ/2), phase shifters D' on the way back, D' = diag(e^{i((φ+α)/2 + π/2)}, e^{i((φ−α)/2 + π/2)}),
so that D' B(θ/2) (−1) B(θ/2) D' = R_angles(θ, α, φ). Delays 4, 3, 2, 1, 0 (needs N_b ≥ 6).
"""
function cz_circuit_elements(θ, α, φ)
    hD = -Matrix(Diagonal([(φ + α) / 2 + π / 2, (φ - α) / 2 + π / 2])) .+ 0im     # exp(−i hD) = D'
    hB = ComplexF64[0 -θ/2; -θ/2 0]                                               # exp(−i hB) = B(θ/2)
    hM = π * Matrix{ComplexF64}(I, 2, 2)                                          # exp(−i hM) = −1
    [(4, hD), (3, hB), (2, hM), (1, hB), (0, hD)]
end
