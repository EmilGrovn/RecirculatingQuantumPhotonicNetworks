# seeds.jl — initial controls for the delay-aware optimisation (two-cavity blocks, rmode = :angles)
#
# Analytic release/recapture protocol for the delay model. Both cavities of a block use the same coupling κ(t):
# sech-shaped release on [0, L], capture of the returning pulse on [τ, τ+L], closed cavities while the Kerr phase
# accumulates, release on [T−τ−L, T−τ], capture on [T−L, T]. Round trip R = R_angles(π/4, −π/4, φ) (50:50, R² ∝
# diag): one pass bunches |11⟩ into (|20⟩ − |02⟩)/√2 (Hong–Ou–Mandel), the second pass un-bunches it and returns
# single photons to their own cavity. A relative wait phase ∫(δ1 − δ2) = π/2 gives s2 = −i s1, and a Kerr phase π/2
# the CZ. Pulse: emitter κ_e(t) = (γ/2)(1 + tanh(γ(t − t_c)/2)) ≤ γ (sech² intensity, centre t_c = L/2), absorber κ_a =
# time reverse. Truncation error ~ e^{−γL/2}.

"""κ(t), δ(t) (2 × P, uniform bins over [0, T]) of the release/recapture protocol with pulse rate γ and pulse length L."""
function hadamard_controls(T, τ, P, γ, L; dsign=+1.0, κfloor=0.0, κcap=Inf)
    tm = ((1:P) .- 0.5) .* (T / P)
    tc = L / 2
    em(t) = (0 <= t <= L) ? γ / 2 * (1 + tanh(γ * (t - tc) / 2)) : 0.0   # emitter, t from pulse start
    ab(t) = (0 <= t <= L) ? γ / 2 * (1 - tanh(γ * (t - tc) / 2)) : 0.0   # absorber
    tw0, tw1 = τ + L, T - τ - L                                          # closed (waiting) window
    tw = max(tw1 - tw0, 1e-3)
    κ = zeros(2, P); δ = zeros(2, P)
    for (p, t) in enumerate(tm)
        k = max(em(t), ab(t - τ), em(t - (T - τ - L)), ab(t - (T - L)))
        κ[:, p] .= clamp(max(k, κfloor), 0, κcap)
        if tw0 <= t <= tw1
            δ[1, p] = dsign * π / (4tw); δ[2, p] = -δ[1, p]
        end
    end
    κ, δ
end
"""Round trip of the release/recapture protocol."""
hadamard_R(φ=0.0) = R_angles(π / 4, -π / 4, φ)

"""
Release/recapture seed with pulse rate γ = 0.98 κmax and pulse length L = c/γ (overlapping windows merged by max).
Returns the parameter vectors of one block (mirror-symmetric CZ).
"""
function seed_hadamard(T, P, τ, pr::Params; c=18.0, γ=0.98 * pr.κmax)
    κ, δ = hadamard_controls(T, τ, P, γ, c / γ; κfloor=0.0, κcap=0.98 * pr.κmax)
    [pack(pr, κ, δ, zeros(pr.M, pr.M); R0=hadamard_R())]
end

"""
The paper's SLH-optimised CZ pulses (`load_slh_cz`, `slh_cz_blocks`) on P uniform bins over [0, T], centred; the idle padding
gets κ = κidle. Returns one parameter vector per block (both blocks: `cz_task_blocks`; first only: `cz_task_mirror`).
"""
function seed_slh(e, blocks, T, P, pr::Params; κidle=0.02 * pr.κmax)
    t0 = max(0.0, (T - e.T) / 2)
    map(blocks) do (C, c)
        cr = resample_pwc(c, T, P; t0=t0)
        κ = copy(cr.κ)
        κ[κ.<κidle] .= κidle
        pack(pr, clamp.(κ, 0, pr.κmax), clamp.(cr.δ, -0.999pr.δmax, 0.999pr.δmax), C)
    end
end

"""Smooth random controls (a few Fourier components) and a random round trip, for `nblocks` blocks."""
function seed_random(T, P, pr::Params, rng; ncomp=6, nblocks=1)
    M = pr.M
    t = ((1:P) .- 0.5) ./ P
    smooth() = (v = zeros(P); for k in 1:ncomp; v .+= randn(rng) / k .* sin.(k * π .* t .+ 2π * rand(rng)); end; v)
    map(1:nblocks) do _
        κ = [pr.κmax * clamp(0.5 + 0.35 * smooth()[p], 0.01, 0.99) for m in 1:M, p in 1:P]
        δ = [pr.δmax * clamp(0.3 * smooth()[p], -0.95, 0.95) for m in 1:M, p in 1:P]
        if pr.rmode === :angles
            pack(pr, κ, δ, zeros(M, M); angles=(π / 2 * rand(rng), 2π * rand(rng), 2π * rand(rng)))
        else
            c = 0.5 + 2rand(rng)
            pack(pr, κ, δ, [0.0 c; c 0.0])
        end
    end
end
