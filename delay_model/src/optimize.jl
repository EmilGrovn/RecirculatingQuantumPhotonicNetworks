# optimize.jl — time grids, control resampling, the L-BFGS driver and the smoothness penalty

"""
    choose_grid(T, P, τ; dtmax=1e-3, Nbmin=8) -> (nsub, dt, Nb)

Time grid for gate time T with P uniform control bins and delay τ: nsub steps per control bin,
dt = T/(P nsub) ≤ min(dtmax, τ/Nbmin), Nb = round(τ/dt) bins per round trip (the simulated delay is Nb dt).
"""
function choose_grid(T, P, τ; dtmax=1e-3, Nbmin=8)
    target = τ > 0 ? min(dtmax, τ / Nbmin) : dtmax
    nsub = max(1, ceil(Int, T / P / target))
    dt = T / (P * nsub)
    Nb = max(1, round(Int, τ / dt))
    nsub, dt, Nb
end

"""
Resample PWC controls onto P uniform bins over [0, T]; the original pulse starts at `t0`
(κ = δ = 0 before t0 and after its end).
"""
function resample_pwc(c::PWC, T, P; t0=0.0)
    edges = cumsum(c.dur)
    M = size(c.κ, 1)
    κ = zeros(M, P); δ = zeros(M, P)
    for p in 1:P
        t = (p - 0.5) * T / P - t0
        (t < 0 || t > edges[end]) && continue
        i = min(searchsortedfirst(edges, t), length(c.dur))
        κ[:, p] = c.κ[:, i]; δ[:, p] = c.δ[:, i]
    end
    PWC(fill(T / P, P), κ, δ)
end

"""
Map PWC controls (κ, δ on uniform bins) from gate time T_old onto P uniform bins over T_new by cutting time out of
(T_new < T_old) or inserting time into (T_new > T_old, middle value held) the middle of the gate. The release and
capture windows at both ends keep their shape and their timing relative to the gate edges.
"""
function midcut_pwc(κ, δ, T_old, T_new, P)
    P_old = size(κ, 2)
    h = min(T_old, T_new) / 2
    κn = zeros(size(κ, 1), P); δn = zeros(size(δ, 1), P)
    for p in 1:P
        t = (p - 0.5) * T_new / P
        to = t <= h ? t : (t >= T_new - h ? t - T_new + T_old : T_old / 2)
        i = clamp(floor(Int, to / T_old * P_old) + 1, 1, P_old)
        κn[:, p] = κ[:, i]; δn[:, p] = δ[:, i]
    end
    κn, δn
end

"""
    reoptimize(pr, θs, C0s, g, task; iters=1500, target=1e-4, verbose=true) -> (θs_opt, F, trace)

L-BFGS (Optim.jl) on log(1 − F) over the concatenated block parameters; stops when 1 − F < target.
`trace` holds the infidelity of every evaluation.
"""
function reoptimize(pr, θs, C0s, g, task; iters=1500, target=1e-4, verbose=true, show_every=100, m=20)
    nb = length(θs)
    L = [length(θ) for θ in θs]
    split(θ) = [θ[sum(L[1:b-1])+1:sum(L[1:b])] for b in 1:nb]
    cache = Ref{Any}((nothing, 0.0, Float64[]))      # Optim asks for f and ∇f separately: evaluate once
    trace = Float64[]
    function compute(θ)
        if cache[][1] === nothing || cache[][1] != θ
            F, grads, _ = fidelity_and_grad(pr, split(θ), C0s, g, task)
            inf = max(1 - F, 1e-15)
            cache[] = (copy(θ), log(inf), -vcat(grads...) ./ inf)
            push!(trace, inf)
            if verbose && length(trace) % show_every == 1
                @printf("    eval %4d  infidelity %.4e\n", length(trace), inf)
                flush(stdout)
            end
        end
        cache[]
    end
    f(θ) = compute(θ)[2]
    g!(G, θ) = (G .= compute(θ)[3])
    cb(state) = !isempty(trace) && minimum(trace) < target
    res = optimize(f, g!, vcat(θs...), LBFGS(m=m),
                   Optim.Options(iterations=iters, g_tol=1e-10, f_reltol=0.0, f_abstol=0.0, x_abstol=0.0,
                                 callback=cb, allow_f_increases=true))
    θopt = Optim.minimizer(res)
    F, _, _ = fidelity_and_grad(pr, split(θopt), C0s, g, task; want_grad=false)
    split(θopt), F, trace
end

"""Σ (Δⁿv)² along the control bins (rows of v; n = 1 or 2: first or second differences) and its gradient w.r.t. v."""
function diffpenalty(v, n)
    Dv = diff(v, dims=2)
    n == 2 && (Dv = diff(Dv, dims=2))
    g = 2 .* Dv
    for _ in 1:n                                       # apply the transposed difference operator n times
        h = zeros(size(g, 1), size(g, 2) + 1)
        h[:, 1:end-1] .-= g; h[:, 2:end] .+= g
        g = h
    end
    sum(abs2, Dv), g
end

"""
    smoothness(pr, θ; sqrtk=false, order=1) -> (S, ∂S/∂θ)

S = Σ_m Σ_p [((κ_{m,p+1} − κ_{m,p})/κmax)² + ((δ_{m,p+1} − δ_{m,p})/δmax)²] of one block (a discrete ∫κ̇², ∫δ̇²
penalty). The R parameters do not enter.
* `sqrtk = true` adds Σ_m Σ_p ((√κ_{m,p+1} − √κ_{m,p})/√κmax)². With κ = κmax sin²x the κ term has no gradient at
  κ → 0, so it cannot remove jitter of switched-off couplings; √κ = √κmax |sin x| is linear in x there.
* `order = 2` uses second differences (κ_{m,p+1} − 2κ_{m,p} + κ_{m,p−1} etc.) in every term: steep but straight ramps
  cost nothing, wiggles do.
"""
function smoothness(pr::Params, θ; sqrtk=false, order=1)
    M, P = pr.M, pr.P
    x = reshape(θ[1:M*P], M, P); y = reshape(θ[M*P+1:2M*P], M, P)
    Sk, gk = diffpenalty(κmap.(x, pr.κmax) ./ pr.κmax, order)
    Sd, gd = diffpenalty(δmap.(y, pr.δmax) ./ pr.δmax, order)
    G = zeros(length(θ))
    G[1:M*P] = vec(gk .* sin.(2 .* x))                     # ∂(κ/κmax)/∂x = sin 2x
    G[M*P+1:2M*P] = vec(gd .* dδmap.(y, pr.δmax) ./ pr.δmax)
    S = Sk + Sd
    if sqrtk                                               # √κ/√κmax = |sin x|
        Sg, gg = diffpenalty(abs.(sin.(x)), order)
        S += Sg
        G[1:M*P] .+= vec(gg .* cos.(x) .* sign.(sin.(x)))
    end
    S, G
end

"""
    refined_infidelity(pr, θs, C0s, g, task; r=2) -> 1 − F

Infidelity of the same piecewise-constant controls and the same delay τ = Nb dt on the time grid refined by r
(dt/r, r Nb bins per round trip).
"""
function refined_infidelity(pr::Params, θs, C0s, g::Grid, task; r=2)
    gr = Grid(pr.M, g.dt / r, r * g.Nb, r * g.nsteps)
    T = g.nsteps * g.dt
    blocks = map(eachindex(θs)) do b
        _, _, κ, δ, R, _, _ = unpack(pr, θs[b], C0s[b])
        (gr, R, PWC(fill(T / pr.P, pr.P), κ, δ))
    end
    1 - gate_fidelity(blocks, task)[1]
end
