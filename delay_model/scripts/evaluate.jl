# evaluate.jl — infidelity of saved solutions on their optimisation grid and on refined grids dt/r (same piecewise-
# constant controls, same delay τ = N_b dt), with the extrapolation to dt → 0.
#
# usage: julia --project=. scripts/evaluate.jl [--r 1,2,4,8,16] [--maxgb 4] [--out out.json] <solution.json | dir>...
#   default r = 1 (optimisation grid; compared with the infidelity stored in the file)
#   With ≥ 3 levels r, 2r, 4r: I_inf = 2 I(dt/4r) − I(dt/2r). The scheme is first order in κ dt, and the ratios of
#   successive differences approach 2 from above, so the remaining tail is at most the last difference.
#   RQPN_METHOD=dense|volterra forces one evaluator; RQPN_CHECK=1 runs both where the dense matrix fits.
#
# Two exact evaluators of the same discretised model (they agree to rounding):
#   dense    — the two-photon state as one D×D matrix (D = M(1 + N_b), 16 D² bytes); see RQPNDelay.simulate2.
#   volterra — the Kerr term acts only inside the active two-photon block of a step: U2_k = U1_k⊗U1_k + Δ_k. The state
#              is the linearly propagated input plus the linearly propagated injections y_k = Δ_k vec(X_k)
#              (X_k = active block before step k). Linear two-photon propagation of a block is Γ y Γᵀ with the
#              one-photon propagator Γ, so X_k obeys a discrete Volterra recursion that needs only one-photon vectors:
#              O(D) memory, O(N²) time for N steps. Used when the dense matrix exceeds --maxgb.
using RQPNDelay, Printf, LinearAlgebra, JSON
import RQPNDelay: step2!, active_idx
BLAS.set_num_threads(1)

function parse_args(args)
    rs, maxGB, out, files = [1], 4.0, nothing, String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--r"
            rs = parse.(Int, split(args[i+1], ",")); i += 2
        elseif a == "--maxgb"
            maxGB = parse(Float64, args[i+1]); i += 2
        elseif a == "--out"
            out = args[i+1]; i += 2
        else
            append!(files, isdir(a) ? sort(filter(f -> endswith(f, ".json"), readdir(a; join=true))) : [a]); i += 1
        end
    end
    rs, maxGB, out, files
end
rs, maxGB, out, files = parse_args(ARGS)
check = get(ENV, "RQPN_CHECK", "0") == "1"
force = get(ENV, "RQPN_METHOD", "")

# symmetric two-photon state of the cavity pair (m, n) (as cavity_state2) as Σ c e_a e_bᵀ
pairstate(m, n) = m == n ? [(1.0 + 0im, m, m)] : [(1 / sqrt(2) + 0im, m, n), (1 / sqrt(2) + 0im, n, m)]

function amp2_dense(g::Grid, R, U1s, U2s, inp, tgt)
    Ψ = zeros(ComplexF64, g.D, g.D)
    for (c, a, b) in pairstate(inp...)
        Ψ[a, b] += c
    end
    bufA = zeros(ComplexF64, g.D, 2g.M); bufB = zeros(ComplexF64, g.D, g.M)
    for k in 0:g.nsteps-1
        step2!(Ψ, U1s[k+1], U2s[k+1], R, g.M, mod(k, g.Nb), bufA, bufB)
    end
    sum(conj(c) * Ψ[a, b] for (c, a, b) in pairstate(tgt...))
end

# hot loop of amp2_volterra: carry the injection y of step k (rows of U = one-photon vectors) through steps
# k+1 … N−1 and add Gᵀ y G to the same-block entries of X_j (G = vectors on the active modes of step j)
function propagate_injection!(acc::Array{ComplexF64,3}, U::Matrix{ComplexF64}, y::Matrix{ComplexF64}, nz::Vector{Tuple{Int,Int}},
                              pairs::Vector{Tuple{Int,Int}}, Ms::Vector{Matrix{ComplexF64}}, k::Int, N::Int, Nb::Int, ::Val{M}) where {M}
    n = 2M                                               # compile-time loop bounds (M = 2: 4×4 blocks unrolled)
    G = zeros(ComplexF64, n, n); Tm = zeros(ComplexF64, n, n)
    @inbounds for j in k+1:N-1
        off = mod(j, Nb) * M                             # active modes of step j: 1:M and off .+ (M+1:2M)
        if j - k < Nb                                    # bin not visited since the injection: still empty
            for r in M+1:n, a in 1:n
                U[a, off+r] = 0
            end
        end
        for r in 1:n, a in 1:n
            G[a, r] = U[a, r <= M ? r : off + r]
        end
        fill!(Tm, 0)
        for (a, b) in nz
            yab = y[a, b]
            for s in 1:n
                Tm[a, s] += yab * G[b, s]
            end
        end
        for (r, s) in pairs
            v = zero(ComplexF64)
            for a in 1:n
                v += G[a, r] * Tm[a, s]
            end
            acc[r, s, j+1] += v
        end
        Mj = Ms[j+1]
        for r in 1:n, a in 1:n
            v = zero(ComplexF64)
            for q in 1:n
                v += G[a, q] * Mj[q, r]
            end
            U[a, r <= M ? r : off + r] = v
        end
    end
    nothing
end

function amp2_volterra(g::Grid, R, U1s, U2s, inp, tgt)
    M, N, Nb, D = g.M, g.nsteps, g.Nb, g.D
    n = 2M
    Ŵ = Matrix{ComplexF64}(I, n, n)
    Ŵ[M+1:n, M+1:n] = R                                 # R on the just-emitted bin
    # per-step data (steps with identical propagators share them by reference)
    cM = IdDict{Any,Matrix{ComplexF64}}(); cΔ = IdDict{Any,Matrix{ComplexF64}}()
    Ms = [get!(() -> Matrix(transpose(Ŵ * U1s[k])), cM, U1s[k]) for k in 1:N]   # row vectors: u[A]ᵀ ← u[A]ᵀ Ms
    Δs = [get!(() -> U2s[k] - kron(U1s[k], U1s[k]), cΔ, U2s[k]) for k in 1:N]
    # Δ only couples pairs inside one (cavity m, bin m) block (cross-block columns vanish up to rounding of U2)
    sb(a, b) = mod(a - 1, M) == mod(b - 1, M)
    for Δ in values(cΔ), j in 1:n, i in 1:n
        sb(i, j) || @assert maximum(abs, view(Δ, :, (j-1)*n+i)) < 1e-10
    end
    pairs = [(r, s) for s in 1:n for r in 1:n if sb(r, s)]
    act(k) = active_idx(M, mod(k, Nb))
    acc = zeros(ComplexF64, n, n, N)                     # X_k (same-block entries)
    fin = zeros(ComplexF64, M, M)                        # final cavity block
    # linear part: propagate the input modes from t = 0
    modes = unique([inp...]); nm = length(modes)
    L = zeros(ComplexF64, nm, D)
    for (i, m) in enumerate(modes)
        L[i, m] = 1
    end
    ps = [(c, findfirst(==(a), modes), findfirst(==(b), modes)) for (c, a, b) in pairstate(inp...)]
    for k in 0:N-1
        A = act(k)
        GL = L[:, A]
        for (c, ia, ib) in ps, (r, s) in pairs
            acc[r, s, k+1] += c * GL[ia, r] * GL[ib, s]
        end
        L[:, A] = GL * Ms[k+1]
    end
    for (c, ia, ib) in ps, s in 1:M, r in 1:M
        fin[r, s] += c * L[ia, r] * L[ib, s]
    end
    # injections y_k = Δ_k vec(X_k), propagated as one-photon vectors (rows of U)
    U = zeros(ComplexF64, n, D)
    y = zeros(ComplexF64, n, n)
    @inbounds for k in 0:N-1
        y .= reshape(Δs[k+1] * vec(view(acc, :, :, k + 1)), n, n)
        nz = [(a, b) for b in 1:n for a in 1:n if y[a, b] != 0]
        isempty(nz) && continue
        A = act(k)
        for a in 1:n, r in 1:n
            U[a, A[r]] = Ŵ[r, a]                         # u_a = Ŵ e_a on the active modes of step k
        end
        propagate_injection!(acc, U, y, nz, pairs, Ms, k, N, Nb, Val(M))
        for s in 1:M, r in 1:M, (a, b) in nz
            fin[r, s] += U[a, r] * y[a, b] * U[b, s]
        end
    end
    sum(conj(c) * fin[a, b] for (c, a, b) in pairstate(tgt...))
end

"""1 − F of solution `s` on the grid refined by r; `method` = :dense or :volterra."""
function infidelity(s, r, method)
    g = Grid(s.pr.M, s.g.dt / r, s.g.Nb * r, s.g.nsteps * r)
    U1s, U2s = step_propagators(g, PWC(fill(s.T / s.pr.P, s.pr.P), s.κ, s.δ))
    cache = Dict{Amp,ComplexF64}()
    for (_, lst) in s.task, (_, a) in lst
        haskey(cache, a) && continue
        cache[a] = a.kind == 1 ? simulate1(g, s.R, U1s, cavity_state1(g, a.input[1]))[a.target[1]] :
                   method === :dense ? amp2_dense(g, s.R, U1s, U2s, a.input, a.target) :
                   amp2_volterra(g, s.R, U1s, U2s, a.input, a.target)
    end
    terms = [sg * prod(cache[a] for (_, a) in lst) for (sg, lst) in s.task]
    1 - abs2(sum(terms) / length(terms)), g
end

res = Dict{String,Any}()
for f in files
    s = load_solution(f)
    key = basename(f)
    ent = Dict{String,Any}("T" => s.T, "tau" => s.tau, "kappa_max" => s.pr.κmax, "dt" => s.g.dt, "levels" => Any[])
    Is = Float64[]
    for r in rs
        D = s.pr.M * (1 + s.g.Nb * r)
        fits = 16 * D^2 / 1e9 <= maxGB
        meths = force == "dense" ? [:dense] : force == "volterra" ? [:volterra] : check && fits ? [:dense, :volterra] :
                fits ? [:dense] : [:volterra]
        lev = Dict{String,Any}("r" => r)
        for m in meths
            t = @elapsed I, g = infidelity(s, r, m)
            lev["I_$m"] = I; lev["dt"] = g.dt
            @printf("%s  r = %2d  dt = %.3e  kappa_max*dt = %.4f  %-8s I = %.6e  (%.0f s)\n", key, r, g.dt, s.pr.κmax * g.dt, m, I, t)
            flush(stdout)
        end
        lev["I"] = get(lev, "I_dense", get(lev, "I_volterra", NaN))
        length(meths) == 2 && @printf("%s  r = %2d  |I_dense - I_volterra| = %.2e\n", key, r, abs(lev["I_dense"] - lev["I_volterra"]))
        r == 1 && haskey(s.meta, "infidelity") && @printf("%s  r =  1  stored I = %.6e  |difference| = %.1e\n", key,
                                                       s.meta["infidelity"], abs(lev["I"] - s.meta["infidelity"]))
        push!(ent["levels"], lev); push!(Is, lev["I"])
    end
    if length(Is) >= 3 && all(rs[i+1] == 2rs[i] for i in 1:length(rs)-1)
        I1, I2, I3 = Is[end-2:end]
        ent["I_inf"] = 2I3 - I2; ent["ratio"] = (I2 - I1) / (I3 - I2)
        @printf("%s  RESULT T = %.4f tau = %.5f kappa_max = %g : I(dt/r) = %s  ratio = %.2f  I_inf = %.4e\n", key, s.T, s.tau,
                s.pr.κmax, join([@sprintf("%.4e", x) for x in Is], " / "), ent["ratio"], ent["I_inf"])
    end
    res[key] = ent
    out === nothing || open(io -> JSON.print(io, res, 1), out, "w")
end
