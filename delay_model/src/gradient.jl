# gradient.jl — optimisation parameters and the gate fidelity with its exact (adjoint) gradient
#
# Parameterisation (control bins aligned with the time grid: control bin p ↔ steps (p−1)nsub+1 … p nsub)
#   coupling amplitude g = √κ = √κmax |sin x|      (κ ∈ [0, κmax]; ∂g/∂x ≠ 0 at κ = 0)
#   detuning δ = δmax (2/π) atan(y) ∈ (−δmax, δmax)
#   round trip R:  rmode = :angles → (M = 2) R = R_angles(θ, α, φ)  (used for all results of the paper)
#                  rmode = :C      → R = Cayley(C), trainable entries of C given by Cmask (i ≤ j)

Base.@kwdef struct Params
    M::Int
    P::Int                 # number of control bins
    nsub::Int              # time steps per control bin
    κmax::Float64 = 15.0
    δmax::Float64 = 7.5
    rmode::Symbol = :angles
    Cmask::Matrix{Bool} = zeros(Bool, 2, 2)
end
gmap(x, κmax) = sqrt(κmax) * abs(sin(x))
dgmap(x, κmax) = sqrt(κmax) * cos(x) * sign(sin(x))
κmap(x, κmax) = gmap(x, κmax)^2
κinv(κ, κmax) = asin(sqrt(clamp(κ / κmax, 0.0, 1.0)))
δmap(y, δmax) = δmax * 2 / π * atan(y)
dδmap(y, δmax) = δmax * 2 / π / (1 + y^2)
δinv(δ, δmax) = tan(π / 2 * clamp(δ / δmax, -1 + 1e-7, 1 - 1e-7))
nRpar(pr::Params) = pr.rmode === :angles ? 3 : count(pr.Cmask[i, j] for j in 1:pr.M for i in 1:j)
nparams(pr::Params) = 2pr.M * pr.P + nRpar(pr)

"""
    unpack(pr, θ, C0) -> (x, y, κ, δ, R, dRs, C)

θ = [x (M×P, column-major); y (M×P); R parameters]. dRs = ∂R/∂(R parameters).
C = C0 with the trained entries in :C mode, Im[R(I − R)⁻¹] in :angles mode.
"""
function unpack(pr::Params, θ, C0)
    M, P = pr.M, pr.P
    x = reshape(θ[1:M*P], M, P)
    y = reshape(θ[M*P+1:2M*P], M, P)
    κ = κmap.(x, pr.κmax)
    δ = δmap.(y, pr.δmax)
    ρ = θ[2M*P+1:end]
    if pr.rmode === :angles
        R = R_angles(ρ...)
        dRs = dR_angles(ρ...)
        C = abs(det(I - R)) > 1e-12 ? C_from_roundtrip(R) : fill(Inf, M, M)
    else
        C = copy(Matrix{Float64}(C0))
        k = 0
        E = Matrix{Float64}[]
        for j in 1:M, i in 1:j
            if pr.Cmask[i, j]
                k += 1
                C[i, j] = C[j, i] = ρ[k]
                Ek = zeros(M, M); Ek[i, j] = Ek[j, i] = 1
                push!(E, Ek)
            end
        end
        R = roundtrip_from_C(C)
        dRs = [dR_dC(C, R, Ek) for Ek in E]
    end
    x, y, κ, δ, R, dRs, C
end
"""
    pack(pr, κ, δ, C0; R0=nothing, angles=nothing) -> θ

Inverse of `unpack`. :C mode: C0 = full C. :angles mode: the angles are `angles` if given, else those of R0,
else those of Cayley(C0).
"""
function pack(pr::Params, κ, δ, C0; R0=nothing, angles=nothing)
    θ = [vec(κinv.(κ, pr.κmax)); vec(δinv.(δ, pr.δmax))]
    if pr.rmode === :angles
        append!(θ, angles !== nothing ? collect(angles) : collect(angles_from_R(R0 === nothing ? roundtrip_from_C(C0) : R0)))
    else
        for j in 1:pr.M, i in 1:j
            pr.Cmask[i, j] && push!(θ, C0[i, j])
        end
    end
    θ
end

# Fréchet derivative of exp: K = V (Φᵀ ∘ V†WV) V†, so that tr(L(−iHt, −it dH) W) = −it Σ_ab dH_ab K_ba
function frechet_kernel(e::ExpData, W::AbstractMatrix)
    Wt = e.V' * W * e.V
    e.V * (transpose(e.Φ) .* Wt) * e.V'
end
# Σ_ab lift2(dh)_ab K_ba for sparse dh given as ((a, b, val), …);  lift2(dh) = I⊗dh + dh⊗I (n×n blocks)
function lift2_contract(K, ents, n)
    s = zero(ComplexF64)
    @inbounds for (i, j, v) in ents, r in 1:n
        s += v * (K[(r-1)*n+j, (r-1)*n+i] + K[(j-1)*n+r, (i-1)*n+r])
    end
    s
end

"""
    fidelity_and_grad(pr, θblocks, C0blocks, g, task; χ=1, want_grad=true) -> (F, grads, terms)

Gate fidelity and its exact gradient with respect to the parameters of every block. All blocks share the
Grid `g` (g.nsteps == pr.P * pr.nsub).
"""
function fidelity_and_grad(pr::Params, θblocks, C0blocks, g::Grid, task; χ=1.0, want_grad=true)
    M, P = pr.M, pr.P
    @assert g.nsteps == P * pr.nsub "nsteps must equal P*nsub"
    nb = length(θblocks)
    dat = map(1:nb) do b
        x, y, κ, δ, R, dRs, C = unpack(pr, θblocks[b], C0blocks[b])
        (x=x, y=y, κ=κ, δ=δ, R=R, dRs=dRs,
         segs=[segprop(M, g.dt, χ, κ[:, p], δ[:, p], g.dt) for p in 1:P])
    end
    amps = unique([(b, a) for (_, lst) in task for (b, a) in lst])
    res = Dict(ba => amp_and_grad(pr, g, dat[ba[1]], ba[2]; want_grad=want_grad) for ba in amps)
    N = length(task)
    terms = [s * prod(res[(b, a)][1] for (b, a) in lst) for (s, lst) in task]
    A = sum(terms) / N
    F = abs2(A)
    want_grad || return F, nothing, terms
    grads = [zeros(length(θblocks[b])) for b in 1:nb]
    for (s, lst) in task, (j, (b, a)) in enumerate(lst)
        other = one(ComplexF64)
        for (jj, (bb, aa)) in enumerate(lst)
            jj != j && (other *= res[(bb, aa)][1])
        end
        grads[b] .+= 2 .* real.(conj(A) .* (s * other / N) .* res[(b, a)][2])
    end
    F, grads, terms
end

"""
Amplitude a = ⟨target ⊗ vac|U|input⟩ and its complex gradient with respect to the block parameters.
Forward sweep, then a backward sweep that undoes each (unitary) step on the state and on the propagated target, so
the trajectory is never stored.
"""
function amp_and_grad(pr::Params, g::Grid, d, a::Amp; want_grad=true)
    M, P, nsub = pr.M, pr.P, pr.nsub
    R = d.R
    two = a.kind == 2
    n = 2M
    bufA = zeros(ComplexF64, g.D, n); bufB = zeros(ComplexF64, g.D, M)
    # ---------------- forward
    if two
        X = cavity_state2(g, a.input...)
        for k in 0:g.nsteps-1
            s = d.segs[k÷nsub+1]
            step2!(X, s.e1.U, s.e2.U, R, M, mod(k, g.Nb), bufA, bufB)
        end
        Λ = cavity_state2(g, a.target...)
    else
        X = cavity_state1(g, a.input[1])
        for k in 0:g.nsteps-1
            step1!(X, d.segs[k÷nsub+1].e1.U, R, M, mod(k, g.Nb))
        end
        Λ = zeros(ComplexF64, g.D); Λ[a.target[1]] = 1
    end
    val = sum(conj.(Λ) .* X)
    want_grad || return val, ComplexF64[]
    # ---------------- backward sweep
    # da = Σ_k ⟨Λ_k|∂P_R E_k|X_{k−1}⟩ + ⟨Λ'_k|∂E_k|X_{k−1}⟩,  Λ'_k = P_R†Λ_k
    W1 = [zeros(ComplexF64, n, n) for _ in 1:P]
    W2 = two ? [zeros(ComplexF64, n^2, n^2) for _ in 1:P] : nothing
    WR = zeros(ComplexF64, M, M)
    Rd = Matrix(R')
    for k in g.nsteps-1:-1:0
        p = mod(k, g.Nb)
        pb = k ÷ nsub + 1
        s = d.segs[pb]
        B = bin_idx(M, p)
        A = active_idx(M, p)
        if two
            WR .+= 2 .* (transpose(X[:, B]) * conj(Λ[:, B]))     # 2 tr(∂R R† X_k[B,:] Λ_k[B,:]†)
            sym_update!(X, B, Rd, bufB)
            sym_update!(Λ, B, Rd, bufB)
            Ud1 = Matrix(s.e1.U')
            sym_update!(X, A, Ud1, bufA; blk=reshape(s.e2.U' * vec(X[A, A]), n, n))
            XA = transpose(X[:, A]); ΛA = transpose(Λ[:, A])
            W1[pb] .+= 2 .* (XA * ΛA' .- X[A, A] * Λ[A, A]')      # 2 X[A,S] Λ'[A,S]†
            W2[pb] .+= vec(X[A, A]) * vec(Λ[A, A])'
            sym_update!(Λ, A, Ud1, bufA; blk=reshape(s.e2.U' * vec(Λ[A, A]), n, n))
        else
            WR .+= X[B] * Λ[B]'
            X[B] = Rd * X[B]
            Λ[B] = Rd * Λ[B]
            X[A] = s.e1.U' * X[A]
            W1[pb] .+= X[A] * Λ[A]'
            Λ[A] = s.e1.U' * Λ[A]
        end
    end
    # ---------------- contract with the parameter derivatives
    # tr(L(−iHt, −it dH) W) = −it Σ_ab dH_ab K_ba with one kernel K per control bin; each parameter then costs
    # O(nnz(dH)) (dH has 1–2 nonzeros, lift2(dH) O(n))
    grad = zeros(ComplexF64, nparams(pr))
    sq = 1 / sqrt(g.dt)
    mit = -im * g.dt
    for p in 1:P
        s = d.segs[p]
        K1 = frechet_kernel(s.e1, W1[p])
        K2 = two ? frechet_kernel(s.e2, W2[p]) : nothing
        for m in 1:M
            # coupling amplitude g_m:  h[m,M+m] = −i g/√dt, h[M+m,m] = +i g/√dt
            ents = ((m, M + m, -im * sq), (M + m, m, im * sq))
            dv = mit * sum(v * K1[b, a] for (a, b, v) in ents)
            two && (dv += mit * lift2_contract(K2, ents, n))
            grad[(p-1)*M+m] = dv * dgmap(d.x[m, p], pr.κmax)
            ents = ((m, m, 1.0 + 0im),)
            dv = mit * K1[m, m]
            two && (dv += mit * lift2_contract(K2, ents, n))
            grad[M*P+(p-1)*M+m] = dv * dδmap(d.y[m, p], pr.δmax)
        end
    end
    for (k, dR) in enumerate(d.dRs)
        grad[2M*P+k] = tr(dR * R' * WR)
    end
    val, grad
end
