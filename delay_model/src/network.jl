# network.jl — round-trip matrix, time grid, controls, and one-/two-photon time stepping

# ─────────────────────────────────────────────────────────────────────────────
# Round-trip matrix R ↔ SLH coupling matrix C
# ─────────────────────────────────────────────────────────────────────────────
"""R = (2C + iI)(2C − iI)⁻¹: round-trip scattering matrix whose SLH (τ → 0) reduction gives C."""
roundtrip_from_C(C::AbstractMatrix) = (2 .* Matrix{ComplexF64}(C) + im * I) / (2 .* Matrix{ComplexF64}(C) - im * I)
"""C = Im[R(I − R)⁻¹]."""
C_from_roundtrip(R::AbstractMatrix) = imag.(R / (I - R))
"""dR = 2(I − R) dC (2C − iI)⁻¹."""
dR_dC(C, R, dC) = 2 .* (I - R) * dC / (2 .* Matrix{ComplexF64}(C) - im * I)

"""
Symmetric 2×2 unitary R = e^{iφ} [e^{iα}cos θ  i sin θ; i sin θ  e^{−iα}cos θ] (two-cavity round trip).
Realised by phase shifters D = diag(e^{i(φ+α)/2}, e^{i(φ−α)/2}) and a coupler: R = D B(θ) D,
B(θ) = [cos θ  i sin θ; i sin θ  cos θ].
"""
R_angles(θ, α, φ) = cis(φ) .* ComplexF64[cis(α)*cos(θ) im*sin(θ); im*sin(θ) cis(-α)*cos(θ)]
"""∂R/∂(θ, α, φ)."""
function dR_angles(θ, α, φ)
    R = R_angles(θ, α, φ)
    dθ = cis(φ) .* ComplexF64[-cis(α)*sin(θ) im*cos(θ); im*cos(θ) -cis(-α)*sin(θ)]
    dα = cis(φ) .* ComplexF64[im*cis(α)*cos(θ) 0; 0 -im*cis(-α)*cos(θ)]
    [dθ, dα, im .* R]
end
"""(θ, α, φ) of a 2×2 symmetric unitary R."""
function angles_from_R(R)
    θ = asin(clamp(abs(R[1, 2]), 0, 1))
    φ = angle(R[1, 2]) - π / 2
    α = angle(R[1, 1]) - φ
    θ, α, φ
end

# ─────────────────────────────────────────────────────────────────────────────
# Grid and controls
# ─────────────────────────────────────────────────────────────────────────────
"""Time grid of one network: M cavities, bin width dt, Nb bins per round trip (τ = Nb dt), nsteps steps (T = nsteps dt)."""
struct Grid
    M::Int
    dt::Float64
    Nb::Int
    nsteps::Int
end
Base.getproperty(g::Grid, s::Symbol) = s === :tau ? getfield(g, :Nb) * getfield(g, :dt) :
                                       s === :D ? getfield(g, :M) * (1 + getfield(g, :Nb)) :
                                       s === :T ? getfield(g, :nsteps) * getfield(g, :dt) : getfield(g, s)

"""make_grid(M, T, dt, tau): N_b = max(1, round(tau/dt)), nsteps = round(T/dt)."""
make_grid(M, T, dt, tau) = Grid(M, dt, max(1, round(Int, tau / dt)), round(Int, T / dt))

"""Piecewise-constant controls: segment p has duration dur[p], couplings κ[:, p], detunings δ[:, p]."""
struct PWC
    dur::Vector{Float64}
    κ::Matrix{Float64}   # M × P
    δ::Matrix{Float64}   # M × P
end

"""
For each step, the list of (duration, segment index) covering [k dt, (k+1) dt].
Time beyond the last segment gets zero controls (index 0).
"""
function step_controls(g::Grid, c::PWC)
    edges = [0.0; cumsum(c.dur)]
    out = Vector{Vector{Tuple{Float64,Int}}}(undef, g.nsteps)
    for k in 0:g.nsteps-1
        t0, t1 = k * g.dt, (k + 1) * g.dt
        segs = Tuple{Float64,Int}[]
        for p in 1:length(c.dur)
            a, b = max(t0, edges[p]), min(t1, edges[p+1])
            b - a > 1e-9 * g.dt && push!(segs, (b - a, p))
        end
        rest = t1 - max(t0, edges[end])
        rest > 1e-9 * g.dt && push!(segs, (rest, 0))
        out[k+1] = segs
    end
    out
end

# ─────────────────────────────────────────────────────────────────────────────
# Step propagators on the active modes [cavities; bins at the active ring position]
# ─────────────────────────────────────────────────────────────────────────────
# single-particle Hamiltonian on the 2M active modes
function h_active(M, dt, κ::AbstractVector, δ::AbstractVector)
    h = zeros(ComplexF64, 2M, 2M)
    for m in 1:M
        h[m, m] = δ[m]
        g = sqrt(max(κ[m], 0.0) / dt)
        h[m, M+m] = -im * g      # a_m† w_m
        h[M+m, m] = +im * g      # w_m† a_m
    end
    h
end

# two-photon generator on vec(Ψ_AA) (column-major):  hΨ + Ψhᵀ + Kerr
function G_active(h::AbstractMatrix, M, χ)
    n = size(h, 1)
    Id = Matrix{ComplexF64}(I, n, n)
    G = kron(Id, h) + kron(h, Id)
    for m in 1:M
        G[(m-1)*n+m, (m-1)*n+m] += -2χ
    end
    G
end

# exp(−i H t) with the eigen-data needed for its Fréchet derivative
struct ExpData
    U::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
    Φ::Matrix{ComplexF64}   # divided differences of exp at x = −i ε t
end
function expdata(H::AbstractMatrix, t)
    F = eigen(Hermitian(Matrix(H)))
    x = -im .* F.values .* t
    ex = exp.(x)
    n = length(x)
    Φ = Matrix{ComplexF64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        d = x[i] - x[j]
        Φ[i, j] = abs(d) < 1e-8 ? ex[j] * (1 + d / 2 + d^2 / 6) : (ex[i] - ex[j]) / d
    end
    ExpData(F.vectors * Diagonal(ex) * F.vectors', F.vectors, Φ)
end

struct SegProp
    e1::ExpData          # one photon (2M × 2M)
    e2::ExpData          # two photons ((2M)² × (2M)²)
end
segprop(M, dt, χ, κ, δ, t) = (h = h_active(M, dt, κ, δ); SegProp(expdata(h, t), expdata(G_active(h, M, χ), t)))

"""Per-step propagators U1 (2M×2M) and U2 ((2M)²×(2M)²) for arbitrary (not grid-aligned) PWC controls."""
function step_propagators(g::Grid, c::PWC; χ=1.0, two=true)
    sc = step_controls(g, c)
    M = g.M
    cache = Dict{Tuple{Int,Float64},SegProp}()
    getseg(p, t) = get!(cache, (p, t)) do
        κ = p == 0 ? zeros(M) : c.κ[:, p]
        δ = p == 0 ? zeros(M) : c.δ[:, p]
        segprop(M, g.dt, χ, κ, δ, t)
    end
    # steps with identical segment lists share (by reference) the same propagator matrices
    stepcache = Dict{Vector{Tuple{Float64,Int}},Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()
    U1s = Vector{Matrix{ComplexF64}}(undef, g.nsteps)
    U2s = Vector{Matrix{ComplexF64}}(undef, g.nsteps)
    Id2 = Matrix{ComplexF64}(I, 4M^2, 4M^2)
    for k in 1:g.nsteps
        key = [(round(t, sigdigits=12), p) for (t, p) in sc[k]]
        U1s[k], U2s[k] = get!(stepcache, key) do
            U1 = Matrix{ComplexF64}(I, 2M, 2M)
            U2 = two ? copy(Id2) : Id2
            for (t, p) in key
                s = getseg(p, t)
                U1 = s.e1.U * U1
                two && (U2 = s.e2.U * U2)
            end
            (U1, U2)
        end
    end
    U1s, U2s
end

active_idx(M, p) = vcat(1:M, M+p*M+1:M+p*M+M)
bin_idx(M, p) = collect(M+p*M+1:M+p*M+M)

# ─────────────────────────────────────────────────────────────────────────────
# States
# ─────────────────────────────────────────────────────────────────────────────
"""One photon in cavity m, waveguides empty."""
cavity_state1(g::Grid, m) = (ψ = zeros(ComplexF64, g.D); ψ[m] = 1; ψ)
"""One photon in cavity m and one in cavity n (two in m if m == n), waveguides empty."""
function cavity_state2(g::Grid, m, n)
    Ψ = zeros(ComplexF64, g.D, g.D)
    if m == n
        Ψ[m, m] = 1
    else
        Ψ[m, n] = Ψ[n, m] = 1 / sqrt(2)
    end
    Ψ
end

# ─────────────────────────────────────────────────────────────────────────────
# Time stepping
# ─────────────────────────────────────────────────────────────────────────────
"""Ψ ↦ VΨVᵀ with V = U on index set `idx` and the identity elsewhere. `blk` (optional) = new idx×idx block."""
function sym_update!(Ψ, idx, U, buf; blk=nothing)
    old = blk === nothing ? Ψ[idx, idx] : nothing
    cols = view(Ψ, :, idx)
    mul!(buf, cols, transpose(U))
    cols .= buf
    Ψ[idx, :] .= transpose(buf)
    Ψ[idx, idx] .= blk === nothing ? U * old * transpose(U) : blk
    Ψ
end
"""One step of the two-photon state: E_k on [cavities; bins at p] (Kerr exact in the active block), then R on the bins at p."""
function step2!(Ψ, U1, U2, R, M, p, bufA, bufB)
    A = active_idx(M, p)
    n = 2M
    newAA = reshape(U2 * vec(Ψ[A, A]), n, n)
    sym_update!(Ψ, A, U1, bufA; blk=newAA)
    sym_update!(Ψ, bin_idx(M, p), R, bufB)
end
"""One step of the one-photon state."""
function step1!(ψ, U1, R, M, p)
    A = active_idx(M, p)
    ψ[A] = U1 * ψ[A]
    B = bin_idx(M, p)
    ψ[B] = R * ψ[B]
    ψ
end

"""Evolve a one-photon state over the grid; returns the final ψ."""
function simulate1(g::Grid, R, U1s, ψ0)
    ψ = copy(ψ0)
    for k in 0:g.nsteps-1
        step1!(ψ, U1s[k+1], R, g.M, mod(k, g.Nb))
    end
    ψ
end
"""Evolve a two-photon state over the grid; returns the final Ψ."""
function simulate2(g::Grid, R, U1s, U2s, Ψ0)
    Ψ = copy(Ψ0)
    bufA = zeros(ComplexF64, g.D, 2g.M); bufB = zeros(ComplexF64, g.D, g.M)
    for k in 0:g.nsteps-1
        step2!(Ψ, U1s[k+1], U2s[k+1], R, g.M, mod(k, g.Nb), bufA, bufB)
    end
    Ψ
end
