# slh.jl — SLH (zero-delay) reference model in the one- and two-photon sectors
# H = Σ δ_m n_m + Σ C_nm √(κ_n κ_m) a_n†a_m − χ Σ a_m†a_m†a_m a_m   (paper Eq. 1, χ3 = 1)

slh_h(C, κ, δ) = Diagonal(δ) + C .* sqrt.(κ * κ')

"""SLH propagators: U1 on the cavities, U2 on vec of the M×M symmetric two-photon matrix."""
function slh_propagators(C, c::PWC; χ=1.0)
    M = size(C, 1)
    U1 = Matrix{ComplexF64}(I, M, M)
    U2 = Matrix{ComplexF64}(I, M^2, M^2)
    Id = Matrix{ComplexF64}(I, M, M)
    for p in eachindex(c.dur)
        h = Matrix{ComplexF64}(slh_h(C, c.κ[:, p], c.δ[:, p]))
        G = kron(Id, h) + kron(h, Id)
        for m in 1:M
            G[(m-1)*M+m, (m-1)*M+m] += -2χ
        end
        U1 = exp(-im * c.dur[p] * Hermitian(h)) * U1
        U2 = exp(-im * c.dur[p] * Hermitian(G)) * U2
    end
    U1, U2
end

"""SLH gate fidelity for a task (same format as the delay model); blocks[b] = (C, PWC). Returns (F, terms)."""
function slh_gate_fidelity(blocks, task; χ=1.0)
    props = [slh_propagators(C, c; χ=χ) for (C, c) in blocks]
    function amp(b, a)
        U1, U2 = props[b]
        M = size(blocks[b][1], 1)
        if a.kind == 1
            return U1[a.target[1], a.input[1]]
        else
            st(m, n) = (X = zeros(ComplexF64, M, M); m == n ? (X[m, m] = 1) : (X[m, n] = X[n, m] = 1 / sqrt(2)); X)
            Ψ = reshape(U2 * vec(st(a.input...)), M, M)
            return sum(conj.(st(a.target...)) .* Ψ)
        end
    end
    terms = [s * prod(amp(b, a) for (b, a) in lst) for (s, lst) in task]
    abs2(sum(terms) / length(terms)), terms
end
