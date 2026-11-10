# tasks.jl — gates as sums of products of transition amplitudes, and the gate fidelity
#
# A gate is a list of terms (s_n, [(b, a), …]): sign s_n and a product of amplitudes a = ⟨target ⊗ vac|U|input⟩
# of network block b. The fidelity (the gate fidelity of the paper, generalised to the delay model) is
#     F = |(1/N) Σ_n s_n Π a_{n,b}|²,
# i.e. the photons must be back in the cavities, with the right phases, at t = T.

"""Transition amplitude ⟨target ⊗ vac|U|input⟩ with one (kind 1) or two (kind 2) photons; inputs/targets are cavity indices."""
struct Amp
    kind::Int                 # 1 = one photon, 2 = two photons
    input::Tuple{Int,Int}     # cavity indices (second ignored for kind 1)
    target::Tuple{Int,Int}
end

"""
CZ on two independent two-cavity networks (block-diagonal C of the paper's CZ solution): blocks A = {cav1, cav4} and
B = {cav2, cav3}, local indices A: cav1 → 1, cav4 → 2; B: cav2 → 1, cav3 → 2.
Dual-rail logical states |00⟩ = |1010⟩, |01⟩ = |1001⟩, |10⟩ = |0110⟩, |11⟩ = |0101⟩; target CZ = diag(1, 1, 1, −1).
"""
cz_task_blocks() = [
    (+1, [(1, Amp(1, (1, 0), (1, 0))), (2, Amp(1, (2, 0), (2, 0)))]),   # |00⟩: cav1, cav3
    (+1, [(1, Amp(2, (1, 2), (1, 2)))]),                                # |01⟩: cav1, cav4
    (+1, [(2, Amp(2, (1, 2), (1, 2)))]),                                # |10⟩: cav2, cav3
    (-1, [(1, Amp(1, (2, 0), (2, 0))), (2, Amp(1, (1, 0), (1, 0)))]),   # |11⟩: cav4, cav2
]
"""
Mirror-symmetric CZ: block B is the mirror image of block A (cav2 uses A's local-2 controls, cav3 uses A's local-1
controls, R_B = σx R_A σx), so κ3 = κ1 and κ2 = κ4. With s1 = ⟨1,0|U_A|1,0⟩, s2 = ⟨0,1|U_A|0,1⟩, d = ⟨1,1|U_A|1,1⟩
the gate needs only block A:  F = |(s1² + 2d − s2²)/4|².
"""
cz_task_mirror() = [
    (+1, [(1, Amp(1, (1, 0), (1, 0))), (1, Amp(1, (1, 0), (1, 0)))]),
    (+1, [(1, Amp(2, (1, 2), (1, 2)))]),
    (+1, [(1, Amp(2, (1, 2), (1, 2)))]),
    (-1, [(1, Amp(1, (2, 0), (2, 0))), (1, Amp(1, (2, 0), (2, 0)))]),
]
"""CZ on a single four-cavity network (qubit 1: cavities 1, 2; qubit 2: cavities 3, 4)."""
cz_task_full() = [(+1, [(1, Amp(2, (1, 3), (1, 3)))]), (+1, [(1, Amp(2, (1, 4), (1, 4)))]),
                  (+1, [(1, Amp(2, (2, 3), (2, 3)))]), (-1, [(1, Amp(2, (2, 4), (2, 4)))])]
"""CNOT (control: cavities 1, 2; target: cavities 3, 4) on a single four-cavity network."""
cnot_task_full() = [(+1, [(1, Amp(2, (1, 3), (1, 3)))]), (+1, [(1, Amp(2, (1, 4), (1, 4)))]),
                    (+1, [(1, Amp(2, (2, 3), (2, 4)))]), (+1, [(1, Amp(2, (2, 4), (2, 3)))])]

"""
    gate_fidelity(blocks, task; χ=1) -> (F, terms)

`blocks[b] = (Grid, R, PWC)`; the controls need not be aligned with the time grid.
Returns the fidelity and the terms s_n Π a_{n,b}.
"""
function gate_fidelity(blocks, task; χ=1.0)
    need2 = [any(a.kind == 2 && bb == b for (_, lst) in task for (bb, a) in lst) for b in eachindex(blocks)]
    props = [step_propagators(g, c; χ=χ, two=need2[b]) for (b, (g, R, c)) in enumerate(blocks)]
    cache = Dict{Tuple{Int,Amp},ComplexF64}()
    amp(b, a::Amp) = get!(cache, (b, a)) do
        g, R, _ = blocks[b]
        U1s, U2s = props[b]
        if a.kind == 1
            simulate1(g, R, U1s, cavity_state1(g, a.input[1]))[a.target[1]]
        else
            Ψ = simulate2(g, R, U1s, U2s, cavity_state2(g, a.input...))
            sum(conj.(cavity_state2(g, a.target...)) .* Ψ)
        end
    end
    terms = [s * prod(amp(b, a) for (b, a) in lst) for (s, lst) in task]
    abs2(sum(terms) / length(terms)), terms
end
