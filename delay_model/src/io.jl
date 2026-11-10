# io.jl — JSON files: the paper's SLH-optimised CZ, and delay-aware (mirror-symmetric CZ) solutions

const RESULTS = normpath(joinpath(@__DIR__, "..", "results"))

rows(A::AbstractMatrix) = [collect(A[m, :]) for m in axes(A, 1)]               # JSON: list of rows
matrix(v) = Matrix{Float64}(reduce(hcat, [Float64.(r) for r in v])')            # inverse of `rows`

"""
    load_slh_cz(path=results/slh_cz_chi3.json) -> (M, C, pwc, T, raw)

The paper's SLH-optimised CZ (four cavities, χ3 nonlinearity) in physical units (written by
`scripts/convert_slh_cz.py` from `src/results/CZ_Chi3_res.pkl`).
"""
function load_slh_cz(path=joinpath(RESULTS, "slh_cz_chi3.json"))
    d = JSON.parsefile(path)
    κ = matrix(d["kappa"]); δ = matrix(d["delta"]); C = matrix(d["C"])
    dur = Float64.(d["dt"])
    (M=Int(d["M"]), C=C, pwc=PWC(dur, κ, δ), T=sum(dur), raw=d)
end

"""
Split the four-cavity CZ into its two independent two-cavity blocks A = (cav1, cav4) and B = (cav2, cav3)
(couplings |C_nm| < 1e-2 between the blocks are dropped). Returns [(C_A, PWC_A), (C_B, PWC_B)].
"""
function slh_cz_blocks(e; drop_small=true)
    sub(idx) = begin
        C = e.C[idx, idx]
        drop_small && (C[abs.(C).<1e-2] .= 0)
        (C, PWC(e.pwc.dur, e.pwc.κ[idx, :], e.pwc.δ[idx, :]))
    end
    [sub([1, 4]), sub([2, 3])]
end

"""Blocks [(R_A, c_A), (R_B, c_B)] of the four-cavity CZ from a mirror-symmetric block A (R_B = σx R_A σx, rows swapped)."""
function mirror_blocks(R, c::PWC)
    σx = [0 1; 1 0]
    [(R, c), (σx * R * σx, PWC(c.dur, c.κ[[2, 1], :], c.δ[[2, 1], :]))]
end

"""
    save_solution(path, pr, θ, g; tau_nominal=g.tau, extra...)

Write a mirror-symmetric CZ solution (one two-cavity block, rmode = :angles) to JSON. Rows of "kappa"/"delta" are
the block's cavities 1 and 2, i.e. κ_1 = κ_3 and κ_4 = κ_2 of the four-cavity gate. `extra` key/value pairs are
stored as well (e.g. infidelities, seed).
"""
function save_solution(path, pr::Params, θ, g::Grid; tau_nominal=g.tau, extra...)
    @assert pr.rmode === :angles && pr.M == 2 "mirror-symmetric CZ solutions with R angles only"
    _, _, κ, δ, R, _, _ = unpack(pr, θ, zeros(2, 2))
    d = Dict{String,Any}("format" => "RQPNDelay CZ solution (mirror-symmetric two-cavity block), v1",
                         "units" => "chi3 = 1; times in 1/chi3, rates in chi3",
                         "T" => g.T, "tau" => g.tau, "tau_nominal" => tau_nominal, "dt" => g.dt, "Nb" => g.Nb,
                         "P" => pr.P, "nsub" => pr.nsub, "kappa_max" => pr.κmax, "delta_max" => pr.δmax,
                         "kappa" => rows(κ), "delta" => rows(δ), "R_angles" => θ[2pr.M*pr.P+1:end],
                         "R_real" => rows(real(R)), "R_imag" => rows(imag(R)))
    for (k, v) in extra
        d[string(k)] = v
    end
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        JSON.print(io, d)
    end
    path
end

"""
    load_solution(path) -> (pr, θ, C0, g, T, tau, κ, δ, R, task, meta)

Read a solution written by `save_solution`: parameters θ (R_angles mode), Grid g and the task `cz_task_mirror()`.
"""
function load_solution(path)
    d = JSON.parsefile(path)
    κ, δ = matrix(d["kappa"]), matrix(d["delta"])
    M, P = size(κ)
    pr = Params(M=M, P=P, nsub=Int(d["nsub"]), κmax=Float64(d["kappa_max"]), δmax=Float64(d["delta_max"]), rmode=:angles)
    angles = Float64.(d["R_angles"])
    θ = pack(pr, κ, δ, zeros(M, M); angles=angles)
    g = Grid(M, Float64(d["dt"]), Int(d["Nb"]), P * pr.nsub)
    (pr=pr, θ=θ, C0=zeros(M, M), g=g, T=Float64(d["T"]), tau=g.tau, κ=κ, δ=δ, R=R_angles(angles...),
     task=cz_task_mirror(), meta=d)
end
