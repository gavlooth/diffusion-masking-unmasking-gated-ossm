module Gating

import Base

using ..Activations: σ

# Utilities for input-dependent gating of the base SSM matrices.

raw"""
    compute_gates(unit, ps, u_t::AbstractVector) -> (gB, gC)

Mathematical definition:
Given parameters ``W_B ∈ ℝ^{2K×d_u}``, ``b_B ∈ ℝ^{2K}``,
``W_C ∈ ℝ^{d_y×d_u}``, ``b_C ∈ ℝ^{d_y}``,
define the maps
```
G_B : ℝ^{d_u} → (0,1)^{2K},\\quad
G_B(u) = σ(W_B u + b_B),                        \\
G_C : ℝ^{d_u} → (0,1)^{d_y},\\quad
G_C(u) = σ(W_C u + b_C).
```
`compute_gates` evaluates these maps.  The return value is a tuple because
Julia allows functions to return multiple outputs without an explicit struct.
"""
function compute_gates(unit, ps, u_t::AbstractVector{<:Real})
    gB = ps.gateB.W * u_t .+ ps.gateB.b
    gB = σ.(gB)

    gC = ps.gateC.W * u_t .+ ps.gateC.b
    gC = σ.(gC)

    return gB, gC
end

"""
    row_scale(M, row_scales) -> Matrix

The map ``R : ℝ^{m×n} × ℝ^m → ℝ^{m×n}`` defined
by ``(R(M, r))_{ij} = r_i · M_{ij}``.  The expression
`row_scales[:, Base.OneTo(C)]` creates a ``m×n`` matrix by broadcasting the
vector over columns.  `Base.OneTo(C)` is Julia's unit range `1:C`.
"""
function row_scale(M::AbstractMatrix{<:Real}, row_scales::AbstractVector{<:Real})
    R, C = Base.size(M)
    @assert Base.length(row_scales) == R "row_scales must match the number of rows"
    return M .* Base.reshape(row_scales, R, 1)
end

"""
    dynamic_BC(unit, ps, gB, gC) -> (B_t, C_t)

Given gates ``g_B ∈ (0,1)^{2K}`` and ``g_C ∈ (0,1)^{d_y}``, compute
``B_t = diag(g_B) B_0`` and ``C_t = diag(g_C) C_0``.
"""
function dynamic_BC(unit, ps, gB::AbstractVector{<:Real}, gC::AbstractVector{<:Real})
    B_t = row_scale(ps.B0, gB)
    C_t = row_scale(ps.C0, gC)
    return B_t, C_t
end

export compute_gates, row_scale, dynamic_BC

end
