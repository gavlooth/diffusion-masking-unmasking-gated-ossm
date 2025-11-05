module Gating

import Base

using ..Activations: σ

# Utilities for input-dependent gating of the base SSM matrices.

function compute_gates(unit, ps, u_t::AbstractVector{<:Real})
    gB = ps.gateB.W * u_t .+ ps.gateB.b
    gB = σ.(gB)

    gC = ps.gateC.W * u_t .+ ps.gateC.b
    gC = σ.(gC)

    return gB, gC
end

"Row-wise scaling: (diag(row_scales) * M)[i,j] = row_scales[i] * M[i,j]."
function row_scale(M::AbstractMatrix{<:Real}, row_scales::AbstractVector{<:Real})
    R, C = Base.size(M)
    @assert Base.length(row_scales) == R "row_scales must match the number of rows"
    return M .* Base.reshape(row_scales, R, 1)
end

function dynamic_BC(unit, ps, gB::AbstractVector{<:Real}, gC::AbstractVector{<:Real})
    B_t = row_scale(ps.B0, gB)
    C_t = row_scale(ps.C0, gC)
    return B_t, C_t
end

export compute_gates, row_scale, dynamic_BC

end
