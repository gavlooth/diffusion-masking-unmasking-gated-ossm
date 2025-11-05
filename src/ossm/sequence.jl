module Sequence

import Base
import Lux
import Random

using ..Unit: GatedOscSSMUnit

# Sequence wrapper layers that reuse the single-step unit across time.

"""
    SequenceGatedOSSM(unit)

Wrap a `GatedOscSSMUnit` to act on sequences:
- Input  U ∈ ℝ^{input_dim×N}, columns U[:, t] = u_t
- Output Y ∈ ℝ^{output_dim×N}, columns Y[:, t] = y_t
"""
struct SequenceGatedOSSM <: Lux.AbstractLuxLayer
    unit::GatedOscSSMUnit
end

Lux.initialparameters(rng::Random.AbstractRNG, m::SequenceGatedOSSM) =
    Lux.initialparameters(rng, m.unit)

Lux.initialstates(rng::Random.AbstractRNG, m::SequenceGatedOSSM) =
    Lux.initialstates(rng, m.unit)

function (m::SequenceGatedOSSM)(U::AbstractMatrix{<:Real}, ps, st)
    d_in, N = Base.size(U)

    ys = Vector{Vector{Float32}}(undef, N)
    st_curr = st

    for t = 1:N
        u_t = Base.view(U, :, t)
        y_t, st_next = m.unit(u_t, ps, st_curr)
        ys[t] = y_t
        st_curr = st_next
    end

    Y = Base.reduce(hcat, ys)  # output_dim×N

    return Y, st_curr
end

export SequenceGatedOSSM

end
