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

The wrapper realises the map
``F : (ℝ^{d_u})^N → (ℝ^{d_y})^N`` defined by
``F(u_1, …, u_N) = (y_1, …, y_N)`` where each pair ``(y_t, x_{t+1})`` is
obtained by applying the underlying `GatedOscSSMUnit`.
"""
struct SequenceGatedOSSM <: Lux.AbstractLuxLayer
    unit::GatedOscSSMUnit
end

"""
    Lux.initialparameters(rng, m::SequenceGatedOSSM)

Delegates to the underlying unit.  Lux locates this method through multiple
dispatch when initialising the composite layer.
"""
Lux.initialparameters(rng::Random.AbstractRNG, m::SequenceGatedOSSM) =
    Lux.initialparameters(rng, m.unit)

"""
    Lux.initialstates(rng, m::SequenceGatedOSSM)

The hidden state is exactly the hidden state of the single-step unit.
"""
Lux.initialstates(rng::Random.AbstractRNG, m::SequenceGatedOSSM) =
    Lux.initialstates(rng, m.unit)

"""
    (m::SequenceGatedOSSM)(U, ps, st) -> (Y, st')

Implements the map ``F``.  The input matrix `U` stores the columns ``u_t``.
The comprehension-style loop is written with `for t = 1:N`, where `1:N`
constructs a `UnitRange` in Julia (it is inclusive of the endpoint).
`Vector{Vector{Float32}}(undef, N)` allocates a length-``N`` Julia vector
whose entries will later hold the ``y_t`` columns.
"""
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
