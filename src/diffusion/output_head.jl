module OutputHead

import Base
import Lux
import Random

using ..Codec: PrimeCodec

# -----------------------------------------------------------------------------
# Output head: digits logits
# -----------------------------------------------------------------------------

"""
    PrimeOutputHead(codec, d_model)

Lux layer:
- input: H ∈ ℝ^{d_model×N},
- output: logits ∈ ℝ^{L×(b+1)×N},

where:
- L = codec.L,
- b+1 = number of subtoken values (digits 0:(b-1) + mask).
"""
struct PrimeOutputHead <: Lux.AbstractLuxLayer
    codec::PrimeCodec
    d_model::Int
end

function Lux.initialparameters(rng::Random.AbstractRNG, h::PrimeOutputHead)
    L = h.codec.L
    V_sub = h.codec.base + 1
    out_dim = L * V_sub
    W = 0.01f0 .* Random.randn(rng, Float32, out_dim, h.d_model)
    b = Base.zeros(Float32, out_dim)
    return (W = W, b = b)
end

Lux.initialstates(::Random.AbstractRNG, ::PrimeOutputHead) = NamedTuple()

function (h::PrimeOutputHead)(H::AbstractMatrix{<:Real}, ps, st)
    d_model, N = Base.size(H)
    L = h.codec.L
    V_sub = h.codec.base + 1
    out_dim = L * V_sub

    logits_flat = ps.W * H .+ ps.b               # ℝ^{out_dim×N}
    logits = Base.reshape(logits_flat, L, V_sub, N)

    return logits, st
end

export PrimeOutputHead

end
