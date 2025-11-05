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

Represents the affine map
``H_{out} : \\mathbb{R}^{d×N} → \\mathbb{R}^{L×(b+1)×N}``,
``H_{out}(H) = \\mathrm{reshape}( W H + b \\mathbf{1}_N^\\top )`` where
`reshape` is the Julia function that views the data with new dimensions.
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
    logits = Base.reshape(logits_flat, L, V_sub, N) # Julia reshape, no copy

    return logits, st
end

export PrimeOutputHead

end
