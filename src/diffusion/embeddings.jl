module Embeddings

import Base
import Lux
import Random

using ..Codec: PrimeCodec

# -----------------------------------------------------------------------------
# Subtoken embedding layer
# -----------------------------------------------------------------------------

"""
    PrimeSubtokenEmbedding(codec, d_sub)

Lux layer:
- input: L×N matrix of Ints (digits 0:(b-1) or mask=b),
- output: (L*d_sub)×N matrix of Float32 (concatenated embeddings per token).

Each subtoken (digit/mask) id j has a learnable vector in ℝ^{d_sub};
the L subtokens for a token are embedded and concatenated.
"""
struct PrimeSubtokenEmbedding <: Lux.AbstractLuxLayer
    codec::PrimeCodec
    d_sub::Int
end

function Lux.initialparameters(rng::Random.AbstractRNG, e::PrimeSubtokenEmbedding)
    V_sub = e.codec.base + 1           # digits 0:(b-1) plus mask
    W = 0.01f0 .* Random.randn(rng, Float32, e.d_sub, V_sub)
    return (W = W,)
end

Lux.initialstates(::Random.AbstractRNG, ::PrimeSubtokenEmbedding) = NamedTuple()

function (e::PrimeSubtokenEmbedding)(x_sub, ps, st)
    codec = e.codec
    L = codec.L
    V_sub = codec.base + 1

    # Normalise to an L×N matrix of Ints.
    x_mat = isa(x_sub, AbstractArray) ? x_sub :
        Base.permutedims(hcat(x_sub...))  # from Vector{NTuple} to (L, N)

    N = Base.size(x_mat, 2)

    # For each token (column), embed its L subtokens and concatenate.
    token_vecs = Base.map(1:N) do j
        ids = Base.view(x_mat, :, j)       # length L (view, no copy)
        subvectors = Base.map(ids) do d
            @assert 0 ≤ d ≤ V_sub - 1 "subtoken id out of range"
            ps.W[:, d+1]                 # embedding in ℝ^{d_sub}
        end
        Base.reduce(vcat, subvectors)      # ℝ^{L*d_sub}
    end

    H = Base.reduce(hcat, token_vecs)      # ℝ^{L*d_sub×N}
    return H, st
end

export PrimeSubtokenEmbedding

end
