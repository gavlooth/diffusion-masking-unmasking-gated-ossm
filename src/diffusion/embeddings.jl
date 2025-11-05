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

Defines the map
``E : \\{0,\\dots,b\\}^{L×N} → \\mathbb{R}^{(L d_{sub})×N}``,
``E(Z) = [e_{z_{1,1}}; …; e_{z_{L,1}}] | … | [e_{z_{1,N}}; …; e_{z_{L,N}}]``,
where the embeddings ``e_j ∈ \\mathbb{R}^{d_{sub}}`` are trainable.
"""
struct PrimeSubtokenEmbedding <: Lux.AbstractLuxLayer
    codec::PrimeCodec
    d_sub::Int
end

"""
    Lux.initialparameters(rng, e::PrimeSubtokenEmbedding)

Returns a `NamedTuple` with a single field `W`.  The matrix has size
``d_{sub} × (b+1)``: one column per digit plus the mask.
"""
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
    # The expression `hcat(x_sub...)` concatenates tuple columns, while
    # `Base.permutedims` swaps axes so that digits become rows.
    x_mat = isa(x_sub, AbstractArray) ? x_sub :
        Base.permutedims(hcat(x_sub...))  # from Vector{NTuple} to (L, N)

    N = Base.size(x_mat, 2)

    # For each token (column), embed its L subtokens and concatenate.
    # `Base.map(1:N) do j` constructs a vector whose j-th entry is the column
    # embedding of token j.
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
