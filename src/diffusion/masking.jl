module Masking

import Base
import Random

using ..Codec: PrimeCodec, mask_id
using ..Tokenizer: Vocab, DEFAULT_SPECIALS

# -----------------------------------------------------------------------------
# Helpers for selecting protected tokens
# -----------------------------------------------------------------------------

"""
    protected_columns(ids, vocab; specials = DEFAULT_SPECIALS) -> Vector{Int}

Return column indices of tokens whose ids correspond to `specials`. Use this to
prevent `<bos>`, `<eos>`, or custom control tokens from being masked.
"""
function protected_columns(
    ids::AbstractVector{<:Integer},
    vocab::Vocab;
    specials::Vector{String} = DEFAULT_SPECIALS,
)
    protected_ids = Int[]
    for tok in specials
        if haskey(vocab.stoi, tok)
            push!(protected_ids, vocab.stoi[tok])
        end
    end
    protected_set = Set(protected_ids)
    keep_cols = Int[]
    for (idx, token_id) in enumerate(ids)
        if token_id in protected_set
            push!(keep_cols, idx)
        end
    end
    return keep_cols
end

# -----------------------------------------------------------------------------
# Forward diffusion (masking)
# -----------------------------------------------------------------------------

"""
    forward_mask(rng, codec, Z0, s; protected_cols = Int[]) -> Zt

Digit-level masking with optional protected columns. Uses keep probability `s`
and fills dropped digits with the codec's mask id.
"""
function forward_mask(
    rng::Random.AbstractRNG,
    codec::PrimeCodec,
    Z0::Matrix{Int},
    s::Float32;
    protected_cols::AbstractVector{<:Integer} = Int[],
)
    L, N = Base.size(Z0)
    Zt = similar(Z0)
    m_id = mask_id(codec)
    protected = Set(protected_cols)

    for j = 1:N
        col_protected = j in protected
        for i = 1:L
            digit = Z0[i, j]
            if col_protected
                Zt[i, j] = digit
                continue
            end
            r = rand(rng, Float32)
            if r < s
                Zt[i, j] = digit
            else
                Zt[i, j] = m_id
            end
        end
    end

    return Zt
end

# -----------------------------------------------------------------------------
# Reverse diffusion (partial unmasking)
# -----------------------------------------------------------------------------

"""
    partial_unmask(rng, codec, Z_prev, Z_ref, keep_mask_prob) -> Z_next

Given a previously masked matrix `Z_prev` and clean reference `Z_ref`, reveal
a subset of masked digits with probability `1 - keep_mask_prob`. Digits already
known stay untouched. This is useful for gradual denoising schedules.
"""
function partial_unmask(
    rng::Random.AbstractRNG,
    codec::PrimeCodec,
    Z_prev::Matrix{Int},
    Z_ref::Matrix{Int},
    keep_mask_prob::Float32,
)
    @assert Base.size(Z_prev) == Base.size(Z_ref) "Matrices must share shape"
    L, N = Base.size(Z_prev)
    Z_next = similar(Z_prev)
    m_id = mask_id(codec)

    for j = 1:N
        for i = 1:L
            prev_digit = Z_prev[i, j]
            if prev_digit == m_id
                # Decide whether to unmask this position.
                if rand(rng, Float32) < keep_mask_prob
                    Z_next[i, j] = m_id
                else
                    Z_next[i, j] = Z_ref[i, j]
                end
            else
                Z_next[i, j] = prev_digit
            end
        end
    end

    return Z_next
end

export forward_mask, partial_unmask, protected_columns

end
