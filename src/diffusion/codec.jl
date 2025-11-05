module Codec

import Base

# -----------------------------------------------------------------------------
# PrimeCodec and helpers
# -----------------------------------------------------------------------------

"""
    PrimeCodec(base, L, vocab_size)

Encode an id v ∈ {0,…,vocab_size-1} as an L-tuple of digits in base `base`.
Requires base^L ≥ vocab_size.
"""
struct PrimeCodec
    base::Int       # b
    L::Int          # digits per token
    vocab_size::Int # V
end

"Encode id v ∈ [0, V-1] into an NTuple{L,Int} of base-b digits (left-padded)."
function encode_id(codec::PrimeCodec, v::Int)
    b = codec.base
    L = codec.L
    Base.ntuple(L) do i
        power = L - i
        (v ÷ b^power) % b
    end
end

"Inverse: digits (NTuple{L,Int}) → id v."
function decode_id(codec::PrimeCodec, digits::NTuple{N,Int}) where {N}
    @assert N == codec.L "Digit tuple length N must equal codec.L"
    b = codec.base
    L = codec.L
    v = 0
    for (i, d) in enumerate(digits)
        power = L - i
        v += d * b^power
    end
    @assert 0 ≤ v < codec.vocab_size "Decoded id out of vocab range"
    return v
end

"Integer code for the mask symbol (we use `base` as mask)."
mask_id(codec::PrimeCodec) = codec.base

"""
    ids_to_digit_matrix(codec, ids) -> Matrix{Int}

Given ids (v₁,…,v_N), return an L×N matrix Z where each column is the L-digit
code of the corresponding id.
"""
function ids_to_digit_matrix(codec::PrimeCodec, ids::Vector{Int})
    L = codec.L
    N = length(ids)
    Z = Base.zeros(Int, L, N)
    for (j, v) in enumerate(ids)
        digits = encode_id(codec, v)
        @inbounds for i = 1:L
            Z[i, j] = digits[i]
        end
    end
    return Z
end

"""
    digit_matrix_to_ids(codec, Z::Matrix{Int}) -> Vector{Int}

Inverse of ids_to_digit_matrix, assuming there are no masks and digits
are all in 0:(base-1).
"""
function digit_matrix_to_ids(codec::PrimeCodec, Z::Matrix{Int})
    L, N = Base.size(Z)
    @assert L == codec.L "Digit matrix has wrong number of rows"
    ids = Vector{Int}(undef, N)
    for j = 1:N
        col = ntuple(i -> Z[i, j], L)
        ids[j] = decode_id(codec, col)
    end
    return ids
end

export PrimeCodec,
    encode_id,
    decode_id,
    mask_id,
    ids_to_digit_matrix,
    digit_matrix_to_ids

end
