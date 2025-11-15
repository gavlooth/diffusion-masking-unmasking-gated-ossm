# Prime codec and tokenizer with partial corruption
# ───────────────────────────────────────────────────────────────────────────────

using StatsBase: sample

# Mathematical definition:
# A prime codec assigns each token index i to the i-th prime p_i, yielding an injective
# map C: {1,…,|V|} → ℕ. Encoding a token sequence τ gives the integer sequence
# C(τ). Partial masking replaces selected entries with C([MASK]), while partial
# unmasking exposes leading factors of p_i to leak controlled information.

struct PrimeCodec
    table::Vector{Int}
end

@inline function is_prime(n::Int)
    n <= 1 && return false
    (n == 2) && return true
    iseven(n) && return false
    limit = floor(Int, sqrt(n))
    return all(k -> n % k != 0, 3:2:limit)
end

function first_n_primes(count::Int)
    count < 0 && throw(ArgumentError("prime count must be non-negative"))
    count == 0 && return Int[]
    primes = Int[]
    candidate = 2
    while length(primes) < count
        # Imperative push keeps the sieve O(count log log count) without reallocations.
        is_prime(candidate) && push!(primes, candidate)
        candidate += candidate == 2 ? 1 : 2
    end
    return primes
end

PrimeCodec(count::Int) = PrimeCodec(first_n_primes(count))

@inline encode_prime(codec::PrimeCodec, idx::Int) = codec.table[idx]
@inline decode_prime(codec::PrimeCodec, value::Int) = findfirst(==(value), codec.table)

function prime_factors(n::Int)
    n < 2 && return Int[]
    residual = n
    factors = Int[]
    while iseven(residual)
        # Writing into `factors` is required to keep multiplicities explicitly.
        push!(factors, 2)
        residual ÷= 2
    end
    divisor = 3
    while divisor * divisor <= residual
        while residual % divisor == 0
            push!(factors, divisor)
            residual ÷= divisor
        end
        divisor += 2
    end
    residual > 1 && push!(factors, residual)
    return factors
end

@inline function partial_prime_view(value::Int)
    facs = prime_factors(value)
    facs_isempty = isempty(facs)
    return facs_isempty ? "[GL:∙]" :
           begin
               visible = max(1, cld(length(facs), 2))
               snippet = join(facs[1:visible], "·")
               "[GL:" * snippet * "]"
           end
end

struct PrimeTokenizer{T<:AbstractString}
    vocabulary::Vector{T}
    mask_token::T
    codec::PrimeCodec
    token_to_prime::Dict{T, Int}
    prime_to_token::Dict{Int, T}
end

function PrimeTokenizer(vocab::AbstractVector{T}; mask_token::T = "[MASK]") where {T<:AbstractString}
    vocab_vec = collect(vocab)
    has_mask = any(==(mask_token), vocab_vec)
    vocab_full = has_mask ? vocab_vec : vcat(vocab_vec, mask_token)
    codec = PrimeCodec(length(vocab_full))
    primes = codec.table
    token_to_prime = Dict{T, Int}(zip(vocab_full, primes))
    prime_to_token = Dict{Int, T}(zip(primes, vocab_full))
    return PrimeTokenizer(
        vocab_full,
        mask_token,
        codec,
        token_to_prime,
        prime_to_token,
    )
end

function prime_encode(tokenizer::PrimeTokenizer, tokens::AbstractVector{<:AbstractString})
    mask_prime = tokenizer.token_to_prime[tokenizer.mask_token]
    return map(token -> get(tokenizer.token_to_prime, token, mask_prime), tokens)
end

function prime_decode(tokenizer::PrimeTokenizer, codes::AbstractVector{<:Integer})
    return map(code -> get(tokenizer.prime_to_token, Int(code), tokenizer.mask_token), codes)
end

struct PrimeSample{T<:AbstractString}
    tokens::Vector{T}
    encoded::Vector{Int}
end

function prime_samples(
    tokenizer::PrimeTokenizer,
    sequences::Vector{<:AbstractVector{<:AbstractString}},
)
    return map(sequences) do tokens
        PrimeSample(collect(tokens), prime_encode(tokenizer, tokens))
    end
end

function corrupt_tokens(
    tokenizer::PrimeTokenizer,
    tokens_or_sample;
    mask_fraction::Real = 0.15,
    unmask_fraction::Real = 0.2,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    tokens = tokens_or_sample isa PrimeSample ? tokens_or_sample.tokens : tokens_or_sample
    encoded_source =
        tokens_or_sample isa PrimeSample ? tokens_or_sample.encoded : prime_encode(tokenizer, tokens)
    total = length(tokens)
    encoded = copy(encoded_source)
    mask_total = clamp(round(Int, mask_fraction * total), 0, total)
    mask_total == 0 && return (
        tokens = collect(tokens),
        encoded = encoded,
        mask_indices = Int[],
        glimpse_indices = Int[],
    )
    selected = sort(StatsBase.sample(rng, 1:total, mask_total; replace = false))
    glimpse_total = clamp(round(Int, unmask_fraction * mask_total), 0, mask_total)
    glimpse_indices =
        glimpse_total == 0 ? Int[] :
        sort(StatsBase.sample(rng, selected, glimpse_total; replace = false))
    mask_indices =
        glimpse_total == 0 ? selected : sort(setdiff(selected, glimpse_indices))
    mask_token = tokenizer.mask_token
    mask_prime = tokenizer.token_to_prime[mask_token]
    corrupted_tokens, corrupted_primes = foldl(
        (state, idx) -> begin
            tokens_vec, primes_vec = state
            tokens_vec[idx] = mask_token
            primes_vec[idx] = mask_prime
            return (tokens_vec, primes_vec)
        end,
        mask_indices;
        init = (collect(tokens), copy(encoded)),
    )
    final_tokens = foldl(
        (tok_vec, idx) -> begin
            tok_vec[idx] = partial_prime_view(encoded[idx])
            return tok_vec
        end,
        glimpse_indices;
        init = corrupted_tokens,
    )
    return (
        tokens = final_tokens,
        encoded = corrupted_primes,
        targets = encoded_source,
        mask_indices = sort(mask_indices),
        glimpse_indices = sort(glimpse_indices),
    )
end
