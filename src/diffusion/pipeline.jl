module Pipeline

import Random

using ..DiffusionTokenizerModule: DiffusionTokenizer, encode_text_to_digits,
    encode_text_to_digits_with_spans
using ..Masking: forward_mask, protected_columns
using ..Tokenizer: DEFAULT_SPECIALS, TokenSpan

# -----------------------------------------------------------------------------
# Convenience wrappers combining tokenisation and masking
# -----------------------------------------------------------------------------

"""
    diffuse_text_step(tok, rng, text, s; protect_specials = true) -> (ids, Z0, Zt)

Encode `text`, compute the clean digit matrix `Z0`, and sample a masked version
`Zt` using keep probability `s`. Special tokens can optionally be protected from
masking (default true).
"""
function diffuse_text_step(
    tok::DiffusionTokenizer,
    rng::Random.AbstractRNG,
    text::AbstractString,
    s::Float32;
    protect_specials::Bool = true,
)
    ids, Z0 = encode_text_to_digits(tok, text)
    protected_cols = protect_specials ?
        protected_columns(ids, tok.vocab; specials = DEFAULT_SPECIALS) : Int[]
    Zt = forward_mask(rng, tok.codec, Z0, s; protected_cols = protected_cols)
    return ids, Z0, Zt
end

"""
    diffuse_text_step_with_spans(tok, rng, text, s; protect_specials = true)
        -> NamedTuple

Variant of `diffuse_text_step` that returns span metadata to assist with
partial masking/unmasking schedules in diffusion training.
"""
function diffuse_text_step_with_spans(
    tok::DiffusionTokenizer,
    rng::Random.AbstractRNG,
    text::AbstractString,
    s::Float32;
    protect_specials::Bool = true,
)
    ids, Z0, spans = encode_text_to_digits_with_spans(tok, text)
    protected_cols = protect_specials ?
        protected_columns(ids, tok.vocab; specials = DEFAULT_SPECIALS) : Int[]
    Zt = forward_mask(rng, tok.codec, Z0, s; protected_cols = protected_cols)
    return (
        ids = ids,
        clean_digits = Z0,
        noised_digits = Zt,
        spans = spans,
        protected_cols = protected_cols,
    )
end

export diffuse_text_step, diffuse_text_step_with_spans

end
