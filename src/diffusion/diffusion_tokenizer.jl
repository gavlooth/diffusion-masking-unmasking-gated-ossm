module DiffusionTokenizerModule

using ..Tokenizer:
    simple_tokenize,
    Vocab,
    tokenize,
    encode_tokens,
    decode_ids,
    detokenize,
    TokenizeConfig,
    TokenSpan,
    DEFAULT_SPECIALS
using ..Codec: PrimeCodec, ids_to_digit_matrix, digit_matrix_to_ids

# -----------------------------------------------------------------------------
# DiffusionTokenizer struct
# -----------------------------------------------------------------------------

"""
    DiffusionTokenizer(vocab, codec; config = TokenizeConfig())

Bundle:
- `vocab :: Vocab`      for text ↔ ids,
- `codec :: PrimeCodec` for ids ↔ digit codes,
- `config :: TokenizeConfig` to keep tokenisation consistent at inference.
"""
struct DiffusionTokenizer
    vocab::Vocab
    codec::PrimeCodec
    config::TokenizeConfig
end

DiffusionTokenizer(
    vocab::Vocab,
    codec::PrimeCodec;
    config::TokenizeConfig = TokenizeConfig(),
) = DiffusionTokenizer(vocab, codec, config)

"""
    encode_text_to_digits_with_spans(tok, text)
        -> (ids, Z0, spans)

"""
function encode_text_to_digits(tok::DiffusionTokenizer, text::AbstractString)
    tokens = simple_tokenize(text)              # no spans, no unsafe indexing
    ids = encode_tokens(tok.vocab, tokens)
    Z0 = ids_to_digit_matrix(tok.codec, ids)
    return ids, Z0
end

"""
    decode_digits_to_text(tok, Z) -> String

Assumes Z has shape L×N and contains only digits 0:(base-1) (no masks).
"""
function decode_digits_to_text(tok::DiffusionTokenizer, Z::Matrix{Int})
    ids = digit_matrix_to_ids(tok.codec, Z)
    tokens = decode_ids(tok.vocab, ids)
    return detokenize(tokens)
end

export DiffusionTokenizer, encode_text_to_digits, decode_digits_to_text

end
