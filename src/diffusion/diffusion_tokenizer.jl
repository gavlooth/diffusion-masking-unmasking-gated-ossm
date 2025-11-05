module DiffusionTokenizerModule

using ..Tokenizer: Vocab, tokenize, encode_tokens, decode_ids, detokenize, TokenizeConfig,
    TokenSpan, tokenize_with_spans, DEFAULT_SPECIALS
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

DiffusionTokenizer(vocab::Vocab, codec::PrimeCodec;
    config::TokenizeConfig = TokenizeConfig()) =
        DiffusionTokenizer(vocab, codec, config)

"""
    encode_text_to_digits_with_spans(tok, text)
        -> (ids, Z0, spans)

Tokenize `text`, then tokens → ids → digit matrix, also returning span metadata.
"""
function encode_text_to_digits_with_spans(tok::DiffusionTokenizer, text::AbstractString)
    spans = tokenize_with_spans(text; config = tok.config)
    ids = encode_tokens(tok.vocab, [span.token for span in spans])
    Z0 = ids_to_digit_matrix(tok.codec, ids)
    return ids, Z0, spans
end

"""
    encode_text_to_digits(tok, text) -> (ids, Z0)

Convenience wrapper returning only ids and digit matrix, matching the previous
API for compatibility.
"""
function encode_text_to_digits(tok::DiffusionTokenizer, text::AbstractString)
    ids, Z0, _ = encode_text_to_digits_with_spans(tok, text)
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

export DiffusionTokenizer,
    encode_text_to_digits,
    encode_text_to_digits_with_spans,
    decode_digits_to_text

end
