module Tokenizer

import Base

"Special tokens reserved at start of vocab."
const DEFAULT_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]

# -----------------------------------------------------------------------------
# Token data structures
# -----------------------------------------------------------------------------

"Character span metadata for a token; useful for partial masking/unmasking."
struct TokenSpan
    token::String          # normalised token used inside the model
    surface::String        # original-case surface form
    span::UnitRange{Int}   # 1-based inclusive character span in original text
end

"""
    TokenizeConfig(; preserve_case=false, special_tokens=DEFAULT_SPECIALS, keep_whitespace=false)

Configuration for the diffusion tokenizer:
- `preserve_case`: keep original case instead of lowercasing (defaults to false).
- `special_tokens`: tokens (e.g. `<mask>`, `<partial_mask>`) that should remain intact.
- `keep_whitespace`: emit explicit whitespace tokens to support very fine-grained masking.
"""
struct TokenizeConfig
    preserve_case::Bool
    special_tokens::Vector{String}
    keep_whitespace::Bool
end

TokenizeConfig(; preserve_case::Bool = false,
    special_tokens::Vector{String} = DEFAULT_SPECIALS,
    keep_whitespace::Bool = false) =
        TokenizeConfig(preserve_case, special_tokens, keep_whitespace)


# -----------------------------------------------------------------------------
# Core vocabulary definitions
# -----------------------------------------------------------------------------

"Simple vocabulary: stoi (token → id), itos (id → token)."
struct Vocab
    stoi::Dict{String,Int}   # string → id in 0:(V-1)
    itos::Vector{String}     # id → string, index = id+1
end

# -----------------------------------------------------------------------------
# Tokenisation utilities
# -----------------------------------------------------------------------------

# Escape tokens so they can be inserted safely into Regex patterns.
_escape_for_regex(tok::AbstractString) = replace(tok, r"([\\.^$|?*+()\[\]{})])" => s"\\\1")

function _build_token_regex(cfg::TokenizeConfig)
    specials = isempty(cfg.special_tokens) ? "" :
        join(_escape_for_regex.(sort(cfg.special_tokens; by = length, rev = true)), "|")

    components = String[]
    if !isempty(specials)
        push!(components, specials)
    end

    # Numbers with optional decimals, e.g. 12.34 or 1_000.
    push!(components, "[0-9]+(?:_[0-9]+)*(?:\\.[0-9]+)?")

    # Words with internal apostrophes/underscores/hyphens.
    push!(components, "[A-Za-z0-9_]+(?:['’_-][A-Za-z0-9_]+)*")

    # Single non-space ASCII punctuation/symbol characters.
    push!(components, "[^\\sA-Za-z0-9_]")

    pattern = join(components, "|")
    return Regex(pattern)
end

"""
    tokenize(text; config = TokenizeConfig())

Tokenise `text` into normalised tokens suitable for the diffusion pipeline.
Returns a `Vector{String}` of canonical tokens.
"""
function tokenize(text::AbstractString; config::TokenizeConfig = TokenizeConfig())
    spans = tokenize_with_spans(text; config = config)
    return [span.token for span in spans]
end

"""
    tokenize_with_spans(text; config = TokenizeConfig()) -> Vector{TokenSpan}

Tokenise while keeping character span metadata and the original surface form.
This is helpful when you need to partially mask/unmask spans of the text
for diffusion training.
"""
function tokenize_with_spans(text::AbstractString; config::TokenizeConfig = TokenizeConfig())
    pattern = _build_token_regex(config)
    raw = String(text)
    processed = config.preserve_case ? raw : Base.lowercase(raw)
    spans = TokenSpan[]

    for m in eachmatch(pattern, processed)
        start_idx = m.offset
        stop_idx = start_idx + ncodeunits(m.match) - 1
        token = m.match

        surface = raw[start_idx:stop_idx]
        norm_token = config.preserve_case ? token : Base.lowercase(token)

        push!(spans, TokenSpan(norm_token, surface, start_idx:stop_idx))
    end

    if config.keep_whitespace
        spans = _inject_whitespace_tokens(spans, raw, config)
    end

    return spans
end

function _inject_whitespace_tokens(spans::Vector{TokenSpan}, raw::String, cfg::TokenizeConfig)
    enriched = TokenSpan[]
    cursor = 1
    for span in spans
        if span.span.start > cursor
            ws = raw[cursor:span.span.start-1]
            push!(enriched, TokenSpan(ws, ws, cursor:span.span.start-1))
        end
        push!(enriched, span)
        cursor = span.span.stop + 1
    end
    if cursor <= lastindex(raw)
        ws = raw[cursor:end]
        push!(enriched, TokenSpan(ws, ws, cursor:lastindex(raw)))
    end
    return enriched
end


# -----------------------------------------------------------------------------
# Vocab construction and conversions
# -----------------------------------------------------------------------------

"""
    build_vocab(texts; vocab_size, specials = DEFAULT_SPECIALS, config = TokenizeConfig())

Build a vocabulary from `texts`. Returns `Vocab`.
Special tokens are pinned at the front with their provided order.
"""
function build_vocab(
    texts::Vector{<:AbstractString};
    vocab_size::Int = 10_000,
    specials::Vector{String} = DEFAULT_SPECIALS,
    config::Union{Nothing,TokenizeConfig} = nothing,
)
    config === nothing && (config = TokenizeConfig(special_tokens = specials))
    counts = Dict{String,Int}()
    for txt in texts
        tokens = tokenize(txt; config = config)
        for tok in tokens
            counts[tok] = get(counts, tok, 0) + 1
        end
    end

    sorted = sort(collect(counts); by = x -> (-x[2], x[1]))
    keep = first(sorted, max(0, vocab_size - length(specials)))
    vocab_tokens = vcat(specials, [p[1] for p in keep])

    stoi = Dict{String,Int}(tok => i - 1 for (i, tok) in enumerate(vocab_tokens))
    itos = Vector{String}(vocab_tokens)

    # Ensure specials stay at fixed ids 0,1,2,...
    for (fixed_id, tok) in enumerate(specials)
        stoi[tok] = fixed_id - 1
        itos[fixed_id] = tok
    end

    return Vocab(stoi, itos)
end

"Return the id of `<unk>` in this vocab."
unk_id(v::Vocab) = v.stoi["<unk>"]

"Token string → id (0-based), with fallback to <unk>."
function encode_token(v::Vocab, tok::String)
    get(v.stoi, tok, unk_id(v))
end

"Vector of tokens → vector of ids."
function encode_tokens(v::Vocab, toks::Vector{String})
    uid = unk_id(v)
    [get(v.stoi, tok, uid) for tok in toks]
end

"Vector of ids → vector of token strings."
function decode_ids(v::Vocab, ids::Vector{Int})
    [v.itos[id+1] for id in ids]
end

"""
    detokenize(tokens::Vector{String}) -> String

Naive detokenization:
- join words with spaces,
- glue punctuation to the previous word.
"""
function detokenize(tokens::Vector{String})
    buff = IOBuffer()
    prev_alnum = false
    for tok in tokens
        is_alnum = !isempty(tok) && all(c -> Base.isalnum(c) || c == '_', tok)
        if is_alnum
            if prev_alnum
                print(buff, ' ')
            end
            print(buff, tok)
        else
            print(buff, tok)
        end
        prev_alnum = is_alnum
    end
    return String(take!(buff))
end


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

export TokenSpan,
    TokenizeConfig,
    Vocab,
    DEFAULT_SPECIALS,
    tokenize,
    tokenize_with_spans,
    build_vocab,
    encode_token,
    encode_tokens,
    decode_ids,
    detokenize,
    unk_id

end
