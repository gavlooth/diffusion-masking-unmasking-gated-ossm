module DiffusionBlocks

# The diffusion namespace now mirrors the dataflow of the diffusion LLM:
# - Tokenizer: text ↔ token ids with span metadata for partial masking.
# - Codec: prime base digitisation utilities.
# - DiffusionTokenizerModule: bundle Vocab+Codec for training and inference.
# - Masking: forward masking and progressive unmasking schedules.
# - Embeddings: subtoken embedding layer for digit streams.
# - TimePos: time/position conditioning layers.
# - OutputHead: digit-level prediction heads.
# - Pipeline: end-to-end helpers wiring tokenisation with masking steps.

include("diffusion/tokenizer.jl")
include("diffusion/codec.jl")
include("diffusion/diffusion_tokenizer.jl")
include("diffusion/masking.jl")
include("diffusion/embeddings.jl")
include("diffusion/timepos.jl")
include("diffusion/output_head.jl")
include("diffusion/pipeline.jl")

using .Tokenizer
using .Codec
using .DiffusionTokenizerModule
using .Masking
using .Embeddings
using .TimePos
using .OutputHead
using .Pipeline

export Tokenizer,
    Codec,
    DiffusionTokenizerModule,
    Masking,
    Embeddings,
    TimePos,
    OutputHead,
    Pipeline

# Re-export individual helpers for convenience.
export Vocab,
    DEFAULT_SPECIALS,
    TokenizeConfig,
    TokenSpan,
    tokenize,
    tokenize_with_spans,
    build_vocab,
    encode_token,
    encode_tokens,
    decode_ids,
    detokenize,
    PrimeCodec,
    encode_id,
    decode_id,
    mask_id,
    ids_to_digit_matrix,
    digit_matrix_to_ids,
    DiffusionTokenizer,
    encode_text_to_digits,
    encode_text_to_digits_with_spans,
    decode_digits_to_text,
    forward_mask,
    partial_unmask,
    protected_columns,
    PrimeSubtokenEmbedding,
    TimePosEncoding,
    PrimeOutputHead,
    diffuse_text_step,
    diffuse_text_step_with_spans

end
