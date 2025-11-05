module dlinoss

import Random
import Lux

# Re-export the submodules that implement the diffusion tokenisation and OSSM
# building blocks.
include("diffusion.jl")
include("ossm.jl")

using .DiffusionBlocks
using .DiffusionBlocks.Embeddings: PrimeSubtokenEmbedding
using .DiffusionBlocks.OutputHead: PrimeOutputHead
using .DiffusionBlocks.TimePos: TimePosEncoding
using .DiffusionBlocks.DiffusionTokenizerModule: DiffusionTokenizer
using .DiffusionBlocks.Codec: PrimeCodec
using .DiffusionBlocks.Pipeline: diffuse_text_step, diffuse_text_step_with_spans
using .ossm
using .ossm: SequenceGatedOSSM, GatedOscSSMUnit

export DiffusionBlocks,
    ossm,
    DiffusionOSSMBackbone,
    DiffusionOSSMBundle,
    build_diffusion_ossm

"""
    DiffusionOSSMBackbone(codec; d_sub, d_model, max_len, num_oscillators)

Composite Lux layer linking the diffusion digit pipeline with the OSSM backbone:

```
digits --PrimeSubtokenEmbedding--> embeddings
      --Dense→TimePosEncoding--> conditioned embeddings
      --SequenceGatedOSSM--> hidden sequence
      --PrimeOutputHead--> digit logits
```

Use `build_diffusion_ossm(tok::DiffusionTokenizer, ...)` as a convenience
initializer when you already have a tokenizer.
"""
struct DiffusionOSSMBackbone <: Lux.AbstractLuxLayer
    embedding::PrimeSubtokenEmbedding
    projector::Lux.Dense
    timepos::TimePosEncoding
    ssm::SequenceGatedOSSM
    head::PrimeOutputHead
end

function DiffusionOSSMBackbone(
    codec::PrimeCodec;
    d_sub::Int,
    d_model::Int,
    max_len::Int,
    num_oscillators::Int,
)
    embedding = PrimeSubtokenEmbedding(codec, d_sub)
    projector = Lux.Dense(codec.L * d_sub, d_model)
    timepos = TimePosEncoding(d_model, max_len)
    ssm_unit = GatedOscSSMUnit(num_oscillators, d_model, d_model)
    ssm = SequenceGatedOSSM(ssm_unit)
    head = PrimeOutputHead(codec, d_model)
    return DiffusionOSSMBackbone(embedding, projector, timepos, ssm, head)
end

DiffusionOSSMBackbone(
    tok::DiffusionTokenizer;
    d_sub::Int,
    d_model::Int,
    max_len::Int,
    num_oscillators::Int,
) = DiffusionOSSMBackbone(tok.codec;
    d_sub = d_sub,
    d_model = d_model,
    max_len = max_len,
    num_oscillators = num_oscillators)

"""
    build_diffusion_ossm(tok_or_codec; kwargs...)

Helper constructor returning the backbone while also exposing the tokenizer,
making it easy to set up:

```
tok = DiffusionTokenizer(...)
model = build_diffusion_ossm(tok; d_sub = 16, d_model = 256, max_len = 1024,
    num_oscillators = 64)
ps, st = Lux.setup(rng, model.layer)
```
"""
struct DiffusionOSSMBundle
    tokenizer::DiffusionTokenizer
    layer::DiffusionOSSMBackbone
end

function build_diffusion_ossm(
    tok::DiffusionTokenizer;
    d_sub::Int,
    d_model::Int,
    max_len::Int,
    num_oscillators::Int,
)
    layer = DiffusionOSSMBackbone(tok;
        d_sub = d_sub,
        d_model = d_model,
        max_len = max_len,
        num_oscillators = num_oscillators)
    return DiffusionOSSMBundle(tok, layer)
end

build_diffusion_ossm(
    codec::PrimeCodec;
    d_sub::Int,
    d_model::Int,
    max_len::Int,
    num_oscillators::Int,
) = DiffusionOSSMBackbone(codec;
    d_sub = d_sub,
    d_model = d_model,
    max_len = max_len,
    num_oscillators = num_oscillators)

function Lux.initialparameters(rng::Random.AbstractRNG, m::DiffusionOSSMBackbone)
    return (
        embedding = Lux.initialparameters(rng, m.embedding),
        projector = Lux.initialparameters(rng, m.projector),
        timepos = Lux.initialparameters(rng, m.timepos),
        ssm = Lux.initialparameters(rng, m.ssm),
        head = Lux.initialparameters(rng, m.head),
    )
end

function forward_diffusion_step(
    bundle::DiffusionOSSMBundle,
    rng::Random.AbstractRNG,
    text::AbstractString,
    keep_prob::Float32,
    ps,
    st;
    protect_specials::Bool = true,
    t::Float32 = 0.0f0,
    start_pos::Int = 1,
)
    ids, clean, masked = diffuse_text_step(bundle.tokenizer, rng, text, keep_prob;
        protect_specials = protect_specials)
    logits, st_next = bundle.layer(masked, ps, st; t = t, start_pos = start_pos)
    return (
        ids = ids,
        clean_digits = clean,
        masked_digits = masked,
        logits = logits,
        state = st_next,
    )
end

function forward_diffusion_step_with_spans(
    bundle::DiffusionOSSMBundle,
    rng::Random.AbstractRNG,
    text::AbstractString,
    keep_prob::Float32,
    ps,
    st;
    protect_specials::Bool = true,
    t::Float32 = 0.0f0,
    start_pos::Int = 1,
)
    sample = diffuse_text_step_with_spans(bundle.tokenizer, rng, text, keep_prob;
        protect_specials = protect_specials)
    logits, st_next = bundle.layer(sample.noised_digits, ps, st; t = t, start_pos = start_pos)
    return (
        ids = sample.ids,
        clean_digits = sample.clean_digits,
        masked_digits = sample.noised_digits,
        spans = sample.spans,
        protected_cols = sample.protected_cols,
        logits = logits,
        state = st_next,
    )
end

export forward_diffusion_step, forward_diffusion_step_with_spans

function Lux.initialstates(rng::Random.AbstractRNG, m::DiffusionOSSMBackbone)
    return (
        embedding = Lux.initialstates(rng, m.embedding),
        projector = Lux.initialstates(rng, m.projector),
        timepos = Lux.initialstates(rng, m.timepos),
        ssm = Lux.initialstates(rng, m.ssm),
        head = Lux.initialstates(rng, m.head),
    )
end

function (m::DiffusionOSSMBackbone)(
    Z::AbstractMatrix{<:Integer},
    ps,
    st;
    t::Float32,
    start_pos::Int = 1,
)
    H_embed, st_embed = m.embedding(Z, ps.embedding, st.embedding)
    H_proj, st_proj = m.projector(H_embed, ps.projector, st.projector)
    H_time, st_time = m.timepos(H_proj, ps.timepos, st.timepos; t = t, start_pos = start_pos)
    H_ssm, st_ssm = m.ssm(H_time, ps.ssm, st.ssm)
    logits, st_head = m.head(H_ssm, ps.head, st.head)

    return logits, (
        embedding = st_embed,
        projector = st_proj,
        timepos = st_time,
        ssm = st_ssm,
        head = st_head,
    )
end

end
