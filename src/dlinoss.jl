"""
dlinoss
========


This tutorial-style module builds a discrete diffusion language model by
introducing each mathematical map before showing the Julia implementation.
All maps are defined between explicit Euclidean spaces, and the code mirrors
the algebra.  Comments highlight Julia semantics (e.g. `NamedTuple`,
multiple dispatch) the first time they appear.
"""
module dlinoss

import Random                      # RNGs passed into Lux.setup
import Lux                         # Lux layers realised via multiple dispatch

# Re-export diffusion and OSSM components defined in the submodules.
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
    build_diffusion_ossm,
    forward_diffusion_step,
    forward_diffusion_step_with_spans

# ---------------------------------------------------------------------------
# Diffusion backbone: mathematical description
# ---------------------------------------------------------------------------

"""
    DiffusionOSSMBackbone(codec; d_sub, d_model, max_len, num_oscillators)

Mathematical map
----------------

Let a codec ``\\phi : \\{0,\\dots,|V|-1\\} → \\{0,\\dots,b-1\\}^L`` be fixed.
Define the following sequence of maps operating on digit matrices
``Z ∈ \\{0,\\dots,b\\}^{L×N}``:

1. **Subtoken embedding**
   ``E : \\{0,\\dots,b\\}^{L×N} → \\mathbb{R}^{(L d_{sub})×N}``
   given by
   ``E(Z) = [e_{z_{1,1}}; …; e_{z_{L,1}}] | … | [e_{z_{1,N}}; …; e_{z_{L,N}}]``
   where each ``e_j ∈ \\mathbb{R}^{d_{sub}}`` is a learnable column vector.

2. **Affine projection**
   ``P : \\mathbb{R}^{(L d_{sub})×N} → \\mathbb{R}^{d_{model}×N}``,
   ``P(H) = W_{proj} H + b_{proj} \\mathbf{1}_N^\\top`` with
   ``W_{proj} ∈ \\mathbb{R}^{d_{model}×(L d_{sub})}`` and
   ``b_{proj} ∈ \\mathbb{R}^{d_{model}}``.

3. **Time-position conditioning**
   ``T : \\mathbb{R}^{d_{model}×N} → \\mathbb{R}^{d_{model}×N}``,
   ``T(H) = H + P_{pos}[:, s:s+N-1] + g(t) \\mathbf{1}_N^\\top`` where
   ``P_{pos} ∈ \\mathbb{R}^{d_{model}×max_len}`` and
   ``g : \\mathbb{R} → \\mathbb{R}^{d_{model}}`` is a two-layer map
   ``g(t) = W_2 \\max(0, W_1 t + b_1) + b_2``.

4. **Gated oscillatory state-space map**
   ``\\mathcal{G} : \\mathbb{R}^{d_{model}×N} → \\mathbb{R}^{d_{model}×N}``.
   Writing the columns of ``H`` as ``(h_1, …, h_N)``, the map is defined by
   the recursion described in `ossm.unit`, yielding columns
   ``(y_1, …, y_N)``.

5. **Digit prediction head**
   ``H_{out} : \\mathbb{R}^{d_{model}×N} → \\mathbb{R}^{L×(b+1)×N}``,
   ``H_{out}(Y) = \\mathrm{reshape}( W_{out} Y + b_{out} \\mathbf{1}_N^\\top )``.

We thus obtain the composite map
``F = H_{out} ∘ \\mathcal{G} ∘ T ∘ P ∘ E``.

Implementation details
----------------------

`DiffusionOSSMBackbone` stores the Lux layers that implement the five maps.
Since it subtypes `Lux.AbstractLuxLayer`, Lux will dispatch to the specialised
methods `Lux.initialparameters`, `Lux.initialstates`, and the call overload
defined below.
"""
struct DiffusionOSSMBackbone <: Lux.AbstractLuxLayer
    embedding::PrimeSubtokenEmbedding  # implements E
    projector::Lux.Dense               # implements P
    timepos::TimePosEncoding           # implements T
    ssm::SequenceGatedOSSM             # implements 𝒢
    head::PrimeOutputHead              # implements H_out
end

function DiffusionOSSMBackbone(
    codec::PrimeCodec;
    d_sub::Int,
    d_model::Int,
    max_len::Int,
    num_oscillators::Int,
)
    embedding = PrimeSubtokenEmbedding(codec, d_sub)
    projector = Lux.Dense(codec.L * d_sub, d_model)  # affine map H ↦ W*H + b*1ᵀ
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
) = DiffusionOSSMBackbone(
    tok.codec;
    d_sub = d_sub,
    d_model = d_model,
    max_len = max_len,
    num_oscillators = num_oscillators,
)

"""
    DiffusionOSSMBundle(tokenizer, layer)

Small container exposing both the tokenizer and the backbone in the tutorial.
"""
struct DiffusionOSSMBundle
    tokenizer::DiffusionTokenizer   # map text → ids → digit matrix
    layer::DiffusionOSSMBackbone    # composite map F
end

function build_diffusion_ossm(
    tok::DiffusionTokenizer;
    d_sub::Int,
    d_model::Int,
    max_len::Int,
    num_oscillators::Int,
)
    layer = DiffusionOSSMBackbone(
        tok;
        d_sub = d_sub,
        d_model = d_model,
        max_len = max_len,
        num_oscillators = num_oscillators,
    )
    return DiffusionOSSMBundle(tok, layer)
end

build_diffusion_ossm(
    codec::PrimeCodec;
    d_sub::Int,
    d_model::Int,
    max_len::Int,
    num_oscillators::Int,
) = DiffusionOSSMBackbone(
    codec;
    d_sub = d_sub,
    d_model = d_model,
    max_len = max_len,
    num_oscillators = num_oscillators,
)

# ---------------------------------------------------------------------------
# Lux interface: parameters and states
# ---------------------------------------------------------------------------

"""
    Lux.initialparameters(rng, m::DiffusionOSSMBackbone) -> ps

Returns a `NamedTuple` whose fields correspond to the sublayers
`(embedding, projector, timepos, ssm, head)`.  Each field itself is a
`NamedTuple` whose entries are the weight matrices and bias vectors described
in the mathematical section.
"""
function Lux.initialparameters(rng::Random.AbstractRNG, m::DiffusionOSSMBackbone)
    return (
        embedding = Lux.initialparameters(rng, m.embedding),
        projector = Lux.initialparameters(rng, m.projector),
        timepos = Lux.initialparameters(rng, m.timepos),
        ssm = Lux.initialparameters(rng, m.ssm),
        head = Lux.initialparameters(rng, m.head),
    )
end

"""
    Lux.initialstates(rng, m::DiffusionOSSMBackbone) -> st

Produces a `NamedTuple` with the same field layout as the parameters.  Only
the `ssm` entry stores a non-trivial hidden state (the vector ``x_1`` in the
state-space recursion).
"""
function Lux.initialstates(rng::Random.AbstractRNG, m::DiffusionOSSMBackbone)
    return (
        embedding = Lux.initialstates(rng, m.embedding),
        projector = Lux.initialstates(rng, m.projector),
        timepos = Lux.initialstates(rng, m.timepos),
        ssm = Lux.initialstates(rng, m.ssm),
        head = Lux.initialstates(rng, m.head),
    )
end

"""
    (m::DiffusionOSSMBackbone)(Z, ps, st; t, start_pos) -> (logits, st')

Implements the composite map ``F``.  The argument `Z` is the digit matrix
``Z ∈ \\{0,\\dots,b\\}^{L×N}``.  The keywords specify the diffusion time `t`
and the positional offset `start_pos`.
"""
function (m::DiffusionOSSMBackbone)(
    Z::AbstractMatrix{<:Integer},
    ps,
    st;
    t::Float32,
    start_pos::Int = 1,
)
    H_embed, st_embed = m.embedding(Z, ps.embedding, st.embedding)
    H_proj, st_proj = m.projector(H_embed, ps.projector, st.projector)
    H_time, st_time =
        m.timepos(H_proj, ps.timepos, st.timepos; t = t, start_pos = start_pos)
    H_ssm, st_ssm = m.ssm(H_time, ps.ssm, st.ssm)
    logits, st_head = m.head(H_ssm, ps.head, st.head)

    return logits,
    (
        embedding = st_embed,
        projector = st_proj,
        timepos = st_time,
        ssm = st_ssm,
        head = st_head,
    )
end

# ---------------------------------------------------------------------------
# Diffusion sampling helpers
# ---------------------------------------------------------------------------

"""
    forward_diffusion_step(bundle, rng, text, keep_prob, ps, st; kwargs...)

Let `bundle.tokenizer` implement the composition
``\\mathrm{Tok} : \\mathrm{Text} → \\{0,\\dots,|V|-1\\}^N`` and let
``\\phi`` be its codec.  Define
``Z_0 = \\phi(\\mathrm{Tok}(text))`` and let
``\\mathrm{Mask}_s`` apply independent Bernoulli masks with keep probability
``s`` to each digit, respecting protected columns.

This function computes:
1. `ids` and `clean_digits` (``Z_0``),
2. `masked_digits = \\mathrm{Mask}_s(Z_0)``,
3. `logits, state = bundle.layer(masked_digits, ps, st; kwargs...)`.

It returns a `NamedTuple` with fields
`(:ids, :clean_digits, :masked_digits, :logits, :state)`.
"""
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
    ids, clean, masked = diffuse_text_step(
        bundle.tokenizer,
        rng,
        text,
        keep_prob;
        protect_specials = protect_specials,
    )
    logits, st_next = bundle.layer(masked, ps, st; t = t, start_pos = start_pos)
    return (
        ids = ids,
        clean_digits = clean,
        masked_digits = masked,
        logits = logits,
        state = st_next,
    )
end

"""
    forward_diffusion_step_with_spans(bundle, rng, text, keep_prob, ps, st; kwargs...)

Same computation as `forward_diffusion_step`, but the named tuple additionally
contains `spans` (the `TokenSpan` metadata) and `protected_cols` (indices kept
fixed during masking).
"""
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
    sample = diffuse_text_step_with_spans(
        bundle.tokenizer,
        rng,
        text,
        keep_prob;
        protect_specials = protect_specials,
    )
    logits, st_next =
        bundle.layer(sample.noised_digits, ps, st; t = t, start_pos = start_pos)
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
# import Revise; include("dlinoss.jl");



end
