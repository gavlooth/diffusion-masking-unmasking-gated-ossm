# Perigee diffusion stack (Gravity 4.0-inspired hybrid)
# -----------------------------------------------------------------------------

# This file keeps the entire hybrid block definition together so it can be
# audited or transplanted as a single unit.

import Statistics: mean
import LossFunctions

const BASE_LOSS = LossFunctions.L2DistLoss()

@inline function _scatter_rngs(rng::Random.AbstractRNG, count::Int)
    return [Random.MersenneTwister(Random.rand(rng, UInt)) for _ in 1:count]
end

struct SqrtWindowAttention <: Lux.AbstractLuxLayer
    attention::RestrictedAttention
    radius_factor::Float32
    min_radius::Int
    max_radius::Int
end

function SqrtWindowAttention(
    model_dim::Int,
    num_heads::Int;
    radius_factor::Real = 4.0,
    min_radius::Int = 1,
    max_radius::Union{Int,Nothing} = nothing,
)
    radius_factor <= 0 && throw(ArgumentError("radius_factor must be positive"))
    min_r = max(1, min_radius)
    max_r = isnothing(max_radius) ? typemax(Int) : max_radius
    base = RestrictedAttention(model_dim, num_heads, min_r)
    return SqrtWindowAttention(base, Float32(radius_factor), min_r, max_r)
end

initialparameters(rng::Random.AbstractRNG, layer::SqrtWindowAttention) =
    initialparameters(rng, layer.attention)

initialstates(rng::Random.AbstractRNG, layer::SqrtWindowAttention) =
    initialstates(rng, layer.attention)

function (layer::SqrtWindowAttention)(
    x::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple,
)
    attn = layer.attention
    q_proj, st_q = attn.q_proj(x, ps.q_proj, st.q_proj)
    k_proj, st_k = attn.k_proj(x, ps.k_proj, st.k_proj)
    v_proj, st_v = attn.v_proj(x, ps.v_proj, st.v_proj)
    heads_q = reshape_heads(q_proj, attn.head_dim, attn.num_heads)
    heads_k = reshape_heads(k_proj, attn.head_dim, attn.num_heads)
    heads_v = reshape_heads(v_proj, attn.head_dim, attn.num_heads)
    time_steps = size(x, 2)
    radius = clamp(
        Int(ceil(layer.radius_factor * sqrt(Float32(time_steps)))),
        layer.min_radius,
        layer.max_radius,
    )
    slices = map(1:attn.num_heads) do head_idx
        q_h = selectdim(heads_q, 2, head_idx)
        k_h = selectdim(heads_k, 2, head_idx)
        v_h = selectdim(heads_v, 2, head_idx)
        restricted_head_attention(q_h, k_h, v_h, radius)
    end
    context_stack = cat(slices...; dims = 3)
    context = permutedims(context_stack, (1, 3, 2))
    merged = reshape(context, attn.head_dim * attn.num_heads, time_steps)
    out, st_out = attn.out_proj(merged, ps.out_proj, st.out_proj)
    return out,
    (
        q_proj = st_q,
        k_proj = st_k,
        v_proj = st_v,
        out_proj = st_out,
    )
end

# -----------------------------------------------------------------------------
# Perigee mixer block: stack of OscMamba mixers with adaptive local attention
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Perigee mixer block: stack of OscMamba mixers with adaptive local attention
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Perigee mixer block: stack of OscMamba mixers with adaptive local attention
# -----------------------------------------------------------------------------

struct PerigeeMixerBlock <: Lux.AbstractLuxLayer
    mamba_layers::Vector{OscMambaMixer}
    attention::SqrtWindowAttention
    fusion_norm::Lux.LayerNorm
    diffusion_proj::Lux.Dense
    energy_proj::Lux.Dense
end

function PerigeeMixerBlock(
    mamba_repeat::Int,
    model_dim::Int,
    oscillator_count::Int,
    num_heads::Int;
    first_input_dim::Union{Nothing,Int}=nothing,
    radius_factor::Real = 4.0,
    min_radius::Int = 1,
    max_radius::Union{Int,Nothing} = nothing,
)
    mamba_repeat > 0 || throw(ArgumentError("mamba_repeat must be positive"))
    first_dim = isnothing(first_input_dim) ? model_dim : first_input_dim
    dims = [idx == 1 ? first_dim : model_dim for idx in 1:mamba_repeat]
    mamba_layers = [
        OscMambaMixer(dims[idx], model_dim, oscillator_count) for idx = 1:mamba_repeat
    ]
    attn = SqrtWindowAttention(
        model_dim,
        num_heads;
        radius_factor = radius_factor,
        min_radius = min_radius,
        max_radius = max_radius,
    )
    return PerigeeMixerBlock(
        mamba_layers,
        attn,
        Lux.LayerNorm(model_dim),
        Lux.Dense(model_dim, model_dim),
        Lux.Dense(model_dim, 1),
    )
end

function initialparameters(rng::Random.AbstractRNG, block::PerigeeMixerBlock)
    rng_total = length(block.mamba_layers) + 4
    rngs = _scatter_rngs(rng, rng_total)
    mamba_ps = map(eachindex(block.mamba_layers)) do idx
        initialparameters(rngs[idx], block.mamba_layers[idx])
    end |> Tuple
    attn_ps = initialparameters(rngs[end-3], block.attention)
    fusion_norm = Lux.initialparameters(rngs[end-2], block.fusion_norm)
    diffusion_proj = Lux.initialparameters(rngs[end-1], block.diffusion_proj)
    energy_proj = Lux.initialparameters(rngs[end], block.energy_proj)
    return (
        mamba_layers = mamba_ps,
        attention = attn_ps,
        fusion_norm = fusion_norm,
        diffusion_proj = diffusion_proj,
        energy_proj = energy_proj,
    )
end

function initialstates(rng::Random.AbstractRNG, block::PerigeeMixerBlock)
    rng_total = length(block.mamba_layers) + 4
    rngs = _scatter_rngs(rng, rng_total)
    mamba_st = map(eachindex(block.mamba_layers)) do idx
        initialstates(rngs[idx], block.mamba_layers[idx])
    end |> Tuple
    attn_st = initialstates(rngs[end-3], block.attention)
    fusion_norm = Lux.initialstates(rngs[end-2], block.fusion_norm)
    diffusion_proj = Lux.initialstates(rngs[end-1], block.diffusion_proj)
    energy_proj = Lux.initialstates(rngs[end], block.energy_proj)
    return (
        mamba_layers = mamba_st,
        attention = attn_st,
        fusion_norm = fusion_norm,
        diffusion_proj = diffusion_proj,
        energy_proj = energy_proj,
    )
end

function (block::PerigeeMixerBlock)(seq::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    x = seq
    mamba_state_acc = ()
    for idx in eachindex(block.mamba_layers)
        x, st_layer =
            block.mamba_layers[idx](x, ps.mamba_layers[idx], st.mamba_layers[idx])
        mamba_state_acc = tuple(mamba_state_acc..., st_layer)
    end
    mamba_states = mamba_state_acc
    attn_in = (seq + x) / 2
    attn_out, st_attn = block.attention(attn_in, ps.attention, st.attention)
    fused = x + attn_out
    normed, st_norm = block.fusion_norm(fused, ps.fusion_norm, st.fusion_norm)
    diffusion_logits, st_diff =
        block.diffusion_proj(normed, ps.diffusion_proj, st.diffusion_proj)
    energy_logits, st_energy = block.energy_proj(normed, ps.energy_proj, st.energy_proj)
    return (sequence = normed, diffusion = diffusion_logits, energy = energy_logits),
    (
        mamba_layers = mamba_states,
        attention = st_attn,
        fusion_norm = st_norm,
        diffusion_proj = st_diff,
        energy_proj = st_energy,
    )
end

# -----------------------------------------------------------------------------
# Perigee diffusion language model with partial-masking helpers
# -----------------------------------------------------------------------------

struct PerigeeDiffusionLM <: Lux.AbstractLuxLayer
    blocks::Vector{PerigeeMixerBlock}
    final_norm::Lux.LayerNorm
    vocab_proj::Lux.Dense
end

function PerigeeDiffusionLM(
    blocks::Vector{PerigeeMixerBlock},
    model_dim::Int,
    vocab_dim::Int,
)
    isempty(blocks) && throw(ArgumentError("must provide at least one block"))
    return PerigeeDiffusionLM(
        blocks,
        Lux.LayerNorm(model_dim),
        Lux.Dense(model_dim, vocab_dim),
    )
end

function initialparameters(rng::Random.AbstractRNG, model::PerigeeDiffusionLM)
    block_ps = map(model.blocks) do block
        block_rng = Random.MersenneTwister(Random.rand(rng, UInt))
        initialparameters(block_rng, block)
    end |> Tuple
    final_norm = Lux.initialparameters(rng, model.final_norm)
    vocab_proj = Lux.initialparameters(
        Random.MersenneTwister(Random.rand(rng, UInt)),
        model.vocab_proj,
    )
    return (blocks = block_ps, final_norm = final_norm, vocab_proj = vocab_proj)
end

function initialstates(rng::Random.AbstractRNG, model::PerigeeDiffusionLM)
    block_st = map(model.blocks) do block
        block_rng = Random.MersenneTwister(Random.rand(rng, UInt))
        initialstates(block_rng, block)
    end |> Tuple
    final_norm = Lux.initialstates(rng, model.final_norm)
    vocab_proj =
        Lux.initialstates(Random.MersenneTwister(Random.rand(rng, UInt)), model.vocab_proj)
    return (blocks = block_st, final_norm = final_norm, vocab_proj = vocab_proj)
end

function (model::PerigeeDiffusionLM)(seq::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    block_indices = eachindex(model.blocks)
    fold_init = (
        sequence = seq,
        diffusion = nothing,
        energy = nothing,
        states = (),
    )
    folded = foldl(
        (acc, idx) -> begin
            block = model.blocks[idx]
            block_out, block_state = block(acc.sequence, ps.blocks[idx], st.blocks[idx])
            next_diffusion = acc.diffusion === nothing ?
                             block_out.diffusion :
                             acc.diffusion + block_out.diffusion
            next_energy = acc.energy === nothing ?
                          block_out.energy :
                          acc.energy + block_out.energy
            return (
                sequence = block_out.sequence,
                diffusion = next_diffusion,
                energy = next_energy,
                states = tuple(acc.states..., block_state),
            )
        end,
        block_indices;
        init = fold_init,
    )
    normed, st_norm = model.final_norm(folded.sequence, ps.final_norm, st.final_norm)
    logits, st_vocab = model.vocab_proj(normed, ps.vocab_proj, st.vocab_proj)
    diffusion_total =
        folded.diffusion === nothing ? zeros(Float32, size(normed)) : folded.diffusion
    energy_total = folded.energy === nothing ?
                   zeros(Float32, 1, size(normed, 2)) :
                   folded.energy
    block_states = folded.states
    return (logits = logits, diffusion = diffusion_total, energy = energy_total),
    (blocks = block_states, final_norm = st_norm, vocab_proj = st_vocab)
end

function build_perigee_model(
    num_layers::Int;
    input_dim::Int,
    model_dim::Int,
    oscillator_count::Int,
    num_heads::Int,
    vocab_dim::Int,
    mamba_repeat::Int = 2,
    radius_factor::Real = 4.0,
    radius_growth::Real = 0.0,
    min_radius::Int = 1,
    max_radius::Union{Int,Nothing} = nothing,
)
    num_layers > 0 || throw(ArgumentError("num_layers must be positive"))
    blocks = [
        PerigeeMixerBlock(
            mamba_repeat,
            model_dim,
            oscillator_count,
            num_heads;
            radius_factor = radius_factor * (1 + radius_growth * (idx - 1)),
            first_input_dim = idx == 1 ? input_dim : nothing,
            min_radius = min_radius,
            max_radius = max_radius,
        ) for idx = 1:num_layers
    ]
    return PerigeeDiffusionLM(blocks, model_dim, vocab_dim)
end

# -----------------------------------------------------------------------------
# Partial masking utilities for diffusion training
# -----------------------------------------------------------------------------

function perigee_prepare_batch(
    tokenizer::PrimeTokenizer,
    sequences::Vector;
    mask_fraction::Real = 0.15,
    unmask_fraction::Real = 0.2,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    masked = map(sequences) do sample
        tokens = sample isa PrimeSample ? sample.tokens : sample
        corrupt_tokens(
            tokenizer,
            sample;
            mask_fraction = mask_fraction,
            unmask_fraction = unmask_fraction,
            rng = rng,
        )
    end
    obs = hcat(map(x -> Float32.(x.encoded), masked)...)
    targets = hcat(map(x -> Float32.(x.targets), masked)...)
    mask_matrix = zeros(Float32, size(obs, 1), length(masked))
    for (col, item) in enumerate(masked)
        for idx in item.mask_indices
            mask_matrix[idx, col] = 1f0
        end
    end
    return (observed = obs, targets = targets, mask_matrix = mask_matrix)
end

function perigee_diffusion_loss(model::PerigeeDiffusionLM, batch::NamedTuple, ps, st)
    output, st_new = model(batch.observed, ps, st)
    pred = output.logits
    base_loss = LossFunctions.mean(BASE_LOSS, pred, batch.targets)
    mask_weight = sum(batch.mask_matrix)
    mask_loss = mask_weight == 0 ?
                0f0 :
                LossFunctions.mean(
                    BASE_LOSS,
                    pred .* batch.mask_matrix,
                    batch.targets .* batch.mask_matrix,
                )
    loss = base_loss + 2f0 * mask_loss
    return loss, st_new
end

# julia --project -e 'using ossmv2; CUDA.functional()' 
# vocab = JSON3.read(read("assets/tokenizers/gpt2- vocab.json", String)) |> keys |> collect.
