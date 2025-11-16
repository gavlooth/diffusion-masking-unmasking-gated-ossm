# Perigee diffusion stack (Gravity 4.0-inspired hybrid)
# -----------------------------------------------------------------------------

# This file keeps the entire hybrid block definition together so it can be
# audited or transplanted as a single unit.

import Statistics: mean
import LossFunctions
using Transducers

const BASE_LOSS = LossFunctions.L2DistLoss()

@inline function mse_mean(pred::AbstractArray, target::AbstractArray)
    diff = pred .- target
    return sum(abs2, diff) / length(diff)
end


@inline function _scatter_rngs(rng::Random.AbstractRNG, count::Int)
    return [Random.MersenneTwister(Random.rand(rng, UInt)) for _ = 1:count]
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
    transformer::LogWindowTransformer
    fusion_norm::Lux.LayerNorm
    diffusion_proj::Lux.Dense
end

function PerigeeMixerBlock(
    mamba_repeat::Int,
    model_dim::Int,
    oscillator_count::Int,
    num_heads::Int;
    radius_factor::Real = 4.0,
    min_radius::Int = 1,
    max_radius::Union{Int,Nothing} = nothing,
)
    mamba_repeat > 0 || throw(ArgumentError("mamba_repeat must be positive"))
    mamba_layers =
        [OscMambaMixer(model_dim, model_dim, oscillator_count) for _ = 1:mamba_repeat]
    transformer = LogWindowTransformer(
        model_dim,
        num_heads;
        radius_factor = radius_factor,
        min_radius = min_radius,
        max_radius = max_radius,
        base_radius = 32,
    )
    return PerigeeMixerBlock(
        mamba_layers,
        transformer,
        Lux.LayerNorm(model_dim),
        Lux.Dense(model_dim, model_dim),
    )
end

function initialparameters(rng::Random.AbstractRNG, block::PerigeeMixerBlock)
    rng_total = length(block.mamba_layers) + 4
    rngs = _scatter_rngs(rng, rng_total)
    mamba_ps = map(eachindex(block.mamba_layers)) do idx
        initialparameters(rngs[idx], block.mamba_layers[idx])
    end |> Tuple
    trans_ps = initialparameters(rngs[end-3], block.transformer)
    fusion_norm = Lux.initialparameters(rngs[end-2], block.fusion_norm)
    diffusion_proj = Lux.initialparameters(rngs[end-1], block.diffusion_proj)
    return (
        mamba_layers = mamba_ps,
        transformer = trans_ps,
        fusion_norm = fusion_norm,
        diffusion_proj = diffusion_proj,
    )
end

function initialstates(rng::Random.AbstractRNG, block::PerigeeMixerBlock)
    rng_total = length(block.mamba_layers) + 4
    rngs = _scatter_rngs(rng, rng_total)
    mamba_st = map(eachindex(block.mamba_layers)) do idx
        initialstates(rngs[idx], block.mamba_layers[idx])
    end |> Tuple
    trans_st = initialstates(rngs[end-3], block.transformer)
    fusion_norm = Lux.initialstates(rngs[end-2], block.fusion_norm)
    diffusion_proj = Lux.initialstates(rngs[end-1], block.diffusion_proj)
    return (
        mamba_layers = mamba_st,
        transformer = trans_st,
        fusion_norm = fusion_norm,
        diffusion_proj = diffusion_proj,
    )
end

function (block::PerigeeMixerBlock)(seq::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    fold_init = (sequence = seq, states = ())
    mamba_fold = Transducers.foldl(
        (acc, idx) -> begin
            x_curr, state_tuple = acc
            x_next, st_layer = block.mamba_layers[idx](
                x_curr,
                ps.mamba_layers[idx],
                st.mamba_layers[idx],
            )
            return (sequence = x_next, states = tuple(state_tuple..., st_layer))
        end,
        Transducers.IdentityTransducer(),
        1:length(block.mamba_layers);
        init = fold_init,
    )
    x = mamba_fold.sequence
    mamba_states = mamba_fold.states
    attn_in = (seq + x) / 2
    trans_out, st_trans = block.transformer(attn_in, ps.transformer, st.transformer)
    fused = x + trans_out
    normed, st_norm = block.fusion_norm(fused, ps.fusion_norm, st.fusion_norm)
    diffusion_logits, st_diff =
        block.diffusion_proj(normed, ps.diffusion_proj, st.diffusion_proj)
    return (sequence = normed, diffusion = diffusion_logits),
    (
        mamba_layers = mamba_states,
        transformer = st_trans,
        fusion_norm = st_norm,
        diffusion_proj = st_diff,
    )
end

# -----------------------------------------------------------------------------
# Perigee diffusion language model with partial-masking helpers
# -----------------------------------------------------------------------------

struct PerigeeDiffusionLM <: Lux.AbstractLuxLayer
    input_embed::Lux.Dense
    blocks::Vector{PerigeeMixerBlock}
    final_norm::Lux.LayerNorm
    vocab_proj::Lux.Dense
end

function PerigeeDiffusionLM(
    blocks::Vector{PerigeeMixerBlock},
    model_dim::Int,
    vocab_dim::Int,
    input_dim::Int,
)
    isempty(blocks) && throw(ArgumentError("must provide at least one block"))
    return PerigeeDiffusionLM(
        Lux.Dense(input_dim, model_dim),
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
    input_embed = Lux.initialparameters(rng, model.input_embed)
    return (
        input_embed = input_embed,
        blocks = block_ps,
        final_norm = final_norm,
        vocab_proj = vocab_proj,
    )
end

function initialstates(rng::Random.AbstractRNG, model::PerigeeDiffusionLM)
    block_st = map(model.blocks) do block
        block_rng = Random.MersenneTwister(Random.rand(rng, UInt))
        initialstates(block_rng, block)
    end |> Tuple
    final_norm = Lux.initialstates(rng, model.final_norm)
    vocab_proj =
        Lux.initialstates(Random.MersenneTwister(Random.rand(rng, UInt)), model.vocab_proj)
    input_embed = Lux.initialstates(rng, model.input_embed)
    return (
        input_embed = input_embed,
        blocks = block_st,
        final_norm = final_norm,
        vocab_proj = vocab_proj,
    )
end

function (model::PerigeeDiffusionLM)(seq::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    embedded, st_embed = model.input_embed(seq, ps.input_embed, st.input_embed)
    fold_init = (sequence = embedded, diffusion = nothing, states = ())
    indices = 1:length(model.blocks)
    folded = Transducers.foldl(
        (acc, idx) -> begin
            block = model.blocks[idx]
            block_out, block_state =
                block(acc.sequence, ps.blocks[idx], st.blocks[idx])
            next_diffusion =
                acc.diffusion === nothing ? block_out.diffusion :
                acc.diffusion + block_out.diffusion
            return (
                sequence = block_out.sequence,
                diffusion = next_diffusion,
                states = tuple(acc.states..., block_state),
            )
        end,
        Transducers.IdentityTransducer(),
        indices;
        init = fold_init,
    )
    normed, st_norm = model.final_norm(folded.sequence, ps.final_norm, st.final_norm)
    logits, st_vocab = model.vocab_proj(normed, ps.vocab_proj, st.vocab_proj)
    diffusion_total =
        folded.diffusion === nothing ? zeros(Float32, size(normed)) : folded.diffusion
    block_states = folded.states
    return (logits = logits, diffusion = diffusion_total),
    (
        input_embed = st_embed,
        blocks = block_states,
        final_norm = st_norm,
        vocab_proj = st_vocab,
    )
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
            min_radius = min_radius,
            max_radius = max_radius,
        ) for idx = 1:num_layers
    ]
    return PerigeeDiffusionLM(blocks, model_dim, vocab_dim, input_dim)
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
    obs = hcat(map(x -> normalize_prime_codes(tokenizer, x.encoded), masked)...)
    targets = hcat(map(x -> normalize_prime_codes(tokenizer, x.targets), masked)...)
    mask_matrix = zeros(Float32, size(obs, 1), length(masked))
    for (col, item) in enumerate(masked)
        for idx in item.mask_indices
            mask_matrix[idx, col] = 1.0f0
        end
    end
    return (observed = obs, targets = targets, mask_matrix = mask_matrix)
end

function perigee_diffusion_loss(model::PerigeeDiffusionLM, batch::NamedTuple, ps, st)
    output, st_new = model(batch.observed, ps, st)
    pred = output.logits
    base_loss = mse_mean(pred, batch.targets)
    mask_weight = sum(batch.mask_matrix)
    mask_loss =
        mask_weight == 0 ? 0.0f0 :
        mse_mean(pred .* batch.mask_matrix, batch.targets .* batch.mask_matrix)
    loss = base_loss + 2.0f0 * mask_loss
    return loss, st_new
end

# julia --project -e 'using ossmv2; CUDA.functional()' 
# vocab = JSON3.read(read("assets/tokenizers/gpt2- vocab.json", String)) |> keys |> collect.
