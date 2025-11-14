# Energy-guided interleaving of OscMamba and restricted attention
# ───────────────────────────────────────────────────────────────────────────────

# Mathematical definition:
# Given an input sequence X, first compute M = Mamba(X) and A = Attention((X + M)/2).
# The fused hidden state is H = LayerNorm(M + A). Two affine heads yield
# diffusion logits D = P_diff(H) and scalar energy estimates E = P_energy(H).
# These tensors guide a diffusion model via gradients ∇_X E while retaining token logits.
struct EnergyInterleaveBlock <: Lux.AbstractLuxLayer
    mamba::OscMambaMixer
    attention::RestrictedAttention
    fusion_norm::Lux.LayerNorm
    energy_proj::Lux.Dense
    diffusion_proj::Lux.Dense
end

function EnergyInterleaveBlock(
    mamba::OscMambaMixer,
    attention::RestrictedAttention,
    model_dim::Int,
)
    return EnergyInterleaveBlock(
        mamba,
        attention,
        Lux.LayerNorm(model_dim),
        Lux.Dense(model_dim, 1),
        Lux.Dense(model_dim, model_dim),
    )
end

function initialparameters(rng::Random.AbstractRNG, block::EnergyInterleaveBlock)
    rngs = Random.split(rng, 5)
    return (
        mamba = initialparameters(rngs[1], block.mamba),
        attention = initialparameters(rngs[2], block.attention),
        fusion_norm = Lux.initialparameters(rngs[3], block.fusion_norm),
        energy_proj = Lux.initialparameters(rngs[4], block.energy_proj),
        diffusion_proj = Lux.initialparameters(rngs[5], block.diffusion_proj),
    )
end

function initialstates(rng::Random.AbstractRNG, block::EnergyInterleaveBlock)
    rngs = Random.split(rng, 5)
    return (
        mamba = initialstates(rngs[1], block.mamba),
        attention = initialstates(rngs[2], block.attention),
        fusion_norm = Lux.initialstates(rngs[3], block.fusion_norm),
        energy_proj = Lux.initialstates(rngs[4], block.energy_proj),
        diffusion_proj = Lux.initialstates(rngs[5], block.diffusion_proj),
    )
end

function (block::EnergyInterleaveBlock)(
    seq::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple,
)
    mamba_out, st_mamba = block.mamba(seq, ps.mamba, st.mamba)
    attn_in = (seq + mamba_out) / 2
    attn_out, st_attn = block.attention(attn_in, ps.attention, st.attention)
    fused = mamba_out + attn_out
    normed, st_norm = block.fusion_norm(fused, ps.fusion_norm, st.fusion_norm)
    energy_logits, st_energy = block.energy_proj(normed, ps.energy_proj, st.energy_proj)
    diffusion_logits, st_diff = block.diffusion_proj(
        normed,
        ps.diffusion_proj,
        st.diffusion_proj,
    )
    return (
        sequence = normed,
        diffusion = diffusion_logits,
        energy = energy_logits,
    ),
    (
        mamba = st_mamba,
        attention = st_attn,
        fusion_norm = st_norm,
        energy_proj = st_energy,
        diffusion_proj = st_diff,
    )
end

# Mathematical definition for the full stack:
# Let {B_ℓ} denote interleave blocks. For ℓ = 1,…,L set (H_ℓ, D_ℓ, E_ℓ) = B_ℓ(H_{ℓ-1})
# with H₀ = token embeddings. Aggregate diffusion logits D = Σ D_ℓ and energies
# E = Σ E_ℓ. Final projections produce vocabulary logits.
struct EnergyGuidedDiffusionLLM{BlocksT<:NTuple, LN<:Lux.LayerNorm, DenseT<:Lux.Dense} <:
       Lux.AbstractLuxLayer
    blocks::BlocksT
    final_norm::LN
    vocab_proj::DenseT
end

function EnergyGuidedDiffusionLLM(
    blocks::NTuple{N, EnergyInterleaveBlock},
    model_dim::Int,
    vocab_dim::Int,
) where {N}
    return EnergyGuidedDiffusionLLM(
        blocks,
        Lux.LayerNorm(model_dim),
        Lux.Dense(model_dim, vocab_dim),
    )
end

function initialparameters(rng::Random.AbstractRNG, model::EnergyGuidedDiffusionLLM)
    block_count = length(model.blocks)
    rngs = Random.split(rng, block_count + 2)
    block_ps = ntuple(
        i -> initialparameters(rngs[i], model.blocks[i]),
        Val(block_count),
    )
    final_norm = Lux.initialparameters(rngs[end - 1], model.final_norm)
    vocab_proj = Lux.initialparameters(rngs[end], model.vocab_proj)
    return (
        blocks = block_ps,
        final_norm = final_norm,
        vocab_proj = vocab_proj,
    )
end

function initialstates(rng::Random.AbstractRNG, model::EnergyGuidedDiffusionLLM)
    block_count = length(model.blocks)
    rngs = Random.split(rng, block_count + 2)
    block_st = ntuple(
        i -> initialstates(rngs[i], model.blocks[i]),
        Val(block_count),
    )
    final_norm = Lux.initialstates(rngs[end - 1], model.final_norm)
    vocab_proj = Lux.initialstates(rngs[end], model.vocab_proj)
    return (
        blocks = block_st,
        final_norm = final_norm,
        vocab_proj = vocab_proj,
    )
end

function (model::EnergyGuidedDiffusionLLM)(
    seq::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple,
)
    block_indices = ntuple(identity, Val(length(model.blocks)))
    fold_init = (
        sequence = seq,
        diffusion = nothing,
        energy = nothing,
        states = Tuple{},
    )
    folded = foldl(
        (acc, idx) -> begin
            block = model.blocks[idx]
            block_out, block_state = block(
                acc.sequence,
                ps.blocks[idx],
                st.blocks[idx],
            )
            next_sequence = block_out.sequence
            next_diff = acc.diffusion === nothing ?
                        block_out.diffusion :
                        acc.diffusion + block_out.diffusion
            next_energy = acc.energy === nothing ?
                          block_out.energy :
                          acc.energy + block_out.energy
            return (
                sequence = next_sequence,
                diffusion = next_diff,
                energy = next_energy,
                states = tuple(acc.states..., block_state),
            )
        end,
        block_indices;
        init = fold_init,
    )
    normed, st_norm = model.final_norm(folded.sequence, ps.final_norm, st.final_norm)
    logits, st_vocab = model.vocab_proj(normed, ps.vocab_proj, st.vocab_proj)
    diffusion_total = folded.diffusion === nothing ?
                      zeros(Float32, size(normed)) :
                      folded.diffusion
    energy_total = folded.energy === nothing ?
                   zeros(Float32, 1, size(normed, 2)) :
                   folded.energy
    return (
        logits = logits,
        diffusion = diffusion_total,
        energy = energy_total,
    ),
    (
        blocks = folded.states,
        final_norm = st_norm,
        vocab_proj = st_vocab,
    )
end

function build_energy_guided_llm(
    num_layers::Int;
    model_dim::Int,
    oscillator_count::Int,
    num_heads::Int,
    window_radius::Int,
    vocab_dim::Int,
)
    num_layers < 1 && throw(ArgumentError("num_layers must be positive"))
    blocks = ntuple(
        _ -> begin
            mamba = OscMambaMixer(model_dim, model_dim, oscillator_count)
            attention = RestrictedAttention(model_dim, num_heads, window_radius)
            EnergyInterleaveBlock(mamba, attention, model_dim)
        end,
        num_layers,
    )
    return EnergyGuidedDiffusionLLM(blocks, model_dim, vocab_dim)
end

# ───────────────────────────────────────────────────────────────────────────────
