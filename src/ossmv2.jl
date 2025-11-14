module ossmv2

import LuxCore, Lux, Random, NNlib
import LuxCore: initialparameters, initialstates
import LinearAlgebra               # for LinearAlgebra.Diagonal

greet() = print("Hello World!")

# ───────────────────────────────────────────────────────────────────────────────
# Layers (shape-only structs)
# ───────────────────────────────────────────────────────────────────────────────

struct OscillatorBank <: Lux.AbstractLuxLayer
    oscillator_count::Int
end

struct GatedOscSSMUnit <: Lux.AbstractLuxLayer
    oscillator_count::Int
    input_dimension::Int
    output_dimension::Int
end

struct OscSSMBlock <: Lux.AbstractLuxLayer
    ssm_unit::GatedOscSSMUnit
    shared_bank::OscillatorBank
end

# Helpers for state dimension (2 reals per oscillator)
@inline state_dimension(bank::OscillatorBank) = 2 * bank.oscillator_count
@inline state_dimension(ssm::GatedOscSSMUnit) = 2 * ssm.oscillator_count

# Small logistic; qualify exp inside a module
σ(x) = 1.0f0 / (1.0f0 + Base.exp(-x))

# ───────────────────────────────────────────────────────────────────────────────
# Parameters / States: OscillatorBank (global A)
# ───────────────────────────────────────────────────────────────────────────────

function initialparameters(rng::Random.AbstractRNG, bank::OscillatorBank)
    osc_count = bank.oscillator_count
    return (
        # later mapped through σ to get α ∈ (0,1)
        logit_contraction = map(
            x -> 0.01f0 * x - 0.05f0,
            Random.randn(rng, Float32, osc_count),
        ),
        # rotation angles (radians)
        rotation_angle = map(x -> 0.20f0 * x, Random.randn(rng, Float32, osc_count)),
    )
end

# Bank has no runtime state
initialstates(::Random.AbstractRNG, ::OscillatorBank) = NamedTuple()

# ───────────────────────────────────────────────────────────────────────────────
# Parameters / States: GatedOscSSMUnit (adapters B/C/D and gates)
# ───────────────────────────────────────────────────────────────────────────────

function initialparameters(rng::Random.AbstractRNG, ssm::GatedOscSSMUnit)
    state_dim = state_dimension(ssm)       # 2 * oscillator_count
    input_dim = ssm.input_dimension
    output_dim = ssm.output_dimension

    B0 = map(x -> 0.01f0 * x, Random.randn(rng, Float32, state_dim, input_dim))
    C0 = map(x -> 0.01f0 * x, Random.randn(rng, Float32, output_dim, state_dim))
    D = Base.zeros(Float32, output_dim, input_dim)
    bias = Base.zeros(Float32, output_dim)

    # Gates: G_B : ℝ^{input_dim} → ℝ^{state_dim}
    B_W = map(x -> 0.01f0 * x, Random.randn(rng, Float32, state_dim, input_dim))
    B_b = Base.zeros(Float32, state_dim)

    # Gates: G_C : ℝ^{input_dim} → ℝ^{output_dim}
    C_W = map(x -> 0.01f0 * x, Random.randn(rng, Float32, output_dim, input_dim))
    C_b = Base.zeros(Float32, output_dim)

    return (
        B0 = B0,
        C0 = C0,
        D = D,
        bias = bias,
        gate_B = (weight = B_W, bias = B_b),
        gate_C = (weight = C_W, bias = C_b),
    )
end

function initialstates(::Random.AbstractRNG, ssm::GatedOscSSMUnit)
    # Runtime recurrent state only
    return (hidden_state = Base.zeros(Float32, 2 * ssm.oscillator_count),)
end

# ───────────────────────────────────────────────────────────────────────────────
# Core A-application (per-oscillator 2×2 damped rotation), functional style
# ───────────────────────────────────────────────────────────────────────────────

@inline function apply_Ax(α::AbstractVector, θ::AbstractVector, x_t::AbstractVector)
    M = length(α)
    @assert length(θ) == M
    @assert length(x_t) == 2M

    pairs = map(1:M) do i
        x1 = x_t[2i-1]
        x2 = x_t[2i]
        a = α[i]
        c = Base.cos(θ[i])
        s = Base.sin(θ[i])
        (a * (c * x1 - s * x2), a * (s * x1 + c * x2))
    end

    return collect(Iterators.flatten(pairs))
end

# ───────────────────────────────────────────────────────────────────────────────
# Unit forward (one step): needs the bank parameters to apply A
# ───────────────────────────────────────────────────────────────────────────────

function (ssm::GatedOscSSMUnit)(
    u_t::AbstractVector,          # input at time t (input_dimension,)
    ps_ssm::NamedTuple,           # unit parameters (B0, C0, D, bias, gate_B, gate_C)
    st_ssm::NamedTuple,           # unit state (hidden_state)
    ps_bank::NamedTuple,           # bank parameters (logit_contraction, rotation_angle)
)
    # Pull params by name (order-independent)
    (; B0, C0, D, bias, gate_B, gate_C) = ps_ssm
    (; logit_contraction, rotation_angle) = ps_bank

    # Gates from input: g_B ∈ ℝ^{state_dim}, g_C ∈ ℝ^{output_dim}
    gB_aff = gate_B.weight * u_t + gate_B.bias
    gC_aff = gate_C.weight * u_t + gate_C.bias
    g_B = map(σ, gB_aff)
    g_C = map(σ, gC_aff)

    # Oscillator params α ∈ (0,1)^M, θ ∈ ℝ^M
    α = map(x -> min(max(σ(x), 1.0f-6), 0.9999f0), logit_contraction)
    θ = rotation_angle

    # State update: x_{t+1} = A x_t + Diagonal(g_B) * (B0 * u_t)
    x_t = st_ssm.hidden_state
    Ax = apply_Ax(α, θ, x_t)
    x_inj = LinearAlgebra.Diagonal(g_B) * (B0 * u_t)
    x_next = Ax + x_inj

    # Output: y_t = Diagonal(g_C) * (C0 * x_next) + D * u_t + bias
    y_core = LinearAlgebra.Diagonal(g_C) * (C0 * x_next)
    y_t = y_core + D * u_t + bias

    return y_t, (hidden_state = x_next,)
end

# ───────────────────────────────────────────────────────────────────────────────
# Block: owns one shared bank + one unit, delegates setup and forward
# ───────────────────────────────────────────────────────────────────────────────

# Parameters: nested under (:shared_bank, :ssm_unit)
function initialparameters(rng::Random.AbstractRNG, blk::OscSSMBlock)
    return (
        shared_bank = initialparameters(rng, blk.shared_bank),
        ssm_unit = initialparameters(rng, blk.ssm_unit),
    )
end

# States: nested under (:shared_bank, :ssm_unit)
function initialstates(rng::Random.AbstractRNG, blk::OscSSMBlock)
    return (
        shared_bank = initialstates(rng, blk.shared_bank),    # empty NamedTuple()
        ssm_unit = initialstates(rng, blk.ssm_unit),       # (hidden_state = …,)
    )
end

# Block forward (one step): wires bank params into the unit
function (blk::OscSSMBlock)(u_t::AbstractVector, ps_blk::NamedTuple, st_blk::NamedTuple)
    # Destructure param/state trees by name
    (; shared_bank = ps_bank, ssm_unit = ps_unit) = ps_blk
    (; shared_bank = st_bank, ssm_unit = st_unit) = st_blk

    # Delegate to unit, passing bank params
    y_t, st_unit′ = blk.ssm_unit(u_t, ps_unit, st_unit, ps_bank)

    # Repack block state
    st_blk′ = (shared_bank = st_bank, ssm_unit = st_unit′)

    return y_t, st_blk′
end

# ───────────────────────────────────────────────────────────────────────────────
# Sequence helpers for Oscillator scans
# ───────────────────────────────────────────────────────────────────────────────

# Mathematical definition:
# Let F_blk: ℝ^{d_in} × Θ_blk × Ξ_blk → ℝ^{d_out} × Ξ_blk be the map implemented by
# OscSSMBlock. For a time-major matrix U ∈ ℝ^{d_in×T}, define the scan
# S_blk(U, θ, ξ₀) = (Y, ξ_T) where
# Y[:, t] = π₁(F_blk(U[:, t], θ, ξ_{t-1})) and ξ_t = π₂(F_blk(U[:, t], θ, ξ_{t-1}))
# for t = 1,…,T with ξ₀ given. The Julia helper below materializes Y and ξ_T.
function propagate_oscillator_sequence(
    block::OscSSMBlock,
    seq::AbstractMatrix,
    ps_block::NamedTuple,
    st_block::NamedTuple,
)
    time_steps = size(seq, 2)
    output_dim = block.ssm_unit.output_dimension
    outputs = Matrix{Float32}(undef, output_dim, time_steps)
    step = function ((state, column_idx), u_t)
        y_t, state_next = block(u_t, ps_block, state)
        outputs[:, column_idx] = y_t                 # single mutation for efficiency
        return (state_next, column_idx + 1)
    end
    if time_steps == 0
        return outputs, st_block
    end
    acc_final = foldl(step, eachcol(seq); init = (st_block, 1))
    return outputs, acc_final[1]
end

# ───────────────────────────────────────────────────────────────────────────────
# Oscillatory Mamba mixer
# ───────────────────────────────────────────────────────────────────────────────

# Mathematical definition:
# Given input X ∈ ℝ^{d_in×T}, define affine maps P_in, P_skip: ℝ^{d_in×T} → ℝ^{d_model×T},
# a state-space scan S_blk from above, a layer normalization map N: ℝ^{d_model×T} → ℝ^{d_model×T},
# and an output affine map P_out: ℝ^{d_model×T} → ℝ^{d_model×T}. The oscillatory Mamba mixer
# implements M(X) = P_out(N(S_blk(P_in(X)) + P_skip(X))).
struct OscMambaMixer <: Lux.AbstractLuxLayer
    oscillator_block::OscSSMBlock
    input_proj::Lux.Dense
    skip_proj::Lux.Dense
    output_proj::Lux.Dense
    norm::Lux.LayerNorm
end

function OscMambaMixer(input_dim::Int, model_dim::Int, oscillator_count::Int)
    bank = OscillatorBank(oscillator_count)
    unit = GatedOscSSMUnit(oscillator_count, model_dim, model_dim)
    block = OscSSMBlock(unit, bank)
    return OscMambaMixer(
        block,
        Lux.Dense(input_dim, model_dim),
        Lux.Dense(input_dim, model_dim),
        Lux.Dense(model_dim, model_dim),
        Lux.LayerNorm(model_dim),
    )
end

function initialparameters(rng::Random.AbstractRNG, layer::OscMambaMixer)
    rngs = Random.split(rng, 5)
    return (
        oscillator_block = initialparameters(rngs[1], layer.oscillator_block),
        input_proj = Lux.initialparameters(rngs[2], layer.input_proj),
        skip_proj = Lux.initialparameters(rngs[3], layer.skip_proj),
        output_proj = Lux.initialparameters(rngs[4], layer.output_proj),
        norm = Lux.initialparameters(rngs[5], layer.norm),
    )
end

function initialstates(rng::Random.AbstractRNG, layer::OscMambaMixer)
    rngs = Random.split(rng, 5)
    return (
        oscillator_block = initialstates(rngs[1], layer.oscillator_block),
        input_proj = Lux.initialstates(rngs[2], layer.input_proj),
        skip_proj = Lux.initialstates(rngs[3], layer.skip_proj),
        output_proj = Lux.initialstates(rngs[4], layer.output_proj),
        norm = Lux.initialstates(rngs[5], layer.norm),
    )
end

function (layer::OscMambaMixer)(
    seq::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple,
)
    pre_proj, st_in = layer.input_proj(seq, ps.input_proj, st.input_proj)
    skip_proj, st_skip = layer.skip_proj(seq, ps.skip_proj, st.skip_proj)
    ssm_out, st_block = propagate_oscillator_sequence(
        layer.oscillator_block,
        pre_proj,
        ps.oscillator_block,
        st.oscillator_block,
    )
    fused = ssm_out + skip_proj
    normed, st_norm = layer.norm(fused, ps.norm, st.norm)
    out, st_out = layer.output_proj(normed, ps.output_proj, st.output_proj)
    return out,
    (
        oscillator_block = st_block,
        input_proj = st_in,
        skip_proj = st_skip,
        output_proj = st_out,
        norm = st_norm,
    )
end

# ───────────────────────────────────────────────────────────────────────────────
# Restricted window attention
# ───────────────────────────────────────────────────────────────────────────────

# Mathematical definition:
# For embeddings X ∈ ℝ^{d_model×T}, define per-head projections producing
# Q_h, K_h, V_h ∈ ℝ^{d_head×T}. For each token index i, attention is confined to
# the window W_i = {j : |i - j| ≤ w}. The head output satisfies
# H_h[:, i] = Σ_{j∈W_i} softmax_i(j) · V_h[:, j], where softmax_i is computed
# from scaled dot products restricted to W_i. Concatenating heads and applying the
# output affine yields the transformer contribution.
struct RestrictedAttention <: Lux.AbstractLuxLayer
    q_proj::Lux.Dense
    k_proj::Lux.Dense
    v_proj::Lux.Dense
    out_proj::Lux.Dense
    window_radius::Int
    num_heads::Int
    head_dim::Int
end

function RestrictedAttention(model_dim::Int, num_heads::Int, window_radius::Int)
    window_radius < 0 && throw(ArgumentError("window radius must be non-negative"))
    model_dim % num_heads == 0 ||
        throw(ArgumentError("model_dim must be divisible by num_heads"))
    head_dim = div(model_dim, num_heads)
    return RestrictedAttention(
        Lux.Dense(model_dim, model_dim),
        Lux.Dense(model_dim, model_dim),
        Lux.Dense(model_dim, model_dim),
        Lux.Dense(model_dim, model_dim),
        window_radius,
        num_heads,
        head_dim,
    )
end

function initialparameters(rng::Random.AbstractRNG, layer::RestrictedAttention)
    rngs = Random.split(rng, 4)
    return (
        q_proj = Lux.initialparameters(rngs[1], layer.q_proj),
        k_proj = Lux.initialparameters(rngs[2], layer.k_proj),
        v_proj = Lux.initialparameters(rngs[3], layer.v_proj),
        out_proj = Lux.initialparameters(rngs[4], layer.out_proj),
    )
end

function initialstates(rng::Random.AbstractRNG, layer::RestrictedAttention)
    rngs = Random.split(rng, 4)
    return (
        q_proj = Lux.initialstates(rngs[1], layer.q_proj),
        k_proj = Lux.initialstates(rngs[2], layer.k_proj),
        v_proj = Lux.initialstates(rngs[3], layer.v_proj),
        out_proj = Lux.initialstates(rngs[4], layer.out_proj),
    )
end

@inline function reshape_heads(x::AbstractMatrix, head_dim::Int, num_heads::Int)
    return reshape(x, head_dim, num_heads, size(x, 2))
end

@inline function band_mask(T::Int, radius::Int)
    radius < 0 && throw(ArgumentError("window radius must be non-negative"))
    return [abs(i - j) <= radius ? 0.0f0 : -Inf32 for i in 1:T, j in 1:T]
end

function restricted_head_attention(
    q_h::AbstractMatrix,
    k_h::AbstractMatrix,
    v_h::AbstractMatrix,
    radius::Int,
)
    time_steps = size(q_h, 2)
    scale = inv(sqrt(Float32(size(q_h, 1))))
    scores = scale .* (transpose(q_h) * k_h)
    masked = scores .+ band_mask(time_steps, radius)
    weights = NNlib.softmax(masked; dims = 2)
    return v_h * transpose(weights)
end

function (layer::RestrictedAttention)(
    x::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple,
)
    q_proj, st_q = layer.q_proj(x, ps.q_proj, st.q_proj)
    k_proj, st_k = layer.k_proj(x, ps.k_proj, st.k_proj)
    v_proj, st_v = layer.v_proj(x, ps.v_proj, st.v_proj)
    q_heads = reshape_heads(q_proj, layer.head_dim, layer.num_heads)
    k_heads = reshape_heads(k_proj, layer.head_dim, layer.num_heads)
    v_heads = reshape_heads(v_proj, layer.head_dim, layer.num_heads)
    context_heads = foldl(
        (tensor, head_idx) -> begin
            q_h = selectdim(q_heads, 2, head_idx)
            k_h = selectdim(k_heads, 2, head_idx)
            v_h = selectdim(v_heads, 2, head_idx)
            tensor[:, head_idx, :] = restricted_head_attention(
                q_h,
                k_h,
                v_h,
                layer.window_radius,
            )
            return tensor
        end,
        1:layer.num_heads;
        init = similar(q_heads),      # mutate per head to avoid reallocating tensors
    )
    merged = reshape(
        context_heads,
        layer.head_dim * layer.num_heads,
        size(x, 2),
    )
    out, st_out = layer.out_proj(merged, ps.out_proj, st.out_proj)
    return out,
    (
        q_proj = st_q,
        k_proj = st_k,
        v_proj = st_v,
        out_proj = st_out,
    )
end

# ───────────────────────────────────────────────────────────────────────────────
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
# Prime codec and tokenizer with partial corruption
# ───────────────────────────────────────────────────────────────────────────────

# Mathematical definition:
# A prime codec assigns each token index i to the i-th prime p_i, yielding an injective
# map C: {1,…,|V|} → ℕ. Encoding a token sequence τ gives the integer sequence
# C(τ). Partial masking replaces selected entries with C([MASK]), while partial
# unmasking exposes leading factors of p_i to leak controlled information.

struct PrimeCodec
    table::Vector{Int}
end

@inline function is_prime(n::Int)
    n <= 1 && return false
    (n == 2) && return true
    iseven(n) && return false
    limit = floor(Int, sqrt(n))
    return all(k -> n % k != 0, 3:2:limit)
end

function first_n_primes(count::Int)
    count < 0 && throw(ArgumentError("prime count must be non-negative"))
    count == 0 && return Int[]
    primes = Int[]
    candidate = 2
    while length(primes) < count
        # Imperative push keeps the sieve O(count log log count) without reallocations.
        is_prime(candidate) && push!(primes, candidate)
        candidate += candidate == 2 ? 1 : 2
    end
    return primes
end

PrimeCodec(count::Int) = PrimeCodec(first_n_primes(count))

@inline encode_prime(codec::PrimeCodec, idx::Int) = codec.table[idx]
@inline decode_prime(codec::PrimeCodec, value::Int) = findfirst(==(value), codec.table)

function prime_factors(n::Int)
    n < 2 && return Int[]
    residual = n
    factors = Int[]
    while iseven(residual)
        # Writing into `factors` is required to keep multiplicities explicitly.
        push!(factors, 2)
        residual ÷= 2
    end
    divisor = 3
    while divisor * divisor <= residual
        while residual % divisor == 0
            push!(factors, divisor)
            residual ÷= divisor
        end
        divisor += 2
    end
    residual > 1 && push!(factors, residual)
    return factors
end

@inline function partial_prime_view(value::Int)
    facs = prime_factors(value)
    facs_isempty = isempty(facs)
    return facs_isempty ? "[GL:∙]" :
           begin
               visible = max(1, cld(length(facs), 2))
               snippet = join(facs[1:visible], "·")
               "[GL:" * snippet * "]"
           end
end

struct PrimeTokenizer{T<:AbstractString}
    vocabulary::Vector{T}
    mask_token::T
    codec::PrimeCodec
    token_to_prime::Dict{T, Int}
    prime_to_token::Dict{Int, T}
end

function PrimeTokenizer(vocab::AbstractVector{T}; mask_token::T = "[MASK]") where {T<:AbstractString}
    vocab_vec = collect(vocab)
    has_mask = any(==(mask_token), vocab_vec)
    vocab_full = has_mask ? vocab_vec : vcat(vocab_vec, mask_token)
    codec = PrimeCodec(length(vocab_full))
    primes = codec.table
    token_to_prime = Dict{T, Int}(zip(vocab_full, primes))
    prime_to_token = Dict{Int, T}(zip(primes, vocab_full))
    return PrimeTokenizer(
        vocab_full,
        mask_token,
        codec,
        token_to_prime,
        prime_to_token,
    )
end

function prime_encode(tokenizer::PrimeTokenizer, tokens::AbstractVector{<:AbstractString})
    mask_prime = tokenizer.token_to_prime[tokenizer.mask_token]
    return map(token -> get(tokenizer.token_to_prime, token, mask_prime), tokens)
end

function prime_decode(tokenizer::PrimeTokenizer, codes::AbstractVector{<:Integer})
    return map(code -> get(tokenizer.prime_to_token, Int(code), tokenizer.mask_token), codes)
end

function corrupt_tokens(
    tokenizer::PrimeTokenizer,
    tokens::AbstractVector{<:AbstractString};
    mask_fraction::Real = 0.15,
    unmask_fraction::Real = 0.2,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    total = length(tokens)
    encoded = prime_encode(tokenizer, tokens)
    mask_total = clamp(round(Int, mask_fraction * total), 0, total)
    mask_total == 0 && return (
        tokens = collect(tokens),
        encoded = encoded,
        mask_indices = Int[],
        glimpse_indices = Int[],
    )
    order = Random.randperm(rng, total)
    selected = order[1:mask_total]
    glimpse_total = clamp(round(Int, unmask_fraction * mask_total), 0, mask_total)
    glimpse_indices = selected[1:glimpse_total]
    mask_indices = selected[glimpse_total+1:end]
    mask_token = tokenizer.mask_token
    mask_prime = tokenizer.token_to_prime[mask_token]
    corrupted_tokens, corrupted_primes = foldl(
        (state, idx) -> begin
            tokens_vec, primes_vec = state
            tokens_vec[idx] = mask_token
            primes_vec[idx] = mask_prime
            return (tokens_vec, primes_vec)
        end,
        mask_indices;
        init = (collect(tokens), copy(encoded)),
    )
    final_tokens = foldl(
        (tok_vec, idx) -> begin
            tok_vec[idx] = partial_prime_view(encoded[idx])
            return tok_vec
        end,
        glimpse_indices;
        init = corrupted_tokens,
    )
    return (
        tokens = final_tokens,
        encoded = corrupted_primes,
        mask_indices = sort(mask_indices),
        glimpse_indices = sort(glimpse_indices),
    )
end

end # module
