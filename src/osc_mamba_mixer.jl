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
    rngs = scatter_rngs(rng, 5)
    return (
        oscillator_block = initialparameters(rngs[1], layer.oscillator_block),
        input_proj = Lux.initialparameters(rngs[2], layer.input_proj),
        skip_proj = Lux.initialparameters(rngs[3], layer.skip_proj),
        output_proj = Lux.initialparameters(rngs[4], layer.output_proj),
        norm = Lux.initialparameters(rngs[5], layer.norm),
    )
end

function initialstates(rng::Random.AbstractRNG, layer::OscMambaMixer)
    rngs = scatter_rngs(rng, 5)
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
