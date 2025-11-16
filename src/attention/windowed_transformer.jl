# Windowed transformer encoder with restricted attention
# ───────────────────────────────────────────────────────────────────────────────

# Mathematical definition:
# For sequence embeddings X ∈ ℝ^{d_model×T}, define two layer-normalized streams
# LN₁(X) and LN₂(Y). Let Attn_W: ℝ^{d_model×T} → ℝ^{d_model×T} denote the
# window-restricted multi-head attention induced by RestrictedAttention with
# window radius w. The encoder computes
# Y = X + Attn_W(LN₁(X))
# Z = Y + W₂·φ(W₁·LN₂(Y)), where W₁ ∈ ℝ^{d_ff×d_model}, W₂ ∈ ℝ^{d_model×d_ff}
# and φ = GELU applied elementwise. Returning Z preserves the local receptive
# field while adding transformer-style residual mixing.
struct WindowedTransformer <: Lux.AbstractLuxLayer
    attention::RestrictedAttention
    attn_norm::Lux.LayerNorm
    ff_norm::Lux.LayerNorm
    ff_up::Lux.Dense
    ff_down::Lux.Dense
end

function WindowedTransformer(
    model_dim::Int,
    num_heads::Int,
    window_radius::Int;
    expansion_factor::Real = 4.0,
)
    expansion_factor > 0 || throw(ArgumentError("expansion_factor must be positive"))
    hidden_dim = max(model_dim, Int(ceil(model_dim * Float32(expansion_factor))))
    return WindowedTransformer(
        RestrictedAttention(model_dim, num_heads, window_radius),
        Lux.LayerNorm(model_dim),
        Lux.LayerNorm(model_dim),
        Lux.Dense(model_dim, hidden_dim),
        Lux.Dense(hidden_dim, model_dim),
    )
end

function initialparameters(rng::Random.AbstractRNG, block::WindowedTransformer)
    rngs = scatter_rngs(rng, 5)
    return (
        attention = initialparameters(rngs[1], block.attention),
        attn_norm = Lux.initialparameters(rngs[2], block.attn_norm),
        ff_norm = Lux.initialparameters(rngs[3], block.ff_norm),
        ff_up = Lux.initialparameters(rngs[4], block.ff_up),
        ff_down = Lux.initialparameters(rngs[5], block.ff_down),
    )
end

function initialstates(rng::Random.AbstractRNG, block::WindowedTransformer)
    rngs = scatter_rngs(rng, 5)
    return (
        attention = initialstates(rngs[1], block.attention),
        attn_norm = Lux.initialstates(rngs[2], block.attn_norm),
        ff_norm = Lux.initialstates(rngs[3], block.ff_norm),
        ff_up = Lux.initialstates(rngs[4], block.ff_up),
        ff_down = Lux.initialstates(rngs[5], block.ff_down),
    )
end

@inline function _gelu(x::AbstractMatrix)
    return NNlib.gelu.(x)
end

function (block::WindowedTransformer)(
    x::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple;
    radius_override::Union{Nothing,Int}=nothing,
)
    normed_attn, st_attn_norm = block.attn_norm(x, ps.attn_norm, st.attn_norm)
    attn_out, st_attn = block.attention(
        normed_attn,
        ps.attention,
        st.attention;
        radius_override = radius_override,
    )
    attn_residual = x + attn_out
    normed_ff, st_ff_norm = block.ff_norm(attn_residual, ps.ff_norm, st.ff_norm)
    ff_hidden, st_ff_up = block.ff_up(normed_ff, ps.ff_up, st.ff_up)
    ff_hidden = _gelu(ff_hidden)
    ff_out, st_ff_down = block.ff_down(ff_hidden, ps.ff_down, st.ff_down)
    final_out = attn_residual + ff_out
    return final_out,
    (
        attention = st_attn,
        attn_norm = st_attn_norm,
        ff_norm = st_ff_norm,
        ff_up = st_ff_up,
        ff_down = st_ff_down,
    )
end

# ───────────────────────────────────────────────────────────────────────────────

# Logarithmic radius adaptor:
# For embeddings X ∈ ℝ^{d_model×T}, define r(T) = clamp(⌈α·log(1+T)⌉, r_min, r_max).
# The adaptor feeds the base WindowedTransformer with radius r(T), yielding overall
# complexity O(T log T) when α is constant while preserving compact neighborhoods
# for shorter contexts.
struct LogWindowTransformer <: Lux.AbstractLuxLayer
    transformer::WindowedTransformer
    radius_factor::Float32
    min_radius::Int
    max_radius::Int
    base_radius::Int
end

function LogWindowTransformer(
    model_dim::Int,
    num_heads::Int;
    radius_factor::Real = 4.0,
    min_radius::Int = 1,
    max_radius::Union{Int,Nothing} = nothing,
    base_radius::Int = 0,
)
    radius_factor <= 0 && throw(ArgumentError("radius_factor must be positive"))
    min_r = max(1, min_radius)
    max_r = isnothing(max_radius) ? typemax(Int) : max_radius
    transformer = WindowedTransformer(model_dim, num_heads, min_r)
    return LogWindowTransformer(transformer, Float32(radius_factor), min_r, max_r, base_radius)
end

initialparameters(rng::Random.AbstractRNG, layer::LogWindowTransformer) =
    initialparameters(rng, layer.transformer)

initialstates(rng::Random.AbstractRNG, layer::LogWindowTransformer) =
    initialstates(rng, layer.transformer)

@inline function _log_window_radius(time_steps::Int, factor::Float32, min_r::Int, max_r::Int, base_r::Int)
    steps = max(time_steps, 1)
    growth = log1p(Float32(steps))
    raw = Int(ceil(factor * growth)) + base_r
    return clamp(raw, min_r, max_r)
end

function (layer::LogWindowTransformer)(
    x::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple,
)
    time_steps = size(x, 2)
    radius = _log_window_radius(time_steps, layer.radius_factor, layer.min_radius, layer.max_radius, layer.base_radius)
    return layer.transformer(x, ps, st; radius_override = radius)
end

const SqrtWindowTransformer = LogWindowTransformer

# ───────────────────────────────────────────────────────────────────────────────
