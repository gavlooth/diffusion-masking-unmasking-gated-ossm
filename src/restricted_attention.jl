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
    rngs = scatter_rngs(rng, 4)
    return (
        q_proj = Lux.initialparameters(rngs[1], layer.q_proj),
        k_proj = Lux.initialparameters(rngs[2], layer.k_proj),
        v_proj = Lux.initialparameters(rngs[3], layer.v_proj),
        out_proj = Lux.initialparameters(rngs[4], layer.out_proj),
    )
end

function initialstates(rng::Random.AbstractRNG, layer::RestrictedAttention)
    rngs = scatter_rngs(rng, 4)
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
    st::NamedTuple;
    radius_override::Union{Nothing,Int}=nothing,
)
    radius = isnothing(radius_override) ? layer.window_radius : radius_override
    radius < 0 && throw(ArgumentError("window radius must be non-negative"))
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
                radius,
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
