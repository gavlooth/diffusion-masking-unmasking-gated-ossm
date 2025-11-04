module dlinoss


import Lux, LuxCore, Random
import Functors              # for fmap
import Setfield              # enables fmap support on Lux layers
import CUDA                  # for CUDA.cu
import LuxCUDA               # trigger package for Lux GPU backend

#=
using Revise
=#

const HAVE_CUDA = try
    @eval using CUDA
    CUDA.has_cuda()
catch
    false
end
# -------------------------
# Basic helpers
# -------------------------

struct OscSSMCell <: Lux.AbstractLuxLayer
    K::Int
    input_dim::Int
    output_dim::Int
end

σ(x) = 1 / (1 + exp(-x))    # sigmoid for α ∈ (0,1)

@inline statedim(cell::OscSSMCell) = 2 * cell.K

function Lux.initialparameters(rng::Random.AbstractRNG, cell::OscSSMCell)
    sdim = statedim(cell)
    logit_α = Random.randn(rng, Float32, cell.K) |> y -> map(x -> x * 0.01f0 + -0.05f0, y)
    θ = map(x -> 0.2f0 * x, Random.randn(rng, Float32, cell.K))
    B = map(x -> 0.01f0 * x, Random.randn(rng, Float32, sdim, cell.input_dim))
    C = map(x -> 0.01f0 * x, Random.randn(rng, Float32, cell.output_dim, sdim))
    D = Base.zeros(Float32, cell.output_dim, cell.input_dim)
    b = Base.zeros(Float32, cell.output_dim)
    (logit_alpha = logit_α, theta = θ, B = B, C = C, D = D, b = b)
end

# the_cell = OscSSMCell(2, 3, 4)

function Lux.initialstates(::Random.AbstractRNG, cell::OscSSMCell)
    return zeros(Float32, statedim(cell))
end


function Lux.initialparameters(rng::Random.AbstractRNG, cell::OscSSMCell)
    sd = statedim(cell)

    logit_alpha = -0.05f0 .+ 0.01f0 .* Random.randn(rng, Float32, cell.K)
    theta = 0.20f0 .* Random.randn(rng, Float32, cell.K)

    B = 0.01f0 .* Random.randn(rng, Float32, sd, cell.input_dim)
    C = 0.01f0 .* Random.randn(rng, Float32, cell.output_dim, sd)
    D = Base.zeros(Float32, cell.output_dim, cell.input_dim)
    b = Base.zeros(Float32, cell.output_dim)

    return (logit_alpha = logit_alpha, theta = theta, B = B, C = C, D = D, b = b)
end

function Lux.initialstates(::Random.AbstractRNG, cell::OscSSMCell)
    return (x = Base.zeros(Float32, statedim(cell)),)
end


function mul_Ax(cell::OscSSMCell, ps, x::AbstractVector{<:Real})
    # α, θ → per-oscillator rotation+decay
    α = Base.clamp.(σ.(ps.logit_alpha), 1.0f-6, 0.9999f0)
    cθ = Base.cos.(ps.theta)
    sθ = Base.sin.(ps.theta)

    K = Base.length(α)

    # functional: build a list of 2D transformed states, then flatten
    blocks = Base.map(1:K) do i
        i1 = 2i - 1
        i2 = 2i
        xi1 = x[i1]
        xi2 = x[i2]

        a = α[i]
        c = cθ[i]
        s = sθ[i]

        # new 2D state
        (a * (c * xi1 - s * xi2), a * (s * xi1 + c * xi2))
    end

    # flatten (x'₁, x'₂) blocks to one vector of length 2K
    return Base.collect(Iterators.flatten(blocks))
end


function (cell::OscSSMCell)(u_t::AbstractVector{<:Real}, ps, st::NamedTuple)
    # u_t: (input_dim,)
    x_t = st.x                    # (2K,)
    x_next = mul_Ax(cell, ps, x_t) .+ ps.B * u_t
    y_t = ps.C * x_next .+ ps.D * u_t .+ ps.b
    return y_t, (x = x_next,)
end

struct SequenceOSSM <: Lux.AbstractLuxLayer
    cell::OscSSMCell
end

# parameters/states come from the inner cell
Lux.initialparameters(rng::Random.AbstractRNG, m::SequenceOSSM) =
    Lux.initialparameters(rng, m.cell)

Lux.initialstates(rng::Random.AbstractRNG, m::SequenceOSSM) = Lux.initialstates(rng, m.cell)


function (m::SequenceOSSM)(H::AbstractMatrix{<:Real}, ps, st::NamedTuple)
    # H: (d_model, N)
    d_model, N = Base.size(H)

    # Turn columns into a list of vectors u_t
    inputs = [Base.view(H, :, t) for t = 1:N]

    # foldl over time dimension, accumulating:
    #   - reversed list of outputs
    #   - state
    init_acc = (Vector{Vector{Float32}}(), st)

    (ys_rev, st_final) = Base.foldl(inputs; init = init_acc) do (acc_ys, acc_st), u_t
        (ys_list, st_t) = (acc_ys, acc_st)
        y_t, st_next = m.cell(u_t, ps, st_t)
        # prepend for O(1) cons; reverse later
        (Vector{Float32}[y_t; ys_list], st_next)
    end

    ys = Base.reverse(ys_rev)
    Y = Base.reduce(hcat, ys)  # (d_model, N)

    return Y, st_final
end

struct OSSMBlock <: Lux.AbstractLuxLayer
    d_model::Int
    d_hidden::Int
    ssm::SequenceOSSM
    norm1::Lux.LayerNorm
    norm2::Lux.LayerNorm
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::OSSMBlock)
    # params for sublayers
    ps_ssm, st_ssm = Lux.setup(rng, block.ssm)
    ps_norm1, st_norm1 = Lux.setup(rng, block.norm1)
    ps_norm2, st_norm2 = Lux.setup(rng, block.norm2)

    # MLP weights
    W1 = 0.01f0 .* Random.randn(rng, Float32, block.d_hidden, block.d_model)
    b1 = Base.zeros(Float32, block.d_hidden)
    W2 = 0.01f0 .* Random.randn(rng, Float32, block.d_model, block.d_hidden)
    b2 = Base.zeros(Float32, block.d_model)

    return (
        ssm = ps_ssm,
        norm1 = ps_norm1,
        norm2 = ps_norm2,
        mlp = (W1 = W1, b1 = b1, W2 = W2, b2 = b2),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::OSSMBlock)
    st_ssm, = (Lux.initialstates(rng, block.ssm),)
    st_norm1, = (Lux.initialstates(rng, block.norm1),)
    st_norm2, = (Lux.initialstates(rng, block.norm2),)
    return (ssm = st_ssm, norm1 = st_norm1, norm2 = st_norm2)
end


function (block::OSSMBlock)(H::AbstractMatrix{<:Real}, ps, st::NamedTuple)
    # H: (d_model, N)
    d_model, N = Base.size(H)

    # 1) Norm1
    Hn, st_norm1 = block.norm1(H, ps.norm1, st.norm1)

    # 2) MLP_in: position-wise (same weight for each column)
    #    h̃ = W2 * relu(W1 * Hn + b1) + b2
    # apply to all positions at once (matrix form)
    z1 = ps.mlp.W1 * Hn .+ ps.mlp.b1  # (d_hidden, N)
    h_act = Base.max.(z1, 0.0f0)      # ReLU
    H_tilde = ps.mlp.W2 * h_act .+ ps.mlp.b2  # (d_model, N)

    # 3) Sequence OSSM along token dimension
    H_ssm, st_ssm = block.ssm(H_tilde, ps.ssm, st.ssm)

    # Residual 1
    H1 = H .+ H_ssm

    # 4) Norm2
    Hn2, st_norm2 = block.norm2(H1, ps.norm2, st.norm2)

    # 5) Second MLP (same params as first here for simplicity; usually separate)
    z2 = ps.mlp.W1 * Hn2 .+ ps.mlp.b1
    h2 = Base.max.(z2, 0.0f0)
    H_mlp = ps.mlp.W2 * h2 .+ ps.mlp.b2

    # Residual 2
    H_out = H1 .+ H_mlp

    return H_out, (ssm = st_ssm, norm1 = st_norm1, norm2 = st_norm2)
end
end
# @show size(gW[1])  # same size as W

