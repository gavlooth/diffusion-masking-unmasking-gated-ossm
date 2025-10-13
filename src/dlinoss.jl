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

struct OscSSM <: Lux.AbstractLuxLayer
    K::Int
    input_dim::Int
    output_dim::Int
end

σ(x) = 1 / (1 + exp(-x))    # sigmoid for α ∈ (0,1)


function Lux.initialparameters(rng::Random.AbstractRNG, m::OscSSM)
    statedim = 2 * m.K
    logit_α = Random.randn(rng, Float32, m.K) |> y -> map(x -> x * 0.01f0 + -0.05f0, y)
    θ = map(x -> 0.2f0 * x, Random.randn(rng, Float32, m.K))
    B = map(x -> 0.01f0 * x, Random.randn(rng, Float32, statedim, m.input_dim))
    C = map(x -> 0.01f0 * x, Random.randn(rng, Float32, m.output_dim, statedim))
    D = Base.zeros(Float32, m.output_dim, m.input_dim)
    b = Base.zeros(Float32, m.output_dim)
    (logit_alpha = logit_α, theta = θ, B = B, C = C, D = D, b = b)
end

function Lux.initialstates(rng::Random.AbstractRNG, m::OscSSM)
    zeros(Float32, m.K * 2)
end

function _mul_Ax!(xout, x, α, cθ, sθ)
    K = length(α)
    @inbounds for i = 1:K
        i1 = 2i - 1
        i2 = 2i
        xi1 = x[i1]
        xi2 = x[i2]
        a = α[i]
        c = cθ[i]
        s = sθ[i]
        xout[i1] = a * (c * xi1 - s * xi2)
        xout[i2] = a * (s * xi1 + c * xi2)
    end
    return xout
end


function (m::OscSSM)(U, ps, st::NamedTuple)
    sd = m.k * 2
    α = clamp.(σ.(ps.logit_alpha), 1.0f-6, 0.9999f0)i
    cθ = cos.(ps.theta)
    sθ = Base.sin.(ps.theta)
    x = copy(st.x)


    if ndims(U) == 1

        @assert length(U) == m.input_dim "Input vector must have length input_dim"

        # x_{t+1} = A x_t
        _mul_Ax!(x, x, α, cθ, sθ)

        # x_{t+1} += B u_t
        x .+= ps.B * U

        # y_t = C x_{t+1} + D u_t + b
        y = ps.C * x .+ ps.D * U .+ ps.b

        return y, (x = x,)

    elseif ndims(U) == 2
        @assert size(U, 1) == m.input_dim "U must have size (input_dim, T)"
        T = size(U, 2)

        # Allocate output Y: (output_dim, T)
        Y = similar(ps.C * x .+ ps.b, m.output_dim, T)

        @inbounds for t = 1:T
            u = view(U, :, t)  # u_t

            # x_{t+1} = A x_t
            _mul_Ax!(x, x, α, cθ, sθ)

            x .+= ps.B * u

            # y_t = C x_{t+1} + D u_t + b
            @views Y[:, t] .= ps.C * x .+ ps.D * u .+ ps.b
        end

        return Y, (x = x,)

    else
        error(
            "Unsupported input U with ndims=$(ndims(U)). Use a vector or (input_dim, T) matrix.",
        )
    end

end


end
# @show size(gW[1])  # same size as W

