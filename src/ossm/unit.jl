module Unit

import Base
import Lux
import Random

using ..Activations: σ
using ..Gating: compute_gates, dynamic_BC

# Core oscillatory state-space unit and its Lux integration hooks.

"""
    GatedOscSSMUnit(K, input_dim, output_dim)

One-step oscillatory state-space unit with:
- hidden state x_t ∈ ℝ^{2K}
- input      u_t ∈ ℝ^{input_dim}
- output     y_t ∈ ℝ^{output_dim}

The update is
    x_{t+1} = A x_t + B_t u_t
    y_t     = C_t x_{t+1} + D u_t + b

where:
- A is block-diagonal with 2×2 damped rotations (oscillators),
- B_t, C_t are data-dependent versions of base matrices B₀, C₀,
  obtained by row-wise gating (g_B(u_t), g_C(u_t)).

Subtyping <: Lux.AbstractLuxLayer tells Lux:
- “this is a layer type”
so Lux can generically do:
- `initialparameters(rng, unit)`,
- `initialstates(rng, unit)`,
- `unit(u, ps, st)` inside larger models via multiple dispatch.
"""
struct GatedOscSSMUnit <: Lux.AbstractLuxLayer
    K::Int         # number of 2D oscillators → state dimension = 2K
    input_dim::Int
    output_dim::Int
end

"""
    statedim(u::GatedOscSSMUnit) -> Int

Returns the hidden state dimension ``2K``.
"""
statedim(u::GatedOscSSMUnit) = 2 * u.K

"""
    Lux.initialparameters(rng, unit::GatedOscSSMUnit) -> ps

Initialise all trainable parameters in a NamedTuple `ps`.

A NamedTuple here is an immutable record-like container, e.g.
    (B0 = ..., C0 = ..., D = ..., b = ...)
that Lux and Functors can traverse, move between devices, etc.

Fields of `ps`:
- `logit_alpha`, `theta` parameterise the diagonal blocks of ``A``.
- `B0`, `C0`, `D`, `b` are the base matrices and bias of the affine map
  ``(x,u) ↦ (A x + B_0 u, C_0 x + D u + b)``.
- `gateB`, `gateC` are `NamedTuple`s whose fields `W` and `b` store the
  matrices in the gate definitions.  Each `NamedTuple` behaves like an immutable
  record; Lux traverses them when moving parameters to devices.
"""
function Lux.initialparameters(rng::Random.AbstractRNG, unit::GatedOscSSMUnit)
    sd = statedim(unit)  # = 2K

    # Oscillatory A via (α, θ)
    logit_alpha = -0.05f0 .+ 0.01f0 .* Random.randn(rng, Float32, unit.K)
    theta = 0.20f0 .* Random.randn(rng, Float32, unit.K)

    # Base SSM matrices: B₀, C₀, D, b
    B0 = 0.01f0 .* Random.randn(rng, Float32, sd, unit.input_dim)
    C0 = 0.01f0 .* Random.randn(rng, Float32, unit.output_dim, sd)
    D = Base.zeros(Float32, unit.output_dim, unit.input_dim)
    b = Base.zeros(Float32, unit.output_dim)

    # Gating maps:
    #   G_B: ℝ^{input_dim}  → ℝ^{2K}
    #   G_C: ℝ^{input_dim}  → ℝ^{output_dim}
    WgB = 0.01f0 .* Random.randn(rng, Float32, sd, unit.input_dim)
    bgB = Base.zeros(Float32, sd)
    WgC = 0.01f0 .* Random.randn(rng, Float32, unit.output_dim, unit.input_dim)
    bgC = Base.zeros(Float32, unit.output_dim)

    return (
        logit_alpha = logit_alpha,
        theta = theta,
        B0 = B0,
        C0 = C0,
        D = D,
        b = b,
        gateB = (W = WgB, b = bgB),
        gateC = (W = WgC, b = bgC),
    )
end

"""
    Lux.initialstates(rng, unit::GatedOscSSMUnit) -> st

Initial hidden state x₁ = 0 stored in a NamedTuple `st`.
"""
function Lux.initialstates(::Random.AbstractRNG, unit::GatedOscSSMUnit)
    return (x = Base.zeros(Float32, statedim(unit)),)
end

raw"""
    mul_Ax(unit, ps, x)

Map ``x ∈ ℝ^{2K}`` to ``A x`` where
``A = diag(α₁ R(θ₁), …, α_K R(θ_K))``.  Each block
``R(θ_i) = \begin{pmatrix} \cos θ_i & -\sin θ_i \\ \sin θ_i & \cos θ_i \end{pmatrix}``
and ``α_i = σ(logit_alpha_i)``.

α_i = σ(logit_alpha[i]) ∈ (0,1), θ_i = theta[i].
"""
function mul_Ax(unit::GatedOscSSMUnit, ps, x::AbstractVector{<:Real})
    α = Base.clamp.(σ.(ps.logit_alpha), 1.0f-6, 0.9999f0)  # elementwise σ
    cθ = Base.cos.(ps.theta)
    sθ = Base.sin.(ps.theta)

    K = Base.length(α)

    # Base.map iterates i = 1,…,K and returns a tuple of 2-vectors; Julia's
    # do-block syntax `do i` introduces the function inline.
    blocks = Base.map(1:K) do i
        i1 = 2i - 1
        i2 = 2i
        xi1 = x[i1]
        xi2 = x[i2]
        a = α[i]
        c = cθ[i]
        s = sθ[i]

        # (c -s; s c) (xi1; xi2) scaled by a.
        (a * (c * xi1 - s * xi2), a * (s * xi1 + c * xi2))
    end

    return Base.collect(Iterators.flatten(blocks))  # length 2K
end

"""
    (unit::GatedOscSSMUnit)(u_t, ps, st) -> (y_t, st')

One time step:
    x_{t+1} = A x_t + B_t u_t
    y_t     = C_t x_{t+1} + D u_t + b

This call method is what makes the unit a callable layer
for Lux: the forward pass is (u_t, ps, st) ↦ (y_t, st').
"""
function (unit::GatedOscSSMUnit)(u_t::AbstractVector{<:Real}, ps, st::NamedTuple)
    x_t = st.x

    # 1) Data-dependent gates from u_t.
    gB, gC = compute_gates(unit, ps, u_t)

    # 2) Dynamic B_t, C_t.
    B_t, C_t = dynamic_BC(unit, ps, gB, gC)

    # 3) State update: x_{t+1} = A x_t + B_t u_t.
    x_next = mul_Ax(unit, ps, x_t) .+ B_t * u_t

    # 4) Output: y_t = C_t x_{t+1} + D u_t + b.
    y_t = C_t * x_next .+ ps.D * u_t .+ ps.b

    return y_t, (x = x_next,)
end

export GatedOscSSMUnit, statedim, mul_Ax

end
