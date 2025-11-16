# Layers (shape-only structs)
# ───────────────────────────────────────────────────────────────────────────────

import ChainRulesCore: rrule, NoTangent
import Zygote

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

const MIN_ALPHA = 1.0f-6
const MAX_ALPHA = 0.9999f0

@inline function sigmoid_map(x::AbstractArray)
    return map(σ, x)
end

function rrule(::typeof(sigmoid_map), x::AbstractArray)
    y = sigmoid_map(x)
    function sigmoid_pullback(Δ)
        grad = map(y, Δ) do y_val, Δ_val
            Δ_val * y_val * (1f0 - y_val)
        end
        return (NoTangent(), grad)
    end
    return y, sigmoid_pullback
end

@inline function contraction_sigmoid(x::AbstractArray)
    return map(value -> clamp(σ(value), MIN_ALPHA, MAX_ALPHA), x)
end

function rrule(::typeof(contraction_sigmoid), x::AbstractArray)
    y = contraction_sigmoid(x)
    function contraction_pullback(Δ)
        grad = map(y, Δ) do y_val, Δ_val
            active = (y_val > MIN_ALPHA && y_val < MAX_ALPHA) ? 1f0 : 0f0
            Δ_val * y_val * (1f0 - y_val) * active
        end
        return (NoTangent(), grad)
    end
    return y, contraction_pullback
end

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

    state = reshape(x_t, 2, M)
    @views begin
        row1 = state[1, :]
        row2 = state[2, :]
        cosθ = Base.cos.(θ)
        sinθ = Base.sin.(θ)
        y1 = α .* (cosθ .* row1 .- sinθ .* row2)
        y2 = α .* (sinθ .* row1 .+ cosθ .* row2)
        combined = permutedims(hcat(y1, y2), (2, 1))
        return reshape(combined, :)
    end
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
    g_B = sigmoid_map(gB_aff)
    g_C = sigmoid_map(gC_aff)

    # Oscillator params α ∈ (0,1)^M, θ ∈ ℝ^M
    α = contraction_sigmoid(logit_contraction)
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
    ps_bank = ps_blk.shared_bank
    ps_unit = ps_blk.ssm_unit
    st_bank = st_blk.shared_bank
    st_unit = st_blk.ssm_unit

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
    if time_steps == 0
        return similar(seq, Float32, output_dim, 0), st_block
    end
    @views first_column = seq[:, 1]
    y_first, state = block(first_column, ps_block, st_block)
    columns = Zygote.Buffer(Vector{typeof(y_first)}(undef, time_steps))
    columns[1] = y_first
    for idx in 2:time_steps
        @views column = seq[:, idx]
        y_t, state = block(column, ps_block, state)
        columns[idx] = y_t
    end
    outputs = hcat(copy(columns)...)
    return outputs, state
end

# ───────────────────────────────────────────────────────────────────────────────
