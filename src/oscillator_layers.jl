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
