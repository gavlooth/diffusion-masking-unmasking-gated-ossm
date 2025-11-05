module TimePos

import Base
import Lux
import Random

# -----------------------------------------------------------------------------
# Time + position encoding
# -----------------------------------------------------------------------------

"""
    TimePosEncoding(d_model, max_len)

Lux layer:
- input: H ∈ ℝ^{d_model×N},
- output: H + P[:,pos:pos+N-1] + e_t,
  where P is a learned position matrix and e_t is a learned encoding
  of scalar time t.
"""
struct TimePosEncoding <: Lux.AbstractLuxLayer
    d_model::Int
    max_len::Int
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::TimePosEncoding)
    P = 0.01f0 .* Random.randn(rng, Float32, l.d_model, l.max_len)

    # time MLP: ℝ → ℝ^{d_model}, via ℝ^{d_model}
    Wt1 = 0.1f0 .* Random.randn(rng, Float32, l.d_model, 1)
    bt1 = Base.zeros(Float32, l.d_model)
    Wt2 = 0.1f0 .* Random.randn(rng, Float32, l.d_model, l.d_model)
    bt2 = Base.zeros(Float32, l.d_model)

    return (P = P, time = (Wt1 = Wt1, bt1 = bt1, Wt2 = Wt2, bt2 = bt2))
end

Lux.initialstates(::Random.AbstractRNG, ::TimePosEncoding) = NamedTuple()

function (l::TimePosEncoding)(
    H::AbstractMatrix{<:Real},
    ps,
    st;
    t::Float32,
    start_pos::Int = 1,
)
    d_model, N = Base.size(H)

    P_slice = Base.view(ps.P, :, start_pos:start_pos+N-1)  # ℝ^{d_model×N}

    # Time embedding: scalar t → ℝ^{d_model}
    t_vec = Float32[t]                      # (1,)
    h1 = ps.time.Wt1 * t_vec .+ ps.time.bt1
    h1 = Base.max.(h1, 0.0f0)               # ReLU
    e_t = ps.time.Wt2 * h1 .+ ps.time.bt2   # (d_model,1)

    E_t = Base.repeat(e_t, 1, N)           # broadcast over positions

    H_out = H .+ P_slice .+ E_t
    return H_out, st
end

export TimePosEncoding

end
