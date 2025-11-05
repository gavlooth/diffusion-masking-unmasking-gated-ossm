module ossm

# The ossm namespace is now organised into focused submodules:
# - Activations: shared nonlinearities used across state updates.
# - Gating: helpers for data-dependent modulation of SSM matrices.
# - Unit: the single-step gated oscillator state-space layer.
# - Sequence: sequence-processing wrappers built on top of the unit.

include("ossm/activations.jl")
include("ossm/gating.jl")
include("ossm/unit.jl")
include("ossm/sequence.jl")

using .Activations: σ
using .Gating: compute_gates, row_scale, dynamic_BC
using .Unit: GatedOscSSMUnit, statedim, mul_Ax
using .Sequence: SequenceGatedOSSM

export σ
export compute_gates, row_scale, dynamic_BC
export GatedOscSSMUnit, statedim, mul_Ax
export SequenceGatedOSSM

end
