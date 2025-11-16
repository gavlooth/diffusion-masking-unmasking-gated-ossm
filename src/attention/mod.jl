module Attention

using ..Lux
using ..LuxCore
import ..LuxCore: initialparameters, initialstates
using ..Random
using ..NNlib

const _ROOT = parentmodule(@__MODULE__)
const scatter_rngs = _ROOT.scatter_rngs

include("restricted_attention.jl")
include("windowed_transformer.jl")

export RestrictedAttention, WindowedTransformer, LogWindowTransformer, SqrtWindowTransformer

end # module Attention
