module Models

using ..Lux
using ..LuxCore
import ..LuxCore: initialparameters, initialstates
using ..Random

using ..Attention: WindowedTransformer, LogWindowTransformer

const _ROOT = parentmodule(@__MODULE__)
const scatter_rngs = _ROOT.scatter_rngs
const OscMambaMixer = _ROOT.OscMambaMixer
const PrimeTokenizer = _ROOT.PrimeTokenizer
const PrimeSample = _ROOT.PrimeSample
const corrupt_tokens = _ROOT.corrupt_tokens
const normalize_prime_codes = _ROOT.normalize_prime_codes

include("energy_guided.jl")
include("perigee_diffusion.jl")

export EnergyInterleaveBlock,
       EnergyGuidedDiffusionLLM,
       build_energy_guided_llm,
       PerigeeMixerBlock,
       PerigeeDiffusionLM,
       build_perigee_model,
       perigee_prepare_batch,
       perigee_diffusion_loss

end # module Models
