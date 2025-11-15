@inline function scatter_rngs(rng::Random.AbstractRNG, count::Int)
    return [Random.MersenneTwister(Random.rand(rng, UInt)) for _ in 1:count]
end

@inline function scatter_rng(rng::Random.AbstractRNG)
    return Random.MersenneTwister(Random.rand(rng, UInt))
end
