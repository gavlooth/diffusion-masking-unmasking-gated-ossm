module Activations

# Nonlinearity helpers that stay shared across the package's oscillatory SSMs.

σ(x) = 1.0f0 / (1.0f0 + Base.exp(-x))

export σ

end
