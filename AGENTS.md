# Repository Guidelines
Use these guidelines to keep ossmv2 contributions reproducible, reviewable, and quick to land.

## Project Structure & Module Organization
The repo follows the standard Julia layout: `Project.toml` and `Manifest.toml` declare dependencies, and the `ossmv2` module currently lives in `src/ossmv2.jl`. Extend functionality by adding files under `src/` and including them from `ossmv2.jl`; mirror each feature under `test/feature_name.jl` once the suite exists so reviewers can trace coverage directly.

## Build, Test, and Development Commands
- `julia --project -e 'using Pkg; Pkg.instantiate()'`: hydrate dependencies from `Project.toml`/`Manifest.toml`.
- `julia --project -e 'using Revise, ossmv2; greet()'`: launch a Revise REPL for manual runs; swap `greet()` for the function you need.
- `julia --project -e 'using Pkg; Pkg.test()'`: run the automated suite (the command CI should call once `test/runtests.jl` exists).

## Coding Style & Naming Conventions
Use 4-space indentation, ~92-character lines, descriptive `snake_case` functions, and `CamelCase` types. Favor functional composition (`map`, `foldl`, comprehensions) over broadcasting or loops; if mutation is required, name the binding and reason. Keep literals typed (`0.01f0`), qualify Base/stdlib calls for clarity, guard GPU paths with capability checks, add docstrings for public APIs, and format via `julia --project -e 'using JuliaFormatter; format("src")'` when available.

In general, express elementwise logic with short functional expressions (`map`, comprehensions, small `do` blocks) instead of hand-written loops or broadcast chains. Custom AD hooks (`rrule`, pullbacks, etc.) should model the preference:

```
function rrule(::typeof(sigmoid_map), x)
    y = sigmoid_map(x)
    function pullback(Δ)
        grad = map(y, Δ) do yᵢ, Δᵢ
            Δᵢ * yᵢ * (1f0 - yᵢ)
        end
        return NoTangent(), grad
    end
    return y, pullback
end
```

**Transformer radii:** Log-window attention must always include a fixed local neighborhood (currently 24 tokens) in addition to the logarithmic schedule. When you increase `sequence_length`, bump `min_radius`/`base_radius` accordingly so each block sees at least those 24 neighbors plus the log-scaled jumps.

### Mathematical Commenting Standard
Before any nontrivial block, state the map with domain/codomain and coordinate formula (e.g., `F: ℝ^{2M} × ℝ^M → ℝ^{2M}`) and reference those equations in comments. Avoid terms like “operator” or “MLP” unless immediately expanded into the explicit affine/nonlinear expression. Show intermediate algebraic steps so reviewers can match each Julia line to the written derivation.

## Testing Guidelines
Use Julia’s `Test` stdlib, keep fixtures in `test/utils.jl`, and name files after the feature under test. Target deterministic coverage for oscillator initialization, gating math, and CUDA execution; favor several focused `@testset`s and run `Pkg.test()` before each PR.

## Commit & Pull Request Guidelines
As of 14 Nov 2025 there is no Git history, so default to Conventional Commits (`feat:`, `fix:`, `docs:`) with ≤72-character subjects. PRs must explain intent, enumerate key changes, link issues, attach evidence (logs/screens) for behavioral shifts, and merge only once CI is green and follow-ups are listed.

## GPU & Configuration Tips
CUDA, LuxCUDA, and KernelAbstractions ship in `Project.toml`; verify the toolkit with `julia --project -e 'using CUDA; CUDA.versioninfo()'`, gate kernels with `CUDA.functional()`, keep secrets in `.env` or shell exports, and document hardware assumptions in PR notes so bugs are reproducible.
