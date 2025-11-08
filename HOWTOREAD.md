# HOW TO READ THIS TUTORIAL REPO

This repository is written as a worked example for learning three things at
the same time:

1. The Julia language features that appear in practical machine-learning code.
2. The Lux.jl neural-network ecosystem (layers, parameter/state handling,
   training loops).
3. The architecture of a discrete diffusion model with an oscillatory
   state-space backbone.

The sections below describe how to approach the codebase, what each file is
trying to teach, and where to look for supporting references.

## Audience and Assumed Background

- Comfortable with another high-level language (Python, Swift, etc.).
- Basic linear algebra notation (matrices, vectors, block matrices).
- Interest in generative modelling, but no requirement to know diffusion or
  state-space models in advance.

If you are entirely new to Julia, skim
[`Julia Manual → Getting Started`](https://docs.julialang.org/en/v1/manual/getting-started/)
and the first two chapters of
[`Julia Manual → Functions`](https://docs.julialang.org/en/v1/manual/functions/)
before diving in.

## Reading Schema (Recommended Sequence)

The tutorial is layered so you can iterate between code and mathematics. Each
stage below lists the intent, the primary files, and optional practice ideas.

| Stage | Goal | Primary files / sections | Optional practice |
| ----- | ---- | ------------------------ | ----------------- |
| 0. Orientation | Skim the high-level story and exported API list. | `src/dlinoss.jl` (intro + exports, lines 1–120). | Run `using dlinoss` in the REPL and tab-complete `dlinoss.` to see available bindings. |
| 1. Julia Essentials | Understand tuples, `NamedTuple`s, multiple dispatch, and broadcasting that keep recurring. | Comments in `src/dlinoss.jl` and `src/ossm/unit.jl:52-167`, plus the docstring examples. | Re-write the small helper `statedim` in the REPL and experiment with `@code_warntype`. |
| 2. Codec & Tokeniser | Learn how text becomes digit matrices. | `src/diffusion/codec.jl`, `src/diffusion/tokenizer.jl`, `src/diffusion/pipeline.jl`. | Use `test/runtests.jl` tokenizer tests as an interactive checklist; try new strings in the REPL. |
| 3. Embeddings & Projection | Follow how digit matrices become model-ready tensors. | `src/diffusion/embeddings.jl`, `src/diffusion/output_head.jl`, and the `DiffusionOSSMBackbone` docstring (lines 43–120 of `src/dlinoss.jl`). | Modify `d_sub` and `d_model` in the REPL and inspect parameter sizes with `Lux.initialparameters`. |
| 4. Time/Masking Mechanics | Study positional encoding and forward/backward masking. | `src/diffusion/timepos.jl`, `src/diffusion/masking.jl`. | Run the masking test block (line ~40 of `test/runtests.jl`) with different seeds to see stochastic behaviour. |
| 5. OSSM Backbone | Understand the oscillatory state-space unit and sequence wrapper. | `src/ossm/unit.jl`, `src/ossm/gating.jl`, `src/ossm/sequence.jl`. | Derive the 2×2 rotation block manually, then compare with `mul_Ax`. |
| 6. Putting It Together | Traverse the end-to-end diffusion step helper. | `src/dlinoss.jl` (forward helpers), `src/diffusion/pipeline.jl`. | Recreate `forward_diffusion_step` in the REPL and inspect outputs; trace shapes with `size`. |

Feel free to loop back—later stages assume comfort with earlier ones, but the
files are intentionally cross-referenced.

## Key Definitions and Where They Appear

| Concept | Definition | First-class example |
| ------- | ---------- | ------------------- |
| `NamedTuple` | Immutable, dictionary-like container accessed via dots. Julia idiom for bundling parameters. | `src/ossm/unit.jl:52-96` shows layer parameters and nested tuples. |
| Multiple dispatch | Method selection based on *all* argument types. Powers Lux’s `layer(u, ps, st)` signature. | `src/ossm/unit.jl:143-168` and `src/dlinoss.jl:200-232`. |
| Broadcasting (`.` ops) | Element-wise application with automatic shape alignment. | `src/ossm/unit.jl:118-138` (`σ.(...)`, `.+`). |
| Lux Layer lifecycle | Three hooks: `initialparameters`, `initialstates`, call overload. | Documented in `src/dlinoss.jl` (backbone) and implemented for each sublayer. |
| Prime codec | Maps token ids to base-`b` digit tuples and back. | `src/diffusion/codec.jl`. |
| Diffusion masking | Stochastic map that replaces digits with mask symbol `b`. | `src/diffusion/masking.jl`. |
| Time-position encoding | Adds deterministic position and learned time conditioning. | `src/diffusion/timepos.jl`. |
| Gated Oscillatory SSM | State-space module with damped rotations and input-dependent gates. | `src/ossm/unit.jl`, `src/ossm/gating.jl`. |

## External References

| Topic | Reference |
| ----- | --------- |
| Julia language | [Julia Manual](https://docs.julialang.org/en/v1/) |
| Multiple dispatch primer | [Julia Manual → Methods](https://docs.julialang.org/en/v1/manual/methods/) |
| Broadcast fusion | [Julia Manual → Arrays](https://docs.julialang.org/en/v1/manual/arrays/#man-array-broadcast) |
| Lux.jl overview | [Lux.jl documentation](https://lux.csail.mit.edu/stable/) |
| Diffusion models | [Lil’Log: Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |
| State-space models | [HiPPO & S4 survey](https://arxiv.org/abs/2206.11893) (background on continuous-time SSMs) |

Use these references as deeper dives after you encounter the concept locally
in the codebase.

## Suggested Study Loop

1. **Read** the docstring or module narrative in the order suggested above.
2. **Experiment** in the Julia REPL (`include("src/dlinoss.jl")`) and inspect
   returned shapes/types.
3. **Test** your understanding by running focused pieces of
   `test/runtests.jl` or writing small REPL snippets (e.g. custom tokens).
4. **Reflect** by mapping code back to the mathematical expressions in the
   docstrings—each formula mirrors the implementation immediately below it.

Sticking to this loop reinforces both Julia and the architecture while keeping
Lux-specific abstractions grounded in concrete examples.

