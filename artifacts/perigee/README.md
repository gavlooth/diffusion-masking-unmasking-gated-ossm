# Perigee Diffusion Pipeline

This folder records the artifacts produced by the Perigee diffusion language model
pipeline introduced in `scripts/train_perigee.jl` and `scripts/generate_perigee.jl`.

## Training summary
- **Dataset**: `data_sets/unified.train.raw` (balanced 50/50 mix of PG-19 and OpenWebText2 samples, each normalized to single-line documents).
- **Vocabulary**: word-level tokens derived from the training corpus and stored in
  `perigee_vocab.json` (50,000 entries plus the special markers `[PAD]`, `[MASK]`,
  `[UNK]`, `[BOS]`, `[EOS]`).
- **Model hyperparameters**: see `configs/perigee_train.toml` (`num_layers=32`,
  `sequence_length=2048`, `model_dim=128`, `oscillator_count=16`, `num_heads=8`,
  `mamba_repeat=2`, `radius_factor=4.0`, base radius = 24).
- **Optimizer**: AdamW (lr = 1.5e-4) for 8 epochs, batch size 1.
- **Hardware note**: the local Tesla M40 (compute capability 5.2) is supported up
  through CUDA 12.8. `LocalPreferences.toml` pins the runtime to 12.8 so
  `train_perigee.jl` initializes the GPU successfully even though the host driver
  advertises CUDA 13.
- **Command**:
  ```bash
  ./scripts/train_perigee.jl configs/perigee_train.toml
  ```
- **Outputs**:
  - Checkpoint: `checkpoints/perigee_epoch1.jls` (contains parameters, states, tokenizer).
  - Log: `logs/training_log.jsonl` (JSONL records of every 25th step + validation loss).

### Data preparation
All corpora live under `data_sets/`. Recreate them with:

```bash
# 1) PG-19 shards → compact .raw (DuckDB)
source .venv-data/bin/activate
python scripts/build_pg19_sample.py \
  --root /mnt/exfat-data/Downloads/pg19/data \
  --train-limit 2000 --valid-limit 200 --test-limit 200

# 2) Stream OpenWebText2 tarball → .raw (HTTP + zstd)
python scripts/build_openwebtext2_sample.py \
  --train-count 2000 --val-count 200 --test-count 200

# 3) Interleave both corpora → unified.{train,valid,test}.raw
python scripts/build_unified_corpus.py
```

`build_openwebtext2_sample.py` streams the Hugging Face tarball on-the-fly, so it only downloads the bytes required for the requested counts.

### Quick hyperparameter overrides
Both `train_perigee.jl` and `generate_perigee.jl` read optional environment variables before constructing the model:

- `PERIGEE_NUM_LAYERS`
- `PERIGEE_NUM_HEADS`
- `PERIGEE_OSCILLATOR_COUNT`

Example:

```bash
PERIGEE_NUM_LAYERS=24 PERIGEE_NUM_HEADS=12 PERIGEE_OSCILLATOR_COUNT=32 \
  JULIA_DEBUG= ./scripts/train_perigee.jl configs/perigee_train.toml
```

If unset, the scripts fall back to the values stored in the config TOML.

### Resuming long trainings
Set `PERIGEE_RESUME=1` to restart `train_perigee.jl` from the latest checkpoint in `checkpoint_dir`. After every epoch the script now saves parameters, optimizer state, RNG state, and the tokenizer, so interrupted multi-hour runs can restart without losing progress:

```bash
PERIGEE_RESUME=1 JULIA_DEBUG= ./scripts/train_perigee.jl configs/perigee_train.toml
```

If the checkpoint already completed all configured epochs, the script exits immediately.

## Smoke Test
- Config: `configs/perigee_smoke.toml` (32-layer stack, 2,048-token windows, 32 sampled sequences for a quick GPU sanity pass).
- Command: `./scripts/train_perigee.jl configs/perigee_smoke.toml`
- Output: `checkpoints/perigee_smoke_epoch1.jls`, log at `logs/smoke_log.jsonl`.

## Generation
- **Command**:
  ```bash
  ./scripts/generate_perigee.jl \
      configs/perigee_train.toml \
      artifacts/perigee/checkpoints/perigee_epoch1.jls \
      "Gallia prepares for battle near the capital" \
      6
  ```
- Like training, generation attempts to use CUDA first; on this machine it also
  falls back to CPU with a warning about the unsupported M40 card when a compatible
  runtime is unavailable. With 2,048-token windows, GPU acceleration is mandatory
  for reasonable throughput.
- Example output from the full config command above (after 6 diffusion refinement steps):
  ```
  Gallia prepares for battle near the capital the the [EOS] [EOS] ...
  ```
- Smoke checkpoint output (`configs/perigee_smoke.toml`, prompt "Gallia prepares", 4 steps):
  ```
  Gallia prepares [PAD] [PAD] [MASK] [PAD] ...
  ```
  The sequence is intentionally short because the model only saw a single epoch.

### Diffusion viewing modes
`./scripts/generate_perigee.jl` now accepts:

```
./scripts/generate_perigee.jl <config> <checkpoint> "<prompt>" <steps> [matrix|live] [batch] [frames.txt] [gif.gif]
```

- `matrix`: prints side-by-side columns each step; masked slots emit `[???]` and changed tokens glow green.
- `live`: rewrites the same terminal rows in-place for the "Matrix rain" effect.
- `frames.txt`: optional plaintext log (one block per step).
- `gif.gif`: optional animated GIF assembled via Pillow inside `.venv`.

Batch inference is now the 6th positional argument (default `1`).

## Reproducibility checklist
1. `julia --project -e 'using Pkg; Pkg.instantiate()'`
2. Ensure `configs/perigee_train.toml` points at your dataset (2048-token windows, 32 layers); adjust `max_sequences`, epochs, etc., if desired.
3. Run the training command. Watch `artifacts/perigee/logs/training_log.jsonl` for
   streaming losses; reruns overwrite the log.
4. Use the generation script with any prompt up to 128 whitespace tokens.
5. Checkpoints are standard `Serialization.serialize` payloads containing
   `params`, `states`, and the `PrimeTokenizer`, so you can `deserialize` them in
   a REPL and resume training or sampling directly.
