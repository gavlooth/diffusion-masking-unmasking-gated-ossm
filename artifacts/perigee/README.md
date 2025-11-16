# Perigee Diffusion Pipeline

This folder records the artifacts produced by the Perigee diffusion language model
pipeline introduced in `scripts/train_perigee.jl` and `scripts/generate_perigee.jl`.

## Training summary
- **Dataset**: `data_sets/wiki.train.raw` (first 4,096 windowed sequences of 128 tokens).
- **Vocabulary**: word-level tokens derived from the training corpus and stored in
  `perigee_vocab.json` (50,000 entries plus the special markers `[PAD]`, `[MASK]`,
  `[UNK]`, `[BOS]`, `[EOS]`).
- **Model hyperparameters**: see `configs/perigee_train.toml` (`num_layers=4`,
  `model_dim=128`, `oscillator_count=16`, `num_heads=8`, `mamba_repeat=2`,
  `radius_factor=4.0`).
- **Optimizer**: AdamW (lr = 2e-4) for 1 epoch, batch size 8.
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

## Smoke Test
- Config: `configs/perigee_smoke.toml` (layers halved, 64-token windows, 64 sequences).
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
  falls back to CPU with a warning about the unsupported M40 card.
- Example output from the full config command above (after 6 diffusion refinement steps):
  ```
  Gallia prepares for battle near the capital the the [EOS] [EOS] ...
  ```
- Smoke checkpoint output (`configs/perigee_smoke.toml`, prompt "Gallia prepares", 4 steps):
  ```
  Gallia prepares [PAD] [PAD] [MASK] [PAD] ...
  ```
  The sequence is intentionally short because the model only saw a single epoch.

## Reproducibility checklist
1. `julia --project -e 'using Pkg; Pkg.instantiate()'`
2. Ensure `configs/perigee_train.toml` points at your dataset, adjust
   `max_sequences`, `sequence_length`, etc., if desired.
3. Run the training command. Watch `artifacts/perigee/logs/training_log.jsonl` for
   streaming losses; reruns overwrite the log.
4. Use the generation script with any prompt up to 128 whitespace tokens.
5. Checkpoints are standard `Serialization.serialize` payloads containing
   `params`, `states`, and the `PrimeTokenizer`, so you can `deserialize` them in
   a REPL and resume training or sampling directly.
