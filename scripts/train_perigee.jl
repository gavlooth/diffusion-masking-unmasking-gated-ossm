#!/usr/bin/env -S julia --project

using TOML
using JSON3
using Statistics
using Random
using Dates
using BSON: @save

using Lux
using LuxCore
using Optimisers
using Zygote
using Functors
using Logging

import CUDA

using ossmv2: PrimeTokenizer, perigee_prepare_batch, perigee_diffusion_loss, build_perigee_model, prime_samples

const SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "[BOS]", "[EOS]"]

struct TrainConfig
    model::Dict{String, Any}
    training::Dict{String, Any}
end

function load_config(path::String)
    cfg = TOML.parsefile(path)
    return TrainConfig(cfg["model"], cfg["training"])
end

function tokenize_line(line::AbstractString)
    stripped = strip(line)
    isempty(stripped) && return String[]
    return split(stripped)
end

function build_vocab(paths::Vector{String}; max_tokens::Int = 50000)
    counts = Dict{String, Int}()
    for path in paths
        open(path, "r") do io
            for line in eachline(io)
                for token in tokenize_line(line)
                    counts[token] = get(counts, token, 0) + 1
                end
            end
        end
    end
    sorted_tokens = sort(collect(counts); by = x -> (-x[2], x[1]))
    keep = max(0, max_tokens - length(SPECIAL_TOKENS))
    vocab = copy(SPECIAL_TOKENS)
    for (idx, (token, _)) in enumerate(sorted_tokens)
        idx > keep && break
        token in SPECIAL_TOKENS && continue
        push!(vocab, token)
    end
    return vocab
end

function load_or_build_vocab(cfg::TrainConfig)
    vocab_path = cfg.training["vocab_path"]
    if isfile(vocab_path)
        raw = JSON3.read(read(vocab_path, String))
        return collect(String, raw)
    end
    mkpath(dirname(vocab_path))
    vocab = build_vocab([cfg.training["train_path"]])
    open(vocab_path, "w") do io
        JSON3.write(io, vocab)
    end
    return vocab
end

function drop_front!(buffer::Vector{String}, count::Int)
    if count >= length(buffer)
        empty!(buffer)
    else
        deleteat!(buffer, 1:count)
    end
    return buffer
end

function load_sequences(path::String; seq_len::Int, stride::Int, max_sequences::Union{Int, Nothing})
    sequences = Vector{Vector{String}}()
    buffer = String[]
    open(path, "r") do io
        for line in eachline(io)
            tokens = tokenize_line(line)
            isempty(tokens) && continue
            append!(buffer, tokens)
            while length(buffer) >= seq_len
                push!(sequences, buffer[1:seq_len])
                drop_front!(buffer, stride)
                if !isnothing(max_sequences) && length(sequences) >= max_sequences
                    return sequences
                end
            end
        end
    end
    return sequences
end

function batches(data::Vector, batch_size::Int)
    return [data[i:min(i + batch_size - 1, length(data))] for i in 1:batch_size:length(data)]
end

function clone_state(st)
    return Functors.fmap(x -> x isa AbstractArray ? copy(x) : x, st)
end

function prepare_batch(tokenizer::PrimeTokenizer, batch_tokens, cfg::TrainConfig, rng)
    return perigee_prepare_batch(
        tokenizer,
        batch_tokens;
        mask_fraction = cfg.training["mask_fraction"],
        unmask_fraction = cfg.training["unmask_fraction"],
        rng = rng,
    )
end

function to_device(array::AbstractArray, use_gpu::Bool)
    use_gpu || return array
    return CUDA.cu(array)
end

function move_tree(tree, mover)
    return Functors.fmap(x -> x isa AbstractArray ? mover(x) : x, tree)
end

function log_entry!(io, data)
    JSON3.write(io, data)
    write(io, '\n')
    flush(io)
end

function build_optimizer(cfg::TrainConfig)
    lr = cfg.training["learning_rate"]
    opt_name = get(cfg.training, "optimizer", "adamw")
    weight_decay = get(cfg.training, "weight_decay", 0.0)
    if opt_name == "adam"
        return Optimisers.Adam(lr)
    else
        return Optimisers.AdamW(lr, (0.9, 0.999), weight_decay)
    end
end

function train()
    config_path = get(ARGS, 1, "configs/perigee_train.toml")
    cfg = load_config(config_path)
    vocab = load_or_build_vocab(cfg)
    sequence_length = cfg.training["sequence_length"]

    println("Loaded vocab of $(length(vocab)) tokens")
    train_raw = load_sequences(
        cfg.training["train_path"];
        seq_len = sequence_length,
        stride = cfg.training["stride"],
        max_sequences = cfg.training["max_sequences"],
    )
    val_raw = load_sequences(
        cfg.training["val_path"];
        seq_len = sequence_length,
        stride = cfg.training["stride"],
        max_sequences = min(512, cfg.training["max_sequences"]),
    )
    tokenizer = PrimeTokenizer(vocab)
    train_sequences = prime_samples(tokenizer, train_raw)
    val_sequences = prime_samples(tokenizer, val_raw)
    println(
        "Loaded $(length(train_sequences)) training samples and $(length(val_sequences)) validation samples",
    )
    seed = cfg.training["seed"]
    data_rng = Random.MersenneTwister(seed)
    model_rng = Random.default_rng()
    Random.seed!(model_rng, seed)

    use_gpu = get(cfg.training, "use_gpu", true)
    if use_gpu
        use_gpu = try
            if CUDA.functional()
                dev = CUDA.device()
                capability = CUDA.capability(dev)
                major = capability.major
                minor = capability.minor
                if major < 6
                    @warn "Compute capability $(major).$(minor) is unsupported on the current CUDA stack; falling back to CPU"
                    false
                else
                    true
                end
            else
                false
            end
        catch err
            @warn "CUDA.functional() failed, falling back to CPU" error = err
            false
        end
    end
    if use_gpu
        CUDA.allowscalar(false)
    else
        @warn "GPU requested but unavailable; training will run on CPU"
    end
    mover = use_gpu ? CUDA.cu : identity

    model = build_perigee_model(
        cfg.model["num_layers"];
        input_dim = sequence_length,
        model_dim = cfg.model["model_dim"],
        oscillator_count = cfg.model["oscillator_count"],
        num_heads = cfg.model["num_heads"],
        vocab_dim = sequence_length,
        mamba_repeat = cfg.model["mamba_repeat"],
        radius_factor = cfg.model["radius_factor"],
        min_radius = cfg.model["min_radius"],
        max_radius = cfg.model["max_radius"],
    )

    ps, st = Lux.setup(model_rng, model)
    ps = move_tree(ps, mover)
    st = move_tree(st, mover)
    st_template = clone_state(st)

    opt = build_optimizer(cfg)
    opt_state = Optimisers.setup(opt, ps)

    checkpoint_dir = cfg.training["checkpoint_dir"]
    mkpath(checkpoint_dir)
    log_path = cfg.training["log_path"]
    mkpath(dirname(log_path))

    open(log_path, "w") do log_io
        global_step = 0
        for epoch in 1:cfg.training["epochs"]
            shuffled = copy(train_sequences)
            Random.shuffle!(data_rng, shuffled)
            st = clone_state(st_template)
            for batch_tokens in batches(shuffled, cfg.training["batch_size"])
                batch = prepare_batch(tokenizer, batch_tokens, cfg, data_rng)
                batch_gpu = (
                    observed = to_device(batch.observed, use_gpu),
                    targets = to_device(batch.targets, use_gpu),
                    mask_matrix = to_device(batch.mask_matrix, use_gpu),
                )

                loss_fn = function (param)
                    l, _ = perigee_diffusion_loss(model, batch_gpu, param, st)
                    return l
                end
                loss, back = Zygote.pullback(loss_fn, ps)
                grads = back(1f0)[1]
                batch_loss, st = perigee_diffusion_loss(model, batch_gpu, ps, st)
                opt_state, ps = Optimisers.update(opt_state, ps, grads)

                global_step += 1
                if global_step % 25 == 0
                    log_entry!(log_io, Dict(
                        "timestamp" => string(now(UTC)),
                        "epoch" => epoch,
                        "step" => global_step,
                        "loss" => float(batch_loss),
                    ))
                    println("[epoch $(epoch)] step $(global_step) loss = $(round(batch_loss, digits=4))")
                end
            end

            val_loss = evaluate(model, tokenizer, val_sequences, cfg, ps, st_template, use_gpu, data_rng)
            log_entry!(log_io, Dict(
                "timestamp" => string(now(UTC)),
                "epoch" => epoch,
                "phase" => "validation",
                "loss" => float(val_loss),
            ))
            println("Epoch $(epoch) validation loss = $(round(val_loss, digits=4))")
        end
    end

    checkpoint_path = joinpath(checkpoint_dir, cfg.training["checkpoint_name"])
    params = move_tree(ps, Array)
    states = move_tree(st_template, Array)
    @save checkpoint_path config_path params states tokenizer
    println("Checkpoint saved to $(checkpoint_path)")
end

function evaluate(model, tokenizer, sequences, cfg, ps, st_template, use_gpu, rng)
    isempty(sequences) && return 0.0f0
    val_batches = batches(sequences, cfg.training["batch_size"])
    limit = min(length(val_batches), 8)
    total = 0.0f0
    st_eval = clone_state(st_template)
    for (idx, batch_tokens) in enumerate(val_batches)
        idx > limit && break
        batch = prepare_batch(tokenizer, batch_tokens, cfg, rng)
        batch_gpu = (
            observed = to_device(batch.observed, use_gpu),
            targets = to_device(batch.targets, use_gpu),
            mask_matrix = to_device(batch.mask_matrix, use_gpu),
        )
        loss, st_eval = perigee_diffusion_loss(model, batch_gpu, ps, st_eval)
        total += loss
    end
    return total / limit
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
