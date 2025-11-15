#!/usr/bin/env -S julia --project

using TOML
using JSON3
using Statistics
using Random
using Dates
using JLD2
using WordTokenizers
using TensorBoardLogger: TBLogger, log_value

using Lux
using LuxCore
using Optimisers
using Zygote
using Functors
using Logging
using LoggingExtras
using ParameterSchedulers

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
    return map(lowercase, WordTokenizers.tokenize(stripped))
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

function tree_l2_norm(tree)
    acc = Ref(0.0f0)
    Functors.fmap(tree) do x
        if x isa AbstractArray
            acc[] += sum(abs2, Float32.(x))
        end
        x
    end
    return sqrt(acc[])
end

function sanitize_log!(data::Dict{String,Any})
    for (k, v) in data
        if v isa AbstractFloat && !isfinite(v)
            data[k] = missing
        end
    end
    return data
end

function json_formatter(io, log)
    payload = Dict{String,Any}(
        "level" => string(log.level),
        "message" => log.message,
    )
    for (k, v) in log.kwargs
        payload[string(k)] = v
    end
    JSON3.write(io, payload)
    write(io, '\n')
end

function log_entry!(logger, data::Dict{String,Any})
    kwargs = Tuple(Symbol(k) => data[k] for k in keys(data))
    LoggingExtras.with_logger(logger) do
        @info "perigee" kwargs...
    end
end

function scale_tree(tree, factor::Real)
    return Functors.fmap(x -> x isa AbstractArray ? factor .* x : x, tree)
end

function build_lr_schedule(cfg::TrainConfig)
    base_lr = cfg.training["learning_rate"]
    sched_cfg = get(cfg.training, "lr_schedule", nothing)
    sched_cfg === nothing && return step -> base_lr
    kind = get(sched_cfg, "type", "cosine")
    if kind == "cosine"
        max_lr = get(sched_cfg, "max_lr", base_lr)
        min_lr = get(sched_cfg, "min_lr", max_lr / 10)
        period = get(sched_cfg, "period", 1000)
        return ParameterSchedulers.CosAnneal(max_lr, min_lr, period)
    elseif kind == "triangle"
        period = get(sched_cfg, "period", 500)
        return ParameterSchedulers.Triangle(base_lr, period)
    else
        return step -> base_lr
    end
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
    base_lr = cfg.training["learning_rate"]
    schedule = build_lr_schedule(cfg)

    checkpoint_dir = cfg.training["checkpoint_dir"]
    mkpath(checkpoint_dir)
    log_path = cfg.training["log_path"]
    mkpath(dirname(log_path))
    json_logger = LoggingExtras.FormatLogger(json_formatter, log_path; append = false)
    logger = LoggingExtras.TeeLogger(json_logger, Logging.current_logger())
    tb_logger = nothing
    if haskey(cfg.training, "tensorboard_dir")
        tb_dir = cfg.training["tensorboard_dir"]
        mkpath(tb_dir)
        tb_logger = TBLogger(tb_dir; mkdir=false)
    end
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
            global_step += 1
            lr_current = schedule(global_step)
            scale_factor = lr_current / base_lr
            scaled_grads = scale_tree(grads, scale_factor)
            opt_state, ps = Optimisers.update(opt_state, ps, scaled_grads)
            grad_norm = tree_l2_norm(scaled_grads)

            if global_step % 25 == 0
                if !(isfinite(batch_loss) && isfinite(grad_norm))
                    continue
                end
                logged_grad = isfinite(grad_norm) ? grad_norm : missing
                logged_loss = isfinite(batch_loss) ? float(batch_loss) : missing
                log_data = Dict(
                    "timestamp" => string(now(UTC)),
                    "epoch" => epoch,
                    "step" => global_step,
                    "loss" => logged_loss,
                    "grad_norm" => logged_grad,
                    "learning_rate" => lr_current,
                )
                log_entry!(logger, sanitize_log!(log_data))
                if tb_logger !== nothing
                    log_value(tb_logger, "train/loss", float(batch_loss), global_step)
                    log_value(tb_logger, "train/grad_norm", grad_norm, global_step)
                    log_value(tb_logger, "train/lr", lr_current, global_step)
                end
                println("[epoch $(epoch)] step $(global_step) loss = $(round(batch_loss, digits=4)) lr=$(lr_current)")
            end
        end

        val_loss = evaluate(model, tokenizer, val_sequences, cfg, ps, st_template, use_gpu, data_rng)
        val_log = Dict(
            "timestamp" => string(now(UTC)),
            "epoch" => epoch,
            "phase" => "validation",
            "loss" => isfinite(val_loss) ? float(val_loss) : missing,
        )
        log_entry!(logger, sanitize_log!(val_log))
        if tb_logger !== nothing && isfinite(val_loss)
            log_value(tb_logger, "val/loss", float(val_loss), epoch)
        end
        println("Epoch $(epoch) validation loss = $(round(val_loss, digits=4))")
    end

    checkpoint_path = joinpath(checkpoint_dir, cfg.training["checkpoint_name"])
    params = move_tree(ps, Array)
    states = move_tree(st_template, Array)
    jldsave(
        checkpoint_path;
        config_path = config_path,
        params = params,
        states = states,
        tokenizer = tokenizer,
    )
    tb_logger !== nothing && close(tb_logger)
    tb_logger !== nothing && close(tb_logger)
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
