#!/usr/bin/env -S julia --project

using TOML
using Lux
using LuxCore
using Functors
using Random
using WordTokenizers
using JLD2

import CUDA

using ossmv2: PrimeTokenizer, build_perigee_model, prime_encode, prime_scale

function load_checkpoint(path)
    return JLD2.load(path)
end

function move_tree(tree, mover)
    return Functors.fmap(x -> x isa AbstractArray ? mover(x) : x, tree)
end

function nearest_token(tokenizer::PrimeTokenizer, value::Real)
    if !isfinite(value)
        return "[nan]"
    end
    primes = tokenizer.codec.table
    differences = abs.(primes .- round(Int, value))
    idx = argmin(differences)
    prime = primes[idx]
    return tokenizer.prime_to_token[prime]
end

function tokenize_prompt(prompt::String)
    stripped = strip(prompt)
    isempty(stripped) && return String[]
    return map(lowercase, WordTokenizers.tokenize(stripped))
end

simple_bool(str) = lowercase(str) in ("1", "true", "yes", "y", "matrix")

function format_cell(token::AbstractString, prime_val::Int, masked::Bool)
    printable = all(isprint, token)
    base = printable ? token : "[p=$(prime_val)]"
    trimmed = length(base) > 12 ? first(base, 11) * "…" : base
    cell = rpad(trimmed, 12)
    return masked ? "*" * cell * "*" : " " * cell * " "
end

function render_matrix(step::Int, tokens, encoded_vals, tokenizer)
    masked_positions = Set(findall(==(tokenizer.mask_token), tokens))
    println("┌ Step $(step) diffusion snapshot" |> String)
    row = join(
        (format_cell(tokens[i], encoded_vals[i], i in masked_positions) for i in eachindex(tokens)),
        "│",
    )
    println(row)
    println("└" * "─" ^ max(length(row) - 1, 1))
end

function load_model(cfg)
    return build_perigee_model(
        cfg["model"]["num_layers"];
        input_dim = cfg["training"]["sequence_length"],
        model_dim = cfg["model"]["model_dim"],
        oscillator_count = cfg["model"]["oscillator_count"],
        num_heads = cfg["model"]["num_heads"],
        vocab_dim = cfg["training"]["sequence_length"],
        mamba_repeat = cfg["model"]["mamba_repeat"],
        radius_factor = cfg["model"]["radius_factor"],
        min_radius = cfg["model"]["min_radius"],
        max_radius = cfg["model"]["max_radius"],
    )
end

function _splitext(name::AbstractString)
    idx = findlast(==('.'), name)
    if idx === nothing
        return (String(name), "")
    end
    stem = idx == 1 ? "" : String(name[1:(idx - 1)])
    return (stem, String(name[idx:end]))
end

function resolve_checkpoint_path(cfg, override::Union{Nothing,String})
    if override !== nothing
        return (path = override, tried = [override])
    end
    training_cfg = get(cfg, "training", Dict{String,Any}())
    dir = get(training_cfg, "checkpoint_dir", "artifacts/perigee/checkpoints")
    name = get(training_cfg, "checkpoint_name", "perigee_epoch1.jls")
    candidate = joinpath(dir, name)
    tried = String[candidate]
    isfile(candidate) && return (path = candidate, tried = tried)
    base, ext = _splitext(name)
    alt_exts = ext == ".jls" ? [".jld2"] :
               ext == ".jld2" ? [".jls"] :
               [".jls", ".jld2"]
    for alt_ext in alt_exts
        alt_candidate = joinpath(dir, base * alt_ext)
        alt_candidate in tried && continue
        push!(tried, alt_candidate)
        isfile(alt_candidate) && return (path = alt_candidate, tried = tried)
    end
    return (path = candidate, tried = tried)
end

function generate()
    config_path = get(ARGS, 1, "configs/perigee_train.toml")
    checkpoint_override = length(ARGS) >= 2 ? ARGS[2] : nothing
    prompt = get(ARGS, 3, "Gallia's forces were preparing for the next offensive.")
    steps = parse(Int, get(ARGS, 4, "6"))
    show_matrix = length(ARGS) >= 5 && simple_bool(ARGS[5])

    cfg = TOML.parsefile(config_path)
    checkpoint_info = resolve_checkpoint_path(cfg, checkpoint_override)
    checkpoint_path = checkpoint_info.path
    if !isfile(checkpoint_path)
        error(
            "No checkpoint found. Checked the following paths: " *
            join(checkpoint_info.tried, ", "),
        )
    end
    sequence_length = cfg["training"]["sequence_length"]
    model = load_model(cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    tokenizer = checkpoint["tokenizer"]
    norm_factor = prime_scale(tokenizer)
    inv_norm = 1f0 / norm_factor
    min_prime = Float32(tokenizer.codec.table[1])
    max_prime = Float32(tokenizer.codec.table[end])

    gpu_requested = get(cfg["training"], "use_gpu", true)
    use_gpu = false
    if gpu_requested
        try
            if CUDA.functional()
                dev = CUDA.device()
                cap = CUDA.capability(dev)
                runtime = try
                    string(CUDA.runtime_version())
                catch
                    "unknown"
                end
                CUDA.allowscalar(false)
                println(
                    "Using CUDA device $(CUDA.name(dev)) (sm_$(cap.major)$(cap.minor)) with runtime $(runtime)",
                )
                use_gpu = true
            else
                @warn "CUDA.functional() returned false; using CPU for generation"
            end
        catch err
            @warn "CUDA.functional() failed during generation; using CPU" error = err
        end
    end
    mover = use_gpu ? CUDA.cu : identity

    ps = move_tree(checkpoint["params"], mover)
    st = move_tree(checkpoint["states"], mover)

    prompt_tokens = tokenize_prompt(prompt)
    tokens = fill("[MASK]", sequence_length)
    limit = min(length(prompt_tokens), sequence_length)
    tokens[1:limit] = prompt_tokens[1:limit]
    mask_indices = findall(==("[MASK]"), tokens)

    function encode_tokens(tok_vec)
        encoded = Float32.(prime_encode(tokenizer, tok_vec)) .* inv_norm
        return reshape(encoded, sequence_length, 1)
    end

    seq = encode_tokens(tokens)
    seq = use_gpu ? CUDA.cu(seq) : seq

    rng = Random.default_rng()
    for step in 1:steps
        output, st = model(seq, ps, st)
        preds = Array(output.logits)
        for idx in mask_indices
            raw = clamp(preds[idx, end] * norm_factor, min_prime, max_prime)
            tokens[idx] = nearest_token(tokenizer, raw)
        end
        seq = encode_tokens(tokens)
        seq = use_gpu ? CUDA.cu(seq) : seq
        if show_matrix
            encoded_vals = vec(round.(Int, Array(seq .* norm_factor)))
            render_matrix(step, tokens, encoded_vals, tokenizer)
        end
    end

    println("Prompt: \"$(prompt)\"")
    println("Generated tokens:\n" * join(tokens, " "))
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate()
end
