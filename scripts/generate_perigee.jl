#!/usr/bin/env -S julia --project

using TOML
using Serialization
using Lux
using LuxCore
using Functors
using Random

import CUDA

using ossmv2: PrimeTokenizer, build_perigee_model, prime_encode

function load_checkpoint(path)
    return Serialization.deserialize(path)
end

function move_tree(tree, mover)
    return Functors.fmap(x -> x isa AbstractArray ? mover(x) : x, tree)
end

function nearest_token(tokenizer::PrimeTokenizer, value::Real)
    primes = tokenizer.codec.table
    differences = abs.(primes .- round(Int, value))
    idx = argmin(differences)
    prime = primes[idx]
    return tokenizer.prime_to_token[prime]
end

function tokenize_prompt(prompt::String)
    stripped = strip(prompt)
    isempty(stripped) && return String[]
    return split(stripped)
end

function load_model(cfg)
    return build_perigee_model(
        cfg["model"]["num_layers"];
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

function generate()
    config_path = get(ARGS, 1, "configs/perigee_train.toml")
    checkpoint_path = get(ARGS, 2, "artifacts/perigee/checkpoints/perigee_epoch1.jls")
    prompt = get(ARGS, 3, "Gallia's forces were preparing for the next offensive.")
    steps = parse(Int, get(ARGS, 4, "6"))

    cfg = TOML.parsefile(config_path)
    sequence_length = cfg["training"]["sequence_length"]
    model = load_model(cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    tokenizer = checkpoint[:tokenizer]

    use_gpu = false
    try
        if CUDA.functional()
            cap = CUDA.capability(CUDA.device())
            if cap.major >= 6
                CUDA.allowscalar(false)
                use_gpu = true
            else
                @warn "Compute capability $(cap.major).$(cap.minor) is unsupported on this CUDA stack; using CPU for generation"
            end
        end
    catch err
        @warn "CUDA.functional() failed during generation; using CPU" error = err
    end
    mover = use_gpu ? CUDA.cu : identity

    ps = move_tree(checkpoint[:params], mover)
    st = move_tree(checkpoint[:states], mover)

    prompt_tokens = tokenize_prompt(prompt)
    tokens = fill("[MASK]", sequence_length)
    limit = min(length(prompt_tokens), sequence_length)
    tokens[1:limit] = prompt_tokens[1:limit]
    mask_indices = findall(==("[MASK]"), tokens)

    function encode_tokens(tok_vec)
        encoded = Float32.(prime_encode(tokenizer, tok_vec))
        return reshape(encoded, sequence_length, 1)
    end

    seq = encode_tokens(tokens)
    seq = use_gpu ? CUDA.asarray(seq) : seq

    rng = Random.default_rng()
    for step in 1:steps
        output, st = model(seq, ps, st)
        preds = Array(output.logits)
        for idx in mask_indices
            tokens[idx] = nearest_token(tokenizer, preds[idx, end])
        end
        seq = encode_tokens(tokens)
        seq = use_gpu ? CUDA.asarray(seq) : seq
    end

    println("Prompt: \"$(prompt)\"")
    println("Generated tokens:\n" * join(tokens, " "))
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate()
end
