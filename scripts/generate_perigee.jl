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

simple_bool(str) = lowercase(str) in ("1", "true", "yes", "y", "matrix", "live")

const DIM = "\e[2m"
const RESET = "\e[0m"

function obfuscate_token(token::AbstractString)
    stripped = replace(token, r"[^A-Za-z0-9]" => "")
    isempty(stripped) && return "[???]"
    if length(stripped) == 1
        return DIM * "[" * string(first(stripped)) * "?]" * RESET
    end
    return DIM * "[" * string(first(stripped)) * "?" * string(last(stripped)) * "]" * RESET
end

function stylize_token(token::AbstractString, prev::AbstractString, masked::Bool, obfuscate::Bool)
    base = if masked
        obfuscate ? obfuscate_token(token) : "[???]"
    else
        token
    end
    prev_base = if masked
        obfuscate ? obfuscate_token(prev) : "[???]"
    else
        prev
    end
    base == prev_base && return base
    return string("\e[32m", base, "\e[0m")
end

function render_inline(step::Int, columns, prev_columns, mask_indices, obfuscate::Bool)
    println("Step $(step) diffusion text view:")
    for (col_idx, col) in enumerate(columns)
        tokens_view = map(1:length(col)) do idx
            masked = idx in mask_indices
            prev_tok = prev_columns[col_idx][idx]
            stylize_token(col[idx], prev_tok, masked, obfuscate)
        end
        println("[column $(col_idx)] " * join(tokens_view, " "))
    end
end

function render_live(columns, prev_columns, mask_indices, obfuscate::Bool)
    lines = length(columns)
    print("\e[" * string(lines) * "A")
    for (col_idx, col) in enumerate(columns)
        tokens_view = map(1:length(col)) do idx
            masked = idx in mask_indices
            prev_tok = prev_columns[col_idx][idx]
            stylize_token(col[idx], prev_tok, masked, obfuscate)
        end
        print("\r\e[2K[column $(col_idx)] " * join(tokens_view, " ") * "\n")
    end
end

function capture_frame(columns)
    return ["[column $(col_idx)] " * join(col, " ") for (col_idx, col) in enumerate(columns)]
end

function record_frame(io::IO, step::Int, frame_lines)
    println(io, "Step $(step)")
    for line in frame_lines
        println(io, line)
    end
    println(io)
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
    render_flag = length(ARGS) >= 5 ? lowercase(ARGS[5]) : ""
    show_matrix = simple_bool(render_flag)
    live_mode = render_flag == "live"
    batch_size = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1
    record_path = length(ARGS) >= 7 ? strip(ARGS[7]) : ""
    gif_path = length(ARGS) >= 8 ? strip(ARGS[8]) : ""
    record_io = record_path == "" ? nothing : open(record_path, "w")
    record_gif = gif_path != ""
    gif_frames = Vector{Vector{String}}()
    obfuscate = true

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
    limit = min(length(prompt_tokens), sequence_length)
    token_columns = [begin
        col = fill("[MASK]", sequence_length)
        col[1:limit] = prompt_tokens[1:limit]
        col
    end for _ in 1:batch_size]
    mask_indices = findall(==("[MASK]"), token_columns[1])

    function encode_tokens_matrix(columns)
        embeddings = map(columns) do tok_vec
            Float32.(prime_encode(tokenizer, tok_vec)) .* inv_norm
        end
        mat = hcat(embeddings...)
        return reshape(mat, sequence_length, length(columns))
    end

    seq = encode_tokens_matrix(token_columns)
    seq = use_gpu ? CUDA.cu(seq) : seq
    prev_columns = deepcopy(token_columns)

    if live_mode
        for _ in 1:batch_size
            println()
        end
    end

    rng = Random.default_rng()
    try
        for step in 1:steps
            output, st = model(seq, ps, st)
            preds = Array(output.logits)
            for (col_idx, col_tokens) in enumerate(token_columns)
                col_preds = preds[:, col_idx]
                for idx in mask_indices
                    raw = clamp(col_preds[idx] * norm_factor, min_prime, max_prime)
                    col_tokens[idx] = nearest_token(tokenizer, raw)
                end
            end
            frame_lines = nothing
            if record_io !== nothing || record_gif
                frame_lines = capture_frame(token_columns)
                record_io !== nothing && record_frame(record_io, step, frame_lines)
                record_gif && push!(gif_frames, frame_lines)
            end
            if live_mode
                render_live(token_columns, prev_columns, mask_indices, obfuscate)
            elseif show_matrix
                render_inline(step, token_columns, prev_columns, mask_indices, obfuscate)
            end
            prev_columns = deepcopy(token_columns)
            seq = encode_tokens_matrix(token_columns)
            seq = use_gpu ? CUDA.cu(seq) : seq
        end
    finally
        record_io !== nothing && close(record_io)
    end

    if record_gif && !isempty(gif_frames)
        mkpath(dirname(gif_path))
        save_gif(gif_path, gif_frames)
        println("Saved diffusion GIF to $(gif_path)")
    end

    println("Prompt: \"$(prompt)\"")
    if batch_size == 1
        println("Generated tokens:\n" * join(token_columns[1], " "))
    else
        println("Generated tokens (each batch column):")
        for (idx, col) in enumerate(token_columns)
            println("[column $(idx)] " * join(col, " "))
        end
    end
end

function save_gif(path::AbstractString, frames::Vector{Vector{String}})
    isempty(frames) && return
    project_root = normpath(joinpath(@__DIR__, ".."))
    python = ensure_python(project_root)
    frame_file = tempname()
    sentinel = "<<FRAME>>"
    open(frame_file, "w") do frame_io
        for frame in frames
            for line in frame
                println(frame_io, line)
            end
            println(frame_io, sentinel)
        end
    end
    frame_file_esc = replace(frame_file, "\\" => "\\\\", "\"" => "\\\"")
    out_esc = replace(path, "\\" => "\\\\", "\"" => "\\\"")
    script = replace("""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
src = Path(r"%FRAME_FILE%")
sentinel = "<<FRAME>>"
frames = []
block = []
with src.open('r', encoding='utf-8') as fh:
    for raw in fh:
        line = raw.rstrip('\\n')
        if line == sentinel:
            if block:
                frames.append(block)
                block = []
        else:
            block.append(line)
if block:
    frames.append(block)
if not frames:
    sys.exit(0)
wrap_width = 160
wrapped_frames = []
for frame in frames:
    wrapped = []
    for line in frame:
        chunks = textwrap.wrap(line, width=wrap_width, break_long_words=False, replace_whitespace=False)
        if not chunks:
            chunks = [""]
        wrapped.extend(chunks)
    wrapped_frames.append(wrapped)
frames = wrapped_frames
font = ImageFont.load_default()
char_w, char_h = font.getbbox('M')[2], font.getbbox('M')[3]
width = max(len(line) for frame in frames for line in frame) * char_w + 20
height = len(frames[0]) * (char_h + 4) + 20
images = []
for frame in frames:
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    y = 10
    for line in frame:
        draw.text((10, y), line, fill=(0, 255, 0), font=font)
        y += char_h + 4
    images.append(img)
images[0].save(r"%OUT%", save_all=True, append_images=images[1:], duration=200, loop=0)
""" , "%FRAME_FILE%" => frame_file_esc, "%OUT%" => out_esc)
    py_file = tempname() * ".py"
    open(py_file, "w") do py_io
        write(py_io, script)
    end
    try
        run(`$python $py_file`)
    finally
        rm(py_file, force=true)
        rm(frame_file, force=true)
    end
end

function ensure_python(root::String)
    venv_dir = joinpath(root, ".venv")
    python_path = joinpath(venv_dir, "bin", "python")
    if !isfile(python_path)
        run(`python3 -m venv $venv_dir`)
    end
    run(`bash -lc ". $venv_dir/bin/activate && pip install -q pillow"`)
    return python_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate()
end
