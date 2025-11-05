using Test
using Random
using Lux
using dlinoss

const DB = dlinoss.DiffusionBlocks
const DT = dlinoss.DiffusionBlocks.DiffusionTokenizerModule
const Masking = dlinoss.DiffusionBlocks.Masking

@testset "Prime codec round-trips" begin
    codec = DB.PrimeCodec(11, 3, 500)
    ids = [0, 1, 42, 499]
    Z = DB.ids_to_digit_matrix(codec, ids)
    @test size(Z) == (codec.L, length(ids))
    @test DB.digit_matrix_to_ids(codec, Z) == ids
end

@testset "Tokenizer span metadata" begin
    text = "Hello, GPT-5!"
    spans = DB.tokenize_with_spans(text; config = DB.TokenizeConfig())
    tokens = [span.token for span in spans]
    surfaces = [span.surface for span in spans]
    ranges = [span.span for span in spans]

    @test tokens == ["hello", ",", "gpt-5", "!"]
    @test surfaces == ["Hello", ",", "GPT-5", "!"]
    @test ranges[1] == 1:5
    @test ranges[end] == lastindex(text):lastindex(text)

    vocab = DB.build_vocab([text]; vocab_size = 16)
    codec = DB.PrimeCodec(13, 3, length(vocab.itos))
    tok = DT.DiffusionTokenizer(vocab, codec)
    ids, digits, span_info = DT.encode_text_to_digits_with_spans(tok, text)

    @test length(ids) == length(span_info)
    @test digits == DB.ids_to_digit_matrix(codec, ids)
end

@testset "Masking and unmasking schedules" begin
    vocab_tokens = vcat(DB.DEFAULT_SPECIALS, ["alpha", "beta", "gamma"])
    stoi = Dict(tok => i - 1 for (i, tok) in enumerate(vocab_tokens))
    vocab = DB.Vocab(stoi, collect(vocab_tokens))
    codec = DB.PrimeCodec(7, 2, length(vocab_tokens))

    ids = collect(0:length(vocab_tokens)-1)
    digits = DB.ids_to_digit_matrix(codec, ids)
    mask_id = DB.mask_id(codec)

    rng = Random.MersenneTwister(1234)
    protected = Masking.protected_columns(ids, vocab)
    keep_prob = 0.4f0
    masked = Masking.forward_mask(rng, codec, digits, keep_prob; protected_cols = protected)

    specials = length(DB.DEFAULT_SPECIALS)
    @test masked[:, 1:specials] == digits[:, 1:specials]
    @test all(x -> x == mask_id || (0 ≤ x < codec.base), masked)
    @test any(x -> x == mask_id, masked[:, specials+1:end])

    rng2 = Random.MersenneTwister(1234)
    masked_again = Masking.forward_mask(rng2, codec, digits, keep_prob; protected_cols = protected)
    @test masked_again == masked

    # Progressive reveal
    fully_masked = fill(mask_id, size(digits))
    rng3 = Random.MersenneTwister(4321)
    revealed = Masking.partial_unmask(rng3, codec, fully_masked, digits, 0.75f0)
    @test all(x -> x == mask_id || (0 ≤ x < codec.base), revealed)
    @test any(x -> x != mask_id, revealed)
end

@testset "Diffusion pipeline wiring" begin
    texts = ["Hello, GPT!", "Prime SSM backbone"]
    vocab = DB.build_vocab(texts; vocab_size = 64)
    codec = DB.PrimeCodec(17, 3, length(vocab.itos))
    tok = DT.DiffusionTokenizer(vocab, codec)
    bundle = dlinoss.build_diffusion_ossm(tok;
        d_sub = 3,
        d_model = 24,
        max_len = 128,
        num_oscillators = 6)

    rng = Random.MersenneTwister(2025)
    ps, st = Lux.setup(rng, bundle.layer)
    sample = dlinoss.forward_diffusion_step_with_spans(
        bundle,
        Random.MersenneTwister(7),
        "Hello, GPT!",
        0.6f0,
        ps,
        st;
        t = 0.2f0,
    )

    @test size(sample.logits, 1) == codec.L
    @test size(sample.logits, 2) == codec.base + 1
    @test size(sample.logits, 3) == size(sample.masked_digits, 2)
    @test sample.clean_digits == DB.ids_to_digit_matrix(codec, sample.ids)
    @test length(sample.spans) == length(sample.ids)
    @test all(col -> 1 ≤ col ≤ length(sample.ids), sample.protected_cols)
end
