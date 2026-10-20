# Finite

**1.0.0** · `tuneout-resampler` · Rust 1.89+

A polyphase FIR sample-rate converter for interleaved f64/f32 PCM, with independent
channels and no runtime dependencies. [Apache 2.0](../LICENSE); see [NOTICE](../NOTICE).

## Get started

```toml
[dependencies]
tuneout-resampler = { git = "https://github.com/studio-nodoka/tuneout-resampler", tag = "v1.0.0" }
```

```rust
use tuneout_resampler::{Error, FiniteResampler};

fn convert(input: &[f64]) -> Result<Vec<f64>, Error> {
    let mut converter = FiniteResampler::new(48_000, 44_100, 2)?;
    let mut block_output = Vec::new();
    let mut complete_output = Vec::new();
    for block in input.chunks(1024 * 2) {
        converter.process_f64_into(block, false, &mut block_output)?;
        complete_output.extend_from_slice(&block_output);
    }
    converter.process_f64_into(&[], true, &mut block_output)?;
    complete_output.extend_from_slice(&block_output);
    Ok(complete_output)
}
```

`Resampler` remains a compatibility alias for `FiniteResampler`.

## Configuration

`new()` selects Generic Standard. Use `with_filter_length` to change the length,
or `with_filter_length_policy` to select both length and policy.

| Option | Behavior |
| --- | --- |
| `FilterLengthPolicy::Generic` | One ratio-based support rule; the default |
| `FilterLengthPolicy::Fixed` | Rate-specific minimum lengths with a ratio-based floor |
| `FilterLength::Standard` | Base support for the selected policy |
| `FilterLength::Long` / `ExtraLong` | 175% / 200% of Standard, rounded up to odd and capped |
| `FilterLength::Custom(taps)` | Exact odd count, at least three and within the rate-dependent limit |

Invalid custom counts return `Error::InvalidFilterLength`. Presets can converge
at resource caps. Create a new converter to change length or policy.

## Streaming

Supply complete interleaved frames. `process_f64_into` replaces the output
vector's contents while reusing capacity; consume each block before the next call.
`process_f64` and `process_f32` return new vectors. Early calls may return no samples.

Flush at the end of the complete stream, as shown above. The final frame count is
`ceil(input_frames * output_rate / input_rate)`; repeated empty flushes emit nothing.
Keep one instance across continuous tracks. Reset for a seek or independent stream,
and construct a new instance when rates or channels change. Equal rates bypass
filtering and preserve sample bits.

## Performance and limits

Construct and process on a worker, then queue PCM for the audio callback. Setup
can take hundreds of milliseconds or longer on a coefficient-cache miss.
Processing can allocate; reuse the converter and output capacity.

`filter_info()` reports effective taps, coefficient bytes, filter lookahead and
a separate SIMD batching bound. Divide input-frame counts by the input rate for
seconds. Rates are fixed positive integers; the converter does not follow clock drift.
Supply finite PCM. Output may overshoot full scale; gain, clipping and dither are
application choices. Use f64 to retain sub-f32 detail.

## Filter design

A centered C1 windowed-sinc FIR uses exact rational timing and a nominal cutoff
at the lower Nyquist frequency. Longer filters narrow the transition at the cost
of CPU, memory, lookahead and ringing duration. Resource caps can limit rejection
at extreme ratios. See the [design overview](design.md).

## Build and check

Run from this crate's directory or the combined workspace root:

```sh
cargo test --release --locked -p tuneout-resampler
cargo clippy --locked --all-targets -p tuneout-resampler -- -D warnings
cargo doc --locked --no-deps -p tuneout-resampler --open
cargo run --release --locked -p tuneout-resampler --example benchmark -- 48000 44100 2 1024
```

The generated API reference describes all constructors, diagnostics and errors.
