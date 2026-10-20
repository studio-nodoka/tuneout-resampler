# Infinite

**1.0.0** · `tuneout-resampler-iir` · Rust 1.89+

A mixed FIR/IIR sample-rate converter for interleaved f64/f32 PCM, with independent
channels and RustFFT 6.4.1 as its runtime dependency. [Apache 2.0](LICENSE);
see [NOTICE.md](NOTICE.md).

## Get started

```toml
[dependencies]
tuneout-resampler-iir = { git = "https://github.com/studio-nodoka/tuneout-resampler", tag = "v1.0.0" }
```

```rust
use tuneout_resampler_iir::{Error, InfiniteResampler};

fn convert(input: &[f64]) -> Result<Vec<f64>, Error> {
    let mut converter = InfiniteResampler::new(48_000, 44_100, 2)?;
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

`BufferedResampler` remains a compatibility alias for `InfiniteResampler`.

## Configuration

`InfiniteResampler::new()` selects predictive endpoints. Choose explicitly with
`InfiniteResampler::with_startup(input_hz, output_hz, channels, startup)`.

| Option | Behavior |
| --- | --- |
| `Startup::Predict` | Estimate missing endpoints from at most 256 observed frames; the default |
| `Startup::Zero` | Assume silence outside the supplied record |

Prediction cannot recover unobserved music or replace continuous track history.
The older `Resampler` type is a separate legacy causal converter.

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

Construct and process on a worker, then queue PCM for the audio callback.
Construction and processing can allocate; reuse the converter and output capacity.

Lookahead is about **410–490 ms** for the documented common pairs.
`filter_info().startup_lookahead_frames` gives the initial input requirement;
divide by the input rate for seconds. Aligned output still needs that future input.

Rates are fixed positive integers with 1–64 channels. Nonfinite input, overflow
and resource-limit violations return errors. The converter does not follow clock
drift. Output may overshoot full scale; gain, clipping and dither are application
choices. Use f64 to retain sub-f32 detail.

## Filter design

A modal IIR core works with a compact FIR prefix and a finite tail/phase correction.
Every accepted ratio uses the same normalized model. The correction is asymmetric;
the complete converter is neither purely IIR nor an ideal zero-phase filter.
See the [design overview](docs/design.md).

## Build and check

Run from this crate's directory or the combined workspace root:

```sh
cargo test --release --locked -p tuneout-resampler-iir
cargo clippy --locked --all-targets -p tuneout-resampler-iir -- -D warnings
cargo doc --locked --no-deps -p tuneout-resampler-iir --open
cargo run --release --locked -p tuneout-resampler-iir --example compare_buffered
```

The generated API reference describes all constructors, diagnostics and errors.
The comparison example uses Finite Standard/Fixed.
