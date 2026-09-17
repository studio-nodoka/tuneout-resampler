# Tuneout Resampler

**Version 0.9.91 C1R3** (`0.9.91-C1R3` in Cargo).

A Rust library for fixed-rate, streaming polyphase FIR sample-rate conversion
with **zero runtime dependencies**. It processes interleaved PCM without an
application framework.

Licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for attribution and
[LICENSING.md](LICENSING.md) for redistribution guidance.

## Features

- **f64 and f32 interfaces** for mono, stereo and multichannel audio, using f64
  coefficients and accumulation.
- **Exact rational timing** with integer clocks. Rate pairs with at most 2,048
  fractional positions use exact phase rows. Larger denominators interpolate
  between 1,024 stored phases.
- **C1 windowed-sinc filters** whose window and first derivative reach zero at
  the support boundary. Every filtered pair uses the same portable high-precision
  coefficient generator and nominal cutoff at the lower Nyquist frequency.
- **Configurable filter length:** Standard, Long and ExtraLong presets, plus
  custom odd tap counts. Generic applies one ratio-based support rule to every
  pair. Fixed combines predefined rate-specific minima with a ratio-based floor.
- **Runtime SIMD selection:** AVX-512F, AVX, SSE2 and AArch64 NEON where applicable,
  with portable fallbacks. Selection uses phase geometry, instruction support
  and common buffering budgets. The AVX path requires neither AVX2 nor FMA.
- **Shared coefficient banks** with up to eight entries and 64 MiB retained in
  the cache. Each stream has independent history and timing.
- Final flush, reset, equal-rate bypass and reusable f64 output buffers.

## Get started

Requires **Rust 1.89 or newer**. Add a Git or local path dependency.

```toml
[dependencies]
tuneout-resampler = { git = "https://github.com/studio-nodoka/tuneout-resampler" }
```

```rust
use tuneout_resampler::{Error, Resampler};

fn convert(input: &[f64]) -> Result<Vec<f64>, Error> {
    let mut converter = Resampler::new(48_000, 44_100, 2)?;
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

Each block must contain complete interleaved frames: one sample per channel.
Early calls can return no output while collecting future input or a SIMD batch.
Mark only the final block as `end_of_stream`, or flush once with an empty final
block. The complete output contains
`ceil(input_frames * output_rate / input_rate)` frames. Repeated empty flushes
produce nothing. Call `reset()` before supplying a new stream.

`process_f64_into` replaces the output vector's contents while reusing capacity.
`process_f64` returns a new vector. `process_f32` uses f64 accumulation and returns
f32 samples. Equal input and output rates bypass filtering and preserve sample bits.

## Choose a filter length

`Resampler::new` uses Standard. Use `Resampler::with_filter_length` to select
another setting when constructing a converter.

| Setting | Filter length |
| --- | --- |
| `FilterLength::Standard` | Base support under the selected length policy |
| `FilterLength::Long` | 175% of Standard, rounded up to odd |
| `FilterLength::ExtraLong` | 200% of Standard, rounded up to odd |
| `FilterLength::Custom(taps)` | Exact odd tap count within the rate-dependent limits |

```rust
use tuneout_resampler::{Error, FilterLength, Resampler};

fn configured_converter(length: FilterLength) -> Result<Resampler, Error> {
    Resampler::with_filter_length(48_000, 44_100, 2, length)
}
```

Longer filters narrow the transition band while increasing processing cost,
memory, lookahead and ringing duration. All settings use the same window,
cutoff and coefficient precision.

The default `FilterLengthPolicy::Generic` uses
`ceil(1664 × input_rate / min(input_rate, output_rate))` for Standard, rounded
up to an odd tap count and capped by the common resource limits.

Use `with_filter_length_policy` to select `FilterLengthPolicy::Fixed`, which
combines predefined rate-specific minimum lengths with a 512/ratio base rule.
Fixed refers to those predefined minima. The actual tap count still depends on
the rates, length preset and resource limits:

```rust
use tuneout_resampler::{Error, FilterLength, FilterLengthPolicy, Resampler};

fn fixed_converter() -> Result<Resampler, Error> {
    Resampler::with_filter_length_policy(
        48_000, 44_100, 2, FilterLength::Standard, FilterLengthPolicy::Fixed,
    )
}
```

Long and ExtraLong scale the selected policy's Standard length. Fixed can
produce shorter filters with different transition and rejection results.
Generic remains the default. Both policies use the same coefficient generator,
window, cutoff, resource limits and generalized SIMD rules. Custom lengths are
independent of the policy. See [design](docs/design.md#configurable-support).

Presets respect the resource limits, so some rate pairs use the same length for
multiple presets. Custom counts must be odd, at least three, and within the
rate-dependent maximum. Invalid requests return `Error::InvalidFilterLength`.
Custom counts are never rounded or clamped. Shorter filters can reduce alias
rejection and retained bandwidth.

`filter_info()` reports the actual tap count, coefficient bytes, filter lookahead
and separate SIMD batching bound. Both buffering fields use input frames. The
selected length and policy stay fixed for the stream, including after `reset()`.
`filter_length_policy()` reports the selected policy.

## Performance and limits

Construct or warm converters on a worker before playback. Building a coefficient
bank can take hundreds of milliseconds or longer. A matching cache hit avoids
that work. Active streams can retain evicted banks beyond the cache's 64 MiB
retention limit.

Construction and processing can allocate. Reusing output buffers reduces
allocations but does not make processing allocation-free. Filter lookahead,
SIMD batching, block delivery and device queues all contribute to latency.

Generic Standard has passed signal-quality checks for **357 rate pairs**, covering
standard rates from 8 to 768 kHz and neighboring rates. Other nonzero integer
rates are accepted, with quality and cost depending on the ratio and resource
limits. Long, ExtraLong and custom lengths have separate configuration and
streaming checks. They do not inherit Standard's signal
measurements. The [verification guide](docs/verification.md) records test scope
and known extreme-rate limits.

## Examples

Measure setup, processing time, coefficient storage and buffering on your host:

```sh
cargo run --locked --release --example benchmark -- 48000 44100 2 1024
cargo run --locked --release --example benchmark -- 48000 44101 2 1024
cargo run --locked --release --example benchmark -- 48000 44100 2 1024 long
cargo run --locked --release --example benchmark -- 48000 44100 2 1024 standard fixed
```

Convert little-endian f64 PCM or export a coefficient bank:

```sh
cargo run --locked --release --example raw -- convert 48000 44100 2 input.f64le output.f64le 1024 f64
cargo run --locked --release --example raw -- coefficients 48000 44100 bank.f64le
```

Both examples accept an optional length (`standard`, `long`, `extra-long`, or an
odd tap count such as `4095`), followed by an optional policy (`generic` or
`fixed`). Omitting both selects Generic Standard. The raw example's
`f32` mode rounds input and output to f32 while keeping f64 files for comparisons.

Use ordinary release builds. Instruction support is detected at runtime.
Finite input is expected, and output can exceed full scale. The application
chooses clipping, dither, normalization and gain policies.

## Verification

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
cargo doc --locked --no-deps
```

The included tests and three documentation examples check streaming,
configuration, independent coefficient references, SIMD equality, cache behavior
and portable fingerprints for 357 rate pairs. The same matrix covers all three
presets under both policies, with 1,071 coefficient-bank fingerprints per policy.
Fixed also has coefficient and output regression references. Bit comparisons
check exact sample representations, including signed zeros and subnormals. See
the [verification guide](docs/verification.md) for evidence and reproduction steps.

CI is configured for Windows, Linux, macOS and Rust 1.89, with cross-compilation
checks for Intel Mac, Apple Silicon and AArch64 Linux.

## Documentation and source

- [Integration](docs/integration.md): stream handling, buffering and application use.
- [Design](docs/design.md): coefficient precision, filter selection, SIMD and resource limits.
- [Verification](docs/verification.md): test coverage, fixtures and measurement guidance.
- `tests/` and `tools/`: regression checks and reference generators for developers.
  Normal application builds do not compile or run them.
- [`src/lib.rs`](src/lib.rs): public API and validation.
- [`src/design.rs`](src/design.rs) and [`src/precision.rs`](src/precision.rs): filter selection and coefficients.
- [`src/cache.rs`](src/cache.rs): shared coefficient banks.
- [`src/fir.rs`](src/fir.rs): streaming, SIMD kernels and endpoint handling.

Unsafe code is confined to private SIMD kernels. The public API, coefficient
generator and cache use safe Rust. The kernels preserve ascending tap order and
separate multiplication/addition. Lane layouts and rounding rules are documented
alongside the code.
