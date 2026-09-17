# Integrating Tuneout Resampler

Add the crate as a Rust Git/path dependency, construct a `Resampler`, and pass
PCM samples. No runtime dependencies, build-time generator or application
framework are required. Generate the API reference with
`cargo doc --no-deps --open`.

## Stream contract

`Resampler::new(input_hz, output_hz, channels)` fixes one stream's format.
Samples are interleaved: stereo is `[L0, R0, L1, R1, ...]`. Channels are filtered
independently, with no speaker interpretation or mixing. Each block must have
a multiple of `channels` samples. Both rates and channel count must be nonzero.

`end_of_stream` means the entire stream has ended. Mark the last nonempty block,
or submit a separate empty final block. Do not mark each decoder block as final.
A repeated empty flush is harmless. Further input requires `reset()`.

`process_f64_into` clears and replaces the supplied output vector on successful
calls. Consume its samples before the next call. It does not append a full track.
Validation errors preserve both the previous output vector and stream state.
The vector may grow. Internal history, padding and SIMD scratch can allocate too.

Equal rates preserve input bits and create no filter bank. After flushing,
resampling returns `ceil(total_input_frames * output_hz / input_hz)` frames.
Per-call output counts vary. An empty return means the converter needs more
input. Any underrun policy belongs to the application.

## Filter-length settings

Expose `Standard`, `Long` and `ExtraLong` as application choices, or pass an exact
tap count for advanced control. Generic Standard is the default. Long and
ExtraLong use 175% and 200% of the selected policy's Standard length, rounded
up to odd and capped to the rate pair's resource limits.

```rust
use tuneout_resampler::{Error, FilterLength, Resampler};

fn main() -> Result<(), Error> {
    let converter = Resampler::with_filter_length(48_000, 44_100, 2, FilterLength::Long)?;
    println!("Long: {} taps", converter.filter_info().taps_per_phase);

    let custom = Resampler::with_filter_length(48_000, 44_100, 2, FilterLength::Custom(4_095))?;
    assert_eq!(custom.filter_info().taps_per_phase, 4_095);
    Ok(())
}
```

To select predefined rate-specific minimum lengths, use
`Resampler::with_filter_length_policy(input_hz, output_hz, channels, length, policy)`
with `FilterLengthPolicy::Fixed`. The other choice, `FilterLengthPolicy::Generic`,
is the default used by `new` and `with_filter_length`. Long and ExtraLong scale
the selected policy's Standard length. Custom counts select the same filter
under either policy. `filter_length_policy()` reports the selected policy.

Fixed combines those minima with a ratio-based floor and can produce shorter
filters with different transition and rejection results. It shares the current
coefficient generator, cutoff, window, resource limits and SIMD selection.
Fixed names the minimum-length policy, not one universal tap count. The rates,
length preset and resource limits determine the effective length.

Custom counts are input-sample taps per phase, not milliseconds or output
samples. They must be odd, at least three, and within the rate-dependent limit.
`Error::InvalidFilterLength { requested, maximum }` reports an invalid request.
The library does not round or clamp it. Presets can converge to the same length
at a resource cap. Display the actual count and lookahead from `filter_info()`
when the distinction matters to your users. Equal rates still bypass filtering
after validating the configuration.

Longer filters need more processing, coefficient memory and future input, and
have longer ringing. Shorter custom filters can reduce bandwidth and alias
rejection. The multi-rate signal measurements cover Generic Standard. The
configuration and streaming checks for optional lengths do not establish a
quality guarantee for arbitrary custom lengths.
See [verification](verification.md) for the measured scope.

Construct a new converter to change length or policy. Apply settings at a stream
boundary, or let your application manage the transition between converters.
`reset()` keeps the selected length and policy. Converters with the same rates
and effective filter share cached coefficients even when their presets, policies
or channel counts differ.

## Buffering and audio callbacks

Construct and prepare the converter on a worker thread. Submit decoded blocks
there and queue the output for the audio callback. Account for your callback
cadence, resampler lookahead, phase batching and measured scheduling variation.
This crate does not provide a device queue or promise allocation-free callbacks.

Every filtered rate pair generates coefficients at extended precision on a cache
miss. Initial setup can take hundreds of milliseconds, and large banks can take
longer. Prepare expected conversions before playback if startup delay matters.
A matching cache hit avoids generation. An evicted bank must be generated again.

```rust
use tuneout_resampler::{Error, Resampler};

fn main() -> Result<(), Error> {
    let converter = Resampler::new(48_000, 44_100, 2)?;
    let info = converter.filter_info();
    let filter_ms = 1000.0 * info.lookahead_input_frames as f64 / info.input_rate as f64;
    let batch_ms = 1000.0 * info.additional_batch_input_frames as f64 / info.input_rate as f64;
    println!("Filter lookahead: {filter_ms:.2} ms. SIMD buffering bound: {batch_ms:.2} ms");
    // Add block delivery and your device queue to these buffering costs.
    Ok(())
}
```

`lookahead_input_frames` describes the centered FIR. The additional field is a
conservative bound for the selected phase batch. It is zero for mono,
multichannel and interpolated stereo. Divide input frames by the input rate for
seconds. The same dispatch rules cap this extra buffering at 64 ms and packed
scratch storage at 16 MiB for every pair. Device buffers, block delivery,
computation and scheduling add further delay. Output stays aligned to input
time without leading delay zeros.

Use ordinary release builds. Globally enabling AVX or `target-cpu=native` can
prevent a binary from running on older CPUs. Internal dispatch selects the
supported SIMD instructions automatically.

## Reset, rate changes, and sharing

Use one converter per continuous logical stream. `reset()` clears its history
and clock and restores initial boundary handling, retaining the filter bank
and reusable allocations. It does not affect other instances.

Reset before supplying audio from a new seek position. Construct a new instance
for rate, channel or filter-length changes. Flush the old one only if its tail
belongs in your output. For gapless material with the same format, keep one
stream across adjacent blocks or tracks and flush only when continuity ends.
Separately finishing tracks applies separate endpoint continuation to each.

Matching converters share immutable banks across channels and threads. Cache
lookup/construction is synchronized. Processing performs no cache lookup or
locking. A cache miss builds the filter. The eight-entry/64 MiB cache limit
bounds retained coefficient data, while active streams can retain evicted banks
beyond that limit. Eviction never invalidates an active converter. The crate
creates no background threads.

## Numeric limits

Finite PCM is expected. Output can overshoot full scale near transients. Choose
clipping, gain or dither policy at your application's final-format boundary.
f32 processing promotes already-rounded input to f64 and rounds output to f32.
Use f64 input/output to retain sub-f32 detail.

Every non-bypass conversion uses the common high-precision coefficient
generator, beta 20 and a nominal cutoff at the lower Nyquist frequency.
`FilterInfo::uses_precise_c1_bank` is a compatibility field and is always false.
Use the phase count, tap count and coefficient byte fields to inspect a filter.

Rates are fixed integers: this is not an asynchronous converter for drifting
hardware clocks. Arbitrary nonzero rates are accepted, but quality and processing
cost depend on the ratio. See [design.md](design.md) for tested boundaries and
extreme-rate limitations.

## License and redistribution

The resampler is available under the [Apache License, Version 2.0](../LICENSE).
Include a copy of the license when redistributing it, and preserve applicable
attribution from [NOTICE](../NOTICE) as required by Section 4. See
[LICENSING.md](../LICENSING.md) for redistribution guidance.
