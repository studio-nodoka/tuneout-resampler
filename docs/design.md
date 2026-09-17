# Tuneout Resampler design

The core is a conventional polyphase windowed-sinc FIR, with custom window
construction, support selection, precise timing and finite-stream boundaries.

## Window and coefficients

For normalized distance `u` from the fractional-phase center, the C1 window uses
the positive tail of the modified Bessel series:

```text
y = beta² (1 - u²) / 4
tail(y) = sum from k=2 to infinity of y^k / (k!)²
window(u) = tail(y) / tail(beta² / 4), for |u| < 1
window(u) = 0, otherwise
```

Removing the first two Bessel terms makes the window and its first derivative
vanish at the support boundary. The window shifts with the sinc's fractional
phase. Summing the positive tail avoids subtracting nearly equal values near
the endpoints. Each phase is normalized for unity DC gain.

Every filtered rate pair uses beta 20 and a nominal cutoff at the lower
Nyquist frequency. The default Generic support formula scales with decimation
for all pairs, including interpolated banks. The declared engineering targets are
at most 0.01 dB gain error through 96% of the lower Nyquist frequency and at
least 120 dB rejection from 104% of it. The intervening region is the transition
band, not a guaranteed deep stopband. These targets apply to the recorded
qualification matrix with Generic Standard support, not every representable
integer rate, policy or custom length.

Each coefficient row is a fractional phase. A final extra row supports linear
interpolation where the exact rational denominator exceeds 2,048 phases. Those
cases use 1,024 phases. Sample positions always advance using integer numerator,
denominator, and remainder state. SIMD lanes process independent outputs or
channels with separate multiplication/addition in ascending tap order. No FMA
or horizontal reduction changes the qualified arithmetic.

## Configurable support

`Resampler::new` selects `FilterLength::Standard` with `FilterLengthPolicy::Generic`.
`with_filter_length` also accepts `Long`, `ExtraLong`, or `Custom(taps)`, retaining
the Generic policy. Generic Standard starts with
`ceil(1664 × input_rate / min(input_rate, output_rate))`, rounds upward to odd,
and applies the common resource caps. Upsampling therefore uses 1,665 taps
before caps. Downsampling widens the support in proportion to the bandwidth
reduction. Scaling both rates by the same factor preserves the coefficients
when the resource caps do not change the effective length.

`with_filter_length_policy` can instead select `FilterLengthPolicy::Fixed`.
Fixed takes the larger of an odd-rounded `ceil(512 / ratio)` base and the
predefined minimum for the rate pair, where `ratio = min(output / input, 1)`.
Rate-specific minima apply only to exact phase banks. Interpolated banks use
the base alone. One private helper preserves the established thresholds and
floating-point rounding. Resource caps still apply. Fixed refers to the minima,
not one universal tap count. This policy can choose different lengths when both
rates are scaled by the same factor.

Long multiplies the selected policy's Standard tap count by 7/4. ExtraLong
multiplies it by 2. Each result rounds upward to an odd count and then respects
the same rate-dependent limits. Capping Standard before scaling is equivalent
to scaling first and capping afterwards for these multipliers, and keeps the
integer arithmetic bounded on 32-bit targets.

Every design is limited to 65,535 taps, 64 MiB of coefficients including the
extra interpolation row, and 160 ms of input support. The smallest limit is
rounded down to odd. A minimum of three taps takes precedence at unusually low
input rates. Centered lookahead is `taps / 2` input frames (integer division).
Custom counts must be odd, at least three, and no greater than that limit.
Invalid counts return the requested and maximum values in
`Error::InvalidFilterLength`, without allocating a bank or silently shortening
the requested filter. Validation also applies to equal-rate bypass.

Length and policy are immutable for a stream and retained by `reset()`. Changing
either requires a new converter. Custom tap counts are independent of policy.
The effective tap count, rather than the preset or policy label, forms
part of the coefficient cache key. A custom count matching a preset can share
its resident bank. Different lengths cannot accidentally reuse one another's
coefficients. `filter_info()` exposes the effective length and resource costs.
`filter_length_policy()` reports the selected policy.

Longer support narrows the transition while increasing computation, memory,
lookahead and ringing duration. Shorter custom support can lose bandwidth and
alias rejection. Length selection changes neither the window shape parameter,
cutoff, phase count, arithmetic nor finite-stream continuation policy.

## Uniform coefficient precision

Every filtered rate pair uses `src/precision.rs`, including interpolated banks.
Its double-double arithmetic stores each intermediate value as a leading f64
and a residual, providing roughly 106 significand bits. The normalized result
is rounded to f64 for the convolution kernels. This improves
coefficient accuracy. It does not make streamed audio mathematically exact.

Error-free addition and Dekker multiplication retain low bits without requiring
FMA. Integer remainder reduction preserves sinc zeros and reduces sine's
argument before multiplication by pi. Sine and the positive Bessel tail use
bounded series. The production beta is 20. Opposite phases share reflected
coefficients, and the final interpolation row is phase zero shifted by one tap.
No platform `sin`, `long double`, build-time generator, or runtime dependency
is required. The standard arithmetic techniques are described in
[Hida, Li and Bailey's double-double paper](https://www.davidhbailey.com/dhbpapers/qd.pdf).

The generator is private and specialized to the bounded filter parameters used
by this crate. It is not a general arbitrary-precision math library. A scratch
row holds two f64 values per tap, at most approximately 1 MiB in addition to the
coefficient bank. Generation occurs on a cache miss, outside the cache mutex.

Under Generic, every pair follows the same support rule. Fixed uses predefined
rate-specific minima with a ratio-based floor. Both policies share cutoff,
precision and dispatch rules. Effective tap counts, costs and quality can differ.

`tools/reference-coefficients.py` provides an independent mpmath calculation
at 80 and 100 decimal digits for the checked-in test reference. Production
fingerprints are separate regression data. See [verification.md](verification.md).

## Boundaries and buffering

A centered FIR needs future input. Streaming waits for that lookahead instead
of adding leading zero frames to the output timeline. The support-time cap
normally bounds filter lookahead below 80 ms. The three-tap minimum takes
precedence at extremely low rates. `filter_info()` reports it in input frames.
Whole-period SIMD can add buffering beyond this lookahead. Its bound is the
number of batched periods multiplied by the reduced input-rate numerator, in
input frames. `additional_batch_input_frames` reports the selected bound for
each converter. Divide by the input sample rate to obtain seconds. Interpolated
SIMD adds no batching delay. A common 64 ms limit bounds additional phase
batching independently of filter lookahead.

At the start, missing samples use odd reflection about the first frame. At the
end, continuation blends mirrored and odd-reflected samples according to the
last 64 frames' normalized roughness, independently per channel. This general
finite-stream policy makes whole-record endpoint handling nonlinear. Interior
convolution remains linear. Very short streams have
boundary-specific behavior and should not be mistaken for steady-state spectra.

Finish a continuous stream once. Separately finishing adjacent blocks or tracks
creates separate boundary continuations. Gapless users should retain one stream
when continuity is required. The library does not join files or manage tracks.

## Precision and resource scope

Both interfaces use f64 coefficients and accumulation. The f32 interface first
promotes its already-rounded input and rounds returned samples back to f32. It
cannot recover information lost in f32 input. Equal-rate conversion copies
samples without constructing a filter.

Each bank is bounded to 64 MiB and 65,535 taps. Matching instances share the bank.
Histories and clocks remain independent. History/output vectors depend on block
size and channels. Generation, buffer growth and allocation are real costs.
`process_f64_into` reuses output capacity without guaranteeing allocation-free
processing.
No claim is made that every supported configuration meets a device deadline.

## Portable SIMD and immutable cache

Exact-phase stereo batching uses eight or sixteen phase periods, depending on
the phase geometry, resource budgets and available instructions. AVX-512F uses
sixteen periods. AVX and NEON use eight. Packing must serve a complete phase
group: at least four phases for AVX-512, or two for AVX/NEON. Single-phase ratios
use immediate kernels. A tile must fit both the 64 ms extra
buffering budget and a conservative 16 MiB scratch-storage budget. If a
sixteen-period tile lacks enough phases or exceeds a budget, dispatch tries
the eight-period AVX tile.
Exact stereo conversions that cannot batch within those limits use immediate-output
SSE2/NEON or scalar kernels. Interpolated stereo uses AVX, SSE2, NEON or scalar
arithmetic across four already-available frames. Fewer available frames are
emitted immediately. Dispatch applies the same rules to every rate pair,
without CPU-model tuning or named-rate lists. The Fixed length policy uses
these same dispatch rules.

The AVX-512F kernel derives its loop bounds and scratch storage from the reduced
ratio and configured tap count. Its final group can contain one through four
outputs. All phase-batch kernels support variable lengths without changing
the accumulation order.

Packed AVX scratch is aligned to 32 bytes (at most three extra doubles), and
AVX-512 scratch to 64 bytes. Unsafe code is confined to the private FIR module:
runtime features gate calls, slice bounds cover loads/stores, and scratch
allocations include alignment padding. There is no public unsafe API.

Banks use `Arc<Vec<f64>>`, keyed by both actual rates, phase/tap counts, and
exact beta/rolloff bits. The LRU cache reserves in-flight construction bytes,
retains at most eight entries/64 MiB of coefficients, and generates outside its
mutex. A per-entry `OnceLock` coalesces requests while the entry remains resident.
No cache locking or Arc cloning occurs in the sample-processing loop. Active
streams can retain evicted banks beyond the cache limit. The cache can retain
64 MiB after streams stop. It is not a whole-process memory cap.

## Extreme-rate quality limits

Interpolated tables reach the bank-size cap at 8,183 taps. At sufficiently
large downsampling ratios, that cap prevents support from widening in proportion
to the bandwidth reduction. The same limitation can occur at the tap or time
cap. The converter uses one filter stage with a bounded support budget.
Coefficient precision cannot compensate for insufficient filter length.
Rate acceptance therefore does not establish a quality guarantee. See
[verification.md](verification.md) for the measured scope.
