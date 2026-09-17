# Tuneout Resampler verification

Version **0.9.91 C1R3** offers Generic and Fixed length policies. Generic applies
one ratio-based support rule. Fixed combines predefined rate-specific minima
with a ratio-based floor. Both share the cutoff, coefficient generator and
streaming implementation. Generic Standard is the default. Long and ExtraLong
scale the selected policy's Standard length.

## Run the included checks

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
cargo doc --locked --no-deps
```

The suite includes three Rustdoc examples. Instruction-specific tests run when
the host supports those instructions. CI is configured for Windows, Linux and
macOS, checks Rust 1.89, and cross-checks Intel Mac, Apple Silicon and AArch64
Linux targets. See [ci.yml](../.github/workflows/ci.yml) and the repository's
Actions tab for current job results.

## One shared rate matrix

`tests/fixtures/rates.txt` lists 17 standard rates from 8 to 768 kHz.
`tests/support/cases.rs` generates every ordered pairing, including bypass,
plus conversions in both directions between each listed rate and its immediate
neighbors. This produces **357 pairs**, with identical coverage for every
listed rate.

| Area | Included checks |
| --- | --- |
| Standard regression | Complete-bank, f64-output and f32-output fingerprints for all 357 pairs under each policy |
| Preset regression | 1,071 complete-bank fingerprints per policy: every pair with Standard, Long and ExtraLong |
| Coefficient accuracy | Independent high-precision samples, phase reflection, exact sinc zeros and interpolation endpoints |
| Generic design | Scaling both rates preserves coefficients when resource limits do not bind |
| Length settings | Preset/custom agreement, invalid requests, odd lengths and resource caps |
| Policy settings | Generic defaults, established Fixed results, cross-policy custom bank sharing and reset |
| Streaming | Frame counts, irregular blocks, channel isolation, reset, final flush and bypass |
| Arithmetic | f32 rounding, separate multiply/add, cancellation, subnormals and signed zeros |
| SIMD | Reference-kernel equality, every phase-group tail, input boundaries, alignment and scratch bounds |
| Dispatch | Common phase, latency and memory rules. Fallback at budget boundaries |
| Cache | Effective-design identity, bounds, eviction, bank lifetime and concurrent initialization |
| Generic signals | Matrix-wide DC gain and sampled passband/stopband checks |

SIMD kernel tests separately exercise every coprime numerator/denominator
combination from 1 through 9. They vary phases, absolute stream positions,
tap counts, block sizes and final-drain behavior. Production preset lengths
are also compared with unbatched output. A final phase group can contain one,
two, three or four outputs. Every shape receives the same checks.

The development folders serve different purposes:

- `tests/` contains executable checks and their reference data.
- `tools/` regenerates those references for review.
- Normal dependency builds compile neither the integration tests nor the
  reference-generator tools. Applications need no Python installation.

## Independent coefficient references

The checked-in reference contains **15,818 sampled coefficients across 342
filtered pairs**: the shared matrix excluding equal-rate bypass, plus two
extreme-integer cases. Independent calculations at 80 and 100 decimal digits
agree after rounding to f64. The Rust test compares production values with
these references, accepting either sign of exact zero.

| File | Purpose |
| --- | --- |
| `tests/fixtures/rates.txt` | Common rate list |
| `tests/fixtures/uniform-precision.csv` | Standard coefficient and output fingerprints |
| `tests/fixtures/preset-banks.csv` | Complete-bank fingerprints for all three presets |
| `tests/fixtures/fixed-precision.csv` | Fixed Standard coefficient and output fingerprints |
| `tests/fixtures/fixed-preset-banks.csv` | Fixed complete-bank fingerprints for all three presets |
| `tests/fixtures/reference-coefficients.csv` | Independently calculated coefficient samples |

Normal tests require only Rust. Regenerating the independent reference requires
Python and mpmath as development tools. Generators print results without
overwriting committed fixtures:

```sh
mkdir -p artifacts
python tools/reference-coefficients.py > artifacts/reference-coefficients.csv
cargo run --release --example fingerprints > artifacts/uniform-precision.csv
cargo run --release --example fingerprints -- --presets > artifacts/preset-banks.csv
cargo run --release --example fingerprints -- --fixed > artifacts/fixed-precision.csv
cargo run --release --example fingerprints -- --presets --fixed > artifacts/fixed-preset-banks.csv
```

Production fingerprints detect changes. They do not independently establish
correctness. Review differences and rerun signal measurements before accepting
new fixtures.

Fixed fixtures cover the same 357-pair matrix. Both policies use common SIMD
dispatch rules, while their filter lengths can change buffering and per-call
output counts. The fixtures record each policy separately. Signal-quality
measurements remain specific to the measured policy and length.

## Bit-exact regression checks

Local before/after verification of the current implementation covered **90
configurations, 540 streams and 15,168 processing calls**. Every byte matched
across coefficients, f64/f32 audio, filter metadata, per-call output lengths
and capacities, and checked errors. Coverage included both policies, all three
presets, custom lengths, exact and interpolated phases, bypass, extreme rates,
and one, two and six channels. Raw-converter results also matched across
**43 success and error cases**. These comparisons use a separate development
harness. The included tests provide the portable regression checks.

For behavior-preserving refactors, retain the reference fixtures and compare
matching release builds with identical inputs, block boundaries and settings.
Check sample bits and output counts after every call, including reset and final
flush. The shared bit-assertion helper checks lengths and exact representations,
so signed-zero and subnormal differences remain visible. SIMD changes also
need host-supported kernel tests and a review of accumulation order.

## Signal measurements and limits

A separate local Generic Standard screen passed **357/357 pairs and 4,437 numerical
checks** using identical criteria for every pair. It checked:

- Gain error at most **0.01 dB through 96% of the lower Nyquist frequency**.
- At least **120 dB rejection from 104% of the lower Nyquist frequency**, wherever
  that stopband lies below the input Nyquist limit.
- Coherent tones in both channels, with f64 and f32 output.
- Residual SNR of at least 120 dB for f64 and 110 dB for f32.
- Streamed alias/image rejection, finite output, exact frame counts and DC gain.
- Filter lookahead within 80 ms for the matrix.

The response screen used up to 17 evenly spaced stored phases per pair, plus
adjacent-phase midpoints for interpolated tables. It checked DC gain across
every stored row. The analyzers were calibrated with ideal signals, a known
gain offset and a known spur. These are finite, sampled measurements, not a
proof over every frequency, phase, input or nonzero integer rate.

The signal screen is a separate development harness. `cargo test` runs the
included checks and reference comparisons. It does not run that external screen.
Generic's optional lengths have matrix-wide configuration, coefficient and
streaming checks, but do not inherit Generic Standard's complete signal screen.
Fixed has matrix-wide preset-bank and Standard-output references, plus focused
policy and streaming checks. Its transition and rejection results can differ
from Generic. Arbitrary custom lengths carry no fixed quality guarantee.

Large downsampling ratios can exhaust the common support budget, especially
with interpolated tables. See [extreme-rate limits](design.md#extreme-rate-quality-limits).
Keep generated reports and audio in ignored `artifacts/`.

## Portable validation

The current local regression run executed on Windows x86-64 with SSE2, AVX
and AVX-512F support. All-target compilation also passed for Intel Mac, Apple
Silicon and AArch64 Linux. Native ARM execution and performance require ARM
hardware. Cross-compilation verifies build compatibility. The CI configuration
above provides the additional platform checks when its jobs run.

## Measure setup and processing cost

```sh
cargo run --locked --release --example benchmark -- 48000 44100 2 1024
cargo run --locked --release --example benchmark -- 48000 44101 2 1024
cargo run --locked --release --example benchmark -- 48000 44100 2 1024 long
cargo run --locked --release --example benchmark -- 48000 44100 2 1024 extra-long
```

The benchmark reports first and cached setup, processing milliseconds per audio
second, actual tap count, coefficient storage, centered lookahead and the
separate SIMD batching bound. Processing is the median of five passes after a
warmup and includes final flush. Setup is measured separately.

Keep rates, channel count, precision, input duration, block size, build profile
and host conditions consistent. Account for cold setup and cache reuse
separately. Software processing time excludes device I/O. Lookahead, batching,
application queues and scheduling also contribute to latency. See the
[integration guide](integration.md).
