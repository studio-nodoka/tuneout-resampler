# Finite filter design

Finite uses a centered polyphase windowed-sinc FIR. Its C1 window and first
derivative reach zero at the support boundary, and each phase is normalized
for unity DC gain. The nominal cutoff is the lower input/output Nyquist frequency.

## Response and timing

Integer rational timing selects exact phase rows for denominators up to 2,048;
larger denominators interpolate between 1,024 stored phases. Generic scales
filter length with the rate ratio. Fixed combines rate-specific minima with
a ratio-based floor. Longer support narrows the transition but increases CPU,
memory, lookahead and ringing duration.

## Streaming and endpoints

The centered response needs future input. SIMD batching can require additional
buffering; both amounts are reported by `filter_info()`. Missing starting samples
use odd reflection. End continuation blends mirrored and odd-reflected samples
based on recent signal roughness. Endpoint handling is therefore adaptive;
retain one converter across continuous tracks.

## Precision and limits

Coefficients are constructed at extended precision and rounded to f64. Convolution
uses separate multiplication/addition in ascending tap order. Runtime SIMD selects
supported instructions without changing that order; portable fallbacks remain.

Filters are capped at 65,535 taps, 64 MiB per coefficient bank and 160 ms of input
support, with a minimum of three taps. These caps can limit rejection at extreme
ratios. Matching converters share immutable banks; the eight-entry/64 MiB cache
retention limit does not include banks still held by active streams.
