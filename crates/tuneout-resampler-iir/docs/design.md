# Infinite filter design

Infinite combines a compact polyphase FIR prefix, a modal IIR core at a lower
internal rate, and a finite tail/phase correction. All accepted ratios use one
normalized model. The finite correction is asymmetric, so the complete response
is a mixed FIR/IIR design with phase advance rather than an ideal zero-phase filter.

## Response and timing

An integer rational clock fixes output positions. Finite contributions use FFT
convolution; recursive states handle the modal tail. The public
`frequency_response()` describes the analytic parent before the finite phase
correction. Measure converted PCM to inspect the complete response.
`prototype()` exposes a legacy diagnostic, not the complete buffered converter.

## Streaming and endpoints

The two-sided response needs future input: roughly 410–490 ms for 44.1→48,
48→44.1, 48→96, 96→44.1 and 192→44.1 kHz. Inspect `filter_info()` for the actual
configuration. Fixed internal blocks keep processing independent of caller chunks.

`Startup::Predict` uses bounded autoregressive fits of observed endpoints;
`Startup::Zero` supplies silence outside the record. Prediction can reduce boundary
transients in predictable signals, but cannot reconstruct unobserved audio.

## Precision and limits

Selected coefficient calculations and sums use a two-component floating-point
representation. SIMD, FFT and compensated kernels have different rounding paths;
bit equality across every architecture is not a general guarantee.

Finite guards approximate the recursive tail. Rate, channel, buffer and phase
limits can reject otherwise valid mathematical ratios. Operation and storage
counters have the scope described in the API; they are not CPU benchmarks.
