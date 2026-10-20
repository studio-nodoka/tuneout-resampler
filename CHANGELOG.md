# Changes

## 1.0.0 — 2026-10-20

- Release Finite and Infinite together as 1.0.0 in one Cargo workspace, with a
  shared version and release tag.
  Existing package names, APIs and Finite defaults remain compatible.
- Name the primary Rust types `FiniteResampler` and `InfiniteResampler`. Keep
  `Resampler` in Finite and `BufferedResampler` in Infinite as compatibility aliases.
- Update Infinite's finite phase correction and bounded endpoint handling.
  `InfiniteResampler` is the maintained mixed FIR/IIR engine; the legacy causal
  `Resampler` remains a separate API.
- Preserve converter output through the subsequent allocation, indexing and
  scalar-target Clippy cleanup.
- Run strict workspace Clippy across the cross-target CI matrix, including
  ARMv7 and WebAssembly with and without SIMD128.
- Use matching, shorter engine guides focused on setup, streaming, configuration,
  playback constraints and high-level filter design.

The workspace includes both engines' tests, examples and license notices.
Crates.io publication is disabled for both packages.
