# Tuneout Resampler

Two Rust sample-rate converters for interleaved PCM: **Finite** and **Infinite**.
Use either independently, without an application framework or audio-device dependency.
Both engines share release version **1.0.0** and are versioned together.

| | Finite | Infinite |
| --- | --- | --- |
| Package | `tuneout-resampler` | `tuneout-resampler-iir` |
| Rust API type | `FiniteResampler` | `InfiniteResampler` |
| Filter | C1 windowed-sinc polyphase FIR | Modal IIR core with finite FIR correction |
| Controls | Length policies, presets and custom taps | Predictive or zero endpoints |
| Runtime dependencies | None | RustFFT 6.4.1 and its dependencies |
| Lookahead | Depends on length, ratio and SIMD batching | About 410–490 ms for documented common pairs |

Both are buffered converters with f64/f32 interfaces, independent channels,
integer rational timing, reusable f64 output buffers, reset and final flush.
Infinite is a mixed FIR/IIR design with an asymmetric phase correction. Neither
engine adapts to drifting hardware clocks.

The previous names remain compatible: `tuneout_resampler::Resampler` aliases
`FiniteResampler`, and `tuneout_resampler_iir::BufferedResampler` aliases
`InfiniteResampler`.

## Get started

Requires **Rust 1.89 or newer**. Choose the package you need:

```toml
[dependencies]
tuneout-resampler = { git = "https://github.com/studio-nodoka/tuneout-resampler", tag = "v1.0.0" }
# Or:
tuneout-resampler-iir = { git = "https://github.com/studio-nodoka/tuneout-resampler", tag = "v1.0.0" }
```

Each engine guide has a complete streaming example and its configuration options.
Construct and process on a worker, then queue PCM for the audio callback.
Both engines can allocate and need future input. Compare setup, warmed CPU cost,
drain cost and lookahead on the intended rate pairs and processor.

## Documentation

| Guide | Finite | Infinite |
| --- | --- | --- |
| Usage and configuration | [Finite](docs/finite.md) | [Infinite](crates/tuneout-resampler-iir/README.md) |
| Design | [Filter and implementation](docs/design.md) | [Filter and implementation](crates/tuneout-resampler-iir/docs/design.md) |

Both usage guides follow the same structure and include streaming, endpoint and
playback constraints. See [CHANGELOG.md](CHANGELOG.md) for package changes.

## Build and check

Finite is the root package; Infinite is an independent workspace member.
Both inherit the version from `[workspace.package]` in the root `Cargo.toml`.

```sh
cargo build --release --locked --workspace
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --release --locked --workspace --all-targets
cargo test --release --locked --workspace --doc
cargo doc --locked --workspace --no-deps
```

Add `--offline` when dependencies are cached. CI configures native tests and
cross-target checks; cross-compilation does not establish execution or performance
on another processor. Tests and examples are development tools.

## License

[Apache 2.0](LICENSE). Retain the applicable [Finite notice](NOTICE) and
[Infinite notice](crates/tuneout-resampler-iir/NOTICE.md); see
[LICENSING.md](LICENSING.md).
