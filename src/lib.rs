//! # Tuneout Resampler
//!
//! Streaming C1 polyphase FIR sample-rate conversion for interleaved PCM.
//!
//! All filtered rate pairs share a portable high-precision coefficient generator
//! and cutoff policy. Integer timing selects exact phase rows or interpolates
//! larger denominators. The default support rule depends on the rate ratio.
//! [`FilterLengthPolicy::Fixed`] uses predefined rate-specific minimum lengths.
//! SIMD batching follows common geometry and resource rules under both policies.
//!
//! ```
//! use tuneout_resampler::Resampler;
//! # fn main() -> Result<(), tuneout_resampler::Error> {
//! let mut converter = Resampler::new(48_000, 44_100, 2)?;
//! let stereo_input = vec![0.0_f64; 2 * 4096];
//! let mut output = converter.process_f64(&stereo_input, false)?;
//! output.extend(converter.process_f64(&[], true)?);
//! assert_eq!(output.len() / 2, (4096_u64 * 44_100).div_ceil(48_000) as usize);
//! # Ok(())
//! # }
//! ```
//!
//! Set `end_of_stream` only for the final block, or flush with an empty final
//! block. Call [`Resampler::reset`] before reusing an instance for a new stream.
//! Equal rates bypass filtering. Construction and processing allocate memory.
//! This crate does not promise allocation-free device-callback operation.
//! Use [`Resampler::with_filter_length`] to choose a preset or an exact tap count.
//! Use [`Resampler::with_filter_length_policy`] to select the length policy too.

#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod cache;
mod design;
mod precision;
#[cfg(test)]
#[path = "../tests/support/assertions.rs"]
mod test_assertions;
#[cfg(test)]
#[path = "../tests/support/cases.rs"]
mod test_cases;
// Audited CPU intrinsics live only in the private FIR implementation.
#[allow(unsafe_code)]
mod fir;

use std::fmt;

/// Invalid configuration, incomplete interleaved frames, or stream misuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Both rates must be greater than zero.
    ZeroSampleRate,
    /// A stream must have at least one channel.
    ZeroChannels,
    /// A custom length must be odd and between three and the rate-dependent limit.
    InvalidFilterLength {
        /// Requested number of taps per phase.
        requested: usize,
        /// Largest permitted odd tap count for this rate pair.
        maximum: usize,
    },
    /// A block must contain a whole number of interleaved frames.
    IncompleteFrame {
        /// Number of samples in the submitted block.
        samples: usize,
        /// Configured number of interleaved channels per frame.
        channels: usize,
    },
    /// Reset before submitting more input after the final block.
    StreamFinished,
    /// The requested frame/sample count exceeds representable storage or timing.
    SizeOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSampleRate => write!(f, "sample rates must be nonzero"),
            Self::ZeroChannels => write!(f, "channel count must be nonzero"),
            Self::InvalidFilterLength { requested, maximum } => write!(
                f,
                "filter length must be odd and between 3 and {maximum} taps (requested {requested})"
            ),
            Self::IncompleteFrame { samples, channels } => {
                write!(
                    f,
                    "{samples} samples do not form complete {channels}-channel frames"
                )
            }
            Self::StreamFinished => write!(f, "reset the resampler before starting another stream"),
            Self::SizeOverflow => write!(f, "sample count exceeds storage or timing limits"),
        }
    }
}

impl std::error::Error for Error {}

/// Filter support chosen at construction, in input samples per phase.
///
/// Presets adapt to the rate pair and respect the existing tap, coefficient-memory
/// and support-time limits. Inspect [`Resampler::filter_info`] for the actual
/// length and lookahead. Longer filters narrow the transition band but increase
/// computation, memory, lookahead and ringing duration. Length alone is not a
/// quality guarantee. The window, cutoff and coefficient precision stay the same.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterLength {
    /// The base length under the selected [`FilterLengthPolicy`].
    /// With the default generic policy, matches [`Resampler::new`] exactly.
    #[default]
    Standard,
    /// 175% of the standard length, rounded up to odd, subject to resource limits.
    Long,
    /// 200% of the standard length, rounded up to odd, subject to resource limits.
    ExtraLong,
    /// An exact odd tap count of at least three, within the rate-dependent limit.
    /// Invalid requests return [`Error::InvalidFilterLength`]. They are not rounded
    /// or clamped. Shorter custom lengths can reduce alias rejection and bandwidth.
    Custom(usize),
}

/// Rule used to select the Standard length before applying a length preset.
///
/// Both policies share the window, cutoff, coefficient precision, resource limits
/// and SIMD dispatch. [`FilterLength::Custom`] selects the same exact tap count
/// under either policy. The policy stays fixed for the life of the converter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterLengthPolicy {
    /// Common ratio-based support: ceil(1664 * input / min(input, output)),
    /// rounded up to odd and capped. Used by all constructors unless overridden.
    #[default]
    Generic,
    /// Predefined rate-specific minimum lengths with a 512/ratio support floor.
    /// Minima apply to exact phase banks. Interpolated banks use the floor alone.
    /// Long and ExtraLong scale this policy's Standard length. Tap counts still
    /// depend on the rates and resource limits, with different transition/rejection results.
    Fixed,
}

/// Filter storage and centered lookahead for the configured conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterInfo {
    /// Input sample rate in Hz.
    pub input_rate: u32,
    /// Output sample rate in Hz.
    pub output_rate: u32,
    /// Number of interleaved channels per frame.
    pub channels: usize,
    /// Number of fractional phases, excluding the final interpolation row.
    pub phases: usize,
    /// Number of filter coefficients in each phase row.
    pub taps_per_phase: usize,
    /// Future input frames needed by the centered filter, excluding batching.
    /// Final output is time-aligned. This lookahead adds no leading frames.
    pub lookahead_input_frames: usize,
    /// Maximum additional input frames held for the selected SIMD phase batch.
    /// Zero on paths that emit every available output immediately.
    pub additional_batch_input_frames: usize,
    /// Coefficient storage only, excluding history and returned output buffers.
    pub coefficient_bytes: usize,
    /// Whether equal sample rates bypass filtering and coefficient allocation.
    pub bypass: bool,
    /// Compatibility field. This is always false.
    /// All filtered conversions generate high-precision C1 coefficients with
    /// the same portable implementation.
    pub uses_precise_c1_bank: bool,
}

/// A fixed-rate resampler with independent interleaved channels.
///
/// Coefficients and convolution use f64. `process_f32` promotes its input to f64
/// and rounds only the returned samples. Finite PCM input is expected. Values
/// outside [-1, 1] are allowed and are not clipped or normalized.
pub struct Resampler {
    inner: Option<fir::Fir>,
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    filter_length_policy: FilterLengthPolicy,
    input_frames: u64,
    finished: bool,
}

impl Resampler {
    /// Construct a converter. Rates are in Hz. Channels use interleaved storage.
    ///
    /// Reuses a matching cached bank, or generates it at extended precision on
    /// a miss. First construction can take hundreds of milliseconds. Construct
    /// outside a time-critical callback. Nonzero rates beyond the documented
    /// qualification matrix are accepted without a quality claim.
    pub fn new(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self, Error> {
        Self::with_filter_length(input_rate, output_rate, channels, FilterLength::Standard)
    }

    /// Construct a converter with a filter-length preset or exact tap count,
    /// using [`FilterLengthPolicy::Generic`].
    ///
    /// Presets are capped to the rate pair's resource limits. Custom counts are
    /// validated even for equal-rate bypass, which still allocates no filter.
    /// The length is fixed for the stream. [`Self::reset`] retains it. Construct
    /// a new converter to change length. Setup and streaming otherwise match
    /// [`Self::new`].
    ///
    /// ```
    /// use tuneout_resampler::{FilterLength, Resampler};
    /// # fn main() -> Result<(), tuneout_resampler::Error> {
    /// let long = Resampler::with_filter_length(48_000, 44_100, 2, FilterLength::Long)?;
    /// let standard = Resampler::new(48_000, 44_100, 2)?;
    /// assert!(long.filter_info().taps_per_phase >= standard.filter_info().taps_per_phase);
    /// let custom = Resampler::with_filter_length(48_000, 44_100, 2, FilterLength::Custom(4_095))?;
    /// assert_eq!(custom.filter_info().taps_per_phase, 4_095);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_filter_length(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        filter_length: FilterLength,
    ) -> Result<Self, Error> {
        Self::with_filter_length_policy(
            input_rate,
            output_rate,
            channels,
            filter_length,
            FilterLengthPolicy::Generic,
        )
    }

    /// Construct a converter with an explicit length preset and support policy.
    ///
    /// Long and ExtraLong scale the selected policy's Standard length. Custom
    /// counts and resource limits are identical under both policies. Fixed uses
    /// predefined rate-specific minimum lengths. Both policies share the same
    /// coefficient generator and SIMD dispatch.
    /// Validation and stream behavior otherwise match [`Self::with_filter_length`].
    ///
    /// ```
    /// use tuneout_resampler::{FilterLength, FilterLengthPolicy, Resampler};
    /// # fn main() -> Result<(), tuneout_resampler::Error> {
    /// let fixed = Resampler::with_filter_length_policy(
    ///     48_000, 44_100, 2, FilterLength::Standard, FilterLengthPolicy::Fixed,
    /// )?;
    /// assert_eq!(fixed.filter_length_policy(), FilterLengthPolicy::Fixed);
    /// assert_eq!(fixed.filter_info().taps_per_phase, 1_663);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_filter_length_policy(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        filter_length: FilterLength,
        policy: FilterLengthPolicy,
    ) -> Result<Self, Error> {
        if input_rate == 0 || output_rate == 0 {
            return Err(Error::ZeroSampleRate);
        }
        if channels == 0 {
            return Err(Error::ZeroChannels);
        }
        if channels > isize::MAX as usize / (65_537 * 8) {
            return Err(Error::SizeOverflow);
        }
        let design = design::select(input_rate, output_rate, filter_length, policy)?;
        Ok(Self {
            inner: (input_rate != output_rate)
                .then(|| fir::Fir::with_design(channels, input_rate, output_rate, design)),
            input_rate,
            output_rate,
            channels,
            filter_length_policy: policy,
            input_frames: 0,
            finished: false,
        })
    }

    /// Process a block of f64 PCM, retaining precision in the returned samples.
    ///
    /// Output may be empty while collecting lookahead. A final block emits all
    /// remaining frames using the documented endpoint continuation. The complete
    /// stream has `ceil(input_frames * output_rate / input_rate)` output frames.
    /// Invalid blocks leave stream state unchanged. An empty call after finishing
    /// returns an empty vector. Nonempty input requires a reset.
    pub fn process_f64(&mut self, input: &[f64], end_of_stream: bool) -> Result<Vec<f64>, Error> {
        let mut output = Vec::new();
        self.process_f64_into(input, end_of_stream, &mut output)?;
        Ok(output)
    }

    /// Replace `output` with this call's f64 samples, reusing its allocation.
    ///
    /// Stream behavior matches [`Self::process_f64`]. Invalid input leaves both
    /// stream state and `output` unchanged. Valid calls clear its old contents,
    /// including empty calls after finishing. Internal buffers can still grow.
    /// This method does not guarantee allocation-free processing.
    pub fn process_f64_into(
        &mut self,
        input: &[f64],
        end_of_stream: bool,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        let frames = self.validate(input.len())?;
        if self.finished {
            output.clear();
            return Ok(());
        }
        match &mut self.inner {
            Some(inner) => inner.process_f64_into(input, end_of_stream, output),
            None => {
                output.clear();
                output.extend_from_slice(input);
            }
        }
        self.input_frames += frames as u64;
        self.finished = end_of_stream;
        Ok(())
    }

    /// Process f32 PCM with f64 coefficients/accumulation and f32 output.
    /// Stream and final-block behavior match [`Self::process_f64`].
    pub fn process_f32(&mut self, input: &[f32], end_of_stream: bool) -> Result<Vec<f32>, Error> {
        let frames = self.validate(input.len())?;
        if self.finished {
            return Ok(Vec::new());
        }
        let output = match &mut self.inner {
            Some(inner) => inner.process_f32(input, end_of_stream),
            None => input.to_vec(),
        };
        self.input_frames += frames as u64;
        self.finished = end_of_stream;
        Ok(output)
    }

    /// Clear history and timing while retaining the coefficient bank/allocation.
    pub fn reset(&mut self) {
        if let Some(inner) = &mut self.inner {
            inner.reset();
        }
        self.input_frames = 0;
        self.finished = false;
    }

    /// Read the length policy selected at construction, including for bypass.
    /// [`Self::reset`] retains this setting.
    pub fn filter_length_policy(&self) -> FilterLengthPolicy {
        self.filter_length_policy
    }

    /// Read the configured filter size and buffering requirement.
    pub fn filter_info(&self) -> FilterInfo {
        FilterInfo {
            input_rate: self.input_rate,
            output_rate: self.output_rate,
            channels: self.channels,
            phases: self.inner.as_ref().map_or(0, |r| r.phases),
            taps_per_phase: self.inner.as_ref().map_or(0, |r| r.taps),
            lookahead_input_frames: self.inner.as_ref().map_or(0, |r| r.half_taps as usize),
            additional_batch_input_frames: self
                .inner
                .as_ref()
                .map_or(0, |r| r.additional_batch_input_frames()),
            coefficient_bytes: self.coefficients().len() * 8,
            bypass: self.inner.is_none(),
            uses_precise_c1_bank: false,
        }
    }

    /// Read phase-major coefficients for inspection, with one extra final row.
    /// Returns an empty slice for equal-rate bypass. The bank is not mutable.
    pub fn coefficients(&self) -> &[f64] {
        self.inner.as_ref().map_or(&[], |r| r.coeffs.as_slice())
    }

    fn validate(&self, samples: usize) -> Result<usize, Error> {
        if self.finished && samples != 0 {
            return Err(Error::StreamFinished);
        }
        if !samples.is_multiple_of(self.channels) {
            return Err(Error::IncompleteFrame {
                samples,
                channels: self.channels,
            });
        }
        let frames = samples / self.channels;
        let total = self.input_frames as u128 + frames as u128;
        // Leave room for mirrored endpoint indices and a final rational step.
        if total > (i64::MAX as u128 / 2 - u32::MAX as u128 - 65_537) {
            return Err(Error::SizeOverflow);
        }
        let info = self.filter_info();
        let pending =
            info.lookahead_input_frames as u128 + info.additional_batch_input_frames as u128;
        let max_output = ((frames as u128 + pending + 2) * self.output_rate as u128)
            .div_ceil(self.input_rate as u128)
            + 8;
        if max_output * self.channels as u128 > isize::MAX as u128 / 8 {
            return Err(Error::SizeOverflow);
        }
        Ok(frames)
    }
}
