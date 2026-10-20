//! Infinite: fixed-rate PCM conversion with a generic modal IIR core.
//!
//! [`InfiniteResampler`] partitions one normalized modal kernel into a
//! compact polyphase FIR prefix and a material demodulated IIR tail processed
//! at a lower internal rate. A finite asymmetric correction shapes its magnitude
//! and phase response. It is a mixed FIR/IIR architecture.
//! Ordinary SIMD dots and compensated modal
//! sums share the same normalized model at every supported rational ratio.
//! Cached phase tiles consume future PCM already admitted by the IIR guard;
//! rolling mono windows reuse overlap under a fixed 64 MiB cache budget.
//! Resource limits bound phase caches and audio buffers. Processing still can
//! allocate and uses substantial lookahead; it is not a device-callback API.
//! [`BufferedResampler`] remains a compatibility name for [`InfiniteResampler`].
//! The older [`Resampler`] is a separate causal converter with frequency-dependent
//! delay and its own rate and accuracy limits. [`Timing`] applies only to it.
//!
//! Preserve the same converter across continuous track boundaries. Send EOS
//! only at the end of the complete stream. The final output contains exactly
//! `ceil(input_frames * output_rate / input_rate)` frames.
//!
//! ```
//! use tuneout_resampler_iir::InfiniteResampler;
//! # fn main() -> Result<(), tuneout_resampler_iir::Error> {
//! let mut converter = InfiniteResampler::new(48_000, 44_100, 2)?;
//! let input = vec![0.0_f64; 2 * 4096];
//! let mut output = converter.process_f64(&input, false)?;
//! output.extend(converter.process_f64(&[], true)?);
//! assert_eq!(output.len() / 2, (4096_u64 * 44_100).div_ceil(48_000) as usize);
//! # Ok(())
//! # }
//! ```

#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

mod alignment;
mod bidirectional;
mod bidirectional_design;
#[allow(unsafe_code)]
mod coarse_fft;
mod coarse_iir;
#[cfg(test)]
mod coarse_iir_fir_reference;
mod complex;
mod core_pipeline;
mod cosine_shape;
mod design;
#[cfg(test)]
mod envelope_stream_tests;
#[allow(unsafe_code)]
mod f64_direct_periods;
#[allow(unsafe_code)]
mod f64_dot;
#[allow(unsafe_code)]
mod f64_pairdot;
#[allow(unsafe_code)]
mod f64_phase_tile;
#[allow(unsafe_code)]
mod f64_stereo_analysis;
#[allow(unsafe_code)]
mod f64_stereo_phase_tile;
#[allow(unsafe_code)]
mod f64_stereo_synthesis;
#[allow(unsafe_code)]
mod f64_symmetric_analysis;
mod joint_edge;
mod late_tail;
mod late_tail_coefficients;
#[allow(unsafe_code)]
mod prefix;
#[allow(unsafe_code)]
mod real_carrier;
#[allow(unsafe_code)]
mod setup_kaiser;
#[allow(unsafe_code)]
mod setup_modal;
#[allow(unsafe_code)]
mod setup_prefix;
mod setup_roots;
mod startup;
#[cfg(test)]
mod whole_tests;
#[allow(unsafe_code)]
#[cfg(test)]
mod wide_dot2;
// Audited CPU intrinsics are confined to this private module, as in the FIR.
#[allow(unsafe_code)]
// Historical output SIMD kernels remain as equivalence-test references;
// production buffered output now uses compensated products and sums.
#[allow(dead_code)]
mod simd;

use alignment::Alignment;
use complex::Complex;
use design::Design;
use std::fmt;

pub use bidirectional::BufferedFilterInfo;
pub use late_tail::InfiniteResampler;

/// Compatibility name for [`InfiniteResampler`].
pub type BufferedResampler = InfiniteResampler;

/// Treatment of the missing history before a new stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Startup {
    /// Infer recursive state from up to 256 leading frames. This adds startup
    /// lookahead and is an estimate, not a recovery of unknown original audio.
    Predict,
    /// Start at rest, preserving the causal zero-state impulse response.
    Zero,
}

/// Placement of the filtered waveform on the returned sample timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timing {
    /// Keep the original causal delay and do not predict past EOS.
    Causal,
    /// Match DC delay to an integer with a one-multiply all-pass, then discard
    /// that integer delay. Requires lookahead and bounded endpoint prediction;
    /// higher frequencies still have the prototype's nonlinear phase.
    DcAligned,
}

/// Construction or streaming contract violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Sample rates must be positive.
    ZeroSampleRate,
    /// One through 64 interleaved channels are supported by this prototype.
    InvalidChannels,
    /// The legacy causal Resampler limits each reduced rational factor to 4096.
    RatioTooLarge { interpolation: u32, decimation: u32 },
    /// The supplied block contains an incomplete interleaved frame.
    IncompleteFrame,
    /// Reset before sending input after end of stream.
    StreamFinished,
    /// Input duration or output storage would overflow.
    SizeOverflow,
    /// Buffered lookahead or one call's output exceeds the documented cap.
    ResourceLimit,
    /// Buffered input samples must be finite; state is unchanged on rejection.
    NonFiniteInput,
    /// Coefficient construction did not produce a finite stable design.
    InvalidDesign,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IIR resampler: {self:?}")
    }
}
impl std::error::Error for Error {}

/// Design diagnostics. Attenuation is a design target, not a measured promise.
#[derive(Debug, Clone, Copy)]
pub struct FilterInfo {
    pub input_rate: u32,
    pub output_rate: u32,
    pub channels: usize,
    pub interpolation: u32,
    pub decimation: u32,
    pub filter_order: usize,
    pub recursive_modes: usize,
    /// Runtime coefficient bank, excluding state and prototype diagnostics.
    pub coefficient_bytes: usize,
    pub passband_fraction_of_lower_nyquist: f64,
    pub stopband_fraction_of_lower_nyquist: f64,
    pub design_attenuation_db: f64,
    /// Largest relative per-phase DC calibration; inspect alongside the target.
    pub max_dc_correction: f64,
    /// Upper bound on real scalar multiplications, per input frame per channel.
    /// Ignores compiler optimizations and setup; this is not a CPU benchmark.
    pub multiplications_per_input_sample: f64,
    pub bypass: bool,
    pub startup_lookahead_frames: usize,
    /// Core prototype group delay at DC, in output frames before alignment.
    pub dc_group_delay_frames: f64,
    /// Output frames advanced after fractional all-pass correction, or zero
    /// for causal timing. These frames require future input to become available.
    pub alignment_advance_frames: usize,
}

/// Legacy causal converter with independent channels and exact integer timing.
/// Use [`InfiniteResampler`] for the current Infinite engine.
pub struct Resampler {
    design: Option<Design>,
    states: Vec<Complex>,
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    next_phase: u32,
    input_frames: u64,
    finished: bool,
    startup: Startup,
    initialized: bool,
    prefix: Vec<f64>,
    alignment: Option<Alignment>,
    output_frames: usize,
    tail: Vec<f64>,
    tail_cursor: usize,
    tail_filled: usize,
}

impl Resampler {
    /// Construct outside the audio callback. Equal rates bypass bit-for-bit.
    pub fn new(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self, Error> {
        Self::with_options(
            input_rate,
            output_rate,
            channels,
            Startup::Predict,
            Timing::DcAligned,
        )
    }

    /// Select initialization with the original causal timing. Keep one instance
    /// across continuous track boundaries; initialization runs after new/reset.
    pub fn with_startup(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        startup: Startup,
    ) -> Result<Self, Error> {
        Self::with_options(input_rate, output_rate, channels, startup, Timing::Causal)
    }

    /// Select initialization and timestamp alignment independently. EOS ends
    /// the complete stream; use `false` at continuous track boundaries.
    pub fn with_options(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        startup: Startup,
        timing: Timing,
    ) -> Result<Self, Error> {
        if input_rate == 0 || output_rate == 0 {
            return Err(Error::ZeroSampleRate);
        }
        if !(1..=64).contains(&channels) {
            return Err(Error::InvalidChannels);
        }
        let design = if input_rate == output_rate {
            None
        } else {
            Some(design::make(input_rate, output_rate)?)
        };
        let states =
            vec![Complex::default(); design.as_ref().map_or(0, |d| d.poles.len()) * channels];
        let alignment = design
            .as_ref()
            .filter(|_| timing == Timing::DcAligned)
            .map(|d| Alignment::new(d, channels));
        let tail = if alignment.is_some() {
            vec![0.0; startup::FRAMES * channels]
        } else {
            Vec::new()
        };
        Ok(Self {
            design,
            states,
            input_rate,
            output_rate,
            channels,
            next_phase: 0,
            input_frames: 0,
            finished: false,
            startup,
            initialized: startup == Startup::Zero || input_rate == output_rate,
            prefix: Vec::new(),
            alignment,
            output_frames: 0,
            tail,
            tail_cursor: 0,
            tail_filled: 0,
        })
    }

    /// Reset history and timing while retaining the coefficient bank.
    pub fn reset(&mut self) {
        self.states.fill(Complex::default());
        self.next_phase = 0;
        self.input_frames = 0;
        self.finished = false;
        self.initialized = self.startup == Startup::Zero || self.design.is_none();
        self.prefix.clear();
        if let Some(alignment) = &mut self.alignment {
            alignment.reset();
        }
        self.output_frames = 0;
        self.tail_cursor = 0;
        self.tail_filled = 0;
    }

    /// Inspect the generated design and arithmetic budget.
    pub fn filter_info(&self) -> FilterInfo {
        let modes = self.design.as_ref().map_or(0, |d| d.poles.len());
        let (interpolation, decimation) = self
            .design
            .as_ref()
            .map_or((1, 1), |d| (d.interpolation, d.decimation));
        FilterInfo {
            input_rate: self.input_rate,
            output_rate: self.output_rate,
            channels: self.channels,
            interpolation,
            decimation,
            filter_order: self
                .design
                .as_ref()
                .map_or(0, |d| 2 * d.coefficients.len() + 1),
            recursive_modes: modes,
            coefficient_bytes: self.design.as_ref().map_or(0, |d| {
                (d.poles.len() + d.phases.len()) * 16
                    + d.direct.len() * 8
                    + if self.alignment.is_some() { 8 } else { 0 }
            }),
            passband_fraction_of_lower_nyquist: design::PASSBAND,
            stopband_fraction_of_lower_nyquist: 1.0,
            design_attenuation_db: design::DESIGN_ATTENUATION_DB,
            max_dc_correction: self.design.as_ref().map_or(0.0, |d| d.max_dc_correction),
            multiplications_per_input_sample: if modes == 0 {
                0.0
            } else {
                4.0 * modes as f64
                    + (2.0 * modes as f64 + 1.0 + f64::from(self.alignment.is_some()))
                        * f64::from(self.output_rate)
                        / f64::from(self.input_rate)
            },
            bypass: self.design.is_none(),
            startup_lookahead_frames: if self.startup == Startup::Predict && self.design.is_some() {
                startup::FRAMES
            } else {
                0
            },
            dc_group_delay_frames: self.alignment.as_ref().map_or_else(
                || self.design.as_ref().map_or(0.0, Design::dc_delay),
                |a| a.dc_delay,
            ),
            alignment_advance_frames: self.alignment.as_ref().map_or(0, |a| a.advance_frames),
        }
    }

    /// Analytic prototype response, including phase, at f / lower_Nyquist.
    /// Does not include sample insertion gain or aliases from the rate change.
    /// Also excludes timestamp alignment and the small fixed per-phase DC
    /// calibration of the runtime; this is the core prototype response.
    pub fn frequency_response(&self, relative_frequency: f64) -> (f64, f64) {
        let h = self
            .design
            .as_ref()
            .map_or(Complex::new(1.0, 0.0), |d| d.response(relative_frequency));
        (h.re, h.im)
    }

    /// Prototype coefficients and frequency warp for independent verification.
    pub fn prototype(&self) -> (&[f64], f64) {
        self.design
            .as_ref()
            .map_or((&[], 0.0), |d| (d.coefficients.as_slice(), d.warp))
    }

    /// Filter a frame-aligned block. Finite PCM is expected; no clipping/dither.
    /// Predictive startup buffers up to 256 frames; EOS also flushes shorter
    /// records. Repeated empty final calls are harmless; reset for a new stream.
    pub fn process_f64(&mut self, input: &[f64], end_of_stream: bool) -> Result<Vec<f64>, Error> {
        let mut output = Vec::new();
        self.process_f64_into(input, end_of_stream, &mut output)?;
        Ok(output)
    }

    /// Clear and reuse an output buffer. Retains internal state between calls.
    pub fn process_f64_into(
        &mut self,
        input: &[f64],
        end_of_stream: bool,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        if !input.len().is_multiple_of(self.channels) {
            return Err(Error::IncompleteFrame);
        }
        if self.finished && (!input.is_empty() || !end_of_stream) {
            return Err(Error::StreamFinished);
        }
        output.clear();
        if self.finished {
            return Ok(());
        }
        let frames = input.len() / self.channels;
        let total = self
            .input_frames
            .checked_add(frames as u64)
            .ok_or(Error::SizeOverflow)?;
        let target_frames = usize::try_from(
            (u128::from(total) * u128::from(self.output_rate))
                .div_ceil(u128::from(self.input_rate)),
        )
        .map_err(|_| Error::SizeOverflow)?;
        let samples = (target_frames - self.output_frames)
            .checked_mul(self.channels)
            .ok_or(Error::SizeOverflow)?;
        output
            .try_reserve(samples)
            .map_err(|_| Error::SizeOverflow)?;
        self.remember_tail(input);
        if !self.initialized && !end_of_stream && total < startup::FRAMES as u64 {
            self.prefix
                .try_reserve(input.len())
                .map_err(|_| Error::SizeOverflow)?;
            self.prefix.extend_from_slice(input);
            output.clear();
            self.input_frames = total;
            return Ok(());
        }
        let mut input = input;
        if !self.initialized {
            let needed = (startup::FRAMES * self.channels - self.prefix.len()).min(input.len());
            self.prefix
                .try_reserve(needed)
                .map_err(|_| Error::SizeOverflow)?;
            self.prefix.extend_from_slice(&input[..needed]);
            input = &input[needed..];
            let prefix = std::mem::take(&mut self.prefix);
            if !prefix.is_empty() {
                startup::initialize(
                    &prefix,
                    self.channels,
                    &self.design.as_ref().unwrap().poles,
                    &mut self.states,
                );
                self.filter(&prefix, output);
                if let Some(alignment) = &mut self.alignment {
                    alignment.initialize(output);
                }
            }
            self.initialized = true;
        }
        self.filter(input, output);
        if let Some(alignment) = &mut self.alignment {
            alignment.process(output);
        }
        if end_of_stream && target_frames != 0 && self.alignment.is_some() {
            // ceil(n*L/M) >= target+advance, respecting the current absolute
            // integer clock. Predict only this finite horizon, never a tail
            // duration based on a pole's decay rate.
            let advance = self.alignment.as_ref().unwrap().advance_frames;
            let raw_target = target_frames
                .checked_add(advance)
                .ok_or(Error::SizeOverflow)?;
            let required = ((raw_target as u128 - 1) * u128::from(self.input_rate))
                / u128::from(self.output_rate)
                + 1;
            let future = usize::try_from(required.saturating_sub(u128::from(total)))
                .map_err(|_| Error::SizeOverflow)?;
            let history = self.endpoint_history();
            let predicted = startup::extend(&history, self.channels, future);
            let mut extension = Vec::new();
            self.filter(&predicted, &mut extension);
            self.alignment.as_mut().unwrap().process(&mut extension);
            let needed = samples - output.len();
            debug_assert!(extension.len() >= needed);
            output.extend_from_slice(&extension[..needed]);
        }
        debug_assert!(output.len() <= samples);
        debug_assert!(!end_of_stream || output.len() == samples);
        self.output_frames += output.len() / self.channels;
        self.input_frames = total;
        self.finished |= end_of_stream;
        Ok(())
    }

    fn remember_tail(&mut self, input: &[f64]) {
        let capacity = self.tail.len();
        if capacity == 0 {
            return;
        }
        if input.len() >= capacity {
            self.tail.copy_from_slice(&input[input.len() - capacity..]);
            self.tail_cursor = 0;
            self.tail_filled = capacity;
        } else {
            let first = input.len().min(capacity - self.tail_cursor);
            self.tail[self.tail_cursor..self.tail_cursor + first].copy_from_slice(&input[..first]);
            self.tail[..input.len() - first].copy_from_slice(&input[first..]);
            self.tail_cursor = (self.tail_cursor + input.len()) % capacity;
            self.tail_filled = (self.tail_filled + input.len()).min(capacity);
        }
    }

    fn endpoint_history(&self) -> Vec<f64> {
        if self.tail_filled < self.tail.len() {
            self.tail[..self.tail_filled].to_vec()
        } else {
            self.tail[self.tail_cursor..]
                .iter()
                .chain(&self.tail[..self.tail_cursor])
                .copied()
                .collect()
        }
    }

    fn filter(&mut self, input: &[f64], output: &mut Vec<f64>) {
        if let Some(design) = &self.design {
            let modes = design.poles.len();
            for frame in input.chunks_exact(self.channels) {
                while self.next_phase < design.interpolation {
                    let phase = self.next_phase as usize;
                    let row = &design.phases[phase * modes..(phase + 1) * modes];
                    for (channel, &sample) in frame.iter().enumerate() {
                        let state = &self.states[channel * modes..(channel + 1) * modes];
                        // Four independent sums reduce dependency depth without
                        // requiring unsafe intrinsics or changing the filter.
                        let mut sums = [0.0; 4];
                        for (i, (&coefficient, &s)) in row.iter().zip(state).enumerate() {
                            sums[i % 4] += coefficient.re * s.re - coefficient.im * s.im;
                        }
                        output.push(
                            (sums[0] + sums[1])
                                + (sums[2] + sums[3])
                                + design.direct[phase] * sample,
                        );
                    }
                    self.next_phase += design.decimation;
                }
                self.next_phase -= design.interpolation;
                for (channel, &sample) in frame.iter().enumerate() {
                    let state = &mut self.states[channel * modes..(channel + 1) * modes];
                    for (s, &pole) in state.iter_mut().zip(&design.poles) {
                        *s = pole * Complex::new(s.re + sample, s.im);
                    }
                }
            }
        } else {
            output.extend_from_slice(input);
        }
    }

    /// Promote input to f64 and narrow only the returned output.
    pub fn process_f32(&mut self, input: &[f32], end_of_stream: bool) -> Result<Vec<f32>, Error> {
        let promoted: Vec<_> = input.iter().copied().map(f64::from).collect();
        Ok(self
            .process_f64(&promoted, end_of_stream)?
            .into_iter()
            .map(|v| v as f32)
            .collect())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod bidirectional_tests;

#[cfg(test)]
mod generic_stream_tests;

#[cfg(test)]
mod coarse_iir_reference;

#[cfg(test)]
mod setup_profile_tests {
    use std::time::Instant;
    #[test]
    fn construction_components() {
        for (input, output) in [
            (44100, 48000),
            (48000, 44100),
            (96000, 44100),
            (192000, 44100),
            (48000, 96000),
        ] {
            let start = Instant::now();
            let a = crate::bidirectional::BufferedResampler::new(input, output, 2).unwrap();
            let core = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let b = crate::prefix::Prefix::new(input, output);
            let prefix = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let c = crate::core_pipeline::CorePipeline::new(input, output);
            let pipeline = start.elapsed().as_secs_f64();
            println!(
                "SETUP,{input},{output},{core:.9},{prefix:.9},{pipeline:.9},{},{}",
                b.bytes(),
                c.coefficient_bytes()
            );
            std::hint::black_box((a, b, c));
        }
    }
}

#[allow(unsafe_code)]
mod projected_synthesis;

#[allow(unsafe_code)]
mod real_fft_post_f64;

mod compact_stopband;
#[cfg(test)]
mod compact_stopband_oracle;
