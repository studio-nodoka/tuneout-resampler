//! Buffered causal/anticausal realization of a normalized continuous modal IIR.
//!
//! The recurring filter is IIR in both directions. Fixed internal blocks make
//! the bounded future approximation independent of callers' chunk boundaries.

use crate::bidirectional_design::{self, TwoSidedDesign};
use crate::complex::{Complex, Wide};
use crate::simd::Kernel;
use crate::{startup, Error, Startup};

const BLOCK_FRAMES: usize = 4096;
/// Each independently allocated audio buffer is capped at 64 MiB (f64).
const MAX_BUFFER_SAMPLES: usize = 8_388_608;

/// Diagnostics for the buffered IIR experiment. Operation counts are estimates;
/// lookahead is real buffering latency even though the returned waveform has
/// almost zero phase.
#[derive(Debug, Clone, Copy)]
pub struct BufferedFilterInfo {
    /// Additional finite recursive terms in the generic tail correction.
    pub finite_correction_modes: usize,
    /// Maximum input lag used by the finite correction.
    pub finite_correction_half_support_frames: usize,
    pub input_rate: u32,
    pub output_rate: u32,
    pub channels: usize,
    pub interpolation: u32,
    pub decimation: u32,
    /// Real rational order of one direction (two per complex conjugate mode).
    pub prototype_order: usize,
    /// Combined causal and anticausal rational order.
    pub filter_order: usize,
    pub causal_modes: usize,
    pub future_modes: usize,
    /// The ideal infinite kernel is real and symmetric.
    /// Rounded coefficients and finite future guards introduce small errors.
    pub ideal_zero_phase: bool,
    pub passband_fraction_of_lower_nyquist: f64,
    pub stopband_fraction_of_lower_nyquist: f64,
    /// Filter-design table bytes, excluding the optional private SIMD layout cache.
    pub coefficient_bytes: usize,
    /// Cached phases, at most 512; zero denotes on-demand coefficient generation.
    pub cached_phases: usize,
    /// Reserved persistent audio/state bytes, excluding the caller's output.
    pub working_buffer_bytes: usize,
    /// Elliptic seed target before flatness shaping; not a realized rejection bound.
    pub design_attenuation_db: f64,
    /// Accepted construction constraints (zero means the original elliptic seed).
    pub flatness_moments: usize,
    pub max_flatness_coefficient_change: f64,
    pub max_dc_correction: f64,
    pub multiplications_per_input_sample: f64,
    pub guard_input_frames: usize,
    /// Total mode updates for the future guard, per channel and ordinary block.
    pub guard_mode_frames: usize,
    pub min_guard_input_frames: usize,
    pub block_input_frames: usize,
    /// Number of input frames accumulated before the first ordinary block.
    /// EOS can emit a shorter stream. Includes the block and its future guard.
    pub startup_lookahead_frames: usize,
    /// Worst phase bound on omitted future for input bounded by one. This
    /// excludes floating point error and unknown audio beyond stream endpoints.
    pub guard_error_bound: f64,
    pub bypass: bool,
}

/// Experimental almost zero-phase, strictly IIR conversion using lookahead.
///
/// Unlike [`crate::Resampler`], this converter buffers thousands of frames to
/// realize a symmetric normalized modal kernel with forward and backward IIR
/// states. Retain the same instance across continuous
/// tracks; pass `true` only at the end of the complete stream. Input must contain
/// complete finite interleaved frames. Construction and processing can allocate.
pub struct BufferedResampler {
    design: Option<TwoSidedDesign>,
    kernel: Kernel,
    #[cfg(test)]
    causal_packed_phases: Vec<[f64; 40]>,
    #[cfg(test)]
    future_packed_phases: Vec<[f64; 40]>,
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    startup: Startup,
    states: Vec<Complex>,
    future_states: Vec<Complex>,
    causal_outputs: Vec<Wide>,
    wide_outputs: Vec<Wide>,
    pending: Vec<f64>,
    end_history: Vec<f64>,
    next_phase: u64,
    input_frames: u64,
    output_frames: usize,
    initialized: bool,
    finished: bool,
}

impl BufferedResampler {
    #[cfg(test)]
    pub fn new(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self, Error> {
        Self::with_startup(input_rate, output_rate, channels, Startup::Predict)
    }

    /// Predict estimates both missing endpoints from at most 256 actual frames.
    /// Zero supplies a known silent past and future. Neither recreates missing
    /// original audio, so independent files cannot guarantee arbitrary gapless
    /// continuity without preserving the converter and rational clock.
    #[cfg(test)]
    pub fn with_startup(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        startup: Startup,
    ) -> Result<Self, Error> {
        Self::with_startup_impl(input_rate, output_rate, channels, startup, true)
    }

    /// Original admission and diagnostic model without an unused runtime bank.
    pub(crate) fn with_startup_metadata(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        startup: Startup,
    ) -> Result<Self, Error> {
        Self::with_startup_impl(input_rate, output_rate, channels, startup, false)
    }
    fn with_startup_impl(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        startup: Startup,
        runtime: bool,
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
            Some(if runtime {
                bidirectional_design::make(input_rate, output_rate)?
            } else {
                bidirectional_design::make_metadata(input_rate, output_rate)?
            })
        };
        let causal_modes = design
            .as_ref()
            .filter(|_| runtime)
            .map_or(0, |d| d.causal_poles.len());
        let future_modes = design
            .as_ref()
            .filter(|_| runtime)
            .map_or(0, |d| d.future_poles.len());
        let capacity = design.as_ref().map_or(0, |d| BLOCK_FRAMES + d.guard_frames);
        let samples = capacity.checked_mul(channels).ok_or(Error::SizeOverflow)?;
        if samples > MAX_BUFFER_SAMPLES {
            return Err(Error::ResourceLimit);
        }
        if let Some(d) = &design {
            // Any accepted converter must be able to flush its largest retained
            // window at EOS. Reject impossible expansion/channel combinations
            // here, before accepting audio that could never be drained.
            let eos_samples = (capacity as u128 * u128::from(d.interpolation))
                .div_ceil(u128::from(d.decimation))
                * channels as u128;
            if eos_samples > (MAX_BUFFER_SAMPLES / 2) as u128 {
                return Err(Error::ResourceLimit);
            }
        }
        let mut pending = Vec::new();
        // Metadata-only cores preserve admission checks, but never filter PCM
        // except for the allocation-free equal-rate bypass.
        if runtime {
            pending
                .try_reserve_exact(samples)
                .map_err(|_| Error::SizeOverflow)?;
        }
        let history_capacity = if runtime && startup == Startup::Predict && design.is_some() {
            startup::FRAMES * channels
        } else {
            0
        };
        let kernel = Kernel::detect();
        #[cfg(test)]
        let causal_packed_phases = design
            .as_ref()
            .filter(|_| runtime)
            .map_or_else(Vec::new, |d| {
                kernel.pack_phase_rows(&d.causal_phases, causal_modes, channels)
            });
        #[cfg(test)]
        let future_packed_phases = design
            .as_ref()
            .filter(|_| runtime)
            .map_or_else(Vec::new, |d| {
                kernel.pack_phase_rows(&d.future_phases, future_modes, channels)
            });
        Ok(Self {
            design,
            kernel,
            #[cfg(test)]
            causal_packed_phases,
            #[cfg(test)]
            future_packed_phases,
            input_rate,
            output_rate,
            channels,
            startup,
            states: vec![Complex::default(); causal_modes * channels],
            future_states: vec![Complex::default(); future_modes * channels],
            causal_outputs: Vec::new(),
            wide_outputs: Vec::new(),
            pending,
            end_history: Vec::with_capacity(history_capacity),
            next_phase: 0,
            input_frames: 0,
            output_frames: 0,
            initialized: startup == Startup::Zero,
            finished: false,
        })
    }

    pub fn reset(&mut self) {
        self.wide_outputs.clear();
        self.states.fill(Complex::default());
        self.future_states.fill(Complex::default());
        self.causal_outputs.clear();
        self.pending.clear();
        self.end_history.clear();
        self.next_phase = 0;
        self.input_frames = 0;
        self.output_frames = 0;
        self.initialized = self.startup == Startup::Zero;
        self.finished = false;
    }

    pub fn filter_info(&self) -> BufferedFilterInfo {
        let d = self.design.as_ref();
        let causal = d.map_or(0, |d| d.causal_poles.len());
        let future = d.map_or(0, |d| d.future_poles.len());
        let guard = d.map_or(0, |d| d.guard_frames);
        let guard_mode_frames = d.map_or(0, |d| d.future_guards.iter().sum::<usize>());
        let ratio = f64::from(self.output_rate) / f64::from(self.input_rate);
        BufferedFilterInfo {
            finite_correction_modes: 0,
            finite_correction_half_support_frames: 0,
            input_rate: self.input_rate,
            output_rate: self.output_rate,
            channels: self.channels,
            interpolation: d.map_or(1, |d| d.interpolation),
            decimation: d.map_or(1, |d| d.decimation),
            prototype_order: d.map_or(0, |d| d.rational_order),
            filter_order: d.map_or(0, |d| 2 * d.rational_order),
            causal_modes: causal,
            future_modes: future,
            ideal_zero_phase: false,
            passband_fraction_of_lower_nyquist: d.map_or(1.0, |d| d.passband),
            stopband_fraction_of_lower_nyquist: d
                .map_or(1.0, |_| crate::bidirectional_design::STOPBAND),
            coefficient_bytes: d.map_or(0, |d| {
                (d.causal_poles.len()
                    + d.future_poles.len()
                    + d.causal_phases.len()
                    + d.future_phases.len())
                    * 16
                    + d.direct.len() * 8
                    + d.future_guards.len() * std::mem::size_of::<usize>()
                    + d.model_bytes()
            }),
            cached_phases: d.map_or(0, |d| d.direct.len()),
            working_buffer_bytes: self.pending.capacity() * 8
                + self.end_history.capacity() * 8
                + (self.states.capacity()
                    + self.future_states.capacity()
                    + self.causal_outputs.capacity()
                    + self.wide_outputs.capacity())
                    * 16,
            design_attenuation_db: bidirectional_design::ATTENUATION_DB,
            flatness_moments: d.map_or(0, |d| d.flatness_moments),
            max_flatness_coefficient_change: d.map_or(0.0, |d| d.max_flatness_coefficient_change),
            max_dc_correction: d.map_or(0.0, |d| d.max_dc_correction),
            multiplications_per_input_sample: if d.is_none() {
                0.0
            } else {
                4.0 * (causal + future) as f64
                    + (4.0 * (causal + future) as f64 + 2.0) * ratio
                    + 4.0 * guard_mode_frames as f64 / BLOCK_FRAMES as f64
            },
            guard_input_frames: guard,
            guard_mode_frames,
            min_guard_input_frames: d
                .map_or(0, |d| d.future_guards.iter().copied().min().unwrap_or(0)),
            block_input_frames: if d.is_some() { BLOCK_FRAMES } else { 0 },
            startup_lookahead_frames: if d.is_some() { BLOCK_FRAMES + guard } else { 0 },
            guard_error_bound: d.map_or(0.0, |d| d.guard_bound),
            bypass: d.is_none(),
        }
    }

    /// Continuous normalized model response, excluding sampling images and
    /// bounded future approximation. Frequency is relative to lower Nyquist.
    pub fn frequency_response(&self, relative_frequency: f64) -> (f64, f64) {
        let h = self
            .design
            .as_ref()
            .map_or(Complex::new(1.0, 0.0), |d| d.response(relative_frequency));
        (h.re, h.im)
    }

    pub fn prototype(&self) -> (&[f64], f64) {
        // Kept for source compatibility; this model has no all-pass prototype.
        (&[], 0.0)
    }

    #[cfg(test)]
    pub fn process_f64(&mut self, input: &[f64], end_of_stream: bool) -> Result<Vec<f64>, Error> {
        let mut output = Vec::new();
        self.process_f64_into(input, end_of_stream, &mut output)?;
        Ok(output)
    }

    /// Clear and reuse output. No ordinary output is ready until an internal
    /// block and its future guard have arrived. At EOS exactly ceil(N*L/M)
    /// frames have been returned in total, with no appended IIR tail.
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
        if input.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteInput);
        }
        output.clear();
        self.wide_outputs.clear();
        if self.finished {
            return Ok(());
        }
        let total = self
            .input_frames
            .checked_add((input.len() / self.channels) as u64)
            .ok_or(Error::SizeOverflow)?;
        let target = usize::try_from(
            (u128::from(total) * u128::from(self.output_rate))
                .div_ceil(u128::from(self.input_rate)),
        )
        .map_err(|_| Error::SizeOverflow)?;
        let emitted = match self.design.as_ref() {
            Some(d) if !end_of_stream => {
                // Reserve only output that this call can release. The rational
                // clock starts at next_phase; lookahead stays in pending.
                let available = self.pending.len() / self.channels + input.len() / self.channels;
                let ready = available.saturating_sub(d.guard_frames) / BLOCK_FRAMES * BLOCK_FRAMES;
                if ready == 0 {
                    0
                } else {
                    let ticks = ready as u128 * u128::from(d.interpolation);
                    usize::try_from(
                        ticks
                            .saturating_sub(u128::from(self.next_phase))
                            .div_ceil(u128::from(d.decimation)),
                    )
                    .map_err(|_| Error::SizeOverflow)?
                }
            }
            _ => target - self.output_frames,
        };
        let samples = emitted
            .checked_mul(self.channels)
            .ok_or(Error::SizeOverflow)?;
        if samples > MAX_BUFFER_SAMPLES {
            return Err(Error::ResourceLimit);
        }
        if self.design.is_some() {
            // Wide scratch is twice as large as returned PCM. Reject before
            // touching any stream state, and reserve before filtering begins.
            if samples > MAX_BUFFER_SAMPLES / 2 {
                return Err(Error::ResourceLimit);
            }
            self.causal_outputs.clear();
            self.causal_outputs
                .try_reserve_exact(samples)
                .map_err(|_| Error::SizeOverflow)?;
            self.wide_outputs
                .try_reserve_exact(samples)
                .map_err(|_| Error::SizeOverflow)?;
        }
        output
            .try_reserve_exact(samples)
            .map_err(|_| Error::SizeOverflow)?;
        if let Some(d) = &self.design {
            let window_samples = (BLOCK_FRAMES + d.guard_frames) * self.channels;
            if self.startup == Startup::Predict {
                self.remember_end(input);
            }
            let mut remaining = input;
            while !remaining.is_empty() {
                let take = (window_samples - self.pending.len()).min(remaining.len());
                self.pending.extend_from_slice(&remaining[..take]);
                remaining = &remaining[take..];
                if self.pending.len() == window_samples {
                    self.initialize();
                    self.future_states.fill(Complex::default());
                    self.filter_block(BLOCK_FRAMES, true, output);
                    self.pending.drain(..BLOCK_FRAMES * self.channels);
                }
            }
            if end_of_stream && !self.pending.is_empty() {
                self.initialize();
                self.future_states.fill(Complex::default());
                if self.startup == Startup::Predict {
                    let reversed: Vec<_> = self
                        .end_history
                        .chunks_exact(self.channels)
                        .rev()
                        .flat_map(|f| f.iter().copied())
                        .collect();
                    startup::initialize_lagged(
                        &reversed,
                        self.channels,
                        &self.design.as_ref().unwrap().future_poles,
                        &mut self.future_states,
                    );
                }
                // Every final frame has the same analytic endpoint. One
                // reverse pass reaches exactly the same per-frame states as
                // restarting there and replaying the suffix for each block.
                let frames = self.pending.len() / self.channels;
                self.filter_block(frames, false, output);
                self.pending.clear();
            }
        } else {
            output.extend_from_slice(input);
        }
        self.input_frames = total;
        self.output_frames += output.len() / self.channels;
        self.finished = end_of_stream;
        debug_assert!(self.output_frames <= target);
        debug_assert!(!end_of_stream || self.output_frames == target);
        Ok(())
    }

    fn remember_end(&mut self, input: &[f64]) {
        let capacity = startup::FRAMES * self.channels;
        if input.len() >= capacity {
            self.end_history.clear();
            self.end_history
                .extend_from_slice(&input[input.len() - capacity..]);
        } else {
            let excess = (self.end_history.len() + input.len()).saturating_sub(capacity);
            self.end_history.drain(..excess);
            self.end_history.extend_from_slice(input);
        }
    }

    fn initialize(&mut self) {
        if !self.initialized {
            let leading = self.pending.len().min(startup::FRAMES * self.channels);
            startup::initialize(
                &self.pending[..leading],
                self.channels,
                &self.design.as_ref().unwrap().causal_poles,
                &mut self.states,
            );
            self.initialized = true;
        }
    }

    fn filter_block(&mut self, frames: usize, truncate_future: bool, output: &mut Vec<f64>) {
        match self.channels {
            1 => self.filter_block_channels::<1>(frames, truncate_future, output),
            2 => self.filter_block_channels::<2>(frames, truncate_future, output),
            _ => self.filter_block_channels::<0>(frames, truncate_future, output),
        }
    }

    fn filter_block_channels<const CHANNELS: usize>(
        &mut self,
        frames: usize,
        truncate_future: bool,
        output: &mut Vec<f64>,
    ) {
        let channels = if CHANNELS == 0 {
            self.channels
        } else {
            CHANNELS
        };
        debug_assert_eq!(channels, self.channels);
        let d = self.design.as_ref().unwrap();
        let causal = d.causal_poles.len();
        let future = d.future_poles.len();
        let first_output = output.len();
        // Keep both temporal directions wide until the final output rounding.
        // Storage is reserved and bounded by the public process entry point.
        self.causal_outputs.clear();
        {
            let mut phase_c = [Complex::default(); bidirectional_design::MODE_COUNT];
            let mut phase_low = phase_c;
            for frame in self.pending[..frames * channels].chunks_exact(channels) {
                while self.next_phase < u64::from(d.interpolation) {
                    let phase = self.next_phase as usize;
                    let (row, direct) = if d.tabled() {
                        (
                            &d.causal_phases[phase * causal..(phase + 1) * causal],
                            d.direct[phase],
                        )
                    } else {
                        let direct = d
                            .past_phase(phase as u32, &mut phase_c, &mut phase_low)
                            .value();
                        (&phase_c[..], direct)
                    };
                    for (channel, &sample) in frame.iter().enumerate() {
                        let states = &self.states[channel * causal..(channel + 1) * causal];
                        self.causal_outputs
                            .push(compensated_dot(row, states) + Wide::product(direct, sample));
                        output.push(0.0);
                    }
                    self.next_phase += u64::from(d.decimation);
                }
                self.next_phase -= u64::from(d.interpolation);
                for (channel, &sample) in frame.iter().enumerate() {
                    let states = &mut self.states[channel * causal..(channel + 1) * causal];
                    self.kernel.causal(&d.causal_poles, states, sample);
                }
            }
        }
        let count = (output.len() - first_output) / channels;
        if count == 0 {
            return;
        }
        let mut left = count;
        // End-of-block forward phase gives the last output's offset from the
        // last input frame. Advancing to the preceding frame adds L; consuming
        // an output subtracts M. No per-frame integer division is needed.
        let mut phase =
            self.next_phase as i64 + i64::from(d.interpolation) - i64::from(d.decimation);
        if truncate_future {
            // The uncomputed future starts from zero separately for each mode.
            // All modes still run over every emitted input frame below. Never
            // undo a recurrence by dividing by a pole: that would amplify error.
            let tail = &self.pending[frames * channels..];
            if channels == 1 {
                interleaved_guards::<1>(
                    self.kernel,
                    &d.future_poles,
                    &d.future_guards,
                    tail,
                    &mut self.future_states,
                );
            } else if channels == 2 {
                interleaved_guards::<2>(
                    self.kernel,
                    &d.future_poles,
                    &d.future_guards,
                    tail,
                    &mut self.future_states,
                );
            } else {
                for (mode, (&pole, &guard)) in
                    d.future_poles.iter().zip(&d.future_guards).enumerate()
                {
                    for frame in tail[..guard * channels].chunks_exact(channels).rev() {
                        for (channel, &sample) in frame.iter().enumerate() {
                            let state = &mut self.future_states[channel * future + mode];
                            *state = pole * *state + Complex::new(sample, 0.0);
                        }
                    }
                }
            }
        } else {
            // At EOS the supplied state describes the forecast beyond the
            // actual endpoint. Propagate through every retained actual sample
            // from there; shorter guards must not relocate or drop that state.
            for frame in self.pending[frames * channels..]
                .chunks_exact(channels)
                .rev()
            {
                for (channel, &sample) in frame.iter().enumerate() {
                    let states = &mut self.future_states[channel * future..(channel + 1) * future];
                    self.kernel.future(&d.future_poles, states, sample);
                }
            }
        }
        let mut phase_b = [Complex::default(); bidirectional_design::MODE_COUNT];
        let mut phase_low = phase_b;
        for frame in self.pending[..frames * channels]
            .chunks_exact(channels)
            .rev()
        {
            while left != 0 && phase >= 0 {
                let index = phase as usize;
                let row = if d.tabled() {
                    &d.future_phases[index * future..(index + 1) * future]
                } else {
                    d.future_phase(index as u32, &mut phase_b, &mut phase_low);
                    &phase_b[..]
                };
                left -= 1;
                for channel in 0..channels {
                    let states = &self.future_states[channel * future..(channel + 1) * future];
                    let at = left * channels + channel;
                    self.causal_outputs[at] =
                        self.causal_outputs[at] + compensated_dot(row, states);
                    output[first_output + at] = self.causal_outputs[at].value();
                }
                phase -= i64::from(d.decimation);
            }
            // Future states are always replaced before another block. Once
            // its earliest output is complete, the remaining updates are dead.
            if left == 0 {
                break;
            }
            for (channel, &sample) in frame.iter().enumerate() {
                let states = &mut self.future_states[channel * future..(channel + 1) * future];
                self.kernel.future(&d.future_poles, states, sample);
            }
            phase += i64::from(d.interpolation);
        }
        debug_assert_eq!(left, 0);
        self.wide_outputs.extend_from_slice(&self.causal_outputs);
    }
}

/// Interleave independent modes while keeping each recurrence's exact sample
/// sequence. The longest guard continues across successive shorter guards;
/// no mode starts earlier, skips input, or performs any extra update.
fn interleaved_guards<const CHANNELS: usize>(
    kernel: Kernel,
    poles: &[Complex],
    guards: &[usize],
    tail: &[f64],
    states: &mut [Complex],
) {
    let modes = poles.len();
    let Some((longest, &maximum)) = guards.iter().enumerate().max_by_key(|(_, guard)| *guard)
    else {
        return;
    };
    let pole = poles[longest];
    let mut main: [Complex; CHANNELS] =
        std::array::from_fn(|channel| states[channel * modes + longest]);
    let mut remaining = maximum;
    for (mode, (&other_pole, &guard)) in poles.iter().zip(guards).enumerate() {
        if mode == longest {
            continue;
        }
        let mut other: [Complex; CHANNELS] =
            std::array::from_fn(|channel| states[channel * modes + mode]);
        let shared = remaining.min(guard);
        let main_tail = &tail[(remaining - shared) * CHANNELS..remaining * CHANNELS];
        let other_tail = &tail[(guard - shared) * CHANNELS..guard * CHANNELS];
        if CHANNELS == 2 {
            let mut paired = [[main[0], main[1]], [other[0], other[1]]];
            kernel.guard_pair([pole, other_pole], &mut paired, [main_tail, other_tail]);
            main[0] = paired[0][0];
            main[1] = paired[0][1];
            other[0] = paired[1][0];
            other[1] = paired[1][1];
        } else {
            for (a, b) in main_tail
                .chunks_exact(CHANNELS)
                .rev()
                .zip(other_tail.chunks_exact(CHANNELS).rev())
            {
                for channel in 0..CHANNELS {
                    main[channel] = pole * main[channel] + Complex::new(a[channel], 0.0);
                    other[channel] = other_pole * other[channel] + Complex::new(b[channel], 0.0);
                }
            }
        }
        remaining -= shared;
        for frame in tail[..(guard - shared) * CHANNELS]
            .chunks_exact(CHANNELS)
            .rev()
        {
            for channel in 0..CHANNELS {
                other[channel] = other_pole * other[channel] + Complex::new(frame[channel], 0.0);
            }
        }
        for channel in 0..CHANNELS {
            states[channel * modes + mode] = other[channel];
        }
    }
    for frame in tail[..remaining * CHANNELS].chunks_exact(CHANNELS).rev() {
        for channel in 0..CHANNELS {
            main[channel] = pole * main[channel] + Complex::new(frame[channel], 0.0);
        }
    }
    for channel in 0..CHANNELS {
        states[channel * modes + longest] = main[channel];
    }
}

// FMA captures each product residual; double-double sums keep cancellation
// across modes and across forward/backward contributions. Recurrences retain
// the existing SIMD binary64 kernels.
fn compensated_dot(row: &[Complex], states: &[Complex]) -> Wide {
    debug_assert_eq!(row.len(), states.len());
    let mut total = Wide::from(0.0);
    for (&c, &s) in row.iter().zip(states) {
        total = total + (Wide::product(c.re, s.re) - Wide::product(c.im, s.im));
    }
    total
}

#[cfg(test)]
fn dot(row: &[Complex], states: &[Complex]) -> f64 {
    // Expose independent accumulators without changing any lane's addition
    // order or the final grouping. A reassociated reduction changes audio bits.
    let mut sums = [0.0; 4];
    let mut coefficients = row.chunks_exact(4);
    let mut state_groups = states.chunks_exact(4);
    for (c, s) in coefficients.by_ref().zip(state_groups.by_ref()) {
        sums[0] += c[0].re * s[0].re - c[0].im * s[0].im;
        sums[1] += c[1].re * s[1].re - c[1].im * s[1].im;
        sums[2] += c[2].re * s[2].re - c[2].im * s[2].im;
        sums[3] += c[3].re * s[3].re - c[3].im * s[3].im;
    }
    for (i, (&c, &s)) in coefficients
        .remainder()
        .iter()
        .zip(state_groups.remainder())
        .enumerate()
    {
        sums[i] += c.re * s.re - c.im * s.im;
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_output_retains_product_and_cross_direction_residuals() {
        let a = 1.0 + 2.0_f64.powi(-27);
        assert_eq!(
            compensated_dot(
                &[Complex::new(a, 1.0 + 2.0_f64.powi(-26))],
                &[Complex::new(a, 1.0)]
            )
            .value(),
            2.0_f64.powi(-54)
        );
        let row = [Complex::new(1e16, 0.0), Complex::new(1.0, 0.0)];
        let state = [Complex::new(1.0, 0.0); 2];
        assert_eq!(
            (compensated_dot(&row, &state) + Wide::from(-1e16)).value(),
            1.0
        );
    }

    #[test]
    fn cumulative_frame_overflow_rejects_before_mutation() {
        let mut r = BufferedResampler::new(65537, 65539, 1).unwrap();
        r.input_frames = u64::MAX;
        assert_eq!(
            r.process_f64(&[0.0], true).unwrap_err(),
            Error::SizeOverflow
        );
        assert_eq!(r.input_frames, u64::MAX);
        assert!(!r.finished && r.pending.is_empty());
    }

    #[test]
    fn diagnostics_report_the_effective_design_edge_and_bypass() {
        let mut converter = BufferedResampler::new(96000, 44100, 2).unwrap();
        assert_eq!(
            converter.filter_info().passband_fraction_of_lower_nyquist,
            bidirectional_design::PASSBAND
        );
        // The constructor's fallback test checks its complete restored bank;
        // this separately checks that diagnostics read the selected shape.
        converter.design.as_mut().unwrap().passband = 0.991;
        assert_eq!(
            converter.filter_info().passband_fraction_of_lower_nyquist,
            0.991
        );
        let bypass = BufferedResampler::new(44100, 44100, 2).unwrap();
        assert_eq!(bypass.filter_info().passband_fraction_of_lower_nyquist, 1.0);
    }

    fn check_interleaved_guards<const CHANNELS: usize>(kernel: Kernel) {
        let poles = [
            Complex::new(0.97, 0.1),
            Complex::new(0.0, 0.0),
            Complex::new(-0.8, -0.2),
            Complex::new(0.997, 0.001),
            Complex::new(0.2, -0.0),
        ];
        for guards in [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 3, 999, 0, 1000],
            [1000, 7, 1, 22, 99],
            [3, 1000, 1000, 0, 1000],
            [1, 0, 0, 0, 0],
        ] {
            for kind in 0..3 {
                let input: Vec<_> = (0..1000 * CHANNELS)
                    .map(|i| match kind {
                        0 => {
                            if i % 2 == 0 {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                        1 => (i as f64 % 3.0 - 1.0) * 2.0e-310,
                        _ => ((i * 731 % 997) as f64 - 498.0) / 1994.0,
                    })
                    .collect();
                let mut expected: Vec<_> = (0..CHANNELS * poles.len())
                    .map(|i| Complex::new(if i % 2 == 0 { -0.0 } else { 2.0e-310 }, -0.0))
                    .collect();
                let mut actual = expected.clone();
                for (mode, (&pole, &guard)) in poles.iter().zip(&guards).enumerate() {
                    for frame in input[..guard * CHANNELS].chunks_exact(CHANNELS).rev() {
                        for (channel, &sample) in frame.iter().enumerate() {
                            let state = &mut expected[channel * poles.len() + mode];
                            *state = pole * *state + Complex::new(sample, 0.0);
                        }
                    }
                }
                interleaved_guards::<CHANNELS>(kernel, &poles, &guards, &input, &mut actual);
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits());
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
            }
        }
    }

    #[test]
    fn interleaved_guards_preserve_each_state_exactly() {
        for kernel in crate::simd::test_kernels() {
            check_interleaved_guards::<1>(kernel);
            check_interleaved_guards::<2>(kernel);
        }
    }

    #[test]
    fn simd_matches_scalar_at_every_call_reset_and_endpoint() {
        let kernels = crate::simd::test_kernels();
        for (from, to) in [
            (96000, 44100),
            (44100, 48000),
            (48000, 96000),
            (192000, 44100),
            (48000, 48000),
            (48000, 44099),
            (65537, 65539),
        ] {
            for channels in [1, 2, 6] {
                for startup in [Startup::Zero, Startup::Predict] {
                    let mut expected =
                        BufferedResampler::with_startup(from, to, channels, startup).unwrap();
                    let mut actual =
                        BufferedResampler::with_startup(from, to, channels, startup).unwrap();
                    expected.kernel = kernels[0];
                    let frames =
                        expected.filter_info().startup_lookahead_frames + BLOCK_FRAMES + 17;
                    let input: Vec<_> = (0..frames * channels)
                        .map(|i| ((i * 731 % 997) as f64 - 498.0) / 1994.0)
                        .collect();
                    for &kernel in &kernels {
                        actual.kernel = kernel;
                        for (chunk, inline_eos) in [(257, false), (4096, true)] {
                            expected.reset();
                            actual.reset();
                            let context=format!("{kernel:?}, {from}->{to}, {channels}ch, {startup:?}, chunk {chunk}, inline_eos {inline_eos}");
                            let equal = |a: &[f64], e: &[f64]| {
                                assert_eq!(a.len(), e.len(), "{context}");
                                for (i, (a, e)) in a.iter().zip(e).enumerate() {
                                    assert_eq!(a.to_bits(), e.to_bits(), "{context}, sample {i}");
                                }
                            };
                            for (index, samples) in input.chunks(chunk * channels).enumerate() {
                                let eos = inline_eos && (index + 1) * chunk >= frames;
                                equal(
                                    &actual.process_f64(samples, eos).unwrap(),
                                    &expected.process_f64(samples, eos).unwrap(),
                                );
                                if !eos {
                                    equal(
                                        &actual.process_f64(&[], false).unwrap(),
                                        &expected.process_f64(&[], false).unwrap(),
                                    );
                                }
                            }
                            if !inline_eos {
                                equal(
                                    &actual.process_f64(&[], true).unwrap(),
                                    &expected.process_f64(&[], true).unwrap(),
                                );
                            }
                            assert!(actual.process_f64(&[], true).unwrap().is_empty());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_phase_cache_keeps_design_diagnostics_and_uncached_audio_bits() {
        for (from, to) in [
            (96000, 44100),
            (192000, 44100),
            (44100, 48000),
            (48000, 96000),
        ] {
            let mut cached = BufferedResampler::new(from, to, 2).unwrap();
            let mut plain = BufferedResampler::new(from, to, 2).unwrap();
            plain.causal_packed_phases = Vec::new();
            plain.future_packed_phases = Vec::new();
            assert_eq!(
                cached.filter_info().coefficient_bytes,
                plain.filter_info().coefficient_bytes
            );
            let bytes = (cached.causal_packed_phases.capacity()
                + cached.future_packed_phases.capacity())
                * std::mem::size_of::<[f64; 40]>();
            assert!(bytes <= 1024 * 1024);
            eprintln!(
                "{from}->{to}: design_bytes={}, private_cache_bytes={bytes}",
                cached.filter_info().coefficient_bytes
            );
            let frames = cached.filter_info().startup_lookahead_frames + 8193;
            let input: Vec<_> = (0..frames * 2)
                .map(|i| ((i * 731 % 997) as f64 - 498.0) / 1994.0)
                .collect();
            for chunk in input.chunks(2054) {
                assert_eq!(
                    cached
                        .process_f64(chunk, false)
                        .unwrap()
                        .iter()
                        .map(|s| s.to_bits())
                        .collect::<Vec<_>>(),
                    plain
                        .process_f64(chunk, false)
                        .unwrap()
                        .iter()
                        .map(|s| s.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(
                cached
                    .process_f64(&[], true)
                    .unwrap()
                    .iter()
                    .map(|s| s.to_bits())
                    .collect::<Vec<_>>(),
                plain
                    .process_f64(&[], true)
                    .unwrap()
                    .iter()
                    .map(|s| s.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        for channels in [1, 3, 6] {
            let converter = BufferedResampler::new(192000, 44100, channels).unwrap();
            assert_eq!(converter.causal_packed_phases.capacity(), 0);
            assert_eq!(converter.future_packed_phases.capacity(), 0);
        }
    }

    #[test]
    fn output_reservation_waits_for_ready_blocks() {
        for (input, rate) in [
            (96_000, 44_100),
            (44_100, 48_000),
            (48_000, 96_000),
            (192_000, 44_100),
        ] {
            for startup in [Startup::Predict, Startup::Zero] {
                let mut converter =
                    BufferedResampler::with_startup(input, rate, 2, startup).unwrap();
                let lookahead = converter.filter_info().startup_lookahead_frames;
                let mut output = Vec::new();
                converter
                    .process_f64_into(&vec![0.25; (lookahead - 1) * 2], false, &mut output)
                    .unwrap();
                assert!(output.is_empty());
                assert_eq!(output.capacity(), 0);
                converter
                    .process_f64_into(&[0.25; 2], false, &mut output)
                    .unwrap();
                let expected =
                    (BLOCK_FRAMES as u64 * u64::from(rate)).div_ceil(u64::from(input)) as usize * 2;
                assert_eq!(output.len(), expected);
                let all_pending_output =
                    (lookahead as u64 * u64::from(rate)).div_ceil(u64::from(input)) as usize * 2;
                assert!(output.capacity() < all_pending_output);
            }
        }
    }

    #[test]
    fn unused_endpoint_history_has_no_buffer() {
        for (input, rate, startup) in [
            (96_000, 44_100, Startup::Zero),
            (48_000, 48_000, Startup::Predict),
        ] {
            let mut converter = BufferedResampler::with_startup(input, rate, 2, startup).unwrap();
            assert_eq!(converter.end_history.capacity(), 0);
            converter.process_f64(&vec![0.25; 2048], false).unwrap();
            assert!(converter.end_history.is_empty());
            assert_eq!(converter.end_history.capacity(), 0);
            converter.process_f64(&[], true).unwrap();
            converter.reset();
            assert_eq!(converter.end_history.capacity(), 0);
        }
    }

    #[test]
    fn grouped_dot_preserves_reduction_bits_including_remainders() {
        let coefficients: Vec<_> = (0..65)
            .map(|i| Complex::new((i as f64 * 0.17).sin(), (i as f64 * 0.23).cos()))
            .collect();
        let states: Vec<_> = (0..65)
            .map(|i| Complex::new((i as f64 * 0.41).cos(), (i as f64 * 0.31).sin()))
            .collect();
        for length in 0..=coefficients.len() {
            let mut sums = [0.0; 4];
            for (i, (&c, &s)) in coefficients[..length]
                .iter()
                .zip(&states[..length])
                .enumerate()
            {
                sums[i % 4] += c.re * s.re - c.im * s.im;
            }
            let original = (sums[0] + sums[1]) + (sums[2] + sums[3]);
            assert_eq!(
                dot(&coefficients[..length], &states[..length]).to_bits(),
                original.to_bits()
            );
        }
    }

    #[test]
    fn reverse_clock_matches_timestamps_for_partial_and_empty_blocks() {
        for (input, output_rate) in [(64, 1), (1, 17), (19, 13), (13, 19)] {
            let mut converter = BufferedResampler::new(input, output_rate, 2).unwrap();
            let decimation = converter.design.as_ref().unwrap().decimation;
            for frames in [1, 3, 17] {
                converter.pending = (0..(frames + 13) * 2)
                    .map(|i| 0.7 * (i as f64 * 0.023).sin())
                    .collect();
                converter.next_phase = u64::from(decimation) - 1;
                for (i, state) in converter.future_states.iter_mut().enumerate() {
                    *state = Complex::new(0.3 * (i as f64 + 1.0), -0.17 * i as f64);
                }
                let expected = common_guard_reference(&converter, frames);
                let mut actual = Vec::new();
                converter.filter_block(frames, false, &mut actual);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{input}->{output_rate}, {frames} frames"
                );
                if input == 64 {
                    assert!(actual.is_empty());
                    assert_eq!(converter.next_phase as usize, 63 - frames);
                }
            }
        }
    }

    // Independent block evaluator retaining the original common future guard.
    // Build output timestamps explicitly rather than sharing filter_block's
    // clock, shortened initializer, or reverse output scheduling.
    fn common_guard_reference(converter: &BufferedResampler, frames: usize) -> Vec<f64> {
        let d = converter.design.as_ref().unwrap();
        let channels = converter.channels;
        let causal = d.causal_poles.len();
        let future = d.future_poles.len();
        let mut states = converter.states.clone();
        let mut backward = converter.future_states.clone();
        let mut tick = converter.next_phase as usize;
        let mut positions = Vec::new();
        let mut output = Vec::new();
        let mut wide = Vec::new();
        let mut phase_c = [Complex::default(); bidirectional_design::MODE_COUNT];
        let mut phase_b = phase_c;
        for (n, frame) in converter.pending[..frames * channels]
            .chunks_exact(channels)
            .enumerate()
        {
            while tick / d.interpolation as usize == n {
                let phase = tick % d.interpolation as usize;
                positions.push((n, phase));
                let direct = d.phase(phase as u32, &mut phase_c, &mut phase_b);
                for (channel, &sample) in frame.iter().enumerate() {
                    wide.push(
                        compensated_dot(
                            &phase_c,
                            &states[channel * causal..(channel + 1) * causal],
                        ) + Wide::product(direct, sample),
                    );
                    output.push(0.0);
                }
                tick += d.decimation as usize;
            }
            for (channel, &sample) in frame.iter().enumerate() {
                for (state, &pole) in states[channel * causal..(channel + 1) * causal]
                    .iter_mut()
                    .zip(&d.causal_poles)
                {
                    *state = pole * Complex::new(state.re + sample, state.im);
                }
            }
        }
        let mut left = positions.len();
        for (n, frame) in converter.pending.chunks_exact(channels).enumerate().rev() {
            while left != 0 && positions[left - 1].0 == n {
                left -= 1;
                let phase = positions[left].1;
                d.phase(phase as u32, &mut phase_c, &mut phase_b);
                for channel in 0..channels {
                    output[left * channels + channel] = (wide[left * channels + channel]
                        + compensated_dot(
                            &phase_b,
                            &backward[channel * future..(channel + 1) * future],
                        ))
                    .value();
                }
            }
            for (channel, &sample) in frame.iter().enumerate() {
                for (state, &pole) in backward[channel * future..(channel + 1) * future]
                    .iter_mut()
                    .zip(&d.future_poles)
                {
                    *state = pole * *state + Complex::new(sample, 0.0);
                }
            }
        }
        assert_eq!(left, 0);
        output
    }

    #[test]
    fn shortened_guard_matches_independent_common_guard_blocks() {
        for (input, output_rate) in [
            (96_000, 44_100),
            (44_100, 48_000),
            (48_000, 96_000),
            (192_000, 44_100),
            (48_000, 44_099),
            (65_537, 65_539),
        ] {
            for channels in [1, 3] {
                let mut converter =
                    BufferedResampler::with_startup(input, output_rate, channels, Startup::Zero)
                        .unwrap();
                let info = converter.filter_info();
                let old_mode_frames = info.guard_input_frames * info.future_modes;
                assert!(info.guard_mode_frames * 2 < old_mode_frames);
                assert!(info.min_guard_input_frames < info.guard_input_frames);
                let expected_operations = 4.0 * (info.causal_modes + info.future_modes) as f64
                    + (4.0 * (info.causal_modes + info.future_modes) as f64 + 2.0)
                        * f64::from(output_rate)
                        / f64::from(input)
                    + 4.0 * info.guard_mode_frames as f64 / BLOCK_FRAMES as f64;
                assert!(
                    (info.multiplications_per_input_sample - expected_operations).abs() < 1e-12
                );
                converter.pending = (0..info.startup_lookahead_frames * channels)
                    .map(|i| ((i * 12_137 % 20_027) as f64 - 10_013.0) / 10_013.0)
                    .collect();
                converter.next_phase = u64::from(converter.design.as_ref().unwrap().decimation) - 1;
                for (i, state) in converter.states.iter_mut().enumerate() {
                    *state = Complex::new((i as f64 * 0.37).sin(), (i as f64 * 0.71).cos());
                }
                let expected = common_guard_reference(&converter, BLOCK_FRAMES);
                let mut actual = Vec::new();
                converter.filter_block(BLOCK_FRAMES, true, &mut actual);
                assert_eq!(actual.len(), expected.len());
                let peak_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                // The analytic bound excludes floating point rounding. The
                // extra allowance is only a few binary64 ulps at unit scale.
                assert!(
                    peak_error <= info.guard_error_bound + 8e-15,
                    "{input}->{output_rate}, {channels} channels: {peak_error:e}"
                );
            }
        }
    }

    #[test]
    fn eos_propagates_forecast_from_actual_endpoint_without_shortening() {
        for (input, output_rate) in [(96_000, 44_100), (44_100, 48_000), (48_000, 96_000)] {
            for emitted in [17, BLOCK_FRAMES] {
                let mut converter = BufferedResampler::new(input, output_rate, 2).unwrap();
                let guard = converter.filter_info().guard_input_frames;
                converter.pending = (0..(emitted + guard / 3) * 2)
                    .map(|i| 0.7 * (i as f64 * 0.023).sin())
                    .collect();
                for (i, state) in converter.future_states.iter_mut().enumerate() {
                    *state = Complex::new(0.3 * (i as f64 + 1.0), -0.17 * i as f64);
                }
                let expected = common_guard_reference(&converter, emitted);
                let mut actual = Vec::new();
                converter.filter_block(emitted, false, &mut actual);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[cfg(test)]
mod metadata_setup_tests {
    use super::*;
    #[test]
    fn metadata_constructor_keeps_admission_diagnostics_and_bypass() {
        for (fi, fo, ch) in [
            (44100, 48000, 2),
            (96000, 44100, 2),
            (192000, 44100, 8),
            (48000, 48000, 64),
            (u32::MAX, u32::MAX - 1, 2),
        ] {
            for startup in [Startup::Zero, Startup::Predict] {
                let full = BufferedResampler::with_startup(fi, fo, ch, startup).unwrap();
                let mut meta =
                    BufferedResampler::with_startup_metadata(fi, fo, ch, startup).unwrap();
                let a = full.filter_info();
                let b = meta.filter_info();
                assert_eq!(a.guard_input_frames, b.guard_input_frames);
                assert_eq!(a.guard_mode_frames, b.guard_mode_frames);
                assert_eq!(a.guard_error_bound.to_bits(), b.guard_error_bound.to_bits());
                assert_eq!(a.interpolation, b.interpolation);
                assert_eq!(a.decimation, b.decimation);
                assert_eq!(
                    a.passband_fraction_of_lower_nyquist,
                    b.passband_fraction_of_lower_nyquist
                );
                assert_eq!(a.bypass, b.bypass);
                assert!(b.coefficient_bytes <= a.coefficient_bytes);
                assert!(b.working_buffer_bytes <= a.working_buffer_bytes);
                for nu in [0., 0.2, 0.997, 1., 1.003, 1.8] {
                    assert_eq!(full.frequency_response(nu), meta.frequency_response(nu));
                }
                meta.reset();
                if fi == fo {
                    let input = vec![0.25; ch * 3];
                    assert_eq!(meta.process_f64(&input, true).unwrap(), input);
                }
            }
        }
        for (fi, fo, ch) in [
            (0, 44100, 2),
            (44100, 0, 2),
            (44100, 48000, 0),
            (44100, 48000, 65),
            (1, 64, 64),
            (u32::MAX, 1, 64),
        ] {
            let a = BufferedResampler::with_startup(fi, fo, ch, Startup::Predict).err();
            let b = BufferedResampler::with_startup_metadata(fi, fo, ch, Startup::Predict).err();
            assert!(a.is_some());
            assert_eq!(a, b);
        }
    }
}
