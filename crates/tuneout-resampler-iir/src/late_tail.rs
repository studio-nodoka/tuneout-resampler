//! Multirate finite correction around the normalized modal IIR.
//!
//! Every rate uses the same continuous intervals and residues. With the reduced
//! clock L/M, K=max(L,M), and output phase r, input distances are (jL+r)/K
//! on the past side and (jL-r)/K on the future side. Integer boundary tests
//! adjust canonical finite states by at most one sample at each endpoint.
//! Tiny correction states use f64; the final merge retains the core's Wide sum.
use crate::bidirectional::{BufferedFilterInfo, BufferedResampler as Core};
use crate::complex::{Complex, Wide};
use crate::late_tail_coefficients::{LOWER_END, MODES, OMITTED_L1};
use crate::{startup, Error, Startup};
use std::collections::VecDeque;

const BLOCK: usize = 4096;
// Ordinary PCM and each channel window are capped at 64 MiB. The retained
// interleaved input and Wide output queues are each capped at 128 MiB. These
// larger queues retain the correction's two-sided support, including 48:1
// four-channel and 64:1 mono clocks, without an arbitrary lag-frame cutoff.
const MAX_AUDIO_SAMPLES: usize = 8_388_608;
const MAX_RETAINED_SAMPLES: usize = 16_777_216;
const MAX_WIDE_SAMPLES: usize = 8_388_608;
fn exponential(re: f64, im: f64) -> Complex {
    let radius = re.exp();
    let (sin, cos) = im.sin_cos();
    Complex::new(radius * cos, radius * sin)
}

fn as_usize(value: u128) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::SizeOverflow)
}

fn sample_count(frames: usize, channels: usize, limit: usize) -> Result<usize, Error> {
    let count = frames.checked_mul(channels).ok_or(Error::SizeOverflow)?;
    if count > limit {
        return Err(Error::ResourceLimit);
    }
    Ok(count)
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| Error::SizeOverflow)?;
    Ok(result)
}

fn reserved_queue<T>(capacity: usize) -> Result<VecDeque<T>, Error> {
    let mut result = VecDeque::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| Error::SizeOverflow)?;
    Ok(result)
}

struct MonoWindow {
    origin: Option<i128>,
    samples: Vec<f64>,
    head: usize,
}

/// Buffered sampling of one normalized modal kernel, with its smooth finite
/// correction and compact FIR prefix. Every ratio uses the same model.
///
/// Preserve the instance across continuous tracks and send EOS only at the end
/// of the complete stream. Construction and processing can allocate; substantial
/// lookahead is required. See [`BufferedFilterInfo`] for actual storage and delay.
pub struct InfiniteResampler {
    envelope: Option<crate::core_pipeline::CorePipeline>,
    envelope_channels: Vec<crate::core_pipeline::PipelineState>,
    prefix: Option<crate::prefix::Prefix>,
    prefix_values: Vec<f64>,
    envelope_values: Vec<Wide>,
    prefix_channels: Vec<crate::prefix::PrefixState>,
    prefix_scratch: Vec<f64>,
    core: Core,
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    startup: Startup,
    interpolation: u32,
    decimation: u32,
    end: usize,
    lookahead: usize,
    omitted_tail_bound: f64,
    samples: VecDeque<f64>,
    sample_start: u64,
    past: Vec<f64>,
    future: Vec<f64>,
    window: Vec<f64>,
    rolling_windows: Vec<MonoWindow>,
    mirrored_windows: bool,
    window_head: usize,
    window_append: Vec<f64>,
    #[cfg(test)]
    rolling_reuses: usize,
    #[cfg(test)]
    rolling_rebuilds: usize,
    input_frames: u64,
    output_frames: u64,
    next_block: u64,
    past_ready: bool,
    finished: bool,
    bypass: bool,
}

impl InfiniteResampler {
    /// Construct with predictive estimates of the missing stream endpoints.
    pub fn new(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self, Error> {
        Self::with_startup(input_rate, output_rate, channels, Startup::Predict)
    }

    /// Select predictive or silent endpoints for one continuous stream.
    pub fn with_startup(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        startup: Startup,
    ) -> Result<Self, Error> {
        let core = Core::with_startup_metadata(input_rate, output_rate, channels, startup)?;
        let info = core.filter_info();
        let prefix = if info.bypass {
            None
        } else {
            Some(crate::prefix::Prefix::new(input_rate, output_rate))
        };
        let l = info.interpolation;
        let m = info.decimation;
        let k = l.max(m);
        let end = if info.bypass {
            0
        } else {
            crate::core_pipeline::CorePipeline::lookahead_for(input_rate, output_rate)
                .max(prefix.as_ref().map_or(0, |p| p.radius()))
        };
        let lookahead = end.max(info.guard_input_frames);
        // Continuous L1 alone is not a discrete sampled-tail bound. For any
        // phase, start the geometric envelope at the cutoff itself; alpha<=1.
        let alpha = f64::from(l) / f64::from(k);
        let omitted_tail_bound = if info.bypass {
            0.0
        } else {
            MODES
                .iter()
                .filter(|entry| entry.1 == LOWER_END)
                .map(|&(a, b, sr, _, cr, ci)| {
                    debug_assert!(sr < 0.0);
                    2.0 * alpha * cr.hypot(ci) * (sr * (b - a) as f64).exp()
                        / -((sr * alpha).exp_m1())
                })
                .sum::<f64>()
                .max(OMITTED_L1)
                * (1.0 + 1e-10)
        };
        let (retained, endpoint_count, window_count) = if info.bypass {
            (0, 0, 0)
        } else {
            let retained_frames = end
                .checked_add(lookahead)
                .and_then(|x| x.checked_add(2 * BLOCK))
                .ok_or(Error::SizeOverflow)?;
            let retained = sample_count(retained_frames, channels, MAX_RETAINED_SAMPLES)?;
            let outstanding_frames = lookahead
                .checked_add(2 * BLOCK)
                .ok_or(Error::SizeOverflow)?;
            let pending_frames =
                as_usize((outstanding_frames as u128 * u128::from(l)).div_ceil(u128::from(m)))?;
            sample_count(pending_frames, channels, MAX_WIDE_SAMPLES)?;
            // An accepted stream must be flushable in one EOS call. The largest
            // retained suffix is strictly shorter than lookahead+BLOCK frames.
            let eos_frames =
                as_usize(((lookahead + BLOCK) as u128 * u128::from(l)).div_ceil(u128::from(m)))?;
            sample_count(eos_frames, channels, MAX_AUDIO_SAMPLES)?;
            let endpoint_count = if startup == Startup::Predict {
                sample_count(
                    end.checked_add(1).ok_or(Error::SizeOverflow)?,
                    channels,
                    MAX_AUDIO_SAMPLES,
                )?
            } else {
                0
            };
            let window_count = end
                .checked_mul(2)
                .and_then(|x| x.checked_add(BLOCK))
                .ok_or(Error::SizeOverflow)?;
            if window_count > MAX_AUDIO_SAMPLES {
                return Err(Error::ResourceLimit);
            }
            let block_frames =
                as_usize(((BLOCK + 1) as u128 * u128::from(l)).div_ceil(u128::from(m)))?;
            sample_count(block_frames, channels, MAX_WIDE_SAMPLES)?;
            let core_frames = as_usize(
                ((info.guard_input_frames + BLOCK) as u128 * u128::from(l)).div_ceil(u128::from(m)),
            )?;
            sample_count(core_frames, channels, MAX_AUDIO_SAMPLES / 2)?;
            (retained, endpoint_count, window_count)
        };
        // All cached mono rows together retain the existing 64 MiB audio cap.
        // High-channel/large-support designs keep the admitted bulk fallback.
        let mirrored_windows = window_count != 0
            && window_count
                .checked_mul(channels)
                .and_then(|n| n.checked_mul(2))
                .is_some_and(|n| n <= MAX_AUDIO_SAMPLES);
        let cached_window_count = if mirrored_windows {
            window_count * 2
        } else {
            window_count
        };
        let mut window = reserved(cached_window_count)?;
        let mut rolling_windows = Vec::new();
        if window_count != 0
            && window_count
                .checked_mul(channels)
                .is_some_and(|n| n <= MAX_AUDIO_SAMPLES)
        {
            rolling_windows = reserved(channels)?;
            rolling_windows.push(MonoWindow {
                origin: None,
                samples: std::mem::take(&mut window),
                head: 0,
            });
            for _ in 1..channels {
                rolling_windows.push(MonoWindow {
                    origin: None,
                    samples: reserved(cached_window_count)?,
                    head: 0,
                });
            }
        }
        Ok(Self {
            envelope: if info.bypass {
                None
            } else {
                Some(crate::core_pipeline::CorePipeline::new(
                    input_rate,
                    output_rate,
                ))
            },
            envelope_channels: (0..channels)
                .map(|_| crate::core_pipeline::PipelineState::default())
                .collect(),
            prefix,
            prefix_values: Vec::new(),
            envelope_values: Vec::new(),
            prefix_channels: (0..channels)
                .map(|_| crate::prefix::PrefixState::default())
                .collect(),
            prefix_scratch: Vec::new(),
            core,
            input_rate,
            output_rate,
            channels,
            startup,
            interpolation: l,
            decimation: m,
            end,
            lookahead,
            omitted_tail_bound,
            samples: reserved_queue(retained)?,
            sample_start: 0,
            past: reserved(endpoint_count)?,
            future: reserved(endpoint_count)?,
            window,
            rolling_windows,
            mirrored_windows,
            window_head: 0,
            window_append: Vec::new(),
            #[cfg(test)]
            rolling_reuses: 0,
            #[cfg(test)]
            rolling_rebuilds: 0,
            input_frames: 0,
            output_frames: 0,
            next_block: 0,
            past_ready: false,
            finished: false,
            bypass: info.bypass,
        })
    }

    /// Clear stream history and the sample clock, retaining coefficient storage.
    pub fn reset(&mut self) {
        self.core.reset();
        for state in &mut self.prefix_channels {
            state.reset();
        }
        for state in &mut self.envelope_channels {
            state.reset();
        }
        self.samples.clear();
        self.past.clear();
        self.future.clear();
        self.window.clear();
        for cache in &mut self.rolling_windows {
            cache.origin = None;
            cache.samples.clear();
            cache.head = 0;
        }
        self.window_head = 0;
        self.window_append.clear();
        #[cfg(test)]
        {
            self.rolling_reuses = 0;
            self.rolling_rebuilds = 0;
        }
        self.sample_start = 0;
        self.input_frames = 0;
        self.output_frames = 0;
        self.next_block = 0;
        self.past_ready = false;
        self.finished = false;
    }

    /// Report the realized model's lookahead, estimated cost and storage.
    pub fn filter_info(&self) -> BufferedFilterInfo {
        let mut info = self.core.filter_info();
        if !self.bypass {
            let env = self.envelope.as_ref().unwrap();
            let prefix = self.prefix.as_ref().unwrap();
            let core_modes = env.coarse.infinite_mode_count() + env.coarse.finite_core_mode_count();
            info.causal_modes = core_modes;
            info.future_modes = core_modes;
            info.prototype_order = 2 * core_modes;
            info.filter_order = 4 * core_modes;
            info.finite_correction_modes = env.coarse.finite_correction_mode_count();
            info.finite_correction_half_support_frames = self.end;
            info.coefficient_bytes += env.coefficient_bytes() + prefix.bytes();
            info.cached_phases = prefix.cached_phases();
            info.working_buffer_bytes += self.envelope_channels.capacity()
                * std::mem::size_of::<crate::core_pipeline::PipelineState>()
                + self
                    .envelope_channels
                    .iter()
                    .map(|state| state.bytes())
                    .sum::<usize>()
                + (self.samples.capacity()
                    + self.past.capacity()
                    + self.future.capacity()
                    + self.window.capacity()
                    + self.prefix_values.capacity()
                    + self.prefix_scratch.capacity()
                    + self.window_append.capacity())
                    * 8
                + self.envelope_values.capacity() * std::mem::size_of::<Wide>()
                + self.rolling_windows.capacity() * std::mem::size_of::<MonoWindow>()
                + self
                    .rolling_windows
                    .iter()
                    .map(|cache| cache.samples.capacity() * 8)
                    .sum::<usize>();
            info.multiplications_per_input_sample = env.multiplications_per_input()
                + prefix.taps_per_output() as f64 * f64::from(self.output_rate)
                    / f64::from(self.input_rate);
            info.guard_input_frames = self.lookahead;
            info.startup_lookahead_frames = self.lookahead + BLOCK;
            info.guard_mode_frames = env.future_seed_mode_frames();
            info.min_guard_input_frames = env.min_guard_input_frames();
            info.guard_error_bound = env.omitted_future_bound() + self.omitted_tail_bound;
        }
        info.working_buffer_bytes += self
            .prefix_channels
            .iter()
            .map(|s| s.bytes())
            .sum::<usize>()
            + self.prefix_channels.capacity() * std::mem::size_of::<crate::prefix::PrefixState>();
        info
    }

    pub fn prototype(&self) -> (&[f64], f64) {
        self.core.prototype()
    }

    /// Analytic parent response before the finite phase correction. This is
    /// not the complete converter's magnitude or phase response.
    /// Also excludes sampled images, recurrence rounding and finite guards.
    pub fn frequency_response(&self, nu: f64) -> (f64, f64) {
        if self.bypass {
            return self.core.frequency_response(nu);
        }
        let mut re = Wide::from(0.);
        let mut im = Wide::from(0.);
        for (j, &numerator) in crate::cosine_shape::SHIFT_WEIGHTS.iter().enumerate() {
            let value = self.parent_frequency_response(nu + (j as f64 - 3.) / 12288.);
            let weight = Wide::from(numerator as f64) / Wide::from(640.);
            re = re + Wide::from(value.0) * weight;
            im = im + Wide::from(value.1) * weight;
        }
        re = re + Wide::from(crate::compact_stopband::response(nu));
        (re.value(), im.value())
    }
    fn parent_frequency_response(&self, nu: f64) -> (f64, f64) {
        let (re, im) = self.core.frequency_response(nu);
        if self.bypass {
            return (re, im);
        }
        let w = std::f64::consts::PI * nu;
        let mut sum = Wide::from(re);
        for &(a, b, sr, si, cr, ci) in MODES {
            let c = Complex::new(cr, ci);
            let length = (b - a) as f64;
            for sign in [-1.0, 1.0] {
                let z = Complex::new(sr, si + sign * w);
                let integral = if z.abs() == 0.0 {
                    Complex::new(length, 0.0)
                } else {
                    (exponential(z.re * length, z.im * length) - Complex::new(1.0, 0.0)) / z
                };
                sum = sum + Wide::from((c * exponential(0.0, sign * w * a as f64) * integral).re);
            }
        }
        (sum.value(), im)
    }

    // Split prehistory, actual PCM and predicted suffix once per channel.
    // Inner copies then avoid i128 clocks, ring dispatch and endpoint tests.
    fn fill_window(&mut self, origin: i128, length: usize, channel: usize) {
        self.window.clear();
        self.append_window(origin, length, channel);
    }

    // Same bulk splitter as fill_window, appending only genuinely new samples.
    fn append_window(&mut self, origin: i128, length: usize, channel: usize) {
        fn strided(dst: &mut Vec<f64>, source: &[f64], start: usize, count: usize, stride: usize) {
            if count == 0 {
                return;
            }
            if stride == 1 {
                dst.extend_from_slice(&source[start..start + count]);
            } else {
                dst.extend(source[start..].iter().step_by(stride).take(count).copied());
            }
        }
        let previous_len = self.window.len();
        let mut index = origin;
        let last = origin + length as i128;
        if index < 0 {
            let end = last.min(0);
            let count = (end - index) as usize;
            if self.startup == Startup::Zero {
                self.window.resize(previous_len + count, 0.);
            } else {
                strided(
                    &mut self.window,
                    &self.past,
                    (index + self.end as i128) as usize * self.channels + channel,
                    count,
                    self.channels,
                );
            }
            index = end;
        }
        let end = last.min(i128::from(self.input_frames));
        if index < end {
            debug_assert!(index >= i128::from(self.sample_start));
            let count = (end - index) as usize;
            let start = (index - i128::from(self.sample_start)) as usize * self.channels + channel;
            let (a, b) = self.samples.as_slices();
            let first = if start < a.len() {
                count.min((a.len() - 1 - start) / self.channels + 1)
            } else {
                0
            };
            strided(&mut self.window, a, start, first, self.channels);
            if first < count {
                let start = start + first * self.channels - a.len();
                strided(&mut self.window, b, start, count - first, self.channels);
            }
            index = end;
        }
        if index < last {
            let count = (last - index) as usize;
            if self.startup == Startup::Zero {
                self.window.resize(previous_len + length, 0.);
            } else {
                strided(
                    &mut self.window,
                    &self.future,
                    (index - i128::from(self.input_frames)) as usize * self.channels + channel,
                    count,
                    self.channels,
                );
            }
        }
        assert_eq!(self.window.len(), previous_len + length);
    }

    fn prepare_window(&mut self, origin: i128, length: usize, channel: usize) {
        if self.rolling_windows.is_empty() {
            self.window_head = 0;
            self.fill_window(origin, length, channel);
            return;
        }
        let previous = self.rolling_windows[channel].origin;
        std::mem::swap(&mut self.window, &mut self.rolling_windows[channel].samples);
        std::mem::swap(
            &mut self.window_head,
            &mut self.rolling_windows[channel].head,
        );
        let full_length = 2 * self.end + BLOCK;
        let stored_length = if self.mirrored_windows {
            length * 2
        } else {
            length
        };
        let shift = previous
            .and_then(|previous| origin.checked_sub(previous))
            .and_then(|distance| usize::try_from(distance).ok())
            .filter(|&shift| {
                length == full_length
                    && self.window.len() == stored_length
                    && shift > 0
                    && shift < length
            });
        if let Some(shift) = shift {
            if self.mirrored_windows {
                // Gather only newly admitted frames using the original endpoint
                // splitter. Keep both physical copies identical across wrap.
                let mut ring = std::mem::take(&mut self.window);
                self.window = std::mem::take(&mut self.window_append);
                self.fill_window(origin + (length - shift) as i128, shift, channel);
                let head = self.window_head;
                let first = shift.min(full_length - head);
                let rest = shift - first;
                ring[head..head + first].copy_from_slice(&self.window[..first]);
                ring[head + full_length..head + full_length + first]
                    .copy_from_slice(&self.window[..first]);
                if rest != 0 {
                    ring[..rest].copy_from_slice(&self.window[first..]);
                    ring[full_length..full_length + rest].copy_from_slice(&self.window[first..]);
                }
                self.window_append = std::mem::replace(&mut self.window, ring);
                self.window_head = (head + shift) % full_length;
            } else {
                self.window.copy_within(shift.., 0);
                self.window.truncate(length - shift);
                self.append_window(origin + (length - shift) as i128, shift, channel);
                self.window_head = 0;
            }
            #[cfg(test)]
            {
                self.rolling_reuses += 1;
            }
        } else {
            self.fill_window(origin, length, channel);
            if self.mirrored_windows {
                self.window.extend_from_within(..length);
            }
            self.window_head = 0;
            #[cfg(test)]
            {
                self.rolling_rebuilds += 1;
            }
        }
        self.rolling_windows[channel].origin = (length == full_length).then_some(origin);
    }

    fn release_window(&mut self, channel: usize) {
        if !self.rolling_windows.is_empty() {
            std::mem::swap(&mut self.window, &mut self.rolling_windows[channel].samples);
            std::mem::swap(
                &mut self.window_head,
                &mut self.rolling_windows[channel].head,
            );
        }
    }

    fn endpoints(&mut self, eos: bool) {
        if self.startup == Startup::Zero {
            self.past_ready = true;
            return;
        }
        if !self.past_ready {
            let count = self.input_frames.min(startup::FRAMES as u64) as usize;
            let prefix: Vec<_> = self
                .samples
                .iter()
                .take(count * self.channels)
                .copied()
                .collect();
            let reversed: Vec<_> = prefix
                .chunks_exact(self.channels)
                .rev()
                .flatten()
                .copied()
                .collect();
            let forecast = startup::extend(&reversed, self.channels, self.end);
            self.past.extend(
                forecast
                    .chunks_exact(self.channels)
                    .rev()
                    .flatten()
                    .copied(),
            );
            self.past_ready = true;
        }
        if eos && self.future.is_empty() {
            let count = self.input_frames.min(startup::FRAMES as u64) as usize;
            let mut suffix: Vec<_> = self
                .samples
                .iter()
                .rev()
                .take(count * self.channels)
                .copied()
                .collect();
            suffix.reverse();
            self.future
                .extend(startup::extend(&suffix, self.channels, self.end + 1));
        }
    }

    fn target(&self, frames: u64) -> Result<u64, Error> {
        let target = (u128::from(frames) * u128::from(self.interpolation))
            .div_ceil(u128::from(self.decimation));
        // Match the core's usize output clock even on a 32-bit host.
        let target = as_usize(target)?;
        u64::try_from(target).map_err(|_| Error::SizeOverflow)
    }

    fn block(&mut self, stop: u64, output: &mut Vec<f64>) {
        let start = self.next_block;
        let count = (stop - start) as usize;
        let first = self.output_frames;
        let last = self.target(stop).expect("prevalidated output clock");
        let outputs = (last - first) as usize;
        // Merge directly into this block's caller-owned interleaved output.
        // Each cell is assigned once; no intermediate Wide output buffer is needed.
        let output_start = output.len();
        output.resize(output_start + outputs * self.channels, 0.);
        let origin = i128::from(start) - self.end as i128;
        let length = count + 2 * self.end;
        // Reuse each mono overlap; deinterleave only the new input suffix.
        // Constructor-bounded designs outside the cache cap retain bulk copies.
        let stereo_windows = self.channels == 2 && !self.rolling_windows.is_empty();
        let stereo = stereo_windows && self.prefix.as_ref().unwrap().stereo_cache_eligible();
        if stereo_windows {
            for ch in 0..2 {
                self.prepare_window(origin, length, ch);
                self.release_window(ch);
            }
            let windows = std::array::from_fn(|ch| {
                let cache = &self.rolling_windows[ch];
                &cache.samples[cache.head..cache.head + length]
            });
            if stereo {
                self.prefix_values.resize(outputs * 2, 0.);
                self.prefix.as_ref().unwrap().output_cached_stereo(
                    &mut self.prefix_channels,
                    first,
                    &mut self.prefix_values,
                    windows,
                    origin,
                    &mut self.prefix_scratch,
                );
            }
            self.envelope_values.resize(outputs * 2, Wide::from(0.));
            self.envelope
                .as_ref()
                .expect("non-bypass envelope")
                .output_block_stereo(
                    &mut self.envelope_channels,
                    first,
                    &mut self.envelope_values,
                    windows,
                    origin,
                );
        }
        for ch in 0..self.channels {
            if stereo_windows {
                std::mem::swap(&mut self.window, &mut self.rolling_windows[ch].samples);
                std::mem::swap(&mut self.window_head, &mut self.rolling_windows[ch].head);
            } else {
                self.prepare_window(origin, length, ch);
            }
            if !stereo {
                self.prefix_values.resize(outputs, 0.);
                self.prefix.as_ref().unwrap().output_cached(
                    &mut self.prefix_channels[ch],
                    first,
                    &mut self.prefix_values,
                    &self.window[self.window_head..self.window_head + length],
                    origin,
                    &mut self.prefix_scratch,
                );
            }
            if !stereo_windows {
                let envelope = self.envelope.as_ref().expect("non-bypass envelope");
                self.envelope_values.resize(outputs, Wide::from(0.));
                envelope.output_block(
                    &mut self.envelope_channels[ch],
                    first,
                    &mut self.envelope_values,
                    &self.window[self.window_head..self.window_head + length],
                    origin,
                );
            }
            for i in 0..outputs {
                let prefix_index = if stereo { i * 2 + ch } else { i };
                let envelope_index = if stereo_windows { i * 2 + ch } else { i };
                output[output_start + i * self.channels + ch] = (self.envelope_values
                    [envelope_index]
                    + Wide::from(self.prefix_values[prefix_index]))
                .value();
            }
            self.release_window(ch);
        }
        self.output_frames = last;
        self.next_block = stop;
        let keep = stop.saturating_sub(self.end as u64);
        let drop = keep.saturating_sub(self.sample_start) as usize * self.channels;
        self.samples.drain(..drop);
        self.sample_start = keep;
    }

    fn drain_ready(&mut self, eos: bool, output: &mut Vec<f64>) {
        if self.input_frames == 0 {
            return;
        }
        if !eos && self.input_frames < (self.lookahead + BLOCK) as u64 {
            return;
        }
        self.endpoints(eos);
        while self.next_block < self.input_frames {
            let stop = self
                .next_block
                .saturating_add(BLOCK as u64)
                .min(self.input_frames);
            if !eos
                && (stop - self.next_block != BLOCK as u64
                    || self.input_frames - stop < self.lookahead as u64)
            {
                break;
            }
            self.block(stop, output);
        }
    }

    /// Convert complete interleaved frames. EOS drains the final exact duration.
    pub fn process_f64(&mut self, input: &[f64], eos: bool) -> Result<Vec<f64>, Error> {
        let mut output = Vec::new();
        self.process_f64_into(input, eos, &mut output)?;
        Ok(output)
    }

    /// Input validation and call-sized resource checks precede any stream
    /// mutation. Caller input is ingested in bounded chunks. Empty repeated EOS
    /// is accepted; the returned total is exactly ceil(N*output/input).
    /// Replace `output` with this call's samples, reusing its allocation.
    /// Rejected input leaves stream state and the caller's output unchanged.
    pub fn process_f64_into(
        &mut self,
        input: &[f64],
        eos: bool,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        if self.bypass {
            return self.core.process_f64_into(input, eos, output);
        }
        if !input.len().is_multiple_of(self.channels) {
            return Err(Error::IncompleteFrame);
        }
        if self.finished && (!input.is_empty() || !eos) {
            return Err(Error::StreamFinished);
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteInput);
        }
        if self.finished {
            output.clear();
            return Ok(());
        }
        let frames = u64::try_from(input.len() / self.channels).map_err(|_| Error::SizeOverflow)?;
        let total = self
            .input_frames
            .checked_add(frames)
            .ok_or(Error::SizeOverflow)?;
        let target = self.target(total)?;
        let ready = if eos {
            target
        } else {
            let stop = total.saturating_sub(self.lookahead as u64) / BLOCK as u64 * BLOCK as u64;
            self.target(stop)?
        };
        let emitted = as_usize(u128::from(ready - self.output_frames))?;
        let count = sample_count(emitted, self.channels, MAX_AUDIO_SAMPLES)?;
        output.clear();
        output
            .try_reserve_exact(count)
            .map_err(|_| Error::SizeOverflow)?;
        // Historical constructor admission bounds and the final
        // EOS suffix. Non-finite/call-output failures are therefore atomic even
        // when the caller supplied more input than the retained ring can hold.
        let chunk_samples = BLOCK * self.channels;
        for chunk in input.chunks(chunk_samples) {
            // The original full-rate core is a constructor/metadata reference
            // only. The material core now runs in the coarse complex branch.
            debug_assert!(self.samples.len() + chunk.len() <= self.samples.capacity());
            self.samples.extend(chunk.iter().copied());
            self.input_frames += (chunk.len() / self.channels) as u64;
            self.drain_ready(false, output);
        }
        if eos {
            self.drain_ready(true, output);
        }
        self.finished = eos;
        debug_assert_eq!(self.input_frames, total);
        debug_assert_eq!(output.len(), count);
        debug_assert!(!eos || self.output_frames == target);
        Ok(())
    }

    /// Promote input to f64, run the same converter, and narrow only final PCM.
    pub fn process_f32(&mut self, input: &[f32], eos: bool) -> Result<Vec<f32>, Error> {
        if input.len() > MAX_AUDIO_SAMPLES {
            return Err(Error::ResourceLimit);
        }
        let mut promoted = reserved(input.len())?;
        promoted.extend(input.iter().map(|&value| f64::from(value)));
        Ok(self
            .process_f64(&promoted, eos)?
            .into_iter()
            .map(|value| value as f32)
            .collect())
    }
}

#[cfg(test)]
#[path = "mirrored_window_tests.rs"]
mod mirrored_window_tests;
#[cfg(test)]
#[path = "rolling_window_tests.rs"]
mod rolling_window_tests;

#[cfg(test)]
mod overwritten_scratch_tests {
    use super::*;
    #[test]
    fn block_outputs_ignore_previous_scratch_even_when_poisoned() {
        for (input, output, channels) in [(44100, 48000, 2), (96000, 44100, 1), (48000, 96000, 2)] {
            let mut clean = InfiniteResampler::new(input, output, channels).unwrap();
            let mut poisoned = InfiniteResampler::new(input, output, channels).unwrap();
            let frames = clean.filter_info().startup_lookahead_frames + 8195;
            let samples: Vec<_> = (0..frames * channels)
                .map(|i| ((i * 37 % 127) as f64 - 63.) / 127.)
                .collect();
            for chunk in samples
                .chunks(997 * channels)
                .chain(std::iter::once(&[][..]))
            {
                poisoned.prefix_values.fill(f64::NAN);
                poisoned.envelope_values.fill(Wide::from(f64::NAN));
                let a = clean.process_f64(chunk, chunk.is_empty()).unwrap();
                let b = poisoned.process_f64(chunk, chunk.is_empty()).unwrap();
                assert_eq!(
                    a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
