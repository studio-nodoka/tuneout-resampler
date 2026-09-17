//! Streaming FIR state, runtime SIMD dispatch, and finite-stream boundaries.
//!
//! Start with `process_f64_into` or `process_f32` to follow an input block.
//! Four-output kernels consume frames that are already available. The f64 phase
//! tiles can wait for several complete rational periods to reuse coefficients.
//! Endpoint helpers supply continuation samples at the start and final flush.
//!
//! SIMD lanes represent independent outputs/channels. Explicit accumulators and
//! unrolled loads keep code generation predictable. Each output visits taps in
//! order with separately rounded multiplication and addition. Reassociation,
//! horizontal reductions, or fused multiply-add would change the output bits.

use crate::cache::{shared_coefficients, Coefficients};
use crate::design::{gcd, Design};
#[cfg(test)]
use crate::{design::select, FilterLength, FilterLengthPolicy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PhaseBatchBackend {
    Disabled,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl PhaseBatchBackend {
    fn periods(self) -> usize {
        match self {
            Self::Disabled => 0,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => 16,
            #[cfg(target_arch = "x86_64")]
            Self::Avx => 8,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => 8,
        }
    }
}

const MAX_BATCH_MILLISECONDS: u64 = 64;
const MAX_BATCH_SCRATCH_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterpolatedStereoBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "x86_64")]
    Avx,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

pub(crate) struct Fir {
    channels: usize,
    pub(crate) phases: usize,
    pub(crate) taps: usize,
    pub(crate) half_taps: i64,
    pub(crate) coeffs: Coefficients,
    buffer: Vec<f64>,        // Retained input frames, with interleaved channels.
    buffer_start_frame: i64, // Absolute input-frame index of buffer[0].
    clock: Clock,
    // Packed rows contain stereo pairs from successive rational periods:
    // [left_period0, right_period0, left_period1, right_period1, ...].
    phase_scratch: Vec<f64>,
    phase_batch_backend: PhaseBatchBackend,
    interpolated_stereo_backend: InterpolatedStereoBackend,
}

// Position of the next output, measured in input frames:
// source_index + phase_numer / step_den. Each output advances step_num / step_den.
// The reduced integer ratio avoids accumulating a floating-point timing error.
struct Clock {
    step_num: u32,
    step_den: u32,
    source_index: i64,
    phase_numer: u32,
}

impl Fir {
    pub(crate) fn additional_batch_input_frames(&self) -> usize {
        self.phase_batch_backend.periods() * self.clock.step_num as usize
    }

    #[cfg(test)]
    fn new(channels: usize, from_rate: u32, to_rate: u32) -> Self {
        let design = select(
            from_rate,
            to_rate,
            FilterLength::Standard,
            FilterLengthPolicy::Generic,
        )
        .unwrap();
        Self::with_design(channels, from_rate, to_rate, design)
    }

    pub(crate) fn with_design(
        channels: usize,
        from_rate: u32,
        to_rate: u32,
        design: Design,
    ) -> Self {
        let divisor = gcd(from_rate, to_rate);
        Self {
            channels,
            phases: design.phases,
            taps: design.taps,
            half_taps: (design.taps / 2) as i64,
            coeffs: shared_coefficients(from_rate, to_rate, design),
            buffer: Vec::new(),
            phase_scratch: Vec::new(),
            phase_batch_backend: if channels == 2 {
                Self::phase_batch_backend_for_rates(from_rate, to_rate, design)
            } else {
                PhaseBatchBackend::Disabled
            },
            interpolated_stereo_backend: Self::select_interpolated_stereo_backend(),
            buffer_start_frame: 0,
            clock: Clock {
                step_num: from_rate / divisor,
                step_den: to_rate / divisor,
                source_index: 0,
                phase_numer: 0,
            },
        }
    }

    pub(crate) fn reset(&mut self) {
        self.buffer.clear();
        self.buffer_start_frame = 0;
        self.clock.source_index = 0;
        self.clock.phase_numer = 0;
    }

    fn current_position(&self) -> (i64, f64) {
        (
            self.clock.source_index,
            self.clock.phase_numer as f64 / self.clock.step_den as f64,
        )
    }

    fn can_emit(&self, total_frames: i64, end_of_stream: bool) -> bool {
        if end_of_stream {
            self.clock.source_index < total_frames
        } else {
            self.clock.source_index + self.half_taps < total_frames
        }
    }

    fn advance(&mut self) {
        let total = u64::from(self.clock.phase_numer) + u64::from(self.clock.step_num);
        self.clock.source_index += (total / u64::from(self.clock.step_den)) as i64;
        self.clock.phase_numer = (total % u64::from(self.clock.step_den)) as u32;
    }

    fn keep_from_frame(&self, end_of_stream: bool, total_frames: i64) -> i64 {
        if end_of_stream {
            return total_frames;
        }
        (self.clock.source_index - self.half_taps - 1)
            .max(self.buffer_start_frame)
            .min(total_frames)
    }

    fn select_interpolated_stereo_backend() -> InterpolatedStereoBackend {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx") {
                return InterpolatedStereoBackend::Avx;
            }
            return InterpolatedStereoBackend::Sse2;
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return InterpolatedStereoBackend::Neon;
            }
        }
        #[allow(unreachable_code)]
        InterpolatedStereoBackend::Scalar
    }

    // Four adjacent outputs, never a whole rational period. If fewer than four
    // are ready, the caller immediately emits them through its original loop.
    // Keep the interpolation setup out of the existing exact-phase hot loops.
    #[inline(never)]
    fn try_process_interpolated_stereo_batch(
        &mut self,
        total_frames: i64,
        eof: bool,
        padding: Option<(&[f64], i64)>,
    ) -> Option<[[f64; 2]; 4]> {
        if self.channels != 2 {
            return None;
        }
        if self.clock.step_den as usize == self.phases {
            return None;
        }
        let origin = self.clock.source_index;
        let numerator = u64::from(self.clock.phase_numer);
        let num = u64::from(self.clock.step_num);
        let den = u64::from(self.clock.step_den);
        let positions: [u64; 4] = std::array::from_fn(|frame| numerator + frame as u64 * num);
        let last_index = origin + (positions[3] / den) as i64;
        let last_output_unavailable = if eof {
            last_index >= total_frames
        } else {
            last_index + self.half_taps >= total_frames
        };
        if last_output_unavailable {
            return None;
        }
        let (source, start) = padding.unwrap_or((&self.buffer, self.buffer_start_frame));
        let first = origin - self.half_taps;
        if first < start
            || last_index - self.half_taps + self.taps as i64 > start + (source.len() / 2) as i64
        {
            return None;
        }
        let samples: [&[f64]; 4] = std::array::from_fn(|frame| {
            let index = origin + (positions[frame] / den) as i64 - self.half_taps;
            let base = ((index - start) as usize) * 2;
            &source[base..base + self.taps * 2]
        });
        // Keep the original division, multiplication and subtraction order.
        let phases: [(usize, usize, f64); 4] = std::array::from_fn(|frame| {
            let fraction = (positions[frame] % den) as f64 / den as f64;
            let exact = fraction * self.phases as f64;
            let lo = exact.floor() as usize;
            (lo, (lo + 1).min(self.phases), exact - lo as f64)
        });
        let lower = std::array::from_fn(|frame| {
            let phase = phases[frame].0;
            &self.coeffs[phase * self.taps..(phase + 1) * self.taps]
        });
        let upper = std::array::from_fn(|frame| {
            let phase = phases[frame].1;
            &self.coeffs[phase * self.taps..(phase + 1) * self.taps]
        });
        let blend = phases.map(|phase| phase.2);
        let output = match self.interpolated_stereo_backend {
            #[cfg(target_arch = "x86_64")]
            InterpolatedStereoBackend::Avx => {
                // SAFETY: selected once after runtime AVX/OS-state detection.
                unsafe { Self::convolve_interpolated_stereo_avx(samples, lower, upper, blend) }
            }
            #[cfg(target_arch = "x86_64")]
            InterpolatedStereoBackend::Sse2 => {
                Self::convolve_interpolated_stereo_sse2(samples, lower, upper, blend)
            }
            #[cfg(target_arch = "aarch64")]
            InterpolatedStereoBackend::Neon => {
                Self::convolve_interpolated_stereo_neon(samples, lower, upper, blend)
            }
            InterpolatedStereoBackend::Scalar => {
                Self::convolve_interpolated_stereo_scalar(samples, lower, upper, blend)
            }
        };
        for _ in 0..4 {
            self.advance();
        }
        Some(output)
    }

    #[inline]
    fn interpolated_stereo_taps(
        samples: [&[f64]; 4],
        lower: [&[f64]; 4],
        upper: [&[f64]; 4],
    ) -> usize {
        (0..4)
            .map(|i| lower[i].len().min(upper[i].len()).min(samples[i].len() / 2))
            .min()
            .unwrap_or(0)
    }

    fn convolve_interpolated_stereo_scalar(
        samples: [&[f64]; 4],
        lower: [&[f64]; 4],
        upper: [&[f64]; 4],
        blend: [f64; 4],
    ) -> [[f64; 2]; 4] {
        let taps = Self::interpolated_stereo_taps(samples, lower, upper);
        let mut sums = [[0.0; 2]; 4];
        for tap in 0..taps {
            for frame in 0..4 {
                let c0 = lower[frame][tap];
                let coefficient = c0 + (upper[frame][tap] - c0) * blend[frame];
                sums[frame][0] += samples[frame][tap * 2] * coefficient;
                sums[frame][1] += samples[frame][tap * 2 + 1] * coefficient;
            }
        }
        sums
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    fn convolve_interpolated_stereo_sse2(
        samples: [&[f64]; 4],
        lower: [&[f64]; 4],
        upper: [&[f64]; 4],
        blend: [f64; 4],
    ) -> [[f64; 2]; 4] {
        use std::arch::x86_64::*;
        let taps = Self::interpolated_stereo_taps(samples, lower, upper);
        let mut sums = [[0.0; 2]; 4];
        // SAFETY: SSE2 is baseline on x86-64. The common bound covers every
        // coefficient and unaligned stereo load. Each lane retains tap order.
        unsafe {
            let (mut a, mut b, mut c, mut d) = (
                _mm_setzero_pd(),
                _mm_setzero_pd(),
                _mm_setzero_pd(),
                _mm_setzero_pd(),
            );
            let blends = blend.map(|value| _mm_set1_pd(value));
            macro_rules! accumulate {
                ($sum:ident, $frame:expr, $tap:expr) => {{
                    let lo = _mm_set1_pd(*lower[$frame].get_unchecked($tap));
                    let hi = _mm_set1_pd(*upper[$frame].get_unchecked($tap));
                    let weight = _mm_add_pd(lo, _mm_mul_pd(_mm_sub_pd(hi, lo), blends[$frame]));
                    $sum = _mm_add_pd(
                        $sum,
                        _mm_mul_pd(_mm_loadu_pd(samples[$frame].as_ptr().add($tap * 2)), weight),
                    );
                }};
            }
            for tap in 0..taps {
                accumulate!(a, 0, tap);
                accumulate!(b, 1, tap);
                accumulate!(c, 2, tap);
                accumulate!(d, 3, tap);
            }
            _mm_storeu_pd(sums[0].as_mut_ptr(), a);
            _mm_storeu_pd(sums[1].as_mut_ptr(), b);
            _mm_storeu_pd(sums[2].as_mut_ptr(), c);
            _mm_storeu_pd(sums[3].as_mut_ptr(), d);
        }
        sums
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline(never)]
    unsafe fn convolve_interpolated_stereo_avx(
        samples: [&[f64]; 4],
        lower: [&[f64]; 4],
        upper: [&[f64]; 4],
        blend: [f64; 4],
    ) -> [[f64; 2]; 4] {
        use std::arch::x86_64::*;
        let taps = Self::interpolated_stereo_taps(samples, lower, upper);
        let mut sums = [[0.0; 2]; 4];
        // SAFETY: caller detects AVX. The common bound covers all loads. Pack
        // two distinct outputs as [left0,right0,left1,right1], with no reduction.
        unsafe {
            let (mut a, mut b) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let blend01 = _mm256_set_pd(blend[1], blend[1], blend[0], blend[0]);
            let blend23 = _mm256_set_pd(blend[3], blend[3], blend[2], blend[2]);
            macro_rules! accumulate {
                ($sum:ident, $first:expr, $second:expr, $blend:ident, $tap:expr) => {{
                    let lo0 = _mm_loaddup_pd(lower[$first].as_ptr().add($tap));
                    let lo1 = _mm_loaddup_pd(lower[$second].as_ptr().add($tap));
                    let hi0 = _mm_loaddup_pd(upper[$first].as_ptr().add($tap));
                    let hi1 = _mm_loaddup_pd(upper[$second].as_ptr().add($tap));
                    let lo = _mm256_insertf128_pd(_mm256_castpd128_pd256(lo0), lo1, 1);
                    let hi = _mm256_insertf128_pd(_mm256_castpd128_pd256(hi0), hi1, 1);
                    let weight = _mm256_add_pd(lo, _mm256_mul_pd(_mm256_sub_pd(hi, lo), $blend));
                    let input0 = _mm_loadu_pd(samples[$first].as_ptr().add($tap * 2));
                    let input1 = _mm_loadu_pd(samples[$second].as_ptr().add($tap * 2));
                    let input = _mm256_insertf128_pd(_mm256_castpd128_pd256(input0), input1, 1);
                    $sum = _mm256_add_pd($sum, _mm256_mul_pd(input, weight));
                }};
            }
            for tap in 0..taps {
                accumulate!(a, 0, 1, blend01, tap);
                accumulate!(b, 2, 3, blend23, tap);
            }
            _mm_storeu_pd(sums[0].as_mut_ptr(), _mm256_castpd256_pd128(a));
            _mm_storeu_pd(sums[1].as_mut_ptr(), _mm256_extractf128_pd(a, 1));
            _mm_storeu_pd(sums[2].as_mut_ptr(), _mm256_castpd256_pd128(b));
            _mm_storeu_pd(sums[3].as_mut_ptr(), _mm256_extractf128_pd(b, 1));
        }
        sums
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(never)]
    fn convolve_interpolated_stereo_neon(
        samples: [&[f64]; 4],
        lower: [&[f64]; 4],
        upper: [&[f64]; 4],
        blend: [f64; 4],
    ) -> [[f64; 2]; 4] {
        use std::arch::aarch64::*;
        let taps = Self::interpolated_stereo_taps(samples, lower, upper);
        let mut sums = [[0.0; 2]; 4];
        // SAFETY: AArch64 NEON supports f64. The common bound covers all loads.
        unsafe {
            let (mut a, mut b, mut c, mut d) = (
                vdupq_n_f64(0.0),
                vdupq_n_f64(0.0),
                vdupq_n_f64(0.0),
                vdupq_n_f64(0.0),
            );
            let blends = blend.map(|value| vdupq_n_f64(value));
            macro_rules! accumulate {
                ($sum:ident, $frame:expr, $tap:expr) => {{
                    let lo = vdupq_n_f64(*lower[$frame].get_unchecked($tap));
                    let hi = vdupq_n_f64(*upper[$frame].get_unchecked($tap));
                    let weight = vaddq_f64(lo, vmulq_f64(vsubq_f64(hi, lo), blends[$frame]));
                    $sum = vaddq_f64(
                        $sum,
                        vmulq_f64(vld1q_f64(samples[$frame].as_ptr().add($tap * 2)), weight),
                    );
                }};
            }
            for tap in 0..taps {
                accumulate!(a, 0, tap);
                accumulate!(b, 1, tap);
                accumulate!(c, 2, tap);
                accumulate!(d, 3, tap);
            }
            vst1q_f64(sums[0].as_mut_ptr(), a);
            vst1q_f64(sums[1].as_mut_ptr(), b);
            vst1q_f64(sums[2].as_mut_ptr(), c);
            vst1q_f64(sums[3].as_mut_ptr(), d);
        }
        sums
    }

    // Independent output frames can share a tap loop without changing the
    // multiplication or accumulation order of any individual channel.
    fn try_process_stereo_batch(&mut self, total_frames: i64) -> Option<[[f64; 2]; 4]> {
        if self.channels != 2 {
            return None;
        }
        let source_index = self.clock.source_index;
        let phase_numer = u64::from(self.clock.phase_numer);
        let step_num = u64::from(self.clock.step_num);
        let step_den = u64::from(self.clock.step_den);
        if step_den != self.phases as u64 {
            return None;
        }
        let first = source_index - self.half_taps;
        let last = source_index + ((phase_numer + 3 * step_num) / step_den) as i64 - self.half_taps;
        if first < self.buffer_start_frame || last + self.taps as i64 > total_frames {
            return None;
        }
        let sums = self.convolve_exact_stereo_batch(&self.buffer, self.buffer_start_frame);
        for _ in 0..4 {
            self.advance();
        }
        Some(sums)
    }

    #[inline(always)]
    fn convolve_exact_stereo_batch(&self, input: &[f64], input_start: i64) -> [[f64; 2]; 4] {
        let source_index = self.clock.source_index;
        let phase_numer = u64::from(self.clock.phase_numer);
        let step_num = u64::from(self.clock.step_num);
        let step_den = u64::from(self.clock.step_den);
        let samples: [&[f64]; 4] = std::array::from_fn(|frame| {
            let position = phase_numer + frame as u64 * step_num;
            let index = source_index + (position / step_den) as i64 - self.half_taps;
            let base = ((index - input_start) as usize) * 2;
            &input[base..base + self.taps * 2]
        });
        let coeffs: [&[f64]; 4] = std::array::from_fn(|frame| {
            let phase = ((phase_numer + frame as u64 * step_num) % step_den) as usize;
            &self.coeffs[phase * self.taps..(phase + 1) * self.taps]
        });
        #[cfg(target_arch = "x86_64")]
        let sums = Self::convolve_stereo_batch_sse2(samples, coeffs);
        #[cfg(target_arch = "aarch64")]
        let sums = Self::convolve_stereo_batch_neon(samples, coeffs);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let sums = Self::convolve_stereo_batch_scalar(samples, coeffs);
        sums
    }

    // Keep each output's left/right samples in one vector. Letting SLP combine
    // all eight scalar sums can instead generate costly AVX-512 gathers across
    // four coefficient rows. Explicit pairs also preserve the original tap
    // order: no horizontal reduction, reassociation, or fused multiply-add.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn convolve_stereo_batch_sse2(samples: [&[f64]; 4], coeffs: [&[f64]; 4]) -> [[f64; 2]; 4] {
        use std::arch::x86_64::{
            _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_setzero_pd, _mm_storeu_pd,
        };

        // Match the portable zipped iterators, including short or empty slices.
        // These bounds make the unaligned loads below safe independently of the
        // caller. In normal playback all four rows have exactly self.taps taps.
        let taps = (0..4)
            .map(|i| coeffs[i].len().min(samples[i].len() / 2))
            .min()
            .unwrap_or(0);
        let mut sums = [[0.0f64; 2]; 4];
        // SAFETY: SSE2 is part of the x86-64 baseline. For every tap, each
        // coefficient and both input samples are within the lengths checked
        // above. Each destination contains two f64 values. Unaligned loads and
        // stores impose no additional alignment requirement.
        unsafe {
            let mut a = _mm_setzero_pd();
            let mut b = _mm_setzero_pd();
            let mut c = _mm_setzero_pd();
            let mut d = _mm_setzero_pd();
            for tap in 0..taps {
                let offset = tap * 2;
                macro_rules! accumulate {
                    ($sum:ident, $frame:expr) => {
                        $sum = _mm_add_pd(
                            $sum,
                            _mm_mul_pd(
                                _mm_loadu_pd(samples[$frame].as_ptr().add(offset)),
                                _mm_set1_pd(*coeffs[$frame].get_unchecked(tap)),
                            ),
                        );
                    };
                }
                accumulate!(a, 0);
                accumulate!(b, 1);
                accumulate!(c, 2);
                accumulate!(d, 3);
            }
            _mm_storeu_pd(sums[0].as_mut_ptr(), a);
            _mm_storeu_pd(sums[1].as_mut_ptr(), b);
            _mm_storeu_pd(sums[2].as_mut_ptr(), c);
            _mm_storeu_pd(sums[3].as_mut_ptr(), d);
        }
        sums
    }

    // AArch64 desktop targets provide NEON as a baseline. Keep left/right in
    // independent lanes, with four output frames sharing the ordered tap loop.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn convolve_stereo_batch_neon(samples: [&[f64]; 4], coeffs: [&[f64]; 4]) -> [[f64; 2]; 4] {
        use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

        let taps = (0..4)
            .map(|i| coeffs[i].len().min(samples[i].len() / 2))
            .min()
            .unwrap_or(0);
        let mut sums = [[0.0f64; 2]; 4];
        // SAFETY: each load covers one stereo pair within the checked slices.
        // Each destination holds two doubles. No alignment is required. Keep
        // multiplication/addition separate to preserve the scalar rounding.
        unsafe {
            let mut a = vdupq_n_f64(0.0);
            let mut b = vdupq_n_f64(0.0);
            let mut c = vdupq_n_f64(0.0);
            let mut d = vdupq_n_f64(0.0);
            for tap in 0..taps {
                let offset = tap * 2;
                macro_rules! accumulate {
                    ($sum:ident, $frame:expr) => {
                        $sum = vaddq_f64(
                            $sum,
                            vmulq_f64(
                                vld1q_f64(samples[$frame].as_ptr().add(offset)),
                                vdupq_n_f64(*coeffs[$frame].get_unchecked(tap)),
                            ),
                        );
                    };
                }
                accumulate!(a, 0);
                accumulate!(b, 1);
                accumulate!(c, 2);
                accumulate!(d, 3);
            }
            vst1q_f64(sums[0].as_mut_ptr(), a);
            vst1q_f64(sums[1].as_mut_ptr(), b);
            vst1q_f64(sums[2].as_mut_ptr(), c);
            vst1q_f64(sums[3].as_mut_ptr(), d);
        }
        sums
    }

    #[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
    #[inline]
    fn convolve_stereo_batch_scalar(samples: [&[f64]; 4], coeffs: [&[f64]; 4]) -> [[f64; 2]; 4] {
        let mut sums = [[0.0f64; 2]; 4];
        let a = samples[0].as_chunks::<2>().0.iter().zip(coeffs[0]);
        let b = samples[1].as_chunks::<2>().0.iter().zip(coeffs[1]);
        let c = samples[2].as_chunks::<2>().0.iter().zip(coeffs[2]);
        let d = samples[3].as_chunks::<2>().0.iter().zip(coeffs[3]);
        for (((s0, &c0), (s1, &c1)), ((s2, &c2), (s3, &c3))) in a.zip(b).zip(c.zip(d)) {
            sums[0][0] += s0[0] * c0;
            sums[0][1] += s0[1] * c0;
            sums[1][0] += s1[0] * c1;
            sums[1][1] += s1[1] * c1;
            sums[2][0] += s2[0] * c2;
            sums[2][1] += s2[1] * c2;
            sums[3][0] += s3[0] * c3;
            sums[3][1] += s3[1] * c3;
        }
        sums
    }

    // The buffer is unchanged during one process call, so each channel's
    // tail estimate can be shared by endpoint padding and fallback frames.
    fn tail_blends<'a>(&self, total_frames: i64, cache: &'a mut Vec<f64>) -> &'a [f64] {
        if cache.is_empty() {
            cache.extend((0..self.channels).map(|ch| {
                endpoint_tail_odd_reflection_blend(
                    &self.buffer,
                    self.channels,
                    self.buffer_start_frame,
                    total_frames,
                    ch,
                )
            }));
        }
        cache
    }

    // Keep ordered f64 arithmetic shared. Only the caller's final cast differs.
    #[inline]
    fn emit_fallback_frame(
        &self,
        total_frames: i64,
        end_of_stream: bool,
        tail_blends: &mut Vec<f64>,
        mut emit: impl FnMut(f64),
    ) {
        let (idx, frac) = self.current_position();
        let (phase0, phase1, phase_blend) = if self.clock.step_den as usize == self.phases {
            let phase = (self.clock.phase_numer as usize).min(self.phases);
            (phase, phase, 0.0)
        } else {
            let exact_phase = frac * self.phases as f64;
            let phase0 = exact_phase.floor() as usize;
            let phase1 = (phase0 + 1).min(self.phases);
            (phase0, phase1, exact_phase - phase0 as f64)
        };
        let phase0_coeffs = &self.coeffs[phase0 * self.taps..(phase0 + 1) * self.taps];
        let phase1_coeffs = &self.coeffs[phase1 * self.taps..(phase1 + 1) * self.taps];

        let first_sample_frame = idx - self.half_taps;
        let kernel_fully_buffered = first_sample_frame >= self.buffer_start_frame
            && first_sample_frame + self.taps as i64 <= total_frames;

        if kernel_fully_buffered && self.channels == 2 {
            let mut left = 0.0f64;
            let mut right = 0.0f64;
            let sample_base =
                ((first_sample_frame - self.buffer_start_frame) as usize) * self.channels;
            let samples = &self.buffer[sample_base..sample_base + self.taps * 2];
            if phase0 == phase1 {
                for (frame, &coeff) in samples.as_chunks::<2>().0.iter().zip(phase0_coeffs) {
                    left += frame[0] * coeff;
                    right += frame[1] * coeff;
                }
            } else {
                for ((frame, &coeff0), &coeff1) in samples
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(phase0_coeffs)
                    .zip(phase1_coeffs)
                {
                    let coeff = coeff0 + (coeff1 - coeff0) * phase_blend;
                    left += frame[0] * coeff;
                    right += frame[1] * coeff;
                }
            }
            emit(left);
            emit(right);
        } else if kernel_fully_buffered {
            let base = ((first_sample_frame - self.buffer_start_frame) as usize) * self.channels;
            for ch in 0..self.channels {
                let mut acc = 0.0f64;
                let mut sample_index = base + ch;
                if phase0 == phase1 {
                    for &coeff in phase0_coeffs {
                        acc += self.buffer[sample_index] * coeff;
                        sample_index += self.channels;
                    }
                } else {
                    for tap in 0..self.taps {
                        let coeff0 = phase0_coeffs[tap];
                        let coeff1 = phase1_coeffs[tap];
                        let coeff = coeff0 + (coeff1 - coeff0) * phase_blend;
                        acc += self.buffer[sample_index] * coeff;
                        sample_index += self.channels;
                    }
                }
                emit(acc);
            }
        } else {
            for ch in 0..self.channels {
                let tail_blend = if end_of_stream {
                    self.tail_blends(total_frames, tail_blends)[ch]
                } else {
                    0.0
                };
                let mut acc = 0.0f64;
                for tap in 0..self.taps {
                    let sample_frame = idx + tap as i64 - self.half_taps;
                    let coeff0 = phase0_coeffs[tap];
                    let coeff1 = phase1_coeffs[tap];
                    let coeff = coeff0 + (coeff1 - coeff0) * phase_blend;
                    acc += endpoint_continued_sample_f64(
                        &self.buffer,
                        self.channels,
                        self.buffer_start_frame,
                        total_frames,
                        sample_frame,
                        ch,
                        tail_blend,
                    ) * coeff;
                }
                emit(acc);
            }
        }
    }

    fn estimated_output_frames(&self, input_frames: usize) -> usize {
        let step_num = u128::from(self.clock.step_num);
        let step_den = u128::from(self.clock.step_den);
        ((input_frames as u128 * step_den).div_ceil(step_num)) as usize + 4
    }

    fn discard_consumed_input(&mut self, end_of_stream: bool, total_frames: i64) {
        let keep_from = self.keep_from_frame(end_of_stream, total_frames);
        let drop_frames = (keep_from - self.buffer_start_frame).max(0) as usize;
        if drop_frames != 0 {
            let samples = drop_frames * self.channels;
            if samples >= self.buffer.len() {
                self.buffer.clear();
            } else {
                self.buffer.drain(..samples);
            }
        }
        self.buffer_start_frame = keep_from;
    }

    pub(crate) fn process_f32(&mut self, input: &[f32], end_of_stream: bool) -> Vec<f32> {
        debug_assert_eq!(input.len() % self.channels, 0);
        self.buffer
            .extend(input.iter().map(|&sample| sample as f64));

        let input_frames = input.len() / self.channels;
        let estimated_output_frames = self.estimated_output_frames(input_frames);
        let total_frames = self.buffer_start_frame + (self.buffer.len() / self.channels) as i64;
        let mut output = Vec::with_capacity(estimated_output_frames * self.channels);
        let mut tail_blends = Vec::new();

        while self.can_emit(total_frames, end_of_stream) {
            if let Some(frames) = self.try_process_stereo_batch(total_frames) {
                output.extend(frames.into_iter().flatten().map(|sample| sample as f32));
                continue;
            }
            if let Some(frames) =
                self.try_process_interpolated_stereo_batch(total_frames, end_of_stream, None)
            {
                output.extend(frames.into_iter().flatten().map(|sample| sample as f32));
                continue;
            }

            self.emit_fallback_frame(total_frames, end_of_stream, &mut tail_blends, |sample| {
                output.push(sample as f32);
            });

            self.advance();
        }

        self.discard_consumed_input(end_of_stream, total_frames);

        output
    }

    // Select once when constructing the resampler. No CPU vendor/model tuning
    // or host-native compiler flags are needed by any of these backends.
    fn select_phase_batch_backend() -> PhaseBatchBackend {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return PhaseBatchBackend::Avx512;
            }
            if std::is_x86_feature_detected!("avx") {
                return PhaseBatchBackend::Avx;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return PhaseBatchBackend::Neon;
            }
        }
        PhaseBatchBackend::Disabled
    }

    fn phase_batch_backend_for_rates(
        from_rate: u32,
        to_rate: u32,
        design: Design,
    ) -> PhaseBatchBackend {
        let divisor = gcd(from_rate, to_rate);
        // Packing must be reused across at least a complete phase group:
        // two phases for AVX/NEON and four for AVX-512. A single-phase ratio
        // uses the immediate kernel, avoiding a repack for every small tile.
        if design.phases < 2 || design.phases != (to_rate / divisor) as usize {
            return PhaseBatchBackend::Disabled;
        }
        // Apply the same latency and scratch budgets to every exact ratio.
        // The packed row count is at most taps + the reduced input numerator.
        let num = u64::from(from_rate / divisor);
        let fits = |backend: PhaseBatchBackend| {
            let periods = backend.periods() as u64;
            design.phases as u64 >= periods / 4
                && num * periods * 1000 <= u64::from(from_rate) * MAX_BATCH_MILLISECONDS
                && (design.taps as u64 + num) * periods * 2 * 8 + 64 <= MAX_BATCH_SCRATCH_BYTES
        };
        let backend = Self::select_phase_batch_backend();
        if fits(backend) {
            return backend;
        }
        #[cfg(target_arch = "x86_64")]
        if backend == PhaseBatchBackend::Avx512
            && std::is_x86_feature_detected!("avx")
            && fits(PhaseBatchBackend::Avx)
        {
            return PhaseBatchBackend::Avx;
        }
        PhaseBatchBackend::Disabled
    }

    // None falls back. Some(0) waits. Some(n) appends n samples and advances.
    fn try_process_phase_tile(
        &mut self,
        total_frames: i64,
        eof: bool,
        output: &mut Vec<f64>,
    ) -> Option<usize> {
        match self.phase_batch_backend {
            PhaseBatchBackend::Disabled => None,
            // SAFETY: construction selects a backend only after detecting its
            // instruction set. Tests also check availability before forcing one.
            #[cfg(target_arch = "x86_64")]
            PhaseBatchBackend::Avx512 => unsafe {
                self.try_process_phase_tile_avx512(total_frames, eof, output)
            },
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            _ => self.try_process_phase_tile_narrow(total_frames, eof, output),
        }
    }

    // A rational phase repeats after step_den outputs and step_num inputs.
    // Filter length is dynamic. Scratch extents and loads use self.taps.
    // Sixteen periods share coefficient loads. Every SIMD lane is a distinct
    // output/channel, accumulated in ascending tap order with separate mul/add.
    // The additional batching bound is sixteen rational periods of input.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[inline(never)]
    unsafe fn try_process_phase_tile_avx512(
        &mut self,
        total_frames: i64,
        eof: bool,
        output: &mut Vec<f64>,
    ) -> Option<usize> {
        if self.channels != 2 || self.taps == 0 {
            return None;
        }
        if self.phases != self.clock.step_den as usize {
            return None;
        }
        let origin = self.clock.source_index;
        let numerator = self.clock.phase_numer as usize;
        let num = self.clock.step_num as usize;
        let den = self.clock.step_den as usize;
        if origin - self.half_taps < self.buffer_start_frame {
            return None;
        }
        let maximum = total_frames + self.half_taps - self.taps as i64;
        if maximum < origin {
            return None;
        }
        let available = (((maximum - origin + 1) as usize * den - 1 - numerator) / num) + 1;

        let cycles = 16usize;
        if available < cycles * den {
            return if eof { None } else { Some(0) };
        }
        use std::arch::x86_64::*;
        let first = (origin - self.half_taps - self.buffer_start_frame) as usize;
        let rows = self.taps + (numerator + (den - 1) * num) / den;
        let stride = cycles * 2;
        let needed = rows * stride + 8;
        if self.phase_scratch.len() < needed {
            self.phase_scratch.resize(needed, 0.0);
        }
        let ptr = self.phase_scratch.as_mut_ptr().cast::<f64>();
        let shift = (64 - (ptr as usize) % 64) % 64 / 8;
        let ptr = unsafe { ptr.add(shift) };
        // The last packed row/cycle is the furthest read. All vector reads
        // cover four complete rows, and the tail uses individual stereo pairs.
        assert!((first + rows - 1 + (cycles - 1) * num + 1) * 2 <= self.buffer.len());
        // SAFETY: the caller detects AVX-512F once at construction. The scratch
        // allocation includes eight doubles of alignment room. The checked source
        // extent covers every packed row and cycle. Every row is fully initialized.
        unsafe {
            let bulk = rows / 4 * 4;
            for row in (0..bulk).step_by(4) {
                for c in (0..cycles).step_by(4) {
                    let source = self.buffer.as_ptr().add((first + row + c * num) * 2);
                    let a = _mm512_loadu_pd(source);
                    let b = _mm512_loadu_pd(source.add(num * 2));
                    let c0 = _mm512_loadu_pd(source.add(num * 4));
                    let d = _mm512_loadu_pd(source.add(num * 6));
                    let ab0 = _mm512_shuffle_f64x2::<0x44>(a, b);
                    let ab1 = _mm512_shuffle_f64x2::<0xee>(a, b);
                    let cd0 = _mm512_shuffle_f64x2::<0x44>(c0, d);
                    let cd1 = _mm512_shuffle_f64x2::<0xee>(c0, d);
                    let r0 = _mm512_shuffle_f64x2::<0x88>(ab0, cd0);
                    let r1 = _mm512_shuffle_f64x2::<0xdd>(ab0, cd0);
                    let r2 = _mm512_shuffle_f64x2::<0x88>(ab1, cd1);
                    let r3 = _mm512_shuffle_f64x2::<0xdd>(ab1, cd1);
                    _mm512_store_pd(ptr.add(row * stride + c * 2), r0);
                    _mm512_store_pd(ptr.add((row + 1) * stride + c * 2), r1);
                    _mm512_store_pd(ptr.add((row + 2) * stride + c * 2), r2);
                    _mm512_store_pd(ptr.add((row + 3) * stride + c * 2), r3);
                }
            }
            for row in bulk..rows {
                for c in 0..cycles {
                    _mm_storeu_pd(
                        ptr.add(row * stride + c * 2),
                        _mm_loadu_pd(self.buffer.as_ptr().add((first + row + c * num) * 2)),
                    );
                }
            }
        }
        let input = unsafe { std::slice::from_raw_parts(ptr, rows * stride) };
        let samples = cycles * den * 2;
        let start = output.len();
        output.resize(start + samples, 0.0);
        let output = &mut output[start..];
        for group in (0..den).step_by(4) {
            let offsets: [usize; 4] = std::array::from_fn(|g| {
                let u = (group + g).min(den - 1);
                ((origin + ((numerator + u * num) / den) as i64
                    - self.half_taps
                    - self.buffer_start_frame) as usize)
                    * 2
            });
            let weights: [&[f64]; 4] = std::array::from_fn(|g| {
                let u = (group + g).min(den - 1);
                let p = (numerator + u * num) % den;
                &self.coeffs[p * self.taps..(p + 1) * self.taps]
            });
            // a{output}_{group} holds one output position across four periods.
            // Each register contains four interleaved left/right pairs.
            unsafe {
                let mut a0_0 = _mm512_setzero_pd();
                let mut a0_1 = _mm512_setzero_pd();
                let mut a0_2 = _mm512_setzero_pd();
                let mut a0_3 = _mm512_setzero_pd();
                let mut a1_0 = _mm512_setzero_pd();
                let mut a1_1 = _mm512_setzero_pd();
                let mut a1_2 = _mm512_setzero_pd();
                let mut a1_3 = _mm512_setzero_pd();
                let mut a2_0 = _mm512_setzero_pd();
                let mut a2_1 = _mm512_setzero_pd();
                let mut a2_2 = _mm512_setzero_pd();
                let mut a2_3 = _mm512_setzero_pd();
                let mut a3_0 = _mm512_setzero_pd();
                let mut a3_1 = _mm512_setzero_pd();
                let mut a3_2 = _mm512_setzero_pd();
                let mut a3_3 = _mm512_setzero_pd();
                let indices: [usize; 4] = offsets.map(|i| (i / 2 - first) * stride);
                for tap in 0..self.taps {
                    macro_rules! accumulate {
                        ($sum:ident, $pointer:expr, $weight:expr) => {
                            $sum = _mm512_add_pd(
                                $sum,
                                _mm512_mul_pd(_mm512_loadu_pd($pointer), $weight),
                            );
                        };
                    }
                    macro_rules! accumulate_phase {
                        ($phase:expr, $a:ident, $b:ident, $c:ident, $d:ident) => {{
                            let weight = _mm512_set1_pd(*weights[$phase].get_unchecked(tap));
                            accumulate!(
                                $a,
                                input.as_ptr().add(indices[$phase] + tap * stride),
                                weight
                            );
                            accumulate!(
                                $b,
                                input.as_ptr().add(indices[$phase] + tap * stride + 8),
                                weight
                            );
                            accumulate!(
                                $c,
                                input.as_ptr().add(indices[$phase] + tap * stride + 16),
                                weight
                            );
                            accumulate!(
                                $d,
                                input.as_ptr().add(indices[$phase] + tap * stride + 24),
                                weight
                            );
                        }};
                    }
                    accumulate_phase!(0, a0_0, a0_1, a0_2, a0_3);
                    accumulate_phase!(1, a1_0, a1_1, a1_2, a1_3);
                    accumulate_phase!(2, a2_0, a2_1, a2_2, a2_3);
                    accumulate_phase!(3, a3_0, a3_1, a3_2, a3_3);
                }

                // Each register holds four complete stereo periods. The final
                // phase group may contain one, two, three or four real outputs.
                macro_rules! store_periods {
                    ($sum:ident, $first_cycle:expr, $group:expr) => {{
                        if $group < den {
                            let mut values = [0.0f64; 8];
                            _mm512_storeu_pd(values.as_mut_ptr(), $sum);
                            for j in 0..4 {
                                let cycle = $first_cycle + j;
                                let target = (cycle * den + $group) * 2;
                                output[target] = values[j * 2];
                                output[target + 1] = values[j * 2 + 1];
                            }
                        }
                    }};
                }
                store_periods!(a0_0, 0, group);
                store_periods!(a0_1, 4, group);
                store_periods!(a0_2, 8, group);
                store_periods!(a0_3, 12, group);
                store_periods!(a1_0, 0, group + 1);
                store_periods!(a1_1, 4, group + 1);
                store_periods!(a1_2, 8, group + 1);
                store_periods!(a1_3, 12, group + 1);
                store_periods!(a2_0, 0, group + 2);
                store_periods!(a2_1, 4, group + 2);
                store_periods!(a2_2, 8, group + 2);
                store_periods!(a2_3, 12, group + 2);
                store_periods!(a3_0, 0, group + 3);
                store_periods!(a3_1, 4, group + 3);
                store_periods!(a3_2, 8, group + 3);
                store_periods!(a3_3, 12, group + 3);
            }
        }
        // Whole rational periods return to the same fractional phase.
        self.clock.source_index += (cycles * num) as i64;
        Some(samples)
    }

    // Eight repeated phases keep the AVX accumulator set within its sixteen
    // vector registers. The same tile is used on NEON. This is an ISA-level
    // layout. Construction applies the common latency and memory budgets.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    // Keep the qualified lane indexing and arithmetic order explicit.
    #[allow(clippy::needless_range_loop)]
    fn try_process_phase_tile_narrow(
        &mut self,
        total_frames: i64,
        eof: bool,
        output: &mut Vec<f64>,
    ) -> Option<usize> {
        const CYCLES: usize = 8;
        const STRIDE: usize = CYCLES * 2;
        if self.channels != 2 || self.taps == 0 {
            return None;
        }
        if self.phases != self.clock.step_den as usize {
            return None;
        }
        let origin = self.clock.source_index;
        let numerator = self.clock.phase_numer as usize;
        let num = self.clock.step_num as usize;
        let den = self.clock.step_den as usize;
        if origin - self.half_taps < self.buffer_start_frame {
            return None;
        }
        let maximum = total_frames + self.half_taps - self.taps as i64;
        if maximum < origin {
            return None;
        }
        let available = (((maximum - origin + 1) as usize * den - 1 - numerator) / num) + 1;
        if available < CYCLES * den {
            return if eof { None } else { Some(0) };
        }
        let first = (origin - self.half_taps - self.buffer_start_frame) as usize;
        let rows = self.taps + (numerator + (den - 1) * num) / den;
        let source_end = (first + rows + (CYCLES - 1) * num) * 2;
        // Slicing validates the full source and scratch extents before SIMD.
        let source = &self.buffer[first * 2..source_end];
        // Vec<f64> does not promise AVX alignment. Keep each packed vector
        // 32-byte aligned regardless of allocations made by the bank cache.
        let needed = rows * STRIDE;
        self.phase_scratch.resize(needed + 3, 0.0);
        let shift = self.phase_scratch.as_ptr().align_offset(32);
        let packed = &mut self.phase_scratch[shift..shift + needed];
        // SAFETY: the selected backend was detected at construction. Packing
        // copies f64 bits only, and the helpers check their own slice bounds.
        unsafe {
            match self.phase_batch_backend {
                #[cfg(target_arch = "x86_64")]
                PhaseBatchBackend::Avx => Self::pack_phase_tile_avx(source, rows, num, packed),
                #[cfg(target_arch = "aarch64")]
                PhaseBatchBackend::Neon => Self::pack_phase_tile_neon(source, rows, num, packed),
                _ => return None,
            }
        }
        let samples = CYCLES * den * 2;
        let start = output.len();
        output.resize(start + samples, 0.0);
        let output = &mut output[start..];
        for group in (0..den).step_by(2) {
            let phases: [usize; 2] = std::array::from_fn(|g| (group + g).min(den - 1));
            let samples: [&[f64]; 2] = std::array::from_fn(|g| {
                let row = (numerator + phases[g] * num) / den;
                &packed[row * STRIDE..(row + self.taps) * STRIDE]
            });
            let weights: [&[f64]; 2] = std::array::from_fn(|g| {
                let phase = (numerator + phases[g] * num) % den;
                &self.coeffs[phase * self.taps..(phase + 1) * self.taps]
            });
            // Every lane is one output/channel. Each receives taps in the
            // original order, with separately rounded multiplication/addition.
            let sums = unsafe {
                match self.phase_batch_backend {
                    #[cfg(target_arch = "x86_64")]
                    PhaseBatchBackend::Avx => Self::convolve_phase_pair_avx(samples, weights),
                    #[cfg(target_arch = "aarch64")]
                    PhaseBatchBackend::Neon => Self::convolve_phase_pair_neon(samples, weights),
                    _ => unreachable!("phase packing already checked the backend"),
                }
            };
            for g in 0..2 {
                if group + g < den {
                    for cycle in 0..CYCLES {
                        let target = (cycle * den + group + g) * 2;
                        output[target..target + 2]
                            .copy_from_slice(&sums[g][cycle * 2..cycle * 2 + 2]);
                    }
                }
            }
        }
        // Whole rational periods return to the same fractional phase.
        self.clock.source_index += (CYCLES * num) as i64;
        Some(samples)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline(never)]
    unsafe fn pack_phase_tile_avx(
        source: &[f64],
        rows: usize,
        cycle_step: usize,
        packed: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        assert!(rows > 0 && source.len() >= (rows + 7 * cycle_step) * 2);
        assert_eq!(packed.len(), rows * 16);
        // Two source frames from two cycles become two packed rows. No gathers,
        // AVX-512 instructions, or extended AVX-512 registers are needed.
        unsafe {
            let bulk = rows / 2 * 2;
            for row in (0..bulk).step_by(2) {
                for cycle in (0..8).step_by(2) {
                    let a = _mm256_loadu_pd(source.as_ptr().add((row + cycle * cycle_step) * 2));
                    let b =
                        _mm256_loadu_pd(source.as_ptr().add((row + (cycle + 1) * cycle_step) * 2));
                    _mm256_storeu_pd(
                        packed.as_mut_ptr().add(row * 16 + cycle * 2),
                        _mm256_permute2f128_pd::<0x20>(a, b),
                    );
                    _mm256_storeu_pd(
                        packed.as_mut_ptr().add((row + 1) * 16 + cycle * 2),
                        _mm256_permute2f128_pd::<0x31>(a, b),
                    );
                }
            }
            if bulk < rows {
                for cycle in 0..8 {
                    _mm_storeu_pd(
                        packed.as_mut_ptr().add(bulk * 16 + cycle * 2),
                        _mm_loadu_pd(source.as_ptr().add((bulk + cycle * cycle_step) * 2)),
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline(never)]
    unsafe fn pack_phase_tile_neon(
        source: &[f64],
        rows: usize,
        cycle_step: usize,
        packed: &mut [f64],
    ) {
        use std::arch::aarch64::*;
        assert!(rows > 0 && source.len() >= (rows + 7 * cycle_step) * 2);
        assert_eq!(packed.len(), rows * 16);
        unsafe {
            for row in 0..rows {
                for cycle in 0..8 {
                    vst1q_f64(
                        packed.as_mut_ptr().add(row * 16 + cycle * 2),
                        vld1q_f64(source.as_ptr().add((row + cycle * cycle_step) * 2)),
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline(never)]
    unsafe fn convolve_phase_pair_avx(
        samples: [&[f64]; 2],
        weights: [&[f64]; 2],
    ) -> [[f64; 16]; 2] {
        use std::arch::x86_64::*;
        let taps = weights[0].len();
        assert_eq!(weights[1].len(), taps);
        assert!(samples.iter().all(|row| row.len() / 16 >= taps));
        let mut sums = [[0.0; 16]; 2];
        // SAFETY: every vector load is within the checked rows. Output stores
        // cover exactly sixteen doubles per phase. No horizontal reductions.
        unsafe {
            let mut a0_0 = _mm256_setzero_pd();
            let mut a0_1 = _mm256_setzero_pd();
            let mut a0_2 = _mm256_setzero_pd();
            let mut a0_3 = _mm256_setzero_pd();
            let mut a1_0 = _mm256_setzero_pd();
            let mut a1_1 = _mm256_setzero_pd();
            let mut a1_2 = _mm256_setzero_pd();
            let mut a1_3 = _mm256_setzero_pd();
            for tap in 0..taps {
                macro_rules! accumulate {
                    ($sum:ident, $pointer:expr, $weight:expr) => {
                        $sum =
                            _mm256_add_pd($sum, _mm256_mul_pd(_mm256_loadu_pd($pointer), $weight));
                    };
                }
                let weight = _mm256_set1_pd(*weights[0].get_unchecked(tap));
                accumulate!(a0_0, samples[0].as_ptr().add(tap * 16), weight);
                accumulate!(a0_1, samples[0].as_ptr().add(tap * 16 + 4), weight);
                accumulate!(a0_2, samples[0].as_ptr().add(tap * 16 + 8), weight);
                accumulate!(a0_3, samples[0].as_ptr().add(tap * 16 + 12), weight);
                let weight = _mm256_set1_pd(*weights[1].get_unchecked(tap));
                accumulate!(a1_0, samples[1].as_ptr().add(tap * 16), weight);
                accumulate!(a1_1, samples[1].as_ptr().add(tap * 16 + 4), weight);
                accumulate!(a1_2, samples[1].as_ptr().add(tap * 16 + 8), weight);
                accumulate!(a1_3, samples[1].as_ptr().add(tap * 16 + 12), weight);
            }
            _mm256_storeu_pd(sums[0].as_mut_ptr().add(0), a0_0);
            _mm256_storeu_pd(sums[0].as_mut_ptr().add(4), a0_1);
            _mm256_storeu_pd(sums[0].as_mut_ptr().add(8), a0_2);
            _mm256_storeu_pd(sums[0].as_mut_ptr().add(12), a0_3);
            _mm256_storeu_pd(sums[1].as_mut_ptr().add(0), a1_0);
            _mm256_storeu_pd(sums[1].as_mut_ptr().add(4), a1_1);
            _mm256_storeu_pd(sums[1].as_mut_ptr().add(8), a1_2);
            _mm256_storeu_pd(sums[1].as_mut_ptr().add(12), a1_3);
        }
        sums
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline(never)]
    unsafe fn convolve_phase_pair_neon(
        samples: [&[f64]; 2],
        weights: [&[f64]; 2],
    ) -> [[f64; 16]; 2] {
        use std::arch::aarch64::*;
        let taps = weights[0].len();
        assert_eq!(weights[1].len(), taps);
        assert!(samples.iter().all(|row| row.len() / 16 >= taps));
        let mut sums = [[0.0; 16]; 2];
        // SAFETY: every vector load is within the checked rows. Output stores
        // cover exactly sixteen doubles per phase. No horizontal reductions.
        unsafe {
            let mut a0_0 = vdupq_n_f64(0.0);
            let mut a0_1 = vdupq_n_f64(0.0);
            let mut a0_2 = vdupq_n_f64(0.0);
            let mut a0_3 = vdupq_n_f64(0.0);
            let mut a0_4 = vdupq_n_f64(0.0);
            let mut a0_5 = vdupq_n_f64(0.0);
            let mut a0_6 = vdupq_n_f64(0.0);
            let mut a0_7 = vdupq_n_f64(0.0);
            let mut a1_0 = vdupq_n_f64(0.0);
            let mut a1_1 = vdupq_n_f64(0.0);
            let mut a1_2 = vdupq_n_f64(0.0);
            let mut a1_3 = vdupq_n_f64(0.0);
            let mut a1_4 = vdupq_n_f64(0.0);
            let mut a1_5 = vdupq_n_f64(0.0);
            let mut a1_6 = vdupq_n_f64(0.0);
            let mut a1_7 = vdupq_n_f64(0.0);
            for tap in 0..taps {
                macro_rules! accumulate {
                    ($sum:ident, $pointer:expr, $weight:expr) => {
                        $sum = vaddq_f64($sum, vmulq_f64(vld1q_f64($pointer), $weight));
                    };
                }
                let weight = vdupq_n_f64(*weights[0].get_unchecked(tap));
                accumulate!(a0_0, samples[0].as_ptr().add(tap * 16), weight);
                accumulate!(a0_1, samples[0].as_ptr().add(tap * 16 + 2), weight);
                accumulate!(a0_2, samples[0].as_ptr().add(tap * 16 + 4), weight);
                accumulate!(a0_3, samples[0].as_ptr().add(tap * 16 + 6), weight);
                accumulate!(a0_4, samples[0].as_ptr().add(tap * 16 + 8), weight);
                accumulate!(a0_5, samples[0].as_ptr().add(tap * 16 + 10), weight);
                accumulate!(a0_6, samples[0].as_ptr().add(tap * 16 + 12), weight);
                accumulate!(a0_7, samples[0].as_ptr().add(tap * 16 + 14), weight);
                let weight = vdupq_n_f64(*weights[1].get_unchecked(tap));
                accumulate!(a1_0, samples[1].as_ptr().add(tap * 16), weight);
                accumulate!(a1_1, samples[1].as_ptr().add(tap * 16 + 2), weight);
                accumulate!(a1_2, samples[1].as_ptr().add(tap * 16 + 4), weight);
                accumulate!(a1_3, samples[1].as_ptr().add(tap * 16 + 6), weight);
                accumulate!(a1_4, samples[1].as_ptr().add(tap * 16 + 8), weight);
                accumulate!(a1_5, samples[1].as_ptr().add(tap * 16 + 10), weight);
                accumulate!(a1_6, samples[1].as_ptr().add(tap * 16 + 12), weight);
                accumulate!(a1_7, samples[1].as_ptr().add(tap * 16 + 14), weight);
            }
            vst1q_f64(sums[0].as_mut_ptr().add(0), a0_0);
            vst1q_f64(sums[0].as_mut_ptr().add(2), a0_1);
            vst1q_f64(sums[0].as_mut_ptr().add(4), a0_2);
            vst1q_f64(sums[0].as_mut_ptr().add(6), a0_3);
            vst1q_f64(sums[0].as_mut_ptr().add(8), a0_4);
            vst1q_f64(sums[0].as_mut_ptr().add(10), a0_5);
            vst1q_f64(sums[0].as_mut_ptr().add(12), a0_6);
            vst1q_f64(sums[0].as_mut_ptr().add(14), a0_7);
            vst1q_f64(sums[1].as_mut_ptr().add(0), a1_0);
            vst1q_f64(sums[1].as_mut_ptr().add(2), a1_1);
            vst1q_f64(sums[1].as_mut_ptr().add(4), a1_2);
            vst1q_f64(sums[1].as_mut_ptr().add(6), a1_3);
            vst1q_f64(sums[1].as_mut_ptr().add(8), a1_4);
            vst1q_f64(sums[1].as_mut_ptr().add(10), a1_5);
            vst1q_f64(sums[1].as_mut_ptr().add(12), a1_6);
            vst1q_f64(sums[1].as_mut_ptr().add(14), a1_7);
        }
        sums
    }

    // Use the original endpoint continuation values with the same SIMD
    // ordered dot products. Padding is local to one input/drain call.
    fn try_process_padded_batch(
        &mut self,
        total_frames: i64,
        eof: bool,
        padding: &[f64],
        pad_start: i64,
    ) -> Option<[[f64; 2]; 4]> {
        if self.channels != 2 {
            return None;
        }
        let source_index = self.clock.source_index;
        let phase_numer = u64::from(self.clock.phase_numer);
        let step_num = u64::from(self.clock.step_num);
        let step_den = u64::from(self.clock.step_den);
        if step_den != self.phases as u64 {
            return None;
        }
        let first = source_index - self.half_taps;
        let last = source_index + ((phase_numer + 3 * step_num) / step_den) as i64 - self.half_taps;
        if last + self.half_taps >= total_frames - if eof { 0 } else { self.half_taps }
            || first < pad_start
            || last + self.taps as i64 > pad_start + (padding.len() / 2) as i64
        {
            return None;
        }
        let sums = self.convolve_exact_stereo_batch(padding, pad_start);
        for _ in 0..4 {
            self.advance();
        }
        Some(sums)
    }

    #[cfg(test)]
    fn process_f64(&mut self, input: &[f64], end_of_stream: bool) -> Vec<f64> {
        let mut output = Vec::new();
        self.process_f64_into(input, end_of_stream, &mut output);
        output
    }

    // Preserve the qualified endpoint padding's channel indexing.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn process_f64_into(
        &mut self,
        input: &[f64],
        end_of_stream: bool,
        output: &mut Vec<f64>,
    ) {
        // Bound the input size eligible for temporary endpoint padding.
        const MAX_PADDED_INPUT_SAMPLES: usize = 131_072;

        debug_assert_eq!(input.len() % self.channels, 0);
        self.buffer.extend_from_slice(input);

        let input_frames = input.len() / self.channels;
        let estimated_output_frames = self.estimated_output_frames(input_frames);
        let total_frames = self.buffer_start_frame + (self.buffer.len() / self.channels) as i64;
        output.clear();
        output.reserve(estimated_output_frames * self.channels);
        let mut tail_blends = Vec::new();
        // Compute endpoint continuation once per source frame, not once per tap.
        let padding = if self.channels == 2
            && self.buffer.len() <= MAX_PADDED_INPUT_SAMPLES
            && self.can_emit(total_frames, end_of_stream)
            && (end_of_stream
                || self.current_position().0 - self.half_taps < self.buffer_start_frame)
        {
            let start = (self.current_position().0 - self.half_taps).min(self.buffer_start_frame);
            let end = total_frames
                + if end_of_stream {
                    self.taps as i64 - self.half_taps
                } else {
                    0
                };
            let blend = if end_of_stream {
                self.tail_blends(total_frames, &mut tail_blends)
            } else {
                &[0.0; 2]
            };
            let mut values = Vec::with_capacity((end - start) as usize * 2);
            for frame in start..end {
                for ch in 0..2 {
                    values.push(endpoint_continued_sample_f64(
                        &self.buffer,
                        self.channels,
                        self.buffer_start_frame,
                        total_frames,
                        frame,
                        ch,
                        blend[ch],
                    ));
                }
            }
            Some((values, start))
        } else {
            None
        };

        while self.can_emit(total_frames, end_of_stream) {
            if let Some(samples) = self.try_process_phase_tile(total_frames, end_of_stream, output)
            {
                if samples == 0 {
                    break;
                }
                continue;
            }
            if let Some((values, start)) = &padding {
                if let Some(frames) =
                    self.try_process_padded_batch(total_frames, end_of_stream, values, *start)
                {
                    output.extend(frames.into_iter().flatten());
                    continue;
                }
            }
            if let Some(frames) = self.try_process_stereo_batch(total_frames) {
                output.extend(frames.into_iter().flatten());
                continue;
            }
            if let Some(frames) = self.try_process_interpolated_stereo_batch(
                total_frames,
                end_of_stream,
                padding
                    .as_ref()
                    .map(|(values, start)| (values.as_slice(), *start)),
            ) {
                output.extend(frames.into_iter().flatten());
                continue;
            }

            self.emit_fallback_frame(total_frames, end_of_stream, &mut tail_blends, |sample| {
                output.push(sample);
            });

            self.advance();
        }

        self.discard_consumed_input(end_of_stream, total_frames);
    }
}

// Read retained input directly. Synthesize missing frames by reflection.
// At the tail, blend=0 mirrors the signal and blend=1 reflects about its edge.
fn endpoint_continued_sample_f64(
    buffer: &[f64],
    channels: usize,
    buffer_start_frame: i64,
    total_frames: i64,
    sample_frame: i64,
    ch: usize,
    tail_blend: f64,
) -> f64 {
    debug_assert!(channels > 0);
    debug_assert!(ch < channels);
    if total_frames <= buffer_start_frame || buffer.is_empty() {
        return 0.0;
    }

    if sample_frame >= buffer_start_frame && sample_frame < total_frames {
        let rel = ((sample_frame - buffer_start_frame) as usize) * channels + ch;
        return buffer[rel];
    }

    let first_frame = buffer_start_frame;
    let last_frame = total_frames - 1;
    if sample_frame < first_frame {
        let mirrored_frame = (first_frame + (first_frame - sample_frame)).min(last_frame);
        let edge_rel = ch;
        let mirror_rel = ((mirrored_frame - buffer_start_frame) as usize) * channels + ch;
        return 2.0 * buffer[edge_rel] - buffer[mirror_rel];
    }

    let mirrored_frame = (last_frame - (sample_frame - last_frame)).max(first_frame);
    let edge_rel = ((last_frame - buffer_start_frame) as usize) * channels + ch;
    let mirror_rel = ((mirrored_frame - buffer_start_frame) as usize) * channels + ch;
    let mirrored = buffer[mirror_rel];
    let odd_reflected = 2.0 * buffer[edge_rel] - mirrored;
    mirrored + (odd_reflected - mirrored) * tail_blend
}

// Estimate tail roughness from successive differences relative to signal RMS.
// Smoother tails favor odd reflection. Rougher tails favor ordinary mirroring.
fn endpoint_tail_odd_reflection_blend(
    buffer: &[f64],
    channels: usize,
    buffer_start_frame: i64,
    total_frames: i64,
    ch: usize,
) -> f64 {
    const ROUGHNESS_FULL_ODD: f64 = 0.40;
    const ROUGHNESS_FULL_MIRROR: f64 = 0.80;
    const WINDOW_FRAMES: i64 = 64;

    let available_frames = total_frames - buffer_start_frame;
    if available_frames <= 2 {
        return 0.0;
    }

    let start_frame = (total_frames - WINDOW_FRAMES).max(buffer_start_frame);
    let mut sample_sq = 0.0f64;
    let mut diff_sq = 0.0f64;
    let mut sample_count = 0usize;
    let mut diff_count = 0usize;
    let mut previous = None;

    for frame in start_frame..total_frames {
        let rel = ((frame - buffer_start_frame) as usize) * channels + ch;
        let sample = buffer[rel];
        sample_sq += sample * sample;
        sample_count += 1;
        if let Some(previous) = previous {
            let diff = sample - previous;
            diff_sq += diff * diff;
            diff_count += 1;
        }
        previous = Some(sample);
    }

    let signal_rms = (sample_sq / sample_count.max(1) as f64).sqrt();
    let diff_rms = (diff_sq / diff_count.max(1) as f64).sqrt();
    let roughness = diff_rms / signal_rms.max(1e-12);
    let x = ((roughness - ROUGHNESS_FULL_ODD) / (ROUGHNESS_FULL_MIRROR - ROUGHNESS_FULL_ODD))
        .clamp(0.0, 1.0);
    let smoothstep = x * x * (3.0 - 2.0 * x);
    1.0 - smoothstep
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
#[path = "simd_tests.rs"]
mod simd_tests;

#[cfg(test)]
#[path = "phase_batch_tests.rs"]
mod phase_batch_tests;

#[cfg(test)]
#[path = "interpolated_tests.rs"]
mod interpolated_tests;
