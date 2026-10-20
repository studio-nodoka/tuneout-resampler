//! Bit-preserving arithmetic kernels. Lanes are independent modes, never time.
//!
//! Every scalar multiplication, subtraction and addition remains separately
//! rounded in its original order. In particular there is no FMA, approximate
//! reciprocal, horizontal reassociation, change to the FP environment, or extra
//! input buffering. `Complex` is repr(C), two contiguous f64 values without
//! padding; all vector memory access is unaligned and slice-bounded.

use crate::complex::Complex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Kernel(Backend);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // Qualified for exactness, but not auto-selected: native screening found
    // the compiler-vectorized scalar fallback faster than this narrow kernel.
    #[allow(dead_code)]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Kernel {
    pub(crate) fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Like FIR dispatch, these checks include OS register-state support.
            if std::is_x86_feature_detected!("avx512f") {
                return Self(Backend::Avx512);
            }
            if std::is_x86_feature_detected!("avx") {
                return Self(Backend::Avx);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Self(Backend::Neon);
        }
        Self(Backend::Scalar)
    }

    #[inline]
    pub(crate) fn causal(self, poles: &[Complex], states: &mut [Complex], sample: f64) {
        self.update::<true>(poles, states, sample);
    }

    #[inline]
    pub(crate) fn future(self, poles: &[Complex], states: &mut [Complex], sample: f64) {
        self.update::<false>(poles, states, sample);
    }

    #[inline]
    fn update<const CAUSAL: bool>(self, poles: &[Complex], states: &mut [Complex], sample: f64) {
        assert_eq!(poles.len(), states.len());
        // SAFETY: non-scalar variants are selected only after feature detection
        // (or directly by tests which check that same feature immediately).
        match self.0 {
            Backend::Scalar => update_scalar::<CAUSAL>(poles, states, sample),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::update_sse2::<CAUSAL>(poles, states, sample) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe { x86::update_avx::<CAUSAL>(poles, states, sample) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe { x86::update_avx512::<CAUSAL>(poles, states, sample) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { neon::update::<CAUSAL>(poles, states, sample) },
        }
    }

    #[inline]
    pub(crate) fn dot(self, row: &[Complex], states: &[Complex]) -> f64 {
        assert_eq!(row.len(), states.len());
        // AVX-512 uses the four-lane AVX reduction: adding more accumulators
        // would change rounding. Wider state updates remain useful separately.
        match self.0 {
            Backend::Scalar => dot_scalar(row, states),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::dot_sse2(row, states) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx | Backend::Avx512 => unsafe { x86::dot_avx(row, states) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { neon::dot(row, states) },
        }
    }
}

#[inline]
fn update_scalar<const CAUSAL: bool>(poles: &[Complex], states: &mut [Complex], sample: f64) {
    for (state, &pole) in states.iter_mut().zip(poles) {
        *state = if CAUSAL {
            pole * Complex::new(state.re + sample, state.im)
        } else {
            pole * *state + Complex::new(sample, 0.0)
        };
    }
}

#[inline]
fn finish_dot(mut sums: [f64; 4], row: &[Complex], states: &[Complex]) -> f64 {
    for (i, (&c, &s)) in row.iter().zip(states).enumerate() {
        sums[i] += c.re * s.re - c.im * s.im;
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

fn dot_scalar(row: &[Complex], states: &[Complex]) -> f64 {
    let mut sums = [0.0; 4];
    let bulk = row.len() / 4 * 4;
    for (c, s) in row[..bulk]
        .chunks_exact(4)
        .zip(states[..bulk].chunks_exact(4))
    {
        sums[0] += c[0].re * s[0].re - c[0].im * s[0].im;
        sums[1] += c[1].re * s[1].re - c[1].im * s[1].im;
        sums[2] += c[2].re * s[2].re - c[2].im * s[2].im;
        sums[3] += c[3].re * s[3].re - c[3].im * s[3].im;
    }
    finish_dot(sums, &row[bulk..], &states[bulk..])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn update_sse2<const CAUSAL: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        sample: f64,
    ) {
        // SAFETY: caller checked SSE2 and equal lengths. Each pair contains two
        // complete repr(C) Complex values. The odd tail is handled separately.
        unsafe {
            let input = _mm_set1_pd(sample);
            let zero = _mm_setzero_pd();
            let bulk = poles.len() / 2 * 2;
            for i in (0..bulk).step_by(2) {
                let p0 = _mm_loadu_pd(poles.as_ptr().add(i).cast());
                let p1 = _mm_loadu_pd(poles.as_ptr().add(i + 1).cast());
                let s0 = _mm_loadu_pd(states.as_ptr().add(i).cast());
                let s1 = _mm_loadu_pd(states.as_ptr().add(i + 1).cast());
                let pr = _mm_unpacklo_pd(p0, p1);
                let pi = _mm_unpackhi_pd(p0, p1);
                let mut sr = _mm_unpacklo_pd(s0, s1);
                let si = _mm_unpackhi_pd(s0, s1);
                if CAUSAL {
                    sr = _mm_add_pd(sr, input);
                }
                let mut re = _mm_sub_pd(_mm_mul_pd(pr, sr), _mm_mul_pd(pi, si));
                let mut im = _mm_add_pd(_mm_mul_pd(pr, si), _mm_mul_pd(pi, sr));
                if !CAUSAL {
                    re = _mm_add_pd(re, input);
                    im = _mm_add_pd(im, zero);
                }
                _mm_storeu_pd(states.as_mut_ptr().add(i).cast(), _mm_unpacklo_pd(re, im));
                _mm_storeu_pd(
                    states.as_mut_ptr().add(i + 1).cast(),
                    _mm_unpackhi_pd(re, im),
                );
            }
            update_scalar::<CAUSAL>(&poles[bulk..], &mut states[bulk..], sample);
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn update_avx<const CAUSAL: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        sample: f64,
    ) {
        // SAFETY: feature and equal lengths checked by caller. Each vector is
        // two complete Complex values; the scalar tail covers an odd count.
        unsafe {
            let input = _mm256_set_pd(0.0, sample, 0.0, sample);
            let bulk = poles.len() / 2 * 2;
            for i in (0..bulk).step_by(2) {
                let p = _mm256_loadu_pd(poles.as_ptr().add(i).cast());
                let mut s = _mm256_loadu_pd(states.as_ptr().add(i).cast());
                if CAUSAL {
                    // Imaginary signed zero must not receive an extra +0.
                    s = _mm256_blend_pd::<5>(s, _mm256_add_pd(s, input));
                }
                let re = _mm256_mul_pd(_mm256_movedup_pd(p), s);
                let im = _mm256_mul_pd(_mm256_permute_pd::<15>(p), _mm256_permute_pd::<5>(s));
                let mut result = _mm256_addsub_pd(re, im);
                if !CAUSAL {
                    result = _mm256_add_pd(result, input);
                }
                _mm256_storeu_pd(states.as_mut_ptr().add(i).cast(), result);
            }
            update_scalar::<CAUSAL>(&poles[bulk..], &mut states[bulk..], sample);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn update_avx512<const CAUSAL: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        sample: f64,
    ) {
        // SAFETY: feature and equal lengths checked by caller. Each vector is
        // four complete Complex values. Unmasked loads never touch the tail.
        unsafe {
            let input = _mm512_set_pd(0.0, sample, 0.0, sample, 0.0, sample, 0.0, sample);
            let bulk = poles.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                let p = _mm512_loadu_pd(poles.as_ptr().add(i).cast());
                let mut s = _mm512_loadu_pd(states.as_ptr().add(i).cast());
                if CAUSAL {
                    s = _mm512_mask_add_pd(s, 0x55, s, input);
                }
                let re = _mm512_mul_pd(_mm512_movedup_pd(p), s);
                let im = _mm512_mul_pd(_mm512_permute_pd::<255>(p), _mm512_permute_pd::<85>(s));
                let mut result = _mm512_mask_sub_pd(_mm512_add_pd(re, im), 0x55, re, im);
                if !CAUSAL {
                    result = _mm512_add_pd(result, input);
                }
                _mm512_storeu_pd(states.as_mut_ptr().add(i).cast(), result);
            }
            update_scalar::<CAUSAL>(&poles[bulk..], &mut states[bulk..], sample);
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn dot_sse2(row: &[Complex], states: &[Complex]) -> f64 {
        // SAFETY: checked equal lengths, four whole values per group, two
        // f64 values per unaligned load. Four accumulator lanes stay separate.
        unsafe {
            let mut a = _mm_setzero_pd();
            let mut b = _mm_setzero_pd();
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                let p0 = _mm_mul_pd(
                    _mm_loadu_pd(row.as_ptr().add(i).cast()),
                    _mm_loadu_pd(states.as_ptr().add(i).cast()),
                );
                let p1 = _mm_mul_pd(
                    _mm_loadu_pd(row.as_ptr().add(i + 1).cast()),
                    _mm_loadu_pd(states.as_ptr().add(i + 1).cast()),
                );
                let p2 = _mm_mul_pd(
                    _mm_loadu_pd(row.as_ptr().add(i + 2).cast()),
                    _mm_loadu_pd(states.as_ptr().add(i + 2).cast()),
                );
                let p3 = _mm_mul_pd(
                    _mm_loadu_pd(row.as_ptr().add(i + 3).cast()),
                    _mm_loadu_pd(states.as_ptr().add(i + 3).cast()),
                );
                a = _mm_add_pd(
                    a,
                    _mm_sub_pd(_mm_unpacklo_pd(p0, p1), _mm_unpackhi_pd(p0, p1)),
                );
                b = _mm_add_pd(
                    b,
                    _mm_sub_pd(_mm_unpacklo_pd(p2, p3), _mm_unpackhi_pd(p2, p3)),
                );
            }
            let mut sums = [0.0; 4];
            _mm_storeu_pd(sums.as_mut_ptr(), a);
            _mm_storeu_pd(sums.as_mut_ptr().add(2), b);
            finish_dot(sums, &row[bulk..], &states[bulk..])
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn dot_avx(row: &[Complex], states: &[Complex]) -> f64 {
        // SAFETY: two Complex per load, two loads per checked group. HSUB only
        // forms each individual complex product's real part, never a reduction.
        unsafe {
            let mut sum = _mm256_setzero_pd();
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                let a = _mm256_mul_pd(
                    _mm256_loadu_pd(row.as_ptr().add(i).cast()),
                    _mm256_loadu_pd(states.as_ptr().add(i).cast()),
                );
                let b = _mm256_mul_pd(
                    _mm256_loadu_pd(row.as_ptr().add(i + 2).cast()),
                    _mm256_loadu_pd(states.as_ptr().add(i + 2).cast()),
                );
                sum = _mm256_add_pd(sum, _mm256_hsub_pd(a, b));
            }
            let mut lanes = [0.0; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), sum);
            // HSUB's physical order is modes 0,2,1,3. Restore the logical
            // order before processing remainder lanes and the final sum.
            finish_dot(
                [lanes[0], lanes[2], lanes[1], lanes[3]],
                &row[bulk..],
                &states[bulk..],
            )
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn update<const CAUSAL: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        sample: f64,
    ) {
        // SAFETY: NEON and equal lengths checked. Each interleaved load/store
        // covers two complete repr(C) Complex values. Scalar tail is separate.
        unsafe {
            let input = vdupq_n_f64(sample);
            let zero = vdupq_n_f64(0.0);
            let bulk = poles.len() / 2 * 2;
            for i in (0..bulk).step_by(2) {
                let p = vld2q_f64(poles.as_ptr().add(i).cast());
                let mut s = vld2q_f64(states.as_ptr().add(i).cast());
                if CAUSAL {
                    s.0 = vaddq_f64(s.0, input);
                }
                let mut re = vsubq_f64(vmulq_f64(p.0, s.0), vmulq_f64(p.1, s.1));
                let mut im = vaddq_f64(vmulq_f64(p.0, s.1), vmulq_f64(p.1, s.0));
                if !CAUSAL {
                    re = vaddq_f64(re, input);
                    im = vaddq_f64(im, zero);
                }
                vst2q_f64(states.as_mut_ptr().add(i).cast(), float64x2x2_t(re, im));
            }
            update_scalar::<CAUSAL>(&poles[bulk..], &mut states[bulk..], sample);
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(row: &[Complex], states: &[Complex]) -> f64 {
        // SAFETY: checked full groups of four Complex values. Each NEON lane
        // accumulates one of the original four scalar lanes in the same order.
        unsafe {
            let mut a = vdupq_n_f64(0.0);
            let mut b = vdupq_n_f64(0.0);
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                let c0 = vld2q_f64(row.as_ptr().add(i).cast());
                let s0 = vld2q_f64(states.as_ptr().add(i).cast());
                let c1 = vld2q_f64(row.as_ptr().add(i + 2).cast());
                let s1 = vld2q_f64(states.as_ptr().add(i + 2).cast());
                a = vaddq_f64(a, vsubq_f64(vmulq_f64(c0.0, s0.0), vmulq_f64(c0.1, s0.1)));
                b = vaddq_f64(b, vsubq_f64(vmulq_f64(c1.0, s1.0), vmulq_f64(c1.1, s1.1)));
            }
            let mut sums = [0.0; 4];
            vst1q_f64(sums.as_mut_ptr(), a);
            vst1q_f64(sums.as_mut_ptr().add(2), b);
            finish_dot(sums, &row[bulk..], &states[bulk..])
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use tests::available as test_kernels;

mod guards;

#[cfg(test)]
mod blocks;

mod coarse;

mod coarse_planes;
pub(crate) use coarse_planes::Planes;

mod coarse_resident;

mod coarse_resident_weighted_real;
mod coarse_weighted_real;

pub(crate) use coarse::RealHistoryMac;
