//! Two independent stereo guards packed across channels and modes.
use super::{Backend, Complex, Kernel};

impl Kernel {
    pub(crate) fn guard_pair(
        self,
        poles: [Complex; 2],
        states: &mut [[Complex; 2]; 2],
        tails: [&[f64]; 2],
    ) {
        assert_eq!(tails[0].len(), tails[1].len());
        assert_eq!(tails[0].len() % 2, 0);
        // SAFETY: Kernel is opaque and contains only feature-checked backends;
        // equal complete stereo frames above cover every unaligned input load.
        match self.0 {
            Backend::Scalar => scalar(poles, states, tails),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::sse2(poles, states, tails) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx | Backend::Avx512 => unsafe { x86::avx(poles, states, tails) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { neon(poles, states, tails) },
        }
    }
}

fn scalar(poles: [Complex; 2], states: &mut [[Complex; 2]; 2], tails: [&[f64]; 2]) {
    for (a, b) in tails[0]
        .chunks_exact(2)
        .rev()
        .zip(tails[1].chunks_exact(2).rev())
    {
        for channel in 0..2 {
            states[0][channel] = poles[0] * states[0][channel] + Complex::new(a[channel], 0.0);
            states[1][channel] = poles[1] * states[1][channel] + Complex::new(b[channel], 0.0);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse2(
        poles: [Complex; 2],
        states: &mut [[Complex; 2]; 2],
        tails: [&[f64]; 2],
    ) {
        // SAFETY: each checked stereo frame supplies two f64 values. State
        // lanes are packed explicitly, so no type-layout assumptions are used.
        unsafe {
            let pr0 = _mm_set1_pd(poles[0].re);
            let pi0 = _mm_set1_pd(poles[0].im);
            let pr1 = _mm_set1_pd(poles[1].re);
            let pi1 = _mm_set1_pd(poles[1].im);
            let mut re0 = _mm_set_pd(states[0][1].re, states[0][0].re);
            let mut im0 = _mm_set_pd(states[0][1].im, states[0][0].im);
            let mut re1 = _mm_set_pd(states[1][1].re, states[1][0].re);
            let mut im1 = _mm_set_pd(states[1][1].im, states[1][0].im);
            let zero = _mm_setzero_pd();
            for i in (0..tails[0].len()).step_by(2).rev() {
                let r0 = _mm_add_pd(
                    _mm_sub_pd(_mm_mul_pd(pr0, re0), _mm_mul_pd(pi0, im0)),
                    _mm_loadu_pd(tails[0].as_ptr().add(i)),
                );
                let i0 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(pr0, im0), _mm_mul_pd(pi0, re0)), zero);
                let r1 = _mm_add_pd(
                    _mm_sub_pd(_mm_mul_pd(pr1, re1), _mm_mul_pd(pi1, im1)),
                    _mm_loadu_pd(tails[1].as_ptr().add(i)),
                );
                let i1 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(pr1, im1), _mm_mul_pd(pi1, re1)), zero);
                re0 = r0;
                im0 = i0;
                re1 = r1;
                im1 = i1;
            }
            let mut re = [0.0; 4];
            let mut im = [0.0; 4];
            _mm_storeu_pd(re.as_mut_ptr(), re0);
            _mm_storeu_pd(re.as_mut_ptr().add(2), re1);
            _mm_storeu_pd(im.as_mut_ptr(), im0);
            _mm_storeu_pd(im.as_mut_ptr().add(2), im1);
            for mode in 0..2 {
                for channel in 0..2 {
                    states[mode][channel] =
                        Complex::new(re[mode * 2 + channel], im[mode * 2 + channel]);
                }
            }
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn avx(
        poles: [Complex; 2],
        states: &mut [[Complex; 2]; 2],
        tails: [&[f64]; 2],
    ) {
        // SAFETY: each low/high 128-bit input load covers a checked complete
        // frame. Four vector lanes are long L/R then short L/R; time is serial.
        unsafe {
            let pr = _mm256_set_pd(poles[1].re, poles[1].re, poles[0].re, poles[0].re);
            let pi = _mm256_set_pd(poles[1].im, poles[1].im, poles[0].im, poles[0].im);
            let mut re = _mm256_set_pd(
                states[1][1].re,
                states[1][0].re,
                states[0][1].re,
                states[0][0].re,
            );
            let mut im = _mm256_set_pd(
                states[1][1].im,
                states[1][0].im,
                states[0][1].im,
                states[0][0].im,
            );
            let zero = _mm256_setzero_pd();
            for i in (0..tails[0].len()).step_by(2).rev() {
                let input = _mm256_insertf128_pd::<1>(
                    _mm256_castpd128_pd256(_mm_loadu_pd(tails[0].as_ptr().add(i))),
                    _mm_loadu_pd(tails[1].as_ptr().add(i)),
                );
                let r = _mm256_add_pd(
                    _mm256_sub_pd(_mm256_mul_pd(pr, re), _mm256_mul_pd(pi, im)),
                    input,
                );
                let j = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(pr, im), _mm256_mul_pd(pi, re)),
                    zero,
                );
                re = r;
                im = j;
            }
            let mut real = [0.0; 4];
            let mut imag = [0.0; 4];
            _mm256_storeu_pd(real.as_mut_ptr(), re);
            _mm256_storeu_pd(imag.as_mut_ptr(), im);
            for mode in 0..2 {
                for channel in 0..2 {
                    states[mode][channel] =
                        Complex::new(real[mode * 2 + channel], imag[mode * 2 + channel]);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(poles: [Complex; 2], states: &mut [[Complex; 2]; 2], tails: [&[f64]; 2]) {
    use std::arch::aarch64::*;
    // SAFETY: each input load covers a checked complete stereo frame. Each
    // local array passed to vld1q/vst1q contains exactly two f64 values.
    unsafe {
        let pr0 = vdupq_n_f64(poles[0].re);
        let pi0 = vdupq_n_f64(poles[0].im);
        let pr1 = vdupq_n_f64(poles[1].re);
        let pi1 = vdupq_n_f64(poles[1].im);
        let mut re0 = vld1q_f64([states[0][0].re, states[0][1].re].as_ptr());
        let mut im0 = vld1q_f64([states[0][0].im, states[0][1].im].as_ptr());
        let mut re1 = vld1q_f64([states[1][0].re, states[1][1].re].as_ptr());
        let mut im1 = vld1q_f64([states[1][0].im, states[1][1].im].as_ptr());
        let zero = vdupq_n_f64(0.0);
        for i in (0..tails[0].len()).step_by(2).rev() {
            let r0 = vaddq_f64(
                vsubq_f64(vmulq_f64(pr0, re0), vmulq_f64(pi0, im0)),
                vld1q_f64(tails[0].as_ptr().add(i)),
            );
            let i0 = vaddq_f64(vaddq_f64(vmulq_f64(pr0, im0), vmulq_f64(pi0, re0)), zero);
            let r1 = vaddq_f64(
                vsubq_f64(vmulq_f64(pr1, re1), vmulq_f64(pi1, im1)),
                vld1q_f64(tails[1].as_ptr().add(i)),
            );
            let i1 = vaddq_f64(vaddq_f64(vmulq_f64(pr1, im1), vmulq_f64(pi1, re1)), zero);
            re0 = r0;
            im0 = i0;
            re1 = r1;
            im1 = i1;
        }
        let mut re = [0.0; 4];
        let mut im = [0.0; 4];
        vst1q_f64(re.as_mut_ptr(), re0);
        vst1q_f64(re.as_mut_ptr().add(2), re1);
        vst1q_f64(im.as_mut_ptr(), im0);
        vst1q_f64(im.as_mut_ptr().add(2), im1);
        for mode in 0..2 {
            for channel in 0..2 {
                states[mode][channel] =
                    Complex::new(re[mode * 2 + channel], im[mode * 2 + channel]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_stereo_guards_preserve_each_recurrence() {
        let tiny = f64::from_bits(1);
        let special = [
            -0.0,
            0.0,
            tiny,
            -tiny,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            0.12345,
            -0.34567,
        ];
        for kernel in crate::simd::test_kernels() {
            for length in (0..=33).chain([257, 1001]) {
                for offset in [0, 1] {
                    for adverse in [false, true] {
                        let a: Vec<_> = (0..2 * length + offset)
                            .map(|i| {
                                if adverse {
                                    special[i % special.len()]
                                } else {
                                    ((i * 731 % 997) as f64 - 498.0) / 1994.0
                                }
                            })
                            .collect();
                        let b: Vec<_> = (0..2 * length + offset)
                            .map(|i| {
                                if adverse {
                                    special[(i * 3 + 1) % special.len()]
                                } else {
                                    ((i * 379 % 997) as f64 - 498.0) / 1994.0
                                }
                            })
                            .collect();
                        let poles = [
                            Complex::new(0.9917, 0.000137),
                            Complex::new(-0.8871, -0.01519),
                        ];
                        let initial = [
                            [Complex::new(-0.0, tiny), Complex::new(tiny, -0.0)],
                            [Complex::new(0.125, -0.00014), Complex::new(-0.25, 0.777)],
                        ];
                        let mut expected = initial;
                        let mut actual = initial;
                        scalar(poles, &mut expected, [&a[offset..], &b[offset..]]);
                        kernel.guard_pair(poles, &mut actual, [&a[offset..], &b[offset..]]);
                        for (a, e) in actual.iter().flatten().zip(expected.iter().flatten()) {
                            assert_eq!(
                                [a.re.to_bits(), a.im.to_bits()],
                                [e.re.to_bits(), e.im.to_bits()],
                                "{kernel:?}, length {length}, offset {offset}, adverse {adverse}"
                            );
                        }
                    }
                }
            }
        }
    }
}
